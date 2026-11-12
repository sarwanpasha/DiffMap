#!/usr/bin/env python
"""
build_real_loci.py -- turn real fine-mapping regions into 1000-SNP loci with an
EXTERNAL 1000 Genomes LD reference.

Input : region .npz files from extract_ukb.py / extract_finngen.py
        (z, alleles, gold-standard in-sample PIP and credible sets, GRCh37 positions)
Output: one .npz per confident gold-standard lead variant, containing
          z        cohort z-scores, allele-aligned to the 1000G ALT allele
          R        external LD = correlation over the 1000G samples of --pop
          gold_pip / gold_cs  in-sample-LD gold standard
          lead_idx index of the gold lead (gold PIP > 0.9 in a 95% CS)
          diag     reference-mismatch diagnostic corr(z, R[:,lead] z[lead])

Steps for each unique confident lead (a lead appearing in several overlapping
regions is built from the region where its gold PIP is highest):
  1. query the 1000G VCF within +/- 1.5 Mb of the lead (bounded scan);
  2. match variants by position AND allele pair; the cohort effect allele is
     allele2, so z is multiplied by +1 if allele2 == 1000G ALT and -1 otherwise;
     variants whose allele pair does not match (indels, multi-allelic) are dropped;
  3. drop variants with missing genotypes or MAF < 0.01 in the reference population;
  4. keep the --window (1000) matched variants nearest the lead -- the model was
     trained on 1000-SNP loci, so real loci are windowed to stay in distribution;
  5. compute the external LD over the reference samples.
The run is resumable: leads whose output file already exists are skipped.

Example
  python build_real_loci.py --regions DATA/ukb_finemap/height_regions --out DATA/ukb_finemap/real_loci_height \
         --vcf-dir DATA/1000g --panel DATA/1000g/integrated_call_samples_v3.20130502.ALL.panel --pop EUR
Requires: cyvcf2.
"""
import argparse
import glob
import os
import sys

import numpy as np
from cyvcf2 import VCF

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from build_ld_refs import corr_clean, read_panel  # noqa: E402
from metrics import mismatch_diag  # noqa: E402

VCF_TPL = "ALL.chr{c}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"


def align_sign(a1, a2, ref, alt):
    """+1 if cohort effect allele (a2) is the 1000G ALT, -1 if it is REF, None if alleles differ."""
    a1, a2, ref, alt = a1.upper(), a2.upper(), ref.upper(), alt.upper()
    if {a1, a2} != {ref, alt}:
        return None
    return 1.0 if a2 == alt else -1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regions", required=True, help="directory of region .npz files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vcf-dir", required=True)
    ap.add_argument("--vcf-tpl", default=VCF_TPL)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--pop", default="EUR", help="1000G super-population used as the external reference")
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--half-width", type=int, default=1_500_000, help="bp scanned on each side of the lead")
    ap.add_argument("--maf", type=float, default=0.01)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pops = read_panel(args.panel)
    v0 = VCF(os.path.join(args.vcf_dir, args.vcf_tpl.format(c=22)), gts012=True)
    sidx = {s: i for i, s in enumerate(v0.samples)}; v0.close()
    cols = np.array([sidx[s] for s in pops[args.pop] if s in sidx], dtype=int)
    ne = len(cols)

    # ---- unique confident leads, each attached to its highest-PIP region ----
    leads = {}
    for fn in sorted(glob.glob(os.path.join(args.regions, "*.npz"))):
        d = np.load(fn, allow_pickle=True)
        gp, cs, pos, ch, a1, a2 = d["gold_pip"], d["gold_cs"], d["pos"], d["chrom"], d["a1"], d["a2"]
        for i in np.where((gp > 0.9) & (cs > 0))[0]:
            key = f"{int(ch[i])}:{int(pos[i])}:{a1[i]}:{a2[i]}"
            if key not in leads or gp[i] > leads[key][0]:
                leads[key] = (float(gp[i]), fn, int(ch[i]), int(pos[i]))
    print(f"{args.pop} samples {ne} | unique confident leads {len(leads)}", flush=True)

    existing = set(os.listdir(args.out))
    built, diags = 0, []
    for key, (_, fn, chrom, lpos) in leads.items():
        out_name = key.replace(":", "_") + ".npz"
        if out_name in existing:
            continue
        d = np.load(fn, allow_pickle=True)
        # (position, allele set) -> cohort record with the highest gold PIP
        umap = {}
        for j in range(len(d["pos"])):
            ak = frozenset([str(d["a1"][j]).upper(), str(d["a2"][j]).upper()])
            rec = (str(d["a1"][j]), str(d["a2"][j]), float(d["z"][j]), float(d["gold_pip"][j]),
                   int(d["gold_cs"][j]), str(d["rsid"][j]))
            cur = umap.get((int(d["pos"][j]), ak))
            if cur is None or rec[3] > cur[3]:
                umap[(int(d["pos"][j]), ak)] = rec
        start = max(int(d["pos"].min()), lpos - args.half_width)
        end = min(int(d["pos"].max()), lpos + args.half_width)

        vcf = VCF(os.path.join(args.vcf_dir, args.vcf_tpl.format(c=chrom)), gts012=True)
        rows = []
        for v in vcf(f"{chrom}:{start}-{end}"):
            if not v.is_snp or len(v.ALT) != 1:
                continue
            rec = umap.get((v.POS, frozenset([v.REF.upper(), v.ALT[0].upper()])))
            if rec is None:
                continue
            a1, a2, z, gp, cs, rsid = rec
            s = align_sign(a1, a2, v.REF, v.ALT[0])
            ge = v.gt_types[cols]
            if s is None or np.any(ge == 3):
                continue
            ge = ge.astype(np.float32)
            af = ge.sum() / (2 * ne)
            if min(af, 1 - af) < args.maf:
                continue
            rows.append((v.POS, ge, s * z, gp, cs, rsid, min(af, 1 - af), f"{chrom}:{v.POS}:{a1}:{a2}"))
        vcf.close()

        lead_rows = [r for r in rows if r[7] == key]
        if not lead_rows or lead_rows[0][3] <= 0.9 or len(rows) < 100:
            print(f"  skip {key} (lead not usable in the reference)", flush=True)
            continue
        rows.sort(key=lambda r: abs(r[0] - lpos))
        rows = sorted(rows[:args.window], key=lambda r: r[0])
        R = corr_clean(np.array([r[1] for r in rows], dtype=np.float32))
        z = np.array([r[2] for r in rows], dtype=np.float32)
        lead_idx = [i for i, r in enumerate(rows) if r[7] == key][0]
        diag, _ = mismatch_diag(z, R, lead_idx)
        diags.append(diag)
        np.savez_compressed(os.path.join(args.out, out_name),
                            R=R, z=z, maf=np.array([r[6] for r in rows], dtype=np.float32),
                            gold_pip=np.array([r[3] for r in rows], dtype=np.float32),
                            gold_cs=np.array([r[4] for r in rows], dtype=np.int64),
                            pos=np.array([r[0] for r in rows], dtype=np.int64),
                            rsid=np.array([r[5] for r in rows]), chrom=chrom, lead_pos=lpos,
                            lead_idx=lead_idx, p=len(rows), n_ref=ne, lead_key=key, diag=diag)
        built += 1
    if diags:
        dg = np.array(diags)
        print(f"BUILT {built} -> {args.out} | diag mean {dg.mean():.3f} median {np.median(dg):.3f} | "
              f"clean {(dg > 0.7).sum()} mid {((dg >= 0.3) & (dg <= 0.7)).sum()} low {(dg < 0.3).sum()}")
    else:
        print(f"BUILT {built} -> {args.out}")


if __name__ == "__main__":
    main()
