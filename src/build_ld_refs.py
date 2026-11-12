#!/usr/bin/env python
"""
build_ld_refs.py -- build the bank of LD blocks used for training and benchmarking.

Reads 1000 Genomes Phase 3 VCFs (GRCh37) and cuts each chromosome into blocks of
`--block-size` consecutive biallelic SNPs (MAF >= --maf in the GWAS population,
default EUR). For every block one .npz file is written containing:

  R            : full in-sample LD of the GWAS population (all 503 EUR samples).
                 This is the TRUE LD used to generate z-scores (README Eq. 1).
  ref_eur200,
  ref_eur100,
  ref_eur50    : LD from random subsets of 200/100/50 EUR samples (same-ancestry,
                 smaller reference panels)
  ref_afr200,
  ref_eas200,
  ref_amr200,
  ref_sas200   : LD from 200 samples of another 1000G super-population
                 (cross-ancestry reference panels)
  maf, pos, rsid, a1, a2, chrom, n

The reference subsets are drawn once (seed --ref-seed) and reused for every
block, mimicking a fixed external panel. Training uses refs
eur200/eur100/eur50/afr200/eas200; amr200 and sas200 are held out of training.
Monomorphic SNPs within a subset get zero off-diagonal correlation and unit
diagonal (corr_clean).

Example (training chromosomes 3-22, held-out test chromosomes 1-2):
  python build_ld_refs.py --vcf-dir DATA/1000g --panel DATA/1000g/integrated_call_samples_v3.20130502.ALL.panel \
         --chroms 3-22 --out-dir DATA/ld_refs_train --workers 4
  python build_ld_refs.py --vcf-dir DATA/1000g --panel DATA/1000g/integrated_call_samples_v3.20130502.ALL.panel \
         --chroms 1,2 --max-blocks 200 --out-dir DATA/ld_refs_test

Requires: cyvcf2 (and tabix-indexed VCFs).
"""
import argparse
import csv
import os

import numpy as np
from cyvcf2 import VCF


def read_panel(panel):
    """1000G panel file -> {super_population: [sample ids]} (super_pop is column 3)."""
    pops = {}
    with open(panel) as f:
        f.readline()
        for line in f:
            p = line.split()
            if len(p) >= 3:
                pops.setdefault(p[2], []).append(p[0])
    return pops


def corr_clean(D):
    """Variant-by-variant correlation of a dosage matrix D (variants x samples).

    NaNs from monomorphic variants are set to 0 and the diagonal forced to 1.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.corrcoef(D).astype(np.float32)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(R, 1.0)
    return R


def flush(dos, meta, out_dir, chrom, idx, mw, eur_cols, ref_cfg):
    """Write one block (.npz) and append a line to the per-chromosome manifest."""
    D = np.asarray(dos, dtype=np.float32)
    ne = len(eur_cols)
    af = D[:, eur_cols].sum(1) / (2 * ne)
    maf = np.minimum(af, 1 - af).astype(np.float32)
    out = dict(R=corr_clean(D[:, eur_cols]), maf=maf,
               pos=np.array([m[1] for m in meta], dtype=np.int64),
               rsid=np.array([m[2] for m in meta]),
               a1=np.array([m[3] for m in meta]), a2=np.array([m[4] for m in meta]),
               chrom=str(chrom), n=ne)
    for name, cols in ref_cfg.items():
        out["ref_" + name] = corr_clean(D[:, cols])
    fn = os.path.join(out_dir, f"chr{chrom}_block{idx:04d}.npz")
    np.savez_compressed(fn, **out)
    mw.writerow([fn, chrom, int(meta[0][1]), int(meta[-1][1]), D.shape[0], ne])


def process_chrom(args, chrom, eur_cols, used, ref_cfg):
    """Stream one chromosome and emit consecutive blocks of block_size SNPs."""
    vcf = VCF(os.path.join(args.vcf_dir, args.vcf_tpl.format(c=chrom)), gts012=True)
    mf = open(os.path.join(args.out_dir, f"manifest_chr{chrom}.csv"), "w", newline="")
    mw = csv.writer(mf)
    mw.writerow(["path", "chrom", "start", "end", "p", "n_eur"])
    dos, meta, idx = [], [], 0
    ne = len(eur_cols)
    for v in vcf(str(chrom)):
        if not v.is_snp or len(v.ALT) != 1:          # biallelic SNPs only
            continue
        g = v.gt_types                                # 0/1/2 ALT dosage, 3 = missing
        if np.any(g[used] == 3):
            continue
        g = g.astype(np.float32)
        ge = g[eur_cols]
        af = ge.sum() / (2 * ne)
        if min(af, 1 - af) < args.maf:
            continue
        dos.append(g)
        meta.append((v.CHROM, v.POS, v.ID or ".", v.REF, v.ALT[0]))
        if len(dos) >= args.block_size:
            flush(dos, meta, args.out_dir, chrom, idx, mw, eur_cols, ref_cfg)
            idx += 1
            dos, meta = [], []
            if args.max_blocks and idx >= args.max_blocks:
                break
    if dos and len(dos) >= args.min_block and (not args.max_blocks or idx < args.max_blocks):
        flush(dos, meta, args.out_dir, chrom, idx, mw, eur_cols, ref_cfg)
        idx += 1
    mf.close()
    vcf.close()
    print(f"chr{chrom}: {idx} blocks", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vcf-dir", required=True, help="directory with 1000G Phase 3 VCFs + .tbi")
    ap.add_argument("--vcf-tpl", default="ALL.chr{c}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz")
    ap.add_argument("--panel", required=True, help="integrated_call_samples_v3.20130502.ALL.panel")
    ap.add_argument("--gwas-pop", default="EUR", help="population whose LD generates the z-scores")
    ap.add_argument("--chroms", default="3-22", help="e.g. '3-22' or '1,2'")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--block-size", type=int, default=1000)
    ap.add_argument("--min-block", type=int, default=200, help="keep a trailing block only if this large")
    ap.add_argument("--maf", type=float, default=0.01)
    ap.add_argument("--max-blocks", type=int, default=0, help="0 = no limit (per chromosome)")
    ap.add_argument("--ref-seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1, help="chromosomes processed in parallel")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    chroms = []
    for part in args.chroms.split(","):
        if "-" in part:
            a, b = part.split("-"); chroms += list(range(int(a), int(b) + 1))
        else:
            chroms.append(int(part))

    pops = read_panel(args.panel)
    v0 = VCF(os.path.join(args.vcf_dir, args.vcf_tpl.format(c=chroms[0])), gts012=True)
    samples = list(v0.samples); v0.close()
    sidx = {s: i for i, s in enumerate(samples)}
    cols = lambda pop: np.array([sidx[s] for s in pops.get(pop, []) if s in sidx], dtype=int)
    eur_cols = cols(args.gwas_pop)
    pc = {p: cols(p) for p in ["AFR", "EAS", "AMR", "SAS"]}
    print(f"samples {len(samples)} | {args.gwas_pop} {len(eur_cols)} "
          + " ".join(f"{p} {len(c)}" for p, c in pc.items()), flush=True)

    # Fixed reference-panel subsets, reused for every block.
    rng = np.random.default_rng(args.ref_seed)
    sub = lambda c, k: np.sort(rng.choice(c, size=min(k, len(c)), replace=False))
    ref_cfg = {"eur200": sub(eur_cols, 200), "eur100": sub(eur_cols, 100), "eur50": sub(eur_cols, 50),
               "afr200": sub(pc["AFR"], 200), "eas200": sub(pc["EAS"], 200),
               "amr200": sub(pc["AMR"], 200), "sas200": sub(pc["SAS"], 200)}
    for k, v in ref_cfg.items():
        print(f"  ref {k}: n={len(v)}", flush=True)
    used = np.unique(np.concatenate([eur_cols] + list(pc.values())))

    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            pool.starmap(process_chrom, [(args, c, eur_cols, used, ref_cfg) for c in chroms])
    else:
        for c in chroms:
            process_chrom(args, c, eur_cols, used, ref_cfg)


if __name__ == "__main__":
    main()
