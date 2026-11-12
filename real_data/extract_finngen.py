#!/usr/bin/env python
"""
extract_finngen.py -- pull per-region data out of FinnGen R12 SuSiE fine-mapping
files and lift coordinates from GRCh38 to GRCh37 (the build of 1000 Genomes).

Each <ENDPOINT>.SUSIE.snp.bgz (public FinnGen release) lists every variant of
every fine-mapped region with beta, se (-> z), maf, prob (gold-standard
in-sample-LD PIP) and cs (credible-set id, -1 if none). For every region that
contains a confident gold signal (prob > 0.9 inside a credible set) we:
  * keep autosomes, strip the 'chr' prefix,
  * lift each position hg38 -> hg19 with the UCSC chain (variants that fail to
    lift, or lift to another chromosome, are dropped; on a minus-strand lift
    both alleles are complemented),
  * write one .npz per region into <src>/<endpoint_lowercase>_regions/
    in exactly the format consumed by build_real_loci.py.
The printed 'lift%' column is the fraction of variants that lifted (typically >99%).

Example
  python extract_finngen.py --src DATA/finngen --chain DATA/finngen/hg38ToHg19.over.chain.gz
Requires: pandas, pyliftover.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from pyliftover import LiftOver

COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
COLS = ["region", "chromosome", "position", "allele1", "allele2", "maf", "beta", "se", "prob", "cs", "rsid"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="directory with <ENDPOINT>.SUSIE.snp.bgz files")
    ap.add_argument("--chain", required=True, help="hg38ToHg19.over.chain.gz")
    ap.add_argument("--min-snps", type=int, default=100)
    args = ap.parse_args()
    lo = LiftOver(args.chain)

    def lift(cstr, pos38):
        r = lo.convert_coordinate(cstr, int(pos38) - 1)          # pyliftover is 0-based
        if not r:
            return None
        return r[0][0], r[0][1] + 1, r[0][2]                      # chrom, 1-based pos, strand

    print(f"{'endpoint':14s}{'regions':>9s}{'conf>0.9':>9s}{'testReg':>9s}{'lift%':>8s}", flush=True)
    for fn in sorted(glob.glob(os.path.join(args.src, "*.SUSIE.snp.bgz"))):
        ep = os.path.basename(fn).replace(".SUSIE.snp.bgz", "")
        df = pd.read_csv(fn, sep="\t", compression="gzip", usecols=COLS)
        df["chrom"] = df["chromosome"].astype(str).str.replace("chr", "", regex=False)
        df = df[df["chrom"].isin([str(i) for i in range(1, 23)])].copy()
        df["chrom"] = df["chrom"].astype(int)
        df["z"] = df["beta"] / df["se"]
        conf = df[(df["prob"] > 0.9) & (df["cs"] > 0)]
        out = os.path.join(args.src, ep.lower() + "_regions")
        os.makedirs(out, exist_ok=True)
        n_ok, n_all, written = 0, 0, 0
        for rg in sorted(conf.region.unique()):
            g = df[df.region == rg].sort_values("position")
            chrom = int(g["chrom"].iloc[0]); cstr = f"chr{chrom}"
            rows = []
            for r in g.itertuples(index=False):
                n_all += 1
                L = lift(cstr, r.position)
                if L is None or L[0] != cstr:
                    continue
                x1, x2 = str(r.allele1).upper(), str(r.allele2).upper()
                if L[2] == "-":
                    x1, x2 = COMP.get(x1, x1), COMP.get(x2, x2)
                rows.append((L[1], str(r.rsid), x1, x2, float(r.maf), float(r.z), float(r.prob), int(r.cs)))
                n_ok += 1
            if len(rows) < args.min_snps:
                continue
            rows.sort(key=lambda q: q[0])
            np.savez(os.path.join(out, f"{chrom}_{rows[0][0]}_{rows[-1][0]}.npz"),
                     rsid=np.array([q[1] for q in rows]), chrom=np.full(len(rows), chrom, dtype=np.int64),
                     pos=np.array([q[0] for q in rows], dtype=np.int64),
                     a1=np.array([q[2] for q in rows]), a2=np.array([q[3] for q in rows]),  # allele2 = effect
                     maf=np.array([q[4] for q in rows]), z=np.array([q[5] for q in rows]),
                     gold_pip=np.array([q[6] for q in rows]), gold_cs=np.array([q[7] for q in rows], dtype=np.int64),
                     region=str(rg))
            written += 1
        print(f"{ep:14s}{df.region.nunique():9d}{len(conf):9d}{written:9d}{100 * n_ok / max(n_all, 1):7.1f}%",
              flush=True)


if __name__ == "__main__":
    main()
