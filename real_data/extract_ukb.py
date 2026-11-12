#!/usr/bin/env python
"""
extract_ukb.py -- pull per-region data for selected traits out of the UK Biobank
in-sample-LD SuSiE fine-mapping release (Zenodo 10.5281/zenodo.7055906).

The single feather file contains, for 10 UK Biobank traits (N = 100k), every
variant of every fine-mapped 3 Mb region with:
  beta, se            -> marginal z = beta / se         (the GWAS summary statistics)
  prob, cs            -> gold-standard PIP and 95% credible-set id from SuSiE run
                         with IN-SAMPLE LD (cs = -1: not in a credible set)
Positions are GRCh37 (same build as 1000 Genomes Phase 3).

For each trait we keep only "test regions" -- regions containing at least one
confident gold-standard signal (a variant with prob > 0.9 inside a credible set)
-- and write one .npz per region into <out-base>/<trait>_regions/.

Example (five quantitative traits):
  python extract_ukb.py --src DATA/ukb_finemap/UKB.susie.100k.snp.feather \
                        --out-base DATA/ukb_finemap --traits Height,Plt,eBMD,HbA1c,RBC
Requires: pyarrow, pandas. Reading the 4.5 GB file needs ~20-30 GB RAM.
"""
import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as pf

COLS = ["rsid", "chromosome", "position", "allele1", "allele2", "maf", "beta", "se", "p",
        "prob", "cs", "region", "trait"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--traits", default="Height,Plt,eBMD,HbA1c,RBC")
    args = ap.parse_args()

    print("reading feather ...", flush=True)
    t = pf.read_table(args.src, columns=COLS)
    print(f"{'trait':8s}{'regions':>9s}{'signals':>9s}{'conf>0.9':>9s}{'testReg':>9s}")
    for tr in args.traits.split(","):
        df = t.filter(pc.equal(t.column("trait"), pa.scalar(tr))).to_pandas()
        df["z"] = df["beta"] / df["se"]
        nsig = df[df.cs > 0].groupby(["region", "cs"]).ngroups
        conf = df[(df["prob"] > 0.9) & (df["cs"] > 0)]
        test = sorted(conf.region.unique())
        out = os.path.join(args.out_base, tr.lower() + "_regions")
        os.makedirs(out, exist_ok=True)
        for rg in test:
            g = df[df.region == rg].sort_values("position")
            np.savez(os.path.join(out, rg.replace(".", "_").replace(":", "_") + ".npz"),
                     rsid=g.rsid.values.astype(str), chrom=g.chromosome.values.astype(np.int64),
                     pos=g.position.values.astype(np.int64),
                     a1=g.allele1.values.astype(str), a2=g.allele2.values.astype(str),  # allele2 = effect allele
                     maf=g.maf.values.astype(float), z=g.z.values.astype(float),
                     gold_pip=g["prob"].values.astype(float), gold_cs=g.cs.values.astype(np.int64),
                     region=str(rg))
        print(f"{tr:8s}{df.region.nunique():9d}{nsig:9d}{len(conf):9d}{len(test):9d}", flush=True)


if __name__ == "__main__":
    main()
