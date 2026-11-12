#!/usr/bin/env python
"""
predict.py -- fine-map ONE locus with a trained DiffMap model (plain-text in, TSV out).

Inputs
  --z    tab-separated file with a header and columns  SNP  Z  [MAF]
         (MAF is optional; if absent 0.25 is used for every variant)
  --ld   p x p LD (correlation) matrix, whitespace separated, rows in the same
         order as the SNPs in --z  (this is the reference panel you supply)
  --n    GWAS sample size
  --ckpt DiffMap checkpoint written by train.py

Output (--out, tab separated): SNP  Z  PIP  CS
  PIP = one-shot posterior inclusion probability (one forward pass, README Eq. 3)
  CS  = 95% credible-set id from iterative conditioning (README Algorithm 1), -1 if none

Example (files shipped in demo/sample_data/):
  python predict.py --z ../demo/sample_data/example_locus_z.tsv \
                    --ld ../demo/sample_data/example_locus_ld.txt --n 100000 \
                    --ckpt ../demo/outputs/demo_model/diffmap_final.pt --out example_pips.tsv
"""
import argparse

import numpy as np
import pandas as pd
import torch

from inference import cond_pip, diffmap_cs, to_tensors
from model import load_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z", required=True)
    ap.add_argument("--ld", required=True)
    ap.add_argument("--n", type=float, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="diffmap_pips.tsv")
    ap.add_argument("--lead-thr", type=float, default=0.1)
    ap.add_argument("--ld-thr", type=float, default=0.4)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    df = pd.read_csv(args.z, sep="\t")
    R = np.loadtxt(args.ld)
    if R.shape != (len(df), len(df)):
        raise SystemExit(f"LD matrix is {R.shape} but the z file has {len(df)} SNPs")
    maf = df["MAF"].to_numpy(float) if "MAF" in df.columns else np.full(len(df), 0.25)

    model, _ = load_checkpoint(args.ckpt, dev)
    zt, mt, Rt = to_tensors(df["Z"].to_numpy(float), maf, R, dev)
    pip = cond_pip(model, zt, mt, Rt, [], dev, n=args.n)
    sets = diffmap_cs(model, zt, mt, Rt, R, dev, n=args.n, lead_thr=args.lead_thr, ld_thr=args.ld_thr)

    cs = np.full(len(df), -1, dtype=int)
    for k, s in enumerate(sets, start=1):
        cs[s] = k
    out = pd.DataFrame({"SNP": df["SNP"], "Z": df["Z"], "PIP": np.round(pip, 5), "CS": cs})
    out.to_csv(args.out, sep="\t", index=False)
    top = out.sort_values("PIP", ascending=False).head(5)
    print(f"{len(sets)} credible set(s); top variants:\n{top.to_string(index=False)}\nwrote {args.out}")


if __name__ == "__main__":
    main()
