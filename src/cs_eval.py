#!/usr/bin/env python
"""
cs_eval.py -- credible-set comparison on the simulation benchmark.

For every benchmark locus and two conditions -- matched LD ("R") and a
mismatched reference (--ref, default ref_eur100) -- compute 95% credible sets with
  * SuSiE (NumPy SuSiE-RSS, or canonical susieR with --with-susieR),
  * RSparsePro (optional, --rsparsepro-src),
  * DiffMap (iterative conditioning, README Algorithm 1; inference.diffmap_cs),
and report per method/condition:
  nCS/locus | coverage (fraction of sets containing a true causal variant)
            | power (fraction of causal variants captured) | mean set size

Example
  python cs_eval.py --bench DATA/bench --ckpt runs/diffmap_big/diffmap_final.pt \
                    --ref ref_eur100 --lead-thr 0.1 --ld-thr 0.4 --with-susieR
"""
import argparse
import glob
import os

import numpy as np
import torch

from baselines import rsparsepro_run, susieR_available, susieR_run
from inference import diffmap_cs, to_tensors
from metrics import cs_counts, summarize_cs
from model import load_checkpoint
from susie_rss import credible_sets as susie_cs, susie_rss


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ref", default="ref_eur100", help="mismatched reference to compare with matched R")
    ap.add_argument("--with-susieR", action="store_true", help="use canonical susieR instead of NumPy SuSiE")
    ap.add_argument("--rsparsepro-src", default=None)
    ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--lead-thr", type=float, default=0.1)
    ap.add_argument("--ld-thr", type=float, default=0.4)
    ap.add_argument("--max-loci", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model, _ = load_checkpoint(args.ckpt, dev)

    use_R = args.with_susieR and susieR_available()
    if args.with_susieR and not use_R:
        print("WARNING: susieR not available; using NumPy SuSiE", flush=True)

    files = sorted(glob.glob(os.path.join(args.bench, "locus*.npz")))
    if args.max_loci:
        files = files[:args.max_loci]
    if not files:
        raise SystemExit(f"no locus*.npz files found in {args.bench}")
    conds = [("matched", "R"), ("mismatch", args.ref)]
    methods = ["susie"] + (["rsparsepro"] if args.rsparsepro_src else []) + ["diffmap"]
    agg = {(m, c): [] for m in methods for c, _ in conds}

    for fi, f in enumerate(files):
        d = np.load(f)
        z = d["z"].astype(np.float64); n = float(d["n"])
        causal = np.where(d["gamma"].astype(int) == 1)[0]
        for cname, rk in conds:
            Rk = d["R"] if rk == "R" else d[rk]
            R64 = Rk.astype(np.float64)
            if use_R:
                sets = susieR_run(z, R64, n, L=args.L)[1]
            else:
                _, alpha = susie_rss(z, R64, n, L=args.L)
                sets = susie_cs(alpha, R64)
            agg[("susie", cname)].append(cs_counts(sets, causal))
            if args.rsparsepro_src:
                agg[("rsparsepro", cname)].append(cs_counts(rsparsepro_run(z, R64, args.rsparsepro_src)[1], causal))
            zt, mt, Rt = to_tensors(z, d["maf"], Rk, dev)
            dm = diffmap_cs(model, zt, mt, Rt, R64, dev, n=n, lead_thr=args.lead_thr, ld_thr=args.ld_thr)
            agg[("diffmap", cname)].append(cs_counts(dm, causal))
        if (fi + 1) % 25 == 0:
            print(f"...{fi + 1}/{len(files)} loci", flush=True)

    print(f"\nbench {args.bench}  loci {len(files)}  mismatch-ref={args.ref}\n")
    print(f"{'method':11s}{'cond':10s}{'nCS/locus':>11s}{'coverage':>10s}{'power':>8s}{'meansize':>10s}")
    for m in methods:
        for cname, _ in conds:
            ncs, cov, pw, sz = summarize_cs(agg[(m, cname)], len(files))
            print(f"{m:11s}{cname:10s}{ncs:11.2f}{cov:10.3f}{pw:8.3f}{sz:10.1f}")


if __name__ == "__main__":
    main()
