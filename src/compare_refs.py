#!/usr/bin/env python
"""
compare_refs.py -- simulation robustness benchmark across LD references.

Every benchmark locus (make_bench_refs.py) was simulated with its TRUE in-sample
LD `R`. Here each method fine-maps the same z-scores while being given a
different LD matrix -- the matched R, or one of the alternative reference
panels -- and we report, per (reference, method):

    AUPRC | precision@PIP>0.9 (number of PIP>0.9 calls)

Methods
  susie        : NumPy SuSiE-RSS (susie_rss.py)                       always run
  susieR       : canonical susieR (needs R + susieR; --with-susieR)   optional
  rsparsepro   : RSparsePro (needs a local clone; --rsparsepro-src)   optional
  <name>       : one column per DiffMap checkpoint given as --ckpt name=path

Example (small and big DiffMap models side by side)
  python compare_refs.py --bench DATA/bench \
      --ckpt diffmap_small=runs/diffmap_small/diffmap_final.pt \
      --ckpt diffmap_big=runs/diffmap_big/diffmap_final.pt \
      --with-susieR --rsparsepro-src /path/to/RSparsePro_LD --save results/compare_refs.csv
"""
import argparse
import glob
import os

import numpy as np
import torch

from baselines import rsparsepro_run, susieR_available, susieR_run
from inference import cond_pip, to_tensors
from metrics import auprc, precision_at
from model import load_checkpoint
from susie_rss import susie_rss

REFS = ["R", "ref_eur200", "ref_eur100", "ref_eur50", "ref_afr200", "ref_eas200", "ref_amr200", "ref_sas200"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--ckpt", action="append", default=[], help="name=path (repeatable)")
    ap.add_argument("--refs", default=",".join(REFS))
    ap.add_argument("--with-susieR", action="store_true")
    ap.add_argument("--rsparsepro-src", default=None)
    ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--max-loci", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save", default="", help="optional CSV of the table")
    args = ap.parse_args()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    models = {}
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        models[name] = load_checkpoint(path, dev)[0]

    files = sorted(glob.glob(os.path.join(args.bench, "locus*.npz")))
    if args.max_loci:
        files = files[:args.max_loci]
    if not files:
        raise SystemExit(f"no locus*.npz files found in {args.bench}")
    with np.load(files[0]) as d0:
        refs = [r for r in args.refs.split(",") if r == "R" or r in d0.files]

    cols = ["susie"]
    if args.with_susieR:
        if susieR_available():
            cols.append("susieR")
        else:
            print("WARNING: susieR not available (set DIFFMAP_RSCRIPT); skipping it", flush=True)
    if args.rsparsepro_src:
        cols.append("rsparsepro")
    cols += list(models)

    res = {c: {rf: [] for rf in refs} for c in cols}
    labels = []
    for k, f in enumerate(files):
        d = np.load(f)
        z = d["z"].astype(np.float64); maf = d["maf"]; n = float(d["n"])
        labels.append(d["gamma"].astype(int))
        for rf in refs:
            Rk = d["R"] if rf == "R" else d[rf]
            R64 = Rk.astype(np.float64)
            res["susie"][rf].append(susie_rss(z, R64, n, L=args.L)[0])
            if "susieR" in res:
                res["susieR"][rf].append(susieR_run(z, R64, n, L=args.L)[0])
            if "rsparsepro" in res:
                res["rsparsepro"][rf].append(rsparsepro_run(z, R64, args.rsparsepro_src)[0])
            zt, mt, Rt = to_tensors(z, maf, Rk, dev)
            for name, m in models.items():
                res[name][rf].append(cond_pip(m, zt, mt, Rt, [], dev, n=n))   # one-shot PIPs
        if (k + 1) % 25 == 0:
            print(f"...{k + 1}/{len(files)} loci", flush=True)

    L = np.concatenate(labels)
    print(f"\nbench {args.bench}  loci {len(files)}   [AUPRC | precision@PIP>0.9 (n)]\n")
    print(f"{'reference':12s}" + "".join(f"{c:>22s}" for c in cols))
    rows = []
    for rf in refs:
        line = f"{rf:12s}"
        for c in cols:
            P = np.concatenate(res[c][rf])
            a = auprc(P, L); pr, nh = precision_at(P, L, 0.9)
            line += f"  {a:7.3f}|{pr:4.2f}({nh:4d})"
            rows.append(dict(reference=rf, method=c, auprc=a, precision_at_0p9=pr, n_calls=nh))
        print(line)
    if args.save:
        import pandas as pd
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        pd.DataFrame(rows).to_csv(args.save, index=False)
        print("saved", args.save)


if __name__ == "__main__":
    main()
