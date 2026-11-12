#!/usr/bin/env python
"""
cs_sweep.py -- operating characteristics of DiffMap credible sets.

Sweeps the two thresholds of the iterative-conditioning extractor (README
Algorithm 1): the lead threshold tau_lead (when to stop adding signals) and the
LD threshold tau_ld (which variants may join a lead's set), for several
references, and prints sets/locus, coverage, power and mean set size.
The default balanced operating point is tau_lead = 0.1, tau_ld = 0.4.

Example
  python cs_sweep.py --bench DATA/bench --ckpt runs/diffmap_big/diffmap_final.pt \
                     --refs R,ref_eur100,ref_afr200
"""
import argparse
import glob
import os

import numpy as np
import torch

from inference import diffmap_cs, to_tensors
from metrics import cs_counts, summarize_cs
from model import load_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--refs", default="R,ref_eur100,ref_afr200")
    ap.add_argument("--lead-grid", default="0.05,0.1,0.2")
    ap.add_argument("--ld-grid", default="0.25,0.4,0.5,0.6")
    ap.add_argument("--max-loci", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model, _ = load_checkpoint(args.ckpt, dev)

    files = sorted(glob.glob(os.path.join(args.bench, "locus*.npz")))
    if args.max_loci:
        files = files[:args.max_loci]
    if not files:
        raise SystemExit(f"no locus*.npz files found in {args.bench}")
    refs = args.refs.split(",")
    grid = [(float(a), float(b)) for a in args.lead_grid.split(",") for b in args.ld_grid.split(",")]

    print(f"loci {len(files)}\n{'ref':12s}{'lead':>6s}{'ld':>6s}{'nCS':>7s}{'cov':>8s}{'power':>8s}{'size':>8s}")
    for rf in refs:
        for lt, ld in grid:
            rows = []
            for f in files:
                d = np.load(f)
                Rk = d["R"] if rf == "R" else d[rf]
                zt, mt, Rt = to_tensors(d["z"], d["maf"], Rk, dev)
                causal = np.where(d["gamma"].astype(int) == 1)[0]
                CSs = diffmap_cs(model, zt, mt, Rt, Rk.astype(np.float64), dev, n=float(d["n"]),
                                 lead_thr=lt, ld_thr=ld)
                rows.append(cs_counts(CSs, causal))
            ncs, cov, pw, sz = summarize_cs(rows, len(files))
            print(f"{rf:12s}{lt:6.2f}{ld:6.2f}{ncs:7.2f}{cov:8.3f}{pw:8.3f}{sz:8.1f}", flush=True)
        print()


if __name__ == "__main__":
    main()
