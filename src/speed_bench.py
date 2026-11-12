#!/usr/bin/env python
"""
speed_bench.py -- per-locus inference time.

Times, on the same benchmark loci and the same supplied LD:
  * DiffMap one-shot PIPs (a single forward pass; GPU if available),
  * NumPy SuSiE-RSS (susie_rss.py, CPU),
  * RSparsePro (optional, --rsparsepro-src; its core solver, CPU).
Timings are per locus and UN-BATCHED, which is conservative for DiffMap
(batching many loci into one forward pass is faster still).

Example
  python speed_bench.py --bench DATA/bench --ckpt runs/diffmap_big/diffmap_final.pt \
                        --n-loci 50 --rsparsepro-src /path/to/RSparsePro_LD
"""
import argparse
import glob
import os
import time

import numpy as np
import torch

from baselines import rsparsepro_run
from inference import cond_pip, to_tensors
from model import load_checkpoint
from susie_rss import susie_rss


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ref", default="R")
    ap.add_argument("--n-loci", type=int, default=50)
    ap.add_argument("--rsparsepro-src", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model, _ = load_checkpoint(args.ckpt, dev)
    files = sorted(glob.glob(os.path.join(args.bench, "locus*.npz")))[:args.n_loci]
    if not files:
        raise SystemExit(f"no locus*.npz files found in {args.bench}")
    loci = []
    for f in files:
        d = np.load(f)
        loci.append((d["z"], d["maf"], (d["R"] if args.ref == "R" else d[args.ref]), float(d["n"])))

    def sync():
        if dev.startswith("cuda"):
            torch.cuda.synchronize()

    # DiffMap: move inputs to the device first so we time the forward pass itself
    tens = [(to_tensors(z, m, R, dev), n) for z, m, R, n in loci]
    for (zt, mt, Rt), n in tens[:3]:                           # warm-up
        cond_pip(model, zt, mt, Rt, [], dev, n=n)
    sync(); t0 = time.perf_counter()
    for (zt, mt, Rt), n in tens:
        cond_pip(model, zt, mt, Rt, [], dev, n=n)
    sync(); t_dm = (time.perf_counter() - t0) / len(tens)

    t0 = time.perf_counter()
    for z, m, R, n in loci:
        susie_rss(z.astype(np.float64), R.astype(np.float64), n)
    t_su = (time.perf_counter() - t0) / len(loci)

    print(f"loci {len(loci)}  p={loci[0][0].shape[0]}  device={dev}")
    print(f"DiffMap one-shot : {1e3 * t_dm:9.2f} ms/locus")
    print(f"SuSiE (NumPy)    : {1e3 * t_su:9.2f} ms/locus   ({t_su / t_dm:6.1f}x DiffMap)")
    if args.rsparsepro_src:
        # NOTE: this includes RSparsePro's process start-up and file I/O; timing its core
        # solver (adaptive_train) directly gives a fairer number.
        t0 = time.perf_counter()
        for z, m, R, n in loci:
            rsparsepro_run(z, R, args.rsparsepro_src)
        t_rs = (time.perf_counter() - t0) / len(loci)
        print(f"RSparsePro (CLI) : {1e3 * t_rs:9.2f} ms/locus   ({t_rs / t_dm:6.1f}x DiffMap)")


if __name__ == "__main__":
    main()
