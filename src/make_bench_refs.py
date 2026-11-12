#!/usr/bin/env python
"""
make_bench_refs.py -- create the held-out simulation benchmark.

For each benchmark locus we pick a held-out LD block (built by build_ld_refs.py
from chromosomes NOT used for training), simulate z-scores from the block's TRUE
in-sample LD `R` (README Eq. 1), and store everything needed to fine-map the
locus under any reference:

  locusXXXX.npz:  z, gamma (true causal indicators), n, maf,
                  R (true LD), ref_eur200 ... ref_sas200 (alternative references)

Each block is loaded fresh (no caching) so memory stays bounded even with many
reference matrices per block.

Example:
  python make_bench_refs.py --bank DATA/ld_refs_test --out DATA/bench --n-loci 200 --seed 1234
"""
import argparse
import glob
import os

import numpy as np

from sim import psd_sqrt, simulate_locus


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True, help="directory of held-out LD blocks (*.npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-loci", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--k-max", type=int, default=3)
    ap.add_argument("--n-min", type=float, default=2e4)
    ap.add_argument("--n-max", type=float, default=5e5)
    ap.add_argument("--ncp-min", type=float, default=4.0)
    ap.add_argument("--ncp-max", type=float, default=12.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    blocks = sorted(glob.glob(os.path.join(args.bank, "*.npz")))
    if not blocks:
        raise SystemExit(f"no LD blocks found in {args.bank}")
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(blocks), size=args.n_loci, replace=len(blocks) < args.n_loci)

    for i, b in enumerate(picks):
        d = np.load(blocks[b], allow_pickle=True)
        R = d["R"].astype(np.float64)
        maf = d["maf"].astype(np.float32)
        z, gamma, n = simulate_locus(R, maf, psd_sqrt(R), rng, k_max=args.k_max,
                                     n_range=(args.n_min, args.n_max),
                                     ncp_range=(args.ncp_min, args.ncp_max))
        out = dict(z=z, gamma=gamma, n=np.float64(n), maf=maf, R=d["R"].astype(np.float32),
                   block=os.path.basename(blocks[b]))
        for k in d.files:
            if k.startswith("ref_"):
                out[k] = d[k].astype(np.float32)
        np.savez_compressed(os.path.join(args.out, f"locus{i:04d}.npz"), **out)
    print(f"wrote {args.n_loci} loci to {args.out}")


if __name__ == "__main__":
    main()
