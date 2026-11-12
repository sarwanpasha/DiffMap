#!/usr/bin/env python
"""
compare_real.py -- real-data validation against in-sample-LD gold standards.

For every real locus built by build_real_loci.py, all methods receive the SAME
cohort z-scores and the SAME external 1000 Genomes LD, and are scored against the
published in-sample-LD fine-mapping (metrics.score_vs_gold):

  lead_pip   PIP given to the gold-standard lead
  top=lead   method's top variant is the gold lead
  inCS       gold lead falls inside one of the method's 95% credible sets
  confW      confidently-wrong calls per locus (method PIP > 0.9 but gold PIP < 0.1)
  prec@.9    fraction of method PIP > 0.9 calls with gold PIP > 0.5
  CSsz       mean credible-set size

Methods: susie (NumPy), susieR (canonical, if available), rsparsepro (if
--rsparsepro-src is given), diffmap (one-shot PIPs + Algorithm 1 credible sets).
Per-locus rows are written to --save (CSV) for aggregate_real.py.

Example
  python compare_real.py --loci DATA/ukb_finemap/real_loci_height --trait ukb_height \
         --ckpt runs/diffmap_big/diffmap_final.pt --n 100000 --with-susieR \
         --rsparsepro-src /path/to/RSparsePro_LD --save results/res_ukb_height.csv
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from baselines import rsparsepro_run, susieR_available, susieR_run  # noqa: E402
from inference import cond_pip, diffmap_cs, to_tensors  # noqa: E402
from metrics import mismatch_diag, score_vs_gold  # noqa: E402
from model import load_checkpoint  # noqa: E402
from susie_rss import credible_sets as susie_cs, susie_rss  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loci", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--trait", default="trait")
    ap.add_argument("--n", type=float, default=100000.0, help="GWAS sample size passed to all methods")
    ap.add_argument("--with-susieR", action="store_true")
    ap.add_argument("--rsparsepro-src", default=None)
    ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--lead-thr", type=float, default=0.1)
    ap.add_argument("--ld-thr", type=float, default=0.4)
    ap.add_argument("--save", default="")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model, _ = load_checkpoint(args.ckpt, dev)

    methods = ["susie"]
    if args.with_susieR:
        if susieR_available():
            methods.append("susieR")
        else:
            print("WARNING: susieR not available (set DIFFMAP_RSCRIPT); skipping it", flush=True)
    if args.rsparsepro_src:
        methods.append("rsparsepro")
    methods.append("diffmap")

    rows = []
    files = sorted(glob.glob(os.path.join(args.loci, "*.npz")))
    for f in files:
        d = np.load(f, allow_pickle=True)
        z = d["z"].astype(np.float64); R = d["R"].astype(np.float64)
        gold = d["gold_pip"].astype(np.float64); lead = int(d["lead_idx"])
        _, tier = mismatch_diag(z, R, lead)
        sc = {}
        pip, alpha = susie_rss(z, R, args.n, L=args.L)
        sc["susie"] = score_vs_gold(pip, susie_cs(alpha, R), lead, gold)
        if "susieR" in methods:
            sc["susieR"] = score_vs_gold(*susieR_run(z, R, args.n, L=args.L), lead, gold)
        if "rsparsepro" in methods:
            sc["rsparsepro"] = score_vs_gold(*rsparsepro_run(z, R, args.rsparsepro_src), lead, gold)
        zt, mt, Rt = to_tensors(z, d["maf"], R, dev)
        pdm = cond_pip(model, zt, mt, Rt, [], dev, n=args.n)
        cdm = diffmap_cs(model, zt, mt, Rt, R, dev, n=args.n, lead_thr=args.lead_thr, ld_thr=args.ld_thr)
        sc["diffmap"] = score_vs_gold(pdm, cdm, lead, gold)
        for m in methods:
            rows.append(dict(trait=args.trait, locus=os.path.basename(f), method=m, tier=tier, **sc[m]))

    if not rows:
        print(f"no loci found for {args.trait}")
        return
    df = pd.DataFrame(rows)
    print(f"\n[{args.trait}] loci={len(files)}")
    for m in methods:
        s = df[df.method == m]
        H, Rg = s.n_high.sum(), s.n_right.sum()
        print("  %-11s lead_pip=%.3f top=%.2f inCS=%.2f confW=%.2f CSsz=%.1f prec@.9=%.2f calls=%.2f" %
              (m, s.lead_pip.mean(), s.top_is_lead.mean(), s.lead_in_cs.mean(), s.confwrong.mean(),
               s.cssize.mean(), (Rg / H if H else float("nan")), H / len(s)))
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        df.to_csv(args.save, index=False)
        print("saved", args.save)


if __name__ == "__main__":
    main()
