#!/usr/bin/env python
"""
aggregate_real.py -- pool per-locus real-data results into summary tables.

Reads the CSV files written by compare_real.py and prints:
  1. same-trait cross-cohort panel for height
     -- any trait whose name contains --focus-trait, per cohort;
  2. per-cohort pooled table over all traits;
  3. confidently-wrong rate and precision by reference-mismatch tier.

Each CSV is assigned to a cohort with --cohort NAME=GLOB (repeatable).

Example
  python aggregate_real.py --cohort UKB='results/res_ukb_*.csv' --cohort FinnGen='results/res_fg_*.csv'
"""
import argparse
import glob

import numpy as np
import pandas as pd

ORDER = ["susie", "susieR", "rsparsepro", "diffmap"]


def line(s):
    if len(s) == 0:
        return "(no loci)"
    H, Rg = s.n_high.sum(), s.n_right.sum()
    return (f"confW={s.confwrong.mean():.2f}  prec@.9={(Rg / H if H else float('nan')):.2f}  "
            f"inCS={s.lead_in_cs.mean():.2f}  lead_pip={s.lead_pip.mean():.3f}  "
            f"calls={H / len(s):.2f}  CSsz={s.cssize.mean():.1f}  n={len(s)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", action="append", required=True, help="NAME=GLOB of result CSVs")
    ap.add_argument("--focus-trait", default="height")
    args = ap.parse_args()

    frames = []
    for spec in args.cohort:
        name, pat = spec.split("=", 1)
        for f in sorted(glob.glob(pat)):
            df = pd.read_csv(f); df["cohort"] = name; frames.append(df)
    if not frames:
        raise SystemExit("no result CSVs matched")
    A = pd.concat(frames, ignore_index=True)
    methods = [m for m in ORDER if m in set(A.method)]

    print(f"\n===== {args.focus_trait.upper()} across cohorts (same trait, external reference vs in-sample gold) =====")
    F = A[A.trait.str.contains(args.focus_trait, case=False)]
    for co in sorted(F.cohort.unique()):
        print(f"\n[{co}  {args.focus_trait}]")
        for m in methods:
            print(f"  {m:11s} " + line(F[(F.cohort == co) & (F.method == m)]))

    print("\n===== each cohort, pooled over its traits =====")
    for co in sorted(A.cohort.unique()):
        print(f"\n[{co}  all traits]")
        for m in methods:
            print(f"  {m:11s} " + line(A[(A.cohort == co) & (A.method == m)]))

    print("\n===== by reference-mismatch tier (confW / prec@.9) =====")
    for co in sorted(A.cohort.unique()):
        for tier in ["clean", "mid", "low"]:
            sub = A[(A.cohort == co) & (A.tier == tier)]
            if len(sub) == 0:
                continue
            print(f"  [{co} | {tier}  n={int((sub.method == methods[-1]).sum())}]")
            for m in methods:
                s = sub[sub.method == m]; H, Rg = s.n_high.sum(), s.n_right.sum()
                print(f"    {m:11s} confW={s.confwrong.mean():.2f}  prec@.9={(Rg / H if H else float('nan')):.2f}")


if __name__ == "__main__":
    main()
