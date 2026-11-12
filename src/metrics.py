"""
metrics.py -- evaluation metrics.

Simulation (known causal variants):
  * auprc(scores, labels)        : area under the precision-recall curve (ranking quality)
  * precision_at(pips, labels)   : fraction of PIP > 0.9 calls that are truly causal (calibration)
  * cs_counts(CSs, causal)       : raw counts used to compute credible-set coverage, power, size

Real data (in-sample-LD gold standard):
  * score_vs_gold(...)           : confidently-wrong calls, precision@0.9, lead-in-CS, ...
  * mismatch_diag(z, R, lead)    : per-locus reference-mismatch diagnostic and its tier
"""
import numpy as np


def auprc(scores, labels):
    """Average precision (step-wise area under the precision-recall curve)."""
    s = np.asarray(scores, dtype=float); l = np.asarray(labels, dtype=int)
    o = np.argsort(-s); ll = l[o]
    tp = np.cumsum(ll); fp = np.cumsum(1 - ll)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(1, ll.sum())
    ap, prev = 0.0, 0.0
    for pp, rr in zip(prec, rec):
        ap += pp * (rr - prev); prev = rr
    return float(ap)


def precision_at(pips, labels, thr=0.9):
    """(precision among calls with PIP > thr, number of such calls)."""
    P = np.asarray(pips); L = np.asarray(labels)
    hi = P > thr
    return (float(L[hi].mean()) if hi.sum() else float("nan")), int(hi.sum())


def cs_counts(CSs, causal):
    """Counts for credible-set summaries.

    Returns (n_sets, n_sets_containing_a_causal, n_causal, n_causal_captured, total_size):
      coverage = n_sets_containing_a_causal / n_sets
      power    = n_causal_captured / n_causal
      size     = total_size / n_sets
    """
    causal = set(int(c) for c in causal)
    n_cs = len(CSs)
    n_cov = sum(1 for cs in CSs if causal & set(cs))
    captured = (set().union(*[set(cs) for cs in CSs]) if CSs else set()) & causal
    return n_cs, n_cov, len(causal), len(captured), sum(len(cs) for cs in CSs)


def summarize_cs(rows, n_loci):
    """Aggregate a list of cs_counts tuples into (sets/locus, coverage, power, mean size)."""
    ncs, ncov, ncausal, capt, tsize = np.asarray(rows).sum(0)
    return (ncs / n_loci,
            ncov / ncs if ncs else float("nan"),
            capt / ncausal if ncausal else float("nan"),
            tsize / ncs if ncs else float("nan"))


def score_vs_gold(pip, CSs, lead, gold):
    """Score one real-data locus against the in-sample-LD gold standard.

    pip  : method PIPs under the external reference
    CSs  : method credible sets (lists of indices)
    lead : index of the gold-standard lead variant (gold PIP > 0.9 in a 95% CS)
    gold : gold-standard (in-sample LD) PIPs
    """
    pip = np.asarray(pip, dtype=float); gold = np.asarray(gold, dtype=float)
    high = pip > 0.9
    return dict(
        lead_pip=float(pip[lead]),                                    # PIP given to the gold lead
        top_is_lead=int(int(np.argmax(pip)) == lead),                 # top variant == gold lead
        lead_in_cs=int(any(lead in c for c in CSs)),                  # gold lead inside a 95% CS
        confwrong=int(np.sum(high & (gold < 0.1))),                   # PIP>0.9 but gold PIP<0.1
        n_high=int(high.sum()),                                       # number of PIP>0.9 calls
        n_right=int(np.sum(high & (gold > 0.5))),                     # PIP>0.9 calls supported by gold
        cssize=(float(np.mean([len(c) for c in CSs])) if CSs else 0.0),
        pipcorr=(float(np.corrcoef(pip, gold)[0, 1]) if np.std(pip) > 0 and np.std(gold) > 0 else 0.0),
    )


def mismatch_diag(z, R, lead):
    """Reference-mismatch diagnostic: corr( z , R[:, lead] * z[lead] ).

    High when the external reference reproduces the cohort's association pattern
    around the lead; low when it does not. Tiers: clean > 0.7, mid 0.3-0.7, low < 0.3.
    """
    z = np.asarray(z, dtype=float)
    pred = np.asarray(R, dtype=float)[:, lead] * z[lead]
    d = float(np.corrcoef(z, pred)[0, 1]) if np.std(pred) > 0 else 0.0
    tier = "clean" if d > 0.7 else ("low" if d < 0.3 else "mid")
    return d, tier
