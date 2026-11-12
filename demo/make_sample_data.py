#!/usr/bin/env python
"""
make_sample_data.py -- build the small SYNTHETIC sample dataset used by the demo.

Nothing here is real genetic data. Genotypes are simulated with a simple
haplotype-copying (Li & Stephens-style) model so that the sample data has
realistic block-like LD and realistic LD *differences between populations*,
which is exactly what DiffMap must be robust to. Everything is deterministic
given --seed, needs only NumPy/SciPy, and runs in well under a minute.

Five synthetic populations stand in for the 1000 Genomes super-populations and
use the same names and sample sizes, so every pipeline script runs on the demo
data unchanged:
  EUR (503)  base population (its LD generates the z-scores)
  SAS (489)  close to EUR      (low divergence)
  EAS (504)  moderately diverged from EUR
  AFR (661)  most diverged, more haplotype diversity and shorter-range LD
  AMR (347)  admixed: mixture of EUR and AFR founder haplotypes
Each LD block stores the same keys as build_ld_refs.py:
  R (EUR in-sample LD), ref_eur200/100/50, ref_afr200, ref_eas200, ref_amr200, ref_sas200, maf, ...

Outputs (in --out, default demo/sample_data):
  ld_bank_train/   LD blocks used to TRAIN the demo model         (like chromosomes 3-22)
  ld_bank_test/    held-out LD blocks for the benchmark            (like chromosomes 1-2)
  real_like_loci/  "real-data-style" loci (format of real_data/build_real_loci.py): a separate
                   cohort population generates z; the gold standard is SuSiE run with
                   the cohort's IN-SAMPLE LD; R stored in the file is an EXTERNAL
                   EUR reference panel. The true causal vector is stored as `gamma`.
  example_locus_z.tsv / example_locus_ld.txt
                   one locus as plain text, the input format of src/predict.py
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from sim import psd_sqrt, simulate_locus  # noqa: E402
from susie_rss import credible_sets, susie_rss  # noqa: E402
from metrics import mismatch_diag  # noqa: E402

POP_N = {"EUR": 503, "SAS": 489, "EAS": 504, "AFR": 661, "AMR": 347}


def corr_clean(D):
    """Variant x variant correlation; monomorphic variants -> 0 off-diagonal, 1 on the diagonal."""
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.corrcoef(D)
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)
    return R.astype(np.float32)


def mutate(H, rate, rng):
    """Flip each allele of the founder haplotypes H (K x S) with probability `rate`."""
    return np.where(rng.random(H.shape) < rate, 1 - H, H)


def copy_haplotypes(F, n_hap, rho, rng):
    """Sample n_hap haplotypes as mosaics of founders F (K x S).

    At every site a haplotype switches to a uniformly chosen founder with
    probability rho[s] (site-specific recombination; hotspots create LD blocks).
    """
    K, S = F.shape
    src = rng.integers(K, size=n_hap)
    out = np.empty((n_hap, S), dtype=np.int8)
    for s in range(S):
        sw = rng.random(n_hap) < rho[s]
        src = np.where(sw, rng.integers(K, size=n_hap), src)
        out[:, s] = F[src, s]
    return out


def make_block(rng, S_raw=170, p=100, cohort_n=0):
    """Simulate one genomic segment for all populations; return dosage matrices.

    Returns dict pop -> (n_individuals x p) dosages, keeping the first p SNPs with
    EUR MAF >= 0.01, or None if fewer than p SNPs pass the filter.
    """
    K0 = 30
    f = np.clip(rng.beta(0.6, 0.6, size=S_raw), 0.02, 0.98)
    anc = (rng.random((K0, S_raw)) < f).astype(np.int8)                 # ancestral haplotypes
    rho = 0.004 * np.where(rng.random(S_raw) < 0.06, 60.0, 1.0)         # background + hotspots

    founders = {
        "EUR": mutate(anc[rng.choice(K0, 14, replace=False)], 0.02, rng),
        "SAS": None, "EAS": None, "AFR": None, "AMR": None,
    }
    founders["SAS"] = mutate(founders["EUR"], 0.04, rng)
    founders["EAS"] = mutate(anc[rng.choice(K0, 12, replace=False)], 0.06, rng)
    founders["AFR"] = mutate(anc, 0.10, rng)                              # more, more diverse founders
    founders["AMR"] = np.vstack([founders["EUR"], founders["AFR"][:10]])  # admixed founder pool
    rho_pop = {"EUR": rho, "SAS": rho, "EAS": rho * 1.3, "AFR": rho * 2.5, "AMR": rho * 1.5}

    sizes = dict(POP_N)
    if cohort_n:
        # a separate GWAS cohort: an isolate derived from EUR (drift = founder loss + mutation)
        founders["COHORT"] = mutate(founders["EUR"][rng.choice(14, 10, replace=False)], 0.03, rng)
        rho_pop["COHORT"] = rho
        sizes["COHORT"] = cohort_n

    G = {}
    for pop, n in sizes.items():
        h = copy_haplotypes(founders[pop], 2 * n, rho_pop[pop], rng)
        G[pop] = (h[0::2] + h[1::2]).astype(np.float32)                   # diploid dosages 0/1/2
    af = G["EUR"].mean(0) / 2
    keep = np.where(np.minimum(af, 1 - af) >= 0.01)[0]
    if len(keep) < p:
        return None
    keep = keep[:p]
    return {pop: g[:, keep] for pop, g in G.items()}


def block_record(G, rng_ref, idx):
    """LD block in the same .npz layout as build_ld_refs.py."""
    eur = G["EUR"]
    af = eur.mean(0) / 2
    sub = lambda X, k: X[np.sort(rng_ref.choice(X.shape[0], size=min(k, X.shape[0]), replace=False))]
    rec = dict(R=corr_clean(eur.T), maf=np.minimum(af, 1 - af).astype(np.float32),
               pos=np.arange(eur.shape[1], dtype=np.int64) * 1000 + 1, chrom=f"synthetic{idx}",
               n=eur.shape[0])
    rec["ref_eur200"] = corr_clean(sub(eur, 200).T)
    rec["ref_eur100"] = corr_clean(sub(eur, 100).T)
    rec["ref_eur50"] = corr_clean(sub(eur, 50).T)
    for pop in ["AFR", "EAS", "AMR", "SAS"]:
        rec[f"ref_{pop.lower()}200"] = corr_clean(sub(G[pop], 200).T)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "sample_data"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--p", type=int, default=100, help="SNPs per LD block (full scale: 1000)")
    ap.add_argument("--n-train", type=int, default=40, help="training LD blocks")
    ap.add_argument("--n-test", type=int, default=12, help="held-out LD blocks")
    ap.add_argument("--n-real", type=int, default=15, help="real-data-style loci")
    ap.add_argument("--cohort-n", type=int, default=2000, help="size of the synthetic GWAS cohort")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rng_ref = np.random.default_rng(args.seed + 1)

    # ---------------- LD banks ----------------
    for name, n_blocks in [("ld_bank_train", args.n_train), ("ld_bank_test", args.n_test)]:
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        i = 0
        while i < n_blocks:
            G = make_block(rng, p=args.p)
            if G is None:
                continue
            np.savez_compressed(os.path.join(d, f"block{i:04d}.npz"), **block_record(G, rng_ref, i))
            i += 1
        print(f"{name}: {n_blocks} blocks of p={args.p}", flush=True)

    # ---------------- real-data-style loci ----------------
    d = os.path.join(args.out, "real_like_loci")
    os.makedirs(d, exist_ok=True)
    n_gwas, built, tries = 100000.0, 0, 0
    while built < args.n_real and tries < 50 * args.n_real:
        tries += 1
        G = make_block(rng, p=args.p, cohort_n=args.cohort_n)
        if G is None:
            continue
        R_in = corr_clean(G["COHORT"].T).astype(np.float64)        # cohort in-sample LD
        R_ext = corr_clean(G["EUR"].T)                              # external reference panel
        af = G["COHORT"].mean(0) / 2
        maf = np.minimum(af, 1 - af).astype(np.float32)
        z, gamma, _ = simulate_locus(R_in, maf, psd_sqrt(R_in), rng, k_max=2, n_range=(n_gwas, n_gwas))
        # gold standard = SuSiE with IN-SAMPLE LD (as in the published releases)
        gold, alpha = susie_rss(z, R_in, n_gwas, L=10)
        sets = credible_sets(alpha, R_in)
        gold_cs = np.full(len(z), -1, dtype=np.int64)
        for k, s in enumerate(sets, start=1):
            gold_cs[s] = k
        conf = np.where((gold > 0.9) & (gold_cs > 0))[0]
        if len(conf) == 0:
            continue                                                # keep loci with a confident gold signal
        lead = int(conf[np.argmax(gold[conf])])
        diag, tier = mismatch_diag(z, R_ext, lead)
        np.savez_compressed(os.path.join(d, f"locus{built:03d}.npz"),
                            R=R_ext, z=z, maf=maf, gold_pip=gold.astype(np.float32), gold_cs=gold_cs,
                            lead_idx=lead, diag=diag, gamma=gamma, p=len(z), lead_key=f"synthetic{built}")
        if built == 0:
            # the same locus as plain text: input format of src/predict.py
            with open(os.path.join(args.out, "example_locus_z.tsv"), "w") as fz:
                fz.write("SNP\tZ\tMAF\n")
                for j in range(len(z)):
                    fz.write(f"snp{j + 1}\t{z[j]:.5f}\t{maf[j]:.4f}\n")
            np.savetxt(os.path.join(args.out, "example_locus_ld.txt"), R_ext, fmt="%.5f")
            causal = ", ".join(f"snp{j + 1}" for j in np.where(gamma == 1)[0])
            with open(os.path.join(args.out, "example_locus_truth.txt"), "w") as ft:
                ft.write(f"true causal variant(s): {causal}\n"
                         f"gold-standard (in-sample LD) lead: snp{lead + 1}  gold PIP={gold[lead]:.3f}\n"
                         f"GWAS sample size n = {int(n_gwas)}\n")
        built += 1
    print(f"real_like_loci: {built} loci (cohort in-sample LD -> gold; external EUR LD stored as R)", flush=True)
    print(f"sample data written to {args.out}")


if __name__ == "__main__":
    main()
