# DiffMap

**Amortized, calibrated statistical fine-mapping that stays robust when the LD reference is wrong.**

Statistical fine-mapping turns a GWAS association signal into posterior inclusion probabilities
(PIPs) and credible sets for the variants of a locus. Summary-statistic methods need an LD
(correlation) matrix that matches the GWAS sample, but in practice an external reference panel
(for example 1000 Genomes) is used instead. When that reference is mismatched, standard methods
produce confident but wrong calls.

DiffMap is a neural network that is trained **once** on simulated loci and then fine-maps any new
locus in **one forward pass**:

* **Amortized.** No per-locus optimization: a locus of 1,000 variants takes milliseconds on a GPU.
* **Robust to LD misspecification.** During training the network is deliberately shown *other*
  LD matrices than the one that generated the data (smaller panels, other ancestries), so it learns
  to stay calibrated when the supplied reference is wrong.
* **Credible sets** are obtained by iteratively conditioning the same network on the signals it has
  already found.

---

## Contents

1. [Installation](#1-installation)
2. [Quick start (about 10 minutes, CPU only)](#2-quick-start-about-10-minutes-cpu-only)
3. [Fine-map your own locus](#3-fine-map-your-own-locus)
4. [How the method works](#4-how-the-method-works)
5. [Repository layout](#5-repository-layout)
6. [Synthetic sample data](#6-synthetic-sample-data)
7. [Full-scale pipeline on public data](#7-full-scale-pipeline-on-public-data)
8. [Baselines: susieR and RSparsePro](#8-baselines-susier-and-rsparsepro)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Installation

Python ≥ 3.9. A GPU is **not** needed for the quick start.

```bash
git clone https://github.com/sarwanpasha/DiffMap.git
cd DiffMap
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas torch            # enough for the quick start
pip install cyvcf2 pyarrow pyliftover           # only for the full-scale pipeline
```

or simply `pip install -r requirements.txt`.

---

## 2. Quick start (about 10 minutes, CPU only)

```bash
cd demo
bash run_demo.sh             # full demo, ~5-15 min on a laptop CPU
QUICK=1 bash run_demo.sh     # shorter training, ~2-4 min
```

No downloads are needed: the script first **generates a small synthetic dataset**
(`demo/sample_data/`, about 13 MB, deterministic, ~10 seconds) and then runs the same scripts
used at full scale. Everything is written to `demo/outputs/`.

| step | script | what to look at |
|---|---|---|
| 1 | `demo/make_sample_data.py` | synthetic LD blocks for five populations and "real-data-style" loci |
| 2 | `src/make_bench_refs.py` | 40 benchmark loci simulated from the *true* LD of held-out blocks |
| 3 | `src/train.py` | training log; `P(causal)` (mean predicted probability at masked causal variants) should rise well above `P(noncausal)` |
| 4 | `src/compare_refs.py` | **AUPRC \| precision at PIP > 0.9 (number of such calls)** for each LD reference given to the methods. SuSiE is near-perfect with the matched matrix `R` but its precision collapses, and its number of confident calls explodes, with mismatched references. Compare the DiffMap column, including `ref_amr200` and `ref_sas200`, which the model never saw during training |
| 5 | `src/cs_eval.py` | 95% credible-set coverage, power and size, matched vs mismatched reference |
| 6 | `real_data/compare_real.py`, `aggregate_real.py` | loci whose "gold standard" is SuSiE run with the cohort's own (in-sample) LD, while the methods only get an external reference: confidently-wrong calls per locus (`confW`), precision, and whether the gold lead is in the credible set (`inCS`) |
| 7 | `src/predict.py` | PIPs and credible sets for one locus given as text files; compare with `sample_data/example_locus_truth.txt` |
| 8 | `src/speed_bench.py` | time per locus of the single forward pass vs SuSiE |

The demo model is deliberately tiny (width 64, 3 layers, a few minutes of CPU training on
100-variant loci), so read its numbers qualitatively. The full-scale configuration
(Section 7) trains a 6.45M-parameter model on 1,000-variant loci built from real 1000 Genomes LD.

Options: `WITH_SUSIER=1` also runs canonical susieR (Section 8); `PYTHON=/path/to/python`
selects the interpreter.

---

## 3. Fine-map your own locus

```bash
cd src
python predict.py --z my_locus_z.tsv --ld my_locus_ld.txt --n 250000 \
                  --ckpt /path/to/diffmap_final.pt --out my_locus_pips.tsv
```

| input | format |
|---|---|
| `--z` | tab-separated with a header: `SNP  Z` and optionally `MAF` |
| `--ld` | p × p correlation matrix (whitespace separated), rows in the order of `--z`, alleles aligned to the z-scores |
| `--n` | GWAS sample size |
| `--ckpt` | a checkpoint written by `src/train.py` |

Output columns: `SNP  Z  PIP  CS` (`CS` = 95% credible-set id, −1 if the variant is in none).
Use loci of the size the model was trained on (1,000 variants at full scale: the 1,000 variants
nearest the lead SNP, as in `real_data/build_real_loci.py`).

From Python:

```python
import sys; sys.path.insert(0, "src")
from model import load_checkpoint
from inference import oneshot_pip
model, _ = load_checkpoint("diffmap_final.pt")
pip = oneshot_pip(model, z, maf, R, n=250000)          # one forward pass -> PIPs
```

---

## 4. How the method works

**Data model.** For a locus with $p$ variants, standardized effects $\beta$, causal indicators
$\gamma \in \{0,1\}^p$, GWAS sample size $n$ and true LD matrix $R$, the marginal z-scores follow
the summary-statistic (RSS) likelihood

$$z \mid \beta, R \sim \mathcal{N}\big(\sqrt{n}\,R\,\beta,\; R\big). \tag{1}$$

Training loci are simulated from Eq. (1) over real LD blocks (`src/sim.py`).

**Amortized posterior as a masked (absorbing-state) discrete diffusion.** Each coordinate of
$\gamma$ is either known (0 or 1) or `MASK`. In the forward process every coordinate is masked
independently with probability $t \in (0, 1]$. The network $f_\theta$ (`src/model.py`) sees the
partly masked $\gamma^{(t)}$ together with $(z, R, \text{MAF}, n)$ and predicts, for every masked
coordinate, $\hat p_j = \Pr(\gamma_j = 1 \mid \gamma^{(t)}, z, R)$. It is trained with the
re-weighted cross-entropy over masked coordinates (`src/train.py`):

$$\mathcal{L}(\theta) = \mathbb{E}_{t,\gamma,z}\;\frac{1}{t\,p}\sum_{j:\,\gamma^{(t)}_j=\text{MASK}}
\Big[-\gamma_j \log \hat p_j - (1-\gamma_j)\log(1-\hat p_j)\Big]. \tag{2}$$

**One-shot PIPs.** With every coordinate masked ($t = 1$), a single forward pass returns the
marginal posterior inclusion probabilities (`inference.oneshot_pip`):

$$\pi_j = \sigma\big(f_\theta(\gamma^{(T)} = \text{MASK}, z, R, \text{MAF}, n)_j\big). \tag{3}$$

**Architecture.** A pre-norm transformer over one token per variant (features: $z$, $|z|$, MAF and
the current symbol of $\gamma_j$), with $t$ and $\log_{10} n$ embedded globally and no positional
embeddings. The supplied LD enters **only** as an additive attention bias: for every layer and
head, the attention logit between variants $i$ and $j$ is increased by a learned monotone
(non-decreasing) function of $R_{ij}$ (`model.MonotoneLDBias`). The LD can thus shape how
information flows between variants, but it is never treated as ground truth.

| configuration | width | heads | layers | parameters | training steps |
|---|---|---|---|---|---|
| small | 128 | 4 | 6 | 1.22M | 30,000 |
| big (recommended) | 256 | 8 | 8 | 6.45M | 60,000 |

**Diverse-reference training (the robustness mechanism).** The z-scores of a training locus are
always generated with the block's true in-sample LD, but the LD matrix *shown* to the network is,
with probability 0.8, drawn from a bank of reference matrices for the same block — 200/100/50-sample
European panels and 200-sample African and East Asian panels — and only with probability 0.2 the
matched matrix. The American and South Asian panels are never shown during training and are used
to test generalization to unseen ancestries.

**Algorithm 1 — credible sets by iterative conditioning** (`inference.diffmap_cs`).
Conditioning on a variant being causal means giving it the symbol 1 instead of `MASK`.

```
F = {}  (variants fixed as causal),  C = []  (credible sets)
repeat
    π   = conditional PIPs from one forward pass with γ_F = 1, all other coordinates MASK
    ℓ   = argmax_{j ∉ F, j unused} π_j ;  stop if π_ℓ < τ_lead
    S   = {ℓ} ∪ {unused j : |R_ℓj| ≥ τ_ld}
    sort S by π; keep the shortest prefix whose normalized π-mass reaches 0.95
    append that prefix to C;  F = F ∪ {ℓ}
```

Default operating point: $\tau_\text{lead} = 0.1$, $\tau_\text{ld} = 0.4$ (`src/cs_sweep.py` maps
the coverage / power / size trade-off over a grid).

**Evaluation metrics** (`src/metrics.py`). Simulation: AUPRC, precision among PIP > 0.9 calls, and
credible-set coverage (fraction of sets containing a causal variant), power (fraction of causal
variants captured) and mean size. Real data, against an in-sample-LD gold standard: confidently-wrong
calls (method PIP > 0.9 but gold PIP < 0.1), precision at PIP > 0.9 (gold PIP > 0.5), whether the gold
lead lies in a 95% credible set, and a per-locus mismatch diagnostic
$\mathrm{corr}(z, R_{\cdot \ell} z_\ell)$ (tiers: clean > 0.7, mid 0.3–0.7, low < 0.3).

---

## 5. Repository layout

```
DiffMap/
├── README.md
├── requirements.txt
├── run_full_pipeline.sh       full-scale pipeline (stages: data, ld, train, sim, real)
├── src/                       method and simulation benchmarks (run scripts from inside src/)
│   ├── model.py               DiffMap network
│   ├── sim.py                 simulation from Eq. (1)
│   ├── train.py               diverse-reference training, Eq. (2)
│   ├── inference.py           one-shot PIPs, Eq. (3), and Algorithm 1
│   ├── predict.py             fine-map one locus given as text files
│   ├── metrics.py             evaluation metrics
│   ├── susie_rss.py           NumPy SuSiE-RSS
│   ├── susie_official.R       canonical susieR wrapper
│   ├── baselines.py           adapters for susieR and RSparsePro
│   ├── build_ld_refs.py       1000 Genomes -> LD blocks with multi-ancestry reference matrices
│   ├── make_bench_refs.py     held-out simulation benchmark
│   ├── compare_refs.py        robustness across LD references
│   ├── cs_eval.py             credible sets, matched vs mismatched reference
│   ├── cs_sweep.py            credible-set operating characteristics
│   └── speed_bench.py         time per locus
├── real_data/                 benchmark against in-sample-LD fine-mapping of real biobanks
│   ├── download_data.sh       public downloads
│   ├── extract_ukb.py         UK Biobank fine-mapping release -> regions
│   ├── extract_finngen.py     FinnGen R12 fine-mapping -> regions (+ GRCh38 -> GRCh37 liftover)
│   ├── build_real_loci.py     1,000-variant loci with external 1000 Genomes LD
│   ├── compare_real.py        all methods vs the gold standard
│   └── aggregate_real.py      cross-cohort, pooled and by-mismatch-tier tables
└── demo/
    ├── run_demo.sh            one-command walkthrough
    └── make_sample_data.py    generator of the synthetic sample data
```

---

## 6. Synthetic sample data

`python demo/make_sample_data.py` (run automatically by `run_demo.sh`) writes
`demo/sample_data/`. It contains **no real genetic data**: genotypes are simulated with a
haplotype-copying model for five populations named and sized like the 1000 Genomes
super-populations (EUR 503, AFR 661, EAS 504, AMR 347, SAS 489), with block-like LD and realistic
differences in LD between populations (AFR most diverged with shorter-range LD, AMR an EUR/AFR
admixture, SAS close to EUR). The output is identical on every run (seed 7).

| path | content |
|---|---|
| `ld_bank_train/block*.npz` (40) | LD blocks for training |
| `ld_bank_test/block*.npz` (12) | held-out LD blocks for the benchmark |
| `real_like_loci/locus*.npz` (15) | loci in the format of `real_data/build_real_loci.py` |
| `example_locus_z.tsv`, `example_locus_ld.txt` | one locus as plain text (input of `predict.py`) |
| `example_locus_truth.txt` | its true causal variants and gold-standard lead |

Keys of an LD block (each matrix p × p; p = 100 in the sample data, 1,000 at full scale):

| key | meaning |
|---|---|
| `R` | full in-sample EUR LD: the TRUE LD that generates the z-scores |
| `ref_eur200`, `ref_eur100`, `ref_eur50` | LD from 200/100/50-sample EUR reference panels |
| `ref_afr200`, `ref_eas200` | cross-ancestry reference panels used in training |
| `ref_amr200`, `ref_sas200` | cross-ancestry panels never shown during training |
| `maf`, `pos`, `chrom`, `n` | allele frequencies, positions, block label, EUR sample size |

Keys of a real-data-style locus: `z` (cohort z-scores), `R` (external reference LD given to every
method), `maf`, `gold_pip` and `gold_cs` (gold standard: SuSiE with the cohort's in-sample LD),
`lead_idx` (gold lead: PIP > 0.9 inside a 95% credible set), `diag` (mismatch diagnostic) and, for
synthetic data only, `gamma` (true causal variants).

```python
import numpy as np
d = np.load("demo/sample_data/ld_bank_test/block0000.npz")
print(d.files, d["R"].shape)
```

---

## 7. Full-scale pipeline on public data

```bash
export DATA=/path/to/data_dir                  # ~25 GB of public data is downloaded here
export RSP_SRC=/path/to/RSparsePro_LD          # optional baseline (Section 8)
export DIFFMAP_RSCRIPT=Rscript                 # optional: R with susieR (Section 8)
bash run_full_pipeline.sh                      # or a subset of stages: data ld train sim real
```

| stage | what it does | resources |
|---|---|---|
| `data` | downloads 1000 Genomes Phase 3 (GRCh37), the UK Biobank in-sample-LD SuSiE fine-mapping release (Zenodo, doi:10.5281/zenodo.7055906), FinnGen R12 SuSiE fine-mapping and the hg38→hg19 liftover chain | internet, ~25 GB |
| `ld` | 1,000-SNP LD blocks (MAF ≥ 0.01) with 8 reference matrices each; chromosomes 3–22 for training, 1–2 held out; 200 benchmark loci | CPU, a few hours, ~100 GB |
| `train` | small and big models (Section 4) | one GPU; the big model takes ~30 min on a modern data-centre GPU |
| `sim` | robustness across references, credible sets, credible-set sweep, timing | GPU + CPU |
| `real` | UK Biobank (height, platelet count, eBMD, HbA1c, red blood cell count) and FinnGen (height, BMI, hypertension, cardiovascular disease): region extraction, liftover, 1,000-variant loci with external 1000 Genomes EUR LD, all methods vs the in-sample-LD gold standard, summary tables | GPU + CPU; ~30 GB RAM for the UK Biobank file |

Results and logs are written to `results/`, checkpoints to `runs/`. Individual scripts document
their arguments (`python script.py --help`). FinnGen asks users to complete a short acknowledgement
form before using its data: https://finngen.gitbook.io/documentation/data-download

---

## 8. Baselines: susieR and RSparsePro

**susieR** (canonical SuSiE). Install R and `install.packages("susieR")` (tested with 0.14.2).
Scripts call R through the command in `DIFFMAP_RSCRIPT` (default `Rscript`). If R runs in a
container:

```bash
apptainer pull r.sif docker://r-base:4.4.1
export R_LIBS_USER=/path/to/Rlibs              # library where susieR is installed
export DIFFMAP_RSCRIPT="apptainer exec --bind /path/to/data r.sif Rscript"
```

Then add `--with-susieR` to `compare_refs.py`, `cs_eval.py` or `compare_real.py`.

**RSparsePro.** `git clone https://github.com/zhwm/RSparsePro_LD` and pass the clone with
`--rsparsepro-src /path/to/RSparsePro_LD`; it runs with its default settings.

Both are optional; without them the scripts compare the NumPy SuSiE implementation
(`src/susie_rss.py`) with DiffMap.

---

## 9. Troubleshooting

* **`ModuleNotFoundError: model`**: run the scripts in `src/` from inside `src/`; scripts in
  `real_data/` and `demo/` find `src/` on their own.
* **Slow training on CPU**: use `QUICK=1` for the demo; full-scale training needs a GPU.
* **`susieR not available`**: check that `$DIFFMAP_RSCRIPT -e 'library(susieR)'` works.
* **Out of memory in `extract_ukb.py`**: use a machine with ≥ 32 GB of RAM.
* **`build_real_loci.py` interrupted**: re-run the same command; finished loci are skipped.

---

## Data sources

All data used by the pipeline are public: the 1000 Genomes Project Phase 3 release, the UK Biobank
in-sample-LD fine-mapping deposit (Zenodo, doi:10.5281/zenodo.7055906) and the FinnGen R12 public
release. The demo uses only synthetic data generated locally.

## License

See [LICENSE](LICENSE).
