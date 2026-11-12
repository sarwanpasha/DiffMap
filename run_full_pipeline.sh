#!/bin/bash
# run_full_pipeline.sh -- full-scale pipeline: data, LD banks, training, simulation and real-data benchmarks.
#
# Needs: a CUDA GPU for training (the big model trains in ~30 min on one modern GPU),
#        the public data (real_data/download_data.sh, ~25 GB), and optionally
#        R + susieR and a clone of RSparsePro for the baselines.
#
# Usage:
#   export DATA=/path/to/data_dir                 # where download_data.sh put the data
#   export RSP_SRC=/path/to/RSparsePro_LD         # optional: git clone https://github.com/zhwm/RSparsePro_LD
#   export DIFFMAP_RSCRIPT="Rscript"              # optional: command that runs R with susieR installed
#   bash run_full_pipeline.sh [stage ...]           # stages: data ld train sim real   (default: all)
#
# Each stage can be run on its own; outputs go to $DATA/... and ./results/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$ROOT/src"; RD="$ROOT/real_data"; RES="$ROOT/results"
DATA=${DATA:?set DATA to the data directory}
PY=${PYTHON:-python3}
PANEL="$DATA/1000g/integrated_call_samples_v3.20130502.ALL.panel"
RSP_FLAG=""; [ -n "${RSP_SRC:-}" ] && RSP_FLAG="--rsparsepro-src $RSP_SRC"
STAGES=${*:-"data ld train sim real"}
mkdir -p "$RES"

for stage in $STAGES; do case $stage in

data)   # public downloads: 1000 Genomes, UK Biobank fine-mapping, FinnGen R12, liftover chain
  bash "$RD/download_data.sh" "$DATA" ;;

ld)     # LD banks: training chromosomes 3-22, held-out test chromosomes 1-2
  ( cd "$SRC"
    $PY build_ld_refs.py --vcf-dir "$DATA/1000g" --panel "$PANEL" --chroms 3-22 --out-dir "$DATA/ld_refs_train" --workers 4
    $PY build_ld_refs.py --vcf-dir "$DATA/1000g" --panel "$PANEL" --chroms 1,2 --max-blocks 200 --out-dir "$DATA/ld_refs_test"
    $PY make_bench_refs.py --bank "$DATA/ld_refs_test" --out "$DATA/bench" --n-loci 200 --seed 1234 ) ;;

train)  # small (1.22M) and big (6.45M) models with diverse-reference training
  ( cd "$SRC"
    $PY train.py --bank "$DATA/ld_refs_train" --out "$ROOT/runs/diffmap_small" --d 128 --heads 4 --layers 6 --steps 30000
    $PY train.py --bank "$DATA/ld_refs_train" --out "$ROOT/runs/diffmap_big"   --d 256 --heads 8 --layers 8 --steps 60000 ) ;;

sim)    # simulation benchmarks: robustness across references, credible sets, timing
  ( cd "$SRC"
    $PY compare_refs.py --bench "$DATA/bench" \
        --ckpt diffmap_small="$ROOT/runs/diffmap_small/diffmap_final.pt" \
        --ckpt diffmap_big="$ROOT/runs/diffmap_big/diffmap_final.pt" \
        --with-susieR $RSP_FLAG --save "$RES/compare_refs.csv" | tee "$RES/compare_refs.log"
    $PY cs_eval.py --bench "$DATA/bench" --ckpt "$ROOT/runs/diffmap_big/diffmap_final.pt" \
        --ref ref_eur100 --lead-thr 0.1 --ld-thr 0.4 --with-susieR $RSP_FLAG | tee "$RES/cs_eval.log"
    $PY cs_sweep.py --bench "$DATA/bench" --ckpt "$ROOT/runs/diffmap_big/diffmap_final.pt" | tee "$RES/cs_sweep.log"
    $PY speed_bench.py --bench "$DATA/bench" --ckpt "$ROOT/runs/diffmap_big/diffmap_final.pt" $RSP_FLAG \
        | tee "$RES/speed.log" ) ;;

real)   # real-data validation: UK Biobank (5 traits) + FinnGen R12 (4 endpoints) vs in-sample-LD gold standards
  CK="$ROOT/runs/diffmap_big/diffmap_final.pt"
  $PY "$RD/extract_ukb.py" --src "$DATA/ukb_finemap/UKB.susie.100k.snp.feather" \
      --out-base "$DATA/ukb_finemap" --traits Height,Plt,eBMD,HbA1c,RBC
  $PY "$RD/extract_finngen.py" --src "$DATA/finngen" --chain "$DATA/finngen/hg38ToHg19.over.chain.gz"
  for d in "$DATA"/ukb_finemap/*_regions "$DATA"/finngen/*_regions; do
    tr=$(basename "$d" _regions); base=$(dirname "$d")
    $PY "$RD/build_real_loci.py" --regions "$d" --out "$base/real_loci_$tr" \
        --vcf-dir "$DATA/1000g" --panel "$PANEL" --pop EUR
  done
  for d in "$DATA"/ukb_finemap/real_loci_*; do
    tr=$(basename "$d" | sed 's/^real_loci_//')
    $PY "$RD/compare_real.py" --loci "$d" --trait "ukb_$tr" --ckpt "$CK" --n 100000 \
        --with-susieR $RSP_FLAG --save "$RES/res_ukb_$tr.csv"
  done
  for d in "$DATA"/finngen/real_loci_*; do
    tr=$(basename "$d" | sed 's/^real_loci_//')
    # n = 100000 is passed to every method for both cohorts
    $PY "$RD/compare_real.py" --loci "$d" --trait "fg_$tr" --ckpt "$CK" --n 100000 \
        --with-susieR $RSP_FLAG --save "$RES/res_fg_$tr.csv"
  done
  $PY "$RD/aggregate_real.py" --cohort "UKB=$RES/res_ukb_*.csv" --cohort "FinnGen=$RES/res_fg_*.csv" \
      | tee "$RES/real_data_tables.log" ;;

*) echo "unknown stage $stage (use: data ld train sim real)"; exit 1 ;;
esac; done
