#!/bin/bash
# run_demo.sh -- end-to-end DiffMap walkthrough on the synthetic sample data.
#
# Runs on a laptop CPU (no GPU, no downloads). Typical time: 5-15 minutes on CPU.
#   QUICK=1 bash run_demo.sh     -> shorter training (~2-4 minutes total), noisier numbers
#   WITH_SUSIER=1 bash run_demo.sh -> also run canonical susieR (needs R + susieR, see README)
#
# Steps
#   1. (re)generate the synthetic sample data             demo/make_sample_data.py
#   2. simulate a held-out benchmark from held-out blocks  src/make_bench_refs.py
#   3. train a SMALL DiffMap with diverse references       src/train.py
#   4. robustness across references                       src/compare_refs.py
#   5. credible sets, matched vs mismatch                 src/cs_eval.py
#   6. real-data-style validation                         real_data/compare_real.py + aggregate_real.py
#   7. fine-map one locus from plain text files            src/predict.py
#   8. per-locus timing                                   src/speed_bench.py
# All outputs and logs go to demo/outputs/.
#
# NOTE: the demo model is tiny (d=64, 3 layers, minutes of CPU training on 100-SNP
# loci) and the data are synthetic, so numbers illustrate the behaviour of the
# method; full-scale results come from the 6.45M-parameter
# model trained for 60k GPU steps on 1000-SNP loci from 1000 Genomes.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/../src"
RD="$HERE/../real_data"
DATA="$HERE/sample_data"
OUT="$HERE/outputs"
PY=${PYTHON:-python3}
STEPS=3000; [ "${QUICK:-0}" = "1" ] && STEPS=800
SUSIER_FLAG=""; [ "${WITH_SUSIER:-0}" = "1" ] && SUSIER_FLAG="--with-susieR"
mkdir -p "$OUT"
banner() { echo; echo "================ $* ================"; }

banner "1/8 synthetic sample data"
$PY "$HERE/make_sample_data.py" --out "$DATA"

banner "2/8 held-out simulation benchmark (40 loci)"
( cd "$SRC" && $PY make_bench_refs.py --bank "$DATA/ld_bank_test" --out "$OUT/bench" --n-loci 40 --seed 1234 )

banner "3/8 train a small DiffMap ($STEPS steps, CPU)"
( cd "$SRC" && $PY train.py --bank "$DATA/ld_bank_train" --out "$OUT/demo_model" \
    --train-refs eur200,eur100,eur50,afr200,eas200 --matched-prob 0.2 \
    --d 64 --heads 4 --layers 3 --steps "$STEPS" --batch-size 8 --lr 1e-3 --warmup 200 \
    --workers 0 --cache-size 40 --log-every 100 --save-every 0 ) | tee "$OUT/train.log"
CK="$OUT/demo_model/diffmap_final.pt"

banner "4/8 robustness across LD references (AMR/SAS were never seen in training)"
( cd "$SRC" && $PY compare_refs.py --bench "$OUT/bench" --ckpt "diffmap_demo=$CK" $SUSIER_FLAG \
    --save "$OUT/compare_refs.csv" ) | tee "$OUT/compare_refs.log"

banner "5/8 credible sets: matched vs mismatched (EUR100) reference"
( cd "$SRC" && $PY cs_eval.py --bench "$OUT/bench" --ckpt "$CK" --ref ref_eur100 \
    --lead-thr 0.1 --ld-thr 0.4 $SUSIER_FLAG ) | tee "$OUT/cs_eval.log"

banner "6/8 real-data-style validation against an in-sample-LD gold standard"
$PY "$RD/compare_real.py" --loci "$DATA/real_like_loci" --trait demo_height --ckpt "$CK" --n 100000 \
    $SUSIER_FLAG --save "$OUT/res_demo_height.csv" | tee "$OUT/compare_real.log"
$PY "$RD/aggregate_real.py" --cohort "Demo=$OUT/res_demo_*.csv" | tee -a "$OUT/compare_real.log"

banner "7/8 fine-map one locus given as plain text"
( cd "$SRC" && $PY predict.py --z "$DATA/example_locus_z.tsv" --ld "$DATA/example_locus_ld.txt" \
    --n 100000 --ckpt "$CK" --out "$OUT/example_locus_pips.tsv" )
cat "$DATA/example_locus_truth.txt"

banner "8/8 per-locus inference time"
( cd "$SRC" && $PY speed_bench.py --bench "$OUT/bench" --ckpt "$CK" --n-loci 20 ) | tee "$OUT/speed.log"

banner "done -- outputs in $OUT"
