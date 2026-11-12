#!/bin/bash
# download_data.sh -- fetch the public data used by the full pipeline.
#
# Usage:  bash download_data.sh DATA_DIR
#
# Everything below is publicly downloadable (no access application needed).
# Total size is large (~14 GB for 1000 Genomes, ~4.5 GB for the UK Biobank
# fine-mapping file, several GB for FinnGen); run on a machine with internet.
#   1. 1000 Genomes Phase 3 (GRCh37) VCFs + sample panel   -> LD blocks / reference LD
#   2. UK Biobank in-sample-LD SuSiE fine-mapping (Zenodo 10.5281/zenodo.7055906)
#   3. FinnGen R12 in-sample-LD SuSiE fine-mapping (public Google Cloud bucket)
#      FinnGen asks users to fill a short acknowledgement form before use:
#      https://finngen.gitbook.io/documentation/data-download
#   4. UCSC hg38 -> hg19 liftover chain (FinnGen is GRCh38; 1000G is GRCh37)
set -euo pipefail
DATA=${1:?usage: bash download_data.sh DATA_DIR}
mkdir -p "$DATA"/1000g "$DATA"/ukb_finemap "$DATA"/finngen

# ---- 1. 1000 Genomes Phase 3 -------------------------------------------------
BASE=https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502
wget -c -P "$DATA"/1000g "$BASE/integrated_call_samples_v3.20130502.ALL.panel"
for c in $(seq 1 22); do
  f=ALL.chr${c}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz
  wget -c -P "$DATA"/1000g "$BASE/$f" "$BASE/$f.tbi"
done

# ---- 2. UK Biobank fine-mapping (gold standard + z-scores in one file) ----------
wget -c -O "$DATA"/ukb_finemap/UKB.susie.100k.snp.feather \
  "https://zenodo.org/records/7055906/files/UKB.susie.100k.snp.feather?download=1"

# ---- 3. FinnGen R12 SuSiE fine-mapping for the benchmark endpoints -----------
for ep in HEIGHT_IRN BMI_IRN I9_HYPTENS FG_CVD GOUT_NOS; do
  wget -c -O "$DATA"/finngen/${ep}.SUSIE.snp.bgz \
    "https://storage.googleapis.com/finngen-public-data-r12/finemap/full/susie/finngen_R12_${ep}.SUSIE.snp.bgz"
done

# ---- 4. liftover chain ------------------------------------------------------------
wget -c -O "$DATA"/finngen/hg38ToHg19.over.chain.gz \
  "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/hg38ToHg19.over.chain.gz"

echo "done -> $DATA"
