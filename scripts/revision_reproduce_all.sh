#!/usr/bin/env bash
set -euo pipefail

# One-click revision reproduction runner:
# - trains UniVI
# - evaluates key metrics
# - runs robustness panels (Reviewer 1)
# - exports Supplemental_Table_S1.xlsx
# - runs CITE-seq benchmark panel script (if present)

OUTDIR="${1:-runs/revision}"
CONFIG="${2:-parameter_files/defaults_cite_seq_raw_counts_nb_lite.json}"
DEVICE="${DEVICE:-cpu}"
LABEL_KEY="${LABEL_KEY:-cell_type}"

mkdir -p "${OUTDIR}"

echo "[1/6] Train UniVI"
python scripts/train_univi.py --config "${CONFIG}" --outdir "${OUTDIR}/train" --device "${DEVICE}"

echo "[2/6] Evaluate UniVI (paired metrics)"
python scripts/evaluate_univi.py --config "${CONFIG}" --checkpoint "${OUTDIR}/train/univi_checkpoint.pt" --outdir "${OUTDIR}/eval" --device "${DEVICE}" --label-key "${LABEL_KEY}"

echo "[3/6] Frequency perturbation robustness"
python scripts/run_frequency_robustness.py --config "${CONFIG}" --checkpoint "${OUTDIR}/train/univi_checkpoint.pt" --outdir "${OUTDIR}/robustness" --device "${DEVICE}" --label-key "${LABEL_KEY}"

echo "[4/6] Do-not-integrate detection"
python scripts/run_do_not_integrate_detection.py --config "${CONFIG}" --checkpoint "${OUTDIR}/train/univi_checkpoint.pt" --outdir "${OUTDIR}/robustness" --device "${DEVICE}" --label-key "${LABEL_KEY}"

echo "[5/6] Export Supplemental_Table_S1.xlsx"
python -m univi export-s1 --config "${CONFIG}" --checkpoint "${OUTDIR}/train/univi_checkpoint.pt" --out "${OUTDIR}/Supplemental_Table_S1.xlsx"

echo "[6/6] Optional: run legacy CITE-seq benchmark figure script"
if [[ -f "scripts/benchmark_univi_citeseq.py" ]]; then
  # Extract RNA/ADT h5ad paths from config
  RNA_H5AD="$(python - <<'PY'
import json,sys
cfg=json.load(open(sys.argv[1]))
print([m['h5ad_path'] for m in cfg['data']['modalities'] if m['name']=='rna'][0])
PY
"${CONFIG}")"
  ADT_H5AD="$(python - <<'PY'
import json,sys
cfg=json.load(open(sys.argv[1]))
print([m['h5ad_path'] for m in cfg['data']['modalities'] if m['name']=='adt'][0])
PY
"${CONFIG}")"
  python scripts/benchmark_univi_citeseq.py \
    --rna-h5ad "${RNA_H5AD}" \
    --adt-h5ad "${ADT_H5AD}" \
    --checkpoint "${OUTDIR}/train/univi_checkpoint.pt" \
    --config-json "${CONFIG}" \
    --output-dir "${OUTDIR}/benchmark_univi_citeseq"
else
  echo "  (skipped; scripts/benchmark_univi_citeseq.py not found)"
fi

echo "[DONE] Outputs in: ${OUTDIR}"
