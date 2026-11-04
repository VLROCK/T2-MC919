#!/usr/bin/env bash
#SBATCH --job-name=t2-finalize
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PY="${PY:-python}"
COLMAP_BIN="${COLMAP_BIN:-colmap}"
WORK_DIR="${WORK_DIR:-work/gpu_compat}"
"$COLMAP_BIN" help >/dev/null
"$PY" -m src.densify --work "$WORK_DIR" --colmap "$COLMAP_BIN" --mesh --existing-only

scale_args=()
if [[ -f "$WORK_DIR/scale.json" ]]; then
  scale_args=(--scale-json "$WORK_DIR/scale.json")
fi
"$PY" -m src.analysis --model "$WORK_DIR/sparse_txt" \
  --manifest "$WORK_DIR/manifest.json" --measurements data/measurements.json \
  --output "$WORK_DIR/analysis" "${scale_args[@]}"
