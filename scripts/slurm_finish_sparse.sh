#!/usr/bin/env bash
#SBATCH --job-name=t2-finish
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PY="${PY:-python}"
WORK_DIR="${WORK_DIR:-work/gpu_compat}"
if [[ ! -f "$WORK_DIR/sparse_txt/images.txt" ]]; then
  echo "Modelo esparso ausente em $WORK_DIR/sparse_txt" >&2
  exit 2
fi

scale_args=()
if [[ -n "${MARKER_SIZE_M:-}" ]]; then
  if "$PY" -m src.scale_marker --model "$WORK_DIR/sparse_txt" --images data/images \
    --marker-id 0 --marker-size-m "$MARKER_SIZE_M" --output "$WORK_DIR/scale.json"; then
    scale_args=(--scale-json "$WORK_DIR/scale.json")
  else
    echo 'Aviso: escala ArUco falhou; análise prosseguirá sem escala métrica.' >&2
  fi
elif [[ -f "$WORK_DIR/scale.json" ]]; then
  scale_args=(--scale-json "$WORK_DIR/scale.json")
fi

"$PY" -m src.analysis --model "$WORK_DIR/sparse_txt" \
  --manifest "$WORK_DIR/manifest.json" --measurements data/measurements.json \
  --output "$WORK_DIR/analysis" "${scale_args[@]}"
"$PY" -m src.visualize --model "$WORK_DIR/sparse_txt" \
  --metrics "$WORK_DIR/analysis/metrics.json" \
  --output "$WORK_DIR/analysis/navigation.mp4"

feature_args=()
if [[ -f "$WORK_DIR/intrinsics.json" ]]; then
  feature_args=(--intrinsics "$WORK_DIR/intrinsics.json")
fi
if ! "$PY" -m src.feature_analysis --images data/images \
  --output "$WORK_DIR/features" "${feature_args[@]}"; then
  echo 'Aviso: comparação de descritores falhou; análise e vídeo permanecem disponíveis.' >&2
fi
