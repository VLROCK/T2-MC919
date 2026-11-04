#!/usr/bin/env bash
#SBATCH --job-name=t2-sfm
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PY="${PY:-python}"
COLMAP_BIN="${COLMAP_BIN:-colmap}"
WORK_DIR="${WORK_DIR:-work/main}"
MATCHING="${MATCHING:-exhaustive}"
mkdir -p "$WORK_DIR"

"$PY" -c 'import cv2, numpy, scipy, matplotlib; print("Python dependencies OK")'
"$COLMAP_BIN" help >/dev/null
"$PY" - <<'PYCODE'
from pathlib import Path
photos = sorted(Path('data/images').glob('*.JPG'))
if len(photos) != 400 or photos[0].open('rb').read(2) != b'\xff\xd8':
    raise SystemExit(f'Esperadas 400 fotos JPEG em data/images; encontradas {len(photos)}.')
print('Fotos OK:', len(photos), '| primeira:', photos[0].name, '| última:', photos[-1].name)
PYCODE

cpu_args=()
if [[ "${CPU_ONLY:-0}" == "1" ]]; then cpu_args=(--cpu); fi
"$PY" -m src.pipeline --images data/images --work "$WORK_DIR" \
  --matching "$MATCHING" --colmap "$COLMAP_BIN" \
  --feature-image-size "${FEATURE_IMAGE_SIZE:-2000}" \
  --max-features "${MAX_FEATURES:-4096}" \
  --feature-threads "${FEATURE_THREADS:-4}" \
  --max-matches "${MAX_MATCHES:-8192}" "${cpu_args[@]}"

scale_args=()
if [[ -n "${MARKER_SIZE_M:-}" ]]; then
  "$PY" -m src.scale_marker --model "$WORK_DIR/sparse_txt" --images data/images \
    --marker-id 0 --marker-size-m "$MARKER_SIZE_M" --output "$WORK_DIR/scale.json"
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
if ! "$PY" -m src.feature_analysis --images data/images --output "$WORK_DIR/features" \
  "${feature_args[@]}"; then
  echo 'Aviso: comparação de descritores falhou; análise e vídeo permanecem disponíveis.' >&2
fi
