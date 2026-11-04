#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
COLMAP_BIN="${COLMAP_BIN:-colmap}"
IMAGES_DIR="${IMAGES_DIR:-data/images}"
WORK_DIR="${WORK_DIR:-work/main}"
MATCHING="${MATCHING:-sequential}"

args=(--images "$IMAGES_DIR" --work "$WORK_DIR" --matching "$MATCHING" --colmap "$COLMAP_BIN")
if [[ "${CPU_ONLY:-0}" == "1" ]]; then args+=(--cpu); fi
if [[ "${DENSE:-0}" == "1" ]]; then args+=(--dense); fi
if [[ "${MESH:-0}" == "1" ]]; then args+=(--mesh); fi
if [[ "${LOOP_DETECTION:-0}" == "1" ]]; then args+=(--loop-detection); fi
if [[ -n "${MAX_IMAGES:-}" ]]; then args+=(--max-images "$MAX_IMAGES"); fi
"$PYTHON_BIN" -m src.pipeline "${args[@]}"
MODEL_IMAGES="$IMAGES_DIR"
if [[ -d "$WORK_DIR/images" ]]; then MODEL_IMAGES="$WORK_DIR/images"; fi

scale_args=()
if [[ -n "${MARKER_SIZE_M:-}" ]]; then
  "$PYTHON_BIN" -m src.scale_marker --model "$WORK_DIR/sparse_txt" --images "$MODEL_IMAGES" \
    --marker-size-m "$MARKER_SIZE_M" --marker-id "${MARKER_ID:-0}" \
    --output "$WORK_DIR/scale.json"
  scale_args=(--scale-json "$WORK_DIR/scale.json")
fi

feature_args=()
if [[ -f "$WORK_DIR/intrinsics.json" ]]; then
  feature_args=(--intrinsics "$WORK_DIR/intrinsics.json")
fi
"$PYTHON_BIN" -m src.feature_analysis --images "$MODEL_IMAGES" \
  --output "$WORK_DIR/features" "${feature_args[@]}"
"$PYTHON_BIN" -m src.analysis --model "$WORK_DIR/sparse_txt" \
  --manifest "$WORK_DIR/manifest.json" --measurements data/measurements.json \
  --output "$WORK_DIR/analysis" "${scale_args[@]}"
"$PYTHON_BIN" -m src.visualize --model "$WORK_DIR/sparse_txt" \
  --metrics "$WORK_DIR/analysis/metrics.json" \
  --output "$WORK_DIR/analysis/navigation.mp4"

echo "Resultados em $WORK_DIR"
