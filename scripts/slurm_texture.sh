#!/usr/bin/env bash
#SBATCH --job-name=t2-texture
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PY="${PY:-python}"
WORK_DIR="${WORK_DIR:-work/gpu_compat}"
COLMAP_TEXTURE_BIN="${COLMAP_TEXTURE_BIN:?Defina COLMAP_TEXTURE_BIN apontando para COLMAP 4.0+}"
"$COLMAP_TEXTURE_BIN" mesh_texturer -h >/dev/null
"$PY" -m src.texture_mesh --work "$WORK_DIR" --colmap "$COLMAP_TEXTURE_BIN"
