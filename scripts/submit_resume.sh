#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
WORK_DIR="${WORK_DIR:-work/gpu_compat}"
if [[ ! -f "$WORK_DIR/sparse_txt/images.txt" ]]; then
  echo "Modelo esparso não encontrado: $WORK_DIR/sparse_txt/images.txt" >&2
  exit 2
fi
COLMAP_BIN="${COLMAP_BIN:-colmap}"
if ! "$COLMAP_BIN" help >/dev/null 2>&1; then
  echo "COLMAP não encontrado ou não executável: $COLMAP_BIN" >&2
  exit 2
fi
mkdir -p results/logs
FINISH_JOB=$(sbatch --parsable --output=results/logs/t2-finish-%j.out \
  --export=ALL scripts/slurm_finish_sparse.sh)
FINISH_JOB="${FINISH_JOB%%;*}"
echo "FINISH_JOB=$FINISH_JOB"

DENSE_JOB=$(sbatch --parsable "$@" --dependency="afterok:$FINISH_JOB" \
  --output=results/logs/t2-mvs-%j.out --export=ALL scripts/slurm_dense.sh)
DENSE_JOB="${DENSE_JOB%%;*}"
echo "DENSE_JOB=$DENSE_JOB"
