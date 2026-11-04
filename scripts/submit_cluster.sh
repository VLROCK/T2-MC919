#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/logs
SPARSE_JOB=$(sbatch --parsable "$@" --output=results/logs/t2-sfm-%j.out \
  --export=ALL scripts/slurm_sparse.sh)
SPARSE_JOB="${SPARSE_JOB%%;*}"
echo "SPARSE_JOB=$SPARSE_JOB"

DENSE_JOB=$(sbatch --parsable "$@" --dependency="afterok:$SPARSE_JOB" \
  --output=results/logs/t2-mvs-%j.out --export=ALL scripts/slurm_dense.sh)
DENSE_JOB="${DENSE_JOB%%;*}"
echo "DENSE_JOB=$DENSE_JOB"
