#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

COLMAP_BIN="${COLMAP_BIN:-colmap}"
PY="${PY:-python}"
if ! "$COLMAP_BIN" help >/dev/null 2>&1; then
  echo "COLMAP não encontrado ou não executável: $COLMAP_BIN" >&2
  echo "Carregue um módulo do cluster ou instale COLMAP antes de submeter jobs." >&2
  exit 2
fi
"$PY" - <<'PYCODE'
from pathlib import Path
photos = sorted(Path('data/images').glob('*.JPG'))
if len(photos) != 400 or photos[0].open('rb').read(2) != b'\xff\xd8':
    raise SystemExit(f'Esperadas 400 fotos JPEG em data/images; encontradas {len(photos)}.')
print('Pré-checagem OK:', len(photos), 'fotos')
PYCODE

mkdir -p results/logs
SPARSE_JOB=$(sbatch --parsable "$@" --output=results/logs/t2-sfm-%j.out \
  --export=ALL scripts/slurm_sparse.sh)
SPARSE_JOB="${SPARSE_JOB%%;*}"
echo "SPARSE_JOB=$SPARSE_JOB"

if [[ "${SKIP_DENSE:-0}" == "1" ]]; then
  echo "SKIP_DENSE=1; job MVS não submetido."
  exit 0
fi

DENSE_JOB=$(sbatch --parsable "$@" --dependency="afterok:$SPARSE_JOB" \
  --output=results/logs/t2-mvs-%j.out --export=ALL scripts/slurm_dense.sh)
DENSE_JOB="${DENSE_JOB%%;*}"
echo "DENSE_JOB=$DENSE_JOB"
