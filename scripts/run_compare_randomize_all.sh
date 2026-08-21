#!/usr/bin/env bash
# Run compare_randomize over every checkpoint directory.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

OUTPUT_DIR="logs/compare"
mkdir -p ${OUTPUT_DIR}

shopt -s nullglob
checkpoints=(checkpoints/*/)
shopt -u nullglob

if [[ ${#checkpoints[@]} -eq 0 ]]; then
  echo "No checkpoint directories found under checkpoints/." >&2
  exit 1
fi

for checkpoint_dir in "${checkpoints[@]}"; do
  checkpoint_name="$(basename "${checkpoint_dir%/}")"
  output_file="${OUTPUT_DIR}/${checkpoint_name}.txt"

  echo "Running ${checkpoint_name} -> ${output_file}"
  JAX_PLATFORMS=cpu uv run -m scripts.compare_randomize \
    --checkpoint_path "$checkpoint_dir" \
    --device cpu \
    --n_actors 20 \
    --n_episodes 100 \
    --render True \
    > "$output_file" 2>&1

done

echo "Done. Outputs are in ${OUTPUT_DIR}/."
