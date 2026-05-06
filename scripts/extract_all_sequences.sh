#!/bin/bash
# Extract per-chain sequences from saved_proteins/ for every completed run.
#
# A run is considered "completed" if its output directory contains
# timing_summary.txt (which is only written at the end of train_and_infer_ab.py).
# In-progress runs (no timing_summary.txt yet) are skipped, so this is safe to
# re-run during a sweep — newly-finished runs get picked up automatically.
#
# Output: <run_dir>/saved_proteins/sequences.csv (one per completed run).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/output}"

shopt -s nullglob
processed=0
skipped=0
for run_dir in "$OUTPUT_DIR"/ab_*; do
    [[ -d "$run_dir" ]] || continue
    name="$(basename "$run_dir")"
    pdb_dir="$run_dir/saved_proteins"

    if [[ ! -f "$run_dir/timing_summary.txt" ]]; then
        echo "[skip] $name (no timing_summary.txt — likely still running)"
        skipped=$((skipped + 1))
        continue
    fi
    if [[ ! -d "$pdb_dir" ]] || ! ls "$pdb_dir"/*.pdb >/dev/null 2>&1; then
        echo "[skip] $name (no PDBs in saved_proteins/)"
        skipped=$((skipped + 1))
        continue
    fi

    echo "[extract] $name"
    python "$REPO_DIR/scripts/extract_sequences.py" "$pdb_dir"
    processed=$((processed + 1))
done

echo
echo "Done: $processed extracted, $skipped skipped."
