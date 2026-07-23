#!/bin/bash
# Parity check for VIDD: the ipTM REWARD the model optimized during design vs
# the FINAL ipTM computed by bonobo's evaluator (the only number we report).
#
# Mirrors ProDifEvo-Refinement/scripts/eval_parity.sh. Final ipTM is computed by
# bonobo ONLY — we no longer cross-check VIDD's / RERD's own evaluators against
# each other. Two tests:
#
#   [CRITICAL] reward-vs-bonobo : design-time reward `iptm` (logged in the VIDD
#                                 run's output.csv) vs bonobo
#                                 eval_compiled_final_iptm.py `final_iptm` on the
#                                 same sequences. mean ~0 = the reward the model
#                                 optimized matches the metric we report.
#   [CYA]      race check        : design-time reward `iptm` vs a fresh VIDD
#                                 re-eval (scripts/eval_iptm.py). mean ~0 = no
#                                 multi-GPU race corrupted the logged reward.
#                                 Skip with RUN_RACE_CHECK=0.
#
# Required:
#   * --input-csv : a VIDD design output.csv with a `sequence` column (and an
#                   `iptm` column = the design-time reward; required for the
#                   critical test).
#   * --antigen   : one of pdl1, bhrf1, il3, il20.
#
# Run on the EC2 box (VIDD is not containerized; runs here):
#
#     cd ~/VIDD
#     git pull
#     bash scripts/eval_parity.sh \
#         --input-csv ~/VIDD/output/ab_vidd_pdl1_.../output.csv \
#         --antigen pdl1
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# ---- Parse CLI args (also accept env vars as fallbacks) ----
INPUT_CSV="${INPUT_CSV:-}"
ANTIGEN="${ANTIGEN:-}"
while [ $# -gt 0 ]; do
    case "$1" in
        --input-csv) INPUT_CSV="$2"; shift 2 ;;
        --antigen)   ANTIGEN="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$INPUT_CSV" ] || [ -z "$ANTIGEN" ]; then
    echo "usage: bash scripts/eval_parity.sh --input-csv <csv> --antigen <name>"
    echo "       (or set INPUT_CSV and ANTIGEN env vars)"
    exit 1
fi
if [ ! -f "$INPUT_CSV" ]; then
    echo "ERROR: --input-csv not found: $INPUT_CSV"
    exit 1
fi
case "$ANTIGEN" in
    pdl1|bhrf1|il3|il20) ;;
    *) echo "ERROR: --antigen must be one of pdl1, bhrf1, il3, il20; got '$ANTIGEN'"; exit 1 ;;
esac

# ---- Knobs (env-overridable) ----
AF_GPU_IDS="${AF_GPU_IDS:-1,2,3}"
VIDD_CONDA_ENV="${VIDD_CONDA_ENV:-vidd}"
BONOBO_CONDA_ENV="${BONOBO_CONDA_ENV:-bonobo}"
BONOBO_REPO="${BONOBO_REPO:-${HOME}/bonobo}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/vidd_parity/${ANTIGEN}}"
RUN_RACE_CHECK="${RUN_RACE_CHECK:-1}"

if [ ! -d "$BONOBO_REPO" ]; then
    echo "ERROR: bonobo repo not found at $BONOBO_REPO (set BONOBO_REPO=...)"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "$OUTPUT_ROOT"
VIDD_EVAL_CACHE_DIR="${OUTPUT_ROOT}/vidd_eval_cache"
BONOBO_STAGING_DIR="${OUTPUT_ROOT}/bonobo_eval"
BONOBO_CACHE_DIR="${OUTPUT_ROOT}/bonobo_cache"
mkdir -p "$VIDD_EVAL_CACHE_DIR" "$BONOBO_STAGING_DIR" "$BONOBO_CACHE_DIR"

# Snapshot the source CSV inside our output dir for reproducibility.
INPUT_BASENAME="$(basename "$INPUT_CSV")"
INPUT_SNAPSHOT="${OUTPUT_ROOT}/${INPUT_BASENAME}"
cp "$INPUT_CSV" "$INPUT_SNAPSHOT"

# Source conda once.
for CONDA_SH in /opt/conda/etc/profile.d/conda.sh "${HOME}/miniconda3/etc/profile.d/conda.sh" "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
    if [ -f "$CONDA_SH" ]; then
        # shellcheck disable=SC1090
        source "$CONDA_SH"
        break
    fi
done

# ============================================================
# Test A [CRITICAL]: bonobo final ipTM (eval_compiled_final_iptm.py, bonobo env)
# ============================================================
# Bonobo's eval reads {compiled_dir}/{method}_{target}.csv. Stage under vidd_*.
echo "=========================================================="
echo "Test A [CRITICAL]: bonobo final ipTM"
echo "  input  : $INPUT_SNAPSHOT"
echo "  antigen: $ANTIGEN"
echo "=========================================================="
STAGED_INPUT="${BONOBO_STAGING_DIR}/vidd_${ANTIGEN}.csv"
cp "$INPUT_SNAPSHOT" "$STAGED_INPUT"
echo "  staged for bonobo: $STAGED_INPUT"

conda activate "$BONOBO_CONDA_ENV"
pushd "$BONOBO_REPO" >/dev/null
python eval_compiled_final_iptm.py \
    --targets "$ANTIGEN" \
    --methods vidd \
    --compiled_dir_template "$BONOBO_STAGING_DIR" \
    --cache_dir "$BONOBO_CACHE_DIR" \
    --write_inplace 0
popd >/dev/null
BONOBO_EVAL_CSV="${BONOBO_STAGING_DIR}/vidd_${ANTIGEN}_w_final_iptm.csv"
if [ ! -f "$BONOBO_EVAL_CSV" ]; then
    echo "ERROR: bonobo eval output not found at $BONOBO_EVAL_CSV"
    exit 1
fi
echo "  -> $BONOBO_EVAL_CSV"

# ============================================================
# Test B [CYA]: VIDD re-eval (scripts/eval_iptm.py, vidd env) — race check
# ============================================================
VIDD_EVAL_CSV=""
if [ "$RUN_RACE_CHECK" = "1" ]; then
    echo
    echo "=========================================================="
    echo "Test B [CYA]: VIDD re-eval (eval_iptm.py) for race check"
    echo "=========================================================="
    conda activate "$VIDD_CONDA_ENV"
    python scripts/eval_iptm.py \
        --input_csv "$INPUT_SNAPSHOT" \
        --antigen "$ANTIGEN" \
        --af_gpu_ids "$AF_GPU_IDS" \
        --cache_dir "$VIDD_EVAL_CACHE_DIR" \
        --write_inplace 0
    VIDD_EVAL_CSV="${INPUT_SNAPSHOT%.csv}_w_final_iptm.csv"
    if [ ! -f "$VIDD_EVAL_CSV" ]; then
        echo "ERROR: VIDD re-eval output not found at $VIDD_EVAL_CSV"
        exit 1
    fi
    echo "  -> $VIDD_EVAL_CSV"
else
    echo
    echo "(skipping Test B race check; RUN_RACE_CHECK=0)"
fi

# ============================================================
# Report
# ============================================================
echo
echo "=========================================================="
echo "Parity report: VIDD reward vs bonobo final ipTM"
echo "=========================================================="
BONOBO_EVAL_CSV="$BONOBO_EVAL_CSV" VIDD_EVAL_CSV="$VIDD_EVAL_CSV" \
INPUT_SNAPSHOT="$INPUT_SNAPSHOT" OUTPUT_ROOT="$OUTPUT_ROOT" \
python - <<'PY'
import os
import pandas as pd

orig = pd.read_csv(os.environ["INPUT_SNAPSHOT"])
bonobo = pd.read_csv(os.environ["BONOBO_EVAL_CSV"])[["sequence", "final_iptm"]].rename(
    columns={"final_iptm": "iptm_bonobo_final"}
)

if "iptm" not in orig.columns:
    raise SystemExit(
        "ERROR: input CSV has no `iptm` column — cannot run the reward-vs-bonobo "
        "test. Pass a VIDD design output.csv (its `iptm` column is the reward)."
    )

m = orig[["sequence", "iptm"]].rename(columns={"iptm": "iptm_reward"}).merge(
    bonobo, on="sequence", how="inner"
)
m["delta_reward_vs_bonobo"] = m["iptm_reward"] - m["iptm_bonobo_final"]

vidd_eval_csv = os.environ.get("VIDD_EVAL_CSV", "")
has_race = bool(vidd_eval_csv) and os.path.exists(vidd_eval_csv)
if has_race:
    vidd = pd.read_csv(vidd_eval_csv)[["sequence", "final_iptm"]].rename(
        columns={"final_iptm": "iptm_vidd_reeval"}
    )
    m = m.merge(vidd, on="sequence", how="inner")
    m["delta_reward_vs_reeval"] = m["iptm_reward"] - m["iptm_vidd_reeval"]

print()
print(f"Sequences compared: {len(m)} / {len(orig)}")
print()

cols = ["iptm_reward", "iptm_bonobo_final", "delta_reward_vs_bonobo"]
if has_race:
    cols = ["iptm_reward", "iptm_vidd_reeval", "iptm_bonobo_final",
            "delta_reward_vs_reeval", "delta_reward_vs_bonobo"]
print("Per-row:")
print(m[cols].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

def stats(s):
    return f"mean={s.mean():+.4f}  std={s.std():.4f}  max={s.max():+.4f}  min={s.min():+.4f}"

print()
print("=== Test A [CRITICAL]: VIDD reward vs bonobo final ipTM ===")
print("    Does the reward the model optimized match the metric we report? ~0 = parity.")
print(f"    delta_reward_vs_bonobo:  {stats(m['delta_reward_vs_bonobo'])}")
if has_race:
    print()
    print("=== Test B [CYA]: VIDD reward vs fresh re-eval (race check) ===")
    print("    Non-zero => multi-GPU race corrupted the logged reward. ~0 = clean.")
    print(f"    delta_reward_vs_reeval:  {stats(m['delta_reward_vs_reeval'])}")

out = os.path.join(os.environ["OUTPUT_ROOT"], "comparison.csv")
m.to_csv(out, index=False)
print()
print(f"Merged comparison written to: {out}")
PY
