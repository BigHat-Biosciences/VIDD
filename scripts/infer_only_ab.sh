#!/bin/bash
# Inference-only driver: re-runs Phase 2 (best-of-N + output.csv) against an
# existing trained checkpoint. By default, auto-picks the most recent
# output/ab_<TARGET>_iptm,plddt,cdr_plddt_*/models/model_best.ckpt.
#
# Typical usage:   TARGET=pdl1 bash scripts/infer_only_ab.sh
# Override:        WANDB_NAME=... CKPT=... TARGET=pdl1 bash scripts/infer_only_ab.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export AF_PARAMS_DIR="${AF_PARAMS_DIR:-$HOME/.mber/af_params}"

ANTIBODY_SEQUENCE="EVQLVESGGGLVQPGGSLRLSCAASGGFTFSSYAMWFRQAPGKEREFAISGSGGSTYYNADSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYCARLSITIRPYYGWGQGTLVTVSS"
CDR_INDICES="${CDR_INDICES:-26,27,28,29,30,31,32,33,34,47,48,49,50,51,52,53,54,55,56,57,95,96,97,98,99,100,101,102,103,104,105,106}"

TARGET="${TARGET:-pdl1}"
TARGET_UPPER="$(echo "$TARGET" | tr '[:lower:]' '[:upper:]')"
TARGET_LOWER="$(echo "$TARGET" | tr '[:upper:]' '[:lower:]')"

ANTIGEN_PDB="${ANTIGEN_PDB:-target_proteins/${TARGET_UPPER}.pdb}"
ANTIGEN_CHAIN="${ANTIGEN_CHAIN:-A}"
BIND_TARGET="${BIND_TARGET:-${TARGET_UPPER}}"

TEMPLATE_DIR="${TEMPLATE_DIR:-target_proteins}"
TEMPLATE_PDB="${TEMPLATE_PDB:-${TEMPLATE_DIR}/template_${TARGET_LOWER}.pdb}"
declare -A HOTSPOTS=(
    [pdl1]="A113"
    [bhrf1]="A60,A61,A63,A71"
    [il3]="A23,A25,A26,A31,A40,A104"
    [il20]="A58,A62,A101"
)
HOTSPOT="${HOTSPOT:-${HOTSPOTS[$TARGET_LOWER]:-}}"

# Auto-discover the most recent output dir for this target if not overridden.
# Match output/ab_<TARGET_UPPER>_<reward>_<wandb_name>_<timestamp>/. Picking
# by mtime handles target_update_interval bumps mid-run; the wandb_name suffix
# (the part after the rewards) is what we feed back as --wandb_name.
if [[ -z "${WANDB_NAME:-}" || -z "${CKPT:-}" ]]; then
    OUT_PATTERN="output/ab_${TARGET_UPPER}_iptm,plddt,cdr_plddt_*"
    LATEST_DIR="$(ls -1dt $OUT_PATTERN 2>/dev/null | head -n 1 || true)"
    if [[ -z "$LATEST_DIR" ]]; then
        echo "ERROR: no matching output dir for $OUT_PATTERN; set WANDB_NAME + CKPT explicitly." >&2
        exit 1
    fi
    # Strip the leading "output/ab_<TARGET>_<reward>_" prefix to recover wandb_name.
    PREFIX="output/ab_${TARGET_UPPER}_iptm,plddt,cdr_plddt_"
    WANDB_NAME="${WANDB_NAME:-${LATEST_DIR#$PREFIX}}"
    CKPT="${CKPT:-$LATEST_DIR/models/model_best.ckpt}"
    echo "[infer_only] using output dir: $LATEST_DIR"
    echo "[infer_only] WANDB_NAME=$WANDB_NAME"
    echo "[infer_only] CKPT=$CKPT"
fi
if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint not found at $CKPT" >&2
    exit 1
fi

python "$(dirname "$0")/train_and_infer_ab.py" \
    --task ab \
    --wandb_mode disabled \
    --wandb_group ab_distillation \
    --wandb_name "$WANDB_NAME" \
    --bind_target "$BIND_TARGET" \
    --antibody_sequence "$ANTIBODY_SEQUENCE" \
    --cdr_indices "$CDR_INDICES" \
    --antigen_pdb "$ANTIGEN_PDB" \
    --antigen_chain "$ANTIGEN_CHAIN" \
    --reward iptm,plddt,cdr_plddt \
    --reward_weight 1,0.1,0.1 \
    --batch_size 16 \
    --inference_batch_size 400 \
    --unmask_K 4 \
    --best_of_N 4 \
    --inference_best_of_N 4 \
    --use_value_xt \
    --rs_gen_model new \
    --reward_step \
    --template_pdb "$TEMPLATE_PDB" \
    --hotspot "$HOTSPOT" \
    --num_recycles 3 \
    --af_params_dir "$AF_PARAMS_DIR" \
    --af_gpu_ids 1,2,3 \
    --skip_train \
    --test_model_path "$CKPT"
