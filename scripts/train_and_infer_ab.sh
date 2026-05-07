#!/bin/bash
# Antibody (VHH) train-then-infer driver for VIDD.
#
# Multi-GPU layout (e.g. g5.12xlarge: 4× A10G):
#   GPU 0     → torch (diffusion student/old/pre)
#   GPU 1,2,3 → AF2 prediction workers (--af_gpu_ids 1,2,3)
#
# CUDA_VISIBLE_DEVICES must expose all four to the process; JAX preallocation
# is disabled so JAX doesn't grab all of GPU 0 — we keep that for torch.
#
# Templates: pre-built combined binder+antigen PDBs (binder=H, antigen=A) are
# loaded from $TEMPLATE_DIR/template_<target>.pdb. Generate offline with
# ProDifEvo-Refinement/scripts/generate_template.py.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

# AF2.3M weights — mirrors mber-open's download_weights.sh default.
export AF_PARAMS_DIR="${AF_PARAMS_DIR:-$HOME/.mber/af_params}"

# VHH (nanobody) seed matching bonobo's framework layout
# (run_bonobo_af_multigpu.py:256 / run_vsd_bonobo.py:280-285):
#   FR1 (26):  EVQLVESGGGLVQPGGSLRLSCAASG
#   CDR1 (9):  GFTFSSYAM      <- filler; masked at AF2 template via rm_binder
#   FR2 (12):  WFRQAPGKEREF   <- canonical VHH FR2 (note WFRQ + REF hallmarks)
#   CDR2 (11): AISGSGGSTYY    <- filler
#   FR3 (37):  NADSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYC
#   CDR3 (12): ARLSITIRPYYG   <- filler
#   FR4 (11):  WGQGTLVTVSS
# Total length: 118 (matches bonobo).
ANTIBODY_SEQUENCE="EVQLVESGGGLVQPGGSLRLSCAASGGFTFSSYAMWFRQAPGKEREFAISGSGGSTYYNADSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYCARLSITIRPYYGWGQGTLVTVSS"

# CDR positions (0-based) matching the bonobo-aligned seed above:
#   CDR1: 26..34   (9)
#   CDR2: 47..57   (11)
#   CDR3: 95..106  (12)
# Total: 32 designed positions out of 118.
CDR_INDICES="${CDR_INDICES:-26,27,28,29,30,31,32,33,34,47,48,49,50,51,52,53,54,55,56,57,95,96,97,98,99,100,101,102,103,104,105,106}"

# Target selection. Override with TARGET=pdl1|bhrf1|il3|il20 (case-insensitive),
# or set ANTIGEN_PDB / BIND_TARGET directly to use a custom PDB.
#   TARGET=bhrf1 bash scripts/train_and_infer_ab.sh
TARGET="${TARGET:-pdl1}"
TARGET_UPPER="$(echo "$TARGET" | tr '[:lower:]' '[:upper:]')"
TARGET_LOWER="$(echo "$TARGET" | tr '[:upper:]' '[:lower:]')"

ANTIGEN_PDB="${ANTIGEN_PDB:-target_proteins/${TARGET_UPPER}.pdb}"
ANTIGEN_CHAIN="${ANTIGEN_CHAIN:-A}"
BIND_TARGET="${BIND_TARGET:-${TARGET_UPPER}}"
WANDB_NAME="${WANDB_NAME:-ab_${TARGET_LOWER}_train_then_infer}"

# Pre-built combined template (binder chain H + antigen chain A) and antigen
# hotspot residues for AF2 prep_binder. Mirrors bonobo's per-target settings.
TEMPLATE_DIR="${TEMPLATE_DIR:-target_proteins}"
TEMPLATE_PDB="${TEMPLATE_PDB:-${TEMPLATE_DIR}/template_${TARGET_LOWER}.pdb}"
declare -A HOTSPOTS=(
    [pdl1]="A113"
    [bhrf1]="A60,A61,A63,A71"
    [il3]="A23,A25,A26,A31,A40,A104"
    [il20]="A58,A62,A101"
)
HOTSPOT="${HOTSPOT:-${HOTSPOTS[$TARGET_LOWER]:-}}"

if [[ ! -f "$ANTIGEN_PDB" ]]; then
    echo "ERROR: antigen PDB not found at $ANTIGEN_PDB" >&2
    echo "       (TARGET=$TARGET, expected target_proteins/${TARGET_UPPER}.pdb)" >&2
    exit 1
fi
if [[ ! -f "$TEMPLATE_PDB" ]]; then
    echo "ERROR: combined template PDB not found at $TEMPLATE_PDB" >&2
    echo "       Generate via ProDifEvo-Refinement/scripts/generate_template.py" >&2
    exit 1
fi
if [[ -z "$HOTSPOT" ]]; then
    echo "ERROR: no baked hotspot for TARGET=$TARGET_LOWER. Set HOTSPOT=... explicitly." >&2
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
    --num_epochs 50 \
    --best_of_N 4 \
    --inference_best_of_N 4 \
    --learning_rate 1e-5 \
    --target_update_interval 20 \
    --gkd_lmbda 0.8 \
    --teacher_alpha 1.0 \
    --use_value_xt \
    --rs_gen_model new \
    --reward_step \
    --template_pdb "$TEMPLATE_PDB" \
    --hotspot "$HOTSPOT" \
    --num_recycles 3 \
    --af_params_dir "$AF_PARAMS_DIR" \
    --af_gpu_ids 1,2,3
