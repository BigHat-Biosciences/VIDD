#!/bin/bash
# Antibody (VHH) train-then-infer driver for VIDD.
#
# Multi-GPU layout (e.g. g5.12xlarge: 4× A10G):
#   GPU 0   → torch (diffusion student/old/pre + NBB2 binder pre-fold)
#   GPU 1,2,3 → AF2 prediction workers (--af_gpu_ids 1,2,3)
#
# CUDA_VISIBLE_DEVICES must expose all four to the process; JAX preallocation
# is disabled so JAX doesn't grab all of GPU 0 — we keep that for torch.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

# AF2.3M weights — mirrors mber-open's download_weights.sh default.
export AF_PARAMS_DIR="${AF_PARAMS_DIR:-$HOME/.mber/af_params}"
# NBB2 weights for the one-shot template fold.
export NBB2_WEIGHTS_DIR="${NBB2_WEIGHTS_DIR:-$HOME/.mber/nbb2_weights}"

# Anti-PDL1 nanobody seed sequence (matches ProDifEvo-Refinement/run_ab_binding.sh).
ANTIBODY_SEQUENCE="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRLSITIRPRYYGLDVWGQGTLVTVSS"

# All three CDR positions (0-based) for the seed above. Approximate Kabat-style
# ranges for this VH framework:
#   CDR-H1 (positions 26-34, 9 res):  G F T F S S Y A M
#   CDR-H2 (positions 50-58, 9 res):  I S G S G G S T Y
#   CDR-H3 (positions 99-111, 13 res): R L S I T I R P R Y Y G L
# Total: 31 designed positions out of 125. Override CDR_INDICES if you want
# H3-only (use 99..111) or to tune the ranges per ANARCI/IMGT preference.
CDR_INDICES="${CDR_INDICES:-26,27,28,29,30,31,32,33,34,50,51,52,53,54,55,56,57,58,99,100,101,102,103,104,105,106,107,108,109,110,111}"

ANTIGEN_PDB="${ANTIGEN_PDB:-target_proteins/PDL1.pdb}"
ANTIGEN_CHAIN="${ANTIGEN_CHAIN:-A}"

python "$(dirname "$0")/train_and_infer_ab.py" \
    --task ab \
    --wandb_mode disabled \
    --wandb_group ab_distillation \
    --wandb_name ab_pdl1_train_then_infer \
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
    --use_template \
    --num_recycles 3 \
    --af_params_dir "$AF_PARAMS_DIR" \
    --nbb2_weights_dir "$NBB2_WEIGHTS_DIR" \
    --af_gpu_ids 1,2,3
