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

# CDR-H3 positions (0-based) for the seed above. Override on the command line
# if you want to design CDR-H1/H2 too.
CDR_INDICES="${CDR_INDICES:-99,100,101,102,103,104,105,106,107,108,109,110,111}"

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
    --unmask_K 4 \
    --num_epochs 50 \
    --best_of_N 4 \
    --inference_best_of_N 32 \
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
