#!/bin/bash
# Minimal antibody smoke test: 2 epochs, batch 4, single GPU.
# Confirms the train + inference loop runs end-to-end and writes timing files.
#
# This is the cheapest end-to-end check for the vidd-antibody container, but it
# is a 1-GPU check only. It does NOT exercise the JAX/torch coexistence path
# that the entrypoint's XLA_PYTHON_CLIENT_* exports exist to protect: JAX
# preallocation only starves torch once AF2 runs on its own devices. A green
# smoke here is not evidence the multi-GPU run will survive -- run
# scripts/train_and_infer_ab.sh on 4 GPUs before trusting the image.
#
# Everything below mirrors scripts/train_and_infer_ab.sh; only the training
# budget is cut down. Keep the two in sync.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export AF_PARAMS_DIR="${AF_PARAMS_DIR:-$HOME/.mber/af_params}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

cd "$(dirname "$0")/.."

# Bonobo-aligned VHH seed, 118aa. The length is load-bearing: the combined
# template PDB below carries the binder on chain H with exactly 118 residues,
# and AF2 prep_binder needs the designed sequence to match it.
ANTIBODY_SEQUENCE="EVQLVESGGGLVQPGGSLRLSCAASGGFTFSSYAMWFRQAPGKEREFAISGSGGSTYYNADSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYCARLSITIRPYYGWGQGTLVTVSS"
CDR_INDICES="26,27,28,29,30,31,32,33,34,47,48,49,50,51,52,53,54,55,56,57,95,96,97,98,99,100,101,102,103,104,105,106"

# The iptm reward path is a binder run, so ab_af2_reward.py hard-requires a
# pre-made combined target+binder template (antigen on chain A, binder on H).
# Without it the run raises before touching a GPU, which makes for a smoke test
# that can never pass.
TEMPLATE_PDB="${TEMPLATE_PDB:-target_proteins/template_pdl1.pdb}"
HOTSPOT="${HOTSPOT:-A113}"

python scripts/train_and_infer_ab.py \
    --task ab \
    --wandb_mode disabled \
    --wandb_name smoke \
    --bind_target PDL1 \
    --antibody_sequence "$ANTIBODY_SEQUENCE" \
    --cdr_indices "$CDR_INDICES" \
    --antigen_pdb target_proteins/PDL1.pdb \
    --antigen_chain A \
    --template_pdb "$TEMPLATE_PDB" \
    --hotspot "$HOTSPOT" \
    --reward iptm,plddt,cdr_plddt \
    --reward_weight 1,0.1,0.1 \
    --batch_size 4 \
    --num_epochs 2 \
    --best_of_N 2 \
    --inference_best_of_N 4 \
    --num_recycles 3 \
    --af_params_dir "$AF_PARAMS_DIR"
