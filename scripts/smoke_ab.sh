#!/bin/bash
# Minimal antibody smoke test: 2 epochs, batch 4, single GPU, no NBB2 template.
# Confirms the train + inference loop runs end-to-end and writes timing files.
set -euo pipefail

export AF_PARAMS_DIR="${AF_PARAMS_DIR:-$HOME/.mber/af_params}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

python "$(dirname "$0")/train_and_infer_ab.py" \
    --task ab \
    --wandb_mode disabled \
    --wandb_name smoke \
    --antibody_sequence EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRLSITIRPRYYGLDVWGQGTLVTVSS \
    --cdr_indices 99,100,101,102,103,104,105,106,107,108,109,110,111 \
    --antigen_pdb target_proteins/PDL1.pdb \
    --antigen_chain A \
    --reward iptm,plddt,cdr_plddt \
    --reward_weight 1,0.1,0.1 \
    --batch_size 4 \
    --num_epochs 2 \
    --best_of_N 2 \
    --inference_best_of_N 4 \
    --af_params_dir "$AF_PARAMS_DIR"
