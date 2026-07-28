#!/bin/bash
# Run the 1-GPU antibody smoke test INSIDE the locally-built vidd-antibody image.
# Intended for a util box with one GPU; no ECR involvement.
#
#   PUSH=0 bash scripts/bh-deploy.sh            # build the image first
#   bash scripts/smoke_in_container.sh          # then this
#
# WHAT A GREEN RUN HERE DOES AND DOES NOT PROVE.
# Proves: the pip resolve produced a working env, torch and jax both see the GPU,
# AF2 weights baked correctly, the combined template loads, colabdesign
# prep_binder accepts the 118aa binder, and the train+infer loop writes output.
# That is where nearly all the build risk lives.
# Does NOT prove: JAX/torch coexistence across devices. VIDD runs JAX (AF2) and
# torch (diffusion) in one process; with --af_gpu_ids unset this takes the SERIAL
# path (evaluations/ab_af2_reward.py:436) and never contends for memory the way
# the real 4-GPU run does. That contention is what killed Germinal in this fleet.
# A green smoke is NOT evidence the entrypoint's XLA_PYTHON_CLIENT_* exports can
# be dropped, and it is NOT a substitute for a 4-GPU run before the 72h launch.
set -euxo pipefail

IMAGE="${IMAGE:-332120041740.dkr.ecr.us-east-1.amazonaws.com/vidd-antibody:latest}"
OUT_DIR="${OUT_DIR:-$PWD/smoke_out}"

mkdir -p "$OUT_DIR"

# CUDA_VISIBLE_DEVICES=0 overrides the entrypoint's 4-GPU default. The entrypoint
# uses := so an inherited value wins, which is why -e works here.
docker run --rm --gpus all \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v "$OUT_DIR":/home/output \
    "$IMAGE" \
    bash scripts/smoke_ab.sh 2>&1 | tee "$OUT_DIR/smoke.log"

echo "--- output written (judge by CONTENT, not exit code) ---"
find "$OUT_DIR" -type f | head -50
