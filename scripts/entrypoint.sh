#!/bin/bash
# Entrypoint for the vidd-antibody container. Activates the vidd conda env and
# execs whatever command SageMaker (or the user) passes in.
#
# Mirrors rerd-antibody's entrypoint. Every export below is here because its
# absence caused a real failure somewhere in this fleet.
set -e

. /opt/conda/etc/profile.d/conda.sh
conda activate vidd

# scipy/sklearn compiled wheels link against a newer libstdc++ (CXXABI_1.3.15+)
# than ubuntu22.04 ships. The conda env has the right one — make sure the
# dynamic loader finds it first. (conda activate intentionally doesn't set this.)
export LD_LIBRARY_PATH="/opt/conda/envs/vidd/lib:${LD_LIBRARY_PATH:-}"

# *** JAX/TORCH COEXISTENCE — DO NOT REMOVE. ***
# VIDD runs JAX (colabdesign/AF2) and torch (diffusion) in the SAME process.
# JAX preallocates ~75% of the device by default, leaving torch nothing, and the
# result is `CUDA out of memory ... allocated by PyTorch` while JAX holds the
# rest. This killed Germinal on a 4-GPU run in this fleet and is KNIFE-EDGE: it
# passes a 1-GPU smoke test and fails the real multi-GPU run, so a green smoke
# is not evidence these can be dropped.
: "${XLA_PYTHON_CLIENT_PREALLOCATE:=false}"
: "${XLA_PYTHON_CLIENT_ALLOCATOR:=platform}"
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"
export XLA_PYTHON_CLIENT_PREALLOCATE XLA_PYTHON_CLIENT_ALLOCATOR PYTORCH_CUDA_ALLOC_CONF

# Multi-GPU default matches rerd-antibody. Caller overrides by setting it first.
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"
export CUDA_VISIBLE_DEVICES

# Weights baked into the image at /root/.mber. Allow override via env.
# NBB2_WEIGHTS_DIR points at a directory that really is populated: nothing on
# the --task ab path constructs NanoBodyBuilder2, but download_weights.sh bakes
# the weights anyway since it has no --skip-nbb2. Note mber pins its own
# weights_dir separately — warming ImmuneBuilder's default cache does NOT warm
# mber's, which caused a corrupt-weights race in this fleet.
: "${AF_PARAMS_DIR:=/root/.mber/af_params}"
: "${NBB2_WEIGHTS_DIR:=/root/.mber/nbb2_weights}"
export AF_PARAMS_DIR NBB2_WEIGHTS_DIR

exec "$@"
