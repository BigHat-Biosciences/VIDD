#!/bin/bash
# VIDD setup: conda env + pip deps. Mirrors the layout used by ProDifEvo-Refinement.
#
# Usage:
#   bash install.sh                     # creates conda env 'vidd', installs everything
#   ENV_NAME=foo bash install.sh        # override env name
#   CUDA=cu121 bash install.sh          # pin pytorch to a CUDA wheel (default: cpu)
#   SKIP_AB=1 bash install.sh           # skip antibody extras (no AF2 / NBB2)
#
# After this script:
# * conda activate $ENV_NAME
# * Run any of scripts/protein_binder_*.sh (--task protein) or
#   scripts/train_and_infer_ab.sh (--task ab) once weights are in place.
#
# Not handled here (manual steps):
# * pyrosetta — license-gated; only needed for --task protein. Get a license
#   at https://www.pyrosetta.org/ and follow their conda channel instructions.
#   The antibody path lazy-imports pyrosetta, so --task ab works without it.
# * AF2 weights — download into $AF_PARAMS_DIR (default ~/.mber/af_params).
# * Combined binder+antigen template PDBs — generate offline via
#   ProDifEvo-Refinement/scripts/generate_template.py and pass through
#   --template_pdb. NBB2 is no longer a runtime dependency for VIDD.
set -euo pipefail

ENV_NAME="${ENV_NAME:-vidd}"
# Default 3.11 matches ProDifEvo-Refinement. VIDD's upstream README claims
# evodiff requires <=3.9, but evodiff imports cleanly on 3.11 and the rest of
# the stack (jax 0.5.2, dm-haiku>=0.0.14) requires >=3.10.
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA="${CUDA:-cpu}"   # cpu | cu118 | cu121 | cu124 ...
SKIP_AB="${SKIP_AB:-0}"

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- conda env -------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH. Install miniconda/anaconda first." >&2
    exit 1
fi

# Source conda for shell activation (works under bash even when not interactive).
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[install] reusing existing conda env: $ENV_NAME"
else
    echo "[install] creating conda env: $ENV_NAME (python=$PYTHON_VERSION)"
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi
conda activate "$ENV_NAME"

# --- conda-only deps -------------------------------------------------------
# pdbfixer + openmm: required by mber-open's AF2 amber relaxation path.
# hmmer: required by ANARCI for CDR auto-detection.
echo "[install] conda installing pdbfixer, openmm, hmmer"
conda install -y -c conda-forge pdbfixer openmm
conda install -y -c bioconda hmmer

# --- torch -----------------------------------------------------------------
# Install torch matching the requested CUDA wheel. Default cpu so the script
# works on machines without CUDA; override with CUDA=cu121 etc.
echo "[install] pip installing torch ($CUDA)"
if [[ "$CUDA" == "cpu" ]]; then
    pip install torch
else
    pip install torch --index-url "https://download.pytorch.org/whl/$CUDA"
fi

# --- requirements (single resolver pass) ----------------------------------
# CRITICAL: install base + antibody extras in ONE pip invocation so the
# resolver sees all constraints at once. Splitting the call lets transitive
# deps from requirements_ab.txt (chex, optax, numpy) silently violate
# evodiff's numpy<2 because evodiff is already "satisfied" on the second pass.
if [[ "$SKIP_AB" == "0" ]]; then
    echo "[install] pip installing base + antibody requirements (single resolve)"
    pip install -r "$REPO_DIR/requirements.txt" -r "$REPO_DIR/requirements_ab.txt"
    # Layer the cuda12 jaxlib on top of jax==0.5.2 for GPU hosts. This only
    # swaps jaxlib's wheel — chex/optax/numpy/etc. stay put.
    if [[ "$CUDA" != "cpu" ]]; then
        echo "[install] pip installing jax[cuda12] (GPU jaxlib for $CUDA host)"
        pip install 'jax[cuda12]==0.5.2'
    fi
else
    echo "[install] SKIP_AB=1 — pip installing base requirements only"
    pip install -r "$REPO_DIR/requirements.txt"
fi

cat <<EOF

[install] done.

Next steps:
  conda activate $ENV_NAME

  # For --task ab (antibody / nanobody design):
  export AF_PARAMS_DIR=\$HOME/.mber/af_params
  # download AF2 weights into \$AF_PARAMS_DIR (see mber-open/download_weights.sh).
  # generate combined binder+antigen template PDBs offline via
  # ProDifEvo-Refinement/scripts/generate_template.py, then pass them via
  # --template_pdb (or place at target_proteins/template_<target>.pdb).
  bash scripts/train_and_infer_ab.sh

  # For --task protein (existing PDL1/IFNAR2 binder paths):
  # also install pyrosetta per https://www.pyrosetta.org/ (license-gated).
  bash scripts/protein_binder_PD_L1.sh
EOF
