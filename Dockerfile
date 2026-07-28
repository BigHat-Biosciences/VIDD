# vidd-antibody: VIDD (reward-guided discrete-diffusion policy distillation,
# AF2.3M backend) for SageMaker processing jobs.
#
# Mirrors the rerd-antibody Dockerfile, which itself mirrors bonobo's: an
# nvidia/cuda runtime base + miniconda + a per-repo conda env + pip deps +
# AF2 weights baked in. Layer order is chosen so the expensive steps (conda
# env, pip resolve, weight download) cache across code-only edits.
#
# Deliberate differences from rerd-antibody, each for a reason:
#   * No environment.yml — VIDD does not ship one. install.sh builds the env
#     imperatively, so the conda step is inlined here rather than copied.
#   * NBB2 is not a runtime dependency. install.sh says so and it checks out:
#     nothing on the --task ab path constructs NanoBodyBuilder2 — combined
#     binder+antigen template PDBs are generated offline and passed via
#     --template_pdb. The weights ARE still baked, because download_weights.sh
#     has no --skip-nbb2 and its step [2/4] runs unconditionally (see Layer 5).
#   * pyrosetta is NOT installed. It is license-gated, and the antibody path
#     lazy-imports it, so --task ab runs without it. --task protein will NOT
#     work in this image; that is intentional and out of scope for the rebuttal.
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Layer 1: system packages. python3.11 via deadsnakes for parity with the conda
# env. Uses the direct keyserver fetch rather than add-apt-repository, which
# calls the Launchpad REST API and hangs from inside Docker on flaky networks —
# the bonobo Dockerfile documents this and rerd hit it too.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gnupg ca-certificates curl && \
    apt-key adv --keyserver keyserver.ubuntu.com \
        --recv-keys F23C5A6CF475977595C89F51BA6932366A755776 && \
    echo "deb https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main" \
        > /etc/apt/sources.list.d/deadsnakes.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        python3.11 python3-pip vim make wget git build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# Layer 2: miniconda from the official image. Same pin as rerd-antibody.
COPY --from=continuumio/miniconda3:24.5.0-0 /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH

WORKDIR /home

# Layer 3: conda env + conda-only deps. Expensive and rarely invalidated.
# python 3.11 matches ProDifEvo-Refinement. VIDD's upstream README claims
# evodiff needs <=3.9, but install.sh notes evodiff imports cleanly on 3.11 and
# the rest of the stack (jax 0.5.2, dm-haiku>=0.0.14) requires >=3.10.
# pdbfixer/openmm and hmmer are conda-only per requirements_ab.txt's header.
RUN conda create -y -n vidd python=3.11 && \
    . /opt/conda/etc/profile.d/conda.sh && conda activate vidd && \
    conda install -y -c conda-forge pdbfixer openmm && \
    conda install -y -c bioconda hmmer && \
    conda clean -afy

# Layer 4: pip deps. Order matters and is taken from install.sh:
#   torch (cu128) FIRST so deepspeed / fair-esm / torch_geometric resolve
#   against a known torch; then base + ab requirements in ONE resolve (install.sh
#   does a single combined resolve deliberately); then jax pinned to 0.5.2.
# The jax pin is load-bearing, not cosmetic: requirements_ab.txt notes that
# without it pip pulls jax 0.10 + incompatible chex/optax/numpy on the first pass.
COPY requirements.txt requirements_ab.txt /home/
RUN . /opt/conda/etc/profile.d/conda.sh && conda activate vidd && \
    pip install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir -r /home/requirements.txt -r /home/requirements_ab.txt && \
    pip install --no-cache-dir 'jax[cuda12]==0.5.2'

# Layer 5: AF2 weights (~3.5GB) baked so containers start instantly on SageMaker
# instead of downloading per job.
# --skip-esm saves ~5GB. Its own help text warns "ESM2 is required", so this was
# checked rather than copied from rerd: the ONLY `import esm` in this repo is
# evaluations/protein_eval.py, which is the --task protein path. Nothing on the
# --task ab path imports it. fair-esm is still pip-installed (it is in
# requirements.txt) so the import would resolve; only the ~5GB of WEIGHTS are
# skipped. If --task protein is ever wanted in this image, drop --skip-esm.
#
# NBB2 IS NOT SKIPPED, despite not being a runtime dependency: download_weights.sh
# offers only --skip-esm / --with-esmfold, so step [2/4] pulls all four
# nanobody_model_* files from zenodo.org every build. Harmless (they are small,
# and their presence makes NBB2_WEIGHTS_DIR in the entrypoint truthful) but it
# puts zenodo on the build's critical path — if this layer fails, check there
# before anything else. Add a --skip-nbb2 flag upstream if that becomes a problem.
COPY mber-open/download_weights.sh /home/mber-open/download_weights.sh
RUN . /opt/conda/etc/profile.d/conda.sh && conda activate vidd && \
    yes "y" | bash /home/mber-open/download_weights.sh /root/.mber --skip-esm

# Layer 6: full source. Everything expensive above is already cached.
COPY . /home/

# Layer 7: mber-open editable — VIDD's AF2 reward path imports the mber AFModel
# subclass, same as RERD's AbAF2RewardCal does.
RUN . /opt/conda/etc/profile.d/conda.sh && conda activate vidd && \
    pip install --no-cache-dir -e /home/mber-open

ENV PYTHONPATH=/home
ENTRYPOINT ["/home/scripts/entrypoint.sh"]
CMD ["python", "scripts/train_and_infer_ab.py", "--help"]
