# VIDD — antibody (VHH) extension

This is BigHat's fork of [VIDD](https://arxiv.org/abs/2507.00445), adapted for
nanobody (VHH) CDR-only design with an AlphaFold2-multimer reward and one-shot
NanoBodyBuilder2 templating. The upstream `protein` and `dna` paths still work
unchanged.

## What's new vs upstream

- **`--task ab`** — new task that wires CDR-only masked diffusion to an AF2
  reward backend. Framework residues are frozen via the seed sequence; only
  positions in `--cdr_indices` get unmasked by the sampler.
- **AF2-multimer reward** (`evaluations/ab_af2_reward.py`) — replaces the
  ESMFold/colabdesign monomer-binder reward with `mk_afdesign_model(
  protocol="binder", use_multimer=True)`. Returns `iptm`, `plddt`, `cdr_plddt`,
  `ptm`, `radius` per generated sequence.
- **NBB2 binder templating** — one-shot NanoBodyBuilder2 fold of the seed
  antibody, combined with the antigen, used as the AF2 binder template with
  CDR positions masked out via `rm_binder`. Per-candidate AF2 cost is just a
  forward pass; no per-candidate NBB2 fold or `_prep_binder`.
- **Multi-GPU AF2 dispatch** — `--af_gpu_ids 1,2,3` builds one AF2 worker per
  JAX device and shards sequence predictions across them via a thread pool.
  Torch stays on GPU 0.
- **Train + inference orchestrator** (`scripts/train_and_infer_ab.py`) — runs
  policy distillation followed by best-of-N inference, emitting per-epoch
  `timing_train.csv`, per-batch `timing_inference.csv`, and a final
  `timing_summary.txt`.
- **Lazy pyrosetta** — pyrosetta only loads when the existing `--task protein`
  eval is constructed. Antibody runs work on a box without pyrosetta.

## Setup

### 1. Conda env + pip deps

```bash
bash install.sh                    # creates conda env 'vidd' (python=3.9)
# Or with CUDA torch:
CUDA=cu121 bash install.sh
```

Knobs:
- `ENV_NAME=foo` — override conda env name
- `PYTHON_VERSION=3.11` — default; matches ProDifEvo-Refinement. Required by
  `jax>=0.4.31` and `dm-haiku>=0.0.14`. Upstream VIDD README's "≤ 3.9" claim
  for evodiff is stale — evodiff imports cleanly on 3.11.
- `CUDA=cpu | cu118 | cu121 | cu124` — pytorch wheel index
- `SKIP_AB=1` — skip AF2 / NBB2 install

`install.sh` handles:
- conda env creation
- conda-only deps: `pdbfixer`, `openmm` (NBB2 deps), `hmmer` (ANARCI dep)
- pip: `torch`, `requirements.txt` (base), `requirements_ab.txt` (AF2 + NBB2)

Manual steps (not in `install.sh`):
- **pyrosetta** — license-gated, only needed for `--task protein`. License at
  https://www.pyrosetta.org/. Antibody runs do not need it.
- **AF2 weights** — see below.
- **NBB2 weights** — auto-downloaded on first use into `$NBB2_WEIGHTS_DIR`.

### 2. AlphaFold2 weights

```bash
export AF_PARAMS_DIR=$HOME/.mber/af_params
mkdir -p "$AF_PARAMS_DIR"
# Then download params_model_*_multimer_v3.npz into $AF_PARAMS_DIR.
# Easiest path: clone mber-open and run its download_weights.sh.
```

The reward backend reads weights from `$AF_PARAMS_DIR` (or `--af_params_dir`).

### 3. NBB2 weights

Auto-downloaded into `$NBB2_WEIGHTS_DIR` (default `~/.mber/nbb2_weights`) the
first time `NanoBodyBuilder2()` is constructed. No manual step needed.

## Running the antibody pipeline

### Quick smoke test (single GPU, no template)

```bash
conda activate vidd
python scripts/train_and_infer_ab.py \
    --task ab \
    --wandb_mode disabled \
    --wandb_name smoke \
    --antibody_sequence EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRLSITIRPRYYGLDVWGQGTLVTVSS \
    --cdr_indices 99,100,101,102,103,104,105,106,107,108,109,110,111 \
    --antigen_pdb target_proteins/PDL1.pdb \
    --antigen_chain A \
    --reward iptm,plddt,cdr_plddt \
    --reward_weight 1,0.1,0.1 \
    --batch_size 4 --num_epochs 2 --best_of_N 2 --inference_best_of_N 4
```

Confirms the train + inference loop runs end-to-end and writes timing files.

### Full run (4× GPU, NBB2 template, anti-PDL1)

```bash
bash scripts/train_and_infer_ab.sh
```

Defaults to:
- `CUDA_VISIBLE_DEVICES=0,1,2,3`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`
- GPU 0 → torch (diffusion student/old/pre + NBB2)
- GPU 1,2,3 → AF2 prediction workers (`--af_gpu_ids 1,2,3`)
- Anti-PDL1 nanobody seed; CDR-H3 indices 99–111

Override anything via env vars or by editing the script:
- `CDR_INDICES=99,100,...` — swap the CDR positions
- `ANTIGEN_PDB=path/to/x.pdb` — swap the antigen
- The full VIDD argparse is forwarded; pass any flag through `python
  scripts/train_and_infer_ab.py ...` directly if you want finer control.

### CLI flags (antibody-specific)

| Flag | Purpose |
|---|---|
| `--task ab` | Switches to CDR-only design + AF2 reward backend. |
| `--antibody_sequence <str>` | Seed VHH (framework frozen, length sets `gen_len`). |
| `--cdr_indices <csv>` | 0-based positions in the seed that get redesigned. |
| `--antigen_pdb <path>` | Antigen structure (required for `iptm`). |
| `--antigen_chain <id>` | Antigen chain ID (default `A`). |
| `--use_template` | One-shot NBB2 fold + AF2 binder templating. |
| `--nbb2_weights_dir <path>` | NBB2 weights (default `$NBB2_WEIGHTS_DIR` or `~/.mber/nbb2_weights`). |
| `--af_params_dir <path>` | AF2 weights (default `$AF_PARAMS_DIR` or `~/.mber/af_params`). |
| `--af_gpu_ids <csv>` | JAX device indices for AF2 workers (e.g. `1,2,3`). Empty/single → serial. |
| `--num_recycles <int>` | AF2 recycles per prediction (default 3). |
| `--inference_best_of_N <int>` | best-of-N for the inference phase (default: reuse `--best_of_N`). |
| `--skip_train` / `--skip_inference` | Run only one phase. |

Reward metrics available for `--reward`: `iptm`, `plddt`, `cdr_plddt`, `ptm`,
`radius`. Weights via `--reward_weight` (csv, same length).

## Outputs

Per run, `output/ab_<bind_target>_<reward>_<wandb_name>/` contains:

- `models/` — `model_best.ckpt`, `model_best_plddt.ckpt`, `last.ckpt`,
  periodic `model_<epoch>.ckpt` snapshots.
- `saved_proteins/` — generated AF2 PDBs for inference-phase best-of-N.
- `timing_train.csv` — one row per training epoch:
  `epoch, wall_seconds, n_sequences, sec_per_seq, reward_seconds, af_calls,
  mean_reward, <metric>_mean ...`
- `timing_inference.csv` — one row per inference-phase best-of-N batch:
  `phase, batch_idx, wall_seconds, n_sequences, sec_per_seq, reward_seconds,
  af_calls, mean_reward, <metric>_mean ...`
- `timing_summary.txt` — human-readable rollup of train + inference wall time,
  AF2 reward time, total sequences scored, avg sec/seq for each phase.

The training tqdm bar shows live `agg`, `s/seq`, and per-metric means from the
previous epoch.

## Multi-GPU layout

Default 4× GPU layout (e.g. g5.12xlarge):

```
GPU 0   torch         diffusion student/old/pre + NBB2 binder fold
GPU 1   AF2 worker A  one mk_afdesign_model pinned to jax.devices()[1]
GPU 2   AF2 worker B  one mk_afdesign_model pinned to jax.devices()[2]
GPU 3   AF2 worker C  one mk_afdesign_model pinned to jax.devices()[3]
```

`--af_gpu_ids` indexes into `jax.devices()`. With `CUDA_VISIBLE_DEVICES=0,1,2,3`,
`jax.devices()[i]` corresponds to physical CUDA device `i`. The
`XLA_PYTHON_CLIENT_PREALLOCATE=false` env var keeps JAX from grabbing all of
GPU 0 — that's reserved for torch.

For a single-GPU box, omit `--af_gpu_ids` (or pass `""`); the reward backend
falls back to a serial single-device path.

## Troubleshooting

- **`ImportError: colabdesign`** — `pip install -r requirements_ab.txt` (or
  re-run `bash install.sh`). Confirm jax is installed too.
- **`KeyError: 'i_ptm'`** — AF2 was built without `use_multimer=True`. The
  antibody backend hardcodes `use_multimer=True`; this only happens if the
  reward stack is mismatched. Rebuild the env.
- **`--af_gpu_ids ... out of range`** — `jax.devices()` doesn't expose those
  indices. Check `CUDA_VISIBLE_DEVICES` matches what JAX sees.
- **`pyrosetta` ImportError on `--task protein`** — pyrosetta isn't installed.
  Antibody runs (`--task ab`) don't need it. For `--task protein`, install
  pyrosetta from its conda channel after accepting the license.
- **NBB2 first run is slow** — the first `NanoBodyBuilder2()` construction
  downloads weights (~hundreds of MB). One-shot.
- **High AF2 recompile time** — colabdesign recompiles when binder length
  changes. Templating keeps `_prep_binder` to one call; without it, every new
  `ab_len` triggers a recompile. Stick to a fixed seed length.

## Layout summary (just the antibody bits)

```
evaluations/
  ab_af2_reward.py          # AbAF2RewardCal: AF2-multimer + NBB2 + multi-GPU + _timings
  eval_models.py            # task='ab' branch dispatches to AbAF2RewardCal
  protein_eval_bind_colabdesign.py  # pyrosetta import made lazy

models/
  gen_models.py             # generate_xt_list now accepts frozen_template_tokens / cdr_indices

finetune_reward_protein.py  # antibody CLI flags + _build_ab_template + per-epoch timing CSV

scripts/
  train_and_infer_ab.py     # train → infer orchestrator (timing CSVs + summary)
  train_and_infer_ab.sh     # bash entry point with multi-GPU env vars

install.sh                  # conda env + pip deps
requirements.txt            # base
requirements_ab.txt         # AF2 + NBB2 extras
```
