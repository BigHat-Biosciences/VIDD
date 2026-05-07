"""Re-score sequences in a CSV through VIDD's AF2 reward backend.

Mirrors ProDifEvo-Refinement/scripts/eval_iptm.py: takes a CSV of antibody
sequences and writes the same CSV with `final_iptm`, `final_plddt`,
`final_cdr_plddt` columns.

Use this when you have a CSV (e.g. compiled from a VIDD design run) and want
ipTM scores under VIDD's exact reward path — same AFModel, same template,
same rm_binder mask, same seed handling — without spinning up the full
diffusion training loop.

Examples
--------

    # Auto-fill template+hotspot for a baked target:
    python scripts/eval_iptm.py \\
        --input_csv ~/Downloads/vidd_il20.csv \\
        --antigen il20 \\
        --af_gpu_ids 1,2,3

    # Custom target (provide template + hotspot explicitly):
    python scripts/eval_iptm.py \\
        --input_csv my_seqs.csv \\
        --antigen_pdb ~/data/custom.pdb \\
        --template_pdb ~/data/template_custom.pdb \\
        --hotspot A45,A46
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Ensure repo root on sys.path so `from evaluations.ab_af2_reward import ...`
# resolves when this script is invoked directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from evaluations.ab_af2_reward import ALPHABET, AbAF2RewardCal  # noqa: E402


# Per-target metadata. Mirrors bonobo + RERD: antigen PDB, combined-template
# PDB (binder=H, antigen=A), and the AF2 hotspot string.
BAKED_TARGETS = {
    "pdl1":  {"hotspot": "A113"},
    "bhrf1": {"hotspot": "A60,A61,A63,A71"},
    "il3":   {"hotspot": "A23,A25,A26,A31,A40,A104"},
    "il20":  {"hotspot": "A58,A62,A101"},
}

# Bonobo-aligned VHH CDR positions (0-based) for the 118-aa scaffold used in
# scripts/train_and_infer_ab.sh. Override with --cdr_indices if your seed
# differs.
DEFAULT_CDR_INDICES = ",".join(
    str(i) for i in (
        list(range(26, 35))   # CDR1 (9)
        + list(range(47, 58)) # CDR2 (11)
        + list(range(95, 107))# CDR3 (12)
    )
)


def tokenize(seq: str) -> List[int]:
    return [ALPHABET.index(c) for c in seq]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.split("\n\n", 1)[0],
    )
    p.add_argument("--input_csv", required=True, help="CSV with a sequence column.")
    p.add_argument("--sequence_col", default="sequence")

    # Target. Either pass --antigen <name> (auto-fills paths/hotspot from
    # target_proteins/) or pass --antigen_pdb + --template_pdb + --hotspot
    # explicitly.
    p.add_argument("--antigen", default=None,
                   help=f"Baked target name; one of {sorted(BAKED_TARGETS)}. "
                        "Auto-resolves --antigen_pdb, --template_pdb, --hotspot.")
    p.add_argument("--antigen_pdb", default=None)
    p.add_argument("--antigen_chain", default="A")
    p.add_argument("--template_pdb", default=None,
                   help="Pre-made multi-chain PDB containing antigen + binder. "
                        "Required when --antigen is not a baked name.")
    p.add_argument("--hotspot", default=None,
                   help="Hotspot residues on antigen, e.g. 'A113'.")

    # CDR positions for cdr_plddt extraction.
    p.add_argument("--cdr_indices", default=DEFAULT_CDR_INDICES,
                   help="Comma-separated 0-based CDR positions. Default matches "
                        "the bonobo-aligned VHH scaffold from train_and_infer_ab.sh.")

    # AF2 backend.
    p.add_argument("--af_params_dir", default=None,
                   help="Falls back to $AF_PARAMS_DIR or ~/.mber/af_params.")
    p.add_argument("--num_recycles", default=3, type=int)
    p.add_argument("--af_gpu_ids", default="",
                   help="Comma-separated JAX device IDs for parallel AF, e.g. '1,2,3'.")

    # Reward selection. Default matches RERD's eval — extract iptm + plddts.
    p.add_argument("--reward", default="iptm,plddt,cdr_plddt",
                   help="Comma-separated reward names AbAF2RewardCal should compute.")
    p.add_argument("--reward_weight", default="1,0,0",
                   help="Weights — only matter for the aggregated reward, not the "
                        "per-metric values we extract.")

    # Output / caching.
    p.add_argument("--output_csv", default=None,
                   help="Where to write augmented CSV. If unset, see --write_inplace.")
    p.add_argument("--write_inplace", default=1, type=int,
                   help="1: overwrite --input_csv. 0: write {stem}_w_final_iptm.csv.")
    p.add_argument("--cache_dir", default="iptm_cache")
    p.add_argument("--cache_name", default=None,
                   help="Cache file name. Default: {input_csv stem}_iptm.csv.")
    p.add_argument("--chunk_size", default=12, type=int,
                   help="Sequences per reward_metrics call. Smaller = more frequent "
                        "checkpoints but more Python overhead.")

    # Determinism. Default matches RERD + bonobo so cross-evaluator parity
    # tests are seed-aligned. Colabdesign's Key() falls back to random.randint
    # when no seed is passed → seeding `random` here is the load-bearing one.
    p.add_argument("--seed", default=1776, type=int,
                   help="Seed for python random / numpy / torch. Default 1776 "
                        "matches bonobo/RERD eval.")

    args = p.parse_args()

    # Resolve baked target → fill in missing paths/hotspot.
    if args.antigen is not None:
        if args.antigen not in BAKED_TARGETS:
            sys.exit(f"--antigen must be one of {sorted(BAKED_TARGETS)}; got {args.antigen!r}")
        repo_root = Path(__file__).resolve().parents[1]
        target_dir = repo_root / "target_proteins"
        args.antigen_pdb = args.antigen_pdb or str(target_dir / f"{args.antigen.upper()}.pdb")
        args.template_pdb = args.template_pdb or str(target_dir / f"template_{args.antigen}.pdb")
        args.hotspot = args.hotspot or BAKED_TARGETS[args.antigen]["hotspot"]

    if args.antigen_pdb is None or args.template_pdb is None or args.hotspot is None:
        sys.exit(
            "Must provide either --antigen (baked target) OR all of "
            "--antigen_pdb / --template_pdb / --hotspot."
        )
    for path_arg, path in [("--antigen_pdb", args.antigen_pdb),
                           ("--template_pdb", args.template_pdb)]:
        if not os.path.exists(path):
            sys.exit(f"{path_arg} not found: {path}")

    return args


def load_cache(path: str) -> Dict[str, Tuple[float, float, float]]:
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    out: Dict[str, Tuple[float, float, float]] = {}
    for _, row in df.iterrows():
        out[row["sequence"]] = (
            float(row["final_iptm"]),
            float(row["final_plddt"]),
            float(row["final_cdr_plddt"]),
        )
    return out


def save_cache(path: str, cache: Dict[str, Tuple[float, float, float]]) -> None:
    rows = [
        {
            "sequence": s,
            "final_iptm": v[0],
            "final_plddt": v[1],
            "final_cdr_plddt": v[2],
        }
        for s, v in cache.items()
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def build_reward_args(cli: argparse.Namespace) -> SimpleNamespace:
    """Bundle CLI args into the namespace AbAF2RewardCal.__init__ expects."""
    return SimpleNamespace(
        reward=cli.reward,
        reward_weight=cli.reward_weight,
        cdr_indices=cli.cdr_indices,
        antigen_pdb=cli.antigen_pdb,
        antigen_chain=cli.antigen_chain,
        af_params_dir=cli.af_params_dir or "",
        num_recycles=cli.num_recycles,
        af_gpu_ids=cli.af_gpu_ids,
        template_pdb=cli.template_pdb,
        hotspot=cli.hotspot,
    )


def main() -> None:
    args = parse_args()

    # Seed early — before any colabdesign / mber-open import paths spin up
    # Key() instances. random.seed() is the one AF actually picks up via
    # colabdesign.shared.utils.Key.__init__'s random.randint fallback;
    # numpy/torch are belt-and-braces.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- Load CSV and validate ----
    df = pd.read_csv(args.input_csv)
    if args.sequence_col not in df.columns:
        sys.exit(
            f"--input_csv missing column {args.sequence_col!r}; got: {list(df.columns)}"
        )
    sequences: List[str] = df[args.sequence_col].astype(str).tolist()
    if not sequences:
        sys.exit("No sequences in input CSV.")

    seq_len = len(sequences[0])
    bad_lens = [(i, len(s)) for i, s in enumerate(sequences) if len(s) != seq_len]
    if bad_lens:
        sys.exit(
            f"All sequences must have the same length ({seq_len}). "
            f"Found mismatches at rows {bad_lens[:5]}{'...' if len(bad_lens) > 5 else ''}"
        )

    print(f"Sequence length: {seq_len}")
    print(f"Sequences to score: {len(sequences)}")

    # ---- Cache setup ----
    os.makedirs(args.cache_dir, exist_ok=True)
    stem = Path(args.input_csv).stem
    cache_name = args.cache_name or f"{stem}_iptm.csv"
    cache_path = os.path.join(args.cache_dir, cache_name)
    cache = load_cache(cache_path)
    todo = [s for s in sequences if s not in cache]
    print(f"Cached: {len(sequences) - len(todo)} | todo: {len(todo)}")

    # ---- Spin up reward backend ----
    if todo:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rew_args = build_reward_args(args)
        reward = AbAF2RewardCal(rew_args, device, result_save_folder="")

        # AbAF2RewardCal.reward_metrics returns (agg_list, per_metric_list)
        # when return_all_reward_term=True. per_metric_list is List[List[float]]
        # where the inner list is aligned with reward.metrics_name order.
        metric_names = reward.metrics_name
        for needed in ("iptm", "plddt", "cdr_plddt"):
            if needed not in metric_names:
                sys.exit(
                    f"--reward must include '{needed}' for this script to populate "
                    f"final_{needed}; got --reward {args.reward!r}"
                )
        idx_iptm = metric_names.index("iptm")
        idx_plddt = metric_names.index("plddt")
        idx_cdr_plddt = metric_names.index("cdr_plddt")

        for chunk_start in range(0, len(todo), args.chunk_size):
            chunk = todo[chunk_start: chunk_start + args.chunk_size]
            tokens = torch.tensor(
                [tokenize(s) for s in chunk], dtype=torch.long, device=device
            )
            _, per_metric = reward.reward_metrics(
                S_sp=tokens, return_all_reward_term=True,
            )
            for s, vals in zip(chunk, per_metric):
                cache[s] = (
                    float(vals[idx_iptm]),
                    float(vals[idx_plddt]),
                    float(vals[idx_cdr_plddt]),
                )
            save_cache(cache_path, cache)
            print(
                f"  chunk {chunk_start + len(chunk)}/{len(todo)} done; "
                f"cache -> {cache_path}"
            )

    # ---- Augment input CSV ----
    df["final_iptm"] = [cache[s][0] for s in sequences]
    df["final_plddt"] = [cache[s][1] for s in sequences]
    df["final_cdr_plddt"] = [cache[s][2] for s in sequences]

    if args.output_csv:
        out_path = args.output_csv
    elif args.write_inplace:
        out_path = args.input_csv
    else:
        in_path = Path(args.input_csv)
        out_path = str(in_path.with_name(f"{in_path.stem}_w_final_iptm.csv"))
    df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
