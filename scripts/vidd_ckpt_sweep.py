"""Checkpoint diversity sweep for the NeurIPS26 VIDD antibody runs (2026-10-26).

VIDD's paper (arXiv 2507.00445 v1, Tables 2/10/11) reports binder results "with diversity >= 0.5
to prevent model collapse"; Table 12 shows the same runs trained on until collapse (diversity
0.0006). Our faithful N = 1 reruns collapse to a single design at the reward-best checkpoint, so
the paper's rule has to be applied to the periodic checkpoints (models/model_N.ckpt, every
--save_every_n_epochs = 50 steps) that the running jobs already wrote: sample from each one and
measure diversity, so the sampling checkpoint can be chosen as the reward-best checkpoint whose
samples still have diversity >= 0.5.

Diversity = VIDD's own set_diversity (evaluations/protein_utils.py: mean pairwise fraction of
differing positions over all ordered pairs, divided by m^2), computed as in the paper "based on
token-level differences among a batch of generated sequences (batch size = 32)": one value per
sampled batch of 32, averaged over --n_batches batches. Two position sets are reported: the 32
designable CDR positions (the analogue of the paper's free-sequence metric for CDR-only design)
and the full 118-mer (what calc_diversity(S_sp) returns in the training loop; the frozen
framework caps it at 32/118 = 0.27 of the CDR value). Set-level values over all sampled
sequences, unique fraction, top-sequence share and distinct CDR3 count are reported too.

Run from /home inside the vidd image, one process per GPU:
    CUDA_VISIBLE_DEVICES=g python scripts/vidd_ckpt_sweep.py --sweep_ckpt_dirs /path/base:0 /path/resume:474 \
        --sweep_out /opt/ml/processing/results/sweep --sweep_worker g --sweep_n_workers 4 \
        <the run's COMMON + ARM + target args>
<dir>:<offset> = a models/ directory and the number of steps trained before its segment (0 for the
base run, pace_step_offset for a resume), so global_epoch = offset + local epoch.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from finetune_reward_protein import _ab_kwargs, _build_ab_template, set_seed  # noqa: E402
from evaluations.protein_utils import set_diversity  # noqa: E402
from models.gen_models import generate_xt_list  # noqa: E402
from models.protein_gen_models import ProteinGenDiffusion  # noqa: E402
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("train_and_infer_ab", os.path.join(_REPO_ROOT, "scripts", "train_and_infer_ab.py"))
_tia = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tia)          # by path: no dependence on scripts/ being importable as a package
_build_argparser = _tia._build_argparser

_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def fast_set_diversity(tokens: np.ndarray, positions: list[int]) -> float:
    """Exact value of set_diversity(tokens, masks=positions) in O(m * P * A):
    (1/P) sum_p (1 - sum_a f_{p,a}^2) with f the token frequency at position p
    (the m^2 normalisation counts the zero diagonal, as in VIDD's function)."""
    sub = tokens[:, positions]
    m = sub.shape[0]
    tot = 0.0
    for p in range(sub.shape[1]):
        _, counts = np.unique(sub[:, p], return_counts=True)
        tot += 1.0 - float(np.sum((counts / m) ** 2))
    return tot / sub.shape[1]


def list_ckpts(spec: str):
    d, _, off = spec.rpartition(":")
    off = int(off)
    out = []
    for p in glob.glob(os.path.join(d, "model_*.ckpt")):
        m = re.fullmatch(r"model_(\d+)\.ckpt", os.path.basename(p))
        if m:
            e = int(m.group(1))
            out.append({"path": p, "kind": "periodic", "local_epoch": e, "global_epoch": off + e, "segment": d, "offset": off})
    for name in ("model_best.ckpt", "last.ckpt"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            out.append({"path": p, "kind": name.removesuffix(".ckpt"), "local_epoch": -1, "global_epoch": -1, "segment": d, "offset": off})
    return sorted(out, key=lambda c: (c["kind"] != "periodic", c["global_epoch"]))


def main() -> None:
    p = _build_argparser()
    p.add_argument("--sweep_ckpt_dirs", nargs="+", required=True, help="<models dir>:<step offset> ...")
    p.add_argument("--sweep_out", required=True)
    p.add_argument("--sweep_worker", type=int, default=0)
    p.add_argument("--sweep_n_workers", type=int, default=1)
    p.add_argument("--sweep_n_batches", type=int, default=32, help="batches of --sweep_batch_size per checkpoint")
    p.add_argument("--sweep_batch_size", type=int, default=32, help="the paper's diversity batch")
    p.add_argument("--sweep_seed", type=int, default=1776)
    args = p.parse_args()
    os.makedirs(args.sweep_out, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpts = [c for spec in args.sweep_ckpt_dirs for c in list_ckpts(spec)]
    mine = [c for i, c in enumerate(ckpts) if i % args.sweep_n_workers == args.sweep_worker]
    print(f"[sweep] worker {args.sweep_worker}/{args.sweep_n_workers}: {len(mine)} of {len(ckpts)} checkpoints, "
          f"{args.sweep_n_batches} x {args.sweep_batch_size} samples each, seed {args.sweep_seed}", flush=True)

    model = ProteinGenDiffusion(args).to(device)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad = False
    args.gen_len = len(args.antibody_sequence)
    args._ab_template_tokens, args._ab_cdr_indices = _build_ab_template(args, model, device)
    cdr = sorted(int(x) for x in args.cdr_indices.split(",") if x.strip())
    full = list(range(args.gen_len))
    cdr3 = [i for i in cdr if i >= 95]
    args.batch_size = args.sweep_batch_size

    rows_path = os.path.join(args.sweep_out, f"sweep_worker{args.sweep_worker}.csv")
    fields = ["segment", "offset", "kind", "local_epoch", "global_epoch", "ckpt_epoch", "ckpt_best_reward",
              "n_seqs", "n_batches", "batch_size", "div_cdr_batch_mean", "div_cdr_batch_std", "div_cdr_batch_min",
              "div_cdr_set", "div_full_batch_mean", "div_full_set", "unique_frac", "top_seq_share", "n_distinct_cdr3",
              "n_distinct_cdr", "load_s", "sample_s", "path"]
    with open(rows_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    for c in mine:
        t0 = time.perf_counter()
        ck = torch.load(c["path"], map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ckpt_epoch = int(ck.get("epoch", -1))
        best_reward = float(ck.get("best_reward", float("nan")))
        if c["kind"] != "periodic":
            c["local_epoch"] = ckpt_epoch
            c["global_epoch"] = c["offset"] + ckpt_epoch
        elif ckpt_epoch != c["local_epoch"]:
            print(f"[sweep] WARNING {c['path']}: file epoch {c['local_epoch']} != stored epoch {ckpt_epoch}", flush=True)
        load_s = time.perf_counter() - t0
        set_seed(args.sweep_seed)
        t1 = time.perf_counter()
        toks, div_cdr_b, div_full_b = [], [], []
        with torch.no_grad():
            for b in range(args.sweep_n_batches):
                sample = generate_xt_list(args=args, generative_model=model, device=device, seq_len=args.gen_len,
                                          unmask_K=args.unmask_K, reward_model=None, num_best_of_N=1, **_ab_kwargs(args))[0]
                arr = sample.detach().cpu().numpy()
                if b == 0 and c is mine[0]:
                    # one-time check that the closed form equals VIDD's O(m^2) function
                    ref = set_diversity(arr, masks=[np.isin(np.arange(arr.shape[1]), cdr).astype(int)] * arr.shape[0])
                    assert abs(ref - fast_set_diversity(arr, cdr)) < 1e-9, (ref, fast_set_diversity(arr, cdr))
                    ref_full = set_diversity(arr)
                    assert abs(ref_full - fast_set_diversity(arr, full)) < 1e-9, (ref_full, fast_set_diversity(arr, full))
                toks.append(arr)
                div_cdr_b.append(fast_set_diversity(arr, cdr))
                div_full_b.append(fast_set_diversity(arr, full))
        sample_s = time.perf_counter() - t1
        all_toks = np.concatenate(toks, 0)
        seqs = ["".join(_ALPHABET[t] if 0 <= t < len(_ALPHABET) else "X" for t in r) for r in all_toks]
        counts: dict[str, int] = {}
        for s in seqs:
            counts[s] = counts.get(s, 0) + 1
        row = {"segment": os.path.basename(os.path.dirname(c["segment"].rstrip("/"))) or c["segment"], "offset": c["offset"],
               "kind": c["kind"], "local_epoch": c["local_epoch"], "global_epoch": c["global_epoch"], "ckpt_epoch": ckpt_epoch,
               "ckpt_best_reward": round(best_reward, 5), "n_seqs": len(seqs), "n_batches": args.sweep_n_batches,
               "batch_size": args.sweep_batch_size,
               "div_cdr_batch_mean": round(float(np.mean(div_cdr_b)), 5), "div_cdr_batch_std": round(float(np.std(div_cdr_b)), 5),
               "div_cdr_batch_min": round(float(np.min(div_cdr_b)), 5), "div_cdr_set": round(fast_set_diversity(all_toks, cdr), 5),
               "div_full_batch_mean": round(float(np.mean(div_full_b)), 5), "div_full_set": round(fast_set_diversity(all_toks, full), 5),
               "unique_frac": round(len(counts) / len(seqs), 5), "top_seq_share": round(max(counts.values()) / len(seqs), 5),
               "n_distinct_cdr3": len({s[cdr3[0]:cdr3[-1] + 1] for s in seqs}),
               "n_distinct_cdr": len({"".join(s[i] for i in cdr) for s in seqs}),
               "load_s": round(load_s, 1), "sample_s": round(sample_s, 1), "path": c["path"]}
        with open(rows_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        seq_path = os.path.join(args.sweep_out, f"seqs_{c['kind']}_g{c['global_epoch']:05d}_w{args.sweep_worker}.csv")
        tmp = seq_path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sequence", "batch_idx", "global_epoch", "kind", "seed"])
            for i, s in enumerate(seqs):
                w.writerow([s, i // args.sweep_batch_size, c["global_epoch"], c["kind"], args.sweep_seed])
        os.replace(tmp, seq_path)
        print(f"[sweep] {c['kind']:10s} g{c['global_epoch']:5d} (local {c['local_epoch']:5d}, stored {ckpt_epoch:5d}) "
              f"div_cdr batch={row['div_cdr_batch_mean']:.3f} set={row['div_cdr_set']:.3f} full={row['div_full_batch_mean']:.3f} "
              f"unique={row['unique_frac']:.3f} top={row['top_seq_share']:.3f} best_reward={best_reward:.4f} "
              f"load {load_s:.1f}s sample {sample_s:.1f}s", flush=True)
    with open(os.path.join(args.sweep_out, f"sweep_worker{args.sweep_worker}.done.json"), "w") as f:
        json.dump({"worker": args.sweep_worker, "n_ckpts": len(mine), "seed": args.sweep_seed,
                   "n_batches": args.sweep_n_batches, "batch_size": args.sweep_batch_size, "cdr": cdr}, f)
    print(f"[sweep] worker {args.sweep_worker} done: {len(mine)} checkpoints", flush=True)


if __name__ == "__main__":
    main()
