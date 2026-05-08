"""Antibody (VHH) train-then-infer driver for VIDD.

Runs the two phases of VIDD's protein binder pipeline back-to-back, against the
AF2-multimer + NBB2-template antibody reward backend:

  1. Train: policy distillation via finetune_reward_protein.run(args).
            Saves checkpoints + per-epoch timing_train.csv into output/ab_*/.

  2. Infer: best-of-N generation from the saved best checkpoint via
            finetune_reward_protein.best_of_n_test(...).
            Emits per-call timing_inference.csv plus a final timing_summary.txt.

CSV schemas:
  timing_train.csv     (already written by finetune_reward_protein.run)
    columns: epoch, wall_seconds, n_sequences, sec_per_seq, reward_seconds,
             af_calls, mean_reward, <metric>_mean for each metric in --reward.

  timing_inference.csv (this script)
    columns: phase, batch_idx, wall_seconds, n_sequences, sec_per_seq,
             reward_seconds, af_calls, mean_reward, <metric>_mean for each metric.

  output.csv           (this script)
    One row per final binder produced by the inference phase. Columns:
      <metric_1>, <metric_2>, ..., total_reward, diversity, sequence,
      cdr_indices, framework_fixed
    Mirrors ProDifEvo-Refinement's per-run output.csv so downstream analysis
    can treat VIDD and RERD outputs interchangeably.

  timing_summary.txt   (this script)
    Free-form rollup of train + infer wall time, AF2 reward time, total seqs,
    avg sec/seq for each phase. Mirrors ProDifEvo-Refinement/ab_refinement.py.

The two phases each construct a fresh AbAF2RewardCal so their _timings counters
start at zero — keeps the train and inference numbers cleanly separable.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

# Make the VIDD repo importable when this script is invoked directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluations.eval_models import initialize_eval_model
from finetune_reward_protein import (
    _build_ab_template,
    best_of_n_test,
    run as run_train,
)
from models.protein_gen_models import ProteinGenDiffusion


def _build_argparser() -> argparse.ArgumentParser:
    """Mirror finetune_reward_protein.py's argparse so users can pass any of its flags here."""
    p = argparse.ArgumentParser(
        description="VIDD antibody train + inference orchestrator (CDR-only design)."
    )
    p.add_argument('--decode_alg', type=str, default="sampling", choices=['SVDDtw', 'sampling'])
    p.add_argument('--SVDD_num_candidate', type=int, default=20)
    p.add_argument('--best_of_N', type=int, default=1)
    p.add_argument('--gkd_lmbda', type=float, default=0.5)

    p.add_argument('--teacher_alpha', type=float, default=1.0)
    p.add_argument('--reward_norm', type=str, default='none')
    p.add_argument('--logits_alpha', type=float, default=1.0)

    p.add_argument('--loss_func', type=str, default="KL")

    p.add_argument('--learning_rate', type=float, default=1e-5)
    p.add_argument('--use_amp', action='store_true')
    p.add_argument('--total_num_steps', type=int, default=1)
    p.add_argument('--num_accum_steps', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--student_initialize_pretrain', type=bool, default=True)

    p.add_argument('--seed', type=int, default=1776,
                   help="Default 1776 matches bonobo + RERD for ipTM reward parity.")
    p.add_argument('--wandb_name', type=str, default="debug")
    p.add_argument('--wandb_mode', type=str, default="disabled")
    p.add_argument('--wandb_group', type=str, default="")
    p.add_argument('--data_base_path', default="", type=str)
    p.add_argument('--eps', type=float, default=1e-5)
    p.add_argument('--target_update_interval', type=int, default=20)
    p.add_argument('--num_epochs', type=int, default=10000)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--save_every_n_epochs', type=int, default=50)

    p.add_argument('--task', type=str, default="ab", choices=['protein', 'ab'])
    p.add_argument('--reward', type=str, default="iptm,plddt,cdr_plddt")
    p.add_argument('--reward_weight', type=str, default='1,0.1,0.1')

    p.add_argument('--define_ss', type=str, default="b", choices=['a', 'b'])
    p.add_argument('--bind_target', type=str, default="PDL1")
    p.add_argument('--bind_target_folder', type=str, default="./target_proteins")
    p.add_argument('--gen_len', type=int, default=120,
                   help="Ignored when --task=ab; overridden to len(antibody_sequence).")
    p.add_argument('--unmask_K', default=4, type=int)
    p.add_argument('--folding_model', default="3b", choices=['650m', '3b'], type=str)

    p.add_argument('--test_model_path', type=str, default="")
    p.add_argument('--only_test', action='store_true')
    p.add_argument('--svdd_baseline', action='store_true')
    p.add_argument('--best_of_n_baseline', action='store_true')
    p.add_argument('--resume_train', action='store_true')
    p.add_argument('--resume_train_path', type=str, default="")

    p.add_argument('--rs_gen_model', default='pre')
    p.add_argument('--old_roll_in', action='store_true')

    p.add_argument("--ratio_clip", type=float, default=1e-4)
    p.add_argument("--adv_norm", action='store_true')

    p.add_argument('--reward_step', action='store_true')
    p.add_argument('--reward_temp', action='store_true')
    p.add_argument('--reward_estimate_times', default=1, type=int)
    p.add_argument('--use_value_xt', action='store_true')
    p.add_argument('--reward_clamp', default=1e6, type=float)

    p.add_argument('--msa_mode', default="none", type=str, choices=['none', 'full_msa'])

    # antibody flags
    p.add_argument('--antibody_sequence', type=str, required=True,
                   help="Seed VHH sequence (framework frozen, CDRs redesigned).")
    p.add_argument('--cdr_indices', type=str, required=True,
                   help="Comma-separated 0-based CDR positions in antibody_sequence.")
    p.add_argument('--antigen_pdb', type=str, default="")
    p.add_argument('--antigen_chain', type=str, default="A")
    p.add_argument('--template_pdb', type=str, default="",
                   help="Pre-built combined binder+antigen PDB (binder=H, antigen=A). "
                        "Required when 'iptm' is in --reward.")
    p.add_argument('--hotspot', type=str, default="",
                   help="Comma-separated antigen hotspot residues (e.g. 'A113').")
    p.add_argument('--af_gpu_ids', type=str, default="")
    p.add_argument('--af_params_dir', type=str, default="")
    p.add_argument('--num_recycles', type=int, default=3)

    # orchestrator-only flags
    p.add_argument('--inference_best_of_N', type=int, default=0,
                   help="best_of_N for the inference phase. 0 → reuse --best_of_N.")
    p.add_argument('--inference_batch_size', type=int, default=0,
                   help="Diffusion batch size for the inference phase (= number of "
                        "final binders produced). 0 → reuse --batch_size.")
    p.add_argument('--skip_train', action='store_true',
                   help="Run only the inference phase against --test_model_path.")
    p.add_argument('--skip_inference', action='store_true',
                   help="Run only the training phase.")
    return p


def _resolve_result_save_folder(args) -> str:
    """Replicate finetune_reward_protein.run()'s result_save_folder convention."""
    if 'iptm' in args.reward:
        return os.path.join('output', f"{args.task}_{args.bind_target}_{args.reward}_{args.wandb_name}")
    return os.path.join('output', f"{args.task}_{args.reward}_{args.wandb_name}")


def _write_output_csv(args, result_save_folder: str, result_dict: dict, reward_metric_names: list) -> None:
    """Write per-sample output.csv mirroring RERD's format.

    Columns: <reward1>,<reward2>,...,total_reward,diversity,sequence,cdr_indices,framework_fixed.
    Reward columns follow the order given in --reward.
    """
    sequences = result_dict.get("sequences")
    per_sample = result_dict.get("per_sample_rewards")
    total_rewards = result_dict.get("per_sample_total_reward")
    diversity = result_dict.get("diversity")
    if sequences is None or per_sample is None:
        print("[infer] best_of_n_test did not return per-sample data; skipping output.csv")
        return

    cdr = sorted(int(x) for x in args.cdr_indices.split(",") if x.strip())
    cdr_str = "[" + ", ".join(str(i) for i in cdr) + "]"

    out_path = os.path.join(result_save_folder, "output.csv")
    header = list(reward_metric_names) + ["total_reward", "diversity", "sequence", "cdr_indices", "framework_fixed"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, seq in enumerate(sequences):
            row = [float(per_sample[i, r_idx]) for r_idx in range(len(reward_metric_names))]
            total = float(total_rewards[i]) if total_rewards is not None else float("nan")
            row.extend([total, float(diversity) if diversity is not None else float("nan"),
                        seq, cdr_str, True])
            w.writerow(row)
    print(f"[infer] wrote per-sample output: {out_path}  ({len(sequences)} rows)")


def _run_inference(args, result_save_folder: str, device: torch.device) -> dict:
    """Standalone inference phase: build a fresh eval model + best_of_n_test."""
    best_ckpt = os.path.join(result_save_folder, "models", "model_best.ckpt")
    if not os.path.exists(best_ckpt):
        # Fallback: --test_model_path or last.ckpt.
        if args.test_model_path and os.path.exists(args.test_model_path):
            best_ckpt = args.test_model_path
        else:
            last_ckpt = os.path.join(result_save_folder, "models", "last.ckpt")
            if os.path.exists(last_ckpt):
                best_ckpt = last_ckpt
            else:
                raise FileNotFoundError(
                    f"No best/last checkpoint at {result_save_folder}/models/, and "
                    f"--test_model_path was not set. Cannot run inference."
                )
    print(f"[infer] loading best checkpoint: {best_ckpt}")

    # Fresh AbAF2RewardCal: zeroed _timings, separate from the train phase counters.
    eval_models = initialize_eval_model(args=args, device=device, result_save_folder=result_save_folder)

    # Inference uses --inference_best_of_N if set, else --best_of_N.
    n_best = args.inference_best_of_N or args.best_of_N

    # Truncate at start so the CSV reflects only this invocation. See the
    # matching change in finetune_reward_protein.run() for timing_train.csv.
    timing_csv = os.path.join(result_save_folder, "timing_inference.csv")
    reward_metric_names = args.reward.split(",")
    header = (
        ["phase", "batch_idx", "wall_seconds", "n_sequences", "sec_per_seq",
         "reward_seconds", "af_calls", "mean_reward"]
        + [f"{m}_mean" for m in reward_metric_names]
    )
    with open(timing_csv, "w", newline="") as f:
        csv.writer(f).writerow(header)

    n_seqs_before = eval_models._timings["n_sequences"]
    reward_s_before = eval_models._timings["reward_seconds"]
    n_calls_before = eval_models._timings["n_calls"]
    t0 = time.perf_counter()

    # Override batch_size for the inference phase if --inference_batch_size set.
    # The diffusion model reads args.batch_size, and best_of_n_test produces
    # one final binder per batch slot, so this controls the final binder count.
    saved_batch_size = args.batch_size
    inference_batch_size = args.inference_batch_size or args.batch_size
    if args.inference_batch_size:
        args.batch_size = args.inference_batch_size

    try:
        with torch.no_grad():
            result_dict = best_of_n_test(
                eval_models=eval_models,
                args=args,
                device=device,
                num_best_of_N=n_best,
                best_model_path=best_ckpt,
            )
    finally:
        args.batch_size = saved_batch_size

    wall = time.perf_counter() - t0
    n_seqs_iter = eval_models._timings["n_sequences"] - n_seqs_before
    reward_s_iter = eval_models._timings["reward_seconds"] - reward_s_before
    af_calls_iter = eval_models._timings["n_calls"] - n_calls_before
    sec_per_seq = wall / max(n_seqs_iter, 1)

    row = [
        "best_of_n",
        0,
        f"{wall:.3f}",
        n_seqs_iter,
        f"{sec_per_seq:.3f}",
        f"{reward_s_iter:.3f}",
        af_calls_iter,
        f"{float(result_dict.get('final_mean_reward_best_of_N', float('nan'))):.4f}",
    ]
    for m in reward_metric_names:
        v = result_dict.get(f"{m}_mean_reward")
        row.append(f"{float(v):.4f}" if v is not None else "nan")
    with open(timing_csv, "a", newline="") as f:
        csv.writer(f).writerow(row)

    # Per-sample output.csv — one row per final binder, columns mirror RERD's
    # output.csv (per-reward scores, diversity, sequence, cdr_indices,
    # framework_fixed). Reward column order follows --reward.
    _write_output_csv(args, result_save_folder, result_dict, reward_metric_names)

    # Final binders generated by THIS inference phase. best_of_n_test saves
    # one PDB per batch slot named f"best_of_{best_of_N}_repeat{i}.pdb", so we
    # use the effective inference batch size as the authoritative count
    # (rather than counting PDBs on disk, which could include stale files
    # from a previous run with a larger batch).
    n_final_binders = inference_batch_size

    return {
        "wall_seconds": wall,
        "n_sequences": n_seqs_iter,
        "reward_seconds": reward_s_iter,
        "af_calls": af_calls_iter,
        "n_final_binders": n_final_binders,
        "result_dict": result_dict,
        "eval_models_timings": dict(eval_models._timings),
    }


def _read_train_totals(result_save_folder: str) -> dict:
    """Read timing_train.csv and roll up to (wall, n_seqs, reward_s, af_calls)."""
    p = os.path.join(result_save_folder, "timing_train.csv")
    totals = {"wall_seconds": 0.0, "n_sequences": 0, "reward_seconds": 0.0, "af_calls": 0, "n_epochs": 0}
    if not os.path.exists(p):
        return totals
    with open(p, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            totals["wall_seconds"] += float(row.get("wall_seconds", 0.0) or 0.0)
            totals["n_sequences"] += int(row.get("n_sequences", 0) or 0)
            totals["reward_seconds"] += float(row.get("reward_seconds", 0.0) or 0.0)
            totals["af_calls"] += int(row.get("af_calls", 0) or 0)
            totals["n_epochs"] += 1
    return totals


def _write_summary(result_save_folder: str, train_totals: dict, infer_totals: dict, run_wall: float) -> None:
    avg = lambda total_s, n: (total_s / n) if n else 0.0
    train_seqs = train_totals["n_sequences"]
    infer_seqs = infer_totals.get("n_sequences", 0)
    train_wall = train_totals["wall_seconds"]
    infer_wall = infer_totals.get("wall_seconds", 0.0)
    n_final = infer_totals.get("n_final_binders", 0)

    lines = [
        "=== VIDD antibody run timing summary ===",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Output dir: {result_save_folder}",
        "",
        f"Total run wall time:                {run_wall:.2f} s",
        f"  Train phase wall time:            {train_wall:.2f} s ({train_totals['n_epochs']} epochs)",
        f"  Inference phase wall time:        {infer_wall:.2f} s",
        "",
        f"Train AF2 reward time:              {train_totals['reward_seconds']:.2f} s"
        f"  ({train_seqs} sequences, {train_totals['af_calls']} calls)",
        f"Inference AF2 reward time:          {infer_totals.get('reward_seconds', 0.0):.2f} s"
        f"  ({infer_seqs} sequences, {infer_totals.get('af_calls', 0)} calls)",
        "",
        # Sequences *scored* during training (each one passed through AF2 for
        # reward) and inference (best-of-N candidates evaluated to pick winners).
        f"Total sequences scored:             {train_seqs + infer_seqs}",
        f"Avg sec/seq scored (train, wall):   {avg(train_wall, train_seqs):.3f}",
        f"Avg sec/seq scored (infer, wall):   {avg(infer_wall, infer_seqs):.3f}",
        "",
        # Final binders = PDBs saved to saved_proteins/ at the end of the
        # inference phase (one per batch slot after best-of-N selection).
        f"Final binders generated:            {n_final}",
        f"Sec / final binder (infer wall):    {avg(infer_wall, n_final):.3f}",
        f"Sec / final binder (run wall):      {avg(run_wall, n_final):.3f}",
        "",
        f"Per-epoch:                          see timing_train.csv",
        f"Per-batch:                          see timing_inference.csv",
    ]
    with open(os.path.join(result_save_folder, "timing_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()

    if args.skip_train and args.skip_inference:
        raise SystemExit("Both --skip_train and --skip_inference set; nothing to do.")

    # CDR-only init constraints surface here too — fail fast before anything heavy.
    if args.task != 'ab':
        raise SystemExit("This driver is antibody-specific. Use --task ab.")
    if not args.antibody_sequence or not args.cdr_indices:
        raise SystemExit("--antibody_sequence and --cdr_indices are required.")

    # Stamp a launch timestamp into wandb_name so reruns with the same nominal
    # name (e.g. 'ab_pdl1_train_then_infer') don't share an output directory.
    # Skipped on --skip_train so resume-style runs land in the original output
    # dir (the user passes the already-stamped wandb_name verbatim).
    if not args.skip_train:
        launch_ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        args.wandb_name = f"{args.wandb_name}_{launch_ts}"

    result_save_folder = _resolve_result_save_folder(args)
    os.makedirs(os.path.join(result_save_folder, "models"), exist_ok=True)
    print(f"[orchestrator] result_save_folder = {result_save_folder}")

    run_t0 = time.perf_counter()

    # ---- Phase 1: train ----
    if not args.skip_train:
        print("[orchestrator] === Phase 1: training (policy distillation) ===")
        run_train(args)
    else:
        print("[orchestrator] --skip_train set; skipping training phase.")

    # Strip in-memory ab template state so the inference phase rebuilds against
    # its own fresh model + reward calculator.
    args._ab_template_tokens = None
    args._ab_cdr_indices = None
    args.only_test = False  # best_of_n_test is invoked directly; don't re-route through run()'s only_test branch.

    # ---- Phase 2: inference ----
    infer_totals = {"wall_seconds": 0.0, "n_sequences": 0, "reward_seconds": 0.0, "af_calls": 0}
    if not args.skip_inference:
        print("[orchestrator] === Phase 2: inference (best-of-N) ===")
        # Pin torch to cuda:0 explicitly; JAX shifts current_device when
        # dispatching AF2 workers to cuda:N. See finetune_reward_protein.run().
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        infer_totals = _run_inference(args, result_save_folder, device)
    else:
        print("[orchestrator] --skip_inference set; skipping inference phase.")

    run_wall = time.perf_counter() - run_t0
    train_totals = _read_train_totals(result_save_folder)
    _write_summary(result_save_folder, train_totals, infer_totals, run_wall)


if __name__ == "__main__":
    main()
