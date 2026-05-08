"""Per-target ipTM summary across VIDD output/ directories.

By default, picks the most recent run per target (by mtime) under
``output/ab_<TARGET>_iptm,*`` and reads its ``output.csv``. Pass explicit dirs
as positional args to override.

    python scripts/summarize_iptm.py
    python scripts/summarize_iptm.py output/ab_PDL1_*  output/ab_IL20_*
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import statistics
import sys


_TARGET_RE = re.compile(r"^ab_(?P<target>[A-Z0-9]+)_iptm")


def _latest_per_target(root: str = "output") -> dict[str, str]:
    """Group output dirs by target, return {target: most_recent_dir_path}."""
    by_target: dict[str, tuple[float, str]] = {}
    for path in glob.glob(os.path.join(root, "ab_*_iptm,*")):
        if not os.path.isdir(path):
            continue
        m = _TARGET_RE.match(os.path.basename(path))
        if not m:
            continue
        target = m.group("target")
        mtime = os.path.getmtime(path)
        prev = by_target.get(target)
        if prev is None or mtime > prev[0]:
            by_target[target] = (mtime, path)
    return {t: p for t, (_, p) in by_target.items()}


def _summarize(csv_path: str) -> dict | None:
    """Return iptm stats from output.csv, or None if unreadable/empty."""
    if not os.path.exists(csv_path):
        return None
    iptm = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        if "iptm" not in (reader.fieldnames or []):
            return None
        for row in reader:
            try:
                iptm.append(float(row["iptm"]))
            except (TypeError, ValueError):
                pass
    if not iptm:
        return None
    iptm_sorted = sorted(iptm)
    return {
        "n": len(iptm),
        "mean": statistics.fmean(iptm),
        "median": statistics.median(iptm),
        "stdev": statistics.pstdev(iptm) if len(iptm) > 1 else 0.0,
        "min": iptm_sorted[0],
        "max": iptm_sorted[-1],
        "top1": iptm_sorted[-1],
        "top10_mean": statistics.fmean(iptm_sorted[-10:]) if len(iptm) >= 10 else statistics.fmean(iptm_sorted),
        "frac_gt_0.5": sum(1 for v in iptm if v > 0.5) / len(iptm),
        "frac_gt_0.7": sum(1 for v in iptm if v > 0.7) / len(iptm),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="*", help="Run dirs to summarize. Default: latest per target under output/.")
    args = p.parse_args()

    if args.dirs:
        # Map provided dirs by parsed target name; deduplicate by latest mtime.
        by_target: dict[str, tuple[float, str]] = {}
        for d in args.dirs:
            d = d.rstrip("/")
            m = _TARGET_RE.match(os.path.basename(d))
            if not m:
                print(f"[warn] skipping {d}: cannot parse target from name", file=sys.stderr)
                continue
            t = m.group("target")
            mt = os.path.getmtime(d) if os.path.exists(d) else 0.0
            prev = by_target.get(t)
            if prev is None or mt > prev[0]:
                by_target[t] = (mt, d)
        targets = {t: p for t, (_, p) in by_target.items()}
    else:
        targets = _latest_per_target()

    if not targets:
        print("No matching output dirs found.", file=sys.stderr)
        sys.exit(1)

    header = f"{'target':<8} {'n':>4} {'mean':>7} {'median':>7} {'std':>6} {'min':>6} {'max':>6} {'top10':>7} {'>0.5':>6} {'>0.7':>6}  dir"
    print(header)
    print("-" * len(header))
    for target in sorted(targets):
        run_dir = targets[target]
        stats = _summarize(os.path.join(run_dir, "output.csv"))
        if stats is None:
            print(f"{target:<8} (no output.csv yet at {run_dir})")
            continue
        print(
            f"{target:<8} {stats['n']:>4d} "
            f"{stats['mean']:>7.3f} {stats['median']:>7.3f} {stats['stdev']:>6.3f} "
            f"{stats['min']:>6.3f} {stats['max']:>6.3f} {stats['top10_mean']:>7.3f} "
            f"{stats['frac_gt_0.5']:>6.1%} {stats['frac_gt_0.7']:>6.1%}  "
            f"{os.path.basename(run_dir)}"
        )


if __name__ == "__main__":
    main()
