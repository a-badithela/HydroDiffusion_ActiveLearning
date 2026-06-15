"""
CLI entry point for the scoring / acquisition step of one AL round.

Usage
-----
python -m active_learning.run_acquisition \\
    --split_csv    runs/al/round0/split.csv \\
    --out_csv      runs/al/round1/split.csv \\
    --acquisition  random|ensemble|representativeness \\
    --k            10 \\
    [--ckpt_dir    <dir>]          # dir with seed subdirs containing checkpoint.pt
    [--ckpt_pattern <glob>]        # shell glob to match seed run dirs (e.g. /path/runs/20260525_al-r0-*_seed*)
    [--device      cuda:0] \\
    [--stride      90] \\
    [--alpha       1.0]            # representativeness only
    [--diversity_pass]             # representativeness only

Checkpoint discovery (mutually exclusive, --ckpt_pattern takes precedence):
  --ckpt_pattern: expands via glob; each matching dir is expected to contain checkpoint.pt
  --ckpt_dir:     scans ckpt_dir/*/checkpoint.pt (legacy layout)

Stopping logic
--------------
If the sample pool is empty, the script exits with code 2.
schedule_al.sh uses this to detect that no more rounds are needed.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

# Make sure the repo root is on the path when run as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active_learning.split_manager import ActiveLearningSplit
from active_learning.acquisition.random import RandomAcquisition
from active_learning.acquisition.ensemble import EnsembleDisagreement
from active_learning.acquisition.representativeness import RepresentativenessAcquisition
from active_learning.basin_scorer import BasinScorer


def find_checkpoints_in_dir(ckpt_dir: str) -> list[Path]:
    """Find checkpoints via legacy ckpt_dir layout: ckpt_dir/*/checkpoint.pt"""
    ckpts = sorted(Path(ckpt_dir).glob("*/checkpoint.pt"))
    if not ckpts:
        direct = Path(ckpt_dir) / "checkpoint.pt"
        if direct.exists():
            ckpts = [direct]
    return ckpts


def find_checkpoints_by_pattern(pattern: str) -> list[Path]:
    """Expand a shell glob; each matching directory must contain checkpoint.pt"""
    run_dirs = sorted(glob.glob(pattern))
    ckpts = []
    for d in run_dirs:
        ckpt = Path(d) / "checkpoint.pt"
        if ckpt.exists():
            ckpts.append(ckpt)
        else:
            print(f"[run_acquisition] WARNING: matched dir has no checkpoint.pt: {d}", file=sys.stderr)
    return ckpts


def main() -> None:
    parser = argparse.ArgumentParser(description="AL acquisition step")
    parser.add_argument("--split_csv",    required=True, help="Current split CSV path")
    parser.add_argument("--ckpt_dir",    default=None, help="Dir containing seed subdirs with checkpoint.pt files (legacy)")
    parser.add_argument("--ckpt_pattern", default=None, help="Shell glob to match seed run dirs (takes precedence over --ckpt_dir)")
    parser.add_argument("--out_csv",     required=True, help="Output path for updated split CSV")
    parser.add_argument("--acquisition", default="random",
                        choices=["random", "ensemble", "representativeness"])
    parser.add_argument("--k",           type=int, default=10, help="Basins to promote per round")
    parser.add_argument("--device",      default="cpu",  help="PyTorch device for inference (e.g. cuda:0)")
    parser.add_argument("--stride",      type=int, default=90, help="Inference stride in days")
    parser.add_argument("--alpha",       type=float, default=1.0,
                        help="Seed-similarity penalty weight (representativeness only)")
    parser.add_argument("--diversity_pass", action="store_true",
                        help="Greedy farthest-point diversity pass (representativeness only)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load split
    # ------------------------------------------------------------------
    split = ActiveLearningSplit(args.split_csv)
    print(f"[run_acquisition] Split loaded: {split.summary()}")
    print(f"[run_acquisition] Acquisition: {args.acquisition}, k={args.k}")

    if len(split.sample_ids) == 0:
        print("[run_acquisition] Sample pool is empty — nothing to promote. Exiting.")
        sys.exit(2)   # schedule_al.sh checks for exit code 2 to stop the loop.

    k = min(args.k, len(split.sample_ids))

    # ------------------------------------------------------------------
    # Build acquisition function
    # ------------------------------------------------------------------
    if args.acquisition == "random":
        acq = RandomAcquisition()

    else:
        # Both ensemble and representativeness need checkpoints.
        if args.ckpt_pattern is None and args.ckpt_dir is None:
            parser.error("--ckpt_pattern or --ckpt_dir is required for ensemble / representativeness")

        if args.ckpt_pattern is not None:
            ckpts = find_checkpoints_by_pattern(args.ckpt_pattern)
            source_desc = f"pattern: {args.ckpt_pattern}"
        else:
            ckpts = find_checkpoints_in_dir(args.ckpt_dir)
            source_desc = f"dir: {args.ckpt_dir}"

        if not ckpts:
            print(f"ERROR: no checkpoint.pt files found via {source_desc}", file=sys.stderr)
            sys.exit(1)
        print(f"[run_acquisition] Found {len(ckpts)} checkpoint(s) via {source_desc}:")
        for c in ckpts:
            print(f"  {c}")

        scorer = BasinScorer(
            checkpoint_paths=[str(c) for c in ckpts],
            device=args.device,
        )

        if args.acquisition == "ensemble":
            acq = EnsembleDisagreement(scorer=scorer, stride=args.stride)

        else:  # representativeness
            acq = RepresentativenessAcquisition(
                scorer=scorer,
                alpha=args.alpha,
                diversity_pass=args.diversity_pass,
                stride=args.stride,
            )

    # ------------------------------------------------------------------
    # Score and promote
    # ------------------------------------------------------------------
    scores = acq.score(
        pool_basin_ids=split.sample_ids,
        seed_basin_ids=split.seed_ids,
        eval_basin_ids=split.eval_ids,
    )

    selected = split.promote_top_k(scores, k=k)
    print(f"[run_acquisition] Promoted {len(selected)} basins: {selected}")
    print(f"[run_acquisition] Updated split: {split.summary()}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    split.save(args.out_csv)
    print(f"[run_acquisition] Saved → {args.out_csv}")

    if len(split.sample_ids) == 0:
        print("[run_acquisition] Sample pool now empty — this is the final round.")
        sys.exit(2)


if __name__ == "__main__":
    main()
