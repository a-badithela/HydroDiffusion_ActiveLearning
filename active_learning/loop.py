"""
Active learning loop orchestrator.

Each round:
  1. Train: submit SLURM jobs (or run directly) with the current split CSV.
  2. Score: run the acquisition function over the 'sample' basin pool.
  3. Query: promote top-k basins from sample → seed.
  4. Save: write the updated split CSV for the next round.

Usage example:
    from active_learning import ActiveLearningLoop, ActiveLearningSplit
    from active_learning.acquisition import RandomAcquisition

    loop = ActiveLearningLoop(
        split_csv="path/to/camels_split_geo_100_100.csv",
        acquisition_fn=RandomAcquisition(seed=42),
        k_per_round=10,
        run_dir="runs/active_learning",
    )
    loop.run(n_rounds=5)
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from .acquisition.base import AcquisitionFunction
from .split_manager import ActiveLearningSplit


SLURM_SCRIPT = Path(__file__).resolve().parents[1] / "slurm_al_train.sh"


class ActiveLearningLoop:
    def __init__(
        self,
        split_csv: str,
        acquisition_fn: AcquisitionFunction,
        k_per_round: int,
        run_dir: str,
        n_ensemble_seeds: int = 5,
        model: str = "encdec_lstm",
    ):
        self.split = ActiveLearningSplit(split_csv)
        self.base_csv = split_csv
        self.acq_fn = acquisition_fn
        self.k = k_per_round
        self.run_dir = Path(run_dir)
        self.n_seeds = n_ensemble_seeds
        self.model = model

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, n_rounds: int, dry_run: bool = False) -> None:
        """
        Run the AL loop for n_rounds iterations.
        Set dry_run=True to skip actual training (useful for testing pipeline).
        """
        for r in range(n_rounds):
            print(f"\n{'='*60}")
            print(f"Active Learning Round {r}  |  {self.split.summary()}")
            print(f"{'='*60}")

            current_csv = ActiveLearningSplit.iteration_path(self.base_csv, r)
            self.split.save(current_csv)
            print(f"[loop] Split saved → {current_csv}")

            if dry_run:
                print(f"[loop] dry_run=True — skipping training")
            else:
                self._train_round(r, current_csv)

            print(f"[loop] Scoring {len(self.split.sample_ids)} sample basins …")
            scores = self.acq_fn.score(
                pool_basin_ids=self.split.sample_ids,
                seed_basin_ids=self.split.seed_ids,
                eval_basin_ids=self.split.eval_ids,
                run_dir=str(self.run_dir),
                round_n=r,
            )

            selected = self.split.promote_top_k(scores, k=self.k)
            print(f"[loop] Promoted {len(selected)} basins: {selected}")

        # Save final state
        final_csv = ActiveLearningSplit.iteration_path(self.base_csv, n_rounds)
        self.split.save(final_csv)
        print(f"\n[loop] Done. Final split → {final_csv}  |  {self.split.summary()}")

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _train_round(self, round_n: int, split_csv: str) -> None:
        """
        Submit n_ensemble_seeds SLURM jobs for the current round.
        Each job trains on 'seed' basins defined by split_csv.
        Blocks until all jobs finish (via --wait) only in interactive mode;
        in typical use you'd poll or chain with --dependency.
        """
        log_dir = self.run_dir / f"round{round_n}" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        first_seed = 3407

        for i in range(self.n_seeds):
            seed = first_seed + i
            run_dir = self.run_dir / f"round{round_n}" / f"seed{seed}"
            log_file = log_dir / f"{self.model}_round{round_n}_seed{seed}_%j.txt"
            cmd = [
                "sbatch",
                (f"--export=ALL,SEED={seed},MODEL={self.model},"
                 f"SPLIT_CSV={split_csv},RUN_DIR={run_dir}"),
                f"--output={log_file}",
                str(SLURM_SCRIPT),
            ]
            print(f"[loop] {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
