"""
Manages the basin split CSV that drives which basins are in 'seed' (labeled),
'sample' (query pool), or 'eval' (held-out) across active learning rounds.

CSV format:
    gauge_id,split
    01022500,eval
    01055000,seed
    01047000,sample
    ...
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import pandas as pd


class ActiveLearningSplit:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._df = pd.read_csv(csv_path, dtype={"gauge_id": str})
        self._df["gauge_id"] = self._df["gauge_id"].str.zfill(8)

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------

    def _ids(self, label: str) -> List[str]:
        return self._df.loc[self._df["split"] == label, "gauge_id"].tolist()

    @property
    def seed_ids(self) -> List[str]:
        return self._ids("seed")

    @property
    def sample_ids(self) -> List[str]:
        return self._ids("sample")

    @property
    def eval_ids(self) -> List[str]:
        return self._ids("eval")

    def summary(self) -> str:
        return (
            f"seed={len(self.seed_ids)}, "
            f"sample={len(self.sample_ids)}, "
            f"eval={len(self.eval_ids)}"
        )

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def promote(self, basin_ids: List[str]) -> None:
        """Move basins from 'sample' → 'seed'. Silently skips non-sample basins."""
        ids = set(b.zfill(8) for b in basin_ids)
        mask = (self._df["gauge_id"].isin(ids)) & (self._df["split"] == "sample")
        if mask.sum() < len(ids):
            actual = set(self._df.loc[mask, "gauge_id"])
            missing = ids - actual
            print(f"[split_manager] Warning: {len(missing)} basin(s) not in sample pool: {missing}")
        self._df.loc[mask, "split"] = "seed"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | None = None) -> str:
        out = path or self.csv_path
        self._df.to_csv(out, index=False)
        return out

    @staticmethod
    def iteration_path(base_csv: str, round_n: int) -> str:
        """
        Returns a path like .../camels_split_geo_100_100_round1.csv
        so each AL round is saved separately and auditable.
        """
        p = Path(base_csv)
        return str(p.parent / f"{p.stem}_round{round_n}{p.suffix}")

    # ------------------------------------------------------------------
    # Scoring helper: promote top-k basins by score dict
    # ------------------------------------------------------------------

    def promote_top_k(self, scores: Dict[str, float], k: int) -> List[str]:
        """
        Given {basin_id: score}, promote the k highest-scoring basins
        from sample → seed. Returns the list of promoted basin IDs.
        """
        pool = set(self.sample_ids)
        ranked = sorted((b for b in scores if b in pool), key=lambda b: scores[b], reverse=True)
        selected = ranked[:k]
        self.promote(selected)
        return selected
