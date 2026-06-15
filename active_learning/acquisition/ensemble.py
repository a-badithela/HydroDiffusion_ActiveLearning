"""
Ensemble disagreement acquisition: score pool basins by how much the ensemble
of trained models disagrees on their predicted streamflow.

High cross-model variance → high uncertainty → high priority to label next.
"""

from __future__ import annotations

from typing import Dict, List

from .base import AcquisitionFunction
from ..basin_scorer import BasinScorer


class EnsembleDisagreement(AcquisitionFunction):
    """
    score[b] = mean over windows and horizon of var_across_models(predictions[b])

    Requires a BasinScorer wrapping at least 2 checkpoints (one per seed).

    Args:
        scorer: BasinScorer holding the ensemble checkpoints.
        split_start / split_end: Temporal window to run inference over
            (default: validation period 1990–1995).
        stride: Sliding-window stride in days (larger = faster, fewer windows).
        batch_size: DataLoader batch size during inference.
    """

    def __init__(
        self,
        scorer: BasinScorer,
        split_start: str = "1990-10-01",
        split_end: str = "1995-09-30",
        stride: int = 90,
        batch_size: int = 256,
    ):
        self.scorer = scorer
        self.split_start = split_start
        self.split_end = split_end
        self.stride = stride
        self.batch_size = batch_size

    def score(self, pool_basin_ids: List[str], **kwargs) -> Dict[str, float]:
        """
        Return {basin_id: disagreement} for every basin in pool_basin_ids.

        predictions[b] has shape (n_models, n_windows, forecast_horizon).
        Disagreement = mean( var over models ) over all windows and steps.
        """
        predictions = self.scorer.predict(
            basin_ids=pool_basin_ids,
            split_start=self.split_start,
            split_end=self.split_end,
            stride=self.stride,
            batch_size=self.batch_size,
        )

        scores: Dict[str, float] = {}
        for bid in pool_basin_ids:
            zp = str(bid).zfill(8)
            preds = predictions.get(zp, predictions.get(bid))
            if preds is None or preds.shape[1] == 0:
                scores[bid] = 0.0
            else:
                # var over models (axis=0) → (n_windows, fh), then mean over all
                scores[bid] = float(preds.var(axis=0).mean())

        return scores
