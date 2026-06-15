"""
Representativeness acquisition: prefer pool basins whose predicted streamflow
distribution is similar to the eval basins and dissimilar to the seed basins.

score[b] = sim(b, eval_set) - alpha * sim(b, seed_set)

where sim(b, S) = mean cosine similarity between b's embedding and all basins in S.

Embedding: ensemble-mean predictions are summarised as [mean, std] over all
windows and forecast steps → a 2-D vector per basin.

An optional diversity pass (greedy farthest-point) can be applied on top to
avoid selecting near-duplicates from the pool when k_per_round is large.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .base import AcquisitionFunction
from ..basin_scorer import BasinScorer


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / denom)


def _mean_sim_to_set(emb: np.ndarray, set_embs: np.ndarray) -> float:
    if set_embs.shape[0] == 0:
        return 0.0
    return float(np.mean([_cosine_sim(emb, e) for e in set_embs]))


def _embed(preds: np.ndarray) -> np.ndarray:
    """
    Compress (n_models, n_windows, fh) predictions to a 2-D vector.
    Takes ensemble mean first, then [global_mean, global_std].
    """
    flat = preds.mean(axis=0).ravel()   # ensemble mean → flatten time×horizon
    return np.array([flat.mean(), flat.std()], dtype=np.float32)


class RepresentativenessAcquisition(AcquisitionFunction):
    """
    Score pool basins by how representative they are of the eval distribution
    relative to what the seed set already covers.

    Args:
        scorer: BasinScorer holding the ensemble checkpoints.
        alpha: Weight on seed-similarity penalty (default 1.0).
            alpha=0 → pure "look like eval" (ignores seed coverage).
            alpha=1 → balanced.  alpha>1 → stronger push toward diversity.
        diversity_pass: If True, after scoring apply greedy farthest-point
            selection in embedding space so the final batch of k basins is
            also internally diverse.
        split_start / split_end: Temporal window for inference.
        stride / batch_size: Efficiency parameters for BasinScorer.
    """

    def __init__(
        self,
        scorer: BasinScorer,
        alpha: float = 1.0,
        diversity_pass: bool = False,
        split_start: str = "1990-10-01",
        split_end: str = "1995-09-30",
        stride: int = 90,
        batch_size: int = 256,
    ):
        self.scorer = scorer
        self.alpha = alpha
        self.diversity_pass = diversity_pass
        self.split_start = split_start
        self.split_end = split_end
        self.stride = stride
        self.batch_size = batch_size

        # Cache the last predictions so select() doesn't re-run inference.
        self._last_predictions: Optional[Dict[str, np.ndarray]] = None
        self._last_embeddings: Optional[Dict[str, np.ndarray]] = None

    def _run_inference(self, basin_ids: List[str]) -> Dict[str, np.ndarray]:
        predictions = self.scorer.predict(
            basin_ids=basin_ids,
            split_start=self.split_start,
            split_end=self.split_end,
            stride=self.stride,
            batch_size=self.batch_size,
        )
        self._last_predictions = predictions
        self._last_embeddings = {
            bid: _embed(preds)
            for bid, preds in predictions.items()
            if preds.shape[1] > 0
        }
        return predictions

    def score(
        self,
        pool_basin_ids: List[str],
        seed_basin_ids: Optional[List[str]] = None,
        eval_basin_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        seed_basin_ids = seed_basin_ids or []
        eval_basin_ids = eval_basin_ids or []

        # Score pool + eval; seed is not scored (it's already labeled).
        all_to_score = list(set(pool_basin_ids) | set(eval_basin_ids))
        self._run_inference(all_to_score)
        embeddings = self._last_embeddings

        eval_embs = np.stack(
            [embeddings[b] for b in eval_basin_ids if b in embeddings], axis=0
        ) if eval_basin_ids else np.zeros((0, 2), dtype=np.float32)

        seed_embs = np.stack(
            [embeddings[b] for b in seed_basin_ids if b in embeddings], axis=0
        ) if seed_basin_ids else np.zeros((0, 2), dtype=np.float32)

        scores: Dict[str, float] = {}
        for bid in pool_basin_ids:
            emb = embeddings.get(bid)
            if emb is None:
                scores[bid] = 0.0
                continue
            scores[bid] = (
                _mean_sim_to_set(emb, eval_embs)
                - self.alpha * _mean_sim_to_set(emb, seed_embs)
            )

        return scores

    def select(
        self,
        pool_basin_ids: List[str],
        k: int,
        seed_basin_ids: Optional[List[str]] = None,
        eval_basin_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:
        scores = self.score(
            pool_basin_ids,
            seed_basin_ids=seed_basin_ids,
            eval_basin_ids=eval_basin_ids,
        )
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)

        if not self.diversity_pass:
            return ranked[:k]

        # Greedy farthest-point selection in embedding space on top of the
        # score-ranked list.  Start with the highest-scoring basin, then pick
        # the basin in the remaining ranked list that is farthest from the
        # already-selected set.
        embeddings = self._last_embeddings or {}

        selected: List[str] = [ranked[0]]
        selected_embs: List[np.ndarray] = [embeddings.get(ranked[0], np.zeros(2))]
        remaining = ranked[1:]

        while len(selected) < k and remaining:
            farthest, max_dist = remaining[0], -1.0
            for bid in remaining:
                emb = embeddings.get(bid, np.zeros(2))
                min_dist = min(np.linalg.norm(emb - s) for s in selected_embs)
                if min_dist > max_dist:
                    max_dist = min_dist
                    farthest = bid
            selected.append(farthest)
            selected_embs.append(embeddings.get(farthest, np.zeros(2)))
            remaining.remove(farthest)

        return selected
