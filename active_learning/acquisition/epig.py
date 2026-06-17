"""
EPIG (Expected Predictive Information Gain) acquisition.

Paper: Bickfordsmith et al., "Prediction-Oriented Bayesian Active Learning"
       arXiv:2304.08151

EPIG scores a pool basin x by how much labeling it is expected to reduce
predictive uncertainty at the eval basins (the "target distribution" p*(x*)
in the paper, approximated here by the held-out eval set).

Under the Gaussian approximation (paper Eq. E.2), the mutual information
between scalar predictions y at pool basin x and y* at eval basin x* is:

    I(y ; y* | x, x*) = -½ log(1 − ρ²)

where ρ is the Pearson correlation of the K-vector of ensemble predictions
across the K models for those two specific (basin, window, horizon) slots.

We approximate E_{x* ~ p*(x*)} by averaging over all eval basins and all
their (window, horizon) slots, and over all pool-basin slots in the same way:

    EPIG(b) ≈ mean_{b*∈eval} mean_{(w,h), (w*,h*)}  I(pool[b,w,h] ; eval[b*,w*,h*])

Each (w, h) slot produces a K-vector of ensemble predictions — one number
per model — giving a K-sample estimate of ρ.

Why average MI per pair rather than correlate mean predictions
--------------------------------------------------------------
Averaging predictions first (mean over w,h → one scalar per model → one ρ)
loses temporal structure.  If a pool basin's ensemble disagreement is high
in summer and near-zero in winter, but an eval basin shares that same
seasonal pattern, the mean-first approach dilutes both signals and
underestimates the correlation.  Computing MI per (w,h)×(w*,h*) pair
preserves that structure: the summer slots will have high ρ and contribute
large MI; the winter slots will contribute near-zero MI.  Jensen's
inequality means these two quantities are genuinely different:
E[f(ρ)] ≠ f(E[ρ]) for nonlinear f = -½ log(1-ρ²).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import AcquisitionFunction
from ..basin_scorer import BasinScorer

_EPS = 1e-8


def _prep(preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten (K, W, fh) → (K, T), center across ensemble, L2-normalise columns.

    Returns
    -------
    normed : (K, T)  — each column is zero-mean across K and has unit L2 norm
    norms  : (T,)    — pre-normalisation L2 norms; near-zero → degenerate slot
    """
    K, W, fh = preds.shape
    flat = preds.reshape(K, W * fh)                        # (K, T)
    centered = flat - flat.mean(axis=0, keepdims=True)     # subtract ensemble mean per slot
    norms = np.linalg.norm(centered, axis=0)               # (T,)  one norm per slot
    normed = centered / (norms + _EPS)                     # (K, T)  unit columns
    return normed, norms


def _mean_pair_mi(
    Pb_n: np.ndarray,
    norms_b: np.ndarray,
    Pe_n: np.ndarray,
    norms_e: np.ndarray,
) -> float:
    """
    Mean Gaussian MI between every slot of pool basin and every slot of eval basin.

    Parameters
    ----------
    Pb_n, Pe_n   : (K, T_b) and (K, T_e) — normalised ensemble predictions
    norms_b/e    : (T_b,) / (T_e,)        — raw norms before normalisation

    Returns
    -------
    Scalar mean MI averaged over all T_b × T_e valid slot pairs.
    """
    # ρ[t, t'] = Pearson correlation between pool slot t and eval slot t'.
    # Because both matrices have unit-norm, zero-mean columns, their dot
    # product is exactly the Pearson correlation coefficient.
    rho = Pb_n.T @ Pe_n                                    # (T_b, T_e)

    # Clamp strictly inside (-1, 1) so log is defined.
    rho = np.clip(rho, -(1.0 - _EPS), 1.0 - _EPS)

    # Paper Eq. E.2: I(y; y*) = -½ log(1 − ρ²).
    # Always ≥ 0 because ρ² ≤ 1.
    mi = -0.5 * np.log(1.0 - rho ** 2)                    # (T_b, T_e)

    # Exclude degenerate slots (zero ensemble variance → ρ is 0 after
    # normalisation, so MI = 0 anyway, but we exclude them from the average
    # denominator so they don't dilute scores for basins with fewer windows).
    valid = (norms_b > _EPS)[:, None] & (norms_e > _EPS)[None, :]  # (T_b, T_e) bool
    n_valid = int(valid.sum())
    if n_valid == 0:
        return 0.0
    return float(mi[valid].sum() / n_valid)


class EPIGAcquisition(AcquisitionFunction):
    """
    Score pool basins by Expected Predictive Information Gain (EPIG).

    Uses the eval set as a finite-sample approximation of the target
    distribution p*(x*).  A high score means: labeling this pool basin
    is expected to reduce predictive uncertainty at the eval basins.

    Args:
        scorer: BasinScorer holding the ensemble checkpoints.
        split_start / split_end: Temporal inference window.
        stride / batch_size: Efficiency parameters for BasinScorer.
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

    def score(
        self,
        pool_basin_ids: List[str],
        seed_basin_ids: Optional[List[str]] = None,
        eval_basin_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Return {basin_id: epig_score} for every basin in pool_basin_ids.

        seed_basin_ids is accepted for interface compatibility but not used:
        EPIG naturally deprioritises basins the ensemble is already confident
        about (those near the seed set will have low ensemble variance).
        """
        eval_basin_ids = eval_basin_ids or []

        all_ids = list(set(pool_basin_ids) | set(eval_basin_ids))
        predictions = self.scorer.predict(
            basin_ids=all_ids,
            split_start=self.split_start,
            split_end=self.split_end,
            stride=self.stride,
            batch_size=self.batch_size,
        )

        def _get(bid: str) -> Optional[np.ndarray]:
            return predictions.get(str(bid).zfill(8), predictions.get(bid))

        # Pre-process all eval basins once.
        eval_prepped: List[Tuple[np.ndarray, np.ndarray]] = []
        for eid in eval_basin_ids:
            p = _get(eid)
            if p is not None and p.shape[1] > 0:
                eval_prepped.append(_prep(p))

        scores: Dict[str, float] = {}
        for bid in pool_basin_ids:
            p = _get(bid)
            if p is None or p.shape[1] == 0 or not eval_prepped:
                scores[bid] = 0.0
                continue

            Pb_n, norms_b = _prep(p)

            total = 0.0
            for Pe_n, norms_e in eval_prepped:
                total += _mean_pair_mi(Pb_n, norms_b, Pe_n, norms_e)
            scores[bid] = total / len(eval_prepped)

        return scores
