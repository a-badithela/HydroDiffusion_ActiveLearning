from __future__ import annotations

import random
from typing import Dict, List

from .base import AcquisitionFunction


class RandomAcquisition(AcquisitionFunction):
    """
    Uniform random acquisition — the baseline every other strategy must beat.

    Assigns each pool basin a random score drawn from U[0, 1].
    A fixed seed can be supplied for reproducibility.
    """

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def score(self, pool_basin_ids: List[str], **kwargs) -> Dict[str, float]:
        return {b: self._rng.random() for b in pool_basin_ids}
