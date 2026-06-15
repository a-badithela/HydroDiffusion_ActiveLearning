from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class AcquisitionFunction(ABC):
    """
    Base class for all active learning acquisition functions.

    Subclasses implement `score()`, which assigns a float to each candidate
    basin. Higher score = higher priority to label in the next round.
    `select()` calls `score()` and returns the top-k basin IDs.
    """

    @abstractmethod
    def score(self, pool_basin_ids: List[str], **kwargs) -> Dict[str, float]:
        """Return {basin_id: score} for every basin in pool_basin_ids."""
        ...

    def select(self, pool_basin_ids: List[str], k: int, **kwargs) -> List[str]:
        """Return the k highest-scoring basins."""
        scores = self.score(pool_basin_ids, **kwargs)
        return sorted(scores, key=scores.__getitem__, reverse=True)[:k]
