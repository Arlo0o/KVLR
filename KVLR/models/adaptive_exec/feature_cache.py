from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class FeatureCache:
    """
    Small in-memory cache keyed by layer id. The MVP stores whole tensors and
    applies token/frame selection outside the cache to keep the API simple.
    """

    enabled: bool = True
    _store: dict[Any, Tensor] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get(self, layer_id: Any, token_ids=None, frame_ids=None) -> Tensor | None:
        if not self.enabled or layer_id not in self._store:
            self.misses += 1
            return None
        self.hits += 1
        value = self._store[layer_id]
        if token_ids is not None:
            value = value[:, token_ids]
        if frame_ids is not None:
            value = value[:, frame_ids]
        return value

    def put(self, layer_id: Any, token_ids=None, frame_ids=None, value: Tensor | None = None) -> None:
        if not self.enabled or value is None:
            return
        self._store[layer_id] = value.detach()

    def invalidate(self, frame_ids=None, token_ids=None) -> None:
        self._store.clear()

    def stats(self) -> dict[str, float]:
        total = self.hits + self.misses
        return {
            "cache_hits": float(self.hits),
            "cache_misses": float(self.misses),
            "cache_hit_ratio": float(self.hits / total) if total > 0 else 0.0,
        }
