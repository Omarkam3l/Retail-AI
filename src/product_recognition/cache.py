"""LRU cache storing computed embedding vectors for tracked objects."""
import collections
import threading
from typing import Optional, Dict, Tuple
import numpy as np


class EmbeddingCache:
    """Thread-safe LRU cache for visual embedding outputs."""

    def __init__(self, maxsize: int = 1000) -> None:
        self._maxsize = maxsize
        # Maps key (track_id or hash) -> value (embedding)
        self._cache: collections.OrderedDict = collections.OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[np.ndarray]:
        with self._lock:
            if key in self._cache:
                self._hits += 1
                # Move to end to represent most recently used
                val = self._cache.pop(key)
                self._cache[key] = val
                return val
            self._misses += 1
            return None

    def put(self, key: str, value: np.ndarray) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
            elif len(self._cache) >= self._maxsize:
                # Evict oldest (FIFO/LRU)
                self._cache.popitem(last=False)
            self._cache[key] = value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> Tuple[int, int]:
        with self._lock:
            return self._hits, self._misses
