"""
DRAM Cache Manager
==================
High-bandwidth DRAM caching layer that sits between the NVMe pipeline and
the CM4 compute modules, minimising latency for hot quantum inference data.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cached item with metadata."""

    key: str
    data: bytes
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = field(init=False)

    def __post_init__(self) -> None:
        self.size_bytes = len(self.data)

    def touch(self) -> None:
        self.last_accessed = time.time()
        self.access_count += 1


class DRAMCacheManager:
    """
    LRU / LFU hybrid cache backed by system DRAM.

    The cache uses an LRU eviction strategy with a configurable capacity
    ceiling.  A ``hit_ratio`` property exposes cache efficiency in real time.

    Parameters
    ----------
    capacity_gb : float
        Maximum DRAM allocated to this cache, in gigabytes.
    eviction_policy : str
        ``"lru"`` (least-recently-used) or ``"lfu"`` (least-frequently-used).
    """

    SUPPORTED_POLICIES = ("lru", "lfu")

    def __init__(
        self,
        capacity_gb: float = 8.0,
        eviction_policy: str = "lru",
    ) -> None:
        if eviction_policy not in self.SUPPORTED_POLICIES:
            raise ValueError(
                f"Unknown eviction policy '{eviction_policy}'. "
                f"Choose from {self.SUPPORTED_POLICIES}."
            )
        self.capacity_bytes = int(capacity_gb * 1024 ** 3)
        self.eviction_policy = eviction_policy
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._used_bytes: int = 0
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def put(self, key: str, data: bytes) -> None:
        """Insert or update a cache entry, evicting stale items if needed."""
        if key in self._store:
            self._used_bytes -= self._store[key].size_bytes
            del self._store[key]

        while self._used_bytes + len(data) > self.capacity_bytes and self._store:
            self._evict()

        entry = CacheEntry(key=key, data=data)
        self._store[key] = entry
        self._used_bytes += entry.size_bytes
        logger.debug("Cache PUT key=%s size=%d bytes.", key[:8], entry.size_bytes)

    def get(self, key: str) -> Optional[bytes]:
        """Return cached data or ``None`` on a cache miss."""
        if key not in self._store:
            self._misses += 1
            logger.debug("Cache MISS key=%s.", key[:8])
            return None

        entry = self._store[key]
        entry.touch()

        if self.eviction_policy == "lru":
            # Move to end (most-recently-used)
            self._store.move_to_end(key)

        self._hits += 1
        logger.debug("Cache HIT  key=%s access_count=%d.", key[:8], entry.access_count)
        return entry.data

    def invalidate(self, key: str) -> bool:
        """Remove a specific entry; returns True if the key was present."""
        if key not in self._store:
            return False
        self._used_bytes -= self._store[key].size_bytes
        del self._store[key]
        logger.info("Cache invalidated key=%s.", key[:8])
        return True

    def flush(self) -> None:
        """Clear all cached entries."""
        self._store.clear()
        self._used_bytes = 0
        logger.info("DRAM cache flushed.")

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _evict(self) -> None:
        if not self._store:
            return
        if self.eviction_policy == "lru":
            evicted_key, evicted_entry = self._store.popitem(last=False)
        else:  # lfu
            evicted_key = min(self._store, key=lambda k: self._store[k].access_count)
            evicted_entry = self._store.pop(evicted_key)
        self._used_bytes -= evicted_entry.size_bytes
        logger.debug("Cache EVICT key=%s policy=%s.", evicted_key[:8], self.eviction_policy)

    # ------------------------------------------------------------------
    # Metrics & introspection
    # ------------------------------------------------------------------

    @property
    def hit_ratio(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    @property
    def utilisation_pct(self) -> float:
        return (self._used_bytes / self.capacity_bytes * 100) if self.capacity_bytes else 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "capacity_gb": round(self.capacity_bytes / 1024 ** 3, 2),
            "used_gb": round(self._used_bytes / 1024 ** 3, 4),
            "utilisation_pct": round(self.utilisation_pct, 2),
            "entries": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_ratio": round(self.hit_ratio, 4),
            "eviction_policy": self.eviction_policy,
        }
