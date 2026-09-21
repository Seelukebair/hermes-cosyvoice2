"""Bounded in-memory cache for CosyVoice reference conditioning.

The cache deliberately holds only runtime objects.  It never serializes model
inputs, writes speaker data, or stores conditioning text alongside telemetry.
"""

from __future__ import annotations

import copy
import hashlib
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ConditioningKey:
    source_revision: str
    model_revision: str
    profile_id: str
    reference_fingerprint: str
    mode: str
    conditioning_text_hash: str


@dataclass(frozen=True)
class CacheLookup:
    value: Any
    hit: bool


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fingerprint_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a content fingerprint; path names and text are never retained."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clone(value: Any) -> Any:
    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    return copy.deepcopy(value)


def _size_bytes(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_size_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_size_bytes(item) for item in value)
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return max(nbytes, 0)
    element_size = getattr(value, "element_size", None)
    numel = getattr(value, "numel", None)
    if callable(element_size) and callable(numel):
        return max(int(element_size()) * int(numel()), 0)
    return 0


class ConditioningCache:
    """Thread-safe LRU cache with entry and byte limits."""

    def __init__(
        self,
        max_entries: int = 8,
        max_bytes: int = 256 * 1024 * 1024,
        *,
        clone_value: Callable[[Any], Any] = _clone,
        size_bytes: Callable[[Any], int] = _size_bytes,
    ) -> None:
        if max_entries < 0 or max_bytes < 0:
            raise ValueError("cache limits must be non-negative")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._clone_value = clone_value
        self._size_bytes = size_bytes
        self._items: OrderedDict[ConditioningKey, tuple[Any, int]] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0 and self.max_bytes > 0

    def get_or_create(self, key: ConditioningKey, factory: Callable[[], Any]) -> CacheLookup:
        with self._lock:
            item = self._items.pop(key, None)
            if item is not None:
                self._items[key] = item
                self._hits += 1
                return CacheLookup(self._clone_value(item[0]), True)
            self._misses += 1

        value = factory()
        if not self.enabled:
            return CacheLookup(value, False)

        size = self._size_bytes(value)
        if size > self.max_bytes:
            return CacheLookup(value, False)

        with self._lock:
            # A second requester may have populated the entry while the factory
            # ran. Reuse that bounded entry and discard this equivalent result.
            item = self._items.pop(key, None)
            if item is not None:
                self._items[key] = item
                self._hits += 1
                return CacheLookup(self._clone_value(item[0]), True)
            self._items[key] = (self._clone_value(value), size)
            self._bytes += size
            self._trim()
        return CacheLookup(value, False)

    def snapshot(self) -> dict[str, int | bool]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "entries": len(self._items),
                "bytes": self._bytes,
                "max_entries": self.max_entries,
                "max_bytes": self.max_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
            }

    def _trim(self) -> None:
        while self._items and (
            len(self._items) > self.max_entries or self._bytes > self.max_bytes
        ):
            _, (_, size) = self._items.popitem(last=False)
            self._bytes -= size
            self._evictions += 1
