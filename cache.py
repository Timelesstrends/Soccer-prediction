"""
Local disk cache for API responses.

Each response is stored as a JSON file under .cache/<source>/<key>.json
alongside a metadata sidecar that records when the entry was written.
The cache is intentionally simple — no eviction policy beyond TTL.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from config import CACHE, CacheConfig

logger = logging.getLogger(__name__)


class DiskCache:
    """
    Thread-safe (via atomic file writes) JSON cache with per-source TTLs.

    Usage:
        cache = DiskCache()
        data = cache.get("fixtures", "epl_2024-03-15")
        if data is None:
            data = fetch_from_api(...)
            cache.set("fixtures", "epl_2024-03-15", data)
    """

    def __init__(self, config: CacheConfig = CACHE) -> None:
        self._config = config
        self._root = config.cache_dir
        self._root.mkdir(parents=True, exist_ok=True)
        logger.info("Cache initialised at %s", self._root.resolve())

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, source: str, key: str) -> Optional[Any]:
        """
        Return cached value for (source, key) if it exists and is still fresh.
        Returns None on a cache miss or expired entry.
        """
        data_path, meta_path = self._paths(source, key)

        if not data_path.exists() or not meta_path.exists():
            logger.debug("Cache miss: %s/%s", source, key)
            return None

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            age = time.time() - meta["written_at"]
            ttl = self._config.ttl.get(source, 3_600)

            if age > ttl:
                logger.debug(
                    "Cache expired: %s/%s (age=%.0fs ttl=%ds)", source, key, age, ttl
                )
                return None

            data = json.loads(data_path.read_text(encoding="utf-8"))
            logger.debug("Cache hit: %s/%s (age=%.0fs)", source, key, age)
            return data

        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.warning("Cache read error for %s/%s: %s", source, key, exc)
            return None

    def set(self, source: str, key: str, value: Any) -> None:
        """
        Persist value under (source, key).
        Writes data and metadata atomically using a temp-file + rename pattern.
        """
        data_path, meta_path = self._paths(source, key)
        data_path.parent.mkdir(parents=True, exist_ok=True)

        meta = {"written_at": time.time(), "source": source, "key": key}

        try:
            self._atomic_write(data_path, json.dumps(value, default=str))
            self._atomic_write(meta_path, json.dumps(meta))
            logger.debug("Cache set: %s/%s", source, key)
        except OSError as exc:
            logger.error("Cache write error for %s/%s: %s", source, key, exc)

    def invalidate(self, source: str, key: str) -> None:
        """Manually expire a single cache entry."""
        data_path, meta_path = self._paths(source, key)
        for path in (data_path, meta_path):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Cache invalidation error: %s", exc)
        logger.info("Cache invalidated: %s/%s", source, key)

    def clear_source(self, source: str) -> None:
        """Remove all cached entries for a given source."""
        source_dir = self._root / source
        if source_dir.exists():
            for f in source_dir.glob("*.json"):
                f.unlink(missing_ok=True)
            logger.info("Cache cleared for source: %s", source)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _paths(self, source: str, key: str) -> tuple[Path, Path]:
        safe_key = key.replace("/", "_").replace(":", "-")
        base = self._root / source / safe_key
        return base.with_suffix(".json"), base.with_suffix(".meta.json")

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        """Write to a temp file then rename — prevents partial writes."""
        tmp = path.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)
