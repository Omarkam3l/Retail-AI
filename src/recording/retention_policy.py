"""Age and disk-quota based file retention with automatic cleanup."""
import os
import time
import logging
from typing import List

logger = logging.getLogger("RetentionPolicy")


class RetentionPolicy:
    """Manages file retention based on age and disk quota."""

    def __init__(self, max_age_hours: float = 72.0,
                 max_size_gb: float = 10.0) -> None:
        self._max_age_seconds = max_age_hours * 3600
        self._max_size_bytes = max_size_gb * 1024 * 1024 * 1024

    def cleanup(self, directory: str) -> int:
        """Removes files exceeding age or quota limits. Returns count of deleted files."""
        if not os.path.isdir(directory):
            return 0

        files = []
        for fname in os.listdir(directory):
            fpath = os.path.join(directory, fname)
            if os.path.isfile(fpath):
                stat = os.stat(fpath)
                files.append((fpath, stat.st_mtime, stat.st_size))

        # Sort oldest first
        files.sort(key=lambda x: x[1])
        now = time.time()
        deleted = 0

        # Delete files exceeding age
        for fpath, mtime, size in files:
            if now - mtime > self._max_age_seconds:
                os.remove(fpath)
                deleted += 1
                logger.info(f"Retention: deleted aged file {fpath}")

        # Recalculate total size and delete oldest if over quota
        remaining = [(f, m, s) for f, m, s in files if os.path.exists(f)]
        total_size = sum(s for _, _, s in remaining)

        for fpath, mtime, size in remaining:
            if total_size <= self._max_size_bytes:
                break
            os.remove(fpath)
            total_size -= size
            deleted += 1
            logger.info(f"Retention: deleted for quota {fpath}")

        return deleted

    def get_stats(self, directory: str) -> dict:
        """Returns retention statistics for a directory."""
        if not os.path.isdir(directory):
            return {"file_count": 0, "total_size_mb": 0.0}

        total_size = 0
        count = 0
        for fname in os.listdir(directory):
            fpath = os.path.join(directory, fname)
            if os.path.isfile(fpath):
                total_size += os.path.getsize(fpath)
                count += 1

        return {
            "file_count": count,
            "total_size_mb": total_size / (1024 * 1024)
        }
