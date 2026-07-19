"""Thread-safe SQLite connection manager with WAL mode."""
import sqlite3
import os
import logging
import threading
from typing import Optional

logger = logging.getLogger("DatabaseConnection")


class DatabaseConnection:
    """Thread-safe SQLite connection manager."""

    def __init__(self, db_path: str = "data/retail_ai.db") -> None:
        self._db_path = db_path
        self._local = threading.local()
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        """Returns a thread-local SQLite connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(self._db_path)
            self._local.connection.row_factory = sqlite3.Row
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA foreign_keys=ON")
        return self._local.connection

    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        conn = self.get_connection()
        cursor = conn.execute(query, params)
        conn.commit()
        return cursor

    def executemany(self, query: str, params_list: list) -> None:
        conn = self.get_connection()
        conn.executemany(query, params_list)
        conn.commit()

    def fetchone(self, query: str, params: tuple = ()) -> Optional[dict]:
        cursor = self.get_connection().execute(query, params)
        row = cursor.fetchone()
        return dict(row) if row else None

    def fetchall(self, query: str, params: tuple = ()) -> list:
        cursor = self.get_connection().execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def close(self) -> None:
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
