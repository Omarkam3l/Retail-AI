"""Versioned schema migrations for SQLite database."""
import logging
from src.database.connection import DatabaseConnection

logger = logging.getLogger("DatabaseMigrations")

MIGRATIONS = [
    # Version 1: Initial schema
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY
    );
    
    CREATE TABLE IF NOT EXISTS alerts (
        id TEXT PRIMARY KEY,
        track_id INTEGER NOT NULL,
        camera_id TEXT NOT NULL,
        timestamp_ms REAL NOT NULL,
        level TEXT NOT NULL,
        event_type TEXT NOT NULL,
        clip_path TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    );
    
    CREATE TABLE IF NOT EXISTS cameras (
        camera_id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        status TEXT DEFAULT 'inactive',
        fps REAL DEFAULT 0.0,
        resolution TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    );
    
    CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY,
        camera_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        track_id INTEGER NOT NULL,
        timestamp_ms REAL NOT NULL,
        details TEXT DEFAULT '',
        created_at TEXT DEFAULT (datetime('now'))
    );
    
    CREATE TABLE IF NOT EXISTS system_logs (
        id TEXT PRIMARY KEY,
        level TEXT NOT NULL,
        module TEXT NOT NULL,
        message TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now'))
    );
    
    CREATE TABLE IF NOT EXISTS benchmark_runs (
        id TEXT PRIMARY KEY,
        model_version TEXT NOT NULL,
        dataset TEXT NOT NULL,
        detection_f1 REAL DEFAULT 0.0,
        tracking_mota REAL DEFAULT 0.0,
        execution_time_s REAL DEFAULT 0.0,
        created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_alerts_camera ON alerts(camera_id);
    CREATE INDEX IF NOT EXISTS idx_alerts_level ON alerts(level);
    CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id);
    CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
    """
]


def get_current_version(db: DatabaseConnection) -> int:
    """Returns the current schema version."""
    try:
        result = db.fetchone("SELECT MAX(version) as v FROM schema_version")
        return result["v"] if result and result["v"] is not None else 0
    except Exception:
        return 0


def run_migrations(db: DatabaseConnection) -> int:
    """Runs all pending migrations and returns the final version."""
    current = get_current_version(db)

    for i, migration in enumerate(MIGRATIONS):
        version = i + 1
        if version <= current:
            continue

        logger.info(f"Running migration v{version}...")
        conn = db.get_connection()
        conn.executescript(migration)
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
        conn.commit()
        logger.info(f"Migration v{version} applied successfully.")

    final = get_current_version(db)
    logger.info(f"Database at schema version {final}.")
    return final
