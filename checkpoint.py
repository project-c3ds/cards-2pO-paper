"""SQLite-based checkpoint store for benchmark results."""

import sqlite3
import json
import threading
from typing import Dict, Any, List, Set, Optional


class CheckpointStore:
    """Thread-safe SQLite checkpoint store for benchmark results.

    Uses a connection-per-thread pattern to handle SQLite's
    limitation of one writer at a time, with WAL mode for
    concurrent reads.
    """

    def __init__(self, db_path: str = "data/results.db"):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self):
        """Create the results table if it doesn't exist."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                model TEXT NOT NULL,
                row_idx INTEGER NOT NULL,
                text TEXT NOT NULL,
                response TEXT NOT NULL,
                usage TEXT NOT NULL,
                PRIMARY KEY (model, row_idx)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS parsed_results (
                model TEXT NOT NULL,
                row_idx INTEGER NOT NULL,
                classification_dict TEXT NOT NULL,
                PRIMARY KEY (model, row_idx)
            )
        """)
        conn.commit()

    def save_result(self, model: str, row_idx: int, text: str, response: str, usage: dict):
        """Save a single result row. Thread-safe via connection-per-thread."""
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO results (model, row_idx, text, response, usage) VALUES (?, ?, ?, ?, ?)",
            (model, row_idx, text, response, json.dumps(usage))
        )
        conn.commit()

    def get_completed_indices(self, model: str) -> Set[int]:
        """Get set of row indices already completed for a model."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT row_idx FROM results WHERE model = ?", (model,)
        )
        return {row[0] for row in cursor.fetchall()}

    def get_results(self, model: str) -> List[Dict[str, Any]]:
        """Get all saved results for a model."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT row_idx, text, response, usage FROM results WHERE model = ? ORDER BY row_idx",
            (model,)
        )
        return [
            {
                "model": model,
                "text": row[1],
                "response": row[2],
                "usage": json.loads(row[3]),
                "row_idx": row[0]
            }
            for row in cursor.fetchall()
        ]

    def save_parsed_result(self, model: str, row_idx: int, classification_dict: dict):
        """Save a parsed classification result. Thread-safe via connection-per-thread."""
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO parsed_results (model, row_idx, classification_dict) VALUES (?, ?, ?)",
            (model, row_idx, json.dumps(classification_dict))
        )
        conn.commit()

    def get_parsed_results(self, model: str) -> Dict[int, dict]:
        """Get all parsed results for a model, keyed by row_idx."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT row_idx, classification_dict FROM parsed_results WHERE model = ?",
            (model,)
        )
        return {row[0]: json.loads(row[1]) for row in cursor.fetchall()}

    def close(self):
        """Close the thread-local connection if it exists."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
