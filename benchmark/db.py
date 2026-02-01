"""SQLite storage for benchmark results."""

import json
import re
import socket
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .runner import BenchmarkResults, IngestResult, SearchResult


# Default database path
DEFAULT_DB_PATH = "results/benchmark.db"


class BenchmarkDatabase:
    """SQLite database for storing benchmark results."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """
        Initialize the benchmark database.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema, creating tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Create runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                database TEXT NOT NULL,
                db_version TEXT,
                dataset TEXT NOT NULL,
                vector_count INTEGER NOT NULL,
                dimensions INTEGER NOT NULL,
                cpus REAL,
                memory_gb REAL,
                config_json TEXT,
                benchmark_config_json TEXT,
                hostname TEXT,
                notes TEXT
            )
        """)

        # Create ingest_results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ingest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                index_type TEXT NOT NULL,
                index_config_json TEXT,
                total_time_s REAL NOT NULL,
                throughput_vps REAL NOT NULL,
                peak_memory_gb REAL,
                final_memory_gb REAL,
                batch_size INTEGER,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            )
        """)

        # Create search_results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                index_type TEXT NOT NULL,
                ef_search INTEGER,
                qps REAL NOT NULL,
                p50_ms REAL,
                p95_ms REAL,
                p99_ms REAL,
                recall_at_10 REAL,
                recall_at_100 REAL,
                avg_cpu_pct REAL,
                avg_memory_gb REAL,
                num_queries INTEGER,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            )
        """)

        # Create resource_samples table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS resource_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                phase TEXT NOT NULL,
                cpu_pct REAL,
                memory_gb REAL,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            )
        """)

        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_runs_database
            ON runs (database)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_runs_timestamp
            ON runs (timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ingest_run_id
            ON ingest_results (run_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_search_run_id
            ON search_results (run_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_resource_run_id
            ON resource_samples (run_id)
        """)

        conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a database connection.

        Note: Uses check_same_thread=False to allow access from multiple threads.
        The benchmark runner may access the database from different threads
        (e.g., main thread and monitoring thread), so we need to allow this.
        SQLite itself handles thread safety at the database level.
        """
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def create_run(
        self,
        database: str,
        dataset: str,
        vector_count: int,
        dimensions: int,
        db_version: Optional[str] = None,
        cpus: Optional[float] = None,
        memory_gb: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None,
        benchmark_config: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> int:
        """
        Create a new benchmark run record.

        Args:
            database: Name of the database being benchmarked
            dataset: Name of the dataset used
            vector_count: Number of vectors in the dataset
            dimensions: Dimensionality of vectors
            db_version: Version of the database
            cpus: CPU limit/allocation
            memory_gb: Memory limit in GB
            config: Database configuration dictionary (stored as JSON)
            benchmark_config: Benchmark configuration dictionary (stored as JSON)
            notes: Optional notes about the run

        Returns:
            run_id: The ID of the created run
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()
        hostname = socket.gethostname()
        config_json = json.dumps(config) if config else None
        benchmark_config_json = json.dumps(benchmark_config) if benchmark_config else None

        cursor.execute("""
            INSERT INTO runs (
                timestamp, database, db_version, dataset, vector_count,
                dimensions, cpus, memory_gb, config_json, benchmark_config_json,
                hostname, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, database, db_version, dataset, vector_count,
            dimensions, cpus, memory_gb, config_json, benchmark_config_json,
            hostname, notes
        ))

        conn.commit()
        return cursor.lastrowid

    def save_ingest_result(
        self,
        run_id: int,
        result: IngestResult,
        index_config: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None,
    ) -> int:
        """
        Save an ingest benchmark result.

        Args:
            run_id: The run ID this result belongs to
            result: IngestResult object
            index_config: Index configuration dictionary
            batch_size: Batch size used for ingestion

        Returns:
            id: The ID of the created record
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        index_config_json = json.dumps(index_config) if index_config else None

        cursor.execute("""
            INSERT INTO ingest_results (
                run_id, index_type, index_config_json, total_time_s,
                throughput_vps, peak_memory_gb, final_memory_gb, batch_size
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, result.index_type, index_config_json,
            result.total_time_seconds, result.vectors_per_second,
            result.peak_memory_gb, result.final_memory_gb, batch_size
        ))

        conn.commit()
        return cursor.lastrowid

    def save_search_result(
        self,
        run_id: int,
        result: SearchResult,
        num_queries: Optional[int] = None,
    ) -> int:
        """
        Save a search benchmark result.

        Args:
            run_id: The run ID this result belongs to
            result: SearchResult object
            num_queries: Number of queries executed

        Returns:
            id: The ID of the created record
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Extract efSearch from search_config if present
        # Handles formats like "efSearch=64" or "-" (for flat index)
        ef_search = None
        if result.search_config:
            match = re.search(r'efSearch=(\d+)', result.search_config)
            if match:
                ef_search = int(match.group(1))

        cursor.execute("""
            INSERT INTO search_results (
                run_id, index_type, ef_search, qps, p50_ms, p95_ms, p99_ms,
                recall_at_10, recall_at_100, avg_cpu_pct, avg_memory_gb, num_queries
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, result.index_type, ef_search, result.qps,
            result.latency_p50_ms, result.latency_p95_ms, result.latency_p99_ms,
            result.recall_at_10, result.recall_at_100,
            result.cpu_percent, result.memory_gb, num_queries
        ))

        conn.commit()
        return cursor.lastrowid

    def save_resource_sample(
        self,
        run_id: int,
        phase: str,
        cpu_pct: Optional[float] = None,
        memory_gb: Optional[float] = None,
    ) -> int:
        """
        Save a resource utilization sample.

        Args:
            run_id: The run ID this sample belongs to
            phase: Phase of benchmark (e.g., 'ingest', 'search')
            cpu_pct: CPU utilization percentage
            memory_gb: Memory usage in GB

        Returns:
            id: The ID of the created record
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO resource_samples (
                run_id, timestamp, phase, cpu_pct, memory_gb
            ) VALUES (?, ?, ?, ?, ?)
        """, (run_id, timestamp, phase, cpu_pct, memory_gb))

        conn.commit()
        return cursor.lastrowid

    def save_benchmark_results(
        self,
        results: BenchmarkResults,
        config: Optional[Dict[str, Any]] = None,
        benchmark_config: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None,
        num_queries: Optional[int] = None,
        db_version: Optional[str] = None,
    ) -> int:
        """
        Save complete benchmark results to the database.

        Args:
            results: BenchmarkResults object containing all results
            config: Database configuration dictionary
            benchmark_config: Benchmark configuration dictionary
            batch_size: Batch size used for ingestion
            num_queries: Number of queries executed
            db_version: Database version string

        Returns:
            run_id: The ID of the created run
        """
        dataset_info = results.dataset_info

        # Create the run record
        run_id = self.create_run(
            database=results.database_name,
            dataset=dataset_info.get("name", "unknown"),
            vector_count=dataset_info.get("num_base_vectors", 0),
            dimensions=dataset_info.get("dimensions", 0),
            db_version=db_version,
            cpus=results.docker_cpu_limit,
            memory_gb=results.docker_memory_limit_gb,
            config=config,
            benchmark_config=benchmark_config,
        )

        # Save ingest results
        for ingest_result in results.ingest_results:
            # Parse index config from description
            index_config = {"description": ingest_result.index_config}
            self.save_ingest_result(
                run_id=run_id,
                result=ingest_result,
                index_config=index_config,
                batch_size=batch_size,
            )

        # Save search results
        for search_result in results.search_results:
            self.save_search_result(
                run_id=run_id,
                result=search_result,
                num_queries=num_queries,
            )

        return run_id

    def get_runs(
        self,
        database: Optional[str] = None,
        dataset: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get benchmark runs, optionally filtered.

        Args:
            database: Filter by database name
            dataset: Filter by dataset name
            limit: Maximum number of results to return

        Returns:
            List of run records as dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM runs WHERE 1=1"
        params = []

        if database:
            query += " AND database = ?"
            params.append(database)

        if dataset:
            query += " AND dataset = ?"
            params.append(dataset)

        query += " ORDER BY timestamp DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_run_details(self, run_id: int) -> Optional[Dict[str, Any]]:
        """
        Get complete details for a specific run.

        Args:
            run_id: The run ID to retrieve

        Returns:
            Dictionary containing run info, ingest results, and search results
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get run info
        cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        run = cursor.fetchone()

        if not run:
            return None

        result = dict(run)

        # Get ingest results
        cursor.execute(
            "SELECT * FROM ingest_results WHERE run_id = ?",
            (run_id,)
        )
        result["ingest_results"] = [dict(row) for row in cursor.fetchall()]

        # Get search results
        cursor.execute(
            "SELECT * FROM search_results WHERE run_id = ?",
            (run_id,)
        )
        result["search_results"] = [dict(row) for row in cursor.fetchall()]

        # Get resource samples
        cursor.execute(
            "SELECT * FROM resource_samples WHERE run_id = ?",
            (run_id,)
        )
        result["resource_samples"] = [dict(row) for row in cursor.fetchall()]

        return result

    def get_comparison(
        self,
        database_names: List[str],
        dataset: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get comparison data for multiple databases.

        Args:
            database_names: List of database names to compare
            dataset: Optional filter by dataset

        Returns:
            Dictionary containing comparison data
        """
        comparison = {}

        for db_name in database_names:
            runs = self.get_runs(database=db_name, dataset=dataset, limit=1)
            if runs:
                run = runs[0]
                run_details = self.get_run_details(run["run_id"])
                comparison[db_name] = run_details

        return comparison


def get_database(db_path: str = DEFAULT_DB_PATH) -> BenchmarkDatabase:
    """
    Get a BenchmarkDatabase instance.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        BenchmarkDatabase instance
    """
    return BenchmarkDatabase(db_path)
