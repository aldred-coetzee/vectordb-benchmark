"""PostgreSQL pgvector database client implementation."""

import time
from typing import Any, Dict, Optional

import numpy as np

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    psycopg2 = None

try:
    from pgvector.psycopg2 import register_vector
except ImportError:
    register_vector = None

from .base import BaseVectorDBClient, IndexConfig, SearchConfig, SearchResult


class PGVectorClient(BaseVectorDBClient):
    """PostgreSQL pgvector database client."""

    def __init__(self):
        """Initialize the pgvector client."""
        if psycopg2 is None:
            raise ImportError(
                "psycopg2 package not installed. "
                "Install with: pip install psycopg2-binary"
            )
        if register_vector is None:
            raise ImportError(
                "pgvector package not installed. "
                "Install with: pip install pgvector"
            )

        self._connection: Optional[Any] = None
        self._cursor: Optional[Any] = None
        self._index_configs: Dict[str, IndexConfig] = {}
        self._metrics: Dict[str, str] = {}  # table_name -> "L2" or "cosine"

    @property
    def name(self) -> str:
        """Return the database name."""
        return "pgvector"

    def connect(self, endpoint: str = "localhost:5432", **kwargs) -> None:
        """
        Connect to PostgreSQL with pgvector.

        Args:
            endpoint: PostgreSQL server endpoint (host:port)
            **kwargs: Additional connection parameters (dbname, user, password)
        """
        # Parse endpoint to extract host and port
        endpoint = endpoint.rstrip("/")
        if endpoint.startswith("postgresql://"):
            endpoint = endpoint[13:]
        elif endpoint.startswith("http://"):
            endpoint = endpoint[7:]
        elif endpoint.startswith("https://"):
            endpoint = endpoint[8:]

        if ":" in endpoint:
            host, port_str = endpoint.split(":", 1)
            port = int(port_str)
        else:
            host = endpoint
            port = 5432

        # Get connection parameters from kwargs or use defaults
        dbname = kwargs.get("dbname", "postgres")
        user = kwargs.get("user", "postgres")
        password = kwargs.get("password", "postgres")

        try:
            self._connection = psycopg2.connect(
                host=host,
                port=port,
                dbname=dbname,
                user=user,
                password=password,
            )
            self._connection.autocommit = True
            self._cursor = self._connection.cursor()

            # Ensure pgvector extension is installed (must be before register_vector)
            self._cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Register pgvector types with psycopg2
            register_vector(self._connection)

            print(f"Connected to PostgreSQL pgvector at {host}:{port}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}")

    def get_version(self) -> str:
        """Return pgvector extension version."""
        if self._cursor is None:
            return "unknown"
        try:
            self._cursor.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )
            row = self._cursor.fetchone()
            return row[0] if row else "unknown"
        except Exception:
            return "unknown"

    def get_client_version(self) -> str:
        """Return psycopg2 Python SDK version."""
        try:
            import importlib.metadata
            return importlib.metadata.version("psycopg2-binary")
        except Exception:
            return "unknown"

    def disconnect(self) -> None:
        """Disconnect from PostgreSQL."""
        self._index_configs.clear()
        if self._cursor:
            self._cursor.close()
            self._cursor = None
        if self._connection:
            self._connection.close()
            self._connection = None
        print("Disconnected from PostgreSQL pgvector")

    def create_table(
        self,
        table_name: str,
        dimension: int,
        index_config: IndexConfig,
    ) -> None:
        """
        Create a table with pgvector column and index.

        Args:
            table_name: Name of the table
            dimension: Vector dimensionality
            index_config: Index configuration
        """
        if self._cursor is None:
            raise RuntimeError("Not connected to database")

        # Sanitize table name to prevent SQL injection
        safe_table_name = self._sanitize_identifier(table_name)

        try:
            # Drop existing table if it exists
            self._cursor.execute(f"DROP TABLE IF EXISTS {safe_table_name}")

            # Create table with id and vector columns
            self._cursor.execute(
                f"CREATE TABLE {safe_table_name} ("
                f"id SERIAL PRIMARY KEY, "
                f"vector vector({dimension}))"
            )

            # Determine distance ops
            metric = index_config.params.get("metric", "L2")
            if metric.lower() in ("cosine", "angular"):
                vector_ops = "vector_cosine_ops"
            else:
                vector_ops = "vector_l2_ops"

            # Create index based on type
            if index_config.index_type == "hnsw":
                # Explicitly validate and cast to int for SQL safety
                m = int(index_config.params.get("M", 16))
                ef_construction = int(index_config.params.get("efConstruction", 64))

                # Validate reasonable ranges for HNSW parameters
                if not (1 <= m <= 100):
                    raise ValueError(f"HNSW M parameter must be between 1 and 100, got {m}")
                if not (1 <= ef_construction <= 2000):
                    raise ValueError(f"HNSW efConstruction must be between 1 and 2000, got {ef_construction}")

                self._cursor.execute(
                    f"CREATE INDEX ON {safe_table_name} "
                    f"USING hnsw (vector {vector_ops}) "
                    f"WITH (m={m}, ef_construction={ef_construction})"
                )
            # For "flat" index type, we don't create an index (exact search)

            self._index_configs[table_name] = index_config
            self._metrics[table_name] = metric
            print(f"Created table '{table_name}' with {index_config.index_type} index")
        except Exception as e:
            raise RuntimeError(f"Failed to create table: {e}")

    def drop_table(self, table_name: str) -> None:
        """
        Drop a table if it exists.

        Args:
            table_name: Name of the table to drop
        """
        if self._cursor is None:
            raise RuntimeError("Not connected to database")

        safe_table_name = self._sanitize_identifier(table_name)

        try:
            self._cursor.execute(f"DROP TABLE IF EXISTS {safe_table_name}")
            print(f"Dropped table '{table_name}'")
        except Exception:
            # Table doesn't exist, which is fine
            pass

        self._index_configs.pop(table_name, None)

    def insert(
        self,
        table_name: str,
        ids: np.ndarray,
        vectors: np.ndarray,
    ) -> None:
        """
        Insert vectors into the table.

        Args:
            table_name: Name of the table
            ids: Array of vector IDs
            vectors: Array of vectors
        """
        if self._cursor is None:
            raise RuntimeError("Not connected to database")

        safe_table_name = self._sanitize_identifier(table_name)

        # Convert to list for psycopg2
        data = [(int(ids[i]), vectors[i].tolist()) for i in range(len(ids))]

        try:
            # Use execute_values for efficient batch insertion
            batch_size = 1000
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                execute_values(
                    self._cursor,
                    f"INSERT INTO {safe_table_name} (id, vector) VALUES %s",
                    batch,
                    template="(%s, %s::vector)",
                )
        except Exception as e:
            raise RuntimeError(f"Failed to insert vectors: {e}")

    def search(
        self,
        table_name: str,
        query_vector: np.ndarray,
        k: int,
        search_config: SearchConfig,
    ) -> SearchResult:
        """
        Search for k nearest neighbors.

        Args:
            table_name: Name of the table
            query_vector: Query vector
            k: Number of neighbors
            search_config: Search configuration

        Returns:
            SearchResult with IDs and latency
        """
        if self._cursor is None:
            raise RuntimeError("Not connected to database")

        safe_table_name = self._sanitize_identifier(table_name)

        # Set ef_search for HNSW index if applicable
        if search_config.index_type == "hnsw":
            ef_search = int(search_config.params.get("efSearch", 64))
            # Validate ef_search is within reasonable bounds
            if not (1 <= ef_search <= 2000):
                raise ValueError(f"HNSW efSearch must be between 1 and 2000, got {ef_search}")
            self._cursor.execute("SET hnsw.ef_search = %s", (ef_search,))

        try:
            # Convert query vector to string format for pgvector
            vector_str = "[" + ",".join(str(x) for x in query_vector.tolist()) + "]"

            # Select distance operator based on metric
            metric = self._metrics.get(table_name, "L2")
            if metric.lower() in ("cosine", "angular"):
                dist_op = "<=>"
            else:
                dist_op = "<->"

            start_time = time.perf_counter()

            self._cursor.execute(
                f"SELECT id, vector {dist_op} %s::vector AS distance "
                f"FROM {safe_table_name} "
                f"ORDER BY distance "
                f"LIMIT %s",
                (vector_str, k)
            )

            results = self._cursor.fetchall()

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            if results:
                ids = np.array([row[0] for row in results], dtype=np.int64)
                distances = np.array([row[1] for row in results], dtype=np.float32)
            else:
                ids = np.array([], dtype=np.int64)
                distances = np.array([], dtype=np.float32)

            return SearchResult(ids=ids, distances=distances, latency_ms=latency_ms)

        except Exception as e:
            raise RuntimeError(f"Search failed: {e}")

    def get_stats(self, table_name: str) -> Dict[str, Any]:
        """
        Get table statistics.

        Args:
            table_name: Name of the table

        Returns:
            Dictionary with table stats
        """
        if self._cursor is None:
            return {
                "row_count": 0,
                "table_name": table_name,
                "error": "Not connected",
            }

        safe_table_name = self._sanitize_identifier(table_name)

        try:
            self._cursor.execute(f"SELECT COUNT(*) FROM {safe_table_name}")
            row_count = self._cursor.fetchone()[0]
            return {
                "row_count": row_count,
                "table_name": table_name,
            }
        except Exception as e:
            return {
                "row_count": 0,
                "table_name": table_name,
                "error": str(e),
            }

    def _sanitize_identifier(self, identifier: str) -> str:
        """
        Sanitize a SQL identifier to prevent SQL injection.

        Args:
            identifier: Table or column name

        Returns:
            Quoted identifier safe for SQL
        """
        # Remove any quotes and special characters, keep only alphanumeric and underscore
        clean = "".join(c for c in identifier if c.isalnum() or c == "_")
        return f'"{clean}"'
