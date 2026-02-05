"""KDB.AI vector database client implementation."""

import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    import kdbai_client as kdbai
except ImportError:
    kdbai = None

from .base import BaseVectorDBClient, IndexConfig, SearchConfig, SearchResult


class KDBAIClient(BaseVectorDBClient):
    """KDB.AI vector database client."""

    def __init__(self):
        """Initialize the KDB.AI client."""
        if kdbai is None:
            raise ImportError(
                "kdbai-client package not installed. "
                "Install with: pip install kdbai-client"
            )

        self._session: Optional[Any] = None
        self._database: Optional[Any] = None
        self._tables: Dict[str, Any] = {}
        self._index_configs: Dict[str, IndexConfig] = {}

    @property
    def name(self) -> str:
        """Return the database name."""
        return "KDB.AI"

    def connect(self, endpoint: str = "http://localhost:8082", **kwargs) -> None:
        """
        Connect to KDB.AI.

        Args:
            endpoint: KDB.AI server endpoint URL
            **kwargs: Additional connection parameters
        """
        try:
            self._session = kdbai.Session(endpoint=endpoint)
            self._database = self._session.database("default")
            print(f"Connected to KDB.AI at {endpoint}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to KDB.AI: {e}")

    def disconnect(self) -> None:
        """Disconnect from KDB.AI."""
        self._tables.clear()
        self._index_configs.clear()
        self._database = None
        if self._session is not None:
            try:
                # Try to close the session properly to avoid socket leaks
                if hasattr(self._session, 'close'):
                    self._session.close()
            except Exception:
                pass  # Ignore errors during cleanup
        self._session = None
        print("Disconnected from KDB.AI")

    def create_table(
        self,
        table_name: str,
        dimension: int,
        index_config: IndexConfig,
    ) -> None:
        """
        Create a table with the specified index configuration.

        Args:
            table_name: Name of the table
            dimension: Vector dimensionality
            index_config: Index configuration
        """
        if self._database is None:
            raise RuntimeError("Not connected to database")

        # Build schema
        schema = [
            {"name": "id", "type": "int64"},
            {"name": "vectors", "type": "float32s"},
        ]

        # Build index configuration
        if index_config.index_type == "flat":
            indexes = [
                {
                    "name": index_config.name,
                    "type": "qFlat",
                    "column": "vectors",
                    "params": {
                        "dims": dimension,
                        "metric": index_config.params.get("metric", "L2"),
                    },
                }
            ]
        elif index_config.index_type == "hnsw":
            indexes = [
                {
                    "name": index_config.name,
                    "type": "qHnsw",
                    "column": "vectors",
                    "params": {
                        "dims": dimension,
                        "M": index_config.params.get("M", 16),
                        "efConstruction": index_config.params.get("efConstruction", 64),
                        "metric": index_config.params.get("metric", "L2"),
                    },
                }
            ]
        else:
            raise ValueError(f"Unsupported index type: {index_config.index_type}")

        try:
            table = self._database.create_table(
                table_name, schema=schema, indexes=indexes
            )
            self._tables[table_name] = table
            self._index_configs[table_name] = index_config
            print(f"Created table '{table_name}' with {index_config.index_type} index")
        except Exception as e:
            raise RuntimeError(f"Failed to create table: {e}")

    def drop_table(self, table_name: str) -> None:
        """
        Drop a table if it exists.

        Args:
            table_name: Name of the table to drop
        """
        if self._database is None:
            raise RuntimeError("Not connected to database")

        try:
            # Try to get and drop the table
            table = self._database.table(table_name)
            table.drop()
            print(f"Dropped table '{table_name}'")
        except (KeyError, ValueError) as e:
            # Table doesn't exist, which is fine
            pass
        except Exception as e:
            # Check if error message indicates table not found
            error_msg = str(e).lower()
            if "not found" in error_msg or "does not exist" in error_msg or "no such" in error_msg:
                pass  # Table doesn't exist, which is fine
            else:
                # Re-raise unexpected errors
                raise RuntimeError(f"Failed to drop table '{table_name}': {e}")

        # Clean up local references
        self._tables.pop(table_name, None)
        self._index_configs.pop(table_name, None)

    def _get_table(self, table_name: str) -> Any:
        """Get a table reference, caching it locally."""
        if table_name not in self._tables:
            if self._database is None:
                raise RuntimeError("Not connected to database")
            self._tables[table_name] = self._database.table(table_name)
        return self._tables[table_name]

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
        table = self._get_table(table_name)

        # Batch inserts to handle large payloads (high-dimensional vectors)
        dims = vectors.shape[1] if len(vectors.shape) > 1 else 128
        max_payload_bytes = 50_000_000  # 50MB conservative
        bytes_per_vector = dims * 4 + 50  # float32 + overhead
        batch_size = max(100, min(len(ids), max_payload_bytes // bytes_per_vector))

        try:
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_vectors = vectors[i:i + batch_size]
                df = pd.DataFrame(
                    {
                        "id": batch_ids.astype(np.int64),
                        "vectors": list(batch_vectors.astype(np.float32)),
                    }
                )
                table.insert(df)
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
        table = self._get_table(table_name)
        index_name = search_config.index_name

        # Build search parameters
        search_vectors = {index_name: [query_vector.tolist()]}

        # Build index params based on index type
        if search_config.index_type == "hnsw":
            index_params = {
                index_name: {"efSearch": search_config.params.get("efSearch", 64)}
            }
        else:
            index_params = {}

        try:
            start_time = time.perf_counter()

            if index_params:
                results = table.search(
                    vectors=search_vectors, n=k, index_params=index_params
                )
            else:
                results = table.search(vectors=search_vectors, n=k)

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            # Extract IDs from results
            if isinstance(results, list) and len(results) > 0:
                # Results is a list of DataFrames, one per query
                result_df = results[0]
                if "id" in result_df.columns:
                    ids = result_df["id"].values.astype(np.int64)
                else:
                    ids = np.array([], dtype=np.int64)

                # Extract distances if available
                distances = None
                if "__nn_distance" in result_df.columns:
                    distances = result_df["__nn_distance"].values.astype(np.float32)
            else:
                ids = np.array([], dtype=np.int64)
                distances = None

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
        table = self._get_table(table_name)

        try:
            # Get row count - this may vary by kdbai-client version
            # Try different methods
            try:
                count = len(table)
            except TypeError:
                try:
                    info = table.info()
                    count = info.get("rowCount", info.get("row_count", 0))
                except Exception:
                    count = 0

            return {
                "row_count": count,
                "table_name": table_name,
            }
        except Exception as e:
            return {
                "row_count": 0,
                "table_name": table_name,
                "error": str(e),
            }
