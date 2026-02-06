"""LanceDB vector database client implementation."""

import time
from typing import Any, Dict, Optional, Set

import numpy as np

try:
    import lancedb
    import pyarrow as pa
except ImportError:
    lancedb = None
    pa = None

from .base import BaseVectorDBClient, IndexConfig, SearchConfig, SearchResult


class LanceDBClient(BaseVectorDBClient):
    """LanceDB vector database client (embedded, no network overhead)."""

    def __init__(self):
        """Initialize the LanceDB client."""
        if lancedb is None:
            raise ImportError(
                "lancedb package not installed. "
                "Install with: pip install lancedb"
            )

        self._db: Optional[Any] = None
        self._tables: Dict[str, Any] = {}
        self._index_configs: Dict[str, IndexConfig] = {}
        self._db_path: str = "./data/lancedb"

    @property
    def name(self) -> str:
        """Return the database name."""
        return "LanceDB"

    @property
    def supported_indexes(self) -> Set[str]:
        """LanceDB only supports IVF-based indexes (IVF_PQ, IVF_HNSW_SQ, IVF_HNSW_PQ).

        Pure HNSW and FLAT indexes are not available. LanceDB will be included
        when IVF-PQ benchmarks are added.
        """
        return set()

    def get_version(self) -> str:
        """Return LanceDB library version."""
        try:
            return lancedb.__version__
        except AttributeError:
            try:
                import importlib.metadata
                return importlib.metadata.version("lancedb")
            except Exception:
                return "unknown"

    def get_client_version(self) -> str:
        """Return LanceDB library version (embedded — same as server)."""
        return self.get_version()

    def connect(self, endpoint: str = "./data/lancedb", **kwargs) -> None:
        """
        Connect to LanceDB (creates/opens a local database).

        Args:
            endpoint: Path to the LanceDB database directory
            **kwargs: Additional connection parameters
        """
        self._db_path = endpoint
        self._db = lancedb.connect(endpoint)
        print(f"LanceDB running embedded at {endpoint}")

    def disconnect(self) -> None:
        """Disconnect from LanceDB (clears references)."""
        self._tables.clear()
        self._index_configs.clear()
        self._db = None
        print("LanceDB disconnected")

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
        if self._db is None:
            raise RuntimeError("Not connected to database")

        # Drop existing table if present
        try:
            self._db.drop_table(table_name)
        except Exception:
            pass  # Table doesn't exist

        # Create empty table with schema
        # LanceDB requires initial data, so we create with a dummy row then delete
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), dimension)),
        ])

        # Create table with schema
        self._tables[table_name] = self._db.create_table(
            table_name,
            schema=schema,
        )
        self._index_configs[table_name] = index_config

        print(f"Created table '{table_name}' with {index_config.index_type} index")

    def drop_table(self, table_name: str) -> None:
        """
        Drop a table if it exists.

        Args:
            table_name: Name of the table to drop
        """
        if self._db is None:
            raise RuntimeError("Not connected to database")

        try:
            self._db.drop_table(table_name)
            print(f"Dropped table '{table_name}'")
        except Exception:
            pass  # Table doesn't exist

        self._tables.pop(table_name, None)
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
        if table_name not in self._tables:
            # Re-open table if needed
            self._tables[table_name] = self._db.open_table(table_name)

        table = self._tables[table_name]

        # Convert to list of dicts for LanceDB
        data = [
            {"id": int(id_val), "vector": vec.tolist()}
            for id_val, vec in zip(ids, vectors)
        ]

        # Add data to table
        table.add(data)

        # Create index after all data is inserted
        # We'll create the index on the last batch (when we have significant data)
        index_config = self._index_configs.get(table_name)
        if index_config and index_config.index_type == "hnsw":
            # Check if this is a large enough batch to create index
            # LanceDB recommends creating index after data is loaded
            try:
                row_count = table.count_rows()
                # Create index when we have all data (1M vectors)
                if row_count >= 1000000:
                    M = index_config.params.get("M", 16)
                    ef_construction = index_config.params.get("efConstruction", 64)
                    lance_metric = index_config.params.get("metric", "L2")
                    if lance_metric.lower() in ("cosine", "angular"):
                        lance_metric = "cosine"
                    table.create_index(
                        metric=lance_metric,
                        num_partitions=256,
                        num_sub_vectors=96,
                        index_type="IVF_HNSW_SQ",
                        # HNSW specific params
                        m=M,
                        ef_construction=ef_construction,
                    )
            except Exception:
                pass  # Index creation may fail if already exists or not enough data

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
        if table_name not in self._tables:
            self._tables[table_name] = self._db.open_table(table_name)

        table = self._tables[table_name]

        start_time = time.perf_counter()

        # Perform search
        query = query_vector.tolist()

        # LanceDB search with IVF_HNSW_SQ index
        # Note: IVF_HNSW doesn't expose HNSW ef_search at query time.
        # Instead, we use refine_factor to control search quality:
        # - nprobes: number of IVF partitions to search (fixed at reasonable value)
        # - refine_factor: re-ranks extra candidates (maps to efSearch semantics)
        if search_config.index_type == "hnsw":
            ef_search = search_config.params.get("efSearch", 64)
            # Map efSearch to refine_factor: ef=64->rf=1, ef=128->rf=2, ef=256->rf=4
            refine_factor = max(1, ef_search // 64)
            results = (
                table.search(query)
                .limit(k)
                .nprobes(50)  # Fixed: search ~20% of 256 partitions
                .refine_factor(refine_factor)
                .to_list()
            )
        else:
            # Flat search (brute force)
            results = (
                table.search(query)
                .limit(k)
                .to_list()
            )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Extract IDs and distances
        if results:
            ids = np.array([r["id"] for r in results], dtype=np.int64)
            distances = np.array([r["_distance"] for r in results], dtype=np.float32)
        else:
            ids = np.array([], dtype=np.int64)
            distances = np.array([], dtype=np.float32)

        return SearchResult(ids=ids, distances=distances, latency_ms=latency_ms)

    def get_stats(self, table_name: str) -> Dict[str, Any]:
        """
        Get table statistics.

        Args:
            table_name: Name of the table

        Returns:
            Dictionary with table stats
        """
        if self._db is None:
            return {
                "row_count": 0,
                "table_name": table_name,
                "error": "Not connected",
            }

        try:
            if table_name not in self._tables:
                self._tables[table_name] = self._db.open_table(table_name)

            table = self._tables[table_name]
            return {
                "row_count": table.count_rows(),
                "table_name": table_name,
            }
        except Exception as e:
            return {
                "row_count": 0,
                "table_name": table_name,
                "error": str(e),
            }
