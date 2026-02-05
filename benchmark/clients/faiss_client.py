"""FAISS vector database client implementation."""

import time
from typing import Any, Dict, List

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from .base import BaseVectorDBClient, IndexConfig, SearchConfig, SearchResult


class FAISSClient(BaseVectorDBClient):
    """FAISS vector database client (in-process, no network overhead)."""

    def __init__(self):
        """Initialize the FAISS client."""
        if faiss is None:
            raise ImportError(
                "faiss-cpu package not installed. "
                "Install with: pip install faiss-cpu"
            )

        self._indexes: Dict[str, faiss.Index] = {}
        self._id_maps: Dict[str, np.ndarray] = {}
        self._index_configs: Dict[str, IndexConfig] = {}
        self._dimensions: Dict[str, int] = {}
        self._metrics: Dict[str, str] = {}  # table_name -> "L2" or "cosine"

    @property
    def name(self) -> str:
        """Return the database name."""
        return "FAISS"

    def connect(self, **kwargs) -> None:
        """
        Connect to FAISS (no-op since FAISS runs in-process).

        Args:
            **kwargs: Ignored for FAISS
        """
        print("FAISS running in-process (no network connection needed)")

    def disconnect(self) -> None:
        """Disconnect from FAISS (clears all indexes)."""
        self._indexes.clear()
        self._id_maps.clear()
        self._index_configs.clear()
        self._dimensions.clear()
        self._metrics.clear()
        print("FAISS indexes cleared")

    def create_table(
        self,
        table_name: str,
        dimension: int,
        index_config: IndexConfig,
    ) -> None:
        """
        Create an index with the specified configuration.

        Args:
            table_name: Name of the index (used as key)
            dimension: Vector dimensionality
            index_config: Index configuration
        """
        metric = index_config.params.get("metric", "L2")
        use_ip = metric.lower() in ("cosine", "ip", "angular")

        if index_config.index_type == "flat":
            if use_ip:
                index = faiss.IndexFlatIP(dimension)
            else:
                index = faiss.IndexFlatL2(dimension)
        elif index_config.index_type == "hnsw":
            M = index_config.params.get("M", 16)
            ef_construction = index_config.params.get("efConstruction", 64)

            if use_ip:
                index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
            else:
                index = faiss.IndexHNSWFlat(dimension, M)
            index.hnsw.efConstruction = ef_construction
        else:
            raise ValueError(f"Unsupported index type: {index_config.index_type}")

        self._indexes[table_name] = index
        self._id_maps[table_name] = np.array([], dtype=np.int64)
        self._index_configs[table_name] = index_config
        self._dimensions[table_name] = dimension
        self._metrics[table_name] = metric
        print(f"Created table '{table_name}' with {index_config.index_type} index (metric={metric})")

    def drop_table(self, table_name: str) -> None:
        """
        Drop an index if it exists.

        Args:
            table_name: Name of the index to drop
        """
        if table_name in self._indexes:
            del self._indexes[table_name]
            del self._id_maps[table_name]
            del self._index_configs[table_name]
            del self._dimensions[table_name]
            print(f"Dropped table '{table_name}'")

    def insert(
        self,
        table_name: str,
        ids: np.ndarray,
        vectors: np.ndarray,
    ) -> None:
        """
        Insert vectors into the index.

        Args:
            table_name: Name of the index
            ids: Array of vector IDs
            vectors: Array of vectors
        """
        if table_name not in self._indexes:
            raise RuntimeError(f"Index '{table_name}' does not exist")

        index = self._indexes[table_name]

        # Ensure vectors are float32 and contiguous
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        # Normalize for cosine metric (FAISS IP requires unit vectors)
        metric = self._metrics.get(table_name, "L2")
        if metric.lower() in ("cosine", "ip", "angular"):
            faiss.normalize_L2(vectors)

        # FAISS uses internal sequential IDs, so we maintain a mapping
        # Append new IDs to our mapping
        self._id_maps[table_name] = np.concatenate([
            self._id_maps[table_name],
            ids.astype(np.int64)
        ])

        # Add vectors to index
        index.add(vectors)

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
            table_name: Name of the index
            query_vector: Query vector
            k: Number of neighbors
            search_config: Search configuration

        Returns:
            SearchResult with IDs and latency
        """
        if table_name not in self._indexes:
            raise RuntimeError(f"Index '{table_name}' does not exist")

        index = self._indexes[table_name]
        id_map = self._id_maps[table_name]

        # Set efSearch for HNSW indexes
        if search_config.index_type == "hnsw":
            ef_search = search_config.params.get("efSearch", 64)
            index.hnsw.efSearch = ef_search

        # Ensure query is float32 and 2D
        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)

        # Normalize for cosine metric
        metric = self._metrics.get(table_name, "L2")
        if metric.lower() in ("cosine", "ip", "angular"):
            faiss.normalize_L2(query)

        start_time = time.perf_counter()
        distances, indices = index.search(query, k)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        # Map FAISS internal indices back to original IDs
        # indices[0] contains the internal FAISS indices for the single query
        faiss_indices = indices[0]

        # Filter out -1 (no result) and map to original IDs
        valid_mask = faiss_indices >= 0
        valid_indices = faiss_indices[valid_mask]

        if len(valid_indices) > 0:
            result_ids = id_map[valid_indices]
            result_distances = distances[0][valid_mask]
        else:
            result_ids = np.array([], dtype=np.int64)
            result_distances = np.array([], dtype=np.float32)

        return SearchResult(
            ids=result_ids,
            distances=result_distances,
            latency_ms=latency_ms
        )

    @property
    def has_batch_search(self) -> bool:
        """FAISS natively supports multi-query search."""
        return True

    def batch_search(
        self,
        table_name: str,
        query_vectors: np.ndarray,
        k: int,
        search_config: SearchConfig,
    ) -> List[SearchResult]:
        """
        Batch search using FAISS native multi-query support.

        FAISS index.search() accepts a 2D array of queries and returns
        results for all queries in a single SIMD-optimized call.
        """
        if table_name not in self._indexes:
            raise RuntimeError(f"Index '{table_name}' does not exist")

        index = self._indexes[table_name]
        id_map = self._id_maps[table_name]

        # Set efSearch for HNSW indexes
        if search_config.index_type == "hnsw":
            ef_search = search_config.params.get("efSearch", 64)
            index.hnsw.efSearch = ef_search

        # Ensure queries are float32 and contiguous 2D array
        queries = np.ascontiguousarray(query_vectors, dtype=np.float32)

        # Normalize for cosine metric
        metric = self._metrics.get(table_name, "L2")
        if metric.lower() in ("cosine", "ip", "angular"):
            faiss.normalize_L2(queries)

        start_time = time.perf_counter()
        distances, indices = index.search(queries, k)
        end_time = time.perf_counter()

        total_ms = (end_time - start_time) * 1000

        # Build per-query SearchResult objects
        results = []
        for i in range(len(queries)):
            faiss_indices = indices[i]
            valid_mask = faiss_indices >= 0
            valid_indices = faiss_indices[valid_mask]

            if len(valid_indices) > 0:
                result_ids = id_map[valid_indices]
                result_distances = distances[i][valid_mask]
            else:
                result_ids = np.array([], dtype=np.int64)
                result_distances = np.array([], dtype=np.float32)

            results.append(SearchResult(
                ids=result_ids,
                distances=result_distances,
                latency_ms=total_ms / len(queries),
            ))

        return results

    def get_stats(self, table_name: str) -> Dict[str, Any]:
        """
        Get index statistics.

        Args:
            table_name: Name of the index

        Returns:
            Dictionary with index stats
        """
        if table_name not in self._indexes:
            return {
                "row_count": 0,
                "table_name": table_name,
                "error": "Index does not exist",
            }

        index = self._indexes[table_name]

        return {
            "row_count": index.ntotal,
            "table_name": table_name,
            "dimension": self._dimensions.get(table_name, 0),
            "index_type": self._index_configs.get(table_name, IndexConfig("", "", {})).index_type,
        }
