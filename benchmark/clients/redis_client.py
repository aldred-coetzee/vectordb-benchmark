"""Redis Stack vector database client implementation."""

import time
from typing import Any, Dict, Optional

import numpy as np

try:
    import redis
    from redis.commands.search.field import VectorField, NumericField
    from redis.commands.search.index_definition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
except ImportError:
    redis = None

from .base import BaseVectorDBClient, IndexConfig, SearchConfig, SearchResult


class RedisClient(BaseVectorDBClient):
    """Redis Stack vector database client."""

    def __init__(self):
        """Initialize the Redis client."""
        if redis is None:
            raise ImportError(
                "redis package not installed. "
                "Install with: pip install redis"
            )

        self._client: Optional[redis.Redis] = None
        self._index_configs: Dict[str, IndexConfig] = {}
        self._dimensions: Dict[str, int] = {}

    @property
    def name(self) -> str:
        """Return the database name."""
        return "Redis"

    def connect(self, endpoint: str = "localhost:6379", **kwargs) -> None:
        """
        Connect to Redis Stack.

        Args:
            endpoint: Redis server endpoint (host:port)
            **kwargs: Additional connection parameters
        """
        # Parse endpoint
        endpoint = endpoint.replace("redis://", "").replace("http://", "")
        if ":" in endpoint:
            host, port_str = endpoint.split(":", 1)
            port = int(port_str)
        else:
            host = endpoint
            port = 6379

        try:
            self._client = redis.Redis(host=host, port=port, decode_responses=False)
            # Test connection
            self._client.ping()
            print(f"Connected to Redis at {host}:{port}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")

    def disconnect(self) -> None:
        """Disconnect from Redis."""
        self._index_configs.clear()
        self._dimensions.clear()
        if self._client:
            self._client.close()
        self._client = None
        print("Disconnected from Redis")

    def create_table(
        self,
        table_name: str,
        dimension: int,
        index_config: IndexConfig,
    ) -> None:
        """
        Create an index with the specified configuration.

        Args:
            table_name: Name of the index
            dimension: Vector dimensionality
            index_config: Index configuration
        """
        if self._client is None:
            raise RuntimeError("Not connected to database")

        # Drop existing index if present
        try:
            self._client.ft(table_name).dropindex(delete_documents=True)
        except Exception:
            pass  # Index doesn't exist

        # Build vector field configuration
        if index_config.index_type == "flat":
            vector_field = VectorField(
                "vector",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": dimension,
                    "DISTANCE_METRIC": "L2",
                },
            )
        elif index_config.index_type == "hnsw":
            M = index_config.params.get("M", 16)
            ef_construction = index_config.params.get("efConstruction", 64)
            vector_field = VectorField(
                "vector",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": dimension,
                    "DISTANCE_METRIC": "L2",
                    "M": M,
                    "EF_CONSTRUCTION": ef_construction,
                    "EF_RUNTIME": 64,  # Default, will be overridden at search time
                },
            )
        else:
            raise ValueError(f"Unsupported index type: {index_config.index_type}")

        # Create schema with vector field and id field
        schema = (
            NumericField("id"),
            vector_field,
        )

        # Create index
        try:
            self._client.ft(table_name).create_index(
                schema,
                definition=IndexDefinition(prefix=[f"{table_name}:"], index_type=IndexType.HASH),
            )
            self._index_configs[table_name] = index_config
            self._dimensions[table_name] = dimension
            print(f"Created index '{table_name}' with {index_config.index_type} index")
        except Exception as e:
            raise RuntimeError(f"Failed to create index: {e}")

    def drop_table(self, table_name: str) -> None:
        """
        Drop an index if it exists.

        Args:
            table_name: Name of the index to drop
        """
        if self._client is None:
            raise RuntimeError("Not connected to database")

        try:
            self._client.ft(table_name).dropindex(delete_documents=True)
            print(f"Dropped index '{table_name}'")
        except Exception:
            pass  # Index doesn't exist

        self._index_configs.pop(table_name, None)
        self._dimensions.pop(table_name, None)

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
        if self._client is None:
            raise RuntimeError("Not connected to database")

        # Use pipeline for batch insertion
        pipe = self._client.pipeline(transaction=False)

        for id_val, vector in zip(ids, vectors):
            key = f"{table_name}:{id_val}"
            # Store as hash with vector as bytes
            pipe.hset(
                key,
                mapping={
                    "id": int(id_val),
                    "vector": vector.astype(np.float32).tobytes(),
                },
            )

        try:
            pipe.execute()
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
            table_name: Name of the index
            query_vector: Query vector
            k: Number of neighbors
            search_config: Search configuration

        Returns:
            SearchResult with IDs and latency
        """
        if self._client is None:
            raise RuntimeError("Not connected to database")

        # Build query
        query_bytes = query_vector.astype(np.float32).tobytes()

        # Set EF_RUNTIME for HNSW
        if search_config.index_type == "hnsw":
            ef_search = search_config.params.get("efSearch", 64)
            query_str = f"*=>[KNN {k} @vector $vec EF_RUNTIME {ef_search}]"
        else:
            query_str = f"*=>[KNN {k} @vector $vec]"

        query = (
            Query(query_str)
            .return_fields("id", "__vector_score")
            .sort_by("__vector_score")
            .dialect(2)
        )

        try:
            start_time = time.perf_counter()

            results = self._client.ft(table_name).search(
                query,
                query_params={"vec": query_bytes},
            )

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            # Extract IDs and distances
            if results.docs:
                # Redis returns IDs as "prefix:id", extract just the numeric part
                ids = np.array([int(doc.id.split(":")[-1]) for doc in results.docs], dtype=np.int64)
                # Redis returns score (lower is better for L2)
                distances = np.array(
                    [float(doc.__vector_score) for doc in results.docs],
                    dtype=np.float32,
                )
            else:
                ids = np.array([], dtype=np.int64)
                distances = np.array([], dtype=np.float32)

            return SearchResult(ids=ids, distances=distances, latency_ms=latency_ms)

        except Exception as e:
            raise RuntimeError(f"Search failed: {e}")

    def get_stats(self, table_name: str) -> Dict[str, Any]:
        """
        Get index statistics.

        Args:
            table_name: Name of the index

        Returns:
            Dictionary with index stats
        """
        if self._client is None:
            return {
                "row_count": 0,
                "table_name": table_name,
                "error": "Not connected",
            }

        try:
            info = self._client.ft(table_name).info()
            return {
                "row_count": int(info.get("num_docs", 0)),
                "table_name": table_name,
            }
        except Exception as e:
            return {
                "row_count": 0,
                "table_name": table_name,
                "error": str(e),
            }
