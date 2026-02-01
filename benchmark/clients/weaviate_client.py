"""Weaviate vector database client implementation."""

import time
from typing import Any, Dict, Optional

import numpy as np

try:
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType, VectorDistances
    from weaviate.classes.query import MetadataQuery
except ImportError:
    weaviate = None

from .base import BaseVectorDBClient, IndexConfig, SearchConfig, SearchResult


class WeaviateClient(BaseVectorDBClient):
    """Weaviate vector database client."""

    def __init__(self):
        """Initialize the Weaviate client."""
        if weaviate is None:
            raise ImportError(
                "weaviate-client package not installed. "
                "Install with: pip install weaviate-client"
            )

        self._client: Optional[weaviate.WeaviateClient] = None
        self._index_configs: Dict[str, IndexConfig] = {}

    @property
    def name(self) -> str:
        """Return the database name."""
        return "Weaviate"

    def connect(self, endpoint: str = "http://localhost:8080", **kwargs) -> None:
        """
        Connect to Weaviate.

        Args:
            endpoint: Weaviate server endpoint URL
            **kwargs: Additional connection parameters
        """
        # Parse endpoint to extract host and port
        endpoint = endpoint.rstrip("/")
        if endpoint.startswith("http://"):
            host = endpoint[7:]
        elif endpoint.startswith("https://"):
            host = endpoint[8:]
        else:
            host = endpoint

        if ":" in host:
            host_part, port_str = host.split(":", 1)
            port = int(port_str)
        else:
            host_part = host
            port = 8080

        try:
            self._client = weaviate.connect_to_local(
                host=host_part,
                port=port,
                grpc_port=kwargs.get("grpc_port", 50051),
            )
            # Test connection
            self._client.is_ready()
            print(f"Connected to Weaviate at {host_part}:{port}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Weaviate: {e}")

    def disconnect(self) -> None:
        """Disconnect from Weaviate."""
        self._index_configs.clear()
        if self._client:
            self._client.close()
        self._client = None
        print("Disconnected from Weaviate")

    def _get_collection_name(self, table_name: str) -> str:
        """Convert table name to valid Weaviate collection name (PascalCase, starts with uppercase)."""
        # Weaviate requires collection names to start with uppercase
        name = table_name.replace("_", " ").title().replace(" ", "")
        return name

    def create_table(
        self,
        table_name: str,
        dimension: int,
        index_config: IndexConfig,
    ) -> None:
        """
        Create a collection with the specified index configuration.

        Args:
            table_name: Name of the collection
            dimension: Vector dimensionality
            index_config: Index configuration
        """
        if self._client is None:
            raise RuntimeError("Not connected to database")

        collection_name = self._get_collection_name(table_name)

        # Build vector index config based on index type
        if index_config.index_type == "flat":
            # Weaviate doesn't have a true flat index
            # Use HNSW with high ef for near-exact search
            vector_index_config = Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.L2_SQUARED,
                ef=256,
                ef_construction=128,
                max_connections=64,
            )
        elif index_config.index_type == "hnsw":
            m = index_config.params.get("M", 16)
            ef_construct = index_config.params.get("efConstruction", 64)
            vector_index_config = Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.L2_SQUARED,
                ef=64,  # Default ef for search
                ef_construction=ef_construct,
                max_connections=m,
            )
        else:
            raise ValueError(f"Unsupported index type: {index_config.index_type}")

        try:
            # Delete if exists
            if self._client.collections.exists(collection_name):
                self._client.collections.delete(collection_name)

            # Create collection with vector index
            self._client.collections.create(
                name=collection_name,
                properties=[
                    Property(name="vector_id", data_type=DataType.INT),
                ],
                vectorizer_config=Configure.Vectorizer.none(),
                vector_index_config=vector_index_config,
            )

            self._index_configs[table_name] = index_config
            print(f"Created collection '{collection_name}' with {index_config.index_type} index")
        except Exception as e:
            raise RuntimeError(f"Failed to create collection: {e}")

    def drop_table(self, table_name: str) -> None:
        """
        Drop a collection if it exists.

        Args:
            table_name: Name of the collection to drop
        """
        if self._client is None:
            raise RuntimeError("Not connected to database")

        collection_name = self._get_collection_name(table_name)

        try:
            if self._client.collections.exists(collection_name):
                self._client.collections.delete(collection_name)
                print(f"Dropped collection '{collection_name}'")
        except Exception:
            # Collection doesn't exist, which is fine
            pass

        self._index_configs.pop(table_name, None)

    def insert(
        self,
        table_name: str,
        ids: np.ndarray,
        vectors: np.ndarray,
    ) -> None:
        """
        Insert vectors into the collection.

        Args:
            table_name: Name of the collection
            ids: Array of vector IDs
            vectors: Array of vectors
        """
        if self._client is None:
            raise RuntimeError("Not connected to database")

        collection_name = self._get_collection_name(table_name)
        collection = self._client.collections.get(collection_name)

        # Convert to lists for Weaviate
        vectors_list = vectors.tolist()
        ids_list = ids.tolist()

        try:
            # Use batch insert for better performance
            batch_size = 5000
            for i in range(0, len(ids_list), batch_size):
                batch_ids = ids_list[i:i + batch_size]
                batch_vectors = vectors_list[i:i + batch_size]

                with collection.batch.dynamic() as batch:
                    for j, (vid, vec) in enumerate(zip(batch_ids, batch_vectors)):
                        batch.add_object(
                            properties={"vector_id": int(vid)},
                            vector=vec,
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
            table_name: Name of the collection
            query_vector: Query vector
            k: Number of neighbors
            search_config: Search configuration

        Returns:
            SearchResult with IDs and latency
        """
        if self._client is None:
            raise RuntimeError("Not connected to database")

        collection_name = self._get_collection_name(table_name)
        collection = self._client.collections.get(collection_name)

        try:
            start_time = time.perf_counter()

            # Perform vector search
            # Note: Weaviate v4 doesn't allow runtime ef changes easily,
            # so we rely on the collection's configured ef
            response = collection.query.near_vector(
                near_vector=query_vector.tolist(),
                limit=k,
                return_metadata=MetadataQuery(distance=True),
            )

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            # Extract IDs and distances
            if response.objects:
                ids = np.array(
                    [obj.properties["vector_id"] for obj in response.objects],
                    dtype=np.int64
                )
                distances = np.array(
                    [obj.metadata.distance for obj in response.objects],
                    dtype=np.float32
                )
            else:
                ids = np.array([], dtype=np.int64)
                distances = np.array([], dtype=np.float32)

            return SearchResult(ids=ids, distances=distances, latency_ms=latency_ms)

        except Exception as e:
            raise RuntimeError(f"Search failed: {e}")

    def get_stats(self, table_name: str) -> Dict[str, Any]:
        """
        Get collection statistics.

        Args:
            table_name: Name of the collection

        Returns:
            Dictionary with collection stats
        """
        if self._client is None:
            return {
                "row_count": 0,
                "table_name": table_name,
                "error": "Not connected",
            }

        collection_name = self._get_collection_name(table_name)

        try:
            collection = self._client.collections.get(collection_name)
            aggregate = collection.aggregate.over_all(total_count=True)
            return {
                "row_count": aggregate.total_count,
                "table_name": table_name,
                "collection_name": collection_name,
            }
        except Exception as e:
            return {
                "row_count": 0,
                "table_name": table_name,
                "error": str(e),
            }
