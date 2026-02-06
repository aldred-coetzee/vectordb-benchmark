"""Qdrant vector database client implementation."""

import time
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from qdrant_client import QdrantClient as QdrantSDK
    from qdrant_client.http.exceptions import UnexpectedResponse
    from qdrant_client.models import (
        Distance,
        HnswConfigDiff,
        PointStruct,
        QueryRequest,
        SearchParams,
        VectorParams,
    )
except ImportError:
    QdrantSDK = None
    UnexpectedResponse = None

from .base import BaseVectorDBClient, IndexConfig, SearchConfig, SearchResult


class QdrantClient(BaseVectorDBClient):
    """Qdrant vector database client."""

    def __init__(self):
        """Initialize the Qdrant client."""
        if QdrantSDK is None:
            raise ImportError(
                "qdrant-client package not installed. "
                "Install with: pip install qdrant-client"
            )

        self._client: Optional[QdrantSDK] = None
        self._index_configs: Dict[str, IndexConfig] = {}

    @property
    def name(self) -> str:
        """Return the database name."""
        return "Qdrant"

    def connect(self, endpoint: str = "http://localhost:6333", **kwargs) -> None:
        """
        Connect to Qdrant.

        Args:
            endpoint: Qdrant server endpoint URL
            **kwargs: Additional connection parameters
        """
        # Parse endpoint to extract host and port
        endpoint = endpoint.rstrip("/")
        if endpoint.startswith("http://"):
            endpoint = endpoint[7:]
        elif endpoint.startswith("https://"):
            endpoint = endpoint[8:]

        if ":" in endpoint:
            host, port_str = endpoint.split(":", 1)
            port = int(port_str)
        else:
            host = endpoint
            port = 6333

        grpc_port = kwargs.get("grpc_port", 6334)
        try:
            self._client = QdrantSDK(
                host=host, port=port, grpc_port=grpc_port,
                prefer_grpc=True, timeout=300,
            )
            # Test connection by getting collections
            self._client.get_collections()
            print(f"Connected to Qdrant at {host}:{port} (gRPC port {grpc_port})")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant: {e}")

    def get_version(self) -> str:
        """Return Qdrant server version via REST API root endpoint."""
        if self._client is None:
            return "unknown"
        try:
            # qdrant-client exposes openapi_client with service API
            info = self._client.http.service_api.root_api()
            return info.version
        except Exception:
            return "unknown"

    def disconnect(self) -> None:
        """Disconnect from Qdrant."""
        self._index_configs.clear()
        if self._client is not None:
            try:
                # Close the client to avoid socket leaks
                if hasattr(self._client, 'close'):
                    self._client.close()
            except Exception:
                pass  # Ignore errors during cleanup
        self._client = None
        print("Disconnected from Qdrant")

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

        # Build HNSW config based on index type
        if index_config.index_type == "flat":
            # For exact search, we use default HNSW config but search with exact=True
            # Qdrant doesn't have a true flat index, so we rely on SearchParams(exact=True)
            hnsw_config = None
        elif index_config.index_type == "hnsw":
            m = index_config.params.get("M", 16)
            ef_construct = index_config.params.get("efConstruction", 64)
            hnsw_config = HnswConfigDiff(
                m=m,
                ef_construct=ef_construct,
            )
        else:
            raise ValueError(f"Unsupported index type: {index_config.index_type}")

        # Determine distance metric
        metric = index_config.params.get("metric", "L2")
        if metric.lower() in ("cosine", "angular"):
            distance = Distance.COSINE
        else:
            distance = Distance.EUCLID

        try:
            self._client.recreate_collection(
                collection_name=table_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=distance,
                    hnsw_config=hnsw_config if index_config.index_type == "hnsw" else None,
                ),
            )
            self._index_configs[table_name] = index_config
            print(f"Created table '{table_name}' with {index_config.index_type} index")
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

        try:
            self._client.delete_collection(table_name)
            print(f"Dropped table '{table_name}'")
        except UnexpectedResponse as e:
            # Collection doesn't exist (404), which is fine
            if e.status_code == 404:
                pass
            else:
                raise RuntimeError(f"Failed to drop collection '{table_name}': {e}")
        except Exception as e:
            # Check if error message indicates collection not found
            error_msg = str(e).lower()
            if "not found" in error_msg or "does not exist" in error_msg:
                pass  # Collection doesn't exist, which is fine
            else:
                raise RuntimeError(f"Failed to drop collection '{table_name}': {e}")

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

        # Convert vectors array to list of lists once for better performance
        vectors_list = vectors.tolist()
        ids_list = ids.tolist()

        # Convert to list of PointStruct
        points = [
            PointStruct(
                id=int(ids_list[i]),
                vector=vectors_list[i],
            )
            for i in range(len(ids))
        ]

        try:
            # Upsert in batches sized to stay under gRPC max message size (default 64MB).
            # With prefer_grpc=True, vectors are sent as protobuf binary (~4 bytes/float).
            dims = len(vectors_list[0]) if vectors_list else 128
            max_payload_bytes = 50_000_000  # 50MB conservative (gRPC max is 64MB)
            bytes_per_point = dims * 4 + 100  # protobuf float32 (4 bytes) + overhead
            batch_size = max(100, min(10000, max_payload_bytes // bytes_per_point))
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self._client.upsert(
                    collection_name=table_name,
                    points=batch,
                    wait=True,
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

        # Build search parameters based on index type
        if search_config.index_type == "flat":
            # Use exact search for flat index
            search_params = SearchParams(exact=True)
        else:
            # HNSW search with ef parameter
            ef_search = search_config.params.get("efSearch", 64)
            search_params = SearchParams(hnsw_ef=ef_search)

        try:
            start_time = time.perf_counter()

            # Use query_points() API (qdrant-client >= 1.7)
            response = self._client.query_points(
                collection_name=table_name,
                query=query_vector.tolist(),
                limit=k,
                search_params=search_params,
            )

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            # Extract IDs and distances from response.points
            points = response.points
            if points:
                ids = np.array([point.id for point in points], dtype=np.int64)
                # Qdrant returns scores; for EUCLID distance, lower is better
                distances = np.array([point.score for point in points], dtype=np.float32)
            else:
                ids = np.array([], dtype=np.int64)
                distances = np.array([], dtype=np.float32)

            return SearchResult(ids=ids, distances=distances, latency_ms=latency_ms)

        except Exception as e:
            raise RuntimeError(f"Search failed: {e}")

    @property
    def has_batch_search(self) -> bool:
        """Qdrant supports batch search via search_batch() API."""
        return True

    def batch_search(
        self,
        table_name: str,
        query_vectors: np.ndarray,
        k: int,
        search_config: SearchConfig,
    ) -> List[SearchResult]:
        """
        Batch search using Qdrant's query_batch_points() API.

        Sends all queries via gRPC. Sub-batches if needed
        to stay under 64MB gRPC message size limit.
        """
        if self._client is None:
            raise RuntimeError("Not connected to database")

        # Build search parameters
        if search_config.index_type == "flat":
            search_params = SearchParams(exact=True)
        else:
            ef_search = search_config.params.get("efSearch", 64)
            search_params = SearchParams(hnsw_ef=ef_search)

        # Calculate sub-batch size to stay under gRPC max message size (default 64MB).
        # Also cap at 500 queries per sub-batch to avoid server-side timeouts
        # (FLAT search of 10K queries against 1M vectors can exceed default timeout)
        dims = query_vectors.shape[1]
        max_payload_bytes = 50_000_000  # 50MB conservative (gRPC max is 64MB)
        bytes_per_query = dims * 4 + 100  # protobuf float32 (4 bytes) + overhead
        payload_batch = max(10, max_payload_bytes // bytes_per_query)
        batch_size = min(len(query_vectors), payload_batch, 500)  # Cap at 500 for timeout safety

        all_results = []

        try:
            for i in range(0, len(query_vectors), batch_size):
                batch = query_vectors[i:i + batch_size]

                requests = [
                    QueryRequest(
                        query=query.tolist(),
                        limit=k,
                        params=search_params,
                    )
                    for query in batch
                ]

                start_time = time.perf_counter()
                responses = self._client.query_batch_points(
                    collection_name=table_name,
                    requests=requests,
                )
                end_time = time.perf_counter()
                batch_ms = (end_time - start_time) * 1000

                for resp in responses:
                    points = resp.points if hasattr(resp, 'points') else resp
                    if points:
                        ids = np.array([point.id for point in points], dtype=np.int64)
                        distances = np.array([point.score for point in points], dtype=np.float32)
                    else:
                        ids = np.array([], dtype=np.int64)
                        distances = np.array([], dtype=np.float32)

                    all_results.append(SearchResult(
                        ids=ids,
                        distances=distances,
                        latency_ms=batch_ms / len(batch),
                    ))

        except Exception as e:
            raise RuntimeError(f"Batch search failed: {e}")

        return all_results

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

        try:
            info = self._client.get_collection(table_name)
            return {
                "row_count": info.points_count,
                "table_name": table_name,
                "status": info.status,
            }
        except Exception as e:
            return {
                "row_count": 0,
                "table_name": table_name,
                "error": str(e),
            }
