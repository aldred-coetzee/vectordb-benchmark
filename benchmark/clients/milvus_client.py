"""Milvus vector database client implementation."""

import time
from typing import Any, Dict, Optional

import numpy as np

try:
    from pymilvus import (
        connections,
        utility,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
    )
except ImportError:
    connections = None

from .base import BaseVectorDBClient, IndexConfig, SearchConfig, SearchResult


class MilvusClient(BaseVectorDBClient):
    """Milvus vector database client."""

    def __init__(self):
        """Initialize the Milvus client."""
        if connections is None:
            raise ImportError(
                "pymilvus package not installed. "
                "Install with: pip install pymilvus"
            )

        self._connection_alias = "default"
        self._index_configs: Dict[str, IndexConfig] = {}

    @property
    def name(self) -> str:
        """Return the database name."""
        return "Milvus"

    def connect(self, endpoint: str = "localhost:19530", **kwargs) -> None:
        """
        Connect to Milvus.

        Args:
            endpoint: Milvus server endpoint (host:port)
            **kwargs: Additional connection parameters
        """
        # Parse endpoint
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
            port = 19530

        try:
            connections.connect(
                alias=self._connection_alias,
                host=host,
                port=port,
            )
            print(f"Connected to Milvus at {host}:{port}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {e}")

    def disconnect(self) -> None:
        """Disconnect from Milvus."""
        self._index_configs.clear()
        try:
            connections.disconnect(self._connection_alias)
        except Exception:
            pass
        print("Disconnected from Milvus")

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
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
        ]
        schema = CollectionSchema(fields=fields, description="Benchmark collection")

        try:
            # Drop if exists
            if utility.has_collection(table_name):
                utility.drop_collection(table_name)

            # Create collection
            collection = Collection(name=table_name, schema=schema)

            # Build index configuration based on index type
            if index_config.index_type == "flat":
                index_params = {
                    "metric_type": "L2",
                    "index_type": "FLAT",
                    "params": {},
                }
            elif index_config.index_type == "hnsw":
                m = index_config.params.get("M", 16)
                ef_construct = index_config.params.get("efConstruction", 64)
                index_params = {
                    "metric_type": "L2",
                    "index_type": "HNSW",
                    "params": {
                        "M": m,
                        "efConstruction": ef_construct,
                    },
                }
            else:
                raise ValueError(f"Unsupported index type: {index_config.index_type}")

            # Create index on vector field
            collection.create_index(field_name="vector", index_params=index_params)

            self._index_configs[table_name] = index_config
            print(f"Created collection '{table_name}' with {index_config.index_type} index")
        except Exception as e:
            raise RuntimeError(f"Failed to create collection: {e}")

    def drop_table(self, table_name: str) -> None:
        """
        Drop a collection if it exists.

        Args:
            table_name: Name of the collection to drop
        """
        try:
            if utility.has_collection(table_name):
                utility.drop_collection(table_name)
                print(f"Dropped collection '{table_name}'")
        except Exception:
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
        try:
            collection = Collection(table_name)

            # Prepare data
            entities = [
                ids.tolist(),
                vectors.tolist(),
            ]

            # Insert
            collection.insert(entities)

            # Flush to ensure data is persisted
            collection.flush()

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
        try:
            collection = Collection(table_name)

            # Ensure collection is loaded
            collection.load()

            # Build search parameters
            if search_config.index_type == "flat":
                search_params = {
                    "metric_type": "L2",
                    "params": {},
                }
            else:
                ef_search = search_config.params.get("efSearch", 64)
                search_params = {
                    "metric_type": "L2",
                    "params": {"ef": ef_search},
                }

            start_time = time.perf_counter()

            # Perform search
            results = collection.search(
                data=[query_vector.tolist()],
                anns_field="vector",
                param=search_params,
                limit=k,
                output_fields=["id"],
            )

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            # Extract results
            if results and len(results) > 0:
                hits = results[0]
                ids = np.array([hit.id for hit in hits], dtype=np.int64)
                distances = np.array([hit.distance for hit in hits], dtype=np.float32)
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
        try:
            collection = Collection(table_name)
            collection.flush()
            return {
                "row_count": collection.num_entities,
                "table_name": table_name,
            }
        except Exception as e:
            return {
                "row_count": 0,
                "table_name": table_name,
                "error": str(e),
            }
