"""ChromaDB vector database client implementation."""

import time
from typing import Any, Dict, Optional

import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

from .base import BaseVectorDBClient, IndexConfig, SearchConfig, SearchResult


class ChromaClient(BaseVectorDBClient):
    """ChromaDB vector database client."""

    def __init__(self):
        """Initialize the ChromaDB client."""
        if chromadb is None:
            raise ImportError(
                "chromadb package not installed. "
                "Install with: pip install chromadb"
            )

        self._client: Optional[Any] = None
        self._collections: Dict[str, Any] = {}
        self._index_configs: Dict[str, IndexConfig] = {}

    @property
    def name(self) -> str:
        """Return the database name."""
        return "Chroma"

    def connect(self, endpoint: str = "http://localhost:8000", **kwargs) -> None:
        """
        Connect to ChromaDB.

        Args:
            endpoint: ChromaDB server endpoint URL
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
            port = 8000

        try:
            self._client = chromadb.HttpClient(host=host, port=port)
            # Test connection
            self._client.heartbeat()
            print(f"Connected to ChromaDB at {host}:{port}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to ChromaDB: {e}")

    def disconnect(self) -> None:
        """Disconnect from ChromaDB."""
        self._collections.clear()
        self._index_configs.clear()
        self._client = None
        print("Disconnected from ChromaDB")

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

        # Delete existing collection if present
        try:
            self._client.delete_collection(table_name)
        except Exception:
            pass  # Collection doesn't exist

        # Build HNSW metadata
        if index_config.index_type == "flat":
            # Chroma doesn't have true flat index, use HNSW with high search_ef
            metadata = {
                "hnsw:space": "l2",
                "hnsw:search_ef": 256,
                "hnsw:construction_ef": 128,
                "hnsw:M": 64,
            }
        elif index_config.index_type == "hnsw":
            M = index_config.params.get("M", 16)
            ef_construction = index_config.params.get("efConstruction", 64)
            metadata = {
                "hnsw:space": "l2",
                "hnsw:construction_ef": ef_construction,
                "hnsw:M": M,
                "hnsw:search_ef": 64,  # Default, will be updated at search time
            }
        else:
            raise ValueError(f"Unsupported index type: {index_config.index_type}")

        try:
            collection = self._client.create_collection(
                name=table_name,
                metadata=metadata,
            )
            self._collections[table_name] = collection
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
        if self._client is None:
            raise RuntimeError("Not connected to database")

        try:
            self._client.delete_collection(table_name)
            print(f"Dropped collection '{table_name}'")
        except Exception:
            pass  # Collection doesn't exist

        self._collections.pop(table_name, None)
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
        if table_name not in self._collections:
            self._collections[table_name] = self._client.get_collection(table_name)

        collection = self._collections[table_name]

        # ChromaDB has a max batch size limit (5461 for 128-dim vectors)
        max_batch_size = 5000  # Stay under the limit

        try:
            for i in range(0, len(ids), max_batch_size):
                batch_ids = ids[i:i + max_batch_size]
                batch_vectors = vectors[i:i + max_batch_size]

                # ChromaDB expects string IDs and list of embeddings
                str_ids = [str(id_val) for id_val in batch_ids]
                embeddings = batch_vectors.tolist()

                collection.add(
                    ids=str_ids,
                    embeddings=embeddings,
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
        if table_name not in self._collections:
            self._collections[table_name] = self._client.get_collection(table_name)

        collection = self._collections[table_name]

        # Update search_ef if HNSW
        if search_config.index_type == "hnsw":
            ef_search = search_config.params.get("efSearch", 64)
            try:
                collection.modify(metadata={"hnsw:search_ef": ef_search})
            except Exception:
                pass  # May fail if collection doesn't support modification

        start_time = time.perf_counter()

        try:
            results = collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=k,
                include=["distances"],
            )

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            # Extract IDs and distances
            if results and results["ids"] and results["ids"][0]:
                # Convert string IDs back to int64
                ids = np.array([int(id_str) for id_str in results["ids"][0]], dtype=np.int64)
                distances = np.array(results["distances"][0], dtype=np.float32) if results["distances"] else np.array([], dtype=np.float32)
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

        try:
            if table_name not in self._collections:
                self._collections[table_name] = self._client.get_collection(table_name)

            collection = self._collections[table_name]
            return {
                "row_count": collection.count(),
                "table_name": table_name,
            }
        except Exception as e:
            return {
                "row_count": 0,
                "table_name": table_name,
                "error": str(e),
            }
