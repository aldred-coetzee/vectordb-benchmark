"""Abstract base class for vector database clients."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class IndexConfig:
    """Configuration for a vector index."""

    name: str
    index_type: str  # e.g., "flat", "hnsw"
    params: Dict[str, Any]

    def get_description(self) -> str:
        """Return a human-readable description of the index config."""
        if self.index_type == "flat":
            dims = self.params.get("dims", "?")
            return f"dims={dims}"
        elif self.index_type == "hnsw":
            m = self.params.get("M", "?")
            ef_construction = self.params.get("efConstruction", "?")
            return f"M={m},efC={ef_construction}"
        else:
            return str(self.params)


@dataclass
class SearchConfig:
    """Configuration for a search operation."""

    index_name: str
    index_type: str
    params: Dict[str, Any]

    def get_description(self) -> str:
        """Return a human-readable description of the search config."""
        if self.index_type == "flat":
            return "-"
        elif self.index_type == "hnsw":
            ef_search = self.params.get("efSearch", "?")
            return f"efSearch={ef_search}"
        else:
            return str(self.params)


@dataclass
class SearchResult:
    """Result from a search operation."""

    ids: np.ndarray  # Shape: (k,) - IDs of nearest neighbors
    distances: Optional[np.ndarray] = None  # Shape: (k,) - distances if available
    latency_ms: float = 0.0


class BaseVectorDBClient(ABC):
    """
    Abstract base class for vector database clients.

    All vector database implementations must inherit from this class
    and implement all abstract methods.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the database (e.g., 'KDB.AI', 'Milvus')."""
        pass

    @abstractmethod
    def connect(self, **kwargs) -> None:
        """
        Connect to the database.

        Args:
            **kwargs: Database-specific connection parameters
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the database."""
        pass

    @abstractmethod
    def create_table(
        self,
        table_name: str,
        dimension: int,
        index_config: IndexConfig,
    ) -> None:
        """
        Create a table/collection with the specified index configuration.

        Args:
            table_name: Name of the table to create
            dimension: Dimensionality of vectors
            index_config: Index configuration (type, parameters)
        """
        pass

    @abstractmethod
    def drop_table(self, table_name: str) -> None:
        """
        Drop a table if it exists.

        Args:
            table_name: Name of the table to drop
        """
        pass

    @abstractmethod
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
            ids: Array of vector IDs (shape: (n,))
            vectors: Array of vectors (shape: (n, dim))
        """
        pass

    @abstractmethod
    def search(
        self,
        table_name: str,
        query_vector: np.ndarray,
        k: int,
        search_config: SearchConfig,
    ) -> SearchResult:
        """
        Search for k nearest neighbors of a query vector.

        Args:
            table_name: Name of the table to search
            query_vector: Query vector (shape: (dim,))
            k: Number of nearest neighbors to return
            search_config: Search configuration (index params, etc.)

        Returns:
            SearchResult containing IDs and optionally distances
        """
        pass

    @abstractmethod
    def get_stats(self, table_name: str) -> Dict[str, Any]:
        """
        Get statistics about the table.

        Args:
            table_name: Name of the table

        Returns:
            Dictionary containing table statistics (row count, etc.)
        """
        pass

    def get_version(self) -> str:
        """Return the version of the database server or library.

        Subclasses should override this to query the actual version.
        Called after connect().
        """
        return "unknown"

    def get_client_version(self) -> str:
        """Return the version of the client library used to connect.

        For embedded databases, this is the same as get_version().
        For client-server databases, this is the Python SDK version.
        """
        return "unknown"

    @property
    def has_batch_search(self) -> bool:
        """Whether this client supports native batch search."""
        return False

    def batch_search(
        self,
        table_name: str,
        query_vectors: np.ndarray,
        k: int,
        search_config: SearchConfig,
    ) -> List[SearchResult]:
        """
        Search for k nearest neighbors for multiple query vectors.

        Default implementation calls search() for each query.
        Subclasses may override for optimized batch search.

        Args:
            table_name: Name of the table to search
            query_vectors: Array of query vectors (shape: (n, dim))
            k: Number of nearest neighbors to return
            search_config: Search configuration

        Returns:
            List of SearchResult objects, one per query
        """
        results = []
        for query in query_vectors:
            result = self.search(table_name, query, k, search_config)
            results.append(result)
        return results
