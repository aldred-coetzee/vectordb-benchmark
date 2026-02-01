"""Data loading utilities for vector benchmark datasets."""

import struct
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np


def _read_vecs(filename: str, dtype: np.dtype) -> np.ndarray:
    """
    Read vectors from a .fvecs or .ivecs file.

    Format: Each vector is prefixed with a 4-byte int32 dimension,
    followed by dim * 4 bytes of data.

    Args:
        filename: Path to the vector file
        dtype: NumPy dtype for the vector elements (np.float32 or np.int32)

    Returns:
        numpy array of shape (num_vectors, dimension) with the specified dtype
    """
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    # Read dimension from first vector to calculate count
    with open(path, "rb") as f:
        dim = struct.unpack("i", f.read(4))[0]

    # Calculate number of vectors from file size
    # Each vector: 4 bytes (dim) + dim * 4 bytes (data)
    file_size = path.stat().st_size
    vector_size = 4 + dim * 4
    num_vectors = file_size // vector_size

    # Pre-allocate array for efficiency
    vectors = np.empty((num_vectors, dim), dtype=dtype)

    with open(path, "rb") as f:
        for i in range(num_vectors):
            # Read and verify dimension
            vec_dim = struct.unpack("i", f.read(4))[0]
            if vec_dim != dim:
                raise ValueError(
                    f"Inconsistent dimension at vector {i}: expected {dim}, got {vec_dim}"
                )

            # Read vector data directly into pre-allocated array
            vectors[i] = np.frombuffer(f.read(dim * 4), dtype=dtype)

    return vectors


def read_fvecs(filename: str) -> np.ndarray:
    """
    Read vectors from a .fvecs file.

    Format: Each vector is prefixed with a 4-byte int32 dimension,
    followed by dim * 4 bytes of float32 data.

    Args:
        filename: Path to the .fvecs file

    Returns:
        numpy array of shape (num_vectors, dimension) with dtype float32
    """
    return _read_vecs(filename, np.float32)


def read_ivecs(filename: str) -> np.ndarray:
    """
    Read vectors from a .ivecs file.

    Format: Each vector is prefixed with a 4-byte int32 dimension,
    followed by dim * 4 bytes of int32 data.

    Args:
        filename: Path to the .ivecs file

    Returns:
        numpy array of shape (num_vectors, dimension) with dtype int32
    """
    return _read_vecs(filename, np.int32)


class SIFTDataset:
    """SIFT-1M dataset loader."""

    def __init__(self, dataset_path: str):
        """
        Initialize the SIFT dataset loader.

        Args:
            dataset_path: Path to the sift directory containing the dataset files
        """
        self.path = Path(dataset_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        self._base_vectors = None
        self._query_vectors = None
        self._ground_truth = None

    @property
    def base_vectors(self) -> np.ndarray:
        """Load and return base vectors (1M vectors, 128 dims)."""
        if self._base_vectors is None:
            base_file = self.path / "sift_base.fvecs"
            print(f"Loading base vectors from {base_file}...")
            self._base_vectors = read_fvecs(str(base_file))
            print(f"Loaded {len(self._base_vectors):,} base vectors")
        return self._base_vectors

    @property
    def query_vectors(self) -> np.ndarray:
        """Load and return query vectors (10K vectors, 128 dims)."""
        if self._query_vectors is None:
            query_file = self.path / "sift_query.fvecs"
            print(f"Loading query vectors from {query_file}...")
            self._query_vectors = read_fvecs(str(query_file))
            print(f"Loaded {len(self._query_vectors):,} query vectors")
        return self._query_vectors

    @property
    def ground_truth(self) -> np.ndarray:
        """Load and return ground truth (10K x 100 nearest neighbor IDs)."""
        if self._ground_truth is None:
            gt_file = self.path / "sift_groundtruth.ivecs"
            print(f"Loading ground truth from {gt_file}...")
            self._ground_truth = read_ivecs(str(gt_file))
            print(f"Loaded ground truth with shape {self._ground_truth.shape}")
        return self._ground_truth

    @property
    def dimensions(self) -> int:
        """Return vector dimensionality."""
        return self.base_vectors.shape[1]

    @property
    def num_base_vectors(self) -> int:
        """Return number of base vectors."""
        return self.base_vectors.shape[0]

    @property
    def num_query_vectors(self) -> int:
        """Return number of query vectors."""
        return self.query_vectors.shape[0]

    def get_info(self) -> dict:
        """Return dataset information."""
        return {
            "name": "SIFT-1M",
            "num_base_vectors": self.num_base_vectors,
            "num_query_vectors": self.num_query_vectors,
            "dimensions": self.dimensions,
            "ground_truth_k": self.ground_truth.shape[1],
        }

    def load_base_vectors(self) -> None:
        """
        Explicitly load base vectors into memory.

        This triggers lazy loading of the base vectors if not already loaded.
        Useful for ensuring data is loaded before timing-sensitive operations.
        """
        # Access the property to trigger lazy loading
        _ = self.base_vectors

    def get_batches(
        self, batch_size: int = 50000
    ) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
        """
        Yield batches of base vectors with their IDs.

        Args:
            batch_size: Number of vectors per batch

        Yields:
            Tuple of (start_id, ids_array, vectors_array)
        """
        num_vectors = self.num_base_vectors
        for start_idx in range(0, num_vectors, batch_size):
            end_idx = min(start_idx + batch_size, num_vectors)
            ids = np.arange(start_idx, end_idx, dtype=np.int64)
            vectors = self.base_vectors[start_idx:end_idx]
            yield start_idx, ids, vectors
