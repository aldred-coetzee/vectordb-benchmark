"""Data loading utilities for vector benchmark datasets."""

import struct
from pathlib import Path
from typing import Iterator, Tuple, Union

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None


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

    # Calculate file size for vector count estimation
    file_size = path.stat().st_size

    # Read all data in a single file open operation for efficiency
    with open(path, "rb") as f:
        # Read dimension from first vector
        dim = struct.unpack("i", f.read(4))[0]

        # Calculate number of vectors from file size
        # Each vector: 4 bytes (dim) + dim * 4 bytes (data)
        vector_size = 4 + dim * 4
        num_vectors = file_size // vector_size

        # Pre-allocate array for efficiency
        vectors = np.empty((num_vectors, dim), dtype=dtype)

        # Read first vector data (dimension already read)
        vectors[0] = np.frombuffer(f.read(dim * 4), dtype=dtype)

        # Read remaining vectors
        for i in range(1, num_vectors):
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


class TexmexDataset:
    """Generic Texmex dataset loader (SIFT, GIST, etc.)."""

    def __init__(self, dataset_path: str, name: str | None = None):
        """
        Initialize the dataset loader.

        Args:
            dataset_path: Path to the dataset directory containing the files
            name: Dataset name (e.g., 'sift', 'gist'). If None, inferred from directory name.
        """
        self.path = Path(dataset_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        # Infer dataset name from directory if not provided
        self.name = name or self.path.name
        self._base_vectors = None
        self._query_vectors = None
        self._ground_truth = None

    @property
    def base_vectors(self) -> np.ndarray:
        """Load and return base vectors."""
        if self._base_vectors is None:
            base_file = self.path / f"{self.name}_base.fvecs"
            print(f"Loading base vectors from {base_file}...")
            self._base_vectors = read_fvecs(str(base_file))
            print(f"Loaded {len(self._base_vectors):,} base vectors")
        return self._base_vectors

    @property
    def query_vectors(self) -> np.ndarray:
        """Load and return query vectors."""
        if self._query_vectors is None:
            query_file = self.path / f"{self.name}_query.fvecs"
            print(f"Loading query vectors from {query_file}...")
            self._query_vectors = read_fvecs(str(query_file))
            print(f"Loaded {len(self._query_vectors):,} query vectors")
        return self._query_vectors

    @property
    def ground_truth(self) -> np.ndarray:
        """Load and return ground truth nearest neighbor IDs."""
        if self._ground_truth is None:
            gt_file = self.path / f"{self.name}_groundtruth.ivecs"
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
            "name": self.name.upper(),
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


class AnnBenchmarkDataset:
    """HDF5 dataset loader for ann-benchmarks format (GloVe, DBpedia-OpenAI, etc.).

    HDF5 keys: 'train' (base vectors), 'test' (query vectors), 'neighbors' (ground truth).
    """

    def __init__(self, dataset_path: str, name: str | None = None):
        """
        Initialize the HDF5 dataset loader.

        Args:
            dataset_path: Path to the .hdf5 file
            name: Dataset name. If None, inferred from filename.
        """
        if h5py is None:
            raise ImportError(
                "h5py package not installed. "
                "Install with: pip install h5py"
            )

        self.path = Path(dataset_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        # Infer name from filename (e.g., "glove-100-angular.hdf5" -> "glove-100")
        if name is None:
            stem = self.path.stem  # "glove-100-angular"
            # Strip common suffixes like "-angular", "-euclidean"
            for suffix in ("-angular", "-euclidean", "-jaccard", "-hamming"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            name = stem
        self.name = name

        self._base_vectors: np.ndarray | None = None
        self._query_vectors: np.ndarray | None = None
        self._ground_truth: np.ndarray | None = None

    def _load_hdf5(self) -> None:
        """Load all data from HDF5 file."""
        print(f"Loading HDF5 dataset from {self.path}...")
        with h5py.File(self.path, "r") as f:
            self._base_vectors = np.array(f["train"], dtype=np.float32)
            self._query_vectors = np.array(f["test"], dtype=np.float32)
            self._ground_truth = np.array(f["neighbors"], dtype=np.int32)
        print(
            f"Loaded {len(self._base_vectors):,} base vectors, "
            f"{len(self._query_vectors):,} queries, "
            f"{self._ground_truth.shape[1]} ground truth neighbors"
        )

    @property
    def base_vectors(self) -> np.ndarray:
        """Load and return base vectors."""
        if self._base_vectors is None:
            self._load_hdf5()
        return self._base_vectors

    @property
    def query_vectors(self) -> np.ndarray:
        """Load and return query vectors."""
        if self._query_vectors is None:
            self._load_hdf5()
        return self._query_vectors

    @property
    def ground_truth(self) -> np.ndarray:
        """Load and return ground truth nearest neighbor IDs."""
        if self._ground_truth is None:
            self._load_hdf5()
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
            "name": self.name.upper(),
            "num_base_vectors": self.num_base_vectors,
            "num_query_vectors": self.num_query_vectors,
            "dimensions": self.dimensions,
            "ground_truth_k": self.ground_truth.shape[1],
        }

    def load_base_vectors(self) -> None:
        """Explicitly load base vectors into memory."""
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


def load_dataset(
    path: str, name: str | None = None
) -> Union[TexmexDataset, AnnBenchmarkDataset]:
    """
    Auto-detect dataset format and return appropriate loader.

    Args:
        path: Path to dataset directory (fvecs) or .hdf5 file
        name: Optional dataset name override

    Returns:
        TexmexDataset for directory-based fvecs, AnnBenchmarkDataset for HDF5
    """
    p = Path(path)
    if p.suffix == ".hdf5" or p.suffix == ".h5":
        return AnnBenchmarkDataset(path, name=name)
    else:
        return TexmexDataset(path, name=name)
