#!/usr/bin/env python3
"""
Generate small dev datasets for fast iteration during development.

Creates *-dev variants of existing datasets with:
  - 10,000 base vectors (from first 10K of full dataset)
  - 100 query vectors (from first 100 of full dataset)
  - Recomputed exact ground truth (brute-force via FAISS)

These run in seconds instead of 30+ minutes, exercising the same
code paths as full datasets. NOT for actual benchmarking.

Usage:
    python scripts/generate_dev_datasets.py                # All available datasets
    python scripts/generate_dev_datasets.py --datasets sift # Specific dataset
"""

import argparse
import struct
import sys
from pathlib import Path

import faiss
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmark.data_loader import read_fvecs


NUM_BASE_VECTORS = 10_000
NUM_QUERY_VECTORS = 100
GROUND_TRUTH_K = 100  # Match typical ground truth depth


def write_fvecs(filename: str, vectors: np.ndarray) -> None:
    """Write vectors to .fvecs format."""
    vectors = vectors.astype(np.float32)
    with open(filename, "wb") as f:
        for vec in vectors:
            dim = len(vec)
            f.write(struct.pack("i", dim))
            f.write(vec.tobytes())


def write_ivecs(filename: str, vectors: np.ndarray) -> None:
    """Write vectors to .ivecs format."""
    vectors = vectors.astype(np.int32)
    with open(filename, "wb") as f:
        for vec in vectors:
            dim = len(vec)
            f.write(struct.pack("i", dim))
            f.write(vec.tobytes())


def compute_ground_truth(
    base_vectors: np.ndarray, query_vectors: np.ndarray, k: int
) -> np.ndarray:
    """Compute exact k-NN ground truth using FAISS brute-force."""
    dim = base_vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(base_vectors)
    _, indices = index.search(query_vectors, k)
    return indices.astype(np.int32)


def generate_dev_dataset(source_dir: Path, data_root: Path) -> None:
    """Generate a dev dataset from a full dataset."""
    dataset_name = source_dir.name
    dev_name = f"{dataset_name}-dev"
    dev_dir = data_root / dev_name

    print(f"\n{'='*60}")
    print(f"Generating {dev_name} from {dataset_name}")
    print(f"{'='*60}")

    # Check source files exist
    base_file = source_dir / f"{dataset_name}_base.fvecs"
    query_file = source_dir / f"{dataset_name}_query.fvecs"

    if not base_file.exists():
        print(f"  Skipping: {base_file} not found")
        return
    if not query_file.exists():
        print(f"  Skipping: {query_file} not found")
        return

    # Load source data
    print(f"  Loading base vectors from {base_file}...")
    base_vectors = read_fvecs(str(base_file))
    print(f"  Loaded {len(base_vectors):,} vectors ({base_vectors.shape[1]}D)")

    print(f"  Loading query vectors from {query_file}...")
    query_vectors = read_fvecs(str(query_file))
    print(f"  Loaded {len(query_vectors):,} queries")

    # Subset
    base_subset = base_vectors[:NUM_BASE_VECTORS]
    query_subset = query_vectors[:NUM_QUERY_VECTORS]
    print(f"  Subset: {len(base_subset):,} base, {len(query_subset):,} queries")

    # Compute ground truth
    print(f"  Computing ground truth (k={GROUND_TRUTH_K})...")
    ground_truth = compute_ground_truth(base_subset, query_subset, GROUND_TRUTH_K)

    # Write dev dataset
    dev_dir.mkdir(parents=True, exist_ok=True)

    write_fvecs(str(dev_dir / f"{dev_name}_base.fvecs"), base_subset)
    write_fvecs(str(dev_dir / f"{dev_name}_query.fvecs"), query_subset)
    write_ivecs(str(dev_dir / f"{dev_name}_groundtruth.ivecs"), ground_truth)

    print(f"  Written to {dev_dir}/")
    print(f"    {dev_name}_base.fvecs     ({len(base_subset):,} x {base_subset.shape[1]})")
    print(f"    {dev_name}_query.fvecs    ({len(query_subset):,} x {query_subset.shape[1]})")
    print(f"    {dev_name}_groundtruth.ivecs ({len(ground_truth):,} x {GROUND_TRUTH_K})")


def main():
    parser = argparse.ArgumentParser(
        description="Generate small dev datasets for fast development iteration"
    )
    parser.add_argument(
        "--datasets",
        help="Comma-separated list of source datasets (default: all found in data/)",
        default=None,
    )
    args = parser.parse_args()

    data_root = Path(__file__).parent.parent / "data"
    if not data_root.exists():
        print(f"Data directory not found: {data_root}")
        sys.exit(1)

    # Find source datasets (directories without -dev suffix)
    if args.datasets:
        dataset_names = [d.strip() for d in args.datasets.split(",")]
        source_dirs = [data_root / name for name in dataset_names]
    else:
        source_dirs = [
            d for d in sorted(data_root.iterdir())
            if d.is_dir() and not d.name.endswith("-dev")
        ]

    if not source_dirs:
        print("No source datasets found in data/")
        sys.exit(1)

    print(f"Dev dataset config: {NUM_BASE_VECTORS:,} vectors, {NUM_QUERY_VECTORS} queries")

    for source_dir in source_dirs:
        generate_dev_dataset(source_dir, data_root)

    print(f"\nDone. Add dev datasets to benchmark.yaml to use them.")


if __name__ == "__main__":
    main()
