#!/usr/bin/env python3
"""
Download benchmark datasets.

Downloads HDF5 datasets from ann-benchmarks and Texmex datasets from IRISA.
Verifies file integrity by checking HDF5 keys and printing vector counts.

Usage:
    # Download all datasets
    python scripts/download_datasets.py

    # Download specific datasets
    python scripts/download_datasets.py --datasets glove-100,dbpedia-openai

    # Download Texmex datasets (sift, gist)
    python scripts/download_datasets.py --datasets sift,gist
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Dataset definitions: name -> (url, local_path, format)
DATASETS = {
    "glove-100": {
        "url": "http://ann-benchmarks.com/glove-100-angular.hdf5",
        "path": "data/glove-100/glove-100-angular.hdf5",
        "format": "hdf5",
    },
    "dbpedia-openai": {
        "url": "https://storage.googleapis.com/ann-datasets/ann-benchmarks/dbpedia-openai-1000k-angular.hdf5",
        "path": "data/dbpedia-openai/dbpedia-openai-1000k-angular.hdf5",
        "format": "hdf5",
    },
    "sift": {
        "url": "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
        "path": "data/sift",
        "format": "fvecs",
    },
    "gist": {
        "url": "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz",
        "path": "data/gist",
        "format": "fvecs",
    },
}


def download_hdf5(name: str, url: str, local_path: str) -> bool:
    """Download an HDF5 dataset and verify its integrity."""
    path = Path(local_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        print(f"  {name}: already exists at {local_path}")
        return verify_hdf5(local_path)

    print(f"  {name}: downloading from {url}...")
    try:
        result = subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(path), url],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Download failed: {e}")
        # Clean up partial download
        if path.exists():
            path.unlink()
        return False

    return verify_hdf5(local_path)


def verify_hdf5(local_path: str) -> bool:
    """Verify HDF5 file has expected keys and print stats."""
    try:
        import h5py
    except ImportError:
        print("  WARNING: h5py not installed, skipping verification")
        return True

    try:
        with h5py.File(local_path, "r") as f:
            keys = list(f.keys())
            required = {"train", "test", "neighbors"}
            missing = required - set(keys)
            if missing:
                print(f"  ERROR: Missing HDF5 keys: {missing}")
                return False

            train = f["train"]
            test = f["test"]
            neighbors = f["neighbors"]
            print(
                f"  Verified: {train.shape[0]:,} vectors ({train.shape[1]}D), "
                f"{test.shape[0]:,} queries, "
                f"{neighbors.shape[1]} ground truth neighbors"
            )
            return True
    except Exception as e:
        print(f"  ERROR: HDF5 verification failed: {e}")
        return False


def download_texmex(name: str, url: str, local_path: str) -> bool:
    """Download and extract a Texmex dataset."""
    path = Path(local_path)

    # Check if already extracted
    base_file = path / f"{name}_base.fvecs"
    if base_file.exists():
        print(f"  {name}: already exists at {local_path}")
        return True

    path.mkdir(parents=True, exist_ok=True)
    tarball = path.parent / f"{name}.tar.gz"

    print(f"  {name}: downloading from {url}...")
    try:
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(tarball), url],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Download failed: {e}")
        return False

    print(f"  {name}: extracting...")
    try:
        subprocess.run(
            ["tar", "-xzf", str(tarball), "-C", str(path.parent)],
            check=True,
        )
        # Clean up tarball
        tarball.unlink()
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Extraction failed: {e}")
        return False

    # Verify files exist
    required = [f"{name}_base.fvecs", f"{name}_query.fvecs", f"{name}_groundtruth.ivecs"]
    for f in required:
        if not (path / f).exists():
            print(f"  ERROR: Missing file after extraction: {f}")
            return False

    print(f"  Verified: {local_path} extracted successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument(
        "--datasets",
        help=f"Comma-separated datasets to download (default: all). Available: {', '.join(DATASETS.keys())}",
        default=",".join(DATASETS.keys()),
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]

    # Validate
    for ds in datasets:
        if ds not in DATASETS:
            print(f"Unknown dataset: {ds}")
            print(f"Available: {', '.join(DATASETS.keys())}")
            sys.exit(1)

    print("=" * 60)
    print("DATASET DOWNLOADER")
    print("=" * 60)
    print(f"Datasets: {', '.join(datasets)}")
    print()

    results = {}
    for ds in datasets:
        info = DATASETS[ds]
        if info["format"] == "hdf5":
            results[ds] = download_hdf5(ds, info["url"], info["path"])
        else:
            results[ds] = download_texmex(ds, info["url"], info["path"])
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for ds, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {ds}: {status}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
