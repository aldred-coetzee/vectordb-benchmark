#!/usr/bin/env python3
"""Download and extract the SIFT-1M dataset."""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path


SIFT_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
EXPECTED_FILES = [
    "sift_base.fvecs",
    "sift_query.fvecs",
    "sift_groundtruth.ivecs",
    "sift_learn.fvecs",
]


def download_progress(block_num: int, block_size: int, total_size: int) -> None:
    """Display download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\rDownloading: {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
        sys.stdout.flush()
    else:
        downloaded_mb = downloaded / (1024 * 1024)
        sys.stdout.write(f"\rDownloading: {downloaded_mb:.1f} MB")
        sys.stdout.flush()


def download_sift(output_dir: str, force: bool = False) -> Path:
    """
    Download and extract the SIFT-1M dataset.

    Args:
        output_dir: Directory to store the dataset
        force: If True, re-download even if files exist

    Returns:
        Path to the extracted dataset directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sift_dir = output_path / "sift"
    tar_path = output_path / "sift.tar.gz"

    # Check if already downloaded and extracted
    if not force and sift_dir.exists():
        all_present = all((sift_dir / f).exists() for f in EXPECTED_FILES)
        if all_present:
            print(f"SIFT-1M dataset already exists at {sift_dir}")
            return sift_dir

    # Download
    print(f"Downloading SIFT-1M dataset from {SIFT_URL}")
    print("This may take a while (~160 MB)...")

    try:
        urllib.request.urlretrieve(SIFT_URL, tar_path, download_progress)
        print("\nDownload complete.")
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        raise

    # Extract (using filter='data' to prevent path traversal attacks - CVE-2007-4559)
    print(f"Extracting to {output_path}...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=output_path, filter='data')
        print("Extraction complete.")
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        raise

    # Cleanup tar file
    if tar_path.exists():
        os.remove(tar_path)
        print("Cleaned up archive file.")

    # Verify files
    print("Verifying extracted files...")
    for filename in EXPECTED_FILES:
        filepath = sift_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Expected file not found: {filepath}")
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  {filename}: {size_mb:.1f} MB")

    print(f"\nSIFT-1M dataset ready at {sift_dir}")
    return sift_dir


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Download SIFT-1M dataset")
    parser.add_argument(
        "--output", "-o",
        default="datasets",
        help="Output directory (default: datasets)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if files exist"
    )

    args = parser.parse_args()
    download_sift(args.output, args.force)


if __name__ == "__main__":
    main()
