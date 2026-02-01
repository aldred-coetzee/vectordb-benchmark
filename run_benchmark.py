#!/usr/bin/env python3
"""
Vector Database Benchmark Tool

Benchmarks vector database performance measuring ingest speed, query speed,
and recall accuracy.

Usage:
    python run_benchmark.py --database kdbai --dataset datasets/sift \
        --container kdbai-bench --cpus 8 --memory 32
"""

import argparse
import sys
from pathlib import Path


def get_client(database: str):
    """
    Get the appropriate database client.

    Args:
        database: Database name (e.g., 'kdbai', 'faiss', 'qdrant', 'pgvector')

    Returns:
        Database client instance
    """
    if database.lower() == "kdbai":
        from benchmark.clients.kdbai_client import KDBAIClient
        return KDBAIClient()
    elif database.lower() == "faiss":
        from benchmark.clients.faiss_client import FAISSClient
        return FAISSClient()
    elif database.lower() == "qdrant":
        from benchmark.clients.qdrant_client import QdrantClient
        return QdrantClient()
    elif database.lower() == "pgvector":
        from benchmark.clients.pgvector_client import PGVectorClient
        return PGVectorClient()
    elif database.lower() == "weaviate":
        from benchmark.clients.weaviate_client import WeaviateClient
        return WeaviateClient()
    else:
        raise ValueError(
            f"Unsupported database: {database}. "
            f"Supported: kdbai, faiss, qdrant, pgvector, weaviate"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Vector Database Benchmark Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run benchmark with KDB.AI
    python run_benchmark.py --database kdbai --dataset datasets/sift \\
        --container kdbai-bench

    # Download dataset first
    python datasets/download_sift.py --output datasets

    # Run without Docker monitoring
    python run_benchmark.py --database kdbai --dataset datasets/sift
        """,
    )

    parser.add_argument(
        "--database", "-d",
        required=True,
        choices=["kdbai", "faiss", "qdrant", "pgvector", "weaviate"],
        help="Database to benchmark (kdbai, faiss, qdrant, pgvector, or weaviate)",
    )

    parser.add_argument(
        "--dataset", "-s",
        required=True,
        help="Path to dataset directory (e.g., datasets/sift)",
    )

    parser.add_argument(
        "--container", "-c",
        default=None,
        help="Docker container name for resource monitoring",
    )

    parser.add_argument(
        "--endpoint", "-e",
        default=None,
        help="Database endpoint URL (default: kdbai=:8082, qdrant=:6333, pgvector=:5432, weaviate=:8080)",
    )

    parser.add_argument(
        "--cpus",
        type=int,
        default=0,
        help="Docker CPU limit (for report metadata only)",
    )

    parser.add_argument(
        "--memory",
        type=int,
        default=0,
        help="Docker memory limit in GB (for report metadata only)",
    )

    parser.add_argument(
        "--output", "-o",
        default="results",
        help="Output directory for results (default: results)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Batch size for vector insertion (default: 50000)",
    )

    parser.add_argument(
        "--ef-search",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128, 256],
        help="HNSW efSearch values to test (default: 8 16 32 64 128 256)",
    )

    args = parser.parse_args()

    # Validate dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        print("Run 'python datasets/download_sift.py' to download the dataset.")
        sys.exit(1)

    # Check for required files
    required_files = ["sift_base.fvecs", "sift_query.fvecs", "sift_groundtruth.ivecs"]
    for filename in required_files:
        if not (dataset_path / filename).exists():
            print(f"Error: Required file not found: {dataset_path / filename}")
            print("Run 'python datasets/download_sift.py' to download the dataset.")
            sys.exit(1)

    # Set default endpoint based on database type
    if args.endpoint is None:
        if args.database.lower() == "kdbai":
            args.endpoint = "http://localhost:8082"
        elif args.database.lower() == "qdrant":
            args.endpoint = "http://localhost:6333"
        elif args.database.lower() == "pgvector":
            args.endpoint = "localhost:5432"
        elif args.database.lower() == "weaviate":
            args.endpoint = "http://localhost:8080"
        else:
            args.endpoint = "http://localhost:8080"

    print("=" * 80)
    print("VECTOR DATABASE BENCHMARK")
    print("=" * 80)
    print(f"Database:    {args.database}")
    print(f"Dataset:     {args.dataset}")
    print(f"Endpoint:    {args.endpoint if args.database.lower() != 'faiss' else 'N/A (in-process)'}")
    print(f"Container:   {args.container or 'None (no monitoring)'}")
    print(f"Output:      {args.output}")
    print(f"efSearch:    {args.ef_search}")
    print("=" * 80)

    # Import here to avoid issues if dependencies aren't installed
    from benchmark.data_loader import SIFTDataset
    from benchmark.docker_monitor import DockerMonitor
    from benchmark.runner import BenchmarkRunner
    from benchmark.report import generate_full_report

    # Load dataset
    print("\nLoading dataset...")
    try:
        dataset = SIFTDataset(str(dataset_path))
        # Force loading to validate files
        _ = dataset.dimensions
        print(f"Dataset: {dataset.num_base_vectors:,} vectors, {dataset.dimensions} dimensions")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Initialize Docker monitor (optional, not used for in-process databases like FAISS)
    monitor = None
    if args.database.lower() == "faiss":
        print("\nFAISS runs in-process - skipping Docker container monitoring")
    elif args.container:
        print(f"\nConnecting to Docker container '{args.container}'...")
        try:
            monitor = DockerMonitor(args.container)
            monitor.connect()
            limits = monitor.get_container_limits()
            print(f"  Memory limit: {limits.get('memory_limit_gb', 0):.1f} GB")
            print(f"  CPU limit: {limits.get('cpu_limit', 0):.1f} cores")
        except Exception as e:
            print(f"Warning: Could not connect to Docker container: {e}")
            print("Continuing without resource monitoring...")
            monitor = None

    # Initialize database client
    print(f"\nConnecting to {args.database}...")
    try:
        client = get_client(args.database)
        client.connect(endpoint=args.endpoint)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

    # Run benchmark
    try:
        runner = BenchmarkRunner(client, dataset, monitor)
        runner.batch_size = args.batch_size

        results = runner.run_full_benchmark(
            hnsw_ef_search_values=args.ef_search,
        )

        # Override Docker limits with CLI args if provided
        if args.cpus > 0:
            results.docker_cpu_limit = args.cpus
        if args.memory > 0:
            results.docker_memory_limit_gb = args.memory

        # Generate reports
        generate_full_report(results, args.output)

        print("\nBenchmark complete!")
        print(f"Results saved to: {args.output}/")

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        print("\nCleaning up...")
        try:
            client.disconnect()
        except Exception:
            pass
        if monitor:
            try:
                monitor.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
