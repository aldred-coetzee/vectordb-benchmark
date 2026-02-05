#!/usr/bin/env python3
"""
Vector Database Benchmark Tool

Benchmarks vector database performance measuring ingest speed, query speed,
and recall accuracy.

Usage:
    # New config-based interface
    python run_benchmark.py --config configs/kdbai.yaml --benchmark benchmark.yaml

    # With optional filters
    python run_benchmark.py --config configs/kdbai.yaml --benchmark benchmark.yaml \
        --dataset sift --indexes flat,hnsw

    # Legacy interface (still supported)
    python run_benchmark.py --database kdbai --dataset data/sift \
        --container kdbai-bench --cpus 8 --memory 32
"""

# Configure warning filters
import warnings

# Ignore noisy pandas internal warnings (from kdbai-client and other libs)
warnings.filterwarnings("ignore", message=".*BlockManager.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*concatenation with empty.*", category=FutureWarning)

# Show all other warnings once
warnings.filterwarnings("once")

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from benchmark.config import load_yaml_config


def parse_memory_string(memory: Any) -> float:
    """
    Parse a memory string (e.g., "8g") into GB as a float.

    Args:
        memory: Memory value as string (e.g., "8g", "16G") or numeric

    Returns:
        Memory in GB as float, or 0.0 if parsing fails
    """
    if isinstance(memory, (int, float)):
        return float(memory)

    if isinstance(memory, str):
        memory_lower = memory.lower().strip()
        if memory_lower.endswith("g"):
            try:
                return float(memory_lower[:-1])
            except ValueError:
                pass
        elif memory_lower.endswith("m"):
            try:
                return float(memory_lower[:-1]) / 1024
            except ValueError:
                pass

    return 0.0


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
    elif database.lower() == "milvus":
        from benchmark.clients.milvus_client import MilvusClient
        return MilvusClient()
    elif database.lower() == "lancedb":
        from benchmark.clients.lancedb_client import LanceDBClient
        return LanceDBClient()
    elif database.lower() == "chroma":
        from benchmark.clients.chroma_client import ChromaClient
        return ChromaClient()
    elif database.lower() == "redis":
        from benchmark.clients.redis_client import RedisClient
        return RedisClient()
    else:
        raise ValueError(
            f"Unsupported database: {database}. "
            f"Supported: kdbai, faiss, qdrant, pgvector, weaviate, milvus, lancedb, chroma, redis"
        )


def get_supported_indexes(client) -> Set[str]:
    """
    Get the set of index types supported by a client.

    Args:
        client: Database client instance

    Returns:
        Set of supported index type names (lowercase)
    """
    # Check if client has a supported_indexes property/method
    if hasattr(client, "supported_indexes"):
        indexes = client.supported_indexes
        if callable(indexes):
            indexes = indexes()
        return set(idx.lower() for idx in indexes)

    # Default: assume flat and hnsw are supported
    return {"flat", "hnsw"}


def filter_indexes(
    requested: List[str],
    supported: Set[str],
    database_name: str,
) -> List[str]:
    """
    Filter requested indexes to only those supported by the database.

    Args:
        requested: List of requested index types
        supported: Set of supported index types
        database_name: Name of the database (for logging)

    Returns:
        List of indexes that are both requested and supported
    """
    filtered = []
    for idx in requested:
        idx_lower = idx.lower()
        if idx_lower in supported:
            filtered.append(idx_lower)
        else:
            print(f"  Index type '{idx}' not supported by {database_name}, skipping")

    return filtered


def run_with_config(
    config: Dict[str, Any],
    benchmark_config: Dict[str, Any],
    dataset_filter: Optional[str],
    indexes_filter: Optional[List[str]],
    output_dir: str,
    container_name: Optional[str],
    skip_docker: bool = False,
    keep_container: bool = False,
) -> None:
    """
    Run benchmark using YAML configuration.

    Args:
        config: Database-specific configuration
        benchmark_config: Shared benchmark configuration
        dataset_filter: Optional dataset name to run (None = all)
        indexes_filter: Optional list of index types to run (None = all)
        output_dir: Directory for output files
        container_name: Optional Docker container name for monitoring
        skip_docker: If True, skip Docker container management (use existing)
        keep_container: If True, don't stop container after benchmark
    """
    import time
    from datetime import datetime
    from benchmark.data_loader import load_dataset
    from benchmark.docker_monitor import DockerMonitor
    from benchmark.docker_manager import DockerManager
    from benchmark.runner import BenchmarkRunner
    from benchmark.report import generate_full_report
    from benchmark.db import BenchmarkDatabase

    # Record start time
    benchmark_start_time = time.time()
    start_time_iso = datetime.now().isoformat()
    print(f"\nBenchmark started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Extract database info from config
    db_config = config.get("database", {})
    database_name = db_config.get("name", "unknown")
    db_version = db_config.get("version")
    endpoint = db_config.get("endpoint")

    # Extract container settings if present
    container_config = config.get("container", {})
    cpus = container_config.get("cpus", 0)
    memory = container_config.get("memory", "0g")

    # Parse memory string (e.g., "8g" -> 8.0)
    memory_gb = parse_memory_string(memory)

    # Initialize Docker manager
    docker_manager = None
    if database_name.lower() != "faiss" and container_config:
        docker_manager = DockerManager(config, database_name)

        if not skip_docker and docker_manager.has_container:
            # Start container
            try:
                docker_manager.start_container()
                if not docker_manager.wait_for_ready():
                    print("Error: Container failed to become ready")
                    sys.exit(1)
            except Exception as e:
                print(f"Error starting container: {e}")
                sys.exit(1)
        elif skip_docker:
            print("Skipping Docker container management (--skip-docker)")

    # Extract search settings from benchmark config
    search_config = benchmark_config.get("search", {})
    batch_size = search_config.get("batch_size", 50000)
    num_queries = search_config.get("num_queries", 10000)
    warmup_queries = search_config.get("warmup", 100)

    # Determine which datasets to run
    datasets_config = benchmark_config.get("datasets", {})
    if dataset_filter:
        if dataset_filter not in datasets_config:
            print(f"Error: Dataset '{dataset_filter}' not found in benchmark config")
            print(f"Available datasets: {list(datasets_config.keys())}")
            sys.exit(1)
        datasets_to_run = {dataset_filter: datasets_config[dataset_filter]}
    else:
        datasets_to_run = datasets_config

    # Determine which indexes to run
    indexes_config = benchmark_config.get("indexes", {})
    all_indexes = list(indexes_config.keys())

    if indexes_filter:
        requested_indexes = [idx.strip() for idx in indexes_filter]
        # Validate requested indexes exist in config
        for idx in requested_indexes:
            if idx.lower() not in [i.lower() for i in all_indexes]:
                print(f"Warning: Index '{idx}' not found in benchmark config, skipping")
    else:
        requested_indexes = all_indexes

    # Initialize database client
    client = None
    db = None
    connect_params = config.get("params", {})

    try:
        print(f"\nConnecting to {database_name}...")
        try:
            client = get_client(database_name)
            # Retry connection — container may need time after health check passes
            # (e.g., Redis loading dataset, pgvector creating extensions)
            max_retries = 8
            for attempt in range(max_retries):
                try:
                    if endpoint:
                        client.connect(endpoint=endpoint, **connect_params)
                    else:
                        client.connect(**connect_params)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = min(2 ** attempt, 10)  # 1,2,4,8,10,10,10s = 45s total
                        print(f"  Connection attempt {attempt + 1} failed: {e}")
                        print(f"  Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        raise
        except Exception as e:
            print(f"Error connecting to database: {e}")
            raise

        # Detect version from the running server/library
        detected_version = client.get_version()
        if detected_version and detected_version != "unknown":
            db_version = detected_version
            print(f"  Detected version: {db_version}")

        # Filter indexes by what the client supports
        supported = get_supported_indexes(client)
        indexes_to_run = filter_indexes(requested_indexes, supported, database_name)

        if not indexes_to_run:
            print(f"\n{database_name} does not support the requested index types. Nothing to benchmark.")
            if not supported:
                print(f"  This database uses different index types (e.g., IVF-based) — it will be included in future benchmarks.")
            else:
                print(f"  Supported indexes: {supported}")
            return

        print(f"Indexes to run: {indexes_to_run}")

        # Initialize SQLite database
        db = BenchmarkDatabase()

        # Run benchmarks for each dataset
        for dataset_name, dataset_info in datasets_to_run.items():
            dataset_path = Path(dataset_info.get("path", f"data/{dataset_name}"))
            dataset_format = dataset_info.get("format", "fvecs")
            dataset_metric = dataset_info.get("metric", "L2")

            # Validate dataset exists
            if dataset_format == "hdf5":
                # HDF5: path is a file
                if not dataset_path.exists():
                    print(f"Warning: HDF5 file not found: {dataset_path}, skipping")
                    continue
            else:
                # fvecs: path is a directory
                if not dataset_path.exists():
                    print(f"Warning: Dataset path not found: {dataset_path}, skipping")
                    continue
                ds_name = dataset_path.name
                required_files = [f"{ds_name}_base.fvecs", f"{ds_name}_query.fvecs", f"{ds_name}_groundtruth.ivecs"]
                missing_files = [f for f in required_files if not (dataset_path / f).exists()]
                if missing_files:
                    print(f"Warning: Missing files in {dataset_path}: {missing_files}, skipping")
                    continue

            print(f"\n{'='*80}")
            print(f"BENCHMARKING: {database_name} with {dataset_name} dataset (metric={dataset_metric})")
            print(f"{'='*80}")

            # Load dataset
            print("\nLoading dataset...")
            try:
                dataset = load_dataset(str(dataset_path))
                dataset.load_base_vectors()  # Explicitly trigger lazy loading
                print(f"Dataset: {dataset.num_base_vectors:,} vectors, {dataset.dimensions} dimensions")
            except Exception as e:
                print(f"Error loading dataset: {e}, skipping")
                continue

            # Initialize Docker monitor
            monitor = None
            effective_container_name = container_name or (
                docker_manager.container_name if docker_manager else None
            )
            if database_name.lower() == "faiss":
                print("\nFAISS runs in-process - skipping Docker container monitoring")
            elif effective_container_name:
                print(f"\nConnecting to Docker container '{effective_container_name}'...")
                try:
                    monitor = DockerMonitor(effective_container_name)
                    monitor.connect()
                    limits = monitor.get_container_limits()
                    print(f"  Memory limit: {limits.get('memory_limit_gb', 0):.1f} GB")
                    print(f"  CPU limit: {limits.get('cpu_limit', 0):.1f} cores")
                except Exception as e:
                    print(f"Warning: Could not connect to Docker container: {e}")
                    monitor = None

            # Prepare index configs
            hnsw_config = indexes_config.get("hnsw", {})
            ef_search_values = hnsw_config.get("efSearch", [8, 16, 32, 64, 128, 256])

            # Run benchmark
            try:
                runner = BenchmarkRunner(
                    client, dataset, monitor,
                    batch_size=batch_size,
                    warmup_queries=warmup_queries,
                )

                # Run the benchmark with filtered indexes
                results = runner.run_full_benchmark(
                    hnsw_ef_search_values=ef_search_values,
                    indexes_to_run=indexes_to_run,
                    metric=dataset_metric,
                )

                # Override Docker limits with config values if provided
                if cpus > 0:
                    results.docker_cpu_limit = float(cpus)
                if memory_gb > 0:
                    results.docker_memory_limit_gb = memory_gb

                # Generate reports (CSV and console)
                generate_full_report(results, output_dir)

                # Calculate end time and duration
                benchmark_end_time = time.time()
                end_time_iso = datetime.now().isoformat()
                duration_seconds = benchmark_end_time - benchmark_start_time

                # Save to SQLite
                run_id = db.save_benchmark_results(
                    results=results,
                    config=config,
                    benchmark_config=benchmark_config,
                    batch_size=batch_size,
                    num_queries=num_queries,
                    db_version=db_version,
                    start_time=start_time_iso,
                    end_time=end_time_iso,
                    duration_seconds=duration_seconds,
                )
                print(f"\nResults saved to SQLite (run_id: {run_id})")

                print(f"\nBenchmark ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Total duration: {duration_seconds:.1f}s ({duration_seconds/60:.1f} minutes)")
                print(f"Results saved to: {output_dir}/")

            except KeyboardInterrupt:
                print("\nBenchmark interrupted by user.")
                break
            except Exception as e:
                print(f"\nError during benchmark: {e}")
                traceback.print_exc()
            finally:
                if monitor:
                    try:
                        monitor.disconnect()
                    except Exception:
                        pass

    finally:
        # Cleanup - always runs even if there's an error
        print("\nCleaning up...")
        if client:
            try:
                client.disconnect()
            except Exception:
                pass
        if db:
            db.close()

        # Stop Docker container if we started it and --keep-container not specified
        if docker_manager and docker_manager.has_container and not skip_docker and not keep_container:
            docker_manager.stop_container()
            docker_manager.close()
        elif docker_manager:
            if keep_container:
                print(f"Keeping container '{docker_manager.container_name}' running (--keep-container)")
            docker_manager.close()


def run_legacy(args: argparse.Namespace) -> None:
    """
    Run benchmark using legacy CLI arguments.

    Args:
        args: Parsed command-line arguments
    """
    from benchmark.data_loader import TexmexDataset
    from benchmark.docker_monitor import DockerMonitor
    from benchmark.runner import BenchmarkRunner
    from benchmark.report import generate_full_report
    from benchmark.db import BenchmarkDatabase

    # Validate dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        print("Run 'python data/download_sift.py' to download the dataset.")
        sys.exit(1)

    # Check for required files (named after the dataset directory)
    ds_name = dataset_path.name
    required_files = [f"{ds_name}_base.fvecs", f"{ds_name}_query.fvecs", f"{ds_name}_groundtruth.ivecs"]
    for filename in required_files:
        if not (dataset_path / filename).exists():
            print(f"Error: Required file not found: {dataset_path / filename}")
            print(f"Expected files: {', '.join(required_files)}")
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
        elif args.database.lower() == "milvus":
            args.endpoint = "localhost:19530"
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

    # Load dataset
    print("\nLoading dataset...")
    try:
        dataset = TexmexDataset(str(dataset_path))
        dataset.load_base_vectors()  # Explicitly trigger lazy loading
        print(f"Dataset: {dataset.num_base_vectors:,} vectors, {dataset.dimensions} dimensions")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Initialize Docker monitor
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
            monitor = None

    # Initialize database client
    print(f"\nConnecting to {args.database}...")
    client = get_client(args.database)
    max_retries = 8
    for attempt in range(max_retries):
        try:
            client.connect(endpoint=args.endpoint)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait = min(2 ** attempt, 10)  # 1,2,4,8,10,10,10s = 45s total
                print(f"  Connection attempt {attempt + 1} failed: {e}")
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"Error connecting to database after {max_retries} attempts: {e}")
                sys.exit(1)

    # Initialize SQLite database
    db = BenchmarkDatabase()

    # Run benchmark
    try:
        runner = BenchmarkRunner(
            client, dataset, monitor,
            batch_size=args.batch_size,
            warmup_queries=100,  # Default warmup for legacy mode
        )

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

        # Save to SQLite
        run_id = db.save_benchmark_results(
            results=results,
            batch_size=args.batch_size,
            num_queries=10000,  # Default from original
        )
        print(f"\nResults saved to SQLite (run_id: {run_id})")

        print("\nBenchmark complete!")
        print(f"Results saved to: {args.output}/")

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during benchmark: {e}")
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
        db.close()


def main():
    parser = argparse.ArgumentParser(
        description="Vector Database Benchmark Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # New config-based interface
    python run_benchmark.py --config configs/kdbai.yaml --benchmark benchmark.yaml

    # With filters
    python run_benchmark.py --config configs/kdbai.yaml --benchmark benchmark.yaml \\
        --dataset sift --indexes flat,hnsw

    # Legacy interface
    python run_benchmark.py --database kdbai --dataset data/sift \\
        --container kdbai-bench

    # Download dataset first
    python data/download_sift.py --output data
        """,
    )

    # Config-based arguments (new interface)
    config_group = parser.add_argument_group("Config-based interface (recommended)")
    config_group.add_argument(
        "--config", "-c",
        help="Path to database-specific YAML config (e.g., configs/kdbai.yaml)",
    )
    config_group.add_argument(
        "--benchmark", "-b",
        help="Path to shared benchmark YAML config (default: benchmark.yaml)",
    )
    config_group.add_argument(
        "--indexes", "-i",
        help="Comma-separated list of index types to run (e.g., flat,hnsw)",
    )

    # Legacy arguments
    legacy_group = parser.add_argument_group("Legacy interface")
    legacy_group.add_argument(
        "--database", "-d",
        choices=["kdbai", "faiss", "qdrant", "pgvector", "weaviate", "milvus"],
        help="Database to benchmark",
    )
    legacy_group.add_argument(
        "--endpoint", "-e",
        default=None,
        help="Database endpoint URL",
    )
    legacy_group.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Batch size for vector insertion (default: 50000)",
    )
    legacy_group.add_argument(
        "--ef-search",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128, 256],
        help="HNSW efSearch values to test (default: 8 16 32 64 128 256)",
    )

    # Common arguments
    common_group = parser.add_argument_group("Common options")
    common_group.add_argument(
        "--dataset", "-s",
        help="Dataset name (config mode) or path (legacy mode)",
    )
    common_group.add_argument(
        "--container",
        default=None,
        help="Docker container name for resource monitoring",
    )
    common_group.add_argument(
        "--cpus",
        type=int,
        default=0,
        help="Docker CPU limit (for report metadata only)",
    )
    common_group.add_argument(
        "--memory",
        type=int,
        default=0,
        help="Docker memory limit in GB (for report metadata only)",
    )
    common_group.add_argument(
        "--output", "-o",
        default="results",
        help="Output directory for results (default: results)",
    )

    # Docker management arguments
    docker_group = parser.add_argument_group("Docker management")
    docker_group.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip Docker container management (use already-running container)",
    )
    docker_group.add_argument(
        "--keep-container",
        action="store_true",
        help="Don't stop container after benchmark (useful for debugging)",
    )

    args = parser.parse_args()

    # Determine which mode to run in
    if args.config:
        # Config-based mode
        if not Path(args.config).exists():
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)

        # Load database config
        config = load_yaml_config(args.config)

        # Load benchmark config
        benchmark_path = args.benchmark or "benchmark.yaml"
        if not Path(benchmark_path).exists():
            print(f"Error: Benchmark config not found: {benchmark_path}")
            sys.exit(1)
        benchmark_config = load_yaml_config(benchmark_path)

        # Parse indexes filter
        indexes_filter = None
        if args.indexes:
            indexes_filter = [idx.strip() for idx in args.indexes.split(",")]

        print("=" * 80)
        print("VECTOR DATABASE BENCHMARK (Config Mode)")
        print("=" * 80)
        print(f"Database config: {args.config}")
        print(f"Benchmark config: {benchmark_path}")
        print(f"Dataset filter:  {args.dataset or 'all'}")
        print(f"Index filter:    {args.indexes or 'all'}")
        print(f"Output:          {args.output}")
        print(f"Skip Docker:     {args.skip_docker}")
        print(f"Keep container:  {args.keep_container}")
        print("=" * 80)

        run_with_config(
            config=config,
            benchmark_config=benchmark_config,
            dataset_filter=args.dataset,
            indexes_filter=indexes_filter,
            output_dir=args.output,
            container_name=args.container,
            skip_docker=args.skip_docker,
            keep_container=args.keep_container,
        )

    elif args.database:
        # Legacy mode
        if not args.dataset:
            print("Error: --dataset is required in legacy mode")
            sys.exit(1)
        run_legacy(args)

    else:
        # No mode specified
        parser.print_help()
        print("\nError: Either --config (recommended) or --database must be specified")
        sys.exit(1)


if __name__ == "__main__":
    main()
