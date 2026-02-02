#!/usr/bin/env python3
"""
Run benchmarks across multiple vector databases sequentially.

This script manages Docker containers for each database, runs benchmarks,
and provides a summary of all results.

Usage:
    # Run all non-placeholder configs
    python run_all.py --configs all --benchmark benchmark.yaml --dataset sift

    # Run specific databases
    python run_all.py --configs configs/kdbai.yaml,configs/qdrant.yaml \
        --benchmark benchmark.yaml --dataset sift --indexes flat,hnsw

    # Run with custom options
    python run_all.py --configs all --benchmark benchmark.yaml \
        --dataset sift --indexes hnsw --output results/comparison
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark.config import load_yaml_config


@dataclass
class BenchmarkResult:
    """Result of a single database benchmark."""

    database: str
    config_path: str
    success: bool
    duration_seconds: float
    error: Optional[str] = None
    run_id: Optional[int] = None


@dataclass
class SummaryResults:
    """Summary of all benchmark results."""

    results: List[BenchmarkResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0

    def add(self, result: BenchmarkResult) -> None:
        """Add a benchmark result."""
        self.results.append(result)

    @property
    def successful(self) -> List[BenchmarkResult]:
        """Get successful results."""
        return [r for r in self.results if r.success]

    @property
    def failed(self) -> List[BenchmarkResult]:
        """Get failed results."""
        return [r for r in self.results if not r.success]


def cleanup_benchmark_containers() -> None:
    """
    Clean up any lingering benchmark containers from previous runs.

    This prevents port conflicts when containers didn't shut down properly.
    """
    try:
        import docker
        client = docker.from_env()

        # Find all containers with benchmark- prefix
        containers = client.containers.list(all=True, filters={"name": "benchmark-"})

        if containers:
            print("Cleaning up lingering benchmark containers...")
            for container in containers:
                try:
                    if container.status == "running":
                        print(f"  Stopping: {container.name}")
                        container.stop(timeout=5)
                    print(f"  Removing: {container.name}")
                    container.remove(force=True)
                except Exception as e:
                    print(f"  Warning: Could not remove {container.name}: {e}")

        # Prune any dangling networks that might hold ports
        client.networks.prune()
        client.close()

    except Exception as e:
        print(f"Warning: Container cleanup failed: {e}")
        print("Continuing anyway...")


def find_config_files(configs_dir: str = "configs") -> List[Path]:
    """Find all YAML config files in the configs directory."""
    configs_path = Path(configs_dir)
    if not configs_path.exists():
        return []
    return sorted(configs_path.glob("*.yaml"))


def is_placeholder_config(config: Dict[str, Any]) -> bool:
    """Check if a config is marked as a placeholder."""
    return config.get("placeholder", False)


def has_container_config(config: Dict[str, Any]) -> bool:
    """Check if a config has a container section."""
    return "container" in config and config["container"]


def run_single_benchmark(
    config_path: str,
    benchmark_config_path: str,
    dataset: Optional[str],
    indexes: Optional[str],
    output_dir: str,
    skip_docker: bool = False,
    keep_container: bool = False,
) -> BenchmarkResult:
    """
    Run a benchmark for a single database.

    Args:
        config_path: Path to database config YAML
        benchmark_config_path: Path to benchmark config YAML
        dataset: Optional dataset filter
        indexes: Optional comma-separated indexes filter
        output_dir: Output directory for results
        skip_docker: Skip Docker management
        keep_container: Keep container after benchmark

    Returns:
        BenchmarkResult with success/failure status
    """
    from benchmark.data_loader import SIFTDataset
    from benchmark.docker_manager import DockerManager
    from benchmark.docker_monitor import DockerMonitor
    from benchmark.runner import BenchmarkRunner
    from benchmark.report import generate_full_report
    from benchmark.db import BenchmarkDatabase

    from datetime import datetime

    start_time = time.time()
    start_time_iso = datetime.now().isoformat()

    # Load configs
    config = load_yaml_config(config_path)
    benchmark_config = load_yaml_config(benchmark_config_path)

    db_config = config.get("database", {})
    database_name = db_config.get("name", "unknown")
    db_version = db_config.get("version")
    endpoint = db_config.get("endpoint")

    container_config = config.get("container", {})
    cpus = container_config.get("cpus", 0)
    memory = container_config.get("memory", "0g")

    # Parse memory using shared utility function
    from run_benchmark import parse_memory_string
    memory_gb = parse_memory_string(memory)

    # Initialize Docker manager
    docker_manager = None
    if database_name.lower() != "faiss" and container_config:
        docker_manager = DockerManager(config, database_name)

        if not skip_docker and docker_manager.has_container:
            try:
                docker_manager.start_container()
                if not docker_manager.wait_for_ready():
                    raise RuntimeError("Container failed to become ready")
            except Exception as e:
                duration = time.time() - start_time
                return BenchmarkResult(
                    database=database_name,
                    config_path=config_path,
                    success=False,
                    duration_seconds=duration,
                    error=f"Docker startup failed: {e}",
                )

    # Run the benchmark
    client = None
    db = None
    run_id = None

    try:
        # Get client
        from run_benchmark import get_client, get_supported_indexes, filter_indexes

        client = get_client(database_name)
        if endpoint:
            client.connect(endpoint=endpoint)
        else:
            client.connect()

        # Determine indexes
        indexes_config = benchmark_config.get("indexes", {})
        all_indexes = list(indexes_config.keys())

        if indexes:
            requested_indexes = [idx.strip() for idx in indexes.split(",")]
        else:
            requested_indexes = all_indexes

        supported = get_supported_indexes(client)
        indexes_to_run = filter_indexes(requested_indexes, supported, database_name)

        if not indexes_to_run:
            raise RuntimeError("No valid indexes to run")

        # Get search config
        search_config = benchmark_config.get("search", {})
        batch_size = search_config.get("batch_size", 50000)
        num_queries = search_config.get("num_queries", 10000)

        # Determine datasets
        datasets_config = benchmark_config.get("datasets", {})
        if dataset:
            if dataset not in datasets_config:
                raise RuntimeError(f"Dataset '{dataset}' not found in config")
            datasets_to_run = {dataset: datasets_config[dataset]}
        else:
            datasets_to_run = datasets_config

        # Initialize SQLite
        db = BenchmarkDatabase()

        # Run for each dataset
        for dataset_name, dataset_info in datasets_to_run.items():
            dataset_path = Path(dataset_info.get("path", f"datasets/{dataset_name}"))
            if not dataset_path.exists():
                print(f"Warning: Dataset not found: {dataset_path}, skipping")
                continue

            # Load dataset
            sift_dataset = SIFTDataset(str(dataset_path))

            # Setup monitor
            monitor = None
            effective_container_name = docker_manager.container_name if docker_manager else None
            if database_name.lower() != "faiss" and effective_container_name:
                try:
                    monitor = DockerMonitor(effective_container_name)
                    monitor.connect()
                except Exception as e:
                    print(f"Warning: Could not connect to Docker monitor: {e}")
                    monitor = None

            # Run benchmark
            try:
                runner = BenchmarkRunner(
                    client, sift_dataset, monitor,
                    batch_size=batch_size,
                )

                hnsw_config = indexes_config.get("hnsw", {})
                ef_search_values = hnsw_config.get("efSearch", [8, 16, 32, 64, 128, 256])

                results = runner.run_full_benchmark(
                    hnsw_ef_search_values=ef_search_values,
                    indexes_to_run=indexes_to_run,
                )

                if cpus > 0:
                    results.docker_cpu_limit = float(cpus)
                if memory_gb > 0:
                    results.docker_memory_limit_gb = memory_gb

                # Generate reports
                db_output_dir = f"{output_dir}/{database_name}"
                generate_full_report(results, db_output_dir)

                # Save to SQLite with timing info
                end_time_iso = datetime.now().isoformat()
                duration_seconds = time.time() - start_time
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
            finally:
                if monitor:
                    try:
                        monitor.disconnect()
                    except Exception:
                        pass

        duration = time.time() - start_time
        return BenchmarkResult(
            database=database_name,
            config_path=config_path,
            success=True,
            duration_seconds=duration,
            run_id=run_id,
        )

    except Exception as e:
        traceback.print_exc()
        duration = time.time() - start_time
        return BenchmarkResult(
            database=database_name,
            config_path=config_path,
            success=False,
            duration_seconds=duration,
            error=str(e),
        )

    finally:
        # Cleanup
        if client:
            try:
                client.disconnect()
            except Exception:
                pass
        if db:
            db.close()

        # Stop container
        if docker_manager and docker_manager.has_container and not skip_docker and not keep_container:
            docker_manager.stop_container()
            docker_manager.close()
        elif docker_manager:
            docker_manager.close()


def print_summary(summary: SummaryResults) -> None:
    """Print a summary of all benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    print(f"\nTotal databases: {len(summary.results)}")
    print(f"Successful: {len(summary.successful)}")
    print(f"Failed: {len(summary.failed)}")
    print(f"Total duration: {summary.total_duration_seconds:.1f}s")

    if summary.successful:
        print("\n--- Successful Benchmarks ---")
        for r in summary.successful:
            run_info = f" (run_id: {r.run_id})" if r.run_id else ""
            print(f"  {r.database}: {r.duration_seconds:.1f}s{run_info}")

    if summary.failed:
        print("\n--- Failed Benchmarks ---")
        for r in summary.failed:
            print(f"  {r.database}: {r.error}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks across multiple vector databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all non-placeholder configs
    python run_all.py --configs all --benchmark benchmark.yaml --dataset sift

    # Run specific databases
    python run_all.py --configs configs/kdbai.yaml,configs/qdrant.yaml \\
        --benchmark benchmark.yaml --dataset sift

    # Run with specific indexes
    python run_all.py --configs all --benchmark benchmark.yaml \\
        --dataset sift --indexes flat,hnsw
        """,
    )

    parser.add_argument(
        "--configs",
        required=True,
        help="Comma-separated config paths or 'all' to scan configs/ directory",
    )
    parser.add_argument(
        "--benchmark", "-b",
        default="benchmark.yaml",
        help="Path to shared benchmark config (default: benchmark.yaml)",
    )
    parser.add_argument(
        "--dataset", "-s",
        help="Dataset to benchmark (e.g., sift)",
    )
    parser.add_argument(
        "--indexes", "-i",
        help="Comma-separated list of index types (e.g., flat,hnsw)",
    )
    parser.add_argument(
        "--output", "-o",
        default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip Docker container management",
    )
    parser.add_argument(
        "--keep-container",
        action="store_true",
        help="Keep containers running after each benchmark",
    )

    args = parser.parse_args()

    # Validate benchmark config
    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        print(f"Error: Benchmark config not found: {args.benchmark}")
        sys.exit(1)

    # Determine which configs to run
    if args.configs.lower() == "all":
        config_files = find_config_files()
        if not config_files:
            print("Error: No config files found in configs/ directory")
            sys.exit(1)
    else:
        config_files = [Path(p.strip()) for p in args.configs.split(",")]
        # Validate all files exist
        for cf in config_files:
            if not cf.exists():
                print(f"Error: Config file not found: {cf}")
                sys.exit(1)

    # Clean up any lingering containers from previous runs
    cleanup_benchmark_containers()

    print("=" * 80)
    print("VECTOR DATABASE BENCHMARK - RUN ALL")
    print("=" * 80)
    print(f"Configs to run: {len(config_files)}")
    print(f"Benchmark config: {args.benchmark}")
    print(f"Dataset: {args.dataset or 'all'}")
    print(f"Indexes: {args.indexes or 'all'}")
    print(f"Output: {args.output}")
    print("=" * 80)

    # Filter out placeholder configs and FAISS (no container)
    configs_to_run = []
    for config_path in config_files:
        config = load_yaml_config(str(config_path))
        db_name = config.get("database", {}).get("name", "unknown")

        if is_placeholder_config(config):
            print(f"Skipping {db_name}: marked as placeholder")
            continue

        configs_to_run.append((config_path, config, db_name))

    if not configs_to_run:
        print("Error: No valid configs to run after filtering placeholders")
        sys.exit(1)

    from datetime import datetime

    print(f"\nWill run benchmarks for: {[c[2] for c in configs_to_run]}\n")

    # Run benchmarks sequentially
    summary = SummaryResults()
    total_start = time.time()
    total_start_time = datetime.now()
    print(f"Benchmark suite started at: {total_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    for i, (config_path, config, db_name) in enumerate(configs_to_run, 1):
        db_start = datetime.now()
        print("\n" + "=" * 80)
        print(f"[{i}/{len(configs_to_run)}] BENCHMARKING: {db_name}")
        print(f"Started at: {db_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        result = run_single_benchmark(
            config_path=str(config_path),
            benchmark_config_path=args.benchmark,
            dataset=args.dataset,
            indexes=args.indexes,
            output_dir=args.output,
            skip_docker=args.skip_docker,
            keep_container=args.keep_container,
        )

        summary.add(result)
        db_end = datetime.now()

        if result.success:
            print(f"\n[{db_name}] Completed successfully")
            print(f"  Started:  {db_start.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Ended:    {db_end.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Duration: {result.duration_seconds:.1f}s ({result.duration_seconds/60:.1f} minutes)")
        else:
            print(f"\n[{db_name}] Failed: {result.error}")

    summary.total_duration_seconds = time.time() - total_start
    total_end_time = datetime.now()

    # Print summary
    print_summary(summary)
    print(f"\nBenchmark suite ended at: {total_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total suite duration: {summary.total_duration_seconds:.1f}s ({summary.total_duration_seconds/60:.1f} minutes)")

    # Exit with error code if any failed
    if summary.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
