"""Benchmark runner for vector database performance testing."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .clients.base import BaseVectorDBClient, IndexConfig, SearchConfig
from .data_loader import SIFTDataset
from .docker_monitor import DockerMonitor, MonitoringResult
from .metrics import (
    calculate_latency_percentiles,
    calculate_qps,
    calculate_recall_at_k,
    calculate_throughput,
)


@dataclass
class IngestResult:
    """Results from an ingest benchmark."""

    index_type: str
    index_config: str  # Human-readable config description
    num_vectors: int
    total_time_seconds: float
    vectors_per_second: float
    peak_memory_gb: float
    final_memory_gb: float


@dataclass
class SearchResult:
    """Results from a search benchmark."""

    index_type: str
    search_config: str  # Human-readable config description
    qps: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    recall_at_10: float
    recall_at_100: float
    cpu_percent: float
    memory_gb: float


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""

    database_name: str
    timestamp: str
    dataset_info: Dict[str, Any]
    docker_cpu_limit: float
    docker_memory_limit_gb: float
    ingest_results: List[IngestResult] = field(default_factory=list)
    search_results: List[SearchResult] = field(default_factory=list)


class BenchmarkRunner:
    """Orchestrates vector database benchmarks."""

    def __init__(
        self,
        client: BaseVectorDBClient,
        dataset: SIFTDataset,
        monitor: Optional[DockerMonitor] = None,
    ):
        """
        Initialize the benchmark runner.

        Args:
            client: Vector database client
            dataset: SIFT dataset loader
            monitor: Optional Docker monitor for resource tracking
        """
        self.client = client
        self.dataset = dataset
        self.monitor = monitor
        self.batch_size = 50000
        self.warmup_queries = 10000
        self.k_values = [10, 100]

    def run_ingest_benchmark(
        self,
        table_name: str,
        index_config: IndexConfig,
    ) -> IngestResult:
        """
        Run an ingest benchmark for a specific index configuration.

        Args:
            table_name: Name of the table to create
            index_config: Index configuration

        Returns:
            IngestResult with timing and resource metrics
        """
        print(f"\n{'='*60}")
        print(f"Running ingest benchmark: {index_config.index_type}")
        print(f"Config: {index_config.get_description()}")
        print(f"{'='*60}")

        # Drop existing table for clean state
        print("Dropping existing table if present...")
        self.client.drop_table(table_name)

        # Create table
        print(f"Creating table with {index_config.index_type} index...")
        self.client.create_table(
            table_name=table_name,
            dimension=self.dataset.dimensions,
            index_config=index_config,
        )

        # Start monitoring
        if self.monitor:
            print("Starting resource monitoring...")
            self.monitor.start_monitoring(interval_seconds=0.5)

        # Insert vectors in batches
        print(f"Inserting {self.dataset.num_base_vectors:,} vectors...")
        total_inserted = 0
        start_time = time.perf_counter()

        for batch_idx, (start_id, ids, vectors) in enumerate(
            self.dataset.get_batches(self.batch_size)
        ):
            batch_start = time.perf_counter()
            self.client.insert(table_name, ids, vectors)
            batch_time = time.perf_counter() - batch_start
            total_inserted += len(ids)

            batch_rate = len(ids) / batch_time if batch_time > 0 else 0
            print(
                f"  Batch {batch_idx + 1}: {total_inserted:,} vectors "
                f"({batch_rate:,.0f} vec/s)"
            )

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Stop monitoring and get results
        monitor_result = MonitoringResult()
        if self.monitor:
            print("Stopping resource monitoring...")
            monitor_result = self.monitor.stop_monitoring()

        # Calculate metrics
        vectors_per_second = calculate_throughput(total_inserted, total_time)

        result = IngestResult(
            index_type=index_config.index_type.upper(),
            index_config=index_config.get_description(),
            num_vectors=total_inserted,
            total_time_seconds=total_time,
            vectors_per_second=vectors_per_second,
            peak_memory_gb=monitor_result.peak_memory_gb,
            final_memory_gb=monitor_result.final_memory_gb,
        )

        print(f"\nIngest complete:")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Throughput: {vectors_per_second:,.0f} vectors/second")
        print(f"  Peak memory: {monitor_result.peak_memory_gb:.2f} GB")
        print(f"  Final memory: {monitor_result.final_memory_gb:.2f} GB")

        return result

    def run_search_benchmark(
        self,
        table_name: str,
        search_config: SearchConfig,
    ) -> SearchResult:
        """
        Run a search benchmark for a specific configuration.

        Args:
            table_name: Name of the table to search
            search_config: Search configuration

        Returns:
            SearchResult with performance metrics
        """
        config_desc = search_config.get_description()
        print(f"\n  Search config: {search_config.index_type} {config_desc}")

        # Get query vectors and ground truth
        queries = self.dataset.query_vectors
        ground_truth = self.dataset.ground_truth
        num_queries = len(queries)
        k = max(self.k_values)  # Search for max k

        # Warmup queries
        print(f"    Running {self.warmup_queries} warmup queries...")
        for i in range(min(self.warmup_queries, num_queries)):
            self.client.search(table_name, queries[i], k, search_config)

        # Start monitoring
        if self.monitor:
            self.monitor.start_monitoring(interval_seconds=0.1)

        # Run timed queries
        print(f"    Running {num_queries:,} timed queries...")
        latencies_ms = []
        all_retrieved_ids = []

        start_time = time.perf_counter()

        for i in range(num_queries):
            result = self.client.search(table_name, queries[i], k, search_config)
            latencies_ms.append(result.latency_ms)
            all_retrieved_ids.append(result.ids)

            if (i + 1) % 2000 == 0:
                elapsed = time.perf_counter() - start_time
                current_qps = (i + 1) / elapsed
                print(f"      Progress: {i + 1:,}/{num_queries:,} ({current_qps:,.0f} QPS)")

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Stop monitoring
        monitor_result = MonitoringResult()
        if self.monitor:
            monitor_result = self.monitor.stop_monitoring()

        # Calculate metrics
        qps = calculate_qps(num_queries, total_time)
        percentiles = calculate_latency_percentiles(latencies_ms)

        # Normalize results to exactly k elements per query
        # Pad with -1 if too few results, truncate if too many
        normalized_results = []
        for ids in all_retrieved_ids:
            if len(ids) < k:
                # Pad with -1 (invalid ID that won't match ground truth)
                normalized_results.append(list(ids) + [-1] * (k - len(ids)))
            elif len(ids) > k:
                normalized_results.append(ids[:k])
            else:
                normalized_results.append(ids)

        # Convert to numpy array for recall calculation
        retrieved_array = np.array(normalized_results)

        # Calculate recall at different k values
        recall_10 = calculate_recall_at_k(retrieved_array, ground_truth, 10)
        recall_100 = calculate_recall_at_k(retrieved_array, ground_truth, 100)

        result = SearchResult(
            index_type=search_config.index_type.upper(),
            search_config=config_desc,
            qps=qps,
            latency_p50_ms=percentiles["p50"],
            latency_p95_ms=percentiles["p95"],
            latency_p99_ms=percentiles["p99"],
            recall_at_10=recall_10,
            recall_at_100=recall_100,
            cpu_percent=monitor_result.avg_cpu_percent,
            memory_gb=monitor_result.final_memory_gb,
        )

        print(f"    Results: QPS={qps:,.0f}, R@10={recall_10:.4f}, R@100={recall_100:.4f}")

        return result

    def run_full_benchmark(
        self,
        hnsw_ef_search_values: List[int] = None,
        indexes_to_run: List[str] = None,
    ) -> BenchmarkResults:
        """
        Run the complete benchmark suite.

        Args:
            hnsw_ef_search_values: List of efSearch values to test for HNSW
            indexes_to_run: List of index types to run (e.g., ['flat', 'hnsw']).
                           If None, runs all indexes (flat and hnsw).

        Returns:
            BenchmarkResults with all metrics
        """
        if hnsw_ef_search_values is None:
            hnsw_ef_search_values = [8, 16, 32, 64, 128, 256]

        # Default to running all indexes if not specified
        if indexes_to_run is None:
            indexes_to_run = ["flat", "hnsw"]

        # Normalize index names to lowercase
        indexes_to_run = [idx.lower() for idx in indexes_to_run]

        # Get timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Get Docker limits if monitor is available
        docker_cpu = 0
        docker_memory_gb = 0
        if self.monitor:
            try:
                limits = self.monitor.get_container_limits()
                docker_cpu = limits.get("cpu_limit", 0)
                docker_memory_gb = limits.get("memory_limit_gb", 0)
            except Exception:
                pass

        results = BenchmarkResults(
            database_name=self.client.name,
            timestamp=timestamp,
            dataset_info=self.dataset.get_info(),
            docker_cpu_limit=docker_cpu,
            docker_memory_limit_gb=docker_memory_gb,
        )

        # Define index configurations
        flat_config = IndexConfig(
            name="flat_index",
            index_type="flat",
            params={"dims": self.dataset.dimensions, "metric": "L2"},
        )

        hnsw_config = IndexConfig(
            name="hnsw_index",
            index_type="hnsw",
            params={
                "dims": self.dataset.dimensions,
                "M": 16,
                "efConstruction": 64,
                "metric": "L2",
            },
        )

        # Track which tables were created for cleanup
        created_tables = []

        # Run Flat index benchmark if requested
        if "flat" in indexes_to_run:
            print("\n" + "=" * 80)
            print("FLAT INDEX BENCHMARK")
            print("=" * 80)

            flat_ingest = self.run_ingest_benchmark("benchmark_flat", flat_config)
            results.ingest_results.append(flat_ingest)
            created_tables.append("benchmark_flat")

            flat_search_config = SearchConfig(
                index_name="flat_index",
                index_type="flat",
                params={},
            )
            flat_search = self.run_search_benchmark("benchmark_flat", flat_search_config)
            results.search_results.append(flat_search)
        else:
            print("\nSkipping FLAT index benchmark (not in indexes_to_run)")

        # Run HNSW index benchmark if requested
        if "hnsw" in indexes_to_run:
            print("\n" + "=" * 80)
            print("HNSW INDEX BENCHMARK")
            print("=" * 80)

            hnsw_ingest = self.run_ingest_benchmark("benchmark_hnsw", hnsw_config)
            results.ingest_results.append(hnsw_ingest)
            created_tables.append("benchmark_hnsw")

            # Test different efSearch values
            print("\n  Testing different efSearch values...")
            for ef_search in hnsw_ef_search_values:
                hnsw_search_config = SearchConfig(
                    index_name="hnsw_index",
                    index_type="hnsw",
                    params={"efSearch": ef_search},
                )
                search_result = self.run_search_benchmark(
                    "benchmark_hnsw", hnsw_search_config
                )
                results.search_results.append(search_result)
        else:
            print("\nSkipping HNSW index benchmark (not in indexes_to_run)")

        # Cleanup only tables that were created
        print("\n" + "=" * 80)
        print("Cleaning up...")
        for table_name in created_tables:
            self.client.drop_table(table_name)

        return results
