"""Report generation for benchmark results."""

import csv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .runner import BenchmarkResults, IngestResult, SearchResult


def format_memory(gb: float) -> str:
    """Format memory in GB with appropriate precision."""
    if gb == 0:
        return "N/A"
    return f"{gb:.1f} GB"


def format_percent(pct: float) -> str:
    """Format percentage."""
    if pct == 0:
        return "N/A"
    return f"{pct:.0f}%"


def print_console_report(results: BenchmarkResults) -> None:
    """
    Print benchmark results to console in formatted table.

    Args:
        results: Complete benchmark results
    """
    dataset = results.dataset_info

    print("\n" + "=" * 100)
    print("VECTOR DATABASE BENCHMARK RESULTS")
    print(f"Database:       {results.database_name}")
    print(f"Timestamp:      {results.timestamp}")
    print(
        f"Dataset:        {dataset['name']} "
        f"({dataset['num_base_vectors']:,} vectors, "
        f"{dataset['dimensions']} dimensions)"
    )
    print(f"Queries:        {dataset['num_query_vectors']:,}")

    if results.docker_cpu_limit > 0:
        print(f"Docker CPU:     {results.docker_cpu_limit:.0f} cores")
    if results.docker_memory_limit_gb > 0:
        print(f"Docker Memory:  {results.docker_memory_limit_gb:.0f} GB")

    # Ingest results
    print("\nINGEST (index built during insert)")
    print("-" * 100)
    print(
        f"{'Index':<10} {'Config':<22} {'Vectors':>12} {'Time(s)':>10} "
        f"{'Vec/sec':>10} {'Peak Mem':>10} {'Final Mem':>10}"
    )
    print("-" * 100)

    for r in results.ingest_results:
        print(
            f"{r.index_type:<10} {r.index_config:<22} {r.num_vectors:>12,} "
            f"{r.total_time_seconds:>10.1f} {r.vectors_per_second:>10,.0f} "
            f"{format_memory(r.peak_memory_gb):>10} {format_memory(r.final_memory_gb):>10}"
        )

    # Search results
    print("\nSEARCH")
    print("-" * 100)
    print(
        f"{'Index':<10} {'Config':<14} {'QPS':>8} {'P50(ms)':>8} "
        f"{'P95(ms)':>8} {'P99(ms)':>8} {'R@10':>7} {'R@100':>7} "
        f"{'CPU%':>6} {'Mem':>8}"
    )
    print("-" * 100)

    for r in results.search_results:
        print(
            f"{r.index_type:<10} {r.search_config:<14} {r.qps:>8,.0f} "
            f"{r.latency_p50_ms:>8.2f} {r.latency_p95_ms:>8.2f} "
            f"{r.latency_p99_ms:>8.2f} {r.recall_at_10:>7.4f} {r.recall_at_100:>7.4f} "
            f"{format_percent(r.cpu_percent):>6} {format_memory(r.memory_gb):>8}"
        )

    print("=" * 100)


def save_ingest_csv(
    results: List[IngestResult],
    output_path: str,
    metadata: dict = None,
) -> None:
    """
    Save ingest results to CSV.

    Args:
        results: List of ingest results
        output_path: Path to output CSV file
        metadata: Optional metadata to include
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "index_type",
            "index_config",
            "num_vectors",
            "total_time_seconds",
            "vectors_per_second",
            "peak_memory_gb",
            "final_memory_gb",
        ])

        # Data
        for r in results:
            writer.writerow([
                r.index_type,
                r.index_config,
                r.num_vectors,
                f"{r.total_time_seconds:.2f}",
                f"{r.vectors_per_second:.2f}",
                f"{r.peak_memory_gb:.3f}",
                f"{r.final_memory_gb:.3f}",
            ])

    print(f"Saved ingest results to {path}")


def save_search_csv(
    results: List[SearchResult],
    output_path: str,
    metadata: dict = None,
) -> None:
    """
    Save search results to CSV.

    Args:
        results: List of search results
        output_path: Path to output CSV file
        metadata: Optional metadata to include
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "index_type",
            "search_config",
            "qps",
            "latency_p50_ms",
            "latency_p95_ms",
            "latency_p99_ms",
            "recall_at_10",
            "recall_at_100",
            "cpu_percent",
            "memory_gb",
        ])

        # Data
        for r in results:
            writer.writerow([
                r.index_type,
                r.search_config,
                f"{r.qps:.2f}",
                f"{r.latency_p50_ms:.3f}",
                f"{r.latency_p95_ms:.3f}",
                f"{r.latency_p99_ms:.3f}",
                f"{r.recall_at_10:.6f}",
                f"{r.recall_at_100:.6f}",
                f"{r.cpu_percent:.2f}",
                f"{r.memory_gb:.3f}",
            ])

    print(f"Saved search results to {path}")


def plot_recall_vs_qps(
    results: List[SearchResult],
    output_path: str,
    title: str = "Recall@10 vs QPS",
) -> None:
    """
    Create a scatter plot of Recall@10 vs QPS.

    Args:
        results: List of search results
        output_path: Path to save the plot
        title: Plot title
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Separate results by index type
    flat_results = [r for r in results if r.index_type.upper() == "FLAT"]
    hnsw_results = [r for r in results if r.index_type.upper() == "HNSW"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot HNSW results
    if hnsw_results:
        recalls = [r.recall_at_10 for r in hnsw_results]
        qps_values = [r.qps for r in hnsw_results]
        labels = [r.search_config for r in hnsw_results]

        ax.scatter(recalls, qps_values, s=100, label="HNSW", marker="o", color="blue")

        # Add labels for each point
        for i, (x, y, label) in enumerate(zip(recalls, qps_values, labels)):
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    # Plot Flat results (should be a single point at recall=1.0)
    if flat_results:
        recalls = [r.recall_at_10 for r in flat_results]
        qps_values = [r.qps for r in flat_results]

        ax.scatter(
            recalls, qps_values, s=100, label="Flat (baseline)",
            marker="s", color="red"
        )

    ax.set_xlabel("Recall@10", fontsize=12)
    ax.set_ylabel("Queries Per Second (QPS)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_yscale("log")  # Log scale for QPS
    ax.set_xlim(0.7, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {path}")


def generate_full_report(
    results: BenchmarkResults,
    output_dir: str = "results",
) -> None:
    """
    Generate all reports (console, CSVs, and plot).

    Args:
        results: Complete benchmark results
        output_dir: Directory to save output files
    """
    # Console output
    print_console_report(results)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Metadata for CSV files
    metadata = {
        "database": results.database_name,
        "timestamp": results.timestamp,
        "dataset": results.dataset_info["name"],
    }

    # Save CSVs
    save_ingest_csv(
        results.ingest_results,
        str(output_path / "benchmark_ingest.csv"),
        metadata,
    )

    save_search_csv(
        results.search_results,
        str(output_path / "benchmark_search.csv"),
        metadata,
    )

    # Generate plot
    plot_recall_vs_qps(
        results.search_results,
        str(output_path / "recall_vs_qps.png"),
        title=f"{results.database_name} - Recall@10 vs QPS",
    )
