"""Metrics calculation for vector database benchmarks."""

from typing import List

import numpy as np


def calculate_recall_at_k(
    retrieved_ids: np.ndarray,
    ground_truth_ids: np.ndarray,
    k: int
) -> float:
    """
    Calculate Recall@k for a set of search results.

    Recall@k measures the fraction of the true k nearest neighbors
    that appear in the retrieved top-k results.

    Args:
        retrieved_ids: Array of shape (num_queries, num_retrieved) containing
                      the IDs of retrieved vectors
        ground_truth_ids: Array of shape (num_queries, num_ground_truth) containing
                         the true nearest neighbor IDs
        k: Number of results to consider

    Returns:
        Mean recall@k across all queries (float between 0 and 1)
    """
    if len(retrieved_ids) != len(ground_truth_ids):
        raise ValueError(
            f"Number of queries mismatch: {len(retrieved_ids)} vs {len(ground_truth_ids)}"
        )

    num_queries = len(retrieved_ids)
    total_recall = 0.0

    for i in range(num_queries):
        # Get top-k from both retrieved and ground truth
        retrieved_k = set(retrieved_ids[i][:k])
        true_k = set(ground_truth_ids[i][:k])

        # Calculate intersection
        hits = len(retrieved_k & true_k)
        total_recall += hits / k

    return total_recall / num_queries


def calculate_latency_percentiles(
    latencies_ms: List[float]
) -> dict:
    """
    Calculate latency percentiles from a list of latencies.

    Args:
        latencies_ms: List of latency measurements in milliseconds

    Returns:
        Dictionary with P50, P95, P99 latencies
    """
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

    arr = np.array(latencies_ms)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def calculate_qps(num_queries: int, total_time_seconds: float) -> float:
    """
    Calculate queries per second.

    Args:
        num_queries: Number of queries executed
        total_time_seconds: Total time taken in seconds

    Returns:
        Queries per second
    """
    if total_time_seconds <= 0:
        return 0.0
    return num_queries / total_time_seconds


def calculate_throughput(num_vectors: int, total_time_seconds: float) -> float:
    """
    Calculate vector throughput (vectors per second).

    Args:
        num_vectors: Number of vectors processed
        total_time_seconds: Total time taken in seconds

    Returns:
        Vectors per second
    """
    if total_time_seconds <= 0:
        return 0.0
    return num_vectors / total_time_seconds
