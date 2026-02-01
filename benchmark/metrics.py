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
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if len(retrieved_ids) != len(ground_truth_ids):
        raise ValueError(
            f"Number of queries mismatch: {len(retrieved_ids)} vs {len(ground_truth_ids)}"
        )

    num_queries = len(retrieved_ids)

    # Vectorized recall calculation using numpy
    # Get top-k from both arrays
    retrieved_k = retrieved_ids[:, :k]
    true_k = ground_truth_ids[:, :k]

    # For each query, count how many retrieved IDs are in the ground truth
    # Use broadcasting: expand dims and compare all pairs
    # retrieved_k: (num_queries, k) -> (num_queries, k, 1)
    # true_k: (num_queries, k) -> (num_queries, 1, k)
    matches = np.any(retrieved_k[:, :, np.newaxis] == true_k[:, np.newaxis, :], axis=2)
    hits_per_query = np.sum(matches, axis=1)

    return float(np.mean(hits_per_query / k))


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
