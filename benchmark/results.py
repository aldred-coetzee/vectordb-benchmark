"""Data classes for benchmark results."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    latency_p50_ms: Optional[float]
    latency_p95_ms: Optional[float]
    latency_p99_ms: Optional[float]
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
