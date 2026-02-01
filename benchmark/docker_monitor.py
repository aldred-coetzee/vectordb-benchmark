"""Docker container resource monitoring."""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import docker

# Configure logger for this module
logger = logging.getLogger(__name__)
from docker.models.containers import Container


@dataclass
class ResourceStats:
    """Container resource statistics."""

    memory_usage_bytes: int = 0
    memory_limit_bytes: int = 0
    cpu_percent: float = 0.0
    cpu_count: int = 0

    @property
    def memory_usage_gb(self) -> float:
        """Memory usage in GB."""
        return self.memory_usage_bytes / (1024**3)

    @property
    def memory_limit_gb(self) -> float:
        """Memory limit in GB."""
        return self.memory_limit_bytes / (1024**3)


@dataclass
class MonitoringResult:
    """Results from a monitoring session."""

    peak_memory_bytes: int = 0
    final_memory_bytes: int = 0
    avg_cpu_percent: float = 0.0
    memory_limit_bytes: int = 0
    cpu_count: int = 0
    samples: list = field(default_factory=list)

    @property
    def peak_memory_gb(self) -> float:
        """Peak memory in GB."""
        return self.peak_memory_bytes / (1024**3)

    @property
    def final_memory_gb(self) -> float:
        """Final memory in GB."""
        return self.final_memory_bytes / (1024**3)

    @property
    def memory_limit_gb(self) -> float:
        """Memory limit in GB."""
        return self.memory_limit_bytes / (1024**3)


class DockerMonitor:
    """Monitor Docker container resource usage."""

    def __init__(self, container_name: str):
        """
        Initialize the Docker monitor.

        Args:
            container_name: Name or ID of the Docker container to monitor
        """
        self.container_name = container_name
        self._client: Optional[docker.DockerClient] = None
        self._container: Optional[Container] = None
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._samples: list = []
        self._lock = threading.Lock()

    def connect(self) -> None:
        """Connect to Docker and find the container."""
        try:
            self._client = docker.from_env()
            self._container = self._client.containers.get(self.container_name)
        except docker.errors.NotFound:
            raise ValueError(f"Container not found: {self.container_name}")
        except docker.errors.DockerException as e:
            raise ConnectionError(f"Failed to connect to Docker: {e}")

    def disconnect(self) -> None:
        """Disconnect from Docker."""
        self.stop_monitoring()
        if self._client:
            self._client.close()
            self._client = None
        self._container = None

    def get_stats(self) -> ResourceStats:
        """
        Get current container resource statistics.

        Returns:
            ResourceStats object with current usage
        """
        if not self._container:
            raise RuntimeError("Not connected to container")

        stats = self._container.stats(stream=False)
        return self._parse_stats(stats)

    def _parse_stats(self, stats: dict) -> ResourceStats:
        """Parse Docker stats API response."""
        result = ResourceStats()

        # Memory stats
        memory_stats = stats.get("memory_stats", {})
        result.memory_usage_bytes = memory_stats.get("usage", 0)
        result.memory_limit_bytes = memory_stats.get("limit", 0)

        # CPU stats - calculate percentage
        cpu_stats = stats.get("cpu_stats", {})
        precpu_stats = stats.get("precpu_stats", {})

        cpu_usage = cpu_stats.get("cpu_usage", {})
        precpu_usage = precpu_stats.get("cpu_usage", {})

        cpu_delta = cpu_usage.get("total_usage", 0) - precpu_usage.get("total_usage", 0)
        system_delta = cpu_stats.get("system_cpu_usage", 0) - precpu_stats.get(
            "system_cpu_usage", 0
        )

        # Get number of CPUs
        online_cpus = cpu_stats.get("online_cpus", 0)
        if online_cpus == 0:
            percpu_usage = cpu_usage.get("percpu_usage", [])
            online_cpus = len(percpu_usage) if percpu_usage else 1

        result.cpu_count = online_cpus

        if system_delta > 0 and cpu_delta > 0:
            result.cpu_percent = (cpu_delta / system_delta) * online_cpus * 100.0

        return result

    def start_monitoring(self, interval_seconds: float = 0.5) -> None:
        """
        Start background monitoring of container resources.

        Args:
            interval_seconds: Time between samples
        """
        # Check if already monitoring
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return

        self._stop_event.clear()
        self._samples = []

        def monitor_loop():
            consecutive_errors = 0
            while not self._stop_event.is_set():
                try:
                    stats = self.get_stats()
                    with self._lock:
                        # Check again after getting stats in case stop was called
                        if not self._stop_event.is_set():
                            self._samples.append(stats)
                    consecutive_errors = 0  # Reset on success
                except Exception as e:
                    consecutive_errors += 1
                    # Log first error and periodic errors to avoid log spam
                    if consecutive_errors == 1 or consecutive_errors % 10 == 0:
                        logger.debug(
                            "Error collecting container stats (count=%d): %s",
                            consecutive_errors,
                            e,
                        )
                # Use wait instead of sleep for faster response to stop
                self._stop_event.wait(timeout=interval_seconds)

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> MonitoringResult:
        """
        Stop monitoring and return aggregated results.

        Returns:
            MonitoringResult with peak/avg statistics
        """
        # Signal the thread to stop
        self._stop_event.set()

        if self._monitor_thread is not None:
            # Wait longer for thread to finish (Docker stats can take time)
            self._monitor_thread.join(timeout=5.0)
            if self._monitor_thread.is_alive():
                # Thread is still running, but we've signaled it to stop
                # It will terminate on next iteration
                pass
            self._monitor_thread = None

        with self._lock:
            samples = self._samples.copy()
            self._samples = []

        result = MonitoringResult()

        if not samples:
            return result

        result.samples = samples
        result.memory_limit_bytes = samples[-1].memory_limit_bytes
        result.cpu_count = samples[-1].cpu_count
        result.final_memory_bytes = samples[-1].memory_usage_bytes
        result.peak_memory_bytes = max(s.memory_usage_bytes for s in samples)

        cpu_samples = [s.cpu_percent for s in samples if s.cpu_percent > 0]
        if cpu_samples:
            result.avg_cpu_percent = sum(cpu_samples) / len(cpu_samples)

        return result

    def get_container_limits(self) -> dict:
        """
        Get container resource limits.

        Returns:
            Dictionary with CPU and memory limits
        """
        if not self._container:
            raise RuntimeError("Not connected to container")

        self._container.reload()
        host_config = self._container.attrs.get("HostConfig", {})

        # Memory limit
        memory_limit = host_config.get("Memory", 0)

        # CPU limit (NanoCPUs is in units of 10^-9 CPUs)
        nano_cpus = host_config.get("NanoCpus", 0)
        cpu_limit = nano_cpus / 1e9 if nano_cpus else 0

        # CPU period/quota based limit
        if cpu_limit == 0:
            cpu_period = host_config.get("CpuPeriod", 0)
            cpu_quota = host_config.get("CpuQuota", 0)
            if cpu_period > 0 and cpu_quota > 0:
                cpu_limit = cpu_quota / cpu_period

        return {
            "memory_limit_bytes": memory_limit,
            "memory_limit_gb": memory_limit / (1024**3) if memory_limit else 0,
            "cpu_limit": cpu_limit,
        }
