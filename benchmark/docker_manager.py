"""Docker container lifecycle management for benchmarks."""

import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import docker
from docker.models.containers import Container


@dataclass
class HealthCheckConfig:
    """Health check configuration."""

    url: Optional[str] = None
    command: Optional[str] = None
    timeout: int = 60
    interval: int = 2


@dataclass
class ContainerConfig:
    """Container configuration parsed from YAML."""

    image: str
    name: str
    cpus: Optional[float] = None
    memory: Optional[str] = None
    cpuset_cpus: Optional[str] = None
    ports: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    volumes: Optional[List[str]] = None
    health_check: Optional[HealthCheckConfig] = None


class DockerManager:
    """Manage Docker container lifecycle for benchmarks."""

    def __init__(self, config: Dict[str, Any], database_name: Optional[str] = None):
        """
        Initialize Docker manager from config.

        Args:
            config: Database YAML config dict containing 'container' section
            database_name: Override database name for container naming
        """
        self._client: Optional[docker.DockerClient] = None
        self._container: Optional[Container] = None
        self._config = config

        # Extract database name
        db_config = config.get("database", {})
        self._database_name = database_name or db_config.get("name", "unknown")

        # Parse container config
        self._container_config = self._parse_container_config()

    def _parse_container_config(self) -> Optional[ContainerConfig]:
        """Parse container configuration from YAML config."""
        container_section = self._config.get("container")
        if not container_section:
            return None

        # Container name derived from database name
        container_name = f"benchmark-{self._database_name}"

        # Parse health check config
        health_check = None
        health_section = self._config.get("health_check")
        if health_section:
            health_check = HealthCheckConfig(
                url=health_section.get("url"),
                command=health_section.get("command"),
                timeout=health_section.get("timeout", 60),
                interval=health_section.get("interval", 2),
            )

        return ContainerConfig(
            image=container_section.get("image", ""),
            name=container_name,
            cpus=container_section.get("cpus"),
            memory=container_section.get("memory"),
            cpuset_cpus=container_section.get("cpuset_cpus"),
            ports=container_section.get("ports"),
            env=container_section.get("env"),
            volumes=container_section.get("volumes"),
            health_check=health_check,
        )

    @property
    def has_container(self) -> bool:
        """Check if config has a container section."""
        return self._container_config is not None

    @property
    def container_name(self) -> Optional[str]:
        """Get the container name."""
        if self._container_config:
            return self._container_config.name
        return None

    def _expand_env_vars(self, value: str) -> str:
        """
        Expand environment variables in a string.

        Supports ${VAR} and ${VAR:-default} syntax.
        """
        def replace_var(match):
            var_expr = match.group(1)
            # Handle ${VAR:-default} syntax
            if ":-" in var_expr:
                var_name, default = var_expr.split(":-", 1)
                return os.environ.get(var_name, default)
            else:
                return os.environ.get(var_expr, "")

        # Match ${...} patterns
        pattern = r"\$\{([^}]+)\}"
        return re.sub(pattern, replace_var, value)

    def _expand_path(self, path: str) -> str:
        """Expand ~ and environment variables in path."""
        # Expand ~ to home directory
        if path.startswith("~"):
            path = str(Path.home()) + path[1:]
        # Expand environment variables
        return self._expand_env_vars(path)

    def _prepare_environment(self) -> Dict[str, str]:
        """Prepare environment variables with expansion."""
        if not self._container_config or not self._container_config.env:
            return {}

        result = {}
        for key, value in self._container_config.env.items():
            if isinstance(value, str):
                result[key] = self._expand_env_vars(value)
            else:
                result[key] = str(value)
        return result

    def _prepare_volumes(self) -> Dict[str, Dict[str, str]]:
        """Prepare volume bindings with path expansion."""
        if not self._container_config or not self._container_config.volumes:
            return {}

        result = {}
        for vol in self._container_config.volumes:
            if ":" in vol:
                parts = vol.split(":")
                host_path = parts[0]
                container_path = parts[1]
                mode = parts[2] if len(parts) > 2 else "rw"

                # Check if it's a named volume or a bind mount
                if "/" in host_path or host_path.startswith("~"):
                    # It's a path, expand it
                    host_path = self._expand_path(host_path)
                    result[host_path] = {"bind": container_path, "mode": mode}
                else:
                    # It's a named volume, use as-is
                    result[host_path] = {"bind": container_path, "mode": mode}
            else:
                # Just a named volume with default mount
                result[vol] = {"bind": f"/{vol}", "mode": "rw"}

        return result

    def _prepare_ports(self) -> Dict[str, int]:
        """Prepare port bindings."""
        if not self._container_config or not self._container_config.ports:
            return {}

        result = {}
        for port in self._container_config.ports:
            if ":" in str(port):
                host_port, container_port = str(port).split(":")
                # Handle protocol suffix like 6333/tcp
                if "/" in container_port:
                    container_port = container_port.split("/")[0]
                result[f"{container_port}/tcp"] = int(host_port)
            else:
                port_num = int(str(port).split("/")[0])
                result[f"{port_num}/tcp"] = port_num

        return result

    def _connect_client(self) -> None:
        """Connect to Docker daemon if not already connected."""
        if not self._client:
            self._client = docker.from_env()

    def _stop_existing_container(self, name: str) -> None:
        """Stop and remove an existing container with the given name."""
        self._connect_client()

        try:
            existing = self._client.containers.get(name)
            print(f"  Stopping existing container '{name}'...")
            existing.stop(timeout=10)
            print(f"  Removing container '{name}'...")
            existing.remove(force=True)
        except docker.errors.NotFound:
            pass  # Container doesn't exist, nothing to do
        except docker.errors.APIError as e:
            print(f"  Warning: Error removing container: {e}")

    def start_container(self) -> bool:
        """
        Start the container based on configuration.

        Returns:
            True if container started successfully, False if no container config
        """
        if not self._container_config:
            print("No container configuration - skipping Docker start")
            return False

        config = self._container_config
        print(f"\nStarting Docker container '{config.name}'...")

        # Stop any existing container with same name
        self._stop_existing_container(config.name)

        self._connect_client()

        # Prepare container arguments
        run_kwargs = {
            "image": config.image,
            "name": config.name,
            "detach": True,
            "remove": False,  # We'll remove manually for control
        }

        # Add resource constraints
        if config.cpus:
            run_kwargs["nano_cpus"] = int(config.cpus * 1e9)

        if config.memory:
            run_kwargs["mem_limit"] = config.memory

        if config.cpuset_cpus:
            run_kwargs["cpuset_cpus"] = config.cpuset_cpus

        # Add ports
        ports = self._prepare_ports()
        if ports:
            run_kwargs["ports"] = ports

        # Add environment
        env = self._prepare_environment()
        if env:
            run_kwargs["environment"] = env

        # Add volumes
        volumes = self._prepare_volumes()
        if volumes:
            run_kwargs["volumes"] = volumes

        # Print configuration
        print(f"  Image: {config.image}")
        if config.cpus:
            print(f"  CPUs: {config.cpus}")
        if config.memory:
            print(f"  Memory: {config.memory}")
        if ports:
            print(f"  Ports: {list(ports.values())}")

        try:
            self._container = self._client.containers.run(**run_kwargs)
            print(f"  Container started with ID: {self._container.short_id}")
            return True
        except docker.errors.ImageNotFound:
            print(f"  Error: Image not found: {config.image}")
            print("  Try: docker pull " + config.image)
            raise
        except docker.errors.APIError as e:
            print(f"  Error starting container: {e}")
            raise

    def stop_container(self) -> bool:
        """
        Stop and remove the container.

        Returns:
            True if container was stopped, False if no container to stop
        """
        if not self._container_config:
            return False

        container_name = self._container_config.name
        print(f"\nStopping Docker container '{container_name}'...")

        self._connect_client()

        try:
            container = self._client.containers.get(container_name)
            container.stop(timeout=10)
            print(f"  Container stopped")
            container.remove(force=True)
            print(f"  Container removed")
            self._container = None
            return True
        except docker.errors.NotFound:
            print(f"  Container '{container_name}' not found")
            return False
        except docker.errors.APIError as e:
            print(f"  Error stopping container: {e}")
            return False

    def is_running(self) -> bool:
        """Check if the container is running."""
        if not self._container_config:
            return False

        self._connect_client()

        try:
            container = self._client.containers.get(self._container_config.name)
            return container.status == "running"
        except docker.errors.NotFound:
            return False
        except docker.errors.APIError:
            return False

    def get_logs(self, tail: int = 100) -> str:
        """
        Get container logs.

        Args:
            tail: Number of lines to retrieve

        Returns:
            Container logs as string
        """
        if not self._container_config:
            return ""

        self._connect_client()

        try:
            container = self._client.containers.get(self._container_config.name)
            return container.logs(tail=tail).decode("utf-8", errors="replace")
        except docker.errors.NotFound:
            return "Container not found"
        except docker.errors.APIError as e:
            return f"Error getting logs: {e}"

    def _check_health_url(self, url: str) -> bool:
        """Check health via HTTP URL using curl."""
        try:
            result = subprocess.run(
                ["curl", "-sf", url],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False

    def _check_health_command(self, command: str) -> bool:
        """Check health via shell command.

        Uses shlex.split() to safely parse the command into arguments,
        avoiding shell injection vulnerabilities.
        """
        try:
            # Expand container name in command
            if self._container_config:
                command = command.replace("${CONTAINER}", self._container_config.name)

            # Parse command safely into list of arguments
            cmd_args = shlex.split(command)

            result = subprocess.run(
                cmd_args,
                shell=False,
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except ValueError as e:
            # shlex.split() can raise ValueError for malformed strings
            print(f"  Warning: Invalid health check command format: {e}")
            return False
        except Exception:
            return False

    def wait_for_ready(
        self,
        timeout: Optional[int] = None,
        interval: Optional[int] = None,
    ) -> bool:
        """
        Wait for container to become ready.

        Uses health check configuration from YAML if available.

        Args:
            timeout: Maximum seconds to wait (default from config or 60)
            interval: Seconds between checks (default from config or 2)

        Returns:
            True if container is ready, False if timeout
        """
        if not self._container_config:
            return True  # No container means nothing to wait for

        health_check = self._container_config.health_check

        # Get timeout and interval from config or defaults
        check_timeout = timeout or (health_check.timeout if health_check else 60)
        check_interval = interval or (health_check.interval if health_check else 2)

        print(f"  Waiting for container to be ready (timeout: {check_timeout}s)...")

        start_time = time.time()
        attempts = 0

        while time.time() - start_time < check_timeout:
            attempts += 1

            # First check if container is running
            if not self.is_running():
                print(f"  Container is not running, waiting...")
                time.sleep(check_interval)
                continue

            # If we have a health check, use it
            if health_check:
                ready = False
                if health_check.url:
                    ready = self._check_health_url(health_check.url)
                elif health_check.command:
                    ready = self._check_health_command(health_check.command)
                else:
                    # No specific check, just verify container is running
                    ready = True

                if ready:
                    elapsed = time.time() - start_time
                    print(f"  Container ready after {elapsed:.1f}s ({attempts} attempts)")
                    return True
            else:
                # No health check config, just wait a bit for startup
                time.sleep(2)
                print(f"  Container running (no health check configured)")
                return True

            time.sleep(check_interval)

        print(f"  Timeout waiting for container after {check_timeout}s")
        print(f"  Container logs:\n{self.get_logs(tail=20)}")
        return False

    def close(self) -> None:
        """Close the Docker client connection."""
        if self._client:
            self._client.close()
            self._client = None


def get_default_health_check(database_name: str) -> Optional[HealthCheckConfig]:
    """
    Get default health check configuration for a database.

    Args:
        database_name: Name of the database

    Returns:
        HealthCheckConfig or None
    """
    defaults = {
        "kdbai": HealthCheckConfig(
            url="http://localhost:8081/api/v2/ready",
            timeout=60,
            interval=2,
        ),
        "qdrant": HealthCheckConfig(
            url="http://localhost:6333/readyz",
            timeout=60,
            interval=2,
        ),
        "pgvector": HealthCheckConfig(
            command="pg_isready -h localhost -p 5432 -U postgres",
            timeout=60,
            interval=2,
        ),
        "milvus": HealthCheckConfig(
            # Placeholder - Milvus health check
            url="http://localhost:9091/healthz",
            timeout=60,
            interval=2,
        ),
        "weaviate": HealthCheckConfig(
            # Placeholder - Weaviate health check
            url="http://localhost:8080/v1/.well-known/ready",
            timeout=60,
            interval=2,
        ),
    }
    return defaults.get(database_name.lower())
