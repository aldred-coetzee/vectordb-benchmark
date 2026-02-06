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
    type: Optional[str] = None  # "http" (default) or "tcp"


@dataclass
class SetupDirectory:
    """Directory to create before container start."""

    path: str
    mode: Optional[str] = None


@dataclass
class SetupFile:
    """File to create before container start."""

    path: str
    content: str
    mode: Optional[str] = None


@dataclass
class SetupConfig:
    """Pre-start setup configuration."""

    directories: Optional[List[SetupDirectory]] = None
    files: Optional[List[SetupFile]] = None


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
    security_opt: Optional[List[str]] = None
    command: Optional[str] = None
    setup: Optional[SetupConfig] = None


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
                type=health_section.get("type"),  # "http" (default) or "tcp"
            )

        # Parse setup config
        setup = None
        setup_section = container_section.get("setup")
        if setup_section:
            directories = None
            if setup_section.get("directories"):
                directories = [
                    SetupDirectory(
                        path=d.get("path", d) if isinstance(d, dict) else d,
                        mode=d.get("mode") if isinstance(d, dict) else None,
                    )
                    for d in setup_section["directories"]
                ]
            files = None
            if setup_section.get("files"):
                files = [
                    SetupFile(
                        path=f["path"],
                        content=f["content"],
                        mode=f.get("mode"),
                    )
                    for f in setup_section["files"]
                ]
            setup = SetupConfig(directories=directories, files=files)

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
            security_opt=container_section.get("security_opt"),
            command=container_section.get("command"),
            setup=setup,
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
        """Expand ~, relative paths, and environment variables in path."""
        # Expand ~ to home directory
        if path.startswith("~"):
            path = str(Path.home()) + path[1:]
        # Expand environment variables
        path = self._expand_env_vars(path)
        # Convert relative paths to absolute
        if path.startswith(".") or (not path.startswith("/") and "/" in path):
            path = str(Path(path).resolve())
        return path

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
                if "/" in host_path or host_path.startswith("~") or host_path.startswith("."):
                    # It's a path, expand it to absolute path
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

    def _run_setup(self) -> None:
        """Run pre-start setup: create directories and files."""
        if not self._container_config or not self._container_config.setup:
            return

        setup = self._container_config.setup

        # Create directories
        if setup.directories:
            for dir_config in setup.directories:
                dir_path = Path(self._expand_path(dir_config.path))
                if not dir_path.exists():
                    print(f"  Creating directory: {dir_path}")
                    dir_path.mkdir(parents=True, exist_ok=True)
                if dir_config.mode:
                    os.chmod(dir_path, int(dir_config.mode, 8))

        # Create files
        if setup.files:
            for file_config in setup.files:
                file_path = Path(self._expand_path(file_config.path))
                # Ensure parent directory exists
                file_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"  Creating file: {file_path}")
                file_path.write_text(file_config.content)
                if file_config.mode:
                    os.chmod(file_path, int(file_config.mode, 8))

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

        # Run pre-start setup (create directories and files)
        self._run_setup()

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

        # Add security options (e.g., seccomp:unconfined for Milvus)
        if config.security_opt:
            run_kwargs["security_opt"] = config.security_opt

        # Add command (e.g., "milvus run standalone")
        if config.command:
            run_kwargs["command"] = config.command

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
            container.stop(timeout=30)
            print(f"  Container stopped")
        except docker.errors.NotFound:
            print(f"  Container '{container_name}' not found")
            return False
        except Exception as e:
            # ReadTimeout, APIError, etc. — don't let container cleanup crash the process
            print(f"  Warning: container stop failed ({type(e).__name__}: {e}), forcing removal")

        try:
            container = self._client.containers.get(container_name)
            container.remove(force=True)
            print(f"  Container removed")
        except docker.errors.NotFound:
            pass  # Already gone
        except Exception as e:
            print(f"  Warning: container remove failed ({type(e).__name__}: {e})")

        self._container = None
        return True

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
        """Check health via HTTP URL.

        Uses urllib as primary method with curl as fallback for broader compatibility.
        """
        # Try urllib first (no external dependency)
        try:
            import urllib.request
            import urllib.error
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            pass
        except Exception:
            pass

        # Fallback to curl if urllib fails (e.g., for some edge cases)
        try:
            result = subprocess.run(
                ["curl", "-sf", url],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        except Exception:
            return False

    def _check_health_tcp(self, url: str) -> bool:
        """Check health via TCP connection.

        Parses host:port from URL (e.g., redis://localhost:6379) and attempts connection.
        """
        import socket

        # Parse host and port from URL
        # Handle formats: redis://localhost:6379, localhost:6379, tcp://host:port
        url = url.replace("redis://", "").replace("tcp://", "")
        if ":" in url:
            host, port_str = url.split(":", 1)
            # Remove any path component
            port_str = port_str.split("/")[0]
            try:
                port = int(port_str)
            except ValueError:
                return False
        else:
            return False

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
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
                    # Use TCP check if type is "tcp", otherwise HTTP
                    if health_check.type == "tcp":
                        ready = self._check_health_tcp(health_check.url)
                    else:
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
