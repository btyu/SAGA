#!/usr/bin/env python3
"""
Main entry point for scorer module.
Supports running as: python -m <scorer> [options]

This version uses Docker Python SDK for robust container management.
"""

import os
import argparse
from loguru import logger
import warnings
import sys
from pathlib import Path
from typing import Optional, List


DEFAULT_IMAGE = os.getenv("SCILEO_RUNTIME_IMAGE", "btyu24/scileo:v4")


class ScorerDockerRunner:
    """Scorer Docker runner using Python Docker SDK."""

    def __init__(self):
        """Initialize Docker client."""
        try:
            import docker
            from docker.errors import DockerException, APIError
            self.docker = docker
            self.DockerException = DockerException
            self.APIError = APIError
            self.client = docker.from_env()
            # Test connection
            self.client.ping()
            logger.debug("Docker client initialized successfully")

            # Track the current container (only one runs at a time)
            self.current_container = None

        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise RuntimeError(f"Docker is not available or not running: {e}")

    def detect_cuda_support(self) -> bool:
        """Detect if CUDA is available and Docker supports GPU access."""
        logger.debug("Starting CUDA support detection...")

        # Step 1: Check if nvidia-smi is available on host machine
        try:
            import subprocess

            logger.debug("Checking for nvidia-smi on host machine...")
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=10,
            )
            if result.returncode != 0:
                logger.debug(
                    f"nvidia-smi failed on host with return code {result.returncode}"
                )
                return False
            logger.debug("nvidia-smi check passed on host machine")
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ) as e:
            logger.debug(f"nvidia-smi check failed on host: {e}")
            return False

        # Step 2: Test GPU access with Docker SDK
        try:
            self.client.containers.run(
                DEFAULT_IMAGE,
                "echo 'GPU test successful'",
                device_requests=[
                    self.docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                ],
                remove=True,
                detach=False,
            )
            logger.debug("Docker GPU support test passed")
            return True
        except Exception as e:
            logger.debug(f"Docker GPU support test failed: {e}")
            return False

    def run_in_docker(
        self,
        transport: str,
        host: str = "localhost",
        port: int = 8000,
        extra_args: Optional[List[str]] = None,
        image: str = DEFAULT_IMAGE,
        no_gpu: bool = False,
    ) -> bool:
        """
        Run the MCP server in a Docker container using the Docker SDK.

        Args:
            transport: Transport method (stdio, streamable-http)
            host: Host to bind to for HTTP transports
            port: Port for HTTP transports
            extra_args: Additional command line arguments
            image: Docker image to use
            no_gpu: Disable GPU support even if available

        Returns:
            True if successful, False otherwise
        """
        if transport == "stdio":
            warnings.warn(
                "`stdio` transport may not work with docker run. Use streamable-http instead."
            )

        if extra_args is None:
            extra_args = []

        # Get the current module directory
        module_dir = Path(__file__).parent.absolute()

        # Pull the image first
        logger.debug(f"Pulling Docker image: {image}")
        self.client.images.pull(image)

        # Check for CUDA support (but respect no_gpu flag)
        cuda_supported = False if no_gpu else self.detect_cuda_support()
        if no_gpu:
            logger.info("GPU support disabled by --no-gpu flag")
        elif cuda_supported:
            logger.info(
                "CUDA support detected, enabling GPU access in Docker container"
            )
        else:
            logger.info(
                "CUDA support not detected, skipping GPU access in Docker container"
            )

        # Prepare container configuration
        volumes = {str(module_dir): {"bind": "/workspace/scorer", "mode": "rw"}}

        environment = {
            "PYTHONUNBUFFERED": "1",  # Ensure Python output is unbuffered for real-time logs
        }

        # Set user permissions for file ownership
        try:
            uid = os.getuid()
            gid = os.getgid()
            environment.update({"HOST_UID": str(uid), "HOST_GID": str(gid)})
        except AttributeError:
            # Windows doesn't have getuid/getgid
            pass

        # Build MCP server command
        mcp_command = f"python -m scorer.mcp_server --transport={transport}"
        if transport == "streamable-http":
            # Use 0.0.0.0 for port mapping to accept connections from host
            mcp_command += f" --host=0.0.0.0 --port={port}"

        # Add extra arguments
        if extra_args:
            mcp_command += " " + " ".join(extra_args)

        # Build the full command with setup and cleanup
        full_command = f"bash scorer/setup.sh && {mcp_command}"

        # Add permission fix if we have uid/gid
        if "HOST_UID" in environment:
            full_command += f" && chown -R {environment['HOST_UID']}:{environment['HOST_GID']} /workspace"

        container_config = {
            "image": image,
            "command": ["/bin/bash", "-c", full_command],
            "volumes": volumes,
            "environment": environment,
            "ports": {f"{port}/tcp": port},  # Port mapping (works on WSL and Linux)
            "remove": True,  # Auto-remove container when it stops
            "detach": True,  # Run in background so we can stream logs
            "stdout": True,
            "stderr": True,
            "tty": True,
        }

        # Add GPU support if available
        if cuda_supported:
            import docker
            container_config["device_requests"] = [
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]

        container_cleaned = False
        try:
            logger.info(
                f"Starting MCP server in Docker container with transport: {transport}"
            )
            if transport == "streamable-http":
                logger.info(f"Server will be available at http://{host}:{port}")

            logger.debug(f"Full command: {full_command}")

            # Create and start the container
            self.current_container = self.client.containers.run(**container_config)
            logger.info("Docker container started successfully")

            # Stream logs continuously (follow=True for long-running servers)
            try:
                for log_line in self.current_container.logs(stream=True, follow=True):
                    if log_line:
                        decoded = log_line.decode("utf-8", errors="replace")
                        sys.stdout.write(decoded)
                        sys.stdout.flush()
            except KeyboardInterrupt:
                logger.info("Received interrupt, stopping container...")
                self.current_container.stop(timeout=10)
                raise
            except Exception as e:
                logger.warning(f"Error streaming logs: {e}")

            # Wait for container (won't be reached for long-running servers)
            try:
                exit_status = self.current_container.wait(timeout=1)
                logger.info(f"Container exited with status: {exit_status}")
            except Exception:
                # Container still running (expected for servers)
                pass
            
            container_cleaned = True
            return True
        except Exception as e:
            logger.error(f"Docker API error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error running Docker container: {e}")
            return False
        finally:
            if self.current_container and not container_cleaned:
                try:
                    self.current_container.stop(timeout=10)
                    self.current_container.remove(force=True)
                    logger.info("Cleaned up Docker container after error")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up container: {cleanup_error}")


def run_local(transport: str, host: str = "localhost", port: int = 3000):
    """Run the MCP server locally."""
    # Import and run the MCP server
    from .mcp_server import run_server

    run_server(transport, host, port)


def main():
    parser = argparse.ArgumentParser(description="Run Local similarity scorer MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="streamable-http",
        help="Transport method for the MCP server (default: streamable-http)",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind to for HTTP-based transports (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to use for HTTP-based transports (default: 8000)",
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help="Docker image to use (default: SCILEO_RUNTIME_IMAGE env or btyu24/scileo:v4)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU support even if available",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run the server locally (default: run in Docker container)",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to the MCP server",
    )

    args = parser.parse_args()

    # Choose execution method (Docker by default)
    if args.local:
        run_local(args.transport, args.host, args.port)
    else:
        runner = ScorerDockerRunner()
        success = runner.run_in_docker(
            transport=args.transport,
            host=args.host,
            port=args.port,
            image=args.image,
            extra_args=args.extra_args,
            no_gpu=args.no_gpu,
        )

        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()

