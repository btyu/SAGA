#!/usr/bin/env python3
"""
Main entry point for SmallWorld similarity scorer module.
Supports running as: python -m <scorer> [options]
"""

import os
import argparse
from loguru import logger
import warnings
import sys
from pathlib import Path
from typing import Optional, List

import docker
from docker.errors import DockerException, APIError


DEFAULT_IMAGE = os.getenv("SCILEO_RUNTIME_IMAGE", "btyu24/scileo:v4")


class ScorerDockerRunner:
    """Scorer Docker runner using Python Docker SDK."""

    def __init__(self):
        try:
            self.client = docker.from_env()
            self.client.ping()
            logger.debug("Docker client initialized successfully")
            self.current_container = None
        except DockerException as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise RuntimeError(f"Docker is not available or not running: {e}")

    def detect_cuda_support(self) -> bool:
        logger.debug("Starting CUDA support detection...")
        try:
            import subprocess

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

        try:
            self.client.containers.run(
                DEFAULT_IMAGE,
                "echo 'GPU test successful'",
                device_requests=[
                    docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
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
        if transport == "stdio":
            warnings.warn(
                "`stdio` transport may not work with docker run. Use streamable-http instead."
            )

        if extra_args is None:
            extra_args = []

        module_dir = Path(__file__).parent.absolute()

        cuda_supported = False if no_gpu else self.detect_cuda_support()
        if no_gpu:
            logger.info("GPU support disabled by --no-gpu flag")
        elif cuda_supported:
            logger.info("CUDA support detected, enabling GPU access in Docker container")
        else:
            logger.info("CUDA support not detected, running without GPU access")

        volumes = {str(module_dir): {"bind": "/workspace/scorer", "mode": "rw"}}
        environment = {}

        try:
            uid = os.getuid()
            gid = os.getgid()
            environment.update({"HOST_UID": str(uid), "HOST_GID": str(gid)})
        except AttributeError:
            pass

        mcp_command = f"python -m scorer.mcp_server --transport={transport}"
        if transport == "streamable-http":
            mcp_command += f" --host=0.0.0.0 --port={port}"

        if extra_args:
            mcp_command += " " + " ".join(extra_args)

        full_command = f"bash scorer/setup.sh && {mcp_command}"

        if "HOST_UID" in environment:
            full_command += f" && chown -R {environment['HOST_UID']}:{environment['HOST_GID']} /workspace"

        container_config = {
            "image": image,
            "command": ["/bin/bash", "-c", full_command],
            "volumes": volumes,
            "environment": environment,
            "ports": {f"{port}/tcp": port},
            "remove": True,
            "detach": True,
            "stdout": True,
            "stderr": True,
            "tty": True,
        }

        if cuda_supported:
            container_config["device_requests"] = [
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]

        container_cleaned = False
        try:
            logger.info(
                f"Starting SmallWorld MCP server in Docker container with transport: {transport}"
            )
            if transport == "streamable-http":
                logger.info(f"Server will be available at http://{host}:{port}")

            logger.debug(f"Full command: {full_command}")
            self.current_container = self.client.containers.run(**container_config)
            logger.info(
                f"Started MCP server container: {self.current_container.short_id}"
            )

            try:
                log_buffer = ""
                for log_chunk in self.current_container.logs(stream=True, follow=True):
                    if log_chunk:
                        chunk_str = log_chunk.decode("utf-8", errors="replace")
                        log_buffer += chunk_str
                        while "\n" in log_buffer:
                            line, log_buffer = log_buffer.split("\n", 1)
                            if line.strip():
                                print(line.rstrip())
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, stopping container...")
                raise
            except Exception as stream_error:
                logger.warning(f"Error streaming container logs: {stream_error}")

            result = self.current_container.wait()
            exit_code = result["StatusCode"]

            if exit_code != 0:
                logger.error(f"Container exited with code {exit_code}")
                return False

            logger.info("MCP server container completed")
            return True

        except KeyboardInterrupt:
            logger.info("Shutting down server...")
            self._cleanup_current_container()
            container_cleaned = True
            raise
        except APIError as e:
            logger.error(f"Docker API error: {e}")
            if not container_cleaned:
                self._cleanup_current_container()
            return False
        except DockerException as e:
            logger.error(f"Docker error: {e}")
            if not container_cleaned:
                self._cleanup_current_container()
            return False
        except Exception as e:
            logger.error(f"Unexpected error running Docker container: {e}")
            if not container_cleaned:
                self._cleanup_current_container()
            return False
        finally:
            if not container_cleaned:
                self._cleanup_current_container()

    def _cleanup_current_container(self):
        if self.current_container:
            try:
                logger.info(f"Stopping container {self.current_container.short_id}...")
                self.current_container.stop(timeout=5)
                logger.info("Container stopped successfully (auto-removed)")
            except Exception as e:
                logger.error(f"Error stopping container: {e}")
                try:
                    self.current_container.kill()
                    logger.warning("Force killed container (auto-removed)")
                except Exception as e2:
                    logger.error(f"Failed to force kill container: {e2}")
            finally:
                self.current_container = None


def run_in_docker(
    transport: str,
    extra_args: Optional[List[str]] = None,
    host: str = "localhost",
    port: int = 8000,
    image: str = DEFAULT_IMAGE,
    no_gpu: bool = False,
):
    runner = None
    try:
        runner = ScorerDockerRunner()
        success = runner.run_in_docker(transport, host, port, extra_args, image, no_gpu)
        if not success:
            logger.error("MCP server Docker container execution failed")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        if runner:
            runner._cleanup_current_container()
        sys.exit(0)


def run_local(transport: str, host: str = "localhost", port: int = 3000):
    from .mcp_server import run_server

    run_server(transport, host, port)


def main():
    parser = argparse.ArgumentParser(
        description="Run SmallWorld similarity MCP server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scorer --transport streamable-http
  python -m scorer --local --transport stdio
        """,
    )

    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="streamable-http",
        help="Transport method for the MCP server",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run the server locally (default: Docker)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to use for HTTP-based transports",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind to for HTTP-based transports",
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help=f"Docker image to use (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU support even if CUDA is available",
    )

    args, extra_args = parser.parse_known_args()

    if args.local:
        run_local(args.transport, args.host, args.port)
    else:
        run_in_docker(
            args.transport, extra_args, args.host, args.port, args.image, args.no_gpu
        )


if __name__ == "__main__":
    main()

