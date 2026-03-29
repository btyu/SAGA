""
"Docker runner using Python Docker SDK for robust container management.

This module provides functions for running scorer commands in Docker containers
using the Python Docker SDK instead of subprocess calls, enabling better
error handling, resource tracking, and cleanup.
""

import argparse
from loguru import logger
import sys
import os
from pathlib import Path
import subprocess

import docker
from docker.errors import DockerException, APIError


DEFAULT_IMAGE = os.getenv("SCILEO_RUNTIME_IMAGE", "btyu24/scileo:v4")


# Configure logging level to INFO
logger.remove()  # Remove default handler
logger.add(sys.stdout, level="INFO")


class DockerRunner:
    """Docker container runner using Python Docker SDK."""

    def __init__(self):
        """Initialize Docker client."""
        try:
            self.client = docker.from_env()
            # Test connection
            self.client.ping()
            self.current_container = None
            logger.debug("Docker client initialized successfully")
        except DockerException as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise RuntimeError(f"Docker is not available or not running: {e}")

    def detect_cuda_support(self) -> bool:
        """Detect if CUDA is available and Docker supports GPU access."""
        logger.debug("Starting CUDA support detection...")

        # Step 1: Check if nvidia-smi is available on host machine
        try:
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
        test_container = None
        try:
            test_container = self.client.containers.run(
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
            # Cleanup test container if it exists and wasn't auto-removed
            if test_container:
                try:
                    test_container.stop(timeout=5)
                    test_container.remove(force=True)
                    logger.debug("Cleaned up GPU test container")
                except Exception as cleanup_e:
                    logger.debug(f"Error cleaning up GPU test container: {cleanup_e}")
            return False

    def get_workspace_dir(self) -> Path:
        """Get the correct workspace directory, handling Docker-in-Docker scenarios."""
        workspace_dir = Path(__file__).parent.parent.absolute()

        # Handle Docker-in-Docker scenario
        if str(workspace_dir) == "/workspace":
            host_workspace = os.environ.get("HOST_WORKSPACE_PATH")
            if host_workspace:
                workspace_dir = Path(host_workspace)
                logger.debug(
                    f"Using host workspace path for nested Docker: {workspace_dir}"
                )
            else:
                logger.warning(
                    "Running in container (/workspace) but HOST_WORKSPACE_PATH not set. Using /workspace as-is."
                )

        return workspace_dir

    def run_in_docker(
        self,
        scorer_name: str,
        running_command: str,
        image: str = DEFAULT_IMAGE,
        enable_gpu: bool = True,
        network_mode: str = "host",
    ) -> bool:
        """
        Run a command in Docker container with scorer workspace mounted.

        Args:
            scorer_name: Name of the scorer module
            running_command: Command to run inside container
            image: Docker image to use
            enable_gpu: Whether to enable GPU access if available
            network_mode: Docker network mode

        Returns:
            True if successful, False otherwise
        """
        workspace_dir = self.get_workspace_dir()

        # Pull the image first
        logger.debug(f"Pulling Docker image: {image}")
        self.client.images.pull(image)

        # Check for CUDA support
        cuda_supported = enable_gpu and self.detect_cuda_support()
        if cuda_supported:
            logger.debug(
                "CUDA support detected, enabling GPU access in Docker container"
            )
        else:
            logger.debug("CUDA support not detected or disabled, skipping GPU access")

        # Prepare container configuration
        volumes = {str(workspace_dir): {"bind": "/workspace", "mode": "rw"}}

        environment = {}

        # Set user permissions for file ownership
        try:
            uid = os.getuid()
            gid = os.getgid()
            environment.update({"HOST_UID": str(uid), "HOST_GID": str(gid)})
        except AttributeError:
            # Windows doesn't have getuid/getgid
            pass

        # Build the full command with setup and cleanup
        full_command = (
            f"cd /workspace && " f"bash {scorer_name}/setup.sh && " f"{running_command}"
        )

        # Add permission fix if we have uid/gid
        if "HOST_UID" in environment:
            full_command += f" && chown -R {environment['HOST_UID']}:{environment['HOST_GID']} /workspace"

        container_config = {
            "image": image,
            "command": ["/bin/bash", "-c", full_command],
            "volumes": volumes,
            "environment": environment,
            "network_mode": network_mode,
            "remove": True,  # Auto-remove container when done
            "detach": True,  # Run in background to enable streaming
            "stdout": True,
            "stderr": True,
            "tty": True,
        }

        # Add GPU support if available
        if cuda_supported:
            container_config["device_requests"] = [
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]

        try:
            logger.debug(f"Running command in Docker container: {running_command}")
            logger.debug(f"Full command: {full_command}")

            # Run the container
            self.current_container = self.client.containers.run(**container_config)

            # Stream logs until container stops
            for log_line in self.current_container.logs(stream=True):
                sys.stdout.write(log_line.decode("utf-8", errors="replace"))

            exit_status = self.current_container.wait()
            logger.info(f"Container exited with status: {exit_status}")
            return exit_status.get("StatusCode", 1) == 0
        except APIError as e:
            logger.error(f"Docker API error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error running Docker container: {e}")
            return False
        finally:
            if self.current_container:
                try:
                    self.current_container.stop(timeout=10)
                    self.current_container.remove(force=True)
                    logger.debug("Cleaned up Docker container")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up container: {cleanup_error}")


def main():
    parser = argparse.ArgumentParser(description="Run scorer command in Docker")
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
        "--network-mode",
        default="host",
        help="Docker network mode to use (default: host)",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run inside the Docker container",
    )

    args = parser.parse_args()

    if not args.command:
        logger.error("You must specify a command to run inside the Docker container")
        sys.exit(1)

    runner = DockerRunner()
    scorer_dir_name = Path(__file__).parent.name
    success = runner.run_in_docker(
        scorer_name=scorer_dir_name,
        running_command=" ".join(args.command),
        image=args.image,
        enable_gpu=not args.no_gpu,
        network_mode=args.network_mode,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

