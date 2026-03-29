"""
Docker runner using Python Docker SDK for robust container management.

This module provides functions for running scorer commands in Docker containers
using the Python Docker SDK instead of subprocess calls, enabling better
error handling, resource tracking, and cleanup.
"""

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
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.debug(f"nvidia-smi failed on host with return code {result.returncode}")
                return False
            logger.debug("nvidia-smi check passed on host machine")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.debug(f"nvidia-smi check failed on host: {e}")
            return False
        
        # Step 2: Test GPU access with Docker SDK
        test_container = None
        try:
            test_container = self.client.containers.run(
                DEFAULT_IMAGE,
                "echo 'GPU test successful'",
                device_requests=[
                    docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
                ],
                remove=True,
                detach=False
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
                logger.debug(f"Using host workspace path for nested Docker: {workspace_dir}")
            else:
                logger.warning("Running in container (/workspace) but HOST_WORKSPACE_PATH not set. Using /workspace as-is.")
        
        return workspace_dir
    
    def run_in_docker(
        self, 
        scorer_name: str, 
        running_command: str,
        image: str = DEFAULT_IMAGE,
        enable_gpu: bool = True,
        network_mode: str = "host"
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
        
        # Check for CUDA support
        cuda_supported = enable_gpu and self.detect_cuda_support()
        if cuda_supported:
            logger.debug("CUDA support detected, enabling GPU access in Docker container")
        else:
            logger.debug("CUDA support not detected or disabled, skipping GPU access")
        
        # Prepare container configuration
        volumes = {
            str(workspace_dir): {'bind': '/workspace', 'mode': 'rw'}
        }

        environment = {}

        # Pass CUDA_VISIBLE_DEVICES to container if set in parent process
        # This ensures Docker container respects GPU visibility restrictions
        cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
        if cuda_visible_devices is not None:
            environment['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
            # logger.debug(f"Forwarding CUDA_VISIBLE_DEVICES={cuda_visible_devices} to Docker container")

        # Set user permissions for file ownership
        try:
            uid = os.getuid()
            gid = os.getgid()
            environment.update({
                'HOST_UID': str(uid),
                'HOST_GID': str(gid)
            })
        except AttributeError:
            # Windows doesn't have getuid/getgid
            pass
        
        # Build the full command with setup and cleanup
        full_command = (
            f"cd /workspace && "
            f"bash {scorer_name}/setup.sh && "
            f"{running_command}"
        )
        
        # Add permission fix if we have uid/gid
        if 'HOST_UID' in environment:
            full_command += f" && chown -R {environment['HOST_UID']}:{environment['HOST_GID']} /workspace"
        
        container_config = {
            'image': image,
            'command': ['/bin/bash', '-c', full_command],
            'volumes': volumes,
            'environment': environment,
            'network_mode': network_mode,
            'remove': True,  # Auto-remove container when done
            'detach': True,  # Run in background to enable streaming
            'stdout': True,
            'stderr': True,
            'tty': True
        }
        
        # Add GPU support if available
        if cuda_supported:
            container_config['device_requests'] = [
                docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
            ]
        
        try:
            logger.debug(f"Running command in Docker container: {running_command}")
            logger.debug(f"Full command: {full_command}")

            # Run the container
            self.current_container = self.client.containers.run(**container_config)

            # Stream output in real-time with proper line buffering
            try:
                # Use attach to get a socket for real-time streaming
                output = self.current_container.attach(stdout=True, stderr=True, stream=True, logs=True)

                buffer = ""
                for chunk in output:
                    # Decode chunk and add to buffer
                    chunk_str = chunk.decode('utf-8')
                    buffer += chunk_str

                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        print(line)

                # Print any remaining content in buffer (without newline)
                if buffer.strip():
                    print(buffer.strip())

            except Exception as stream_e:
                logger.debug(f"Error streaming logs: {stream_e}")
                # Fallback to getting all logs at once
                logs = self.current_container.logs()
                if logs:
                    print(logs.decode('utf-8'))

            # Wait for container to complete
            exit_status = self.current_container.wait()
            logger.debug(f"Container exited with status: {exit_status}")

            self.current_container = None

            # Return True if exit code is 0, False otherwise
            return exit_status.get('StatusCode', 1) == 0

        except APIError as e:
            logger.error(f"Docker API error: {e}")
            return False
        except DockerException as e:
            logger.error(f"Docker error: {e}")
            return False
        except KeyboardInterrupt:
            logger.debug("Shutting down container...")
            try:
                # Only stop and remove the container we started
                if self.current_container:
                    logger.debug(f"Stopping and removing container {self.current_container.short_id}...")
                    self.current_container.stop(timeout=5)
                    self.current_container.remove(force=True)
                    self.current_container = None
            except Exception as e:
                logger.debug(f"Error stopping/removing container: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error running Docker container: {e}")
            return False


def run_in_docker(scorer_name: str, running_command: str, image: str = DEFAULT_IMAGE):
    """
    Mount the current directory to the Docker container, run the setup.sh script 
    if it exists, and then run the command using Docker SDK.
    """
    try:
        runner = DockerRunner()
        success = runner.run_in_docker(scorer_name, running_command, image)
        
        if not success:
            logger.critical("Docker container execution failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.debug("Interrupted by user")
        sys.exit(0)


def main():
    """Main entry point for the Docker runner."""
    parser = argparse.ArgumentParser(
        description="Run a command in a Docker container with the scorer workspace mounted (using Docker SDK)"
    )
    parser.add_argument("--scorer_name", type=str, required=True, help="Name of the scorer")
    parser.add_argument("command", nargs="+", help="Command to run in the container")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image to use")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU support")

    args = parser.parse_args()
    
    # Join the command arguments into a single string
    running_command = " ".join(args.command)
    
    # Use the new Docker SDK runner
    run_in_docker(args.scorer_name, running_command, args.image)


if __name__ == "__main__":
    main()
