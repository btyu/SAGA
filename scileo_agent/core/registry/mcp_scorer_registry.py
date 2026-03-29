"""
MCP Scorer Manager for the SciLeo Agent framework.

This module provides a robust scoring system for managing MCP (Model Context Protocol)
server-based scorers. Each scorer module handles its own containerization.
"""

from typing import Dict, Any, Optional, List, Union
import os
import socket
import time
import threading
import importlib.util
import asyncio
import atexit
import subprocess
from pathlib import Path
from dataclasses import dataclass
from copy import deepcopy

from ...utils.logging import get_logger

logger = get_logger()

# MCP SDK imports
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


@dataclass
class McpServerInfo:
    """Information about a running MCP server."""
    module_name: str
    module_path: Path
    port: int
    container: Optional[object] = None  # subprocess.Popen object or Docker container object
    url: Optional[str] = None
    is_running: bool = False


@dataclass
class ScorerInfo:
    """Information about a scorer within an MCP module."""
    name: str
    description: str
    tool_description: str
    type: str = "candidate-wise"  # "candidate-wise", "population-wise", or "filter"
    population_wise: bool = False  # Deprecated, kept for backward compatibility


def load_module_scorers(module_path: Union[Path, str], return_raw: bool = False) -> Dict[str, Any]:
    scorers = {}

    if isinstance(module_path, str):
        module_path = Path(module_path)

    # Look for __init__.py which should contain scorer definitions
    init_file = module_path / "__init__.py"
    if not init_file.exists():
        logger.critical(f"No __init__.py found in module {module_path}")
        raise FileNotFoundError(f"No __init__.py found in module {module_path}")

    try:
        # Load the module to get scorer info
        module_name = module_path.name
        spec = importlib.util.spec_from_file_location(module_name, init_file)
        if spec is None or spec.loader is None:
            logger.error(f"Could not load module spec from {init_file}")
            return scorers

        temp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(temp_module)

        if return_raw:
            return deepcopy(temp_module.scorers)

        # Look for scorer definitions
        for scorer_name, scorer_config in temp_module.scorers.items():
            # Get type with fallback to population_wise for backward compatibility
            scorer_type = scorer_config.get('type')
            if scorer_type is None:
                # Infer from population_wise for backward compatibility
                population_wise = scorer_config.get('population_wise', False)
                scorer_type = "population-wise" if population_wise else "candidate-wise"
            else:
                population_wise = (scorer_type == "population-wise")

            scorers[scorer_name] = ScorerInfo(
                name=scorer_name,
                description=scorer_config['description'],
                tool_description=scorer_config['tool_description'],
                type=scorer_type,
                population_wise=population_wise
            )

    except Exception as e:
        logger.critical(f"Error loading scorers from {module_path}: {e}")
        raise e

    return scorers


class McpScorerManager:
    """
    Manages MCP scorer modules by running them with `python -m module_name`.
    Each module handles its own Docker container setup.

    This class implements the singleton pattern to ensure only one instance exists.

    This class handles:
    - Adding scorer modules from directories
    - Starting MCP servers by running modules (modules create their own Docker containers)
    - Managing server process lifecycle (start, stop, cleanup)
    - Calling scorers via MCP protocol
    - Automatic cleanup on exit
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, run_in_docker: bool = True):
        """
        Create a new instance only if one doesn't exist (singleton pattern).

        Args:
            run_in_docker: Whether to run scorer modules in Docker containers
        """
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super(McpScorerManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, run_in_docker: bool = True):
        """
        Initialize the MCP scorer manager (only called once due to singleton).

        Args:
            run_in_docker: Whether to run scorer modules in Docker containers (default: True)
        """
        # Prevent re-initialization if already initialized
        if hasattr(self, '_initialized'):
            return

        self.run_in_docker = run_in_docker

        # Storage for modules and servers
        self._modules: Dict[str, Dict[str, ScorerInfo]] = {}  # module_name -> {scorer_name: ScorerInfo}
        self._servers: Dict[str, McpServerInfo] = {}  # module_name -> McpServerInfo
        self._scorer_to_module: Dict[str, str] = {}  # scorer_name -> module_name
        self._module_paths: Dict[str, str] = {}  # module_name -> absolute path string
        self._lock = threading.Lock()

        # Port management
        self._next_port = 8000
        self._used_ports = set()

        # Register cleanup on exit
        atexit.register(self._cleanup_all)

        # Mark as initialized
        self._initialized = True

    def set_run_in_docker(self, run_in_docker: bool) -> None:
        """
        Set whether to run scorer modules in Docker containers.

        Args:
            run_in_docker: Whether to run scorer modules in Docker containers

        Note:
            This only affects new servers started after this setting is changed.
            Existing running servers will not be affected.
        """
        self.run_in_docker = run_in_docker
        logger.debug(f"Set McpScorerManager run_in_docker to {run_in_docker}")

    @classmethod
    def reset_instance(cls):
        """
        Reset the singleton instance (mainly for testing purposes).
        This will clean up the current instance and allow a new one to be created.
        """
        with cls._lock:
            if cls._instance is not None:
                try:
                    cls._instance._cleanup_all()
                except Exception as e:
                    logger.warning(f"Error during cleanup when resetting instance: {e}")
                cls._instance = None

    def __del__(self):
        """Cleanup when instance is garbage collected."""
        try:
            self._cleanup_all()
        except Exception:
            # Ignore errors during cleanup in destructor
            pass

    def _load_module_scorers(self, module_path: Path) -> Dict[str, ScorerInfo]:
        """Load scorer information from a module directory."""
        return load_module_scorers(module_path)

    def _allocate_port(self) -> int:
        """Allocate an available port by checking system availability."""
        # Start from the next port and find one that's both not used by us and available on the system
        port = self._next_port
        max_attempts = 1000  # Prevent infinite loop
        attempts = 0
        
        while attempts < max_attempts:
            # Check if port is not in our used ports and is available on the system
            if port not in self._used_ports and self._is_port_available(port):
                self._used_ports.add(port)
                self._next_port = port + 1
                return port
            
            port += 1
            attempts += 1
        
        # If we couldn't find an available port, raise an error
        raise RuntimeError(f"Could not find an available port after checking {max_attempts} ports starting from {self._next_port}")
    
    def _is_port_available(self, port: int) -> bool:
        """
        Check if a port is available on the system by attempting to bind to it.
        
        Args:
            port: Port number to check
            
        Returns:
            True if port is available, False otherwise
        """
        try:
            # Try to create a socket and bind to the port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('localhost', port))
                return True
        except (OSError, socket.error):
            # Port is not available (already in use or permission denied)
            return False

    def _release_port(self, port: int) -> None:
        """Release a port back to the pool."""
        self._used_ports.discard(port)

    def _start_mcp_server(self, module_name: str) -> bool:
        """
        Start an MCP server for the given module by running `python -m module_name`.
        The module handles its own Docker container setup.

        Args:
            module_name: Name of the module to start

        Returns:
            True if server started successfully, False otherwise
        """
        if module_name not in self._modules:
            logger.error(f"Module '{module_name}' not found")
            return False

        # Check if server is already running
        if module_name in self._servers and self._servers[module_name].is_running:
            logger.debug(f"MCP server for '{module_name}' already running")
            return True

        # Find module path
        if module_name not in self._module_paths:
            logger.error(f"Module path not found for '{module_name}'")
            return False

        module_path = Path(self._module_paths[module_name])
        port = self._allocate_port()

        try:
            # Convert module path to Python module name
            # e.g., /path/to/modules/dna_design/scorer_mcp/stability_scorer_mcp
            #       -> modules.dna_design.scorer_mcp.stability_scorer_mcp
            try:
                # Get the relative path from project root
                project_root = Path.cwd()
                relative_path = module_path.relative_to(project_root)
                # Convert path to module name (replace / with .)
                python_module_name = str(relative_path).replace(os.sep, '.')
            except ValueError:
                # If module_path is not relative to cwd, just use module_name
                python_module_name = module_name

            # Build command to run the module
            # The module's __main__.py will handle Docker container creation (or run locally)
            command = [
                'python', '-m', python_module_name,
                '--transport', 'streamable-http',
                '--host', '0.0.0.0',
                '--port', str(port)
            ]

            # Add --local flag if not running in Docker
            if not self.run_in_docker:
                command.append('--local')

            # If running in Docker and CUDA_VISIBLE_DEVICES is set in the environment,
            # pass it as an environment variable to the subprocess so it can be inherited
            # by the Docker container
            # Note: This is handled by passing the full environment to subprocess.Popen

            logger.debug(f"Starting MCP server for '{module_name}' ('{python_module_name}') on port {port} (Docker: {self.run_in_docker})")
            # logger.debug(f"Command: {' '.join(command)}")
            # logger.debug(f"Python module name: {python_module_name}")

            # Prepare environment variables to pass to subprocess
            # This ensures CUDA_VISIBLE_DEVICES is inherited if set in the main program
            env = os.environ.copy()

            # Log if CUDA_VISIBLE_DEVICES is being propagated to the subprocess
            if 'CUDA_VISIBLE_DEVICES' in env:
                logger.debug(f"Propagating CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} to MCP server subprocess")

            # Start the module process (which will create its own Docker container)
            # Don't capture output to avoid blocking the subprocess
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env
            )

            # Store server info
            server_info = McpServerInfo(
                module_name=module_name,
                module_path=module_path,
                port=port,
                container=process,  # Store the process instead of container
                url=f"http://localhost:{port}/mcp",
                is_running=True
            )

            self._servers[module_name] = server_info

            # Wait for server to be ready by checking port
            if self._wait_for_port_ready(port):
                logger.debug(f"MCP server for '{module_name}' started successfully on port {port}")
                # Give server extra time to fully initialize
                time.sleep(2)
                return True
            else:
                logger.error(f"MCP server for '{module_name}' failed to start properly")
                self._stop_server(module_name)
                return False

        except Exception as e:
            logger.error(f"Failed to start MCP server for '{module_name}': {e}")
            self._release_port(port)
            return False

    def _wait_for_port_ready(self, port: int, timeout: int = 600) -> bool:
        """
        Wait for MCP server to be ready by checking if port is listening.

        Args:
            port: Port to check
            timeout: Maximum wait time in seconds

        Returns:
            True if server is ready, False if timeout or error
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Check if port is listening
                if self._is_port_listening(port):
                    # logger.debug(f"Port {port} is listening")
                    return True

                time.sleep(2)

            except Exception as e:
                logger.error(f"Error checking server readiness: {e}")
                return False

        logger.error(f"Server not ready after {timeout}s timeout")
        return False

    def _wait_for_server_ready(self, container, port: int, timeout: int = 600) -> bool:
        """
        Wait for MCP server to be ready by checking container logs and port.
        (Deprecated - kept for backward compatibility)

        Args:
            container: Docker container object
            port: Port to check
            timeout: Maximum wait time in seconds

        Returns:
            True if server is ready, False if timeout or error
        """
        ready_indicators = [
            "Uvicorn running on",
            "Application startup complete",
            "MCP server with transport:",
            "Started server process"
        ]

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Check if container is still running
                container.reload()
                if container.status != 'running':
                    logger.warning(f"Container stopped while waiting for ready state")
                    return False

                # Check logs for ready indicators
                logs = container.logs(tail=20).decode('utf-8', errors='replace')
                for indicator in ready_indicators:
                    if indicator in logs:
                        # logger.debug(f"Found ready indicator: {indicator}")
                        return True

                # Also check if port is listening
                if self._is_port_listening(port):
                    # logger.debug(f"Port {port} is listening")
                    return True

                time.sleep(2)

            except Exception as e:
                logger.error(f"Error checking server readiness: {e}")
                return False

        logger.error(f"Server not ready after {timeout}s timeout")
        return False

    def _is_port_listening(self, port: int) -> bool:
        """Check if a port is listening."""
        try:
            with socket.create_connection(('localhost', port), timeout=1):
                return True
        except (socket.error, ConnectionRefusedError):
            return False

    def _cleanup_docker_containers_on_port(self, port: int) -> None:
        """
        Clean up any Docker containers that are listening on the specified port.

        This is needed because when we kill the Python subprocess, it doesn't get a chance
        to clean up its Docker container, leaving orphaned containers running.

        Args:
            port: Port number to check for containers
        """
        try:
            import docker
            client = docker.from_env()

            # Get all running containers
            containers = client.containers.list()

            for container in containers:
                try:
                    # Reload container to get fresh attrs
                    container.reload()

                    # Check if this container is using host networking
                    network_mode = container.attrs.get('HostConfig', {}).get('NetworkMode', '')
                    if network_mode != 'host':
                        continue

                    # Get the full command as a single string
                    cmd_list = container.attrs.get('Config', {}).get('Cmd', [])
                    cmd = ' '.join(cmd_list) if cmd_list else ''

                    # Check for MCP server patterns and port
                    # Pattern: "scorer/mcp_server" or "mcp run" with "--port=XXXX" or "--port XXXX"
                    is_mcp_server = (
                        'scorer.mcp_server' in cmd or
                        'scorer/mcp_server' in cmd or
                        'mcp run' in cmd or
                        'streamable-http' in cmd
                    )

                    # Check if this specific port is mentioned
                    has_port = (
                        f'--port={port}' in cmd or
                        f'--port {port}' in cmd or
                        f':{port}' in cmd
                    )

                    if is_mcp_server and has_port:
                        # This looks like one of our MCP containers on this port
                        logger.debug(f"Cleaning up MCP Docker container {container.short_id} on port {port}")
                        try:
                            container.stop(timeout=5)
                            logger.debug(f"Stopped Docker container {container.short_id} (auto-removed)")
                        except Exception as stop_error:
                            # Try to kill if stop fails
                            logger.debug(f"Stop failed, force killing Docker container {container.short_id}: {stop_error}")
                            try:
                                container.kill()
                                logger.debug(f"Killed Docker container {container.short_id} (auto-removed)")
                            except Exception as kill_error:
                                logger.warning(f"Failed to kill container {container.short_id}: {kill_error}")

                except Exception as e:
                    logger.debug(f"Error checking/stopping container {getattr(container, 'short_id', 'unknown')}: {e}")

        except Exception as e:
            logger.debug(f"Could not clean up Docker containers on port {port}: {e}")

    def _stop_server(self, module_name: str) -> bool:
        """
        Stop the MCP server for a module.

        Args:
            module_name: Name of the module

        Returns:
            True if stopped successfully, False otherwise
        """
        # Get server info and remove from servers dict
        server_info = None
        with self._lock:
            if module_name not in self._servers:
                logger.debug(f"No server found for module '{module_name}'")
                return True
            
            server_info = self._servers[module_name]
            del self._servers[module_name]
        
        # Stop the container outside the lock
        return self._stop_server_container(server_info)

    def _stop_server_container(self, server_info: McpServerInfo) -> bool:
        """
        Stop a server process/container without using locks (to avoid deadlock).

        Args:
            server_info: Server info object containing process/container and port info

        Returns:
            True if stopped successfully, False otherwise
        """
        try:
            if server_info.container:
                # Check if it's a subprocess.Popen object (new behavior)
                if hasattr(server_info.container, 'terminate'):
                    logger.debug(f"Terminating process for module '{server_info.module_name}'")
                    server_info.container.terminate()
                    try:
                        # Wait for process to terminate gracefully
                        server_info.container.wait(timeout=5)
                        logger.debug(f"Process terminated for module '{server_info.module_name}'")
                    except:
                        # If it doesn't terminate, kill it
                        logger.debug(f"Force killing process for module '{server_info.module_name}'")
                        server_info.container.kill()
                        server_info.container.wait()
                        logger.debug(f"Process killed for module '{server_info.module_name}'")
                # Otherwise, it's a Docker container object (old behavior)
                else:
                    logger.debug(f"Stopping container for module '{server_info.module_name}'")
                    # Use a shorter timeout to prevent hanging
                    server_info.container.stop(timeout=2)
                    logger.debug(f"Container stopped for module '{server_info.module_name}'")

            # Clean up any Docker containers that might still be running
            # (In case the subprocess was killed before it could clean up its container)
            self._cleanup_docker_containers_on_port(server_info.port)

            # Release port
            self._release_port(server_info.port)
            return True

        except Exception as e:
            logger.warning(f"Error stopping process/container for '{server_info.module_name}': {e}")
            try:
                if server_info.container:
                    if hasattr(server_info.container, 'kill'):
                        logger.debug(f"Force killing process/container for module '{server_info.module_name}'")
                        server_info.container.kill()
                        if hasattr(server_info.container, 'wait'):
                            try:
                                server_info.container.wait(timeout=2)
                            except:
                                pass
                        logger.debug(f"Process/container force killed for module '{server_info.module_name}'")

                # Clean up any Docker containers that might still be running
                self._cleanup_docker_containers_on_port(server_info.port)

                # Release port even if kill failed
                self._release_port(server_info.port)
                return True
            except Exception as e2:
                logger.error(f"Failed to force stop process/container for '{server_info.module_name}': {e2}")
                # Still try to clean up Docker containers and release port
                try:
                    self._cleanup_docker_containers_on_port(server_info.port)
                    self._release_port(server_info.port)
                except Exception:
                    pass
                return False

    async def call_scorer(self, scorer_name: str, samples: List[str], **kwargs) -> List[float]:
        """
        Call a scorer via MCP protocol (async).

        Args:
            scorer_name: Name of the scorer to call
            samples: List of serialized samples
            **kwargs: Additional arguments for the scorer

        Returns:
            List of scores
        """
        # Find which module contains this scorer
        module_name = None
        for mod_name, scorers in self._modules.items():
            if scorer_name in scorers:
                module_name = mod_name
                break

        if module_name is None:
            raise ValueError(f"Scorer '{scorer_name}' not found in any module")

        # Ensure server is running
        if not self._ensure_server_running(module_name):
            raise RuntimeError(f"Failed to start MCP server for module '{module_name}'")

        server_info = self._servers[module_name]

        # Call scorer via MCP (directly call async method)
        return await self._async_call_scorer(server_info.url, scorer_name, samples, **kwargs)

    def _ensure_server_running(self, module_name: str) -> bool:
        """Ensure MCP server is running for the given module."""
        server_info_to_stop = None

        with self._lock:
            if module_name in self._servers and self._servers[module_name].is_running:
                # Check if process/container is actually running
                server_info = self._servers[module_name]
                try:
                    if server_info.container:
                        # Check if it's a subprocess.Popen object (new behavior)
                        if hasattr(server_info.container, 'poll'):
                            # Check if process is still running
                            if server_info.container.poll() is None:
                                # Process is still running
                                return True
                            else:
                                logger.warning(f"Process for '{module_name}' is not running, restarting...")
                                # Remove from servers dict and prepare to stop process outside lock
                                del self._servers[module_name]
                                server_info_to_stop = server_info
                        # Otherwise, it's a Docker container object (old behavior)
                        else:
                            server_info.container.reload()
                            if server_info.container.status == 'running':
                                return True
                            else:
                                logger.warning(f"Container for '{module_name}' is not running, restarting...")
                                # Remove from servers dict and prepare to stop container outside lock
                                del self._servers[module_name]
                                server_info_to_stop = server_info
                except Exception:
                    logger.warning(f"Error checking process/container status for '{module_name}', restarting...")
                    # Remove from servers dict and prepare to stop process/container outside lock
                    del self._servers[module_name]
                    server_info_to_stop = server_info

        # Stop process/container outside the lock to avoid blocking while holding lock
        if server_info_to_stop:
            self._stop_server_container(server_info_to_stop)

        # Now try to start the server
        return self._start_mcp_server(module_name)

    async def _async_call_scorer(self, server_url: str, scorer_name: str, samples: List[str], **kwargs) -> List[float]:
        """Async call to MCP scorer."""
        # logger.debug(f"Connecting to MCP server at {server_url}")

        read_stream = None
        write_stream = None
        session = None

        try:
            # Connect to MCP server using the pattern from the example
            async with streamablehttp_client(server_url) as (
                read_stream,
                write_stream,
                _,
            ):
                # Create a session using the client streams
                async with ClientSession(read_stream, write_stream) as session:
                    # logger.debug("Initializing MCP session...")
                    await session.initialize()
                    # logger.debug("MCP session initialized successfully")

                    # List available tools to verify scorer exists
                    tools = await session.list_tools()
                    tool_names = [tool.name for tool in tools.tools]
                    # logger.debug(f"Available tools: {tool_names}")

                    if scorer_name not in tool_names:
                        raise ValueError(f"Scorer '{scorer_name}' not found. Available: {tool_names}")

                    # Call the scorer tool
                    # logger.debug(f"Calling MCP tool '{scorer_name}' with samples: {samples}")
                    result = await session.call_tool(
                        scorer_name,
                        arguments={
                            'samples': samples,
                            **kwargs
                        }
                    )

                    if result.isError:
                        raise RuntimeError(f"MCP tool error: {result.content}")

                    # Parse results
                    # logger.debug(f"MCP tool result: {result}")

                    # Check if we have structured content first (preferred)
                    if result.structuredContent and 'result' in result.structuredContent:
                        scores = result.structuredContent['result']
                        # logger.debug(f"Using structured content: {scores}, type: {type(scores)}")
                        return scores

                    # Fallback to text content parsing
                    if result.content and len(result.content) > 0:
                        content = result.content[0]
                        # Import types for type checking
                        from mcp import types
                        logger.debug(f"MCP result content type: {type(content)}, content: {content}")
                        if isinstance(content, types.TextContent):
                            import json
                            # logger.debug(f"Raw text content: {content.text}")
                            scores = json.loads(content.text)
                            # logger.debug(f"Parsed scores: {scores}, type: {type(scores)}")
                            return scores
                        else:
                            raise ValueError(f"Unexpected result format: {type(content)}")
                    else:
                        raise ValueError("Empty result from MCP scorer")

        except Exception as e:
            logger.error(f"Error calling MCP scorer '{scorer_name}': {e}")
            raise

    def get_available_scorers(self) -> Dict[str, ScorerInfo]:
        """Get all available scorers across all modules."""
        all_scorers = {}
        for module_name, scorers in self._modules.items():
            for scorer_name, scorer_info in scorers.items():
                all_scorers[scorer_name] = scorer_info
        return all_scorers

    def get_scorers(self) -> List[str]:
        """Get list of all available scorer names (for compatibility with old interface)."""
        all_scorers = []
        for module_name, scorers in self._modules.items():
            for scorer_name in scorers.keys():
                all_scorers.append(scorer_name)
        return all_scorers

    def get_scorer_info(self, scorer_name: str) -> Optional[Dict[str, Any]]:
        """Get scorer info for compatibility with old interface."""
        for module_name, scorers in self._modules.items():
            if scorer_name in scorers:
                scorer_info = scorers[scorer_name]
                return {
                    'module': module_name,
                    'description': scorer_info.description,
                    'tool_description': scorer_info.tool_description,
                    'type': scorer_info.type,
                    'population_wise': scorer_info.population_wise  # Keep for backward compatibility
                }
        return None

    def get_module_path(self, module_name: str) -> Optional[str]:
        """
        Get the absolute path of a registered module.

        Args:
            module_name: Name of the module

        Returns:
            Absolute path string to the module, or None if not found
        """
        return self._module_paths.get(module_name)

    def get_all_module_paths(self) -> Dict[str, str]:
        """
        Get all registered module paths.

        Returns:
            Dictionary mapping module names to their absolute path strings
        """
        return dict(self._module_paths)

    def _cleanup_all(self) -> None:
        """Clean up all running servers and containers."""
        logger.debug("Cleaning up MCP servers and containers...")

        with self._lock:
            module_names = list(self._servers.keys())

        for module_name in module_names:
            try:
                self._stop_server(module_name)
            except Exception as e:
                logger.error(f"Error stopping server for '{module_name}': {e}")

        logger.debug("MCP scorer manager cleanup completed")

    # Update add_module to store module paths
    def add_module(self, module_path: Union[str, Path]) -> bool:
        """Add a single MCP module by path."""
        module_path = Path(module_path).resolve()
        if not module_path.exists():
            logger.error(f"Module path does not exist: {module_path}")
            return False

        if module_path.is_file():
            module_path = module_path.parent

        module_name = module_path.name

        with self._lock:
            if module_name in self._modules:
                logger.warning(f"Module '{module_name}' already added")
                return True

            # Store absolute module path as string
            self._module_paths[module_name] = str(module_path.resolve())

            # Load scorer information
            scorers_info = self._load_module_scorers(module_path)
            if not scorers_info:
                logger.warning(f"No scorers found in module '{module_name}'")
                return False

            self._modules[module_name] = scorers_info
            
            # Track scorer-to-module mapping
            for scorer_name in scorers_info.keys():
                self._scorer_to_module[scorer_name] = module_name
            
            return True

    def unregister_scorer(self, scorer_name: str) -> bool:
        """
        Unregister a specific MCP scorer.
        
        If this is the last scorer from a module, automatically stops the module server.
        
        Args:
            scorer_name: Name of the scorer to unregister
            
        Returns:
            True if scorer was found and removed, False otherwise
        """
        # Check if scorer exists and get module info
        module_name = None
        server_info = None
        should_stop_server = False
        
        with self._lock:
            if scorer_name not in self._scorer_to_module:
                logger.debug(f"Scorer '{scorer_name}' not found in MCP manager")
                return False
            
            module_name = self._scorer_to_module[scorer_name]
            
            # Remove scorer from module
            if module_name in self._modules and scorer_name in self._modules[module_name]:
                del self._modules[module_name][scorer_name]
                logger.debug(f"Removed scorer '{scorer_name}' from module '{module_name}'")
            
            # Remove from scorer-to-module mapping
            del self._scorer_to_module[scorer_name]
            
            # Check if this was the last scorer in the module
            if module_name in self._modules and not self._modules[module_name]:
                # No more scorers in this module, stop the server
                logger.debug(f"No more scorers in module '{module_name}', stopping server")
                should_stop_server = True
                # Get server info before removing
                if module_name in self._servers:
                    server_info = self._servers[module_name]
                    del self._servers[module_name]
                # Remove empty module
                del self._modules[module_name]
        
        # Stop the server outside the lock to avoid deadlock
        if should_stop_server and server_info:
            try:
                self._stop_server_container(server_info)
            except Exception as e:
                logger.warning(f"Error stopping server for module '{module_name}': {e}")
        
        logger.debug(f"Unregistered MCP scorer '{scorer_name}' from module '{module_name}'")
        return True

    def unregister_module(self, module_name: str) -> bool:
        """
        Unregister all scorers from a module and stop its server.
        
        Args:
            module_name: Name of the module to unregister
            
        Returns:
            True if module was found and removed, False otherwise
        """
        # First, check if module exists and get server info
        server_info = None
        with self._lock:
            if module_name not in self._modules:
                logger.debug(f"Module '{module_name}' not found in MCP manager")
                return False
            
            # Get all scorers from this module
            scorers_in_module = list(self._modules[module_name].keys())
            
            # Get server info before removing from servers dict
            if module_name in self._servers:
                server_info = self._servers[module_name]
            
            # Remove all scorers from scorer-to-module mapping
            for scorer_name in scorers_in_module:
                if scorer_name in self._scorer_to_module:
                    del self._scorer_to_module[scorer_name]
            
            # Remove module
            del self._modules[module_name]

            # Remove module path
            if module_name in self._module_paths:
                del self._module_paths[module_name]

            # Remove from servers dict
            if module_name in self._servers:
                del self._servers[module_name]
        
        # Stop the server outside the lock to avoid deadlock
        if server_info:
            try:
                self._stop_server_container(server_info)
            except Exception as e:
                logger.warning(f"Error stopping server for module '{module_name}': {e}")
        
        logger.debug(f"Unregistered module '{module_name}' with {len(scorers_in_module)} scorers")
        return True

    def stop_all_mcp_servers(self) -> None:
        """
        Stop all running MCP servers without unregistering scorers or modules.

        This is useful for freeing up memory at the end of iterations while keeping
        scorer and module registrations intact. Servers will be restarted automatically
        when needed.
        """
        # Get list of server infos to stop (outside the lock to avoid deadlock)
        server_infos = []
        with self._lock:
            server_infos = list(self._servers.values())

        # Stop all servers outside the lock
        stopped_count = 0
        for server_info in server_infos:
            try:
                self._stop_server_container(server_info)
                stopped_count += 1
            except Exception as e:
                logger.warning(f"Error stopping server for module '{server_info.module_name}': {e}")

        with self._lock:
            # Clear all servers (but keep modules, scorers, and paths intact)
            self._servers.clear()

        if stopped_count > 0:
            logger.debug(f"Stopped {stopped_count} MCP server(s) to free memory")

    def clear_all_scorers(self) -> None:
        """
        Clear all scorers, modules, and stop all servers.

        This properly cleans up all resources including stopping running servers.
        """
        # Get list of server infos to stop (outside the lock to avoid deadlock)
        server_infos = []
        with self._lock:
            server_infos = list(self._servers.values())

        # Stop all servers outside the lock
        for server_info in server_infos:
            try:
                self._stop_server_container(server_info)
            except Exception as e:
                logger.warning(f"Error stopping server for module '{server_info.module_name}': {e}")

        with self._lock:
            # Clear all scorer-to-module mappings
            self._scorer_to_module.clear()

            # Clear all modules
            self._modules.clear()

            # Clear all module paths
            self._module_paths.clear()

            # Clear all servers
            self._servers.clear()

            # Reset port tracking
            self._used_ports.clear()
            self._next_port = 8000

            logger.debug("Cleared all MCP scorers, modules, and servers")


def get_mcp_scorer_manager() -> McpScorerManager:
    """
    Get the singleton MCP scorer manager instance.
    
    This function now simply returns the singleton instance, as the singleton
    pattern is handled within the McpScorerManager class itself.
    """
    return McpScorerManager()