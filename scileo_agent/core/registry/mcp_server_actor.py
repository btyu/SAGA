"""
Ray actor for managing MCP server Docker containers.

This actor runs in the main process and manages the lifecycle of MCP server containers,
providing server URLs to Ray workers without requiring them to start their own containers.
"""

import ray
from pathlib import Path
from typing import Dict, Optional
from loguru import logger


@ray.remote
class McpServerActor:
    """
    Ray actor that manages MCP server Docker containers.

    This actor:
    - Starts MCP server Docker containers in the main process
    - Provides server URLs to worker processes
    - Handles server health checks
    - Manages clean shutdown of containers
    """

    def __init__(self):
        """Initialize the MCP server actor."""
        self.servers = {}  # module_name -> {"url": str, "port": int, "manager": McpScorerManager}
        self.mcp_manager = None

    def start_server(self, module_path: str, serializer_name: str) -> Dict[str, str]:
        """
        Start an MCP server for the given module.

        Args:
            module_path: Path to the MCP module
            serializer_name: Name of the serializer to use

        Returns:
            Dictionary with server info: {"url": str, "port": int, "module_name": str}
        """
        from scileo_agent.core.registry.mcp_scorer_registry import McpScorerManager

        module_name = Path(module_path).name

        # Check if server already running
        if module_name in self.servers:
            logger.info(f"MCP server for '{module_name}' already running at {self.servers[module_name]['url']}")
            return self.servers[module_name]

        # Initialize MCP manager if needed
        if self.mcp_manager is None:
            self.mcp_manager = McpScorerManager()

        # Add module to manager
        success = self.mcp_manager.add_module(module_path)
        if not success:
            raise RuntimeError(f"Failed to add MCP module at path: {module_path}")

        # Start the server
        server_started = self.mcp_manager._ensure_server_running(module_name)
        if not server_started:
            raise RuntimeError(f"Failed to start MCP server for module '{module_name}'")

        # Get server info
        server_info = self.mcp_manager._servers[module_name]

        # Store server info
        self.servers[module_name] = {
            "url": server_info.url,
            "port": server_info.port,
            "module_name": module_name
        }

        logger.info(f"✓ MCP server for '{module_name}' started successfully at {server_info.url}")

        return self.servers[module_name]

    def get_server_url(self, module_name: str) -> Optional[str]:
        """
        Get the URL for a running MCP server.

        Args:
            module_name: Name of the module

        Returns:
            Server URL or None if not running
        """
        if module_name in self.servers:
            return self.servers[module_name]["url"]
        return None

    def get_all_servers(self) -> Dict[str, Dict]:
        """
        Get information about all running servers.

        Returns:
            Dictionary mapping module names to server info
        """
        return self.servers.copy()

    def shutdown(self):
        """Shutdown all MCP servers and clean up Docker containers."""
        if self.mcp_manager is None:
            return

        logger.info(f"Shutting down {len(self.servers)} MCP servers...")

        for module_name in list(self.servers.keys()):
            try:
                if module_name in self.mcp_manager._servers:
                    server_info = self.mcp_manager._servers[module_name]
                    self.mcp_manager._stop_server_container(server_info)
                    logger.info(f"✓ Stopped MCP server for '{module_name}'")
            except Exception as e:
                logger.error(f"Error stopping MCP server for '{module_name}': {e}")

        self.servers.clear()
        logger.info("All MCP servers shut down")

    def health_check(self, module_name: str) -> bool:
        """
        Check if an MCP server is healthy.

        Args:
            module_name: Name of the module to check

        Returns:
            True if server is healthy, False otherwise
        """
        if module_name not in self.servers:
            return False

        if self.mcp_manager is None:
            return False

        try:
            import requests
            url = self.servers[module_name]["url"]
            response = requests.get(url, timeout=2.0)
            return response.status_code in [200, 404, 405]  # Server responding
        except Exception:
            return False
