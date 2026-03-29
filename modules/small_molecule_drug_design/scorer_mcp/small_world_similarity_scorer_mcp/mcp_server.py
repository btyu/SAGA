from mcp.server.fastmcp import FastMCP
import logging
import sys
import inspect
from typing import get_type_hints

from .base import Scorer

logger = logging.getLogger(__name__)


mcp = FastMCP("small-world-scorer")


def register_scorers_as_tools(scorer_class):
    """
    Register all scorer methods as separate MCP tools.
    """
    scorer_instance = scorer_class()
    available_scorers = scorer_instance.get_available_scorers()
    registered_tools = []

    for scorer_name, scorer_info in available_scorers.items():
        scorer_method = scorer_instance.scorers[scorer_name]["method"]
        signature = inspect.signature(scorer_method)
        type_hints = get_type_hints(scorer_method)

        def create_wrapper(method, name):
            def wrapper_func(*args, **kwargs):
                fresh_instance = scorer_class()
                fresh_method = fresh_instance.scorers[name]["method"]
                return fresh_method(*args, **kwargs)

            wrapper_func.__doc__ = method.__doc__
            wrapper_func.__annotations__ = {
                k: v for k, v in type_hints.items() if k != "self"
            }
            new_params = [
                param
                for param_name, param in signature.parameters.items()
                if param_name != "self"
            ]
            wrapper_func.__signature__ = signature.replace(parameters=new_params)
            wrapper_func.__name__ = f"{name}_scorer"
            return wrapper_func

        wrapper_func = create_wrapper(scorer_method, scorer_name)

        tool_decorator = mcp.tool(
            name=scorer_name,
            title=f"{scorer_name} scorer",
            description=scorer_info["description"]
            or f"Calculate {scorer_name} score",
        )

        registered_tool = tool_decorator(wrapper_func)
        registered_tools.append(registered_tool)
        logger.info(f"Registered MCP tool: {scorer_name}")

    return registered_tools


register_scorers_as_tools(Scorer)


def run_server(transport: str, host: str = "localhost", port: int = 8000):
    logger.info(f"Starting MCP server with transport: {transport}")

    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "streamable-http":
        mcp.settings.host = host
        mcp.settings.port = port
        mcp.run(transport=transport)
    else:
        logger.error(f"Unsupported transport: {transport}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SmallWorld MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="streamable-http",
        help="Transport method",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transport",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host for HTTP transport",
    )

    args = parser.parse_args()
    logger.info(args)
    run_server(args.transport, args.host, args.port)

