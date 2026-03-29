from mcp.server.fastmcp import FastMCP
import logging
import sys
import inspect
from typing import get_type_hints

from .base import Scorer

# Configure logging to output to stderr so it's visible in Docker logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
    force=True
)
logger = logging.getLogger(__name__)


# Create MCP server
mcp = FastMCP("mcp-server")

# Cache scorer instance to avoid recreating it for each call
_scorer_instance_cache = None
_scorer_instance_lock = None

def _get_scorer_instance(scorer_class):
    """Get or create a cached scorer instance."""
    global _scorer_instance_cache, _scorer_instance_lock
    if _scorer_instance_cache is None:
        import threading
        if _scorer_instance_lock is None:
            _scorer_instance_lock = threading.Lock()
        with _scorer_instance_lock:
            # Double-check after acquiring lock
            if _scorer_instance_cache is None:
                _scorer_instance_cache = scorer_class()
    return _scorer_instance_cache


def register_scorers_as_tools(scorer_class):
    """
    Register all individual scorers in a scorer class as separate MCP tools.
    
    This function introspects the scorer class to find all methods decorated with @scorer
    and registers each one as a separate MCP tool with proper input/output schema.
    
    Args:
        scorer_class: A scorer class that inherits from BaseScorer
    """
    # Create an instance to access the scorers registry
    scorer_instance = scorer_class()
    
    # Get all available scorers from the registry
    available_scorers = scorer_instance.get_available_scorers()
    
    registered_tools = []
    
    for scorer_name, scorer_info in available_scorers.items():
        # Get the actual scorer method
        scorer_method = scorer_instance.scorers[scorer_name]['method']
        
        # Get method signature and type hints
        signature = inspect.signature(scorer_method)
        type_hints = get_type_hints(scorer_method)
        
        # Create a wrapper function for this specific scorer
        def create_wrapper(method, name):
            def wrapper_func(*args, **kwargs):
                """Dynamically generated wrapper function for individual scorer method."""
                import sys
                samples = kwargs.get('samples', args[0] if args else [])
                num_samples = len(samples) if isinstance(samples, list) else 1
                print(f"[LOCAL_SIMILARITY] Starting {name} scoring for {num_samples} samples", file=sys.stderr, flush=True)
                sys.stderr.flush()
                
                # Reuse cached scorer instance (FAISS index is class-level cached)
                scorer_instance = _get_scorer_instance(scorer_class)
                # Get the specific method from the cached instance
                scorer_method = scorer_instance.scorers[name]['method']
                
                try:
                    result = scorer_method(*args, **kwargs)
                    print(f"[LOCAL_SIMILARITY] Completed {name} scoring for {num_samples} samples", file=sys.stderr, flush=True)
                    sys.stderr.flush()
                    return result
                except Exception as e:
                    print(f"[LOCAL_SIMILARITY] Error in {name}: {e}", file=sys.stderr, flush=True)
                    sys.stderr.flush()
                    raise
            
            # Copy the original docstring and annotations
            wrapper_func.__doc__ = method.__doc__
            wrapper_func.__annotations__ = {
                k: v for k, v in type_hints.items() 
                if k != 'self'  # Exclude 'self' parameter
            }
            
            # Create the correct signature (excluding 'self')
            new_params = [
                param for param_name, param in signature.parameters.items() 
                if param_name != 'self'
            ]
            wrapper_func.__signature__ = signature.replace(parameters=new_params)
            
            # Set the function name dynamically
            wrapper_func.__name__ = f"{name}_scorer"
            
            return wrapper_func
        
        # Create wrapper for this scorer
        wrapper_func = create_wrapper(scorer_method, scorer_name)
        
        # Register as MCP tool with proper metadata
        tool_decorator = mcp.tool(
            name=scorer_name,
            title=f"{scorer_name} scorer",
            description=scorer_info['description'] or f"Calculate {scorer_name} score"
        )
        
        registered_tool = tool_decorator(wrapper_func)
        registered_tools.append(registered_tool)
        
        logger.info(f"Registered MCP tool: {scorer_name}")
    
    return registered_tools


# Register all scorer tools
register_scorers_as_tools(Scorer)


def run_server(transport: str, host: str = "localhost", port: int = 8000):
    """Run the MCP server with the specified transport and configuration"""
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
    
    parser = argparse.ArgumentParser(description="Run MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="streamable-http",
        help="Transport method for the MCP server (default: streamable-http)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to use for HTTP-based transports (default: 8000)"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind to for HTTP-based transports (default: localhost)"
    )
    
    args = parser.parse_args()
    logger.info(f"{args}")
    run_server(args.transport, args.host, args.port)

