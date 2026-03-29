import os
import asyncio
import json
import uuid
from typing import Optional, List, Dict, Any, Union
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AgentDefinition, CLINotFoundError, ProcessError, CLIJSONDecodeError
import pickle
import subprocess

import docker
from docker.errors import APIError as DockerAPIError
from scileo_agent.utils import get_logger
from scileo_agent.utils.llm import YAMLLLMConfig


logger = get_logger()


def setup_claude_code_credentials(model_name, models: dict, credentials: dict):
    """Set up environment variables for Claude Code based on model configuration."""
    model_config = models.get(model_name)
    if not model_config:
        raise ValueError(f"Model {model_name} not found in models config")
        
    tag = model_config.get("credentials")
    provider = model_config.get("provider")
    
    if tag is None:
        return
    
    credential = credentials.get(tag)
    if credential is None:
        raise ValueError(f"Credential tag {tag} not found in credentials")

    # Unset the environment variables if they are set
    for key in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_USE_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME", "CLAUDE_CODE_OAUTH_TOKEN"):
        os.environ.pop(key, None)

    if provider == "claude_code":
        api_key = credential.get("api_key")
        if api_key:
            os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = api_key
    elif provider == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = credential.get("api_key")
    elif provider == "bedrock":
        os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"
        os.environ["AWS_ACCESS_KEY_ID"] = credential.get("aws_access_key_id")
        os.environ["AWS_SECRET_ACCESS_KEY"] = credential.get("aws_secret_access_key")
        os.environ["AWS_REGION_NAME"] = credential.get("aws_region_name")
    else:
        raise ValueError(f"Provider {provider} not supported")


def _detect_provider():
    """
    Detect and log which provider is being used based on Claude Code SDK environment variables.
    """
    # Check Claude Code specific environment variables
    use_bedrock = os.getenv('CLAUDE_CODE_USE_BEDROCK') == '1'
    use_vertex = os.getenv('CLAUDE_CODE_USE_VERTEX') == '1'
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

    provider = None

    if use_bedrock:
        provider = "bedrock"
        logger.debug("☁️ Using Amazon Bedrock (CLAUDE_CODE_USE_BEDROCK=1)")
    elif use_vertex:
        provider = "vertex"
        logger.debug("🌐 Using Google Vertex AI (CLAUDE_CODE_USE_VERTEX=1)")
    elif anthropic_api_key:
        provider = "anthropic"
        logger.debug("🔑 Using Anthropic API with API key")
    else:
        provider = "claude_code"
        logger.debug("👤 Using Anthropic Claude Pro account")

    return provider


class ClaudeAgent:
    """
    Self-contained Claude agent that handles configuration, credentials, and execution.
    
    This class encapsulates all Claude Code functionality including:
    - Model configuration and credential management
    - Docker-based execution with proper isolation
    - Native execution fallback
    """
    
    def __init__(
        self,
        model_name: str = "claude_code/claude-sonnet-4-5-20250929",
        models_file: str = "llm_configs/claude_code.yaml",
        credentials_file: str = "llm_configs/credentials.yaml",
        run_in_docker: bool = True,
        docker_image: str = "btyu24/scileo:claude-agent-runner-251117",
        system_prompt: Optional[Union[str, Dict[str, Any]]] = None,
        agents: Optional[Dict[str, AgentDefinition]] = None
    ):
        """
        Initialize Claude Code agent with configuration.

        Args:
            model_name: Key in models config (e.g., "bedrock/claude-sonnet-4-20250514")
            models_file: Path to models configuration file
            credentials_file: Path to credentials configuration file
            run_in_docker: Whether to run Claude Code in Docker container
            docker_image: Docker image name for containerized execution (prebuilt)
            system_prompt: System prompt configuration. Can be:
                         - None: defaults to Claude Code preset
                         - str: custom prompt string
                         - dict: {"type": "preset", "preset": "claude_code"} for Claude Code's system prompt,
                                or add "append" key to extend the preset
            agents: Optional dictionary of subagent definitions (name -> AgentDefinition)
        """
        self.model_name = model_name
        self.run_in_docker = run_in_docker
        self.docker_image = docker_image

        # Use default Claude Code system prompt if none provided
        self.system_prompt = system_prompt if system_prompt is not None else {"type": "preset", "preset": "claude_code"}
        self.tools = ["Read", "Grep", "Glob", "Write", "Edit", "MultiEdit", "Delete", "Bash", "WebSearch", "WebFetch", "Task", "NotebookEdit", "TodoWrite", "BashOutput", "KillBash", "ExitPlanMode", "ListMcpResources", "ReadMcpResource"]
        self.agents = agents or {}

        # Container management - now tracked per-run instead of per-instance
        # This allows the same agent to handle multiple parallel runs
        self._containers = {}  # Dict[run_id, container] for tracking multiple containers
        self._container_lock = None  # Will be initialized in async context if needed

        # Session-specific token tracking
        self._session_id = str(uuid.uuid4())

        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            if run_in_docker:
                raise RuntimeError(f"Failed to initialize Docker client: {e}")
            else:
                self.docker_client = None

        # Load configuration
        try:
            self.yaml_config = YAMLLLMConfig(
                models_file=models_file,
                credentials_file=credentials_file
            )
        except Exception as e:
            raise ValueError(f"Failed to load Claude Code configuration: {e}")
        
        # Get model configuration
        model_config = self.yaml_config.models.get(model_name)
        if not model_config:
            raise ValueError(f"Model configuration '{model_name}' not found in {models_file}")
        
        self.actual_model = model_config.get("model")
        self.provider = model_config.get("provider")

        if not self.actual_model:
            raise ValueError(f"Model configuration '{model_name}' missing 'model' field")
        
        # Set up credentials
        self._setup_credentials()

        # logger.debug(f"Initialized Claude Code agent with model: {self.actual_model} (provider: {self.provider})")

    def __del__(self):
        """Destructor to ensure container cleanup on object destruction."""
        try:
            self._cleanup_all_containers()
        except Exception:
            pass  # Ignore errors during cleanup in destructor
    
    def _setup_credentials(self):
        """Set up environment credentials for the configured model."""
        try:
            setup_claude_code_credentials(
                self.model_name, 
                self.yaml_config.models, 
                self.yaml_config.credentials
            )
            # logger.debug(f"Credentials set up for model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to set up credentials for {self.model_name}: {e}")
            raise

    def _cleanup_container(self, run_id: str):
        """
        Force stop and remove a specific container by run_id.

        Args:
            run_id: The unique identifier for this run
        """
        container = self._containers.get(run_id)
        if container is None:
            return

        container_id = None
        try:
            container_id = container.short_id
            # Reload container to get current status
            container.reload()

            # Stop container if it's still running
            if container.status in ['running', 'paused']:
                logger.info(f"Stopping container {container_id} (run_id: {run_id})...")
                container.stop(timeout=10)
                logger.info(f"Container {container_id} stopped")

                # After stopping, give auto-remove a moment to work
                import time
                time.sleep(0.5)

                # Check if container still exists after auto-remove
                try:
                    container.reload()
                    # If we reach here, container still exists, so remove it manually
                    logger.debug(f"Container {container_id} still exists, removing manually")
                    container.remove(force=True)
                    logger.debug(f"Container {container_id} manually removed")
                except docker.errors.NotFound:
                    # Container was auto-removed, which is expected
                    logger.debug(f"Container {container_id} auto-removed successfully")
            else:
                # Container is not running, try to remove it
                try:
                    container.remove(force=True)
                    logger.debug(f"Container {container_id} removed (was in {container.status} state)")
                except docker.errors.NotFound:
                    logger.debug(f"Container {container_id} already removed")
                except docker.errors.APIError as api_error:
                    if "removal of container" in str(api_error) and "is already in progress" in str(api_error):
                        logger.debug(f"Container {container_id} removal already in progress (auto-remove)")
                    else:
                        logger.warning(f"API error removing container {container_id}: {api_error}")

        except docker.errors.NotFound:
            # Container already gone
            logger.debug(f"Container {container_id or 'unknown'} not found (already removed)")
        except Exception as e:
            # Container might already be gone or auto-removed
            logger.debug(f"Container cleanup completed (container may be auto-removed): {e}")
        finally:
            # Remove from tracking dict
            if run_id in self._containers:
                del self._containers[run_id]

    def _cleanup_all_containers(self):
        """Clean up all tracked containers (called on destruction)."""
        run_ids = list(self._containers.keys())
        for run_id in run_ids:
            try:
                self._cleanup_container(run_id)
            except Exception as e:
                logger.debug(f"Error cleaning up container for run_id {run_id}: {e}")
    
    async def run(
        self,
        user_prompt: str,
        cwd: str,
        add_dirs: Optional[List[str]] = None,
        max_thinking_tokens: int = 4096,
        permission_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run Claude Code asynchronously (async version for parallel execution).

        This method allows multiple Claude Code sessions to run in parallel.

        Args:
            user_prompt: The prompt to send to Claude
            cwd: Working directory for Claude Code
            add_dirs: Additional directories to include
            max_thinking_tokens: Maximum thinking tokens
            permission_mode: Permission mode for Claude Code.
                           If None, defaults to "bypassPermissions".
                           Note: IS_SANDBOX=1 is set in Docker to allow bypassPermissions with root.
                           Valid options: acceptEdits, bypassPermissions, default, plan.

        Returns:
            Dict: Token usage statistics including:
                - input_tokens: Number of input tokens
                - output_tokens: Number of output tokens
                - cache_creation_input_tokens: Number of cache creation tokens
                - cache_read_input_tokens: Number of cache read tokens
                - total_tokens: Total tokens used
                - cost: Cost in USD
                - model: Model name used
        """
        _detect_provider()

        if add_dirs is None:
            add_dirs = []

        # Set default permission mode based on execution environment
        if permission_mode is None:
            permission_mode = "bypassPermissions"

        if self.run_in_docker:
            # Run Docker operations in thread pool to avoid blocking
            return await asyncio.to_thread(
                self._run_in_docker,
                user_prompt, cwd, add_dirs, max_thinking_tokens, permission_mode
            )
        else:
            # Native execution already has async implementation
            return await self._run_native_async(
                user_prompt, cwd, add_dirs, max_thinking_tokens, permission_mode
            )
    
    async def _run_native_async(
        self,
        user_prompt: str,
        cwd: str,
        add_dirs: List[str],
        max_thinking_tokens: int,
        permission_mode: str
    ) -> Dict[str, Any]:
        """Run Claude Code natively (async implementation)."""

        logger.debug(f"Running Claude Code natively with model: {self.actual_model}")

        # Container to store usage info
        usage_result = {}

        try:
            options = ClaudeAgentOptions(
                model=self.actual_model,  # Use actual model identifier
                max_thinking_tokens=max_thinking_tokens,
                cwd=cwd,
                add_dirs=add_dirs,
                permission_mode=permission_mode,
                system_prompt=self.system_prompt,
                allowed_tools=self.tools,
                agents=self.agents if self.agents else None,
            )

            async with ClaudeSDKClient(options=options) as client:
                await client.query(user_prompt)

                messages = []
                async for message in client.receive_response():
                    if hasattr(message, 'content'):
                        for block in message.content:
                            if hasattr(block, 'text'):
                                logger.debug(block.text.strip())
                    messages.append(message)

                last_message = messages[-1]
                if not hasattr(last_message, 'is_error'):
                    with open("messages.pkl", "wb") as f:
                        pickle.dump(messages, f)
                    raise RuntimeError(f"Claude Code doesn't return a valid ResultMessage. Messages are saved to `messages.pkl`. The last message is {last_message}")
                if last_message.is_error:
                    with open("messages_error.pkl", "wb") as f:
                        pickle.dump(messages, f)
                    assistant_message = messages[-2]
                    error_msg = assistant_message.content[-1].text
                    logger.error(f"Claude Code returns an error: {error_msg}")
                    raise RuntimeError(f"Claude Code returns an error: {error_msg} Messages are saved to `messages_error.pkl`.")

                # Extract usage information from ResultMessage
                if hasattr(last_message, 'usage') and last_message.usage:
                    usage = last_message.usage
                    usage_result['input_tokens'] = usage.get('input_tokens', 0)
                    usage_result['output_tokens'] = usage.get('output_tokens', 0)
                    usage_result['cache_creation_input_tokens'] = usage.get('cache_creation_input_tokens', 0)
                    usage_result['cache_read_input_tokens'] = usage.get('cache_read_input_tokens', 0)
                    usage_result['total_tokens'] = (
                        usage_result['input_tokens'] +
                        usage_result['output_tokens'] +
                        usage_result['cache_creation_input_tokens'] +
                        usage_result['cache_read_input_tokens']
                    )
                else:
                    logger.warning("ResultMessage does not contain usage information")
                    usage_result['input_tokens'] = 0
                    usage_result['output_tokens'] = 0
                    usage_result['cache_creation_input_tokens'] = 0
                    usage_result['cache_read_input_tokens'] = 0
                    usage_result['total_tokens'] = 0

                # Extract cost from ResultMessage
                if hasattr(last_message, 'total_cost_usd'):
                    usage_result['cost'] = last_message.total_cost_usd
                else:
                    logger.warning("ResultMessage does not contain total_cost_usd")
                    usage_result['cost'] = 0.0

                usage_result['model'] = self.actual_model
                usage_result['model_name'] = self.model_name

                # Write usage info to file in working directory
                usage_file_path = os.path.join(cwd, 'claude_code_usage.json')
                try:
                    with open(usage_file_path, 'w') as f:
                        json.dump(usage_result, f, indent=2)
                    logger.debug(f"Token usage written to {usage_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to write token usage file: {e}")

        except CLINotFoundError:
            logger.error("Claude Code CLI is not installed. Please install it via npm.")
            raise
        except ProcessError as e:
            error_msg = f"Claude Code process failed with exit code {e.exit_code}"
            if e.stderr:
                stderr_lower = e.stderr.lower()
                if "not_found_error" in stderr_lower and "model:" in stderr_lower:
                    error_msg += f"\n❌ Model Error: The specified model was not found. Please check your model name."
                elif "api error" in stderr_lower or "authentication" in stderr_lower:
                    error_msg += f"\n❌ API Error: Please check your API configuration and credentials."
                elif "permission" in stderr_lower or "unauthorized" in stderr_lower:
                    error_msg += f"\n❌ Permission Error: Please check your API permissions."
                else:
                    error_msg += f"\nError details: {e.stderr}"
            logger.error(error_msg)
            raise
        except CLIJSONDecodeError:
            logger.error("Received invalid JSON from Claude Code")
            raise
        except Exception as e:
            logger.error(f"Unexpected error running Claude Code: {e}")
            raise

        logger.debug(f"Token usage: {usage_result}")
        return usage_result
    
    def _run_in_docker(
        self,
        user_prompt: str,
        cwd: str,
        add_dirs: List[str],
        max_thinking_tokens: int,
        permission_mode: str
    ) -> Dict[str, Any]:
        """Run Claude Code in Docker container using Docker SDK."""
        # Generate unique run_id for this execution
        run_id = str(uuid.uuid4())
        logger.debug(f"Starting Docker run with run_id: {run_id}")

        # Ensure prebuilt image exists locally
        if not self._image_exists(self.docker_image):
            logger.debug(f"Pulling prebuilt Docker image: {self.docker_image}")
            try:
                self.docker_client.images.pull(self.docker_image)
                logger.debug(f"Successfully pulled Docker image: {self.docker_image}")
            except Exception as e:
                raise RuntimeError(f"Failed to pull Docker image {self.docker_image}: {e}")

        workspace_abs = os.path.abspath(cwd)

        # Prepare volumes dictionary for Docker SDK
        volumes = {
            workspace_abs: {'bind': '/workspace', 'mode': 'rw'},
            '/var/run/docker.sock': {'bind': '/var/run/docker.sock', 'mode': 'rw'}
        }

        # Map additional directories into container and translate paths
        container_add_dirs = []
        for path in add_dirs or []:
            if not path:
                continue
            host_path = os.path.abspath(path)
            if not os.path.exists(host_path):
                logger.warning(f"Directory {host_path} does not exist. Skipping.")
                continue
            logger.critical(f"Support for adding additional directories is not implemented yet.")
            raise NotImplementedError(f"Support for adding additional directories is not implemented yet.")

        # Prepare environment variables
        environment = {}

        # Add credential environment variables securely
        for key in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_USE_BEDROCK", "AWS_ACCESS_KEY_ID",
                   "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME", "AWS_SESSION_TOKEN",
                   "CLAUDE_CODE_OAUTH_TOKEN", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
            val = os.getenv(key)
            if val is not None:
                environment[key] = val

        # Serialize agents for Docker environment
        agents_json = None
        if self.agents:
            # Convert AgentDefinition instances to dict for JSON serialization
            agents_data = {}
            for agent_name, agent_def in self.agents.items():
                agents_data[agent_name] = {
                    'description': agent_def.description,
                    'prompt': agent_def.prompt,
                    'tools': agent_def.tools,
                    'model': agent_def.model
                }
            agents_json = json.dumps(agents_data)

        # Add runtime configuration
        environment.update({
            'CLAUDE_USER_PROMPT': user_prompt,
            'CLAUDE_CWD': '/workspace',
            'CLAUDE_ADD_DIRS': '::'.join(container_add_dirs),
            'CLAUDE_MODEL': self.actual_model,
            'CLAUDE_MODEL_NAME': self.model_name,
            'CLAUDE_PERMISSION_MODE': permission_mode,
            'CLAUDE_MAX_TOKENS': str(max_thinking_tokens),
            'CLAUDE_ALLOWED_TOOLS': json.dumps(self.tools),
            'CLAUDE_SYSTEM_PROMPT': json.dumps(self.system_prompt),
            'HOST_WORKSPACE_PATH': workspace_abs,
            'IS_SANDBOX': '1'  # Allow bypassPermissions with root privileges in Docker
        })

        # Add agents if configured
        if agents_json:
            environment['CLAUDE_AGENTS'] = agents_json

        # Add host user info for file ownership
        try:
            environment['HOST_UID'] = str(os.getuid())
            environment['HOST_GID'] = str(os.getgid())
        except Exception:
            pass

        # Check for GPU availability and respect CUDA_VISIBLE_DEVICES
        device_requests = None
        cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')

        # Step 1: Check if nvidia-smi is available on host machine
        host_has_gpu = False
        try:
            gpu_check = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
            if gpu_check.returncode == 0:
                host_has_gpu = True
                logger.debug("nvidia-smi check passed on host machine")
            else:
                logger.debug(f"nvidia-smi failed on host with return code {gpu_check.returncode}")
        except Exception as e:
            logger.debug(f"nvidia-smi check failed on host: {e}")

        # Step 2: Test GPU access with Docker SDK
        docker_has_gpu = False
        if host_has_gpu:
            try:
                import docker.types
                # Try to run a simple GPU test container
                self.docker_client.containers.run(
                    self.docker_image,
                    "echo 'GPU test successful'",
                    device_requests=[
                        docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
                    ],
                    remove=True,
                    detach=False,
                    stdout=True,
                    stderr=True
                )
                docker_has_gpu = True
                logger.debug("Docker GPU support test passed")
            except Exception as e:
                logger.debug(f"Docker GPU support test failed: {e}")
                logger.warning("GPU detected on host but Docker GPU support (nvidia-container-toolkit) is not available. Running without GPU.")

        # Step 3: Configure GPU access if both host and Docker support GPU
        if docker_has_gpu:
            import docker.types
            if cuda_visible_devices is not None:
                # Pass CUDA_VISIBLE_DEVICES to container to limit GPU access
                environment['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

                # Parse device IDs to determine count
                if cuda_visible_devices.strip() == '':
                    # Empty string means no GPUs
                    logger.debug("CUDA_VISIBLE_DEVICES is empty - no GPU access for container")
                else:
                    # Count specified GPUs (handles formats like "0", "0,1", "0,1,2")
                    device_ids = [d.strip() for d in cuda_visible_devices.split(',') if d.strip()]
                    device_count = len(device_ids)
                    device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])]
                    logger.debug(f"GPU support enabled for container with CUDA_VISIBLE_DEVICES={cuda_visible_devices} ({device_count} GPU(s))")
            else:
                # No restriction - expose all GPUs
                device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])]
                logger.debug("GPU support enabled for container (all GPUs)")
        else:
            logger.debug("GPU support not available for Docker container")

        # Read Python script from template file
        template_dir = os.path.dirname(__file__)
        template_path = os.path.join(template_dir, 'template', 'claude_agent_runner.py.txt')

        try:
            with open(template_path, 'r') as f:
                python_script = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Template file not found: {template_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to read template file {template_path}: {e}")

        # Use Docker SDK to run the container
        # Include run_id in container name for unique identification
        container_name = f"claude-agent-{os.getpid()}-{run_id[:8]}"

        logger.info(f"Running Claude Agent in Docker with model: {self.actual_model} (run_id: {run_id})")

        try:
            # Create container configuration - use detach=True for streaming
            container_config = {
                'image': self.docker_image,
                'name': container_name,
                'command': ['python', '-c', python_script],
                'environment': environment,
                'volumes': volumes,
                'working_dir': '/workspace',
                'network_mode': 'host',
                'remove': True,   # Auto-remove container when it stops, but we'll also handle manual cleanup
                'detach': True,   # Run in background so we can stream logs
                'stdout': True,
                'stderr': True
            }

            if device_requests:
                container_config['device_requests'] = device_requests

            # Create and start the container
            try:
                container = self.docker_client.containers.run(**container_config)
                # Store container reference with run_id for tracking
                self._containers[run_id] = container
                logger.debug(f"Started Claude Code container: {container.short_id} (run_id: {run_id})")
            except Exception as container_error:
                # If container creation fails, try to clean up by name
                try:
                    failed_container = self.docker_client.containers.get(container_name)
                    failed_container.remove(force=True)
                    logger.debug(f"Cleaned up failed container: {container_name}")
                except Exception:
                    pass
                raise container_error

            # Stream logs in real-time
            try:
                log_buffer = ""
                message_buffer = []  # Buffer to accumulate multi-line messages
                current_log_level = None

                for log_chunk in container.logs(stream=True, follow=True):
                    if log_chunk:
                        # Decode the chunk and add to buffer
                        chunk_str = log_chunk.decode('utf-8', errors='replace')
                        log_buffer += chunk_str

                        # Process complete lines
                        while '\n' in log_buffer:
                            line, log_buffer = log_buffer.split('\n', 1)
                            line_stripped = line.rstrip()

                            if not line_stripped:  # Empty line
                                continue

                            # Check if this is a new message with a log level prefix
                            new_message = False
                            message_content = None
                            log_level = None

                            if line_stripped.startswith('[Debug]'):
                                message_content = line_stripped[7:].lstrip()
                                log_level = 'debug'
                                new_message = True
                            elif line_stripped.startswith('[Warning]'):
                                message_content = line_stripped[9:].lstrip()
                                log_level = 'warning'
                                new_message = True
                            elif line_stripped.startswith('[Error]'):
                                message_content = line_stripped[7:].lstrip()
                                log_level = 'error'
                                new_message = True
                            elif line_stripped.startswith('[Claude]'):
                                # message_content = line_stripped[8:].lstrip()
                                message_content = line_stripped
                                log_level = 'debug'
                                new_message = True
                            elif line_stripped.startswith('Traceback') or line_stripped.startswith('  File '):
                                # Python traceback - always error
                                message_content = line_stripped
                                log_level = 'error'
                                new_message = not message_buffer  # New if buffer is empty
                            else:
                                # Continuation of previous message or standalone info
                                message_content = line_stripped
                                log_level = current_log_level or 'info'
                                new_message = not message_buffer  # New if buffer is empty

                            # If starting a new message, flush the previous one
                            if new_message and message_buffer:
                                full_message = '\n'.join(message_buffer)
                                if current_log_level == 'debug':
                                    logger.debug(full_message)
                                elif current_log_level == 'warning':
                                    logger.warning(full_message)
                                elif current_log_level == 'error':
                                    logger.error(full_message)
                                else:
                                    logger.info(full_message)
                                message_buffer = []

                            # Add current line to buffer
                            if new_message:
                                current_log_level = log_level
                            message_buffer.append(message_content)

                # Flush any remaining message
                if message_buffer:
                    full_message = '\n'.join(message_buffer)
                    if current_log_level == 'debug':
                        logger.debug(full_message)
                    elif current_log_level == 'warning':
                        logger.warning(full_message)
                    elif current_log_level == 'error':
                        logger.error(full_message)
                    else:
                        logger.info(full_message)

            except Exception as stream_error:
                logger.warning(f"Error streaming container logs: {stream_error}")

            # Wait for container to complete
            result = container.wait()
            exit_code = result['StatusCode']

            if exit_code != 0:
                # Get final logs for error details (handle auto-removal race condition)
                try:
                    final_logs = container.logs().decode('utf-8', errors='replace')
                    logger.error(f"Final container logs:\n{final_logs}")
                except DockerAPIError as api_error:
                    if "dead or marked for removal" in str(api_error):
                        logger.debug("Container was auto-removed before logs could be retrieved")
                    else:
                        logger.warning(f"Could not retrieve container logs: {api_error}")
                except Exception as log_error:
                    logger.debug(f"Could not retrieve final container logs: {log_error}")

                logger.error(f"Container exited with code {exit_code}")
                raise RuntimeError(f"Claude Code container failed with exit code {exit_code}")
            else:
                logger.debug("Claude Code container completed successfully (auto-removed)")

            # Read token usage from the working directory
            usage_file_path = os.path.join(workspace_abs, 'claude_code_usage.json')
            usage_result = {
                'input_tokens': 0,
                'output_tokens': 0,
                'cache_creation_input_tokens': 0,
                'cache_read_input_tokens': 0,
                'total_tokens': 0,
                'cost': 0.0,
                'model': self.actual_model,
                'model_name': self.model_name
            }

            try:
                with open(usage_file_path, 'r') as f:
                    usage_data = json.load(f)
                    usage_result.update(usage_data)
                logger.info(f"Token usage: {usage_result}")
            except FileNotFoundError:
                logger.warning(f"Token usage file not found: {usage_file_path}")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse token usage file: {e}")
            except Exception as e:
                logger.warning(f"Failed to read token usage: {e}")

            return usage_result

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, cleaning up...")
            raise
        except Exception as e:
            logger.error(f"Claude Code runner container failed: {str(e)}")

            # Try to get container logs for debugging if container was created
            container = self._containers.get(run_id)
            if container:
                try:
                    logs = container.logs().decode('utf-8')
                    logger.error(logs)
                except Exception as log_error:
                    logger.debug(f"Could not retrieve container logs: {log_error}")
            else:
                # Try to get container by name if reference not stored yet
                try:
                    failed_container = self.docker_client.containers.get(container_name)
                    logs = failed_container.logs().decode('utf-8')
                    logger.error(logs)
                except Exception as log_error:
                    logger.debug(f"Could not retrieve container logs: {log_error}")

            raise
        finally:
            # Always cleanup container for this specific run
            self._cleanup_container(run_id)

    def _image_exists(self, image: str) -> bool:
        """Check if Docker image exists locally using Docker SDK."""
        try:
            self.docker_client.images.get(image)
            return True
        except docker.errors.ImageNotFound:
            return False
        except Exception:
            return False


def main():
    """
    Main function that can be called from both CLI and Jupyter environments.
    """
    
    # Simple example usage
    user_prompt = "Create a simple Python function that calculates the factorial of a number and save it to a file called 'factorial.py'"
    cwd = os.getcwd()  # Use current working directory
    add_dirs = []  # No additional directories to add
    
    try:
        agent = ClaudeAgent(
            model_name="claude_code/claude-sonnet-4-5-20250929",
            run_in_docker=True
        )
        asyncio.run(
            agent.run(
                user_prompt=user_prompt,
                cwd=os.path.join(cwd, 'tmp', 'claude_agent_test_run'),
                add_dirs=add_dirs,
            )
        )
        logger.info("✅ Claude Agent execution completed successfully!")
    except Exception as e:
        logger.error(f"❌ Claude Agent execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
