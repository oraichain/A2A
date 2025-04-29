import os
import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta
import re
from agents.autogen.memory_manager import MemoryManager
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_ext.tools.mcp import mcp_server_tools, McpServerParams
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    MultiModalMessage,
)
from autogen_agentchat.base import TaskResult
from autogen_core import Image
from autogen_core.models import RequestUsage
from typing import AsyncIterable, Dict, Any, TypedDict, AsyncGenerator, Tuple, List
from datetime import timezone
from litellm import completion_cost
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SessionData(TypedDict):
    generator: AsyncGenerator[BaseAgentEvent | BaseChatMessage | TaskResult, None]
    last_accessed: datetime


class Agent:
    def __init__(
        self,
        label: str,
        system_instruction: str,
        supported_content_types: list[str] = ["text", "text/plain"],
        timeout: int = 900,
        max_concurrent_tasks: int = 10,
        in_mem_vector_store: bool = False,
        use_memory: bool = False,
    ):
        self.LABEL = label
        self.SYSTEM_INSTRUCTION = system_instruction
        self.SUPPORTED_CONTENT_TYPES = supported_content_types
        self.model_client = None
        self.mcp_agent = None
        self.code_executor_agent = None
        self.sessions: Dict[str, SessionData] = defaultdict(dict)
        # Track cumulative token metrics per session
        self.session_token_metrics: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "accumulated_cost": 0}
        )
        self.session_lock = asyncio.Lock()
        self.timeout = timeout
        self.session_timeout = timedelta(seconds=self.timeout)
        self.max_concurrent_tasks = max_concurrent_tasks
        self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        if use_memory:
            self.memory_manager = MemoryManager(label, in_mem_vector_store)
        else:
            self.memory_manager = None
        
        self.sse_mcp_server_urls = []
        self.mcp_server_params = []
        self.sse_mcp_server_sessions = []
        self.code_executor = DockerCommandLineCodeExecutor(work_dir="coding", container_name="code-executor")

    async def initialize(self, mcp_server_params: list[McpServerParams] = []):
        api_key = os.getenv("API_KEY")
        model = os.getenv("LLM_MODEL")
        if not api_key or not model:
            raise ValueError("API_KEY or LLM_MODEL not set")
        if re.match(r"^claude", model):
            self.model_client = AnthropicChatCompletionClient(model=model, api_key=api_key)
        else:
            self.model_client = OpenAIChatCompletionClient(model=model, api_key=api_key)
        # FIXME: This is a hack to get the model name. With autogen, there could be multiple models in the config.
        self.model_name = model
        self.tools = []
        for mcp_server_param in mcp_server_params:
            server_tools = await mcp_server_tools(mcp_server_param)
            self.tools.extend(server_tools)
        asyncio.create_task(self.cleanup_sessions())
        logger.info(f"{self.LABEL} initialized and cleanup_sessions task scheduled")
    async def initialize_with_mcp_sse_urls(self, sse_mcp_server_urls: list[str]):
        api_key = os.getenv("API_KEY")
        model = os.getenv("LLM_MODEL")
        if not api_key or not model:
            raise ValueError("API_KEY or LLM_MODEL not set")
        if re.match(r"^claude", model):
            self.model_client = AnthropicChatCompletionClient(model=model, api_key=api_key)
        else:
            self.model_client = OpenAIChatCompletionClient(model=model, api_key=api_key)
        # FIXME: This is a hack to get the model name. With autogen, there could be multiple models in the config.
        self.model_name = model
        self.tools = []
        
        # Close any existing SSE sessions before creating new ones
        await self.close_sse_sessions()
        
        self.sse_mcp_server_urls = sse_mcp_server_urls
        self.sse_mcp_server_sessions = []
        self.mcp_server_params = []
        
        for url in sse_mcp_server_urls:
            try:
                streams = await sse_client(url)
                session = ClientSession(streams[0], streams[1])
                await session.initialize()
                self.sse_mcp_server_sessions.append(session)
                self.mcp_server_params.append(McpServerParams(url=url))
                
                # Get tools from this server and add to our tools list
                server_tools = await mcp_server_tools(McpServerParams(url=url), session=session)
                self.tools.extend(server_tools)
                logger.info(f"Connected to MCP server at {url}")
            except Exception as e:
                logger.error(f"Error connecting to MCP server at {url}: {e}")
        
        asyncio.create_task(self.cleanup_sessions())
        logger.info(f"{self.LABEL} initialized and cleanup_sessions task scheduled")
        
    async def shutdown(self):
        """Shutdown the agent and cleanup resources."""
        logger.info(f"Shutting down {self.LABEL} agent")
        
        # Close all sessions
        async with self.session_lock:
            for sid, data in list(self.sessions.items()):
                try:
                    await data["generator"].aclose()
                    logger.info(f"Closed session {sid}")
                except Exception as e:
                    logger.error(f"Error closing session {sid}: {e}")
                    
        # Close SSE sessions
        await self.close_sse_sessions()
        
        # Clean up any other resources
        if self.memory_manager:
            # Free memory if needed
            logger.info("Cleaning up memory manager")
            for sid in list(self.session_token_metrics.keys()):
                self.memory_manager.delete_session(sid)
                
        logger.info(f"{self.LABEL} agent shutdown complete")

    def accumulate_token_metrics(self, session_id: str, model_usage: RequestUsage):
        self.session_token_metrics[session_id]["prompt_tokens"] += model_usage.prompt_tokens
        self.session_token_metrics[session_id]["completion_tokens"] += model_usage.completion_tokens
        self.session_token_metrics[session_id]["total_tokens"] += model_usage.prompt_tokens + model_usage.completion_tokens
        calc_cost_request = {
            "model": self.model_name,
            "usage": {
                "prompt_tokens": model_usage.prompt_tokens,
                "completion_tokens": model_usage.completion_tokens,
                "total_tokens": model_usage.prompt_tokens + model_usage.completion_tokens
            }
        }
        self.session_token_metrics[session_id]["accumulated_cost"] += completion_cost(calc_cost_request)
        logger.info(
            f"Token Metrics for session {session_id}: Prompt Tokens={model_usage.prompt_tokens}, "
            f"Completion Tokens={model_usage.completion_tokens}, "
            f"Cumulative Total={self.session_token_metrics[session_id]['total_tokens']}"
            f"Accumulated Cost=${self.session_token_metrics[session_id]['accumulated_cost']:.6f}"
        )

    async def process_event(self, event, session_id: str) -> Dict[str, Any]:
        try:
            if isinstance(event, BaseAgentEvent):
                try:
                    content, images = self.extract_message_content(event)
                except Exception as e:
                    logger.error(f"Error extracting message content: {e}")
                    content = str(event)
                    images = []
                    
                model_usage = event.models_usage
                if model_usage:
                    self.accumulate_token_metrics(session_id, model_usage)
                return {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "model_usage": model_usage,
                    "content": content,
                    "images": images,
                }
            elif isinstance(event, TaskResult):
                async with self.session_lock:
                    logger.info(f"TaskResult event model usages in all messages: {[msg.models_usage for msg in event.messages]}")
                    await self.sessions[session_id]["generator"].aclose()
                    del self.sessions[session_id]
                    # Log final aggregated token metrics
                    logger.info(
                        f"Final Model Usage for session {session_id}: "
                        f"Prompt Tokens={self.session_token_metrics[session_id]['prompt_tokens']}, "
                        f"Completion Tokens={self.session_token_metrics[session_id]['completion_tokens']}, "
                        f"Total Tokens={self.session_token_metrics[session_id]['total_tokens']}, "
                        f"Estimated Cost=${self.session_token_metrics[session_id]['accumulated_cost']:.6f}"
                    )
                    del self.session_token_metrics[session_id]
                    if self.memory_manager:
                        logger.info(
                            f"Adding memory {event.messages} for session {session_id}"
                        )
                        for message in event.messages:
                            if "content" in message.to_text():
                                self.memory_manager.add_memory(
                                    session_id,
                                    [
                                        {
                                            "role": "assistant",
                                            "content": message.to_text(),
                                        },
                                ],
                            )

                content = self.extract_task_result_content(event)
                images = []
                if len(event.messages) > 0:
                    try:
                        last_message = event.messages[-1]
                        content_text, content_images = self.extract_message_content(
                            last_message
                        )
                        content = content_text
                        images = content_images
                    except Exception as e:
                        logger.error(f"Error extracting message content from task result: {e}")

                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "model_usage": None,
                    "content": content,
                    "images": images,
                }
            elif isinstance(event, BaseChatMessage):
                try:
                    content, images = self.extract_message_content(event)
                except Exception as e:
                    logger.error(f"Error extracting message content: {e}")
                    content = event.to_text() if hasattr(event, 'to_text') else str(event)
                    images = []
                    
                model_usage = event.models_usage
                if model_usage:
                    self.accumulate_token_metrics(session_id, model_usage)
                
                return {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "model_usage": model_usage,
                    "content": content,
                    "images": images,
                }
        except Exception as e:
            logger.error(f"Error in process_event: {e}")
        return {
            "is_task_complete": False,
            "require_user_input": False,
            "model_usage": None,
            "content": "Unknown event",
            "images": [],
        }

    async def cleanup_sessions(self):
        while True:
            async with self.session_lock:
                now = datetime.now(timezone.utc)
                expired = [
                    sid
                    for sid, data in self.sessions.items()
                    if now - data["last_accessed"] > self.session_timeout
                ]
                for sid in expired:
                    gen = self.sessions[sid]["generator"]
                    await gen.aclose()
                    # Log final aggregated token metrics for expired session
                    logger.info(
                        f"Final Model Usage for expired session {sid}: "
                        f"Prompt Tokens={self.session_token_metrics[sid]['prompt_tokens']}, "
                        f"Completion Tokens={self.session_token_metrics[sid]['completion_tokens']}, "
                        f"Total Tokens={self.session_token_metrics[sid]['total_tokens']}, "
                        f"Estimated Cost=${self.session_token_metrics[sid]['accumulated_cost']:.6f}"
                    )
                    del self.sessions[sid]
                    del self.session_token_metrics[sid]
                    if self.memory_manager:
                        self.memory_manager.delete_session(sid)
            await asyncio.sleep(60)
            
    async def close_sse_sessions(self):
        """Close all SSE sessions."""
        for session in self.sse_mcp_server_sessions:
            if session:
                try:
                    await session.close()
                except Exception as e:
                    logger.error(f"Error closing SSE session: {e}")
        self.sse_mcp_server_sessions = []

    @staticmethod
    def extract_message_content(
        message: BaseAgentEvent | BaseChatMessage,
    ) -> Tuple[str, List[str]]:
        if isinstance(message, MultiModalMessage):
            text_parts = [item for item in message.content if isinstance(item, str)]
            image_parts = [
                item.to_base64() for item in message.content if isinstance(item, Image)
            ]
        elif isinstance(message, BaseChatMessage):
            text_parts = [message.to_model_text()]
            image_parts = []
        else:
            text_parts = [message.to_text()]
            image_parts = []
        return "\n".join(text_parts), image_parts

    @staticmethod
    def extract_task_result_content(message: TaskResult) -> str:
        output = (
            f"Number of messages: {len(message.messages)}\n"
            f"Finish reason: {message.stop_reason}\n"
        )
        return output

    async def stream(
        self, query: str, session_id: str
    ) -> AsyncIterable[Dict[str, Any]]:
        logger.info(f"Starting stream for session {session_id} with query: {query}")
        async with self.semaphore:
            try:
                # Check if we have an existing agent for this session
                async with self.session_lock:
                    mcp_agent = AssistantAgent(
                        name=self.LABEL,
                        model_client=self.model_client,
                        tools=self.tools,
                        system_message=self.SYSTEM_INSTRUCTION,
                    )
                    await self.code_executor.start()
                    code_executor_agent = CodeExecutorAgent(
                        name="HyperwhalesAnalyst",
                        model_client=self.model_client,
                        code_executor=self.code_executor,
                        system_message=self.SYSTEM_INSTRUCTION,
                        description="Expert analyst who can execute code to navigate database, analyze data and provide insights",
                    )
                    orchestrator_agent = MagenticOneGroupChat(
                        participants=[mcp_agent], model_client=self.model_client
                    )
                    
                    task = query
                    if self.memory_manager:
                        relevant_memories = self.memory_manager.relevant_memories(
                            session_id=session_id, query=query
                        )
                        logger.info(
                            f"Relevant memories: {relevant_memories} for session {session_id} at query {query}"
                        )
                        
                        if len(relevant_memories) > 0:
                            flatten_relevant_memories = "\n".join(
                                [m["memory"] for m in relevant_memories]
                            )
                            task = f"Answer the user question considering the memories. Keep answers clear and concise. Memories:{flatten_relevant_memories}\n\nQuestion: {query}"
                            
                        self.memory_manager.add_memory(
                            session_id,
                            [
                                {"role": "user", "content": query},
                            ],
                        )
                        
                    self.sessions[session_id] = {
                        "generator": orchestrator_agent.run_stream(task=task),
                        "last_accessed": datetime.now(timezone.utc),
                    }
                    
                async with asyncio.timeout(self.timeout):
                    async for event in self.sessions[session_id]["generator"]:
                        async with self.session_lock:
                            self.sessions[session_id]["last_accessed"] = datetime.now(
                                timezone.utc
                            )
                        yield await self.process_event(event, session_id)
            except asyncio.TimeoutError:
                logger.warning(f"Stream for session {session_id} timed out")
                async with self.session_lock:
                    if session_id in self.sessions:
                        await self.sessions[session_id]["generator"].aclose()
                        del self.sessions[session_id]
                yield {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": "Task timed out",
                    "images": [],
                }
            except Exception as e:
                logger.error(f"Error in stream for session {session_id}: {e}")
                async with self.session_lock:
                    if session_id in self.sessions:
                        await self.sessions[session_id]["generator"].aclose()
                        del self.sessions[session_id]
                yield {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": f"Error: {str(e)}",
                    "images": [],
                }
            finally:
                logger.info(f"Stream for session {session_id} completed")
