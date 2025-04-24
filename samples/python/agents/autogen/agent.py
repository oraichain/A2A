import os
import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from agents.autogen.memory_manager import MemoryManager
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_ext.tools.mcp import mcp_server_tools, McpServerParams
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    MultiModalMessage,
)
from autogen_agentchat.base import TaskResult
from autogen_core import Image
from typing import AsyncIterable, Dict, Any, TypedDict, AsyncGenerator, Tuple, List
from datetime import timezone

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
    ):
        self.LABEL = label
        self.SYSTEM_INSTRUCTION = system_instruction
        self.SUPPORTED_CONTENT_TYPES = supported_content_types
        self.model_client = None
        self.mcp_agent = None
        self.sessions: Dict[str, SessionData] = defaultdict(dict)
        self.session_lock = asyncio.Lock()
        self.timeout = timeout
        self.session_timeout = timedelta(seconds=self.timeout)
        self.max_concurrent_tasks = max_concurrent_tasks
        self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        self.memory_manager = MemoryManager(label, in_mem_vector_store)

    async def initialize(self, mcp_server_params: list[McpServerParams] = []):
        api_key = os.getenv("API_KEY")
        model = os.getenv("LLM_MODEL")
        if not api_key or not model:
            raise ValueError("API_KEY or LLM_MODEL not set")
        self.model_client = AnthropicChatCompletionClient(model=model, api_key=api_key)
        self.tools = []
        for mcp_server_param in mcp_server_params:
            server_tools = await mcp_server_tools(mcp_server_param)
            logger.info("Server tools:", server_tools)
            self.tools.extend(server_tools)
        asyncio.create_task(self.cleanup_sessions())
        logger.info(f"{self.LABEL} initialized and cleanup_sessions task scheduled")

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
                    del self.sessions[sid]
                    self.memory_manager.delete_session(sid)
            await asyncio.sleep(60)

    async def process_event(self, event, session_id: str) -> Dict[str, Any]:
        try:
            if isinstance(event, BaseAgentEvent):
                content, images, type = self.extract_message_content(event)
                return {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": content,
                    "images": images,
                    "type": type,
                }
            elif isinstance(event, TaskResult):
                # async with self.session_lock:
                #     await self.sessions[session_id]["generator"].aclose()
                #     del self.sessions[session_id]
                #     logger.info(
                #         f"Adding memory {event.messages[-1].content} for session {session_id}"
                #     )
                #     if "content" in event.messages[-1]:
                #         self.memory_manager.add_memory(
                #             session_id,
                #             [
                #                 {
                #                     "role": "assistant",
                #                     "content": event.messages[-1].content,
                #                 },
                #             ],
                #         )

                # get last message in event.messages

                content = self.extract_task_result_content(event)
                images = []
                type = None
                if len(event.messages) > 0:
                    last_message = event.messages[-1]
                    content, images, type = self.extract_message_content(last_message)
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": content,
                    "images": images,
                    "type": type,
                }
            elif isinstance(event, BaseChatMessage):
                content, images, type = self.extract_message_content(event)
                return {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": content,
                    "images": images,
                    "type": type,
                }
        except Exception as e:
            logger.info(f"Error in process_event: {e}")
        return {
            "is_task_complete": False,
            "require_user_input": False,
            "model_usage": None,
            "content": "Unknown event",
            "images": [],
        }

    @staticmethod
    def extract_message_content(
        message: BaseAgentEvent | BaseChatMessage,
    ) -> Tuple[str, List[str], str | None]:
        message_type = message.model_dump().get("type", None)
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
        return "\n".join(text_parts), image_parts, message_type

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
                    orchestrator_agent = MagenticOneGroupChat(
                        participants=[mcp_agent], model_client=self.model_client
                    )
                    # relevant_memories = self.memory_manager.relevant_memories(
                    #     session_id=session_id, query=query
                    # )
                    # logger.info(
                    #     f"Relevant memories: {relevant_memories} for session {session_id} at query {query}"
                    # )
                    task = query
                    # if len(relevant_memories) > 0:
                    #     flatten_relevant_memories = "\n".join(
                    #         [m["memory"] for m in relevant_memories]
                    #     )
                    #     task = f"Answer the user question considering the memories. Keep answers clear and concise. Memories:{flatten_relevant_memories}\n\nQuestion: {query}"
                    self.sessions[session_id] = {
                        "generator": orchestrator_agent.run_stream(task=task),
                        "last_accessed": datetime.now(timezone.utc),
                    }
                    # self.memory_manager.add_memory(
                    #     session_id,
                    #     [
                    #         {"role": "user", "content": query},
                    #     ],
                    # )
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
                }
            finally:
                logger.info(f"Stream for session {session_id} completed")
