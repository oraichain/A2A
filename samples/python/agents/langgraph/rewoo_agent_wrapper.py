import asyncio
import json
import logging
import os
import traceback
from common.types import FilePart, TextPart
from langgraph.checkpoint.memory import MemorySaver
from typing import Any, Dict, AsyncIterable, List, Tuple
from rewoo import InitState, PlannerModel, AnalysisModel, PlannerState, SolverState, WorkerState, create_rewoo_agent, is_init_state, is_planner_state, is_worker_state, is_solver_state
from langchain_mcp_adapters.client import SSEConnection

memory = MemorySaver()
logger = logging.getLogger(__name__)

class ReWOOAgentWrapper:

    def __init__(self, sse_mcp_server_sessions: dict[str, SSEConnection]):
        self.model_name = os.getenv("LLM_MODEL")
        self.api_key = os.getenv("API_KEY")
        self.planner_model = PlannerModel(model=self.model_name, api_key=self.api_key)
        self.analysis_model = AnalysisModel(model=self.model_name, api_key=self.api_key)
        self.sse_mcp_server_sessions = sse_mcp_server_sessions
        
    async def initialize(self):
        self.graph = await create_rewoo_agent(
            sse_mcp_server_sessions=self.sse_mcp_server_sessions,
            planner_model=self.planner_model,
            analysis_model=self.analysis_model,
        )

    async def stream(self, query: str, session_id: str, task_id: str) -> AsyncIterable[Dict[str, Any]]:
        inputs = InitState(task=query)
        config = {"configurable": {"thread_id": f"{session_id}_{task_id}"}}

        try:
            async for event in self.graph.astream(inputs, config, stream_mode="values"):
                yield await self.process_event(event, session_id, task_id)
        except asyncio.TimeoutError:
                logger.warning(f"Stream for session {session_id} with task {task_id} timed out")
                yield {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": "Task timed out",
                }
        except Exception as e:
            logger.error(f"Error in stream for session {session_id} with task {task_id}: {e}")
            logger.error(traceback.format_exc())
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"Error: {str(e)}",
            }
        finally:
            logger.info(f"Stream for session {session_id} and task {task_id} completed")
        
    async def process_event(self, event, session_id: str, task_id: str) -> Dict[str, Any]:
        try:
            state = is_init_state(event)
            if state:
                return {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "model_usage": None,
                    "content": [],
                    "images": [],
                }
            
            state = is_planner_state(event) or is_worker_state(event)
            if state:
                content, images = self.extract_message_content(state)
                model_usage = state.model_usage
                logger.info(f"Model usage: {model_usage}")
                logger.info(f"Content of planner/worker: {content}")
                return {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "model_usage": model_usage,
                    "content": content,
                    "images": images,
                }

            state = is_solver_state(event)
            if state:
                content, images = self.extract_message_content(state)
                model_usage = state.model_usage
                logger.info(f"Model usage: {model_usage}")
                logger.info(f"Content of solver: {content}")
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "model_usage": model_usage,
                    "content": content,
                    "images": images,
                }
            else:
                logger.error(f"Unknown event type: {event}")
                return {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "model_usage": None,
                    "content": [],
                    "images": [],
                }

        except Exception as e:
            logger.error(f"Error in process_event with session {session_id} and task {task_id}: {e}")
            logger.error(traceback.format_exc())
        return {
            "is_task_complete": False,
            "require_user_input": False,
            "model_usage": None,
            "content": "Unknown event",
            "images": [],
        }

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    @staticmethod
    def extract_message_content(
        message: PlannerState | WorkerState | SolverState,
    ) -> Tuple[List[TextPart], List[FilePart]]:
        text_parts: List[TextPart] = []
        image_parts: List[FilePart] = []
        if is_planner_state(message):
            text_parts = [TextPart(type="text", text=json.dumps(item, default=lambda _o: "<not serializable>")) for item in message.plan]
            image_parts = []
        elif is_worker_state(message):
            text_parts = [TextPart(type="text", text=item) for item in message.mcp_results]
            image_parts = []
        elif is_solver_state(message):
            text_parts = [TextPart(type="text", text=message.analysis)]
            image_parts = []
        return text_parts, image_parts