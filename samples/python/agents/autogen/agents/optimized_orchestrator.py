import json
import os
import asyncio
import logging

from autogen_core import Component
from agents.summary_agent import SummaryAnalysisMessage
import aiofiles
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.teams._group_chat._magentic_one._magentic_one_orchestrator import MagenticOneOrchestrator
from autogen_agentchat.messages import (
    MultiModalMessage,
)
from typing import Any, Callable, List, Mapping, override
from autogen_core.models import (
    AssistantMessage,
    LLMMessage,
    UserMessage,
)
from autogen_agentchat.messages import (
    ToolCallRequestEvent,
    ToolCallExecutionEvent,
    StopMessage,
    HandoffMessage,
    TextMessage,
    ToolCallSummaryMessage,
    BaseAgentEvent,
    BaseChatMessage,
    MessageFactory,
)
from autogen_agentchat.base import TerminationCondition, ChatAgent, TerminationCondition
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, MessageFactory
from autogen_agentchat.teams._group_chat._base_group_chat import GroupChatTermination
from autogen_agentchat.teams._group_chat._base_group_chat import BaseGroupChat
from autogen_agentchat.teams._group_chat._magentic_one._magentic_one_group_chat import MagenticOneGroupChatConfig
from autogen_agentchat.teams._group_chat._magentic_one._prompts import ORCHESTRATOR_FINAL_ANSWER_PROMPT

import asyncio
import logging
from typing import Callable, List

from autogen_core import AgentRuntime, Component, ComponentModel
from autogen_core.models import ChatCompletionClient
from pydantic import BaseModel
from typing_extensions import Self

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedOrchestrator(MagenticOneOrchestrator):
  
  @override
  def _thread_to_context(self) -> List[LLMMessage]:
        """Convert the message thread to a context for the model."""
        context: List[LLMMessage] = []
        for m in self._message_thread:
            if isinstance(m, ToolCallRequestEvent | ToolCallExecutionEvent):
                # Ignore tool call messages.
                continue
            # elif isinstance(m, ToolCallExecutionEvent):
            #     context.append(AssistantMessage(content=[content.model_dump_json() for content in m.content], source=m.source))
            elif isinstance(m, StopMessage | HandoffMessage):
                context.append(UserMessage(content=m.content, source=m.source))
            elif m.source == self._name:
                assert isinstance(m, TextMessage | ToolCallSummaryMessage)
                # if not isinstance(m, ToolCallSummaryMessage):
                context.append(AssistantMessage(content=m.content, source=m.source))
            else:
                assert isinstance(m, (TextMessage, MultiModalMessage, ToolCallSummaryMessage))
                # if not isinstance(m, ToolCallSummaryMessage):
                context.append(UserMessage(content=m.content, source=m.source))
        return context

class MagenticOneGroupChatWithPersistState(BaseGroupChat, Component[MagenticOneGroupChatConfig]):
    
    component_config_schema = MagenticOneGroupChatConfig
    component_provider_override = "agents.optimized_orchestrator.OptimizedOrchestrator"
    
    def __init__(
        self,
        participants: List[ChatAgent],
        model_client: ChatCompletionClient,
        *,
        termination_condition: TerminationCondition | None = None,
        max_turns: int | None = 20,
        runtime: AgentRuntime | None = None,
        max_stalls: int = 3,
        final_answer_prompt: str = ORCHESTRATOR_FINAL_ANSWER_PROMPT,
        emit_team_events: bool = False,
        session_id: str | None = None,
        task_id: str | None = None,
    ):  
        super().__init__(
            participants,
            group_chat_manager_name="OptimizedOrchestrator",
            group_chat_manager_class=OptimizedOrchestrator,
            termination_condition=termination_condition,
            max_turns=max_turns,
            runtime=runtime,
            emit_team_events=emit_team_events,
        )

        # Validate the participants.
        if len(participants) == 0:
            raise ValueError("At least one participant is required for MagenticOneGroupChat.")
        self._model_client = model_client
        self._max_stalls = max_stalls
        self._final_answer_prompt = final_answer_prompt
        self.session_id = session_id
        self.task_id = task_id
        self.emit_team_events = emit_team_events
        root_path = os.getenv("ROOT_PATH", os.getcwd())
        self.base_path = f"{root_path}/agents/autogen/hyperwhales/sessions/{self.session_id}/tasks/{self.task_id}"
        if not self.session_id or not self.task_id:
            raise ValueError("session_id and task_id are required")
    
    @override
    async def save_state(self):
        state = await super().save_state()
        await asyncio.to_thread(
            os.makedirs,
            os.path.dirname(f"{self.base_path}/agent_state.json"),
            exist_ok=True
            )
        async with aiofiles.open(f"{self.base_path}/agent_state.json", "w") as f:
            await f.write(json.dumps(state))
            await f.flush()
        logger.info(f"Saved orchestrator state for session {self.session_id} and task {self.task_id}")
        return state
    
    @override
    async def load_state(self, _: Mapping[str, Any]):
        if not os.path.exists(f"{self.base_path}/agent_state.json"):
            return None
        async with aiofiles.open(f"{self.base_path}/agent_state.json", "r") as f:
            state_str = await f.read()
            state = json.loads(state_str)
            await super().load_state(state)
        logger.info(f"Loaded orchestrator state for session {self.session_id} and task {self.task_id}")
        return state

    def _create_group_chat_manager_factory(
        self,
        name: str,
        group_topic_type: str,
        output_topic_type: str,
        participant_topic_types: List[str],
        participant_names: List[str],
        participant_descriptions: List[str],
        output_message_queue: asyncio.Queue[BaseAgentEvent | BaseChatMessage | GroupChatTermination],
        termination_condition: TerminationCondition | None,
        max_turns: int | None,
        message_factory: MessageFactory,
        
    ) -> Callable[[], OptimizedOrchestrator]:
        return lambda: OptimizedOrchestrator(
            name,
            group_topic_type,
            output_topic_type,
            participant_topic_types,
            participant_names,
            participant_descriptions,
            max_turns,
            message_factory,
            self._model_client,
            self._max_stalls,
            self._final_answer_prompt,
            output_message_queue,
            termination_condition,
            self.emit_team_events,
        )

    def _to_config(self) -> MagenticOneGroupChatConfig:
        participants = [participant.dump_component() for participant in self._participants]
        termination_condition = self._termination_condition.dump_component() if self._termination_condition else None
        return MagenticOneGroupChatConfig(
            participants=participants,
            model_client=self._model_client.dump_component(),
            termination_condition=termination_condition,
            max_turns=self._max_turns,
            max_stalls=self._max_stalls,
            final_answer_prompt=self._final_answer_prompt,
        )

    @classmethod
    def _from_config(cls, config: MagenticOneGroupChatConfig) -> Self:
        participants = [ChatAgent.load_component(participant) for participant in config.participants]
        model_client = ChatCompletionClient.load_component(config.model_client)
        termination_condition = (
            TerminationCondition.load_component(config.termination_condition) if config.termination_condition else None
        )
        return cls(
            participants,
            model_client,
            termination_condition=termination_condition,
            max_turns=config.max_turns,
            max_stalls=config.max_stalls,
            final_answer_prompt=config.final_answer_prompt,
        )