# mypy: ignore-errors
import asyncio
from uuid import uuid4

from common.client import A2ACardResolver, A2AClient
from common.types import Message, TaskSendParams, TaskState, TaskStatusUpdateEvent

class A2AAgent:
    def __init__(self, a2a_server_url: str, session: str = None, history: bool = False):
        self.session = session
        self.history = history

        self.card_resolver = A2ACardResolver(a2a_server_url)
        self.card = self.card_resolver.get_agent_card()

        print('======= Agent Card ========')
        print(self.card.model_dump_json(exclude_none=True))

        self.client = A2AClient(agent_card=self.card)
        if session:
            self.sessionId = session
        else:
            self.sessionId = uuid4().hex

    async def step(self, messages: list[str]):
        continue_loop = True
        streaming = self.card.capabilities.streaming

        while continue_loop:
            taskId = uuid4().hex
            print('=========  starting a new task ======== ')
            continue_loop = await self.completeTask(
                streaming, taskId, messages
            )

            if self.history and continue_loop:
                print('========= history ======== ')
                task_response = await self.client.get_task(
                    {'id': taskId, 'historyLength': 10}
                )
                print(
                    task_response.model_dump_json(include={'result': {'history': True}})
                )

    async def completeTask(self, streaming: bool, taskId: str, messages: list[str]):
        parts = [
            {
              "type": "text",
              "text": message,
            }
            for message in messages
        ]
        
        conversation_id = self.sessionId
        request: TaskSendParams = TaskSendParams(
            id=str(taskId),
            sessionId=self.sessionId,
            message=Message(
                role='user',
                parts=parts,
                metadata={},
            ),
            acceptedOutputModes=['text', 'text/plain', 'image/png'],
            metadata={'conversation_id': conversation_id},
        )

        taskResult = None
        if streaming:
            response_stream = self.client.send_task_streaming(request)
            async for result in response_stream:
                result = result.result
                print(f'stream event => {result.model_dump_json(exclude_none=True)}')
                if (
                    result
                    and isinstance(result, TaskStatusUpdateEvent)
                    and result.final
                ):
                    return False
        else:
            taskResult = await self.client.send_task(request)
            print(f'\ntask result => {taskResult.model_dump_json(exclude_none=True)}')
            ## if the result is that more input is required, loop again.
            state = TaskState(taskResult.result.status.state)
            if state.name == TaskState.INPUT_REQUIRED.name:
                return await self.completeTask(streaming, taskId)
            else:
                ## task is complete
                return False
        return True


if __name__ == '__main__':
    sessionId = "743712a8805942dc991d3e060b8753c0"
    a2a_server_url = 'http://localhost:10002'
    agent = A2AAgent(a2a_server_url=a2a_server_url, session=sessionId, history=True)
    
    PROMPT_TO_TEST = """
Here will be your task, please do it from step by step, one task is done you will able to move to next task. DO NOT use liquidity heatmap tool, function for analyzing:

Fetching every whales on some markets Find trading patterns and strategies identified based on latest whales activity, histocial trading pnl Risk assessment of all current positions Analyze market trend based on 30 days of tokens Define short-term trades as many as possible that can be executed with safety scoring and entries, stop loss, take profit, concise description, bias including short-term or long-term trades. The entries should be closest to latest price, stop loss and take profit should be realistic which is not too far from entry. Write report into a md file and remain the wallet address for checking instead of shorting it, and give me a time that i should generate this report again since the trades may change    
    
"""
    asyncio.run(agent.step([PROMPT_TO_TEST]))
