# mypy: ignore-errors
import asyncio
from typing import AsyncGenerator
from uuid import uuid4

from common.client import A2ACardResolver, A2AClient
from common.types import Message, SendTaskResponse, SendTaskStreamingResponse, Task, TaskArtifactUpdateEvent, TaskSendParams, TaskState, TaskStatusUpdateEvent

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
            try:
                async for task_response in self.send_task(
                    streaming, taskId, messages
                ):
                    if task_response is None or task_response.result is None:
                        continue
                    result = task_response.result

                    if isinstance(result, TaskStatusUpdateEvent):
                        # TODO: We need a mechanism to store TaskStatusUpdateEvent, in case there's something wrong -> return everything so far / summarize the task so far.
                        print(
                            f"""Task state: {result.status.state}\n
                            Task message: {result.status.message}"""
                        )
                        # yield A2ASendTaskUpdateObservation(
                        #     agent_name=action.agent_name,
                        #     task_update_event=result,
                        #     content=result.model_dump_json(),
                        # )
                    elif isinstance(result, TaskArtifactUpdateEvent):
                        # TODO: Need to verify the artifact's quality. It should be a quality summary of all the task status update events.
                        # Worst case, it should be list of all the task status update events.
                        # yield A2ASendTaskArtifactObservation(
                        #     agent_name=action.agent_name,
                        #     task_artifact_event=result,
                        #     content=result.model_dump_json(),
                        # )
                        pass
                    elif isinstance(result, Task):
                        # yield A2ASendTaskResponseObservation(
                        #     agent_name=action.agent_name,
                        #     task=result,
                        #     content=result.model_dump_json(),
                        # )
                        pass
            except Exception as e:
                print(f'Error sending task: {e}')
                # we should handle error more gracefully.
                # Eg: Keep all of the progress so far somewhere, and ask the user if they want to continue.
            continue_loop = await self.send_task(
                streaming, taskId, messages
            )

            if self.history and continue_loop:
                print('========= history ======== ')
            # don't pass historyLength. By default get all history.
                task_response = await self.client.get_task(
                    {'id': taskId}
                )
                print(
                    task_response.model_dump_json(include={'result': {'history': True}})
                )

    async def send_task(
        self, streaming: bool, taskId: str, messages: list[str]
    ) -> AsyncGenerator[SendTaskStreamingResponse | SendTaskResponse, None]:
        """Send a task to a remote agent and yield task responses.

        Args:
            streaming: Whether to stream the task response
            taskId: ID of the task
            messages: List of messages to send to the agent

        Yields:
            TaskStatusUpdateEvent or Task: Task response updates
        """
        
        parts = [
            {
              "type": "text",
              "text": message,
            }
            for message in messages
        ]
        
        request: TaskSendParams = TaskSendParams(
            id=taskId,
            sessionId=self.sessionId,
            message=Message(
                role='user',
                parts=parts,
                metadata={},
            ),
            acceptedOutputModes=['text', 'text/plain', 'image/png'],
            metadata={'conversation_id': self.sessionId},
        )

        if streaming:
            async for response in self.client.send_task_streaming(request):
                yield response
        else:
            response = await self.client.send_task(request)
            yield response

if __name__ == '__main__':
    sessionId = "743712a8805942dc991d3e060b8753c0"
    a2a_server_url = 'http://localhost:10000'
    agent = A2AAgent(a2a_server_url=a2a_server_url, session=sessionId, history=True)
    
    PROMPT_TO_TEST = """
You are a Perpetual Whales Agent agent who is an expert analyst specializing in detecting whale trading patterns with years of experience understanding deeply crypto trading behavior, on-chain metrics, and derivatives markets, you have developed a keen understanding of whale trading strategies.

    You can identify patterns in whale positions, analyze their portfolio changes over time, and evaluate the potential reasons behind their trading decisions. Your analysis helps traders decide whether to follow whale trading moves or not.

    Here will be your task, please do it from step by step, one task is done you will able to move to next task. DO NOT use liquidity heatmap tool, function for analyzing:

    - Fetching every whales on some markets
    - Find trading patterns and strategies identified based on latest whales activity, histocial trading pnl
    - Risk assessment of all current positions
    - Analyze market trend based on 30 days of tokens
    - Define short-term trades as many as possible that can be executed with safety scoring and entries, stop loss, take profit, concise description, bias including short-term or long-term trades. The entries should be closest to latest price, stop loss and take profit should be realistic which is not too far from entry.

    Write report into a md file and remain the wallet address for checking instead of shorting it
"""
    asyncio.run(agent.step([PROMPT_TO_TEST]))
