# test_server_stall.py
import asyncio
import logging
from uuid import uuid4
from datetime import datetime
from common.client import A2ACardResolver, A2AClient
from common.types import Message, TaskSendParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServerStallTest:
    def __init__(self, a2a_server_url: str):
        self.a2a_server_url = a2a_server_url
        self.card_resolver = A2ACardResolver(a2a_server_url)
        self.card = self.card_resolver.get_agent_card()
        self.client = A2AClient(agent_card=self.card)
        self.sessionId = uuid4().hex
    
    async def test_server_stall(self):
        """Test case that demonstrates artifacts loss when server stalls"""
        logger.info("Starting server stall test...")
        
        # 1. Create a task likely to cause a stall
        task_id = uuid4().hex
        request = TaskSendParams(
            id=str(task_id),
            sessionId=self.sessionId,
            message=Message(
                role='user',
                parts=[{
                    "type": "text",
                    # Complex query that might cause the server to stall
                    "text": "Analyze complex transaction patterns across 1000 Oraichain wallets, including detailed token flow visualization, validator reputation metrics, and compute-intensive simulations."
                }],
                metadata={},
            ),
            acceptedOutputModes=['text', 'text/plain'],
            metadata={'conversation_id': self.sessionId},
        )
        
        # 2. Start streaming with client-side timeout
        try:
            logger.info(f"Starting potentially stall-inducing task {task_id}")
            response_stream = self.client.send_task_streaming(request)
            
            # Process the stream with timeout detection
            received_events = 0
            last_event_time = datetime.now()
            stall_timeout = 10  # seconds to wait before considering server stalled
            
            async for result in response_stream:
                # Reset timer when we get an event
                current_time = datetime.now()
                time_since_last = (current_time - last_event_time).total_seconds()
                last_event_time = current_time
                
                received_events += 1
                logger.info(f"Received event #{received_events} after {time_since_last:.2f}s")
                
                if hasattr(result.result, 'artifact'):
                    logger.info(f"Partial artifact: {result.result.artifact.parts[0].text[:50]}...")
                
                # Add client-side timeout detection between events
                try:
                    # Wait for next event with timeout
                    await asyncio.wait_for(
                        asyncio.sleep(0.1),  # Just a small delay to let next event arrive
                        timeout=stall_timeout
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Server appears STALLED - no events for {stall_timeout} seconds!")
                    logger.error("This demonstrates the second bug - server stuck without yielding")
                    break
        
        except Exception as e:
            logger.error(f"Stream error: {e}")
        
        # 3. Try to retrieve the task after stall to see if artifacts were saved
        await asyncio.sleep(2)
        
        try:
            logger.info(f"Retrieving task after potential stall")
            task_response = await self.client.get_task(
                {'id': task_id, 'historyLength': 5}
            )
            
            logger.info(f"Task status: {task_response.result.status.state}")
            
            # Check if artifacts are present
            if hasattr(task_response.result, 'artifacts') and task_response.result.artifacts:
                logger.info(f"Retrieved artifacts: {len(task_response.result.artifacts)}")
                for artifact in task_response.result.artifacts:
                    logger.info(f"Artifact content: {artifact.parts[0].text[:100]}...")
            else:
                logger.warning("NO ARTIFACTS FOUND - Intermediate results lost due to server stall!")
            
            # Check task state - likely stuck in "working" state
            logger.info(f"Task final state: {task_response.result.status.state}")
            if task_response.result.status.state == "working":
                logger.error("Task stuck in 'working' state with no completion mechanism!")
            
        except Exception as e:
            logger.error(f"Error retrieving task: {e}")

async def main():
    a2a_server_url = 'http://localhost:10002'  # Update with your server URL
    test = ServerStallTest(a2a_server_url=a2a_server_url)
    await test.test_server_stall()

if __name__ == '__main__':
    asyncio.run(main())