# test_artifact_loss.py
import asyncio
import os
import logging
from uuid import uuid4
from common.client import A2ACardResolver, A2AClient
from common.types import Message, TaskSendParams, TaskState, TaskStatusUpdateEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArtifactLossTest:
    def __init__(self, a2a_server_url: str):
        self.a2a_server_url = a2a_server_url
        self.card_resolver = A2ACardResolver(a2a_server_url)
        self.card = self.card_resolver.get_agent_card()
        self.client = A2AClient(agent_card=self.card)
        self.sessionId = uuid4().hex
        self.artifacts_received = []
    
    async def test_artifact_loss(self):
        """Test case that demonstrates artifact loss when streaming fails"""
        logger.info("Starting artifact loss test...")
        
        # 1. Create a long-running task
        task_id = uuid4().hex
        request = TaskSendParams(
            id=str(task_id),
            sessionId=self.sessionId,
            message=Message(
                role='user',
                parts=[{
                    "type": "text",
                    "text": "Perform a long-running analysis of Oraichain ecosystem. Include detailed information about validators, staking, and tokenomics."
                }],
                metadata={},
            ),
            acceptedOutputModes=['text', 'text/plain'],
            metadata={'conversation_id': self.sessionId},
        )
        
        # 2. Start streaming but interrupt it partway through
        try:
            logger.info(f"Starting streaming task {task_id}")
            response_stream = self.client.send_task_streaming(request)
            
            # Process only part of the stream before interruption
            count = 0
            async for result in response_stream:
                count += 1
                result = result.result
                logger.info(f"Received stream event: {type(result)}")
                
                # Store any artifacts we receive
                if hasattr(result, 'artifact'):
                    self.artifacts_received.append(result.artifact)
                    logger.info(f"Captured artifact: {result.artifact}")
                
                # Simulate connection interruption after a few events
                if count >= 3:
                    logger.info("Simulating connection interruption")
                    await response_stream.aclose()
                    break
        
        except Exception as e:
            logger.error(f"Stream interrupted: {e}")
        
        # 3. After interruption, try to retrieve the task to see if artifacts are lost
        await asyncio.sleep(2)  # Wait a moment
        
        try:
            logger.info(f"Retrieving task after interruption")
            task_response = await self.client.get_task(
                {'id': task_id, 'historyLength': 10}
            )
            
            logger.info(f"Task status: {task_response.result.status.state}")
            
            # Check if artifacts are present
            if hasattr(task_response.result, 'artifacts') and task_response.result.artifacts:
                logger.info(f"Retrieved artifacts: {len(task_response.result.artifacts)}")
                for artifact in task_response.result.artifacts:
                    logger.info(f"Artifact part: {artifact.parts[0].text[:100]}...")
            else:
                logger.warning("NO ARTIFACTS FOUND IN TASK RESPONSE - Demonstrating the bug!")
            
            # Compare with artifacts we captured during streaming
            logger.info(f"Artifacts captured during streaming: {len(self.artifacts_received)}")
            
            # 4. Demonstrate lack of retry mechanism
            logger.info("Attempting to resume the task (will fail due to lack of retry mechanism)")
            try:
                # Try to reconnect to the same task
                # This should fail because there's no retry mechanism
                resume_response = await self.client.send_task(request)
                logger.info(f"Resume response: {resume_response}")
            except Exception as e:
                logger.error(f"Failed to resume task: {e}")
                
        except Exception as e:
            logger.error(f"Error retrieving task: {e}")

async def main():
    a2a_server_url = 'http://localhost:10002'  # Update with your server URL
    test = ArtifactLossTest(a2a_server_url=a2a_server_url)
    await test.test_artifact_loss()

if __name__ == '__main__':
    asyncio.run(main())