import asyncio
import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from common.types import (
    SendTaskStreamingRequest,
    TaskSendParams,
    Message,
    TextPart,
    TaskState,
    JSONRPCResponse,
)
from agents.autogen.task_manager import AgentTaskManager
from agents.autogen.agent import Agent


class MockAgent:
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]
    
    def __init__(self, delay=0.1):
        self.stream_calls = []
        self.delay = delay
    
    async def stream(self, query, session_id):
        self.stream_calls.append((query, session_id))
        
        # First message - working state
        yield {
            "content": f"Working on {query} for session {session_id}",
            "images": [],
            "is_task_complete": False,
            "require_user_input": False,
        }
        
        # Simulate processing time
        await asyncio.sleep(self.delay)
        
        # Final message - task complete
        yield {
            "content": f"Completed {query} for session {session_id}",
            "images": [],
            "is_task_complete": True,
            "require_user_input": False,
        }


@pytest.fixture
def mock_agent():
    return MockAgent()


@pytest.fixture
def task_manager(mock_agent):
    return AgentTaskManager(agent=mock_agent)


def create_streaming_request():
    task_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    
    return SendTaskStreamingRequest(
        id=1,
        method="sendTaskStreaming",
        params=TaskSendParams(
            id=task_id,
            sessionId=session_id,
            message=Message(
                role="user",
                parts=[{"type": "text", "text": f"Test query for {task_id}"}],
            ),
            acceptedOutputModes=["text", "text/plain"],
        ),
    )


async def collect_streaming_responses(generator):
    responses = []
    async for response in generator:
        responses.append(response)
    return responses


@pytest.mark.asyncio
async def test_single_request(task_manager):
    """Test that a single request is processed correctly."""
    request = create_streaming_request()
    
    # Get the response generator
    response_generator = await task_manager.on_send_task_subscribe(request)
    
    # Collect all responses
    responses = await collect_streaming_responses(response_generator)
    
    # Verify we got exactly 2 responses (working and completed)
    assert len(responses) == 2
    
    # First response should be WORKING state
    assert responses[0].result.status.state == TaskState.WORKING
    assert "Working on" in responses[0].result.status.message.parts[0]["text"]
    
    # Second response should be COMPLETED state and final=True
    assert responses[1].result.status.state == TaskState.COMPLETED
    assert responses[1].result.final == True


@pytest.mark.asyncio
async def test_multiple_concurrent_requests(task_manager):
    """Test that multiple concurrent requests are handled correctly with proper isolation."""
    # Create 5 concurrent requests
    requests = [create_streaming_request() for _ in range(5)]
    
    # Start all requests concurrently
    tasks = []
    for request in requests:
        response_generator = await task_manager.on_send_task_subscribe(request)
        task = asyncio.create_task(collect_streaming_responses(response_generator))
        tasks.append((request, task))
    
    # Wait for all responses
    results = []
    for request, task in tasks:
        responses = await task
        results.append((request.params.id, request.params.sessionId, responses))
    
    # Verify each request was handled correctly
    for task_id, session_id, responses in results:
        # Each request should get exactly 2 responses
        assert len(responses) == 2
        
        # First response should be WORKING state
        assert responses[0].result.status.state == TaskState.WORKING
        
        # Second response should be COMPLETED state and final=True
        assert responses[1].result.status.state == TaskState.COMPLETED
        assert responses[1].result.final == True
        
    # Verify that the agent was called with the correct parameters for each request
    assert len(task_manager.agent.stream_calls) == 5
    
    # Verify each request used its own task ID and session ID
    session_ids = set(session_id for _, session_id, _ in results)
    assert len(session_ids) == 5


@pytest.mark.asyncio
async def test_input_required_state(task_manager):
    """Test handling of a request that requires user input."""
    
    async def mock_stream(query, session_id):
        yield {
            "content": "Need more information",
            "images": [],
            "is_task_complete": False,
            "require_user_input": True,
        }
    
    # Replace the stream method temporarily
    original_stream = task_manager.agent.stream
    task_manager.agent.stream = mock_stream
    
    request = create_streaming_request()
    response_generator = await task_manager.on_send_task_subscribe(request)
    responses = await collect_streaming_responses(response_generator)
    
    # Restore original stream method
    task_manager.agent.stream = original_stream
    
    # Should get one response with INPUT_REQUIRED state and final=True
    assert len(responses) == 1
    assert responses[0].result.status.state == TaskState.INPUT_REQUIRED
    assert responses[0].result.final == True


@pytest.mark.asyncio
async def test_task_resubscription(task_manager):
    """Test resubscription to an existing task."""
    # First create and process a task
    original_request = create_streaming_request()
    response_generator = await task_manager.on_send_task_subscribe(original_request)
    await collect_streaming_responses(response_generator)
    
    # Now attempt to resubscribe to this task
    from common.types import TaskResubscriptionRequest, TaskIdParams
    
    resubscribe_request = TaskResubscriptionRequest(
        id=2,
        method="resubscribeToTask", 
        params=TaskIdParams(id=original_request.params.id)
    )
    
    # Should return an error because the task is already completed
    response = await task_manager.on_resubscribe_to_task(resubscribe_request)
    assert isinstance(response, JSONRPCResponse)
    assert response.error is not None


@pytest.mark.asyncio
async def test_error_handling(task_manager):
    """Test error handling in the task manager."""
    
    async def mock_stream_with_error(query, session_id):
        raise Exception("Test error")
    
    # Replace the stream method temporarily
    original_stream = task_manager.agent.stream
    task_manager.agent.stream = mock_stream_with_error
    
    request = create_streaming_request()
    response_generator = await task_manager.on_send_task_subscribe(request)
    responses = await collect_streaming_responses(response_generator)
    
    # Restore original stream method
    task_manager.agent.stream = original_stream
    
    # Should get one response with an error
    assert len(responses) == 1
    assert responses[0].error is not None
    assert "An error occurred" in responses[0].error.message


if __name__ == "__main__":
    pytest.main(["-xvs", "test_task_manager.py"]) 