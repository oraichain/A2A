import asyncio
import pytest
import multiprocessing
import time
import requests
import json
import uuid
from unittest.mock import patch, MagicMock

from common.client import A2AClient


class TestServerMultipleRequests:
    """
    Tests for the server's handling of multiple concurrent requests.
    
    Note: These are integration tests that require starting the actual server.
    They should be run in a controlled environment with proper setup.
    """
    
    @pytest.fixture
    def server_process(self):
        """Start the server in a separate process for testing."""
        # Mock the environment variables
        env_patcher = patch.dict('os.environ', {
            'API_KEY': 'test_api_key',
            'LLM_MODEL': 'test_model',
            'MCP_SERVER_URL': 'http://localhost:4010/sse'
        })
        env_patcher.start()
        
        # Mock the Agent class to avoid actual LLM calls
        from agents.autogen.agent import Agent
        original_init = Agent.__init__
        original_stream = Agent.stream
        original_initialize = Agent.initialize_with_mcp_sse_urls
        
        def mock_init(self, *args, **kwargs):
            self.SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]
            self.label = kwargs.get('label', 'TestAgent')
            self.system_instruction = kwargs.get('system_instruction', '')
            return None
            
        async def mock_stream(self, query, session_id):
            await asyncio.sleep(0.1)  # Simulate processing time
            yield {
                "content": f"Response for {query} in session {session_id}",
                "images": [],
                "is_task_complete": True,
                "require_user_input": False,
            }
            
        async def mock_initialize(*args, **kwargs):
            return None
            
        Agent.__init__ = mock_init
        Agent.stream = mock_stream
        Agent.initialize_with_mcp_sse_urls = mock_initialize
        
        # Import main module
        from agents.autogen.oraichain.__main__ import main
        
        # Start server in a separate process
        host = "localhost"
        port = 10099  # Use a different port for testing
        
        # Create and start the process
        server_proc = multiprocessing.Process(
            target=main, 
            args=(host, port)
        )
        server_proc.start()
        
        # Wait for server to start
        time.sleep(2)
        
        yield f"http://{host}:{port}"
        
        # Cleanup
        server_proc.terminate()
        server_proc.join(timeout=5)
        
        if server_proc.is_alive():
            server_proc.kill()
            
        Agent.__init__ = original_init
        Agent.stream = original_stream
        Agent.initialize_with_mcp_sse_urls = original_initialize
        env_patcher.stop()
    
    def test_server_health(self, server_process):
        """Test that the server is running and returns an agent card."""
        url = f"{server_process}/.well-known/agent.json"
        response = requests.get(url)
        assert response.status_code == 200
        agent_card = response.json()
        assert "name" in agent_card
        assert "description" in agent_card
        
    def test_multiple_concurrent_requests(self, server_process):
        """Test sending multiple concurrent requests to the server."""
        # First get the agent card
        url = f"{server_process}/.well-known/agent.json"
        response = requests.get(url)
        agent_card = response.json()
        
        # Create client
        client = A2AClient(agent_card)
        
        # Send multiple concurrent streaming requests
        async def send_request(request_id):
            task_id = str(uuid.uuid4())
            session_id = str(uuid.uuid4())
            
            from common.types import TaskSendParams, Message, SendTaskStreamingRequest
            
            params = TaskSendParams(
                id=task_id,
                sessionId=session_id,
                message=Message(
                    role="user",
                    parts=[{"type": "text", "text": f"Test request {request_id}"}],
                ),
                acceptedOutputModes=["text", "text/plain"],
            )
            
            results = []
            async for result in client.send_task_streaming(params):
                results.append(result)
            
            return task_id, session_id, results
        
        # Run 5 concurrent requests
        async def run_concurrent_requests():
            tasks = [send_request(i) for i in range(5)]
            return await asyncio.gather(*tasks)
            
        results = asyncio.run(run_concurrent_requests())
        
        # Validate the results
        assert len(results) == 5
        
        for task_id, session_id, task_results in results:
            # Should get at least one result per request
            assert len(task_results) >= 1
            
            # Final response should have a completed status
            final_result = task_results[-1].result
            assert final_result.status.state.name == "COMPLETED"
            
            # Verify session isolation - each response should include its own session ID
            for result in task_results:
                if hasattr(result.result.status, "message") and result.result.status.message:
                    assert session_id in result.result.status.message.parts[0]["text"] or task_id in result.result.status.message.parts[0]["text"]


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 