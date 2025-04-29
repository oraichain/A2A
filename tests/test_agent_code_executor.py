import asyncio
import sys
import platform
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_core import CancellationToken


async def run_code_executor_agent() -> None:
    try:
        # Create a code executor agent that uses a Docker container to execute code.
        code_executor = DockerCommandLineCodeExecutor(work_dir="coding", container_name="code-executor")
        await code_executor.start()
        code_executor_agent = CodeExecutorAgent("code_executor", code_executor=code_executor)

        # Run the agent with a given code snippet.
        task = TextMessage(
            content='''Here is some code
```python
import os
os.system('pip install numpy')

import numpy as np
print(np.__version__)
```
''',
            source="user",
        )
        response = await code_executor_agent.on_messages([task], CancellationToken())
        print(response.chat_message)

        # Stop the code executor.
        await code_executor.stop()
    
    except RuntimeError as e:
        if "Failed to connect to Docker" in str(e):
            is_macos = platform.system() == "Darwin"
            print("\nError: Docker is not available or not running.")
            
            if is_macos:
                print("\nMacBook-specific troubleshooting:")
                print("1. Ensure Docker Desktop is running - check the Docker icon in your menu bar")
                print("2. If Docker Desktop isn't starting, try restarting it from Applications")
                print("3. Run this command to verify Docker is running: 'docker ps'")
                print("4. Check Docker Desktop settings for any resource limitations")
                print("5. Try running 'docker system prune' to clear unused Docker resources")
                print("\nIf you still have issues, try restarting Docker Desktop or your Mac")
            else:
                print("To fix this issue:")
                print("1. Make sure Docker is installed: https://docs.docker.com/get-docker/")
                print("2. Ensure Docker daemon is running")
                print("3. Verify your user has permissions to access Docker")
            
            print("\nAlternative: If you don't want to use Docker, you can try using a local code executor instead.")
            print("Example:")
            print("from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor")
            print("code_executor = LocalCommandLineCodeExecutor(work_dir='coding')")
            sys.exit(1)
        else:
            raise


if __name__ == "__main__":
    asyncio.run(run_code_executor_agent())
