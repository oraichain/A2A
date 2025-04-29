import asyncio
import sys
import os
import platform
from pathlib import Path
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken


async def run_code_executor_agent() -> None:
    """Run a code executor agent with fallback from Docker to local executor."""
    
    # Create coding directory if it doesn't exist
    coding_dir = Path("coding")
    coding_dir.mkdir(exist_ok=True)
    
    # First try Docker executor
    try:
        from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
        
        is_macos = platform.system() == "Darwin"
        if is_macos:
            print("MacOS detected - checking Docker Desktop status...")
            # On macOS, check if Docker Desktop is running
            import subprocess
            result = subprocess.run(["ps", "-ef"], capture_output=True, text=True)
            if "Docker Desktop.app" not in result.stdout and "com.docker.docker" not in result.stdout:
                print("Warning: Docker Desktop process not detected. Make sure it's running.")
                print("Tip: Check the Docker icon in your menu bar - it should be running (not initializing).")
        
        print("Attempting to use Docker code executor...")
        code_executor = DockerCommandLineCodeExecutor(work_dir=str(coding_dir))
        await code_executor.start()
        print("Docker executor started successfully.")
    except (RuntimeError, ImportError) as e:
        # Fall back to local executor if Docker is not available
        print(f"Docker executor failed: {e}")
        print("Falling back to local code executor...")
        from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
        code_executor = LocalCommandLineCodeExecutor(work_dir=str(coding_dir))
        await code_executor.start()
        print("Local executor started successfully.")
    
    # Create the agent with whichever executor was successful
    code_executor_agent = CodeExecutorAgent("code_executor", code_executor=code_executor)

    # Run the agent with a given code snippet
    task = TextMessage(
        content='''Here is some code
```python
print('Hello world')
import platform
print(f"Running on Python {platform.python_version()}")
print(f"Nguyen Xuan Thanh")
print(3+5)
```
''',
        source="user",
    )
    
    try:
        response = await code_executor_agent.on_messages([task], CancellationToken())
        print("\nExecution result:")
        print(response.chat_message)
    finally:
        # Always stop the executor when done
        await code_executor.stop()


if __name__ == "__main__":
    asyncio.run(run_code_executor_agent()) 