from common.server import A2AServer
from common.types import AgentCard, AgentCapabilities, AgentSkill, MissingAPIKeyError
from agents.langgraph.task_manager import AgentTaskManager
from agents.langgraph.rewoo_agent_wrapper import ReWOOAgentWrapper
from autogen_ext.tools.mcp import SseServerParams
import click
import os
import logging
import asyncio
from dotenv import load_dotenv
from litellm import verbose_logger as logger

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=10000)
def main(host, port):
    url = os.getenv('DOMAIN_URL', f'http://{host}:{port}/')
    print(f"URL: {url}")
    try:
        if not os.getenv("PLANNER_API_KEY") or not os.getenv("ANALYSIS_API_KEY"):
            raise MissingAPIKeyError("API_KEY environment variable not set.")
        if not os.getenv("PLANNER_LLM_MODEL") or not os.getenv("ANALYSIS_LLM_MODEL"):
            raise MissingAPIKeyError("LLM_MODEL environment variable not set.")

        description = """
        Analyze whale trading patterns, positions, and portfolio changes over time. Provide insights and trade suggestions.
        """
        capabilities = AgentCapabilities(streaming=True)
        skill = AgentSkill(
            id="hyperwhales",
            name="Hyperwhales",
            description=description,
            tags=["hyperwhales"],
            examples=["Find me some trades now based on whales data and history"],
        )
        agent_card = AgentCard(
            name="Hyperwhales",
            description=description,
            url=url,
            version="1.0.0",
            defaultInputModes=["text", "text/plain"],
            defaultOutputModes=["text", "text/plain"],
            capabilities=capabilities,
            skills=[skill],
        )
        agent = ReWOOAgentWrapper(
            sse_mcp_server_sessions={
                'hyperwhales': {
                    'url': "http://localhost:4000/sse",
                    'transport': 'sse',
                }
            }
        )
        asyncio.run(
            agent.initialize()
        )
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=agent),
            host=host,
            port=port,
        )
        server.start()
    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        exit(1)


if __name__ == "__main__":
    main()
