from common.server import A2AServer
from common.types import AgentCard, AgentCapabilities, AgentSkill, MissingAPIKeyError
from agents.autogen.task_manager import AgentTaskManager
from agents.autogen.agent import Agent
from autogen_ext.tools.mcp import SseServerParams
import click
import os
import logging
import asyncio
from dotenv import load_dotenv

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
        if not os.getenv("API_KEY"):
            raise MissingAPIKeyError("API_KEY environment variable not set.")
        if not os.getenv("LLM_MODEL"):
            raise MissingAPIKeyError("LLM_MODEL environment variable not set.")

        capabilities = AgentCapabilities(streaming=True)
        skill = AgentSkill(
            id="hyperwhales",
            name="Hyperwhales",
            description="Analyze whale trading patterns, positions, and portfolio changes over time. Provide insights and trade suggestions.",
            tags=["hyperwhales"],
            examples=["Find me some trades now based on whales data and history"],
        )
        agent_card = AgentCard(
            name="Hyperwhales",
            description="Analyze whale trading patterns, positions, and portfolio changes over time. Provide insights and trade suggestions.",
            url=url,
            version="1.0.0",
            defaultInputModes=["text", "text/plain"],
            defaultOutputModes=["text", "text/plain"],
            capabilities=capabilities,
            skills=[skill],
        )
        agent = Agent(
            label="HyperwhalesAgent",
            ## TODO: add system instruction to folder
            system_instruction="""You are a Perpetual Whales Agent agent who is an expert analyst specializing in detecting whale trading patterns with years of experience understanding deeply crypto trading behavior, on-chain metrics, and derivatives markets, you have developed a keen understanding of whale trading strategies.You can identify patterns in whale positions, analyze their portfolio changes over time, and evaluate the potential reasons behind their trading decisions. Your analysis helps traders decide whether to follow whale trading moves or not.""",
            supported_content_types=["text", "text/plain"],
        )
        asyncio.run(
            agent.initialize(
                mcp_server_params=[
                    SseServerParams(url="http://0.0.0.0:4000/sse")
                ]
            )
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
