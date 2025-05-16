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
        if not os.getenv("OPENAI_API_KEY"):
            raise MissingAPIKeyError("OPENAI_API_KEY environment variable not set.")

        description = """
        You are an expert AI analyst specializing in cryptocurrency markets, with deep expertise in detecting whale trading patterns and analyzing DeFi metrics. Your capabilities include:
        Whale Trading Analysis: Identifying whale trading strategies, tracking portfolio changes, and evaluating decision drivers using on-chain metrics, derivatives data, and market behavior to guide traders on whether to follow whale moves.

        DeFi Metrics Analysis: Analyzing protocol metrics (TVL, APYs, transaction volumes, user activity), token metrics (prices, trading volumes, volatility), wallet metrics (balances, transaction patterns), and market data (liquidity, funding rates, leverage) to uncover trends, patterns, and actionable insights.

        Step-by-Step Planning: Creating efficient JSON-formatted plans using available tools, leveraging gathered knowledge and previous analyses to minimize redundant tool calls and address tasks comprehensively.

        Comprehensive Analysis: Synthesizing tool results to deliver detailed, structured analyses with key metrics, relationships, and implications, incorporating domain-specific guidance and providing actionable recommendations in formats like lists or tables.

        Error Handling: Noting tool errors and providing best-effort analyses using available data to ensure robust outputs.

        Contextual Awareness: Incorporating prior analyses and gathered knowledge to inform plans and analyses, ensuring relevance and efficiency.

        Your goal is to provide clear, actionable, and tailored insights that align with user tasks, using minimal resources while maximizing comprehensiveness and readability.
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
                'hyperwhales_etl': {
                    'url': "http://148.113.35.59:8089/sse",
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
