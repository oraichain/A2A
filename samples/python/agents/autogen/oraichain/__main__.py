from qdrant_client import QdrantClient
from common.server import A2AServer
from common.types import AgentCard, AgentCapabilities, AgentSkill, MissingAPIKeyError
from agents.autogen.task_manager import AgentTaskManager
from agents.autogen.agent import Agent
from autogen_ext.tools.mcp import McpServerParams
import click
import os
import logging
import asyncio
import signal
import sys
from dotenv import load_dotenv
import pathlib

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the path to the .env file
# current_dir = os.path.abspath(os.getcwd())
# env_path = os.path.join(current_dir, '.env')
# logger.info(f"Loading .env from: {env_path}")
# load_dotenv(dotenv_path=env_path)

# # Debug prints
# logger.info(f"API_KEY is set: {bool(os.getenv('API_KEY'))}")
# logger.info(f"LLM_MODEL is set: {bool(os.getenv('LLM_MODEL'))}")
# logger.info(f"MCP_SERVER_URL is set: {bool(os.getenv('MCP_SERVER_URL'))}")
# logger.info(f"Current working directory: {os.getcwd()}")


@click.command()
@click.option("--host", default="localhost")
@click.option("--port", default=10002)
def main(host, port):
    try:
        api_key = os.getenv("API_KEY")
        llm_model = os.getenv("LLM_MODEL")
        mcp_server_url = os.getenv("MCP_SERVER_URL") or "http://15.235.225.246:4010/sse"
        
        if api_key:
            logger.info(f"API_KEY: {api_key[:5]}... (truncated)")
        logger.info(f"LLM_MODEL: {llm_model}")
        logger.info(f"MCP_SERVER_URL: {mcp_server_url}")
        
        if not api_key:
            raise MissingAPIKeyError("API_KEY environment variable not set.")
        if not llm_model:
            raise MissingAPIKeyError("LLM_MODEL environment variable not set.")

        capabilities = AgentCapabilities(streaming=True)
        skill = AgentSkill(
            id="OraichainExplorer",
            name="OraichainExplorerAgent",
            description="Agent capable of providing Oraichain-related data, from wallet balances, different token types, delegation, validators to transaction history.",
            tags=["oraichain", "balance", "delegation", "validators", "transactions"],
            examples=["Query the balance of these wallets: orai1234, orai1235, orai1236"],
        )
        agent_card = AgentCard(
            name="OraichainExplorerAgent",
            description="Explore the Oraichain and provide insights on the best stablecoin farming strategies",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=["text", "text/plain"],
            defaultOutputModes=["text", "text/plain"],
            capabilities=capabilities,
            skills=[skill],
        )
        agent = Agent(
            label="OraichainExplorerAgent",
            system_instruction="You are an expert Oraichain Explorer. Your role is to query Oraichain-related data, from wallet balances, different token types, delegation, validators to transaction history. You can use the tools provided to you to get the data. If there are multiple wallets, you can query the data for each wallet separately. The task is finished when you have fully retrieved the required data.",
            supported_content_types=["text", "text/plain"],
            in_mem_vector_store=True,
        )
        
        # Initialize the agent with the MCP server URL
        asyncio.run(
            agent.initialize_with_mcp_sse_urls(
                sse_mcp_server_urls=[mcp_server_url]
            )
        )
        
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=agent),
            host=host,
            port=port,
        )
        
        # Set up signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Shutting down server...")
            # Run the shutdown method in the event loop
            loop = asyncio.get_event_loop()
            loop.run_until_complete(agent.shutdown())
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start the server
        server.start()
    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        import traceback
        logger.error(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    main()
