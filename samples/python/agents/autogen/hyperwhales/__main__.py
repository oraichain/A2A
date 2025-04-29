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
            system_instruction=f"""You are an expert analyst specializing in detecting whale trading patterns. With years of experience understanding deeply crypto trading behavior, on-chain metrics, and derivatives markets, you have developed a keen understanding of whale trading strategies. You can identify patterns in whale positions, analyze their portfolio changes over time, and evaluate the potential reasons behind their trading decisions. Your analysis helps traders decide whether to follow whale trading moves or not. You can execute code to navigate database, analyze data and provide insights. You only generate and run code in python and bash (```bash and ```python), you can install packages using pip install. The database is MongoDB with URI {os.getenv('MONGO_URI')}. Here is the schema of the database: 
### 🗂️ MongoDB Schema Description for `hyperwhales` Database

You are interacting with a MongoDB database that contains three main collections: `positions`, `markets`, and `traderstatistics`. Each collection stores structured, timestamped trading data. Here's a detailed schema breakdown:

---

#### 📄 `positions`
This collection stores **individual open positions** taken by traders on various markets.

Each document contains:
- `address` (string): Wallet address of the trader.
- `token` (string): The token being traded (e.g., BTC).
- `side` (string): Trade direction (`LONG` or `SHORT`).
- `usdSize` (float): Position size in USD.
- `entryPrice` (float): Entry price of the position.
- `leverage` (int): Leverage used for the position.
- `pnl` (float): Current profit or loss.
- `fundingFee` (float): Accumulated funding fee.
- `market` (string): Market platform name (e.g., GMX).
- `snapshotTime` (datetime): Timestamp when the snapshot was taken.

---

#### 📄 `markets`
This collection captures **aggregated market-wide statistics** at snapshot intervals.

Each document contains:
- `positionInfo`: 
  - `longPosition`, `shortPosition`, `totalPosition` (float): Aggregated open positions.
- `marginInfo`: 
  - `longMargin`, `shortMargin`, `totalMargin` (float): Margin amounts used.
- `pnlInfo`: 
  - `longPnl`, `shortPnl`, `totalPnl` (float): Profit or loss across all positions.
- `fundingFeeInfo`: 
  - `longFundingFee`, `shortFundingFee`, `totalFundingFee` (float): Total funding fees.
- `createdAt`, `updatedAt` (datetime): Timestamps for data recording.

---

#### 📄 `traderstatistics`
This collection tracks **per-trader performance metrics** across different time windows.

Each document contains:
- `address` (string): Wallet address of the trader.
- `market` (string): Market platform name (e.g., HYPERLIQUID).
- `totalPnl` (float): Lifetime profit or loss.
- `totalPnl24H`, `totalPnl48H`, `totalPnl7D`, `totalPnl30D` (float): PnL over recent time windows.
- `snapshotTime` (datetime): Time when the statistics were captured.

""",
        supported_content_types=["text", "text/plain"],
        )
        asyncio.run(
            agent.initialize(
                mcp_server_params=[
                    SseServerParams(url="http://15.235.225.246:4010/sse")
                    # SseServerParams(url="http://localhost:4000/sse")
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
