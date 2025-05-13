# Prerequisites

# Installation

1. Create venv and activate

```bash
# enter A2A sample dirs
cd samples/python

uv venv --python 3.12
source .venv/bin/activate  # On Unix/macOS
```

2. Install deps

```bash
# install deps for the overall workspace
uv pip install -r pyproject.toml
# install deps for autogen workspace member
uv pip install -e agents/langgraph
```

3. Create an .env file

```bash
cp .rewoo.env.example .env
```

Then, fill the .env with your keys and appropriate MCP servers

# Start A2A server

```bash
uv run agents/langgraph/__main__.py
```

# Start a client asking some questions

```bash
uv run agents/langgraph/oh_client.py
```