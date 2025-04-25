from mem0 import Memory
from qdrant_client import QdrantClient
import os

class MemoryManager:
    def __init__(self, name: str, in_mem_vector_store: bool = False):
        self.memory = Memory.from_config(
            {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": name,
                        "client": QdrantClient(":memory:") if in_mem_vector_store else QdrantClient(host="localhost", port=6333),
                        "embedding_model_dims": 768,  # Change this according to your local model's dimensions
                    },
                },
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": "gpt-4o-mini",
                        "temperature": 0,
                        "max_tokens": 40000,
                        # "ollama_base_url": "https://litellm.distilled.ai/",
                        "api_key": os.getenv("API_KEY"),
                    },
                },
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "model": "text-embedding-3-small",
                        # "ollama_base_url": "https://litellm.distilled.ai/",
                        "api_key": os.getenv("API_KEY"),
                    },
                },
            }
        )

    def get_memory(self):
        return self.memory

    def add_memory(self, session_id: str, messages: list[dict]):
        self.memory.add(messages=messages, user_id=session_id)

    def relevant_memories(self, session_id: str, query: str):
        return self.memory.search(query, user_id=session_id).get("results")

    def delete_session(self, session_id: str):
        self.memory.delete_all(user_id=session_id)
