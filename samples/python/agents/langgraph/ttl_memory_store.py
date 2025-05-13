import time
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import NOT_PROVIDED, NotProvided
from typing import Any, Literal, Optional, Union, override
import logging
logger = logging.getLogger(__name__)

class TTLInMemoryStore(InMemoryStore):
  
    @override
    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Optional[Union[Literal[False], list[str]]] = None,
        *,
        ttl: Union[Optional[float], "NotProvided"] = NOT_PROVIDED):
        super().put(namespace, key, value, index)
        # cleanup old data
        # NOTE: limit=10 for simplicity. In practice, we should use a more comprehensive approach.
        old_data = self.search(namespace, limit=10)
        for item in old_data:
            ttl_seconds = ttl * 60
            current_time = time.time()
            if item.updated_at.timestamp() < current_time - ttl_seconds:
                logger.info(f"Deleting old data with namespace {namespace} and key {item.key}")
                self.delete(namespace, item.key)
                
    @override
    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Optional[Union[Literal[False], list[str]]] = None,
        *,
        ttl: Union[Optional[float], "NotProvided"] = NOT_PROVIDED):
        await super().aput(namespace, key, value, index)
        # cleanup old data
        # NOTE: limit=10 for simplicity. In practice, we should use a more comprehensive approach.
        old_data = self.search(namespace, limit=10)
        for item in old_data:
            ttl_seconds = ttl * 60
            current_time = time.time()
            if item.updated_at.timestamp() < current_time - ttl_seconds:
                logger.info(f"Deleting old data with namespace {namespace} and key {item.key}")
      
        