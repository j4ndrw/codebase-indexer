from typing import Any

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult


class ChainStreamHandler(BaseCallbackHandler):
    def __init__(self, queue) -> None:
        super().__init__()
        self._queue = queue
        self._stop_signal = None

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self._queue.put(token)

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        self._queue.put(self._stop_signal)
