import threading
import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult


class StatsCallbackHandler(BaseCallbackHandler):
    """Callback handler that tracks LLM calls, tool calls, and token usage.

    Extended (spec 2026-08-24 §14): records per-call telemetry rows with the
    requested model, latency, and token usage so the lane can attribute cost
    and behavior to individual graph nodes. ``on_llm_start`` receives the
    model name via ``serialized`` kwargs; ``on_llm_end`` adds latency and
    token counts. Raw prompts are NOT retained (privacy; §14).
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.tool_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        # One row per completed call: {model, latency_ms, input_tokens,
        # output_tokens, started_at}
        self.calls: list[dict[str, Any]] = []
        self._pending: dict[str, dict[str, Any]] = {}

    def _model_from_serialized(self, serialized: dict[str, Any]) -> str | None:
        """Best-effort model name from the serialized LLM descriptor."""
        if not serialized:
            return None
        kwargs = serialized.get("kwargs") or {}
        model = kwargs.get("model_name") or kwargs.get("model")
        if model:
            return str(model)
        # fallback: langchain id path tail (e.g. .../ChatOpenAI)
        id_path = serialized.get("id") or []
        if id_path:
            return str(id_path[-1])
        return None

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when an LLM starts."""
        with self._lock:
            self.llm_calls += 1
            run_id = kwargs.get("run_id")
            self._pending[str(run_id)] = {
                "model": self._model_from_serialized(serialized),
                "started_at": time.time(),
            }

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when a chat model starts."""
        with self._lock:
            self.llm_calls += 1
            run_id = kwargs.get("run_id")
            self._pending[str(run_id)] = {
                "model": self._model_from_serialized(serialized),
                "started_at": time.time(),
            }

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Extract token usage and latency from LLM response."""
        try:
            generation = response.generations[0][0]
        except (IndexError, TypeError):
            return

        usage_metadata = None
        if hasattr(generation, "message"):
            message = generation.message
            if isinstance(message, AIMessage) and hasattr(message, "usage_metadata"):
                usage_metadata = message.usage_metadata

        with self._lock:
            run_id = str(kwargs.get("run_id"))
            pending = self._pending.pop(run_id, None)
            if usage_metadata:
                ti = usage_metadata.get("input_tokens", 0)
                to = usage_metadata.get("output_tokens", 0)
                self.tokens_in += ti
                self.tokens_out += to
            else:
                ti, to = 0, 0
            latency_ms = None
            if pending:
                latency_ms = round((time.time() - pending["started_at"]) * 1000)
            self.calls.append({
                "model": (pending or {}).get("model"),
                "latency_ms": latency_ms,
                "input_tokens": ti,
                "output_tokens": to,
            })

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Increment tool call counter when a tool starts."""
        with self._lock:
            self.tool_calls += 1

    def get_stats(self) -> dict[str, Any]:
        """Return current statistics."""
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "tool_calls": self.tool_calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
            }

    def get_calls(self) -> list[dict[str, Any]]:
        """Return per-call telemetry rows (completed calls only)."""
        with self._lock:
            return list(self.calls)
