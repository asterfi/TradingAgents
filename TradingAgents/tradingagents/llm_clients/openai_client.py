import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, ToolCall
from langchain_openai import ChatOpenAI

from .api_key_env import get_api_key_env
from .base_client import BaseLLMClient, normalize_content
from .capabilities import get_capabilities
from .validators import validate_model


class NormalizedChatOpenAI(ChatOpenAI):
    """ChatOpenAI with normalized content output and capability-aware binding.

    The Responses API returns content as a list of typed blocks
    (reasoning, text, etc.). ``invoke`` normalizes to string for
    consistent downstream handling.

    ``with_structured_output`` consults the per-model capability table
    (``capabilities.get_capabilities``) to pick the method and to decide
    whether ``tool_choice`` may be sent. Models that reject ``tool_choice``
    (e.g. DeepSeek V4 and reasoner — per their official tool-calling
    guide) still bind the schema as a tool, but no ``tool_choice``
    parameter is sent.

    Provider-specific quirks beyond structured-output (e.g. DeepSeek's
    reasoning_content roundtrip) live in subclasses so this base class
    stays small.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))

    def with_structured_output(self, schema, *, method=None, **kwargs):
        caps = get_capabilities(self.model_name)
        if caps.preferred_structured_method == "none":
            raise NotImplementedError(
                f"{self.model_name} has no structured-output method available; "
                f"agent factories will fall back to free-text generation."
            )
        method = method or caps.preferred_structured_method
        # When the model rejects tool_choice, suppress langchain's hardcoded
        # value. The schema is still bound as a tool — exactly what
        # DeepSeek's official tool-calling examples do.
        if method == "function_calling" and not caps.supports_tool_choice:
            kwargs.setdefault("tool_choice", None)
        return super().with_structured_output(schema, method=method, **kwargs)


class LocalCompatibleChatOpenAI(NormalizedChatOpenAI):
    """OpenAI-compatible client for arbitrary local servers (LM Studio, vLLM,
    llama.cpp via the generic ``openai_compatible`` provider).

    Their tool-calling support varies, and many reject the object-form
    ``tool_choice`` langchain sends for function-calling structured output. Bind
    the schema as a tool but don't force tool_choice, so structured output works
    across local servers regardless of the model ID's capabilities (#1057).
    """

    def with_structured_output(self, schema, *, method=None, **kwargs):
        resolved = method or get_capabilities(self.model_name).preferred_structured_method
        if resolved == "function_calling":
            kwargs.setdefault("tool_choice", None)
        return super().with_structured_output(schema, method=method, **kwargs)


def _input_to_messages(input_: Any) -> list:
    """Normalise a langchain LLM input to a list of message objects.

    Accepts a list of messages, a ``ChatPromptValue`` (from a
    ChatPromptTemplate), or anything else (treated as no messages).
    Used by providers that need to walk the outgoing message history;
    in particular DeepSeek thinking-mode propagation must work for
    both bare-list invocations and ChatPromptTemplate-driven ones, so
    treating only ``list`` here would silently skip half the call sites.
    """
    if isinstance(input_, list):
        return input_
    if hasattr(input_, "to_messages"):
        return input_.to_messages()
    return []


class DeepSeekChatOpenAI(NormalizedChatOpenAI):
    """DeepSeek-specific overrides on top of the OpenAI-compatible client.

    Thinking-mode round-trip is the only DeepSeek-specific behavior that
    stays here. When DeepSeek's thinking models return a response with
    ``reasoning_content``, that field must be echoed back as part of the
    assistant message on the next turn or the API fails with HTTP 400.
    ``_create_chat_result`` captures it on receive and
    ``_get_request_payload`` re-attaches it on send.

    Tool-choice handling for V4 and reasoner — those models reject the
    ``tool_choice`` parameter — is handled by the capability dispatch in
    ``NormalizedChatOpenAI.with_structured_output``, not here.
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        outgoing = payload.get("messages", [])
        for message_dict, message in zip(outgoing, _input_to_messages(input_), strict=False):
            if not isinstance(message, AIMessage):
                continue
            reasoning = message.additional_kwargs.get("reasoning_content")
            if reasoning is not None:
                message_dict["reasoning_content"] = reasoning
        return payload

    def _create_chat_result(self, response, generation_info=None):
        chat_result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(
                exclude={"choices": {"__all__": {"message": {"parsed"}}}}
            )
        )
        for generation, choice in zip(
            chat_result.generations, response_dict.get("choices", []), strict=False
        ):
            reasoning = choice.get("message", {}).get("reasoning_content")
            if reasoning is not None:
                generation.message.additional_kwargs["reasoning_content"] = reasoning
        return chat_result


class MinimaxChatOpenAI(NormalizedChatOpenAI):
    """MiniMax-specific overrides on top of the OpenAI-compatible client.

    M2.x reasoning models embed ``<think>...</think>`` blocks directly in
    ``message.content`` by default, which would pollute saved reports.
    Per platform.minimax.io/docs/api-reference/text-openai-api,
    ``reasoning_split=True`` redirects the thinking block into
    ``reasoning_details`` so ``content`` stays clean. It is sent via
    ``extra_body`` (not a top-level kwarg) because the openai SDK validates
    top-level params and rejects unknown ones like reasoning_split (#826).

    The flag is gated by ``ModelCapabilities.requires_reasoning_split`` so
    only M2.x reasoning models receive it; non-reasoning MiniMax endpoints
    (Coding Plan, MiniMax-Text-01) never see it.

    Tool-choice handling for M2.x — those models accept only the string
    enum ``{"none", "auto"}`` and reject langchain's function-spec dict —
    is handled by the capability dispatch in
    ``NormalizedChatOpenAI.with_structured_output``, not here.
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if get_capabilities(self.model_name).requires_reasoning_split:
            # Pass via extra_body, not as a top-level kwarg: the openai SDK
            # (>=1.56) validates top-level params against Completions.create
            # and rejects unknown ones like reasoning_split (#826). extra_body
            # is forwarded into the request body untouched.
            extra_body = payload.setdefault("extra_body", {})
            extra_body.setdefault("reasoning_split", True)
        return payload


# Kwargs forwarded from user config to ChatOpenAI
_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "reasoning_effort", "temperature", "top_p",
    "api_key", "callbacks", "http_client", "http_async_client",
    # Output-token cap. Critical for the Nous portal: without it the model's
    # hidden-thinking phase is unbounded (measured 2026-08-23).
    "max_tokens",
)

# OpenAI's ``reasoning_effort`` is only accepted by reasoning models — the GPT-5
# family and the o-series. Non-reasoning models (gpt-4.1, gpt-4o, ...) 400 with
# "Unsupported parameter: 'reasoning.effort' is not supported with this model".
# Drop the kwarg for those rather than crash the run.
_OPENAI_REASONING_MODEL = re.compile(r"^(gpt-5|o[1-9])")


def _supports_reasoning_effort(model: str) -> bool:
    """Whether the (native OpenAI) model accepts ``reasoning_effort``."""
    return bool(_OPENAI_REASONING_MODEL.match(model.lower().strip()))


@dataclass(frozen=True)
class ProviderSpec:
    """Declarative config for one OpenAI-compatible provider.

    The OpenAI-compatible family (OpenAI, xAI, DeepSeek, Qwen, GLM, MiniMax,
    OpenRouter, Ollama, and any user endpoint) all speak the same Chat
    Completions API and differ only by these fields — so one row here replaces
    the former per-provider base-URL dict, auth handling, and client-class
    branches. Native Anthropic / Google use their own clients (genuinely
    different APIs) and are intentionally NOT in this registry.

    The API-key env var stays in ``api_key_env.PROVIDER_API_KEY_ENV`` (the single
    source consulted by both this client and the CLI prompt); only behavior that
    is provider-specific (base URL, key optionality, wire-format quirks via
    ``chat_class``) lives here.
    """

    chat_class: type = NormalizedChatOpenAI   # provider quirks live in the subclass
    base_url: str | None = None            # default endpoint (None -> SDK default)
    base_url_env: str | None = None        # env var that overrides base_url (e.g. OLLAMA_BASE_URL)
    key_optional: bool = False                # don't require/prompt; send a placeholder if unset
    placeholder_key: str = "EMPTY"            # sent when no key is available (keyless local servers)
    require_base_url: bool = False            # error if no base_url is resolved (generic endpoint)
    use_responses_api: bool = False           # native OpenAI Responses API


class NousChatOpenAI(NormalizedChatOpenAI):
    """Nous Research portal quirk guard: retry-on-empty for stealth models.

    ox-alpha (stealth preview) intermittently returns bursty empty completions
    (observed 2026-08-23 on the Hermes gateway: 3 consecutive empty responses).
    langchain's max_retries only retries on API-level errors, not on
    content-free 200s — so a single empty completion would poison an agent's
    report (e.g. an analyst writing "" instead of analysis) and cascade into a
    garbage final verdict. Retry up to 3x on truly-empty text; give up honestly
    after that rather than fabricating output.
    """

    _NOUS_EMPTY_RETRIES = 3

    # 401s (key rejected) are retried this many times with a fresh key before
    # giving up — the portal flaps 401s in brief waves then recovers. With the
    # 10-20s backoff this rides out ~3-4 min of continuous 401s.
    _AUTH_RETRIES = 12

    # Capacity-wave survival: the Nous portal's stealth models intermittently
    # return 429 "temporarily at capacity upstream" (not a key rate limit).
    # Observed 2026-08-23: a wave long enough to exhaust langchain_openai's
    # retry budget (~2 min) and kill the run. This lane is an unattended daily
    # batch with no deadline, so ride waves out: long exponential backoff with
    # jitter (60s → 8m ceiling), up to 12 attempts (~45 min worst case per call).
    _CAPACITY_ATTEMPTS = 12
    _CAPACITY_BASE_S = 60
    _CAPACITY_CEIL_S = 480

    def _refresh_key(self):
        """Re-read Hermes' live Nous key from auth.json (rotates hourly).

        Long graph runs (25-40 min) outlive the rotating agent_key; without
        this a 401 mid-run kills the whole stock lane. Refreshing before every
        attempt costs one tiny file read and keeps the key current.

        IMPORTANT: langchain's OpenAI client caches the key at build time
        (self.root_client). Merely setting self.openai_api_key does NOT affect
        in-flight requests. The client builds LAZILY (`if not self.client:`),
        so resetting client/root_client to None forces a rebuild with the new
        key on the next invoke.
        """
        try:
            with open(os.path.expanduser("~/.hermes/auth.json")) as f:
                nous = json.load(f).get("providers", {}).get("nous", {})
            key = nous.get("agent_key") or nous.get("access_token")
            if not key:
                return
            old = getattr(self, "openai_api_key", None)
            os.environ["NOUS_API_KEY"] = key
            try:
                self.openai_api_key = key
            except Exception:
                pass
            # langchain's OpenAI client caches the key at build time inside
            # validate_environment() (self.root_client). If the key rotated,
            # re-run validate_environment() so the next invoke uses the fresh
            # key. Only rebuild when the key actually changed.
            if old != key:
                try:
                    self.validate_environment()
                except Exception:
                    pass
        except (OSError, json.JSONDecodeError):
            pass

    def invoke(self, input, config=None, **kwargs):
        import random
        import time as _time

        def _has_payload(r):
            return isinstance(r, AIMessage) and (bool((r.content or "").strip()) or bool(r.tool_calls))

        # Capture the base-class stream method once; super() inside a nested
        # closure has no class context.
        _base_stream = super().stream

        def _stream_invoke(input, config, **kwargs):
            """Stream the request internally and aggregate to an AIMessage.

            The Nous portal is dramatically faster and more timeout-resilient
            when streaming (190s non-stream vs 53s stream for the market
            analyst's long report, measured 2026-08-23). Streaming keeps the
            same return shape (AIMessage) so all graph nodes benefit without
            changing their code.

            Tool calls are aggregated the SDK-blessed way: AIMessageChunk
            supports additive `+` merging, which handles tool-call fragments
            (id on first chunk, args split across later chunks) correctly.
            """
            from langchain_core.messages import AIMessageChunk
            acc = None
            for ch in _base_stream(input, config=config, **kwargs):
                m = getattr(ch, "message", ch)
                acc = m if acc is None else (acc + m)
            if acc is None:
                return AIMessage(content="")
            if type(acc).__name__ == "AIMessageChunk":
                # normalize to a plain AIMessage so downstream (langgraph)
                # treats it identically to a non-streaming result
                from langchain_core.messages import AIMessageChunk
                if isinstance(acc, AIMessageChunk):
                    acc = AIMessage(
                        content=acc.content,
                        tool_calls=getattr(acc, "tool_calls", None) or [],
                        additional_kwargs=getattr(acc, "additional_kwargs", None) or {},
                    )
            return acc

        for attempt in range(self._CAPACITY_ATTEMPTS):
            self._refresh_key()
            try:
                result = _stream_invoke(input, config, **kwargs)
                break
            except Exception as e:
                body = getattr(e, "body", None)
                status = getattr(e, "status_code", None) or (
                    body.get("status") if isinstance(body, dict) else None)
                ename = str(type(e).__name__)
                # 401 = key rejected. Portal-down or key-revoked signature. The
                # 2026-08-23 portal flap strikes 401s in brief waves then
                # recovers (observed repeatedly mid-graph). Refresh the key +
                # rebuild the client and retry a few times BEFORE giving up —
                # but cap it so a genuinely dead portal fails fast instead of
                # grinding the full backoff.
                if status == 401 or "AuthenticationError" in ename or "401" in str(e)[:200]:
                    self._refresh_key()
                    if attempt >= self._AUTH_RETRIES:
                        raise RuntimeError(
                            "nous: 401 persists after key refresh — "
                            "portal down or key revoked. Not a capacity issue."
                        ) from e
                    # short backoff so we ride out a 1-3 min 401 flap
                    # (observed repeatedly 2026-08-23) without hammering
                    _time.sleep(random.uniform(10, 20))
                    print(f"[nous] key 401, refreshed (attempt {attempt + 1}/"
                          f"{self._AUTH_RETRIES + 1}); retrying", flush=True)
                    continue
                # Retryable upstream pain: explicit 429s AND timeouts (the
                # 2026-08-23 drought first showed as multi-minute response
                # stalls, then 429s — same capacity queue, two symptoms).
                # ALSO retry mid-stream drops (RemoteProtocolError /
                # incomplete reads) — the portal drops long streaming
                # generations intermittently; a retry typically succeeds.
                rate_limited = (status == 429 or "429" in ename
                                or "Timeout" in ename or "timeout" in str(e)[:200]
                                or "RemoteProtocolError" in ename
                                or "incomplete" in str(e)[:200]
                                or "connection" in str(e)[:200].lower()
                                or "chunked" in str(e)[:200].lower())
                if not rate_limited:
                    raise
                # Last attempt: back off and retry once more instead of
                # raising — the portal's long-generations can exceed any
                # fixed timeout during capacity waves. We only give up after
                # the FULL budget (CAPACITY_ATTEMPTS) is consumed.
                if attempt == self._CAPACITY_ATTEMPTS - 1:
                    raise RuntimeError(
                        "nous: capacity/timeout budget exhausted after "
                        f"{self._CAPACITY_ATTEMPTS} attempts"
                    ) from e
                delay = min(self._CAPACITY_CEIL_S,
                            self._CAPACITY_BASE_S * (2 ** attempt))
                delay = random.uniform(delay * 0.5, delay)
                reason = "429 capacity" if (status == 429 or "429" in ename) else "upstream timeout"
                print(f"[nous] {reason}, backing off {delay:.0f}s "
                      f"(attempt {attempt + 1}/{self._CAPACITY_ATTEMPTS})",
                      flush=True)
                _time.sleep(delay)
        else:
            raise RuntimeError("nous: invoke loop exhausted without result")
        for _ in range(self._NOUS_EMPTY_RETRIES):
            if _has_payload(result):
                break
            result = _stream_invoke(input, config, **kwargs)
        if not _has_payload(result):
            raise RuntimeError(
                "nous: model returned an empty response after "
                f"{self._NOUS_EMPTY_RETRIES + 1} attempts"
            )
        return result


# Single source of truth for the OpenAI-compatible provider family. Dual-region
# providers (qwen/glm/minimax) keep separate endpoints because international and
# China accounts cannot share credentials (#758).
OPENAI_COMPATIBLE_PROVIDERS: dict[str, ProviderSpec] = {
    "openai":     ProviderSpec(use_responses_api=True),
    "xai":        ProviderSpec(base_url="https://api.x.ai/v1"),
    "deepseek":   ProviderSpec(base_url="https://api.deepseek.com", chat_class=DeepSeekChatOpenAI),
    "qwen":       ProviderSpec(base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    "qwen-cn":    ProviderSpec(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "glm":        ProviderSpec(base_url="https://api.z.ai/api/paas/v4/"),
    "glm-cn":     ProviderSpec(base_url="https://open.bigmodel.cn/api/paas/v4/"),
    "minimax":    ProviderSpec(base_url="https://api.minimax.io/v1", chat_class=MinimaxChatOpenAI),
    "minimax-cn": ProviderSpec(base_url="https://api.minimaxi.com/v1", chat_class=MinimaxChatOpenAI),
    "openrouter": ProviderSpec(base_url="https://openrouter.ai/api/v1"),
    "nous":       ProviderSpec(base_url="https://inference-api.nousresearch.com/v1",
                               chat_class=NousChatOpenAI),
    "mistral":    ProviderSpec(base_url="https://api.mistral.ai/v1"),
    "kimi":       ProviderSpec(base_url="https://api.moonshot.ai/v1"),
    "groq":       ProviderSpec(base_url="https://api.groq.com/openai/v1"),
    "nvidia":     ProviderSpec(base_url="https://integrate.api.nvidia.com/v1"),
    "ollama":     ProviderSpec(base_url="http://localhost:11434/v1", base_url_env="OLLAMA_BASE_URL",
                               key_optional=True, placeholder_key="ollama"),
    # Generic endpoint: user supplies base_url; key optional (keyless local).
    "openai_compatible": ProviderSpec(
        require_base_url=True, key_optional=True, chat_class=LocalCompatibleChatOpenAI
    ),
}


def is_openai_compatible(provider: str) -> bool:
    """Whether ``provider`` is served by the OpenAI-compatible registry."""
    return provider.lower() in OPENAI_COMPATIBLE_PROVIDERS


def _is_native_openai_base_url(base_url: str | None) -> bool:
    """True when ``base_url`` is unset or points at api.openai.com.

    The Responses API (/v1/responses) only exists on native OpenAI. A custom
    base_url on the ``openai`` provider (a proxy, gateway, or local server)
    speaks only Chat Completions, so the Responses API must stay off there even
    though the provider spec enables it (#1024).
    """
    if not base_url:
        return True
    if "://" not in base_url:
        base_url = "https://" + base_url
    host = urlparse(base_url).hostname or ""
    return host == "api.openai.com" or host.endswith(".openai.com")


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI, Ollama, OpenRouter, and xAI providers.

    For native OpenAI models, uses the Responses API (/v1/responses) which
    supports reasoning_effort with function tools across all model families
    (GPT-4.1, GPT-5). Third-party compatible providers (xAI, OpenRouter,
    Ollama) use standard Chat Completions.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        provider: str = "openai",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """Return a configured ChatOpenAI instance, driven by the provider registry."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}
        spec = OPENAI_COMPATIBLE_PROVIDERS.get(self.provider)
        chat_cls = NormalizedChatOpenAI

        if spec is not None:
            chat_cls = spec.chat_class

            # base_url precedence: explicit client base_url (carries the config /
            # TRADINGAGENTS_LLM_BACKEND_URL value) > provider env override (e.g.
            # OLLAMA_BASE_URL) > provider default. None means use the SDK default.
            env_base_url = os.environ.get(spec.base_url_env) if spec.base_url_env else None
            base_url = self.base_url or env_base_url or spec.base_url
            if spec.require_base_url and not base_url:
                raise ValueError(
                    f"Provider '{self.provider}' requires a base_url. Set it via "
                    "backend_url / TRADINGAGENTS_LLM_BACKEND_URL to your endpoint, "
                    "e.g. http://localhost:8000/v1 (vLLM) or http://localhost:1234/v1 "
                    "(LM Studio)."
                )
            if base_url:
                llm_kwargs["base_url"] = base_url

            # API key: required unless key_optional; keyless local servers get a
            # placeholder. The env-var name is the single source in api_key_env.
            api_key_env = get_api_key_env(self.provider)
            api_key = os.environ.get(api_key_env) if api_key_env else None
            if api_key:
                llm_kwargs["api_key"] = api_key
            elif spec.key_optional:
                llm_kwargs["api_key"] = spec.placeholder_key
            elif api_key_env:
                raise ValueError(
                    f"API key for provider '{self.provider}' is not set. "
                    f"Please set the {api_key_env} environment variable "
                    f"(e.g. add {api_key_env}=your_key to your .env file)."
                )

            # The Responses API only exists on native OpenAI; if the user points
            # the openai provider at a custom base_url (proxy/gateway/local), it
            # only speaks Chat Completions, so keep Responses off there (#1024).
            if spec.use_responses_api and _is_native_openai_base_url(base_url):
                llm_kwargs["use_responses_api"] = True
        elif self.base_url:
            llm_kwargs["base_url"] = self.base_url

        # Forward user-provided kwargs
        for key in _PASSTHROUGH_KWARGS:
            if key not in self.kwargs:
                continue
            if key == "reasoning_effort" and self.provider == "openai" and not _supports_reasoning_effort(self.model):
                continue
            llm_kwargs[key] = self.kwargs[key]

        # The subclass (provider quirks) comes from the registry spec.
        return chat_cls(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for the provider."""
        return validate_model(self.provider, self.model)
