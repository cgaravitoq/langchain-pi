from __future__ import annotations

import json
import os
from typing import Any, Callable, Iterator, Optional, Sequence, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr

from .claude_code_auth import ClaudeCodeAuth
from .claude_code_client import DEFAULT_ANTHROPIC_BASE_URL, ClaudeCodeClient
from .claude_code_conversions import (
    apply_claude_code_transforms,
    build_request_body,
    messages_to_anthropic,
    tool_to_anthropic,
)
from .claude_code_models import CC_VERSION, PROVIDER_ID


def apply_stop(text: str, stop: Optional[list[str]]) -> str:
    if not stop:
        return text
    idxs = [i for i in (text.find(s) for s in stop) if i >= 0]
    return text[: min(idxs)] if idxs else text


def to_tool_calls(raw: list[dict]) -> list[dict]:
    return [
        {
            "name": tc.get("name"),
            "args": tc.get("arguments") or {},
            "id": tc.get("id"),
            "type": "tool_call",
        }
        for tc in raw
    ]


def to_usage_metadata(usage: Optional[dict]) -> Optional[dict]:
    if not usage:
        return None
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": usage.get("total_tokens", input_tokens + output_tokens),
        "input_token_details": {
            "cache_read": usage.get("cache_read", 0),
            "cache_creation": usage.get("cache_write", 0),
        },
    }


def to_response_metadata(
    model: str, stop_reason: Optional[str], usage: Optional[dict]
) -> dict:
    metadata: dict[str, Any] = {
        "provider": PROVIDER_ID,
        "model": model,
        "stopReason": stop_reason,
    }
    if usage:
        metadata["usage"] = usage
    return metadata


class ChatClaudeCode(BaseChatModel):
    """LangChain chat model for Claude via the Claude Code subscription (OAuth)."""

    model: str
    reasoning: str = "medium"
    system: Optional[str] = None
    creds_path: Optional[str] = None
    base_url: str = DEFAULT_ANTHROPIC_BASE_URL
    max_tokens: Optional[int] = None
    entrypoint: Optional[str] = None
    long_context: bool = False
    thinking_budgets: Optional[dict] = None

    _client: Optional[ClaudeCodeClient] = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return PROVIDER_ID

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "provider": PROVIDER_ID,
            "model": self.model,
            "reasoning": self.reasoning,
        }

    def _entrypoint(self) -> str:
        return self.entrypoint or os.environ.get("CLAUDE_CODE_ENTRYPOINT", "sdk-cli")

    def _get_client(self) -> ClaudeCodeClient:
        if self._client is None:
            self._client = ClaudeCodeClient(
                ClaudeCodeAuth(self.creds_path),
                base_url=self.base_url,
                entrypoint=self._entrypoint(),
                long_context=self.long_context,
            )
        return self._client

    def _build_body(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict:
        system_texts, anthropic_messages = messages_to_anthropic(messages)
        if self.system:
            system_texts = [self.system, *system_texts]
        body = build_request_body(
            self.model,
            system_texts,
            anthropic_messages,
            max_tokens=self.max_tokens,
            tools=kwargs.get("tools"),
            reasoning=self.reasoning,
            thinking_budgets=self.thinking_budgets,
            stop=stop,
        )
        version = os.environ.get("ANTHROPIC_CLI_VERSION", CC_VERSION)
        return apply_claude_code_transforms(
            body, version=version, entrypoint=self._entrypoint()
        )

    @staticmethod
    def _tool_names(kwargs: dict) -> list[str]:
        return [t["name"] for t in (kwargs.get("tools") or []) if t.get("name")]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        text_parts: list[str] = []
        raw_tool_calls: list[dict] = []
        usage: Optional[dict] = None
        stop_reason: Optional[str] = None

        body = self._build_body(messages, stop=stop, **kwargs)
        for event in self._get_client().stream(
            body, model=self.model, tool_names=self._tool_names(kwargs)
        ):
            kind = event["type"]
            if kind == "text_delta":
                text_parts.append(event["delta"])
            elif kind == "tool_call":
                raw_tool_calls.append(event["tool_call"])
            elif kind == "done":
                usage = event.get("usage")
                stop_reason = event.get("stop_reason")

        text = apply_stop("".join(text_parts), stop)
        message = AIMessage(
            content=text,
            tool_calls=to_tool_calls(raw_tool_calls),
            usage_metadata=to_usage_metadata(usage),
            response_metadata=to_response_metadata(self.model, stop_reason, usage),
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        body = self._build_body(messages, stop=stop, **kwargs)
        for event in self._get_client().stream(
            body, model=self.model, tool_names=self._tool_names(kwargs)
        ):
            kind = event["type"]
            if kind == "text_delta":
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=event["delta"])
                )
                if run_manager:
                    run_manager.on_llm_new_token(event["delta"], chunk=chunk)
                yield chunk
            elif kind == "tool_call":
                tc = event["tool_call"]
                args = tc.get("arguments")
                if not isinstance(args, str):
                    args = json.dumps(args or {})
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {
                                "name": tc.get("name"),
                                "args": args,
                                "id": tc.get("id"),
                                "index": event.get("index", 0),
                                "type": "tool_call_chunk",
                            }
                        ],
                    )
                )
            elif kind == "done":
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        usage_metadata=to_usage_metadata(event.get("usage")),
                        response_metadata=to_response_metadata(
                            self.model, event.get("stop_reason"), event.get("usage")
                        ),
                    )
                )

    def bind_tools(
        self,
        tools: Sequence[Union[dict, type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[str, dict, bool]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        anthropic_tools = [tool_to_anthropic(tool) for tool in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self.bind(tools=anthropic_tools, **kwargs)
