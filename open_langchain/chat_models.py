from __future__ import annotations

import json
from typing import Any, Callable, Iterator, Optional, Sequence, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr

from .auth import CodexAuth
from .client import CodexClient
from .codex_conversions import (
    apply_stop,
    build_request_body,
    messages_to_responses,
    to_response_metadata,
    to_tool_calls,
    to_usage_metadata,
    tool_to_responses,
)
from .constants import DEFAULT_CODEX_BASE_URL, PROVIDER_ID
from .models import clamp_thinking_level

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


class ChatCodex(BaseChatModel):
    """LangChain chat model for OpenAI Codex (ChatGPT subscription)."""

    model: str
    reasoning: str = "low"
    system: Optional[str] = DEFAULT_SYSTEM_PROMPT
    auth_path: Optional[str] = None
    base_url: str = DEFAULT_CODEX_BASE_URL
    session_id: Optional[str] = None

    _client: Optional[CodexClient] = PrivateAttr(default=None)

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

    def _get_client(self) -> CodexClient:
        if self._client is None:
            self._client = CodexClient(
                CodexAuth(self.auth_path), base_url=self.base_url
            )
        return self._client

    def _build_body(self, messages: list[BaseMessage], **kwargs: Any) -> dict:
        instructions, input_items = messages_to_responses(messages, self.system)
        effort = clamp_thinking_level(self.model, self.reasoning)
        return build_request_body(
            model=self.model,
            instructions=instructions,
            input_items=input_items,
            tools=kwargs.get("tools"),
            tool_choice=kwargs.get("tool_choice"),
            reasoning_effort=effort,
            session_id=self.session_id,
        )

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

        body = self._build_body(messages, **kwargs)
        for event in self._get_client().stream(body, session_id=self.session_id):
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
        body = self._build_body(messages, **kwargs)
        for event in self._get_client().stream(body, session_id=self.session_id):
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
                                "id": tc.get("call_id") or tc.get("id"),
                                "index": 0,
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
        codex_tools = [tool_to_responses(tool) for tool in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self.bind(tools=codex_tools, **kwargs)
