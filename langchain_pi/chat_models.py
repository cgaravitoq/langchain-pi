"""ChatPi: a LangChain BaseChatModel adapter for Pi (@earendil-works/pi-ai),
driven through a Node sidecar. 1:1 mirror of langchain-pi-ts ChatPi."""

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

from .pi_conversions import (
    apply_stop,
    messages_to_pi_payload,
    to_response_metadata,
    to_tool_calls,
    to_usage_metadata,
    tool_to_pi,
)
from .sidecar import PiSidecar

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def _error_message(event: dict) -> str:
    return (event.get("error") or {}).get("errorMessage") or "pi-ai request failed"


class ChatPi(BaseChatModel):
    """Chat model backed by Pi. Resolves provider/model/credentials through pi's
    own ModelRegistry/AuthStorage (inside the Node sidecar), so any provider
    authenticated in ``~/.pi`` works with no extra configuration."""

    provider: str
    model: str
    reasoning: str = "low"
    system: Optional[str] = DEFAULT_SYSTEM_PROMPT

    node_path: str = "node"
    """Executable used to run the sidecar."""
    sidecar_cwd: Optional[str] = None
    """Working directory for the sidecar process."""
    node_modules_dir: Optional[str] = None
    """A node_modules directory containing @earendil-works/pi-ai, for when the
    package is not installed alongside one (e.g. pip into site-packages)."""

    _sidecar: Optional[PiSidecar] = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "pi"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "reasoning": self.reasoning,
        }

    def _get_sidecar(self) -> PiSidecar:
        if self._sidecar is None:
            self._sidecar = PiSidecar(
                node_path=self.node_path,
                cwd=self.sidecar_cwd,
                node_modules_dir=self.node_modules_dir,
            )
        return self._sidecar

    def _build_request(self, messages: list[BaseMessage], **kwargs: Any) -> dict:
        system_prompt, history = messages_to_pi_payload(messages, self.system)
        request: dict[str, Any] = {
            "provider": self.provider,
            "modelId": self.model,
            "reasoning": self.reasoning,
            "messages": history,
        }
        if system_prompt:
            request["systemPrompt"] = system_prompt
        tools = kwargs.get("tools")
        if tools:
            request["tools"] = tools
        return request

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

        for event in self._get_sidecar().stream(self._build_request(messages, **kwargs)):
            kind = event.get("type")
            if kind == "text_delta":
                text_parts.append(event["delta"])
            elif kind == "toolcall_end":
                raw_tool_calls.append(event["toolCall"])
            elif kind == "done":
                usage = event.get("usage")
                stop_reason = event.get("stopReason")
            elif kind == "error":
                raise RuntimeError(_error_message(event))

        text = apply_stop("".join(text_parts), stop)
        message = AIMessage(
            content=text,
            tool_calls=to_tool_calls(raw_tool_calls),
            usage_metadata=to_usage_metadata(usage),
            response_metadata=to_response_metadata(
                self.provider, self.model, stop_reason, usage
            ),
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for event in self._get_sidecar().stream(self._build_request(messages, **kwargs)):
            kind = event.get("type")
            if kind == "text_delta":
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=event["delta"])
                )
                if run_manager:
                    run_manager.on_llm_new_token(event["delta"], chunk=chunk)
                yield chunk
            elif kind == "toolcall_end":
                tc = event["toolCall"]
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {
                                "name": tc.get("name"),
                                "args": json.dumps(tc.get("arguments") or {}),
                                "id": tc.get("id"),
                                "index": event.get("contentIndex", 0),
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
                            self.provider,
                            self.model,
                            event.get("stopReason"),
                            event.get("usage"),
                        ),
                    )
                )
            elif kind == "error":
                raise RuntimeError(_error_message(event))

    def bind_tools(
        self,
        tools: Sequence[Union[dict, type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[str, dict, bool]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        pi_tools = [tool_to_pi(tool) for tool in tools]
        return self.bind(tools=pi_tools, **kwargs)
