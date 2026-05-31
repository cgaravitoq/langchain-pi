from typing import Type

from langchain_pi import ChatPi
from langchain_tests.unit_tests import ChatModelUnitTests


class TestChatPiUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatPi]:
        return ChatPi

    @property
    def chat_model_params(self) -> dict:
        return {"provider": "openai-codex", "model": "gpt-5.3-codex-spark"}

    @property
    def has_tool_calling(self) -> bool:
        return True
