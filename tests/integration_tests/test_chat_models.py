from typing import Type

from langchain_pi import ChatPi
from langchain_tests.integration_tests import ChatModelIntegrationTests


class TestChatPiIntegration(ChatModelIntegrationTests):
    """Hits a real Pi provider via the Node sidecar. Requires Node + the pi
    packages installed and a provider authenticated in ~/.pi."""

    @property
    def chat_model_class(self) -> Type[ChatPi]:
        return ChatPi

    @property
    def chat_model_params(self) -> dict:
        return {"provider": "openai-codex", "model": "gpt-5.3-codex-spark"}

    @property
    def has_tool_calling(self) -> bool:
        return True
