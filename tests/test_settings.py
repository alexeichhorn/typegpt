import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

import pytest

from typegpt import BaseLLMResponse, LLMArrayElementOutput, LLMArrayOutput, LLMOutput, PromptSettings, PromptTemplate


class TestSettings:
    class InjectableSimplePrompt(PromptTemplate):
        class Output(BaseLLMResponse):
            title: str

        def __init__(self, settings: PromptSettings):
            self.settings = settings

        def system_prompt(self) -> str:
            return "Some system prompt"

        def user_prompt(self) -> str:
            return "Some user prompt"

    class NoDirectFormattingInstructionPrompt(PromptTemplate):
        class Output(BaseLLMResponse):
            title: str

        settings = PromptSettings(disable_formatting_instructions=True)

        def system_prompt(self) -> str:
            return "Some system prompt"

        def user_prompt(self) -> str:
            return "Some user prompt"

    def test_disable_formatting_instructions(self):
        enabled_settings = PromptSettings(disable_formatting_instructions=False)
        disabled_settings = PromptSettings(disable_formatting_instructions=True)

        enabled_messages = self.InjectableSimplePrompt(settings=enabled_settings).generate_messages(
            token_limit=1000, token_counter=lambda x: 0
        )
        disabled_messages = self.InjectableSimplePrompt(settings=disabled_settings).generate_messages(
            token_limit=1000, token_counter=lambda x: 0
        )
        direct_disabled_messages = self.NoDirectFormattingInstructionPrompt().generate_messages(token_limit=1000, token_counter=lambda x: 0)

        assert len(enabled_messages) == 2
        assert enabled_messages[0]["role"] == "system"
        assert enabled_messages[0]["content"].count("Always return the answer in the following format:") == 1

        assert disabled_messages == [
            {"role": "system", "content": "Some system prompt"},
            {"role": "user", "content": "Some user prompt"},
        ]

        assert direct_disabled_messages == disabled_messages
