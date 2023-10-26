import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

from typing import List, Optional, Union
from unittest.mock import Mock

import pytest

from llm_lib import BaseLLMResponse, PromptTemplate
from llm_lib.openai import OpenAIChatCompletion
from llm_lib.openai.chat_completion import OpenAIChatModel


class TestOpenAIChatCompletion:
    def test_token_counter(self):
        test_messages = [
            {"role": "system", "content": "This is a system message"},
            {"role": "user", "content": "This is a user message 🧑🏾"},
        ]

        # check if test covers all models (increase if new models are added)
        assert len(OpenAIChatModel.__args__) == 11  #  type: ignore

        assert OpenAIChatCompletion.num_tokens_from_messages(test_messages, model="gpt-3.5-turbo") == 27
        assert OpenAIChatCompletion.num_tokens_from_messages(test_messages, model="gpt-3.5-turbo-0301") == 29
        assert OpenAIChatCompletion.num_tokens_from_messages(test_messages, model="gpt-3.5-turbo-0613") == 27
        assert OpenAIChatCompletion.num_tokens_from_messages(test_messages, model="gpt-3.5-turbo-16k") == 27
        assert OpenAIChatCompletion.num_tokens_from_messages(test_messages, model="gpt-3.5-turbo-16k-0613") == 27
        assert OpenAIChatCompletion.num_tokens_from_messages(test_messages, model="gpt-4") == 27
        assert OpenAIChatCompletion.num_tokens_from_messages(test_messages, model="gpt-4-0314") == 27
        assert OpenAIChatCompletion.num_tokens_from_messages(test_messages, model="gpt-4-0613") == 27
        assert OpenAIChatCompletion.num_tokens_from_messages(test_messages, model="gpt-4-32k") == 27
        assert OpenAIChatCompletion.num_tokens_from_messages(test_messages, model="gpt-4-32k-0314") == 27
        assert OpenAIChatCompletion.num_tokens_from_messages(test_messages, model="gpt-4-32k-0613") == 27

    # -

    @pytest.fixture
    def mock_openai_completion(self, mocker):
        async def async_mock(*args, **kwargs):
            return {"choices": [{"message": {"content": "TITLE: This is a test completion\nCOUNT: 09"}}]}

        mocker.patch("llm_lib.openai.chat_completion.openai.ChatCompletion.acreate", new=async_mock)

    @pytest.mark.asyncio
    async def test_mock_end_to_end(self, mock_openai_completion):
        class FullExamplePrompt(PromptTemplate):
            def system_prompt(self) -> str:
                return "This is a random system prompt"

            def user_prompt(self) -> str:
                return "This is a random user prompt"

            class Output(BaseLLMResponse):
                title: str
                count: int

        result = await OpenAIChatCompletion.acreate(
            model="gpt-3.5-turbo",
            prompt=FullExamplePrompt(),
            max_output_tokens=100,
        )

        assert isinstance(result, FullExamplePrompt.Output)
        assert result.title == "This is a test completion"
        assert result.count == 9
