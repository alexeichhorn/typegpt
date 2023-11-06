import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

from typing import List, Optional, Union
from unittest.mock import Mock

import pytest

from gpt_condom import BaseLLMResponse, LLMArrayOutput, PromptTemplate
from gpt_condom.exceptions import LLMTokenLimitExceeded
from gpt_condom.openai import OpenAIChatCompletion
from gpt_condom.openai.chat_completion import OpenAIChatModel, openai


class TestOpenAIChatCompletion:
    def test_token_counter(self):
        test_messages = [
            {"role": "system", "content": "This is a system message"},
            {"role": "user", "content": "This is a user message 🧑🏾"},
        ]

        # check if test covers all models (increase if new models are added)
        assert len(OpenAIChatModel.__args__) == 14  #  type: ignore

        assert OpenAIChatCompletion.num_tokens_from_messages(test_messages, model="gpt-3.5-turbo") == 27
        assert OpenAIChatCompletion.num_tokens_from_messages(test_messages, model="gpt-3.5-turbo-0301") == 29
        assert OpenAIChatCompletion.num_tokens_from_messages(test_messages, model="gpt-3.5-turbo-0613") == 27
        assert OpenAIChatCompletion.num_tokens_from_messages(test_messages, model="gpt-3.5-turbo-1106") == 27
        assert OpenAIChatCompletion.num_tokens_from_messages(test_messages, model="gpt-3.5-turbo-16k") == 27
        assert OpenAIChatCompletion.num_tokens_from_messages(test_messages, model="gpt-3.5-turbo-16k-0613") == 27
        assert OpenAIChatCompletion.num_tokens_from_messages(test_messages, model="gpt-4") == 27
        assert OpenAIChatCompletion.num_tokens_from_messages(test_messages, model="gpt-4-0314") == 27
        assert OpenAIChatCompletion.num_tokens_from_messages(test_messages, model="gpt-4-0613") == 27
        assert OpenAIChatCompletion.num_tokens_from_messages(test_messages, model="gpt-4-32k") == 27
        assert OpenAIChatCompletion.num_tokens_from_messages(test_messages, model="gpt-4-32k-0314") == 27
        assert OpenAIChatCompletion.num_tokens_from_messages(test_messages, model="gpt-4-32k-0613") == 27
        assert OpenAIChatCompletion.num_tokens_from_messages(test_messages, model="gpt-4-1106-preview") == 27
        assert OpenAIChatCompletion.num_tokens_from_messages(test_messages, model="gpt-4-vision-preview") == 27

    # -

    @pytest.fixture
    def mock_openai_completion(self, mocker):
        async def async_mock(*args, **kwargs):
            return {
                "id": "test",
                "model": "gpt-3.5-turbo",
                "object": "x",
                "created": 123,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 1,
                        "message": {"role": "assistant", "content": "TITLE: This is a test completion\nCOUNT: 09"},
                    }
                ],
            }

        mocker.patch("gpt_condom.openai.chat_completion.openai.ChatCompletion.acreate", new=async_mock)

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

        result = await OpenAIChatCompletion.generate_output(
            model="gpt-3.5-turbo",
            prompt=FullExamplePrompt(),
            output_type=FullExamplePrompt.Output,
            max_output_tokens=100,
        )

        assert isinstance(result, FullExamplePrompt.Output)
        assert result.title == "This is a test completion"
        assert result.count == 9

        result_base = await OpenAIChatCompletion.generate_output(
            model="gpt-3.5-turbo",
            prompt=FullExamplePrompt(),
            max_output_tokens=100,
        )

        assert isinstance(result, FullExamplePrompt.Output)
        assert result.title == "This is a test completion"
        assert result.count == 9

        # -

        class AlternativeOutput(BaseLLMResponse):
            count: int

        result_alt = await OpenAIChatCompletion.generate_output(
            model="gpt-3.5-turbo",
            prompt=FullExamplePrompt(),
            output_type=AlternativeOutput,
            max_output_tokens=100,
        )

        assert isinstance(result_alt, AlternativeOutput)
        assert result_alt.count == 9
        assert not hasattr(result_alt, "title")

    @pytest.fixture
    def mock_openai_retry_completion(self, mocker):
        call_count = 0

        async def async_mock(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 6:
                content_res = "TITLE: Some title n\nCOUNT: 42\nITEM 1: abc"
            elif call_count == 5:
                content_res = "TITLE: Some title n\nCOUNT: 42"
            elif call_count == 4:
                content_res = "Random stuff"  # no content
            elif call_count == 3:
                content_res = "TITLE: Some title\nCOUNT: 99999\nITEM 1: abc\nITEM 2: def\nITEM 3: ghi"  # too many items
            elif call_count == 2:
                content_res = "TITLE: Some title\nCOUNT: random string\nITEM 1: abc"  # wrong type
            else:
                content_res = "TITLE: Only title\nITEM 1: abc"

            return {
                "id": "test",
                "model": "gpt-3.5-turbo",
                "object": "x",
                "created": 123,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 1,
                        "message": {"role": "assistant", "content": content_res},
                    }
                ],
            }

        mocker.patch("gpt_condom.openai.chat_completion.openai.ChatCompletion.acreate", new=async_mock)

    @pytest.mark.asyncio
    async def test_mock_end_to_end_parse_retry(self, mock_openai_retry_completion):
        class FullExamplePrompt(PromptTemplate):
            def system_prompt(self) -> str:
                return "This is a random system prompt"

            def user_prompt(self) -> str:
                return "This is a random user prompt"

            class Output(BaseLLMResponse):
                title: str
                items: list[str] = LLMArrayOutput((1, 2), instruction=lambda _: "Put the items here")
                count: int

        result = await OpenAIChatCompletion.generate_output(
            model="gpt-3.5-turbo", prompt=FullExamplePrompt(), max_output_tokens=100, retry_on_parse_error=5
        )

        assert isinstance(result, FullExamplePrompt.Output)
        assert result.title == "Some title n"
        assert result.items == ["abc"]
        assert result.count == 42

    @pytest.mark.asyncio
    async def test_mock_reduce_prompt(self, mock_openai_completion):
        class NonAutomaticReducingPrompt(PromptTemplate):
            def __init__(self, number: int):
                self.lines = [f"This is line {i}" for i in range(number)]

            def system_prompt(self) -> str:
                return "This is a random system prompt"

            def user_prompt(self) -> str:
                return "My lines:\n\n" + "\n".join(self.lines)

            class Output(BaseLLMResponse):
                lines: list[str]

        non_reducing_prompt_100 = NonAutomaticReducingPrompt(100)

        result = await OpenAIChatCompletion.generate_output(
            model="gpt-3.5-turbo",
            prompt=non_reducing_prompt_100,
            max_output_tokens=100,
        )

        non_reducing_prompt_1000 = NonAutomaticReducingPrompt(1000)

        with pytest.raises(LLMTokenLimitExceeded):
            result = await OpenAIChatCompletion.generate_output(
                model="gpt-3.5-turbo",
                prompt=non_reducing_prompt_1000,
                max_output_tokens=100,
            )

        class ReducingTestPrompt(PromptTemplate):
            def __init__(self, number: int):
                self.lines = [f"This is line {i}" for i in range(number)]

            def system_prompt(self) -> str:
                return "This is a random system prompt"

            def user_prompt(self) -> str:
                return "My lines:\n\n" + "\n".join(self.lines)

            class Output(BaseLLMResponse):
                lines: list[str]

            def reduce_if_possible(self) -> bool:
                if len(self.lines) > 10:
                    # remove last 10 lines
                    self.lines = self.lines[:-10]
                    return True
                return False

        reducing_prompt_100 = ReducingTestPrompt(100)

        result = await OpenAIChatCompletion.generate_output(
            model="gpt-3.5-turbo",
            prompt=reducing_prompt_100,
            max_output_tokens=100,
        )

        assert len(reducing_prompt_100.lines) == 100

        reducing_prompt_1000 = ReducingTestPrompt(1000)

        result = await OpenAIChatCompletion.generate_output(
            model="gpt-3.5-turbo",
            prompt=reducing_prompt_1000,
            max_output_tokens=100,
        )
