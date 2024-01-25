import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

from typing import List, Optional, Union
from unittest.mock import Mock

import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from typegpt import BaseLLMResponse, LLMArrayOutput, LLMOutput, PromptTemplate
from typegpt.exceptions import LLMTokenLimitExceeded
from typegpt.openai import AsyncTypeAzureOpenAI, AsyncTypeOpenAI, OpenAIChatModel, TypeAzureOpenAI, TypeOpenAI


class TestOpenAIChatCompletion:
    def test_token_counter(self):
        test_messages = [
            {"role": "system", "content": "This is a system message"},
            {"role": "user", "content": "This is a user message 🧑🏾"},
        ]

        # check if test covers all models (increase if new models are added)
        assert len(OpenAIChatModel.__args__) == 16  #  type: ignore

        client = AsyncTypeOpenAI(api_key="mock")

        assert client.chat.completions.num_tokens_from_messages(test_messages, model="gpt-3.5-turbo") == 27
        assert client.chat.completions.num_tokens_from_messages(test_messages, model="gpt-3.5-turbo-0301") == 29
        assert client.chat.completions.num_tokens_from_messages(test_messages, model="gpt-3.5-turbo-0613") == 27
        assert client.chat.completions.num_tokens_from_messages(test_messages, model="gpt-3.5-turbo-1106") == 27
        assert client.chat.completions.num_tokens_from_messages(test_messages, model="gpt-3.5-turbo-16k") == 27
        assert client.chat.completions.num_tokens_from_messages(test_messages, model="gpt-3.5-turbo-16k-0613") == 27
        assert client.chat.completions.num_tokens_from_messages(test_messages, model="gpt-4") == 27
        assert client.chat.completions.num_tokens_from_messages(test_messages, model="gpt-4-0314") == 27
        assert client.chat.completions.num_tokens_from_messages(test_messages, model="gpt-4-0613") == 27
        assert client.chat.completions.num_tokens_from_messages(test_messages, model="gpt-4-32k") == 27
        assert client.chat.completions.num_tokens_from_messages(test_messages, model="gpt-4-32k-0314") == 27
        assert client.chat.completions.num_tokens_from_messages(test_messages, model="gpt-4-32k-0613") == 27
        assert client.chat.completions.num_tokens_from_messages(test_messages, model="gpt-4-turbo-preview") == 27
        assert client.chat.completions.num_tokens_from_messages(test_messages, model="gpt-4-1106-preview") == 27
        assert client.chat.completions.num_tokens_from_messages(test_messages, model="gpt-4-0125-preview") == 27
        assert client.chat.completions.num_tokens_from_messages(test_messages, model="gpt-4-vision-preview") == 27

    def test_max_token_counter(self):
        # check if test covers all models (increase if new models are added)
        assert len(OpenAIChatModel.__args__) == 16  #  type: ignore

        client = AsyncTypeOpenAI(api_key="mock")

        assert client.chat.completions.max_tokens_of_model("gpt-3.5-turbo") == 4096
        assert client.chat.completions.max_tokens_of_model("gpt-3.5-turbo-0301") == 4096
        assert client.chat.completions.max_tokens_of_model("gpt-3.5-turbo-0613") == 4096
        assert client.chat.completions.max_tokens_of_model("gpt-3.5-turbo-1106") == 16384
        assert client.chat.completions.max_tokens_of_model("gpt-3.5-turbo-16k") == 16384
        assert client.chat.completions.max_tokens_of_model("gpt-3.5-turbo-16k-0613") == 16384
        assert client.chat.completions.max_tokens_of_model("gpt-4") == 8192
        assert client.chat.completions.max_tokens_of_model("gpt-4-0314") == 8192
        assert client.chat.completions.max_tokens_of_model("gpt-4-0613") == 8192
        assert client.chat.completions.max_tokens_of_model("gpt-4-32k") == 32768
        assert client.chat.completions.max_tokens_of_model("gpt-4-32k-0314") == 32768
        assert client.chat.completions.max_tokens_of_model("gpt-4-32k-0613") == 32768
        assert client.chat.completions.max_tokens_of_model("gpt-4-turbo-preview") == 128_000
        assert client.chat.completions.max_tokens_of_model("gpt-4-1106-preview") == 128_000
        assert client.chat.completions.max_tokens_of_model("gpt-4-0125-preview") == 128_000
        assert client.chat.completions.max_tokens_of_model("gpt-4-vision-preview") == 128_000

    # -

    @pytest.fixture
    def mock_openai_completion(self, mocker):
        async def async_mock(*args, **kwargs):
            return ChatCompletion(
                id="test",
                model="gpt-3.5-turbo",
                object="chat.completion",
                created=123,
                choices=[
                    Choice(
                        finish_reason="stop",
                        index=1,
                        message=ChatCompletionMessage(role="assistant", content="TITLE: This is a test completion\nCOUNT: 09"),
                    )
                ],
            )

        mocker.patch("typegpt.openai._async.chat_completion.AsyncTypeChatCompletion.create", new=async_mock)

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

        client = AsyncTypeOpenAI(api_key="mock")

        result = await client.chat.completions.generate_output(
            model="gpt-3.5-turbo",
            prompt=FullExamplePrompt(),
            output_type=FullExamplePrompt.Output,
            max_output_tokens=100,
        )

        assert isinstance(result, FullExamplePrompt.Output)
        assert result.title == "This is a test completion"
        assert result.count == 9

        result_base = await client.chat.completions.generate_output(
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

        result_alt = await client.chat.completions.generate_output(
            model="gpt-3.5-turbo",
            prompt=FullExamplePrompt(),
            output_type=AlternativeOutput,
            max_output_tokens=100,
        )

        assert isinstance(result_alt, AlternativeOutput)
        assert result_alt.count == 9
        assert not hasattr(result_alt, "title")

    @pytest.mark.asyncio
    async def test_mock_end_to_end_azure(Self, mock_openai_completion):
        class FullExamplePrompt(PromptTemplate):
            def system_prompt(self) -> str:
                return "This is a random system prompt"

            def user_prompt(self) -> str:
                return "This is a random user prompt"

            class Output(BaseLLMResponse):
                title: str
                count: int

        client = AsyncTypeAzureOpenAI(api_key="mock", azure_endpoint="mock", api_version="mock")

        result = await client.chat.completions.generate_output(
            model="gpt-3.5-turbo",
            prompt=FullExamplePrompt(),
            output_type=FullExamplePrompt.Output,
            max_output_tokens=100,
        )

        assert isinstance(result, FullExamplePrompt.Output)
        assert result.title == "This is a test completion"
        assert result.count == 9

        result_base = await client.chat.completions.generate_output(
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

        result_alt = await client.chat.completions.generate_output(
            model="gpt-3.5-turbo",
            prompt=FullExamplePrompt(),
            output_type=AlternativeOutput,
            max_output_tokens=100,
        )

        assert isinstance(result_alt, AlternativeOutput)
        assert result_alt.count == 9
        assert not hasattr(result_alt, "title")

    @pytest.fixture
    def mock_openai_completion_sync(self, mocker):
        def sync_mock(*args, **kwargs):
            return ChatCompletion(
                id="test",
                model="gpt-3.5-turbo",
                object="chat.completion",
                created=123,
                choices=[
                    Choice(
                        finish_reason="stop",
                        index=1,
                        message=ChatCompletionMessage(role="assistant", content="TITLE: This is a test completion\nCOUNT: 09"),
                    )
                ],
            )

        mocker.patch("typegpt.openai._sync.chat_completion.TypeChatCompletion.create", new=sync_mock)

    def test_mock_end_to_end_sync(self, mock_openai_completion_sync):
        class FullExamplePrompt(PromptTemplate):
            def system_prompt(self) -> str:
                return "This is a random system prompt"

            def user_prompt(self) -> str:
                return "This is a random user prompt"

            class Output(BaseLLMResponse):
                title: str
                count: int

        client = TypeOpenAI(api_key="mock")

        result = client.chat.completions.generate_output(
            model="gpt-3.5-turbo",
            prompt=FullExamplePrompt(),
            output_type=FullExamplePrompt.Output,
            max_output_tokens=100,
        )

        assert isinstance(result, FullExamplePrompt.Output)
        assert result.title == "This is a test completion"
        assert result.count == 9

        result_base = client.chat.completions.generate_output(
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

        result_alt = client.chat.completions.generate_output(
            model="gpt-3.5-turbo",
            prompt=FullExamplePrompt(),
            output_type=AlternativeOutput,
            max_output_tokens=100,
        )

        assert isinstance(result_alt, AlternativeOutput)
        assert result_alt.count == 9
        assert not hasattr(result_alt, "title")

    def test_mock_end_to_end_sync_azure(self, mock_openai_completion_sync):
        class FullExamplePrompt(PromptTemplate):
            def system_prompt(self) -> str:
                return "This is a random system prompt"

            def user_prompt(self) -> str:
                return "This is a random user prompt"

            class Output(BaseLLMResponse):
                title: str
                count: int

        client = TypeAzureOpenAI(api_key="mock", azure_endpoint="mock", api_version="mock")

        result = client.chat.completions.generate_output(
            model="gpt-3.5-turbo",
            prompt=FullExamplePrompt(),
            output_type=FullExamplePrompt.Output,
            max_output_tokens=100,
        )

        assert isinstance(result, FullExamplePrompt.Output)
        assert result.title == "This is a test completion"
        assert result.count == 9

        result_base = client.chat.completions.generate_output(
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

        result_alt = client.chat.completions.generate_output(
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

            return ChatCompletion(
                id="test",
                model="gpt-3.5-turbo",
                object="chat.completion",
                created=123,
                choices=[
                    Choice(
                        finish_reason="stop",
                        index=1,
                        message=ChatCompletionMessage(role="assistant", content=content_res),
                    )
                ],
            )

        mocker.patch("typegpt.openai._async.chat_completion.AsyncTypeChatCompletion.create", new=async_mock)

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

        client = AsyncTypeOpenAI(api_key="mock")

        result = await client.chat.completions.generate_output(
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

        client = AsyncTypeOpenAI(api_key="mock")

        result = await client.chat.completions.generate_output(
            model="gpt-3.5-turbo",
            prompt=non_reducing_prompt_100,
            max_output_tokens=100,
        )

        non_reducing_prompt_1000 = NonAutomaticReducingPrompt(1000)

        with pytest.raises(LLMTokenLimitExceeded):
            result = await client.chat.completions.generate_output(
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

        result = await client.chat.completions.generate_output(
            model="gpt-3.5-turbo",
            prompt=reducing_prompt_100,
            max_output_tokens=100,
        )

        assert len(reducing_prompt_100.lines) == 100

        reducing_prompt_1000 = ReducingTestPrompt(1000)

        result = await client.chat.completions.generate_output(
            model="gpt-3.5-turbo",
            prompt=reducing_prompt_1000,
            max_output_tokens=100,
        )

    # -

    def test_dynamic_output_type(self, mock_openai_completion_sync):
        class FullExamplePrompt(PromptTemplate):
            def __init__(self, name: str):
                self.name = name

            def system_prompt(self) -> str:
                return "This is a random system prompt"

            def user_prompt(self) -> str:
                return "This is a random user prompt"

            @property
            def Output(self):
                class Output(BaseLLMResponse):
                    title: str = LLMOutput(f"The title of {self.name}")
                    count: int

                return Output

        client = TypeOpenAI(api_key="mock")

        prompt = FullExamplePrompt("test")

        result = client.chat.completions.generate_output(
            model="gpt-3.5-turbo",
            prompt=prompt,
            output_type=prompt.Output,
            max_output_tokens=100,
        )

        assert result.title == "This is a test completion"
        assert result.count == 9
