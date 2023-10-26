from typing import Any, AsyncGenerator, Awaitable, Generic, Literal, TypeVar, overload

import openai
import tiktoken

from ..message_collection_builder import EncodedMessage, MessageCollectionFactory
from ..prompt_definition.prompt_template import PromptTemplate, _Output
from .views import (
    AzureChatModel,
    AzureConfig,
    ChatCompletionChunk,
    ChatCompletionResult,
    EncodedFunction,
    FunctionCallBehavior,
    OpenAIChatModel,
    OpenAIConfig,
)

# Prompt = TypeVar("Prompt", bound=PromptTemplate)


class OpenAIChatCompletion(openai.ChatCompletion):
    @overload
    @classmethod
    async def acreate(
        cls,
        model: OpenAIChatModel | AzureChatModel,
        messages: list[dict],
        stream: Literal[True],
        frequency_penalty: float | None = None,  # [-2, 2]
        function_call: FunctionCallBehavior | None = None,
        functions: list[EncodedFunction] = [],
        logit_bias: dict[int, float] | None = None,  # [-100, 100]
        stop: list[str] | None = None,
        max_tokens: int = 1000,
        n: int | None = None,
        presence_penalty: float | None = None,  # [-2, 2]
        temperature: float | None = None,
        top_p: float | None = None,
        user: str | None = None,
        config: OpenAIConfig | AzureConfig | None = None,
    ) -> tuple[AsyncGenerator[ChatCompletionChunk, None]]:
        ...

    @overload
    @classmethod
    async def acreate(
        cls,
        model: OpenAIChatModel | AzureChatModel,
        messages: list[dict],
        stream: Literal[False] = False,
        frequency_penalty: float | None = None,  # [-2, 2]
        function_call: FunctionCallBehavior | None = None,
        functions: list[EncodedFunction] = [],
        logit_bias: dict[int, float] | None = None,  # [-100, 100]
        stop: list[str] | None = None,
        max_tokens: int = 1000,
        n: int | None = None,
        presence_penalty: float | None = None,  # [-2, 2]
        temperature: float | None = None,
        top_p: float | None = None,
        user: str | None = None,
        config: OpenAIConfig | AzureConfig | None = None,
    ) -> ChatCompletionResult:
        ...

    @classmethod
    async def acreate(
        cls,
        model: OpenAIChatModel | AzureChatModel,
        messages: list[dict],
        stream: bool = False,
        frequency_penalty: float | None = None,  # [-2, 2]
        function_call: FunctionCallBehavior | None = None,
        functions: list[EncodedFunction] = [],
        logit_bias: dict[int, float] | None = None,  # [-100, 100]
        stop: list[str] | None = None,
        max_tokens: int = 1000,
        n: int | None = None,
        presence_penalty: float | None = None,  # [-2, 2]
        temperature: float | None = None,
        top_p: float | None = None,
        user: str | None = None,
        config: OpenAIConfig | AzureConfig | None = None,
    ) -> ChatCompletionResult | tuple[AsyncGenerator[ChatCompletionChunk, None]]:
        kwargs = {
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        if isinstance(model, AzureChatModel):
            kwargs["deployment_id"] = model.deployment_id
        else:
            kwargs["model"] = model

        if frequency_penalty is not None:
            kwargs["frequency_penalty"] = frequency_penalty

        if functions:
            kwargs["functions"] = functions

            if function_call:
                kwargs["function_call"] = function_call

        if logit_bias:
            kwargs["logit_bias"] = logit_bias

        if stop:
            kwargs["stop"] = stop

        if n is not None:
            kwargs["n"] = n

        if presence_penalty is not None:
            kwargs["presence_penalty"] = presence_penalty

        if temperature is not None:
            kwargs["temperature"] = temperature

        if top_p is not None:
            kwargs["top_p"] = top_p

        if user is not None:
            kwargs["user"] = user

        if config:
            kwargs = {**kwargs, **config.__dict__}

        if stream:

            async def _process_stream(raw_stream: AsyncGenerator[list[dict] | dict, None]) -> AsyncGenerator[ChatCompletionChunk, None]:
                async for chunk in raw_stream:
                    if isinstance(chunk, dict):
                        chunk = [chunk]

                    for c in chunk:
                        yield ChatCompletionChunk(**c)

            raw_stream: AsyncGenerator[list[dict] | dict, None] = await openai.ChatCompletion.acreate(**kwargs)  # type: ignore
            return (_process_stream(raw_stream),)  # tuple used to have correct static type

        else:
            raw_result: dict[str, Any] = await openai.ChatCompletion.acreate(**kwargs)  # type: ignore
            return ChatCompletionResult(**raw_result)

    @classmethod
    async def generate_completion(
        cls,
        model: OpenAIChatModel,
        messages: list[dict],
        frequency_penalty: float | None = None,  # [-2, 2]
        function_call: FunctionCallBehavior | None = None,
        functions: list[EncodedFunction] = [],
        logit_bias: dict[int, float] | None = None,  # [-100, 100]
        stop: list[str] | None = None,
        max_tokens: int = 1000,
        n: int | None = None,
        presence_penalty: float | None = None,  # [-2, 2]
        temperature: float | None = None,
        top_p: float | None = None,
        user: str | None = None,
    ) -> str:
        result = await cls.acreate(
            model=model,
            messages=messages,
            frequency_penalty=frequency_penalty,
            function_call=function_call,
            functions=functions,
            logit_bias=logit_bias,
            stop=stop,
            max_tokens=max_tokens,
            n=n,
            presence_penalty=presence_penalty,
            temperature=temperature,
            top_p=top_p,
            user=user,
        )

        return result.choices[0].message.content or ""

    @classmethod
    async def generate_output(
        cls,
        model: OpenAIChatModel,
        prompt: PromptTemplate[_Output],
        max_output_tokens: int,
        max_input_tokens: int | None = None,
        frequency_penalty: float | None = None,  # [-2, 2]
        n: int | None = None,
        presence_penalty: float | None = None,  # [-2, 2]
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> _Output:
        """
        Calls OpenAI Chat API, generates assistant response, and fits it into the output class
        """

        max_prompt_length = cls.max_tokens_of_model(model) - max_output_tokens

        if max_input_tokens:
            max_prompt_length = min(max_prompt_length, max_input_tokens)

        messages = prompt.generate_messages(
            token_limit=max_prompt_length, token_counter=lambda messages: cls.num_tokens_from_messages(messages, model=model)
        )

        completion = await cls.generate_completion(
            model=model,
            messages=messages,
            max_tokens=max_output_tokens,
            frequency_penalty=frequency_penalty,
            n=n,
            presence_penalty=presence_penalty,
            temperature=temperature,
            top_p=top_p,
        )

        return prompt.Output.parse_response(completion)

    # TODO: implement sync variant `create`

    @staticmethod
    def max_tokens_of_model(model: OpenAIChatModel) -> int:
        match model:
            case "gpt-3.5-turbo" | "gpt-3.5-turbo-0301" | "gpt-3.5-turbo-0613":
                return 4096
            case "gpt-3.5-turbo-16k" | "gpt-3.5-turbo-16k-0613":
                return 16384
            case "gpt-4" | "gpt-4-0314" | "gpt-4-0613":
                return 8192
            case "gpt-4-32k" | "gpt-4-32k-0314" | "gpt-4-32k-0613":
                return 32768

    # copied from OpenAI: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    @classmethod
    def num_tokens_from_messages(cls, messages: list[EncodedMessage], model: OpenAIChatModel | None = None) -> int:
        """Returns the number of tokens used by a list of messages."""
        if model is None:
            model = "gpt-3.5-turbo-0613"  # default model

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model == "gpt-3.5-turbo":
            return cls.num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif model == "gpt-3.5-turbo-16k":
            return cls.num_tokens_from_messages(messages, model="gpt-3.5-turbo-16k-0613")
        elif model == "gpt-4":
            return cls.num_tokens_from_messages(messages, model="gpt-4-0613")
        elif model == "gpt-4-32k":
            return cls.num_tokens_from_messages(messages, model="gpt-4-32k-0613")
        elif model in "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model in ("gpt-4-0314", "gpt-4-0613", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613", "gpt-4-32k-0314", "gpt-4-32k-0613"):
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
