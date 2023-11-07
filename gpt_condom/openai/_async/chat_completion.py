from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar, cast, overload

import tiktoken
from openai import BadRequestError, resources
from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    completion_create_params,
)

from ...base import BaseLLMResponse
from ...exceptions import LLMParseException
from ...message_collection_builder import EncodedMessage, MessageCollectionFactory
from ...prompt_definition.prompt_template import PromptTemplate
from ...utils.internal_types import _UseDefault, _UseDefaultType
from ..exceptions import AzureContentFilterException
from ..views import AzureChatModel, OpenAIChatModel

# Prompt = TypeVar("Prompt", bound=PromptTemplate)
_Output = TypeVar("_Output", bound=BaseLLMResponse)


class AsyncChatCompletionCondom(resources.chat.AsyncCompletions):
    # region - legacy code

    # @overload
    # @classmethod
    # async def acreate(
    #     cls,
    #     model: OpenAIChatModel | AzureChatModel,
    #     messages: list[dict],
    #     stream: Literal[True],
    #     frequency_penalty: float | None = None,  # [-2, 2]
    #     function_call: FunctionCallBehavior | None = None,
    #     functions: list[EncodedFunction] = [],
    #     logit_bias: dict[int, float] | None = None,  # [-100, 100]
    #     stop: list[str] | None = None,
    #     max_tokens: int = 1000,
    #     n: int | None = None,
    #     presence_penalty: float | None = None,  # [-2, 2]
    #     temperature: float | None = None,
    #     top_p: float | None = None,
    #     seed: int | None = None,
    #     response_format: ResponseFormat = ResponseFormat(type="text"),
    #     user: str | None = None,
    #     request_timeout: float | None = None,
    #     config: OpenAIConfig | AzureConfig | None = None,
    # ) -> tuple[AsyncGenerator[ChatCompletionChunk, None]]:
    #     ...

    # @overload
    # @classmethod
    # async def acreate(
    #     cls,
    #     model: OpenAIChatModel | AzureChatModel,
    #     messages: list[dict],
    #     stream: Literal[False] = False,
    #     frequency_penalty: float | None = None,  # [-2, 2]
    #     function_call: FunctionCallBehavior | None = None,
    #     functions: list[EncodedFunction] = [],
    #     logit_bias: dict[int, float] | None = None,  # [-100, 100]
    #     stop: list[str] | None = None,
    #     max_tokens: int = 1000,
    #     n: int | None = None,
    #     presence_penalty: float | None = None,  # [-2, 2]
    #     temperature: float | None = None,
    #     top_p: float | None = None,
    #     seed: int | None = None,
    #     response_format: ResponseFormat = ResponseFormat(type="text"),
    #     user: str | None = None,
    #     request_timeout: float | None = None,
    #     config: OpenAIConfig | AzureConfig | None = None,
    # ) -> ChatCompletionResult:
    #     ...

    # @classmethod
    # async def acreate(
    #     cls,
    #     model: OpenAIChatModel | AzureChatModel,
    #     messages: list[dict],
    #     stream: bool = False,
    #     frequency_penalty: float | None = None,  # [-2, 2]
    #     function_call: FunctionCallBehavior | None = None,
    #     functions: list[EncodedFunction] = [],
    #     logit_bias: dict[int, float] | None = None,  # [-100, 100]
    #     stop: list[str] | None = None,
    #     max_tokens: int = 1000,
    #     n: int | None = None,
    #     presence_penalty: float | None = None,  # [-2, 2]
    #     temperature: float | None = None,
    #     top_p: float | None = None,
    #     seed: int | None = None,
    #     response_format: ResponseFormat = ResponseFormat(type="text"),
    #     user: str | None = None,
    #     request_timeout: float | None = None,
    #     config: OpenAIConfig | AzureConfig | None = None,
    # ) -> ChatCompletionResult | tuple[AsyncGenerator[ChatCompletionChunk, None]]:
    #     kwargs = {
    #         "messages": messages,
    #         "max_tokens": max_tokens,
    #         "stream": stream,
    #         "response_format": response_format,
    #     }

    #     if isinstance(model, AzureChatModel):
    #         is_azure = True
    #         kwargs["deployment_id"] = model.deployment_id
    #     else:
    #         is_azure = False
    #         kwargs["model"] = model

    #     if frequency_penalty is not None:
    #         kwargs["frequency_penalty"] = frequency_penalty

    #     if functions:
    #         kwargs["functions"] = functions

    #         if function_call:
    #             kwargs["function_call"] = function_call

    #     if logit_bias:
    #         kwargs["logit_bias"] = logit_bias

    #     if stop:
    #         kwargs["stop"] = stop

    #     if n is not None:
    #         kwargs["n"] = n

    #     if presence_penalty is not None:
    #         kwargs["presence_penalty"] = presence_penalty

    #     if temperature is not None:
    #         kwargs["temperature"] = temperature

    #     if top_p is not None:
    #         kwargs["top_p"] = top_p

    #     if seed is not None:
    #         kwargs["seed"] = seed

    #     if user is not None:
    #         kwargs["user"] = user

    #     if request_timeout is not None:
    #         kwargs["request_timeout"] = request_timeout

    #     if config:
    #         kwargs = {**kwargs, **config.__dict__}

    #     if stream:

    #         async def _process_stream(raw_stream: AsyncGenerator[list[dict] | dict, None]) -> AsyncGenerator[ChatCompletionChunk, None]:
    #             async for chunk in raw_stream:
    #                 if isinstance(chunk, dict):
    #                     chunk = [chunk]

    #                 for c in chunk:
    #                     yield ChatCompletionChunk(**c)

    #         raw_stream: AsyncGenerator[list[dict] | dict, None] = await openai.ChatCompletion.acreate(**kwargs)  # type: ignore
    #         return (_process_stream(raw_stream),)  # tuple used to have correct static type

    #     else:
    #         try:
    #             raw_result: dict[str, Any] = await openai.ChatCompletion.acreate(**kwargs)  # type: ignore
    #             result = ChatCompletionResult(**raw_result)

    #             if is_azure and result.choices[0].finish_reason == "content_filter":
    #                 raise AzureContentFilterException(reason="completion")

    #             return result

    #         except InvalidRequestError as e:
    #             if is_azure and e.code == "content_filter":
    #                 raise AzureContentFilterException(reason="prompt")
    #             else:
    #                 raise e

    # endregion

    async def generate_completion(
        self,
        model: OpenAIChatModel | AzureChatModel,
        messages: list[ChatCompletionMessageParam],
        frequency_penalty: float | None | NotGiven = NOT_GIVEN,  # [-2, 2]
        function_call: completion_create_params.FunctionCall | NotGiven = NOT_GIVEN,
        functions: list[completion_create_params.Function] | NotGiven = NOT_GIVEN,
        logit_bias: dict[str, int] | None | NotGiven = NOT_GIVEN,  # [-100, 100]
        max_tokens: int | NotGiven = 1000,
        n: int | None | NotGiven = NOT_GIVEN,
        presence_penalty: float | None | NotGiven = NOT_GIVEN,  # [-2, 2]
        response_format: completion_create_params.ResponseFormat | NotGiven = NOT_GIVEN,
        seed: int | None | NotGiven = NOT_GIVEN,
        stop: str | list[str] | None | NotGiven = NOT_GIVEN,
        temperature: float | None | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: list[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_p: float | None | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        timeout: float | None | NotGiven = NOT_GIVEN,
    ) -> str:
        raw_model: OpenAIChatModel | str
        if isinstance(model, AzureChatModel):
            raw_model = model.deployment_id
            is_azure = True
        else:
            raw_model = model
            is_azure = False

        try:
            result = await self.create(
                model=raw_model,
                messages=messages,
                frequency_penalty=frequency_penalty,
                function_call=function_call,
                functions=functions,
                logit_bias=logit_bias,
                max_tokens=max_tokens,
                n=n,
                presence_penalty=presence_penalty,
                response_format=response_format,
                seed=seed,
                stop=stop,
                stream=False,
                temperature=temperature,
                tool_choice=tool_choice,
                tools=tools,
                top_p=top_p,
                user=user,
                timeout=timeout,
            )

            if is_azure and result.choices[0].finish_reason == "content_filter":
                raise AzureContentFilterException(reason="completion")

            return result.choices[0].message.content or ""

        except BadRequestError as e:
            if is_azure and e.code == "content_filter":
                raise AzureContentFilterException(reason="prompt")
            elif (
                is_azure and "filtered due to the prompt triggering Azure OpenAI" in e.message
            ):  # temporary fix: OpenAI library doesn't correctly parse code into error object
                raise AzureContentFilterException(reason="prompt")
            else:
                raise e

    @overload
    async def generate_output(
        self,
        model: OpenAIChatModel | AzureChatModel,
        prompt: PromptTemplate,
        max_output_tokens: int,
        output_type: type[_Output],
        max_input_tokens: int | None = None,
        frequency_penalty: float | None = None,  # [-2, 2]
        n: int | None | NotGiven = NOT_GIVEN,
        presence_penalty: float | None | NotGiven = NOT_GIVEN,  # [-2, 2]
        temperature: float | NotGiven = NOT_GIVEN,
        seed: int | None | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        timeout: float | None | NotGiven = NOT_GIVEN,
        retry_on_parse_error: int = 0,
    ) -> _Output:
        ...

    @overload
    async def generate_output(
        self,
        model: OpenAIChatModel | AzureChatModel,
        prompt: PromptTemplate,
        max_output_tokens: int,
        output_type: _UseDefaultType = _UseDefault,
        max_input_tokens: int | None = None,
        frequency_penalty: float | None = None,  # [-2, 2]
        n: int | None | NotGiven = NOT_GIVEN,
        presence_penalty: float | None | NotGiven = NOT_GIVEN,  # [-2, 2]
        temperature: float | NotGiven = NOT_GIVEN,
        seed: int | None | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        timeout: float | None | NotGiven = NOT_GIVEN,
        retry_on_parse_error: int = 0,
    ) -> BaseLLMResponse:
        ...

    async def generate_output(
        self,
        model: OpenAIChatModel | AzureChatModel,
        prompt: PromptTemplate,
        max_output_tokens: int,
        output_type: type[_Output] | _UseDefaultType = _UseDefault,
        max_input_tokens: int | None = None,
        frequency_penalty: float | None = None,  # [-2, 2]
        n: int | None | NotGiven = NOT_GIVEN,
        presence_penalty: float | None | NotGiven = NOT_GIVEN,  # [-2, 2]
        temperature: float | NotGiven = NOT_GIVEN,
        seed: int | None | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        timeout: float | None | NotGiven = NOT_GIVEN,
        retry_on_parse_error: int = 0,
    ) -> _Output | BaseLLMResponse:
        """
        Calls OpenAI Chat API, generates assistant response, and fits it into the output class

        :param model: model to use as `OpenAIChatModel` or `AzureChatModel`
        :param prompt: prompt, which is a subclass of `PromptTemplate`
        :param max_output_tokens: maximum number of tokens to generate
        :param output_type: output class used to parse the response, subclass of `BaseLLMResponse`. If not specified, the output defined in the prompt is used
        :param max_input_tokens: maximum number of tokens to use from the prompt. If not specified, the maximum number of tokens is calculated automatically
        :param request_timeout: timeout for the request in seconds
        :param retry_on_parse_error: number of retries if the response cannot be parsed (i.e. any `LLMParseException`). If set to 0, it has no effect.
        :param config: additional OpenAI/Azure config if needed (e.g. no global api key)
        """

        if isinstance(model, AzureChatModel):
            model_type = model.base_model
        else:
            model_type = model

        max_prompt_length = self.max_tokens_of_model(model_type) - max_output_tokens

        if max_input_tokens:
            max_prompt_length = min(max_prompt_length, max_input_tokens)

        messages = prompt.generate_messages(
            token_limit=max_prompt_length, token_counter=lambda messages: self.num_tokens_from_messages(messages, model=model_type)
        )

        completion = await self.generate_completion(
            model=model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            max_tokens=max_output_tokens,
            frequency_penalty=frequency_penalty,
            n=n,
            presence_penalty=presence_penalty,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            timeout=timeout,
        )

        try:
            if isinstance(output_type, _UseDefaultType):
                return prompt.Output.parse_response(completion)
            else:
                return output_type.parse_response(completion)
        except LLMParseException as e:
            if retry_on_parse_error > 0:
                return await self.generate_output(
                    model=model,
                    prompt=prompt,
                    max_output_tokens=max_output_tokens,
                    output_type=output_type,
                    max_input_tokens=max_input_tokens,
                    frequency_penalty=frequency_penalty,
                    n=n,
                    presence_penalty=presence_penalty,
                    temperature=temperature,
                    seed=seed,
                    top_p=top_p,
                    timeout=timeout,
                    retry_on_parse_error=retry_on_parse_error - 1,
                )
            else:
                raise e

    # region - token counting

    @staticmethod
    def max_tokens_of_model(model: OpenAIChatModel) -> int:
        match model:
            case "gpt-3.5-turbo" | "gpt-3.5-turbo-0301" | "gpt-3.5-turbo-0613":
                return 4096
            case "gpt-3.5-turbo-16k" | "gpt-3.5-turbo-16k-0613" | "gpt-3.5-turbo-1106":
                return 16384
            case "gpt-4" | "gpt-4-0314" | "gpt-4-0613":
                return 8192
            case "gpt-4-32k" | "gpt-4-32k-0314" | "gpt-4-32k-0613":
                return 32768
            case "gpt-4-1106-preview" | "gpt-4-vision-preview":
                return 128_000

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
        elif model in ("gpt-3.5-turbo-0301"):
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model in (
            "gpt-4-0314",
            "gpt-4-0613",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-32k-0314",
            "gpt-4-32k-0613",
            "gpt-4-1106-preview",
            "gpt-4-vision-preview",
        ):
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

    # endregion
