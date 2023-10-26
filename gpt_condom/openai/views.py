from dataclasses import dataclass
from typing import Literal, TypedDict

OpenAIChatModel = Literal[
    "gpt-3.5-turbo",  # 3.5 turbo
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k",  # 3.5 turbo 16k
    "gpt-3.5-turbo-16k-0613",
    "gpt-4",  # gpt-4
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",  # gpt-4 32k
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
]


class FunctionCallForceBehavior(TypedDict):
    name: str  # function name


FunctionCallBehavior = Literal["auto", "none"] | FunctionCallForceBehavior


EncodedFunction = dict[str, "EncodedFunction | str | list[str] | list[EncodedFunction] | list[str] | None"]


# region - Outputs


ChatCompletionRole = Literal["function", "system", "user", "assistant"]


class FunctionCall(TypedDict):
    name: str
    arguments: str


class Message(TypedDict):
    content: str | None
    role: ChatCompletionRole
    function_call: FunctionCall | None


class Choice(TypedDict):
    finish_reason: Literal["stop", "lenght", "function_call", "content_filter"]
    index: int
    message: Message


class CompletionUsage(TypedDict):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class ChatCompletionResult(TypedDict):
    id: str
    model: str
    object: str
    created: int
    choices: list[Choice]
    usage: CompletionUsage | None


# - Streaming


@dataclass
class ChatCompletionChunk:
    @dataclass
    class Choice:
        class Delta:
            content: str | None
            function_call: FunctionCall | None
            role: ChatCompletionRole | None

        delta: Delta
        finish_reason: Literal["stop", "length", "function_call", "content_filter", None]
        index: int

    id: str
    model: str
    choices: list[Choice]
    created: int
    object: str


# endregion
