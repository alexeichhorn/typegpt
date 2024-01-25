from dataclasses import dataclass
from typing import Literal, TypedDict

OpenAIChatModel = Literal[
    "gpt-3.5-turbo",  # 3.5 turbo
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-16k",  # 3.5 turbo 16k
    "gpt-3.5-turbo-16k-0613",
    "gpt-4",  # gpt-4
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",  # gpt-4 32k
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
    "gpt-4-turbo-preview",  # gpt-4 turbo
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-vision-preview",  # gpt-4 vision
]


@dataclass
class AzureChatModel:
    deployment_id: str
    base_model: OpenAIChatModel  # only used for token counting


@dataclass
class AzureConfig:
    api_key: str
    api_base: str
    api_version: str
    api_type: str = "azure"


EncodedFunction = dict[str, "EncodedFunction | str | list[str] | list[EncodedFunction] | list[str] | None"]
