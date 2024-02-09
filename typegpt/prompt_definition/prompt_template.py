from __future__ import annotations

from typing import TYPE_CHECKING, Callable, ClassVar, Generic, Protocol, TypeVar

from typegpt.prompt_definition.few_shot_example import FewShotExample
from typegpt.prompt_definition.prompt_settings import PromptSettings

from ..base import BaseLLMResponse
from ..fields import LLMArrayOutput
from ..message_collection_builder import EncodedMessage, MessageCollectionFactory

# _Output = TypeVar("_Output", bound=BaseLLMResponse)


class PromptTemplate(Protocol):  # , Generic[_Output]):
    def system_prompt(self) -> str:
        """System prompt for the LLM (required)"""
        ...

    def user_prompt(self) -> str:
        """User prompt for the LLM (required)"""
        ...

    def reduce_if_possible(self) -> bool:
        """
        Override this method to reduce the parameters of the prompt, which gets called when the token limit is exceeded
        @returns: whether the parameters could be further reduced
        """
        return False

    def few_shot_examples(self) -> list[FewShotExample[BaseLLMResponse]]:  # list[FewShotExample[_Output]]:
        """
        Override this method to provide few shot examples
        """
        return []

    Output: type[BaseLLMResponse]  # type[_Output]

    settings: PromptSettings = PromptSettings()

    def generate_messages(self, token_limit: int, token_counter: Callable[[list[EncodedMessage]], int]):
        """
        Generates messages dictionary that can be sent to any OpenAI equivalent API, ensuring that the total number of tokens is below the specified limit
        Messages that do not fit in are removed inside the object permanently
        """
        return MessageCollectionFactory(self, token_counter=token_counter).generate_messages(token_limit=token_limit)


# if TYPE_CHECKING:
#     from ..message_collection_builder import EncodedMessage, MessageCollectionFactory
