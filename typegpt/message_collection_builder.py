from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Callable, Generic, TypeVar
from typegpt.example_builder import ExampleOutputFactory

from typegpt.prompt_definition.few_shot_example import FewShotExample

from .exceptions import LLMTokenLimitExceeded
from .prompt_builder import OutputPromptFactory

Prompt = TypeVar("Prompt", bound="PromptTemplate")

# EncodedMessage = dict[str, dict[str, str] | str | None]
EncodedMessage = dict[str, str]


class MessageCollectionFactory(Generic[Prompt]):
    def __init__(self, prompt: Prompt, token_counter: Callable[[list[EncodedMessage]], int]):
        self.prompt = prompt
        self.token_counter = token_counter
        self.output_prompt_factory = OutputPromptFactory(
            list(prompt.Output.__fields__.values())
        )  #  TODO: add config for `threaten` and more

    def _generate_single_fewshot_example_messages(self, example: FewShotExample) -> list[EncodedMessage]:
        encoded_output = ExampleOutputFactory(example.output).generate()
        return [
            {"role": "system", "name": "example_user", "content": example.input},
            {"role": "system", "name": "example_assistant", "content": encoded_output},
        ]

    def _generate_fewshot_example_messages(self, examples: list[FewShotExample]) -> list[EncodedMessage]:
        return sum([self._generate_single_fewshot_example_messages(example) for example in examples], [])

    def _generate_messages_from_prompt(self, prompt: Prompt) -> list[EncodedMessage]:
        system_prompt = prompt.system_prompt()

        if not prompt.settings.disable_formatting_instructions:
            system_prompt += "\n\n"
            system_prompt += self.output_prompt_factory.generate()

        result: list[EncodedMessage] = [{"role": "system", "content": system_prompt}]

        if few_shot_examples := prompt.few_shot_examples():
            result += self._generate_fewshot_example_messages(few_shot_examples)

        result.append({"role": "user", "content": prompt.user_prompt()})

        return result

    def generate_messages(self, token_limit: int):
        """
        Generates messages dictionary that can be sent to any OpenAI equivalent API, ensuring that the total number of tokens is below the specified limit
        Messages that do not fit in are removed inside the object permanently
        """

        generated_messages = self._generate_messages_from_prompt(self.prompt)

        num_tokens = self.token_counter(generated_messages)
        if num_tokens <= token_limit:
            return generated_messages

        # try to reduce the length of the prompt
        prompt = copy.deepcopy(self.prompt)
        while prompt.reduce_if_possible():
            generated_messages = self._generate_messages_from_prompt(prompt)

            num_tokens = self.token_counter(generated_messages)
            if num_tokens <= token_limit:
                self.prompt = prompt  # update the prompt if successful
                return generated_messages

        raise LLMTokenLimitExceeded(
            f"Prompt can't be reduced to fit within the token limit ({token_limit})",
            system_prompt=prompt.system_prompt(),
            user_prompt=prompt.user_prompt(),
        )


if TYPE_CHECKING:
    from .prompt_definition.prompt_template import PromptTemplate
