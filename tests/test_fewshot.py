import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

import pytest

from typegpt import BaseLLMArrayElement, BaseLLMResponse, LLMArrayElementOutput, LLMArrayOutput, LLMOutput, PromptTemplate, FewShotExample


class TestFewShot:
    class SimplePrompt(PromptTemplate):
        class Output(BaseLLMResponse):
            title: str

        def system_prompt(self) -> str:
            return "Some system prompt"

        def user_prompt(self) -> str:
            return "Some user prompt"

    def test_simple_prompt(self):
        prompt = self.SimplePrompt()
        messages = prompt.generate_messages(token_limit=1000, token_counter=lambda x: 0)

        expected_system_prompt = f"""
Some system prompt

Always return the answer in the following format:
\"""
TITLE: <Put the title here>
\"""
""".strip()

        assert messages == [
            {"role": "system", "content": expected_system_prompt},
            {"role": "user", "content": "Some user prompt"},
        ]

    class SimplePromptWithFewShot(PromptTemplate):
        class Output(BaseLLMResponse):
            title: str

        def __init__(self, num_shots: int):
            self.num_shots = num_shots

        def system_prompt(self) -> str:
            return "Some system prompt"

        def user_prompt(self) -> str:
            return "Some user prompt"

        def few_shot_examples(self) -> list[FewShotExample[BaseLLMResponse]]:
            return [FewShotExample(input=f"Some input {i}", output=self.Output(title=f"Some title {i}")) for i in range(self.num_shots)]

    def test_simple_few_shot_prompt(self):
        prompt_one = self.SimplePromptWithFewShot(num_shots=1)
        prompt_three = self.SimplePromptWithFewShot(num_shots=3)
        messages_one = prompt_one.generate_messages(token_limit=1000, token_counter=lambda x: 0)
        messages_three = prompt_three.generate_messages(token_limit=1000, token_counter=lambda x: 0)

        expected_system_prompt = f"""
Some system prompt

Always return the answer in the following format:
\"""
TITLE: <Put the title here>
\"""
""".strip()

        assert messages_one == [
            {"role": "system", "content": expected_system_prompt},
            {"role": "system", "name": "example_user", "content": "Some input 0"},
            {"role": "system", "name": "example_assistant", "content": "TITLE: Some title 0"},
            {"role": "user", "content": "Some user prompt"},
        ]

        assert messages_three == [
            {"role": "system", "content": expected_system_prompt},
            {"role": "system", "name": "example_user", "content": "Some input 0"},
            {"role": "system", "name": "example_assistant", "content": "TITLE: Some title 0"},
            {"role": "system", "name": "example_user", "content": "Some input 1"},
            {"role": "system", "name": "example_assistant", "content": "TITLE: Some title 1"},
            {"role": "system", "name": "example_user", "content": "Some input 2"},
            {"role": "system", "name": "example_assistant", "content": "TITLE: Some title 2"},
            {"role": "user", "content": "Some user prompt"},
        ]
