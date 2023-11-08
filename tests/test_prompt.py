import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

import pytest

from typegpt import BaseLLMResponse, LLMArrayOutput, LLMOutput
from typegpt.fields import ExamplePosition, LLMArrayOutputInfo
from typegpt.prompt_builder import OutputPromptFactory


class TestPromptFactory:
    class SimpleTestOutput(BaseLLMResponse):
        title: str
        description: str | None
        tags: list[str]
        cool_integer: int
        connected_floats: list[float]
        mice: list[str]
        is_active: bool

    def test_simple_output_fields(self):
        fields = list(self.SimpleTestOutput.__fields__.values())
        prompt = OutputPromptFactory(fields, threaten=True).generate()
        expected_prompt = f"""
Always return the answer in the following format, otherwise people might die and you are the only one to blame:
\"""
TITLE: <Put the title here>
DESCRIPTION: <Put the description here>
TAG 1: <Put the first tag here>
TAG 2: <Put the second tag here>
...

COOL INTEGER: <Put the cool integer here>
CONNECTED FLOAT 1: <Put the first connected float here>
CONNECTED FLOAT 2: <Put the second connected float here>
...

MOUSE 1: <Put the first mouse here>
MOUSE 2: <Put the second mouse here>
...

IS ACTIVE: <'true' if is active, else 'false'>
\"""
""".strip()

        assert prompt == expected_prompt

    # -

    class CustomExplainedTestOutput(BaseLLMResponse):
        abcd: str = LLMOutput("Just put a random string here (perferably 'abcd')", multiline=False)
        longer_sentences: str = LLMOutput("Put longer sentences here please", multiline=True)
        items: list[str] = LLMArrayOutput((1, 2), instruction=lambda i: f"Put item {i} here")

    def test_custom_explained_output_fields(self):
        fields = list(self.CustomExplainedTestOutput.__fields__.values())
        prompt = OutputPromptFactory(fields).generate()
        expected_prompt = f"""
Always return the answer in the following format:
\"""
ABCD: <Just put a random string here (perferably 'abcd')>
LONGER SENTENCES: <Put longer sentences here please>
ITEM 1: <Put item 1 here>
ITEM 2: <Put item 2 here>
\"""
""".strip()

        assert prompt == expected_prompt

    # -

    class CustomRestrictedArrayOutput(BaseLLMResponse):
        fixed: list[str] = LLMArrayOutput(expected_count=3, instruction=lambda i: f"Put the {i.ordinal} fixed string here")
        fixed_longer: list[str] = LLMArrayOutput(expected_count=10, instruction=lambda i: f"Put the {i.ordinal} longer fixed string here")
        min_max_integers: list[int] = LLMArrayOutput((2, 5), instruction=lambda i: f"Put the {i.ordinal} integer here")

    def test_custom_restricted_array_output(self):
        fields = list(self.CustomRestrictedArrayOutput.__fields__.values())
        prompt = OutputPromptFactory(fields).generate()
        expected_prompt = f"""
Always return the answer in the following format:
\"""
FIXED 1: <Put the first fixed string here>
FIXED 2: <Put the second fixed string here>
FIXED 3: <Put the third fixed string here>

FIXED LONGER 1: <Put the first longer fixed string here>
FIXED LONGER 2: <Put the second longer fixed string here>
...
FIXED LONGER 10: <Put the 10th longer fixed string here>

MIN MAX INTEGER 1: <Put the first integer here>
MIN MAX INTEGER 2: <Put the second integer here>
...
MIN MAX INTEGER 5: <Put the 5th integer here>
\"""
""".strip()

        assert prompt == expected_prompt
