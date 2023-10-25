import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

import pytest

from llm_lib import BaseLLMResponse, LLMArrayOutput, LLMOutput
from llm_lib.fields import ExamplePosition, LLMArrayOutputInfo
from llm_lib.prompt_builder import OutputPromptFactory


class TestPromptFactory:
    class SimpleTestOutput(BaseLLMResponse):
        title: str
        description: str | None
        tags: list[str]
        cool_integer: int
        connected_floats: list[float]
        mice: list[str]

    def test_simple_output_fields(self):
        fields = list(self.SimpleTestOutput.__fields__.values())
        prompt = OutputPromptFactory(fields, threaten=True).generate()
        expected_prompt = f"""
Return the answer in the following format, otherwise people might die and you are the only one to blame:
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
\"""
""".strip()

        assert prompt == expected_prompt

    # -

    class CustomExplainedTestOutput(BaseLLMResponse):
        abcd: str = LLMOutput("Just put a random string here (perferably 'abcd')", multiline=False)
        longer_sentences: str = LLMOutput("Put longer sentences here please", multiline=True)

    def test_custom_explained_output_fields(self):
        fields = list(self.CustomExplainedTestOutput.__fields__.values())
        prompt = OutputPromptFactory(fields).generate()
        expected_prompt = f"""
Return the answer in the following format:
\"""
ABCD: <Just put a random string here (perferably 'abcd')>
LONGER SENTENCES: <Put longer sentences here please>
\"""
""".strip()

        assert prompt == expected_prompt
