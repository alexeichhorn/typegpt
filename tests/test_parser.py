import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

import logging

import pytest

from typegpt import BaseLLMResponse, LLMArrayOutput, LLMOutput, PromptTemplate
from typegpt.exceptions import LLMOutputFieldInvalidLength, LLMOutputFieldMissing, LLMOutputFieldWrongType
from typegpt.fields import ExamplePosition, LLMArrayOutputInfo
from typegpt.utils.internal_types import _NoDefault


class TestFields:
    # region - 1

    class SimpleTestOutput(BaseLLMResponse):
        title: str
        description: str | None
        tags: list[str]
        cool_integer: int
        connected_floats: list[float]
        mice: list[str]
        sample_with_default: str = "some default value"

    def test_parse_simple_output(self):
        completion_output = """
TITLE: Some title
DESCRIPTION: Some description
TAG 1: first tag
some irrelevant stuff that should be ignored
COOL INTEGER: 33
blabla
CONNECTED FLOAT 1: 1.0
CONNECTED FLOAT 2: 2.0
CONNECTED FLOAT 3: 3.14
"""

        parsed_output = self.SimpleTestOutput.parse_response(completion_output)
        assert parsed_output.title == "Some title"
        assert parsed_output.description == "Some description"
        assert parsed_output.tags == ["first tag"]
        assert parsed_output.cool_integer == 33
        assert parsed_output.connected_floats == [1.0, 2.0, 3.14]
        assert parsed_output.mice == []
        assert parsed_output.sample_with_default == "some default value"
        assert parsed_output.__raw_completion__ == completion_output

    def test_parse_simple_output_2(self):
        completion_output = """
\"""
TITLE: 3.58
COOL INTEGER: -55
MOUSE 1: Mickey
MOUSE 2: Minnie
SAMPLE WITH DEFAULT: some other value
\"""
"""

        parsed_output = self.SimpleTestOutput.parse_response(completion_output)
        assert parsed_output.title == "3.58"
        assert parsed_output.description is None
        assert parsed_output.tags == []
        assert parsed_output.cool_integer == -55
        assert parsed_output.connected_floats == []
        assert parsed_output.mice == ["Mickey", "Minnie"]
        assert parsed_output.sample_with_default == "some other value"
        assert parsed_output.__raw_completion__ == completion_output

    # endregion
    # region - 2

    class MultilineSingleTestOutput(BaseLLMResponse):
        text: str = LLMOutput("Put the text here", multiline=True)

    def test_parse_multiline_single_output(self):
        completion_output = """
TEXT: Text line 1
Text line 2
Text line 3
"""

        parsed_output = self.MultilineSingleTestOutput.parse_response(completion_output)
        assert parsed_output.text == "Text line 1\nText line 2\nText line 3"

        completion_output_2 = """
\"""
TEXT: Text line 1
Text line 2
\"""
"""

        parsed_output_2 = self.MultilineSingleTestOutput.parse_response(completion_output_2)
        assert parsed_output_2.text == "Text line 1\nText line 2"

    # endregion
    # region - 3

    class MultilineMultipleTestOutput(BaseLLMResponse):
        text: str = LLMOutput("Put the text here", multiline=True)
        value: int

    def test_parse_multiline_multiple_output(self):
        completion_output = """
TEXT: Text line 1
Text line 2
VALUE: 12345
"""

        parsed_output = self.MultilineMultipleTestOutput.parse_response(completion_output)
        assert parsed_output.text == "Text line 1\nText line 2"
        assert parsed_output.value == 12345

        completion_output_2 = """
VALUE: 45
77
TEXT: L1
L2
"""

        parsed_output_2 = self.MultilineMultipleTestOutput.parse_response(completion_output_2)
        assert parsed_output_2.text == "L1\nL2"
        assert parsed_output_2.value == 45

        completion_output_3 = """
VALUE: 78
"""

        with pytest.raises(LLMOutputFieldMissing):
            self.MultilineMultipleTestOutput.parse_response(completion_output_3)

        completion_output_4 = """
TEXT: L1
"""

        with pytest.raises(LLMOutputFieldMissing):
            self.MultilineMultipleTestOutput.parse_response(completion_output_4)

        completion_output = """
TEXT: L1
VALUE: 8xz
"""

        with pytest.raises(LLMOutputFieldWrongType):
            self.MultilineMultipleTestOutput.parse_response(completion_output)

    # endregion
    # region - 4

    class LimitedArrayTestOutput(BaseLLMResponse):
        geese: list[str] = LLMArrayOutput((2, 3), lambda _: "test")

    # endregion
    # region - 5

    class MultilineArrayTestOutput(BaseLLMResponse):
        apples: list[str] = LLMArrayOutput((2, 3), lambda _: "test", multiline=True)

    def test_parse_multiline_array_output(self):
        completion_output = """
APPLE 1: L1
L2
APPLE 2: L3
L4
"""

        parsed_output = self.MultilineArrayTestOutput.parse_response(completion_output)
        assert parsed_output.apples == ["L1\nL2", "L3\nL4"]

        completion_output_2 = """
APPLE 1: L1
"""

        with pytest.raises(LLMOutputFieldInvalidLength):
            self.MultilineArrayTestOutput.parse_response(completion_output_2)

        completion_output_3 = """
APPLE 1: L1
APPLE 2: L2
APPLE 3: L3
"""

        parsed_output_3 = self.MultilineArrayTestOutput.parse_response(completion_output_3)
        assert parsed_output_3.apples == ["L1", "L2", "L3"]

        completion_output_4 = """
APPLE 1: L1
APPLE 2: L2
APPLE 3: L3
APPLE 4: L4
"""

        with pytest.raises(LLMOutputFieldInvalidLength):
            self.MultilineArrayTestOutput.parse_response(completion_output_4)

    # endregion
    # region - 6

    def test_dyamic_parse_output(self):
        class DynamicTestPrompt(PromptTemplate):
            def __init__(self, num_items: int, actor_needed: bool):
                self.num_items = num_items
                self.actor_needed = actor_needed

            @property
            def Output(self):
                class Output(BaseLLMResponse):
                    items: list[str] = LLMArrayOutput(self.num_items, lambda _: "test")
                    actor: str = LLMOutput("Some actor", default=_NoDefault if self.actor_needed else "some default")

                return Output

            def system_prompt(self) -> str:
                return "..."

            def user_prompt(self) -> str:
                return "..."

        completion_output_1 = """
ITEM 1: L1
"""

        completion_output_2 = """
ITEM 1: L1
ITEM 2: L2
"""

        completion_output_3 = """
ITEM 1: L1
ITEM 2: L2
ACTOR: Some actor
"""

        parsed_output_1 = DynamicTestPrompt(1, False).Output.parse_response(completion_output_1)
        assert parsed_output_1.items == ["L1"]
        assert parsed_output_1.actor == "some default"

        parsed_output_2 = DynamicTestPrompt(2, False).Output.parse_response(completion_output_2)
        assert parsed_output_2.items == ["L1", "L2"]
        assert parsed_output_2.actor == "some default"

        parsed_output_3 = DynamicTestPrompt(2, True).Output.parse_response(completion_output_3)
        assert parsed_output_3.items == ["L1", "L2"]
        assert parsed_output_3.actor == "Some actor"

        parsed_output_4 = DynamicTestPrompt(2, False).Output.parse_response(completion_output_3)
        assert parsed_output_4.items == ["L1", "L2"]
        assert parsed_output_4.actor == "Some actor"

        with pytest.raises(LLMOutputFieldMissing):
            DynamicTestPrompt(2, True).Output.parse_response(completion_output_2)

        with pytest.raises(LLMOutputFieldInvalidLength):
            DynamicTestPrompt(2, False).Output.parse_response(completion_output_1)

    # endregion
