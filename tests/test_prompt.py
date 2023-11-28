import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

import pytest

from typegpt import BaseLLMArrayElement, BaseLLMResponse, LLMArrayElementOutput, LLMArrayOutput, LLMOutput
from typegpt.fields import ExamplePosition, LLMArrayElementOutputInfo, LLMArrayOutputInfo
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

    # -

    class SubtypeTestOutput(BaseLLMResponse):
        class Item(BaseLLMArrayElement):
            subtitle: str
            description: str = LLMArrayElementOutput(lambda pos: f"Put the {pos.ordinal} item description here")
            abstract: str = LLMArrayElementOutput(lambda _: "Some instruction", default="default abstract")

        class DirectItem(BaseLLMResponse):
            title: str

        title: str
        strings: list[str]
        items: list[Item]
        subitem: DirectItem
        optional_subitem: DirectItem | None = None

    def test_subtype_output_fields(self):
        fields = list(self.SubtypeTestOutput.__fields__.values())
        prompt = OutputPromptFactory(fields).generate()
        expected_prompt = f"""
Always return the answer in the following format:
\"""
TITLE: <Put the title here>
STRING 1: <Put the first string here>
STRING 2: <Put the second string here>
...

ITEM 1 SUBTITLE: <Put the first subtitle here>
ITEM 1 DESCRIPTION: <Put the first item description here>
ITEM 1 ABSTRACT: <Some instruction>
ITEM 2 SUBTITLE: <Put the second subtitle here>
ITEM 2 DESCRIPTION: <Put the second item description here>
ITEM 2 ABSTRACT: <Some instruction>
...

SUBITEM TITLE: <Put the title here>

OPTIONAL SUBITEM TITLE: <Put the title here>
\"""
""".strip()

        assert prompt == expected_prompt

    class UltraSubtypeTestOutput(BaseLLMResponse):
        class Item(BaseLLMArrayElement):
            class InnerItem(BaseLLMResponse):
                title: str
                description: str

            class InnerElement(BaseLLMArrayElement):
                value: int

            subtitle: str
            description: str
            abstract: str
            inner_item: InnerItem
            inner_elements: list[InnerElement] = LLMArrayOutput(2, instruction=lambda _: "...")

        class DirectItem(BaseLLMResponse):
            class InnerDirectElement(BaseLLMArrayElement):
                subtitle: str

            title: str
            x: list[InnerDirectElement]

        title: str
        subitem: DirectItem
        items: list[Item]

    def test_ultra_subtype_output_fields(self):
        fields = list(self.UltraSubtypeTestOutput.__fields__.values())
        prompt = OutputPromptFactory(fields).generate()
        expected_prompt = f"""
Always return the answer in the following format:
\"""
TITLE: <Put the title here>

SUBITEM TITLE: <Put the title here>
SUBITEM X 1 SUBTITLE: <Put the first subtitle here>
SUBITEM X 2 SUBTITLE: <Put the second subtitle here>
...

ITEM 1 SUBTITLE: <Put the first subtitle here>
ITEM 1 DESCRIPTION: <Put the first description here>
ITEM 1 ABSTRACT: <Put the first abstract here>

ITEM 1 INNER ITEM TITLE: <Put the title here>
ITEM 1 INNER ITEM DESCRIPTION: <Put the description here>

ITEM 1 INNER ELEMENT 1 VALUE: <Put the first value here>
ITEM 1 INNER ELEMENT 2 VALUE: <Put the second value here>
ITEM 2 SUBTITLE: <Put the second subtitle here>
ITEM 2 DESCRIPTION: <Put the second description here>
ITEM 2 ABSTRACT: <Put the second abstract here>

ITEM 2 INNER ITEM TITLE: <Put the title here>
ITEM 2 INNER ITEM DESCRIPTION: <Put the description here>

ITEM 2 INNER ELEMENT 1 VALUE: <Put the first value here>
ITEM 2 INNER ELEMENT 2 VALUE: <Put the second value here>
...
\"""
""".strip()

        assert prompt == expected_prompt
