import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

import logging

import pytest

from typegpt import BaseLLMArrayElement, BaseLLMResponse, LLMArrayElementOutput, LLMArrayOutput, LLMOutput, PromptTemplate
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
        optional_integer: int | None
        filled_optional_integer: int | None
        optional_bool: bool | None
        connected_floats: list[float]
        mice: list[str]
        sample_with_default: str = "some default value"

    def test_parse_simple_output(self):
        completion_output = """
TITLE: Some title
DESCRIPTION: "Some description"
TAG 1: first tag
some irrelevant stuff that should be ignored
COOL INTEGER: 33
FILLED OPTIONAL INTEGER: 44
OPTIONAL BOOL: yes
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
        assert parsed_output.optional_integer is None
        assert parsed_output.filled_optional_integer == 44
        assert parsed_output.optional_bool == True
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
MOUSE 2: 'Minnie'
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
    # region - 7

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

    def test_parse_subtype_output(self):
        completion_output_1 = """
TITLE: Hello world
STRING 1: s1
STRING 2: s2
STRING 3: s3
ITEM 1 SUBTITLE: subtitle one
ITEM 1 DESCRIPTION: description one
ITEM 1 ABSTRACT: Just an abstract...
ITEM 2 SUBTITLE: subtitle two
ITEM 2 DESCRIPTION: Description TWO
ITEM 2 ABSTRACT: More abstract...

SUBITEM TITLE: A subitem title (!!)

OPTIONAL SUBITEM TITLE: Optional subitem title (but filled)
""".strip()

        parsed_output = self.SubtypeTestOutput.parse_response(completion_output_1)
        assert parsed_output.title == "Hello world"
        assert parsed_output.strings == ["s1", "s2", "s3"]
        assert len(parsed_output.items) == 2
        assert parsed_output.items[0].subtitle == "subtitle one"
        assert parsed_output.items[0].description == "description one"
        assert parsed_output.items[0].abstract == "Just an abstract..."
        assert parsed_output.items[1].subtitle == "subtitle two"
        assert parsed_output.items[1].description == "Description TWO"
        assert parsed_output.items[1].abstract == "More abstract..."
        assert parsed_output.subitem.title == "A subitem title (!!)"
        assert parsed_output.optional_subitem is not None
        assert parsed_output.optional_subitem.title == "Optional subitem title (but filled)"

        completion_output_2 = """
TITLE: Hello world

SUBITEM TITLE: A subitem title (!!)
""".strip()

        parsed_output_2 = self.SubtypeTestOutput.parse_response(completion_output_2)
        assert parsed_output_2.title == "Hello world"
        assert parsed_output_2.strings == []
        assert len(parsed_output_2.items) == 0
        assert parsed_output_2.subitem.title == "A subitem title (!!)"
        assert parsed_output_2.optional_subitem is None

    # endregion
    # region - 8
    class UltraSubtypeTestOutput(BaseLLMResponse):
        class Item(BaseLLMArrayElement):
            class InnerItem(BaseLLMResponse):
                title: str
                description: str

            class InnerElement(BaseLLMArrayElement):
                value: float
                is_accurate: bool

            subtitle: str
            description: str | None = LLMArrayElementOutput(lambda pos: f"Put the {pos.ordinal} item description here")
            abstract: str = LLMArrayElementOutput(lambda _: "...", multiline=True)
            inner_item: InnerItem
            inner_elements: list[InnerElement] = LLMArrayOutput(2, instruction=lambda _: "...")
            multiline_text: list[str] = LLMArrayOutput((0, None), lambda _: "...", multiline=True)

        class DirectItem(BaseLLMResponse):
            class InnerDirectElement(BaseLLMArrayElement):
                subtitle: str

            title: str
            x: list[InnerDirectElement]

        title: str
        subitem: DirectItem
        items: list[Item]

    def test_parse_ultra_subtype_output(self):
        completion_output = """
TITLE: Main head title

SUBITEM TITLE: subtitle
disregarded!
SUBITEM X 1 SUBTITLE: sub1
? also disregarded ?
SUBITEM X 2 SUBTITLE: sub2

ITEM 1 SUBTITLE: First item subtitle
ITEM 1 DESCRIPTION: first descr.
ITEM 1 ABSTRACT: Jsut some random abstract
ITEM 1 INNER ITEM TITLE: INNER TITLE 1
ITEM 1 INNER ITEM DESCRIPTION: DESCription 1
ITEM 1 INNER ELEMENT 1 VALUE: 1.0
ITEM 1 INNER ELEMENT 1 IS ACCURATE: yes
ITEM 1 INNER ELEMENT 2 VALUE: 3.14
ITEM 1 INNER ELEMENT 2 IS ACCURATE: False
ITEM 1 MULTILINE TEXT 1: line 1
line 2
line 3
ITEM 1 MULTILINE TEXT 2: line 4
line 5
line 6
ITEM 1 MULTILINE TEXT 3: line 7
line 8
line 9

ITEM 2 SUBTITLE: Second item subtitle
ITEM 2 ABSTRACT: Another abstract but this time
with multiple lines
ITEM 2 INNER ITEM TITLE: tt2
ITEM 2 INNER ITEM DESCRIPTION: dd2
ITEM 2 INNER ELEMENT 1 VALUE: 2.0
ITEM 2 INNER ELEMENT 1 IS ACCURATE: no
ITEM 2 INNER ELEMENT 2 VALUE: 8.958
ITEM 2 INNER ELEMENT 2 IS ACCURATE: True
"""
        parsed_output = self.UltraSubtypeTestOutput.parse_response(completion_output)
        assert parsed_output.title == "Main head title"
        assert parsed_output.subitem.title == "subtitle"
        assert len(parsed_output.subitem.x) == 2
        assert parsed_output.subitem.x[0].subtitle == "sub1"
        assert parsed_output.subitem.x[1].subtitle == "sub2"

        assert len(parsed_output.items) == 2
        assert parsed_output.items[0].subtitle == "First item subtitle"
        assert parsed_output.items[0].description == "first descr."
        assert parsed_output.items[0].abstract == "Jsut some random abstract"
        assert parsed_output.items[0].inner_item.title == "INNER TITLE 1"
        assert parsed_output.items[0].inner_item.description == "DESCription 1"
        assert len(parsed_output.items[0].inner_elements) == 2
        assert parsed_output.items[0].inner_elements[0].value == 1.0
        assert parsed_output.items[0].inner_elements[0].is_accurate == True
        assert parsed_output.items[0].inner_elements[1].value == 3.14
        assert parsed_output.items[0].inner_elements[1].is_accurate == False
        assert parsed_output.items[0].multiline_text == ["line 1\nline 2\nline 3", "line 4\nline 5\nline 6", "line 7\nline 8\nline 9"]

        assert parsed_output.items[1].subtitle == "Second item subtitle"
        assert parsed_output.items[1].description is None
        assert parsed_output.items[1].abstract == "Another abstract but this time\nwith multiple lines"
        assert parsed_output.items[1].inner_item.title == "tt2"
        assert parsed_output.items[1].inner_item.description == "dd2"
        assert len(parsed_output.items[1].inner_elements) == 2
        assert parsed_output.items[1].inner_elements[0].value == 2.0
        assert parsed_output.items[1].inner_elements[0].is_accurate == False
        assert parsed_output.items[1].inner_elements[1].value == 8.958
        assert parsed_output.items[1].inner_elements[1].is_accurate == True

    # endregion
