import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

import pytest

from typegpt import BaseLLMArrayElement, BaseLLMResponse, LLMArrayElementOutput, LLMArrayOutput, LLMOutput
from typegpt.fields import ExamplePosition, LLMArrayElementOutputInfo, LLMArrayOutputInfo


class TestFields:
    class SimpleTestOutput(BaseLLMResponse):
        title: str
        description: str | None
        tags: list[str]
        cool_integer: int
        connected_floats: list[float]
        mice: list[str]
        sample_with_default: str = "some default value"

    def test_simple_output_fields(self):
        expected_fields = {
            "title": LLMOutput("Put the title here"),
            "description": LLMOutput("Put the description here", default=None),
            "tags": LLMArrayOutput((0, None), lambda pos: f"Put the {pos.ordinal} tag here"),
            "cool_integer": LLMOutput("Put the cool integer here"),
            "connected_floats": LLMArrayOutput((0, None), lambda pos: f"Put the {pos.ordinal} connected float here"),
            "mice": LLMArrayOutputInfo(lambda pos: f"Put the {pos.ordinal} mouse here", 0, None, multiline=False),
            "sample_with_default": LLMOutput("Put the sample with default here", default="some default value"),
        }

        for key, value in self.SimpleTestOutput.__fields__.items():
            assert key in expected_fields
            if isinstance(value.info, LLMArrayOutputInfo):
                for i in range(1, 10):
                    assert value.info.instruction(ExamplePosition(i)) == expected_fields[key].instruction(ExamplePosition(i))
                expected_fields[key].instruction = value.info.instruction
            assert value.info == expected_fields[key], f"Field '{key}' does not match expected value"

        # also make sure it is exhaustive
        assert len(self.SimpleTestOutput.__fields__) == len(expected_fields)

    # -

    class ExtendedTestOutput(BaseLLMResponse):
        title: str | None = LLMOutput("Put the title here")
        some_text: str = LLMOutput("...", default="empty")

    def test_extended_output_fields(self):
        expected_fields = {
            "title": LLMOutput("Put the title here", default=None),
            "some_text": LLMOutput("...", default="empty"),
        }

        for key, value in self.ExtendedTestOutput.__fields__.items():
            assert key in expected_fields
            assert value.info == expected_fields[key], f"Field '{key}' does not match expected value"

    # -

    class SubelementsTestOutput(BaseLLMResponse):
        class Item(BaseLLMArrayElement):
            title: str
            description: str
            inner_array: list[str]

        class Coordinates(BaseLLMResponse):
            x: int
            y: float

        title: str
        coordinates: Coordinates
        subelements: list[BaseLLMArrayElement]

    def test_subelements_output_fields(self):
        expected_fields = {
            "title": LLMOutput("Put the title here"),
            "coordinates": LLMOutput("Put the coordinates here"),
            "subelements": LLMArrayOutput((0, None), lambda pos: f"Put the {pos.ordinal} subelement here"),
        }

        for key, value in self.SubelementsTestOutput.__fields__.items():
            assert key in expected_fields
            if isinstance(value.info, LLMArrayOutputInfo):
                for i in range(1, 10):
                    assert value.info.instruction(ExamplePosition(i)) == expected_fields[key].instruction(ExamplePosition(i))
                expected_fields[key].instruction = value.info.instruction
            assert value.info == expected_fields[key], f"Field '{key}' does not match expected value"

        # - items

        expected_item_fields = {
            "title": LLMArrayElementOutput(lambda pos: f"Put the {pos.ordinal} title here"),
            "description": LLMArrayElementOutput(lambda pos: f"Put the {pos.ordinal} description here"),
            "inner_array": LLMArrayOutput((0, None), lambda pos: f"Put the {pos.ordinal} inner array here"),
        }

        for key, value in self.SubelementsTestOutput.Item.__fields__.items():
            assert key in expected_item_fields
            if isinstance(value.info, LLMArrayOutputInfo):
                for i in range(1, 10):
                    assert value.info.instruction(ExamplePosition(i)) == expected_item_fields[key].instruction(ExamplePosition(i))
                expected_item_fields[key].instruction = value.info.instruction
            elif isinstance(value.info, LLMArrayElementOutputInfo):
                for i in range(1, 10):
                    assert value.info.instruction(ExamplePosition(i)) == expected_item_fields[key].instruction(ExamplePosition(i))
                expected_item_fields[key].instruction = value.info.instruction
            assert value.info == expected_item_fields[key], f"Field '{key}' does not match expected value"

        # also make sure it is exhaustive
        assert len(self.SubelementsTestOutput.__fields__) == len(expected_fields)
