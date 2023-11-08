import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

from typing import Any, List, Literal, Optional

import pytest

from typegpt import BaseLLMResponse, LLMArrayOutput, LLMOutput
from typegpt.exceptions import LLMOutputFieldInvalidLength, LLMOutputFieldMissing, LLMOutputFieldWrongType
from typegpt.fields import ExamplePosition, LLMArrayOutputInfo


class TestResponseObject:
    class SimpleTestOutput(BaseLLMResponse):
        title: str
        description: str | None
        tags: list[str]

    def test_correct_initialization(self):
        x = self.SimpleTestOutput(title="Some title", description="Some description", tags=["first tag"])
        assert x.title == "Some title"
        assert x.description == "Some description"
        assert x.tags == ["first tag"]

        dict_init = {"title": "Some title", "description": "Some description", "tags": ["first tag", "second"]}
        y = self.SimpleTestOutput(**dict_init)
        assert y.title == "Some title"
        assert y.description == "Some description"
        assert y.tags == ["first tag", "second"]

    def test_partial_initialization(self):
        # doesn't throw as list is by default not required (default: [])
        x = self.SimpleTestOutput(title="Some title", description=None)  # type: ignore
        assert x.title == "Some title"
        assert x.description is None
        assert x.tags == []

        # doesn't throw as optional type is by default None
        y = self.SimpleTestOutput(title="Some title", tags=["first tag"])  # type: ignore
        assert y.title == "Some title"
        assert y.description is None
        assert y.tags == ["first tag"]

        dict_init: dict = {"title": "Some title", "description": "desc"}
        z = self.SimpleTestOutput(**dict_init)
        assert z.title == "Some title"
        assert z.description == "desc"
        assert z.tags == []

        # title is required as it has no default value and is not optional
        with pytest.raises(ValueError):
            self.SimpleTestOutput(description="dd", tags=["x"])  # type: ignore

        with pytest.raises(ValueError):
            dict_init: dict = {"description": "desc"}
            self.SimpleTestOutput(**dict_init)

    def test_incorrect_type_initialization(self):
        with pytest.raises(LLMOutputFieldWrongType):
            self.SimpleTestOutput(title=1, description="Some description", tags=["first tag"])  # type: ignore

        with pytest.raises(LLMOutputFieldWrongType):
            self.SimpleTestOutput(title="Some title", description="Some description", tags="first tag")  # type: ignore

        with pytest.raises(LLMOutputFieldWrongType):
            self.SimpleTestOutput(title="Some title", description="Some description", tags=["first tag", 1])  # type: ignore

        with pytest.raises(TypeError):
            self.SimpleTestOutput(title=None, description="Some description", tags=["first tag"])  # type: ignore

        with pytest.raises(LLMOutputFieldWrongType):
            self.SimpleTestOutput(title="Some title", description=None, tags=["first tag", None])  # type: ignore

    def test_parameter_change(self):
        x = self.SimpleTestOutput(title="Some title", description="Some description", tags=["first tag"])
        assert x.title == "Some title"
        assert x.description == "Some description"
        assert x.tags == ["first tag"]

        x.title = "New title"
        x.description = None
        x.tags = ["first tag", "second tag"]

        assert x.title == "New title"
        assert x.description is None
        assert x.tags == ["first tag", "second tag"]

    # -

    class ExtendedTestOutput(BaseLLMResponse):
        title: str = ""
        description: int | None = None
        tags: list[str] = LLMArrayOutput((1, 3), lambda pos: f"Put the {pos.ordinal} tag here")

    def test_extended_correct_initialization(self):
        x = self.ExtendedTestOutput(title="Some title", description=1, tags=["first tag"])
        assert x.title == "Some title"
        assert x.description == 1
        assert x.tags == ["first tag"]

        dict_init = {"title": "Some title", "description": 5, "tags": ["first tag", "second"]}
        y = self.ExtendedTestOutput(**dict_init)
        assert y.title == "Some title"
        assert y.description == 5
        assert y.tags == ["first tag", "second"]

        z = self.ExtendedTestOutput(tags=["a", "b", "c"])
        assert z.title == ""
        assert z.description is None
        assert z.tags == ["a", "b", "c"]

    def test_extended_partial_initialization(self):
        with pytest.raises(ValueError):
            self.ExtendedTestOutput(title="Some title", description=None)  # type: ignore

        # not enough tags
        with pytest.raises(LLMOutputFieldInvalidLength):
            self.ExtendedTestOutput(tags=[])

        # too many tags
        with pytest.raises(LLMOutputFieldInvalidLength):
            self.ExtendedTestOutput(tags=["a", "b", "c", "d"])

    def test_extended_incorrect_type_initialization(self):
        with pytest.raises(LLMOutputFieldWrongType):
            dict_init = {"title": "Some title", "description": None, "tags": ["first tag", "second", None]}
            self.ExtendedTestOutput(**dict_init)

        with pytest.raises(LLMOutputFieldWrongType):
            self.ExtendedTestOutput(title="Some title", description="Some description", tags=["first tag"])  # type: ignore

        with pytest.raises(LLMOutputFieldWrongType):
            dict_init: dict = {"title": "Some title", "description": "desc"}
            self.ExtendedTestOutput(**dict_init)

    # -

    def test_incorrect_type_definition(self):
        with pytest.raises(TypeError):

            class A(BaseLLMResponse):
                title: str = 1  #  type: ignore

        with pytest.raises(TypeError):

            class B(BaseLLMResponse):
                title: str = None  # type: ignore

        with pytest.raises(TypeError):

            class C(BaseLLMResponse):
                test: Any

        with pytest.raises(TypeError):

            class D(BaseLLMResponse):
                z: str
                test: str | int

        with pytest.raises(TypeError):

            class E(BaseLLMResponse):
                z: str = LLMOutput("test", default=1)

        with pytest.raises(TypeError):

            class F(BaseLLMResponse):
                z: str = LLMOutput("test", default=None)

        # - matching wrong field description (array <-> non-array)

        with pytest.raises(TypeError):

            class H(BaseLLMResponse):
                z: str = LLMArrayOutput((1, 3), lambda pos: f"Put the {pos.ordinal} tag here")

        with pytest.raises(TypeError):

            class I(BaseLLMResponse):
                arr: list[str] = LLMOutput("test")

        # - arrays

        # with pytest.raises(TypeError):

        #     class J(BaseLLMResponse):
        #         arr: list[str] = LLMArrayOutput((1, 3), lambda _: "test")
