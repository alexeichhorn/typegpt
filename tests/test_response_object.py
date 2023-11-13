import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

from typing import Any, List, Literal, Optional

import pytest

from typegpt import BaseLLMResponse, LLMArrayOutput, LLMOutput, BaseLLMArrayElement, LLMArrayElementOutput
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

    class SubtypeTestOutput(BaseLLMResponse):
        class Item(BaseLLMArrayElement):
            subtitle: str
            description: str = LLMArrayElementOutput(lambda pos: f"Put the {pos.ordinal} item description here")
            abstract: str = LLMArrayElementOutput(lambda _: "...", default="default abstract")

        class DirectItem(BaseLLMResponse):
            subitem_title: str

        title: str
        strings: list[str]
        items: list[Item]
        subitem: DirectItem
        optional_subitem: DirectItem | None = None

    def test_subtype_correct_initialization(self):
        x = self.SubtypeTestOutput(
            title="Some title", strings=["a", "b", "c"], items=[], subitem=self.SubtypeTestOutput.DirectItem(subitem_title="sub")
        )
        assert x.title == "Some title"
        assert x.strings == ["a", "b", "c"]
        assert x.items == []
        assert x.subitem.subitem_title == "sub"
        assert x.optional_subitem is None

        y = self.SubtypeTestOutput(
            title="T",
            strings=["a"],
            items=[self.SubtypeTestOutput.Item(subtitle="sub", description="desc")],
            subitem=self.SubtypeTestOutput.DirectItem(subitem_title="sub2"),
            optional_subitem=self.SubtypeTestOutput.DirectItem(subitem_title="sub3"),
        )
        assert y.title == "T"
        assert y.strings == ["a"]
        assert len(y.items) == 1
        assert y.items[0].subtitle == "sub"
        assert y.items[0].description == "desc"
        assert y.items[0].abstract == "default abstract"
        assert y.subitem.subitem_title == "sub2"
        assert y.optional_subitem is not None
        assert y.optional_subitem.subitem_title == "sub3"

        dict_init = {
            "title": "b",
            "strings": ["a"],
            "items": [self.SubtypeTestOutput.Item(subtitle="sub", description="...", abstract="something")],
            "subitem": self.SubtypeTestOutput.DirectItem(subitem_title="sub2"),
        }
        z = self.SubtypeTestOutput(**dict_init)
        assert z.title == "b"
        assert z.strings == ["a"]
        assert len(z.items) == 1
        assert z.items[0].subtitle == "sub"
        assert z.items[0].description == "..."
        assert z.items[0].abstract == "something"
        assert z.subitem.subitem_title == "sub2"
        assert z.optional_subitem is None

    def test_subtype_partial_initialization(self):
        with pytest.raises(TypeError):
            self.SubtypeTestOutput(title="Some title", strings=["a", "b", "c"], items=[], subitem=None)  # type: ignore

        with pytest.raises(ValueError):
            self.SubtypeTestOutput.DirectItem()  # type: ignore

        with pytest.raises(ValueError):
            self.SubtypeTestOutput.DirectItem(subitem_title="a", description="b")  # type: ignore

        # one parameter too little
        with pytest.raises(ValueError):
            self.SubtypeTestOutput.Item(subtitle="a")  # type: ignore

        # one parameter too much
        with pytest.raises(ValueError):
            self.SubtypeTestOutput.Item(subtitle="a", description="b", abstract="c", extra="d")  # type: ignore

    def test_subtype_incorrect_type_initialization(self):
        with pytest.raises(LLMOutputFieldWrongType):
            self.SubtypeTestOutput(title="Some title", strings=["a"], items=[], subitem="some string")  # type: ignore

        class IncorrectDirectItem(BaseLLMResponse):
            subitem_title: int

        with pytest.raises(LLMOutputFieldWrongType):
            self.SubtypeTestOutput(title="Some title", strings=["a"], items=[], subitem=IncorrectDirectItem(subitem_title=1))  # type: ignore

        class IncorrectItem(BaseLLMArrayElement):
            subtitle: int

        with pytest.raises(LLMOutputFieldWrongType):
            self.SubtypeTestOutput(title="Some title", strings=["a"], items=[IncorrectItem(subtitle=1)], subitem=self.SubtypeTestOutput.DirectItem(subitem_title="sub2"))  # type: ignore

        # sub-items directly

        with pytest.raises(LLMOutputFieldWrongType):
            self.SubtypeTestOutput.DirectItem(subitem_title=1)  # type: ignore

        with pytest.raises(LLMOutputFieldWrongType):
            self.SubtypeTestOutput.Item(subtitle="a", description=1)  # type: ignore

        with pytest.raises(LLMOutputFieldWrongType):
            self.SubtypeTestOutput.Item(subtitle="a", description="b", abstract=1)  # type: ignore

        with pytest.raises(LLMOutputFieldWrongType):
            self.SubtypeTestOutput.Item(subtitle="a", description=self.SubtypeTestOutput.Item(subtitle="a", description="b"))  # type: ignore

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

        with pytest.raises(ValueError):

            class G(BaseLLMResponse):
                z = ""

        # - matching wrong field description (array <-> non-array)

        with pytest.raises(TypeError):

            class H(BaseLLMResponse):
                z: str = LLMArrayOutput((1, 3), lambda pos: f"Put the {pos.ordinal} tag here")

        with pytest.raises(TypeError):

            class I(BaseLLMResponse):
                arr: list[str] = LLMOutput("test")

        # - matching wrong field description (element <-> array element <-> array)

        with pytest.raises(TypeError):
            # not allowed to use `LLMArrayElementOutput` on normal BaseLLMResponse
            class J(BaseLLMResponse):
                arr: list[str] = LLMArrayElementOutput(lambda pos: f"Put the {pos.ordinal} tag here")

        with pytest.raises(TypeError):
            # not allowed to use `LLMOutput` on BaseLLMArrayElement
            class K(BaseLLMArrayElement):
                arr: str = LLMOutput("test")

        class TestArrayElement(BaseLLMArrayElement):
            x: int

        class TestOutput(BaseLLMResponse):
            t: str

        with pytest.raises(TypeError):
            # should be normal LLMArrayOutput
            class L(BaseLLMResponse):
                arr: list[TestArrayElement] = LLMArrayElementOutput(lambda pos: f"Put the {pos.ordinal} tag here")

        with pytest.raises(TypeError):
            # should be LLMArrayElementOutput
            class M(BaseLLMArrayElement):
                arr: TestOutput = LLMOutput("test")

        with pytest.raises(TypeError):
            # should be LLMArrayElementOutput (since not a array)
            class N(BaseLLMArrayElement):
                arr: TestOutput = LLMArrayOutput((1, 3), lambda pos: f"Put the {pos.ordinal} tag here")

        with pytest.raises(TypeError):
            # not allowed to use normal LLMOutput in list
            class O(BaseLLMArrayElement):
                arr: list[TestOutput] = LLMArrayOutput((1, 3), lambda pos: f"Put the {pos.ordinal} tag here")

        # - arrays

        # with pytest.raises(TypeError):

        #     class J(BaseLLMResponse):
        #         arr: list[str] = LLMArrayOutput((1, 3), lambda _: "test")
