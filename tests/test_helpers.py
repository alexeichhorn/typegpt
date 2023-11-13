import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

from typing import Dict, List, Literal, Optional, Set, Tuple, Union

import pytest

from typegpt import BaseLLMArrayElement, BaseLLMResponse
from typegpt.utils.type_checker import *


class TestHelpers:
    class SomeClass:
        pass

    class SomeResponseOutput(BaseLLMResponse):
        x: int

    class SomeArrayElement(BaseLLMArrayElement):
        title: str

    def test_is_optional(self):
        assert is_optional(list[str]) == False
        assert is_optional(Optional[str]) == True
        assert is_optional(Union[str, None]) == True
        assert is_optional(str | None) == True
        assert is_optional(str) == False
        assert is_optional(list[float] | None) == True
        assert is_optional(Optional[list[float]]) == True
        assert is_optional(list[str | None]) == False
        assert is_optional(Optional[self.SomeClass]) == True
        assert is_optional(Optional[self.SomeResponseOutput]) == True
        assert is_optional(Optional[self.SomeArrayElement]) == True
        assert is_optional(self.SomeClass | None) == True
        assert is_optional(self.SomeResponseOutput | None) == True
        assert is_optional(self.SomeArrayElement | None) == True
        assert is_optional(self.SomeClass) == False
        assert is_optional(self.SomeResponseOutput) == False
        assert is_optional(self.SomeArrayElement) == False

    def test_is_array(self):
        assert is_array(list[str]) == True
        assert is_array(List[str]) == True
        assert is_array(list[list[str]]) == True
        assert is_array(List[List[str]]) == True
        assert is_array(list[int] | None) == True
        assert is_array(List[int] | None) == True
        assert is_array(Optional[list[int]]) == True
        assert is_array(list[str | None]) == True
        assert is_array(None | list[float]) == True
        assert is_array(str | None) == False
        assert is_array(Optional[str]) == False
        assert is_array(float) == False
        assert is_array(list[self.SomeClass]) == True
        assert is_array(List[self.SomeClass]) == True
        assert is_array(self.SomeClass) == False
        assert is_array(self.SomeResponseOutput) == False
        assert is_array(self.SomeArrayElement) == False

    def test_array_item_type(self):
        assert array_item_type(list[str]) == str
        assert array_item_type(List[str]) == str
        assert array_item_type(list[list[str]]) == list[str]
        assert array_item_type(List[List[str]]) == List[str]
        assert array_item_type(list[int] | None) == int
        assert array_item_type(List[int] | None) == int
        assert array_item_type(Optional[list[int]]) == int
        assert array_item_type(list[str | None]) == str | None
        assert array_item_type(None | list[float]) == float
        assert array_item_type(list[int] | list[str]) == int  # we currently only take first one

        assert array_item_type(list[self.SomeClass]) == self.SomeClass
        assert array_item_type(List[self.SomeClass]) == self.SomeClass
        assert array_item_type(list[self.SomeArrayElement]) == self.SomeArrayElement

        with pytest.raises(TypeError):
            array_item_type(str | None)

        with pytest.raises(TypeError):
            array_item_type(Optional[str])

        with pytest.raises(TypeError):
            array_item_type(float)

        with pytest.raises(TypeError):
            array_item_type(self.SomeClass)

    def test_is_supported_output_type(self):
        assert is_supported_output_type(bool) == True
        assert is_supported_output_type(int) == True
        assert is_supported_output_type(float) == True
        assert is_supported_output_type(str) == True
        assert is_supported_output_type(type(None)) == False
        assert is_supported_output_type(list) == False
        assert is_supported_output_type(set) == False
        assert is_supported_output_type(dict) == False
        assert is_supported_output_type(self.SomeClass) == False
        assert is_supported_output_type(self.SomeResponseOutput) == True
        assert is_supported_output_type(self.SomeArrayElement) == False

        assert is_supported_output_type(bool | None) == True
        assert is_supported_output_type(int | None) == True
        assert is_supported_output_type(float | None) == True
        assert is_supported_output_type(str | None) == True
        assert is_supported_output_type(self.SomeClass | None) == False
        assert is_supported_output_type(list | None) == False
        assert is_supported_output_type(set | None) == False
        assert is_supported_output_type(dict | None) == False
        assert is_supported_output_type(list[str] | None) == False
        assert is_supported_output_type(List[int] | None) == False
        assert is_supported_output_type(self.SomeResponseOutput | None) == True
        assert is_supported_output_type(self.SomeArrayElement | None) == False
        assert is_supported_output_type(list[self.SomeArrayElement] | None) == False
        assert is_supported_output_type(list[self.SomeResponseOutput] | None) == False

        assert is_supported_output_type(str | int | None) == False
        assert is_supported_output_type(str | int) == False
        assert is_supported_output_type(int | float) == False
        assert is_supported_output_type(self.SomeResponseOutput | str) == False

        assert is_supported_output_type(Optional[bool]) == True
        assert is_supported_output_type(Optional[int]) == True
        assert is_supported_output_type(Optional[float]) == True
        assert is_supported_output_type(Optional[str]) == True
        assert is_supported_output_type(Optional[self.SomeClass]) == False
        assert is_supported_output_type(Optional[self.SomeResponseOutput]) == True
        assert is_supported_output_type(Optional[self.SomeArrayElement]) == False
        assert is_supported_output_type(Optional[list]) == False
        assert is_supported_output_type(Optional[list[int]]) == False
        assert is_supported_output_type(Optional[list[self.SomeResponseOutput]]) == False
        assert is_supported_output_type(Optional[list[self.SomeArrayElement]]) == False

        assert is_supported_output_type(list[int]) == True
        assert is_supported_output_type(List[int]) == True
        assert is_supported_output_type(list[float]) == True
        assert is_supported_output_type(List[float]) == True
        assert is_supported_output_type(list[bool]) == True
        assert is_supported_output_type(List[bool]) == True
        assert is_supported_output_type(list[str]) == True
        assert is_supported_output_type(List[str]) == True
        assert is_supported_output_type(list[self.SomeClass]) == False
        assert is_supported_output_type(List[self.SomeClass]) == False
        assert is_supported_output_type(list[self.SomeResponseOutput]) == False
        assert is_supported_output_type(List[self.SomeResponseOutput]) == False
        assert is_supported_output_type(list[self.SomeArrayElement]) == True
        assert is_supported_output_type(List[self.SomeArrayElement]) == True
        assert is_supported_output_type(list[list[int]]) == False
        assert is_supported_output_type(List[List[int]]) == False
        assert is_supported_output_type(list[int | str]) == False
        assert is_supported_output_type(List[int | str]) == False

        assert is_supported_output_type(list[list[int]]) == False
        assert is_supported_output_type(list[int | None]) == False
        assert is_supported_output_type(List[str | None]) == False

        assert is_supported_output_type(set[str]) == False
        assert is_supported_output_type(set[float]) == False
        assert is_supported_output_type(Set[str]) == False
        assert is_supported_output_type(tuple[str, int]) == False
        assert is_supported_output_type(Tuple[str, str]) == False
        assert is_supported_output_type(dict[str, int]) == False
        assert is_supported_output_type(Dict[str, int]) == False
        assert is_supported_output_type(set[self.SomeArrayElement]) == False
        assert is_supported_output_type(dict[str, self.SomeResponseOutput]) == False
        assert is_supported_output_type(dict[str, self.SomeArrayElement]) == False

        assert is_supported_output_type(Union[str, int]) == False
        assert is_supported_output_type(Union[str, None]) == True
        assert is_supported_output_type(Union[str, int, None]) == False

        assert is_supported_output_type(Literal["a", "b", "c"]) == False  # type: ignore # TODO: support in future
