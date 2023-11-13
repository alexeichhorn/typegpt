from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

from .utils.internal_types import _NoDefault, _NoDefaultType

T = TypeVar("T")


@dataclass
class ExamplePosition:
    _position: int

    def __str__(self) -> str:
        return str(self._position)

    @property
    def ordinal(self) -> str:
        return {1: "first", 2: "second", 3: "third"}.get(self._position, f"{self._position}th")


@dataclass
class LLMOutputInfo(Generic[T]):
    instruction: str
    default: T | _NoDefaultType
    required: bool
    multiline: bool


@dataclass
class LLMArrayOutputInfo(Generic[T]):
    instruction: Callable[[ExamplePosition], str]
    min_count: int
    max_count: int | None
    multiline: bool


@dataclass
class LLMArrayElementOutputInfo(Generic[T]):
    instruction: Callable[[ExamplePosition], str]
    default: T | _NoDefaultType
    required: bool
    multiline: bool


@dataclass
class LLMFieldInfo(Generic[T]):
    key: str
    name: str
    type_: type[T]
    info: LLMOutputInfo[T] | LLMArrayOutputInfo[T] | LLMArrayElementOutputInfo[T]


def LLMOutput(
    instruction: str,
    default: SupportedBaseTypes | None | _NoDefaultType = _NoDefault,
    # required: bool = True,
    multiline: bool = False,
) -> Any:
    return LLMOutputInfo(instruction=instruction, default=default, required=(default is _NoDefault), multiline=multiline)


def LLMArrayOutput(
    expected_count: int | None | tuple[int | None, int | None],
    instruction: Callable[[ExamplePosition], str],
    multiline: bool = False,
) -> Any:
    min_count, max_count = 0, None
    if isinstance(expected_count, tuple):
        min_count, max_count = expected_count
        min_count = min_count or 0
    elif expected_count is not None:
        min_count = expected_count
        max_count = expected_count

    return LLMArrayOutputInfo(instruction=instruction, min_count=min_count, max_count=max_count, multiline=multiline)


def LLMArrayElementOutput(
    instruction: Callable[[ExamplePosition], str],
    default: SupportedBaseTypes | None | _NoDefaultType = _NoDefault,
    multiline: bool = False,
) -> Any:
    return LLMArrayElementOutputInfo(instruction=instruction, default=default, required=(default is _NoDefault), multiline=multiline)


def ClassPlaceholder(init: bool, value: Any = None) -> Any:
    return value


if TYPE_CHECKING:
    from .utils.type_checker import SupportedBaseTypes, array_item_type
