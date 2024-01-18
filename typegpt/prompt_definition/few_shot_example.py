from dataclasses import dataclass
from typing import Generic, TypeVar
from typegpt.base import BaseLLMResponse


_Output = TypeVar("_Output", bound=BaseLLMResponse)


@dataclass
class FewShotExample(Generic[_Output]):
    """
    A few shot example
    """

    input: str
    output: _Output
