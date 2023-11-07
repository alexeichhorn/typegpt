from __future__ import annotations

import re
from typing import TYPE_CHECKING, Generic, TypeVar

from .exceptions import LLMOutputFieldMissing, LLMOutputFieldWrongType
from .fields import LLMArrayOutputInfo, LLMFieldInfo, LLMOutputInfo
from .utils.utils import symmetric_strip

_Output = TypeVar("_Output", bound="BaseLLMResponse")


class Parser(Generic[_Output]):
    def __init__(self, output_type: type[_Output]):
        self.output_type = output_type
        self.fields = self.output_type.__fields__.values()

    def _regex_for_field(self, field: LLMFieldInfo) -> str:
        other_fields = [f for f in self.fields if f.key != field.key]
        other_field_names = ["\n" + f.name for f in other_fields]

        excluded_lookahead = other_field_names

        if not field.info.multiline:
            excluded_lookahead.append("\n")

        # also add current field if it's an array
        if isinstance(field.info, LLMArrayOutputInfo):
            excluded_lookahead.append("\n" + field.name)

        exclusion_cases_regex = "|".join(excluded_lookahead)
        if exclusion_cases_regex:
            exclusion_cases_regex = f"(?!{exclusion_cases_regex})"

        if isinstance(field.info, LLMOutputInfo):
            return rf"{field.name}: *\n?(?P<content>({exclusion_cases_regex}[\s\S])+)"
        elif isinstance(field.info, LLMArrayOutputInfo):
            if max_count := field.info.max_count:
                max_decimal_places = len(str(max_count))
                count_regex = f"\\d{{1,{max_decimal_places}}}"
            else:
                count_regex = "\\d{1,3}"
            return rf"{field.name} {count_regex}: ?(?P<content>({exclusion_cases_regex}[\s\S])+)"
        else:
            raise ValueError(f"Invalid field info type: {field.info}")

    def parse(self, response: str) -> _Output:
        field_values: dict[str, str | list[str]] = {}

        # preprocess full response
        raw_response = response
        response = symmetric_strip(response.strip(), ["'", '"', "`"])

        for field in self.fields:
            pattern = self._regex_for_field(field)

            if isinstance(field.info, LLMOutputInfo):
                match = re.search(pattern, response)
                if match:
                    field_values[field.key] = match.group("content").strip()
                else:
                    if field.info.required:
                        raise LLMOutputFieldMissing(f'Field "{field.name}" is missing')

            elif isinstance(field.info, LLMArrayOutputInfo):
                matches = re.finditer(pattern, response)
                items = [m.group("content").strip() for m in matches]
                items = [i for i in items if i]
                field_values[field.key] = items

        output = self.output_type(**field_values)
        output._set_raw_completion(raw_response)
        return output


if TYPE_CHECKING:
    from .base import BaseLLMResponse
