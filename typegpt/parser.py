from __future__ import annotations

import re
from typing import TYPE_CHECKING, Generic, TypeVar

from .exceptions import LLMOutputFieldMissing, LLMOutputFieldWrongType
from .fields import LLMArrayOutputInfo, LLMFieldInfo, LLMOutputInfo, LLMArrayElementOutputInfo
from .utils.utils import symmetric_strip

_Output = TypeVar("_Output", bound="BaseLLMResponse | BaseLLMArrayElement")


class Parser(Generic[_Output]):
    def __init__(self, output_type: type[_Output]):
        self.output_type = output_type
        self.fields = self.output_type.__fields__.values()

    def _regex_for_field(self, field: LLMFieldInfo) -> str:
        from .utils.type_checker import if_response_type, is_response_type, is_array_element_list_type, if_array_element_list_type

        other_fields = [f for f in self.fields if f.key != field.key]
        other_field_names = ["\n" + f.name for f in other_fields]

        excluded_lookahead = other_field_names

        if not field.info.multiline and not is_response_type(field.type_) and not is_array_element_list_type(field.type_):
            excluded_lookahead.append("\n")

        # also add current field if it's an array
        if isinstance(field.info, LLMArrayOutputInfo) or is_response_type(field.type_):
            excluded_lookahead.append("\n" + field.name)

        exclusion_cases_regex = "|".join(excluded_lookahead)
        if exclusion_cases_regex:
            exclusion_cases_regex = f"(?!{exclusion_cases_regex})"

        if isinstance(field.info, LLMOutputInfo) or isinstance(field.info, LLMArrayElementOutputInfo):
            if field_type := if_response_type(field.type_):
                return rf"(?:^|\n){field.name} (?P<subfield_name>((?!:|\n)[\S ])+): ?(?P<content>({exclusion_cases_regex}[\s\S])+)"
            else:
                return rf"(?:^|\n){field.name}: *\n?(?P<content>({exclusion_cases_regex}[\s\S])+)"

        elif isinstance(field.info, LLMArrayOutputInfo):
            if max_count := field.info.max_count:
                max_decimal_places = len(str(max_count))
                count_regex = f"\\d{{1,{max_decimal_places}}}"
            else:
                count_regex = "\\d{1,3}"

            if field_type := if_array_element_list_type(field.type_):
                return rf"(?:^|\n){field.name} (?P<i>{count_regex}) (?P<subfield_name>((?!:|\n)[\S ])+): ?(?P<content>({exclusion_cases_regex}[\s\S])+)"
            else:
                return rf"(?:^|\n){field.name} {count_regex}: ?(?P<content>({exclusion_cases_regex}[\s\S])+)"

        else:
            raise ValueError(f"Invalid field info type: {field.info}")

    def parse(self, response: str) -> _Output:
        from .utils.type_checker import if_response_type, if_array_element_list_type

        field_values: dict[str, str | list[str] | BaseLLMResponse | list[BaseLLMArrayElement]] = {}

        # preprocess full response
        raw_response = response
        response = symmetric_strip(response.strip(), ["'", '"', "`"])

        for field in self.fields:
            pattern = self._regex_for_field(field)

            if isinstance(field.info, LLMOutputInfo) or isinstance(field.info, LLMArrayElementOutputInfo):
                if field_type := if_response_type(field.type_):
                    matches = re.finditer(pattern, response)
                    inner_response = "\n".join(f"{m.group('subfield_name')}: {m.group('content')}" for m in matches)

                    if field.info.required:
                        field_values[field.key] = field_type.parse_response(inner_response)
                    else:
                        try:
                            field_values[field.key] = field_type.parse_response(inner_response)
                        except:
                            pass

                else:
                    match = re.search(pattern, response)
                    if match:
                        field_values[field.key] = symmetric_strip(match.group("content").strip(), ["'", '"', "`"]).strip()
                    else:
                        if field.info.required:
                            raise LLMOutputFieldMissing(f'Field "{field.name}" is missing in {self.output_type.__name__}')

            elif isinstance(field.info, LLMArrayOutputInfo):
                if field_type := if_array_element_list_type(field.type_):
                    matches = re.finditer(pattern, response)
                    inner_responses: dict[int, str] = {}
                    for m in matches:
                        i = int(m.group("i"))
                        if not i in inner_responses:
                            inner_responses[i] = ""
                        inner_responses[i] += "\n" + m.group("subfield_name") + ": " + m.group("content")

                    # sort by index
                    inner_responses = dict(sorted(inner_responses.items(), key=lambda x: x[0]))

                    array_items: list[BaseLLMArrayElement] = []
                    for i, inner_response in inner_responses.items():
                        item = field_type.parse_response(inner_response)
                        array_items.append(item)

                    field_values[field.key] = array_items

                else:
                    matches = re.finditer(pattern, response)
                    items: list[str] = [m.group("content").strip() for m in matches]
                    items = [symmetric_strip(item, ["'", '"', "`"]).strip() for item in items]
                    items = [i for i in items if i]
                    field_values[field.key] = items

        output = self.output_type(**field_values)
        output._set_raw_completion(raw_response)
        return output


if TYPE_CHECKING:
    from .base import BaseLLMResponse, BaseLLMArrayElement
