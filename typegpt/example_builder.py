from typing import TypeVar

from typegpt.base import BaseLLMResponse
from typegpt.fields import LLMArrayElementOutputInfo, LLMArrayOutputInfo, LLMOutputInfo
from typegpt.utils.type_checker import SupportedBaseTypes, if_array_element_list_type, if_response_type, is_response_type
from typegpt.utils.utils import limit_newlines

BaseTypeT = TypeVar("BaseTypeT", bound=SupportedBaseTypes)


class ExampleOutputFactory:
    def __init__(self, example: BaseLLMResponse, name_prefixes: list[str] = []):
        self.example = example
        self.name_prefixes = name_prefixes

    def _transform_value(self, value: BaseTypeT) -> str | BaseTypeT:
        """Transforms single values into values that can be printed in the output prompt"""
        if isinstance(value, bool):
            return "true" if value else "false"
        return value

    def generate(self) -> str:
        lines: list[str] = []

        for field in self.example.__fields__.values():
            field_name = " ".join(self.name_prefixes + [field.name])

            if isinstance(field.info, LLMOutputInfo):
                value = getattr(self.example, field.key)

                if is_response_type(field.type_):
                    subelement_factory = ExampleOutputFactory(value, name_prefixes=self.name_prefixes + [field.name])
                    lines.append("")  # add a newline before each subelement
                    lines.append(subelement_factory.generate())

                else:
                    if value is None:
                        continue
                    value = self._transform_value(value)
                    lines.append(f"{field_name}: {value}")

            elif isinstance(field.info, LLMArrayOutputInfo):
                if if_array_element_list_type(field.type_):
                    lines.append("")
                    for i, subelement in enumerate(getattr(self.example, field.key)):
                        subelement_factory = ExampleOutputFactory(subelement, name_prefixes=self.name_prefixes + [field.name, str(i + 1)])

                        subelement_content = subelement_factory.generate()

                        # append newline for each entry that is complex enough (i.e. contains multiple lines)
                        if "\n" in subelement_content:
                            subelement_content += "\n"

                        lines.append(subelement_content)

                else:
                    for i, subelement in enumerate(getattr(self.example, field.key)):
                        lines.append(f"{field_name} {i + 1}: {self._transform_value(subelement)}")

            elif isinstance(field.info, LLMArrayElementOutputInfo):
                value = getattr(self.example, field.key)

                if is_response_type(field.type_):
                    subelement_factory = ExampleOutputFactory(value, name_prefixes=self.name_prefixes + [field.name])
                    lines.append("")
                    lines.append(subelement_factory.generate())

                else:
                    if value is None:
                        continue
                    value = self._transform_value(value)
                    lines.append(f"{field_name}: {value}")

        return limit_newlines("\n".join(lines).strip())
