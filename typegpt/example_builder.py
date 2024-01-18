from typegpt.base import BaseLLMResponse
from typegpt.fields import LLMOutputInfo
from typegpt.utils.type_checker import if_response_type


class ExampleOutputFactory:
    def __init__(self, example: BaseLLMResponse):
        self.example = example

    def generate(self) -> str:
        lines: list[str] = []

        for field in self.example.__fields__.values():
            if isinstance(field.info, LLMOutputInfo):
                if field_type := if_response_type(field.type_):
                    subfields = list(field_type.__fields__.values())
                    print(subfields)

                else:
                    lines.append(f"{field.name}: {getattr(self.example, field.key)}")

        return "\n".join(lines)
