from .example_formatter import LimitedExampleListFormatter
from .fields import ExamplePosition, LLMArrayOutputInfo, LLMFieldInfo, LLMOutputInfo


class OutputPromptFactory:
    def __init__(self, fields: list[LLMFieldInfo], threaten: bool = False):
        self.fields = fields
        self.threaten = threaten

    def _generate_array(self, field: LLMFieldInfo, info: LLMArrayOutputInfo) -> str:
        is_unlimited = info.max_count is None
        max_count = info.max_count or 2

        examples = [f"{field.name} {i+1}: <{info.instruction(ExamplePosition(i+1))}>" for i in range(max_count)]

        if is_unlimited:
            return LimitedExampleListFormatter(max_count, "\n").format(examples) + "\n...\n"
        else:
            return LimitedExampleListFormatter(3, separator="\n").format(examples) + "\n"

    def generate(self) -> str:
        if self.threaten:
            prompt = "Always return the answer in the following format, otherwise people might die and you are the only one to blame:"
        else:
            prompt = "Always return the answer in the following format:"

        prompt += '\n"""\n'

        for field in self.fields:
            if isinstance(field.info, LLMOutputInfo):
                field_prompt = f"{field.name}: <{field.info.instruction}>"
                prompt += field_prompt

            elif isinstance(field.info, LLMArrayOutputInfo):
                prompt += self._generate_array(field, field.info)

            prompt += "\n"

        prompt = prompt.rstrip() + "\n"

        prompt += '"""'

        return prompt
