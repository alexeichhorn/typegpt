from .example_formatter import LimitedExampleListFormatter
from .fields import ExamplePosition, LLMArrayElementOutputInfo, LLMArrayOutputInfo, LLMFieldInfo, LLMOutputInfo
from .utils.type_checker import if_array_element_list_type, if_response_type


class OutputPromptFactory:
    def __init__(self, fields: list[LLMFieldInfo], threaten: bool = False, name_prefixes: list[str] = []):
        self.fields = fields
        self.threaten = threaten
        self.name_prefixes = name_prefixes

    def _generate_array(self, field: LLMFieldInfo, info: LLMArrayOutputInfo) -> str:
        is_unlimited = info.max_count is None
        max_count = info.max_count or 2

        if element_type := if_array_element_list_type(field.type_):
            subfields = list(element_type.__fields__.values())
            # subprompt_factory = OutputPromptFactory(subfields, name_prefixes=self.name_prefixes + [field.name])
            # examples = [subprompt_factory._generate_schema(offset=i + 1) for i in range(max_count)]
            examples = [
                OutputPromptFactory(subfields, name_prefixes=self.name_prefixes + [field.name + f" {i+1}"])._generate_schema(offset=i + 1)
                for i in range(max_count)
            ]
        else:
            field_name = " ".join(self.name_prefixes + [field.name])
            examples = [f"{field_name} {i+1}: <{info.instruction(ExamplePosition(i+1))}>" for i in range(max_count)]

        if is_unlimited:
            return LimitedExampleListFormatter(max_count, "\n").format(examples) + "\n...\n"
        else:
            return LimitedExampleListFormatter(3, separator="\n").format(examples) + "\n"

    def _generate_schema(self, offset: int = 0) -> str:
        prompt = ""

        for field in self.fields:
            field_name = " ".join(self.name_prefixes + [field.name])
            # if offset > 0:
            #     field_name += f" {offset}"

            if isinstance(field.info, LLMOutputInfo):
                if field_type := if_response_type(field.type_):
                    subfields = list(field_type.__fields__.values())
                    prompt = prompt.rstrip() + "\n\n"
                    prompt += (
                        OutputPromptFactory(subfields, name_prefixes=self.name_prefixes + [field.name])._generate_schema(offset=offset)
                        + "\n"
                    )

                else:
                    field_prompt = f"{field_name}: <{field.info.instruction}>"
                    prompt += field_prompt

            elif isinstance(field.info, LLMArrayElementOutputInfo):
                if field_type := if_response_type(field.type_):
                    subfields = list(field_type.__fields__.values())
                    prompt = prompt.rstrip() + "\n\n"
                    prompt += (
                        OutputPromptFactory(subfields, name_prefixes=self.name_prefixes + [field.name])._generate_schema(offset=offset)
                        + "\n"
                    )

                else:
                    field_prompt = f"{field_name}: <{field.info.instruction(ExamplePosition(offset))}>"
                    prompt += field_prompt

            elif isinstance(field.info, LLMArrayOutputInfo):
                prompt += self._generate_array(field, field.info)

            prompt += "\n"

        return prompt.rstrip()

    def generate(self) -> str:
        if self.threaten:
            prompt = "Always return the answer in the following format, otherwise people might die and you are the only one to blame:"
        else:
            prompt = "Always return the answer in the following format:"

        prompt += '\n"""\n'

        prompt += self._generate_schema() + "\n"

        prompt += '"""'

        return prompt
