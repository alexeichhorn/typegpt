from abc import ABCMeta
from typing import Any, Callable

import inflect
from typing_extensions import dataclass_transform

from .fields import ClassPlaceholder, ExamplePosition, LLMArrayOutput, LLMArrayOutputInfo, LLMFieldInfo, LLMOutput, LLMOutputInfo
from .helper import ClassAttribute, generate_model_signature
from .utils.internal_types import _NoDefault
from .utils.type_checker import is_array, is_optional, is_supported_output_type

inflect_engine = inflect.engine()


@dataclass_transform(kw_only_default=True, field_specifiers=(LLMArrayOutput, LLMOutput, ClassPlaceholder))
class LLMMeta(ABCMeta):
    def __new__(mcls: type, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs: Any) -> "LLMMeta":
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        # print(name)
        # print(namespace)
        annotations: dict[str, type] = namespace.get("__annotations__", {})

        fields: dict[str, LLMFieldInfo] = {}

        for field_name, field_type in annotations.items():
            namespace_field = namespace.get(field_name, _NoDefault)
            is_array_type = is_array(field_type)

            if not is_supported_output_type(field_type):
                raise TypeError(f"Field {field_name} has unsupported type {field_type}")

            if isinstance(namespace_field, LLMOutputInfo):
                if is_array_type:
                    raise TypeError(f"Field {field_name} is not an array, but has `LLMArrayOutput` annotation. Use `LLMOutput` instead")
                displayed_name = LLMMeta.generate_field_name(field_name)
                namespace[field_name] = namespace_field.default
                fields[field_name] = LLMFieldInfo(name=displayed_name, key=field_name, type_=field_type, info=namespace_field)
            elif isinstance(namespace_field, LLMArrayOutputInfo):
                if not is_array_type:
                    raise TypeError(f"Field {field_name} is an array, but has normal `LLMOutput` annotation. Use `LLMArrayOutput` instead")
                displayed_name = LLMMeta.generate_field_name(field_name, is_array=True)
                namespace[field_name] = []
                fields[field_name] = LLMFieldInfo(name=displayed_name, key=field_name, type_=field_type, info=namespace_field)
            else:
                if is_array_type:
                    displayed_name = LLMMeta.generate_field_name(field_name, is_array=True)
                    field = LLMArrayOutputInfo(
                        instruction=LLMMeta.generate_default_array_instruction(displayed_name),
                        min_count=0,
                        max_count=None,
                        multiline=False,
                    )
                else:
                    displayed_name = LLMMeta.generate_field_name(field_name)
                    default = namespace_field  # or (None if is_optional(field_type) else _NoDefault)
                    if is_optional(field_type) and default is _NoDefault:
                        default = None

                    field = LLMOutputInfo(
                        instruction=LLMMeta.generate_default_instruction(displayed_name, field_type),
                        default=default,
                        required=(default is _NoDefault),
                        multiline=False,
                    )
                fields[field_name] = LLMFieldInfo(name=displayed_name, key=field_name, type_=field_type, info=field)

            fields[field_name] = LLMMeta._verify_and_fix_field_info(fields[field_name])

            cls.__fields__ = fields

        for var_name, value in namespace.items():
            if var_name.startswith("__"):
                continue
            if var_name.startswith("_BaseLLMResponse_"):
                continue
            if var_name in (
                "_Self",
                "parse_response",
                "_prepare_and_validate_field",
                "_prepare_and_validate_dict",
                "_prepare_field_value",
                "_set_raw_completion",
            ):
                continue
            if not var_name in fields:
                raise ValueError(f"Field {var_name} has no type annotation")

        cls.__signature__ = ClassAttribute("__signature__", generate_model_signature(cls.__init__, fields))

        return cls

    @staticmethod
    def generate_field_name(key: str, is_array: bool = False) -> str:
        name = key.replace("_", " ")
        if is_array:
            if singular_name := inflect_engine.singular_noun(name):
                name = singular_name
        return name.upper()

    @staticmethod
    def generate_default_instruction(name: str, _type: type) -> str:
        if _type is bool:
            return f"'true' if {name.lower()}, else 'false'"
        return f"Put the {name.lower()} here"

    @staticmethod
    def generate_default_array_instruction(name: str) -> Callable[[ExamplePosition], str]:
        # singular_word = inflect_engine.singular_noun(name.lower())
        singular_word = name.lower()
        return lambda position: f"Put the {position.ordinal} {singular_word} here"

    # - Verification

    # def _prepare_and_validate_field(self, __name: str, __value: Any) -> Any:
    #     if __name not in self.__fields__:
    #         raise ValueError(f'"{self.__class__.__name__}" object has no field "{__name}"')

    #     field_info = self.__fields__[__name]
    #     if isinstance(field_info.info, LLMOutputInfo):
    #         __value = self._prepare_field_value(__value, field_info.type_)

    #         if __value is None and field_info.info.required:
    #             raise TypeError(f'"{self.__class__.__name__}" field "{__name}" is required')
    #         if not isinstance(__value, field_info.type_):
    #             raise TypeError(f'"{self.__class__.__name__}" field "{__name}" must be of type {field_info.type_}')

    #     elif isinstance(field_info.info, LLMArrayOutputInfo):
    #         item_type = array_item_type(field_info.type_)

    #         if not isinstance(__value, list):
    #             raise TypeError(f'"{self.__class__.__name__}" field "{__name}" must be a list')
    #         if field_info.info.min_count is not None and len(__value) < field_info.info.min_count:
    #             raise ValueError(f'"{self.__class__.__name__}" field "{__name}" must have at least {field_info.info.min_count} items')
    #         if field_info.info.max_count is not None and len(__value) > field_info.info.max_count:
    #             raise ValueError(f'"{self.__class__.__name__}" field "{__name}" must have at most {field_info.info.max_count} items')

    #         __value = [self._prepare_field_value(v, item_type) for v in __value]
    #         if not all(isinstance(v, item_type) for v in __value):
    #             raise TypeError(f'"{self.__class__.__name__}" field "{__name}" must be a list of type {field_info.type_}')

    #     return __value

    @staticmethod
    def _verify_and_fix_field_info(field: LLMFieldInfo) -> LLMFieldInfo:
        """@throws ValueError, TypeError"""

        # TODO: check types and more

        if isinstance(field.info, LLMOutputInfo):
            if field.info.required and field.info.default is not _NoDefault:
                raise ValueError(f'"{field.key}" is required but has default value')

            if field.info.default is not _NoDefault:
                if field.info.default == _NoDefault:
                    raise ValueError(f'"{field.key}" has invalid default value')
                if not isinstance(field.info.default, field.type_):
                    raise TypeError(f'"{field.key}" default value ({field.info.default}) must be of specified type {field.type_}')

            elif is_optional(field.type_):  # optional, but no default
                field.info.default = None
                field.info.required = False
                # raise ValueError(f'"{field.key}" is optional but has no default value. Use e.g. `None` as default value')

        # TODO: arrays

        return field
