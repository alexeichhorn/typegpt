from __future__ import annotations

import inspect
from abc import ABCMeta
from typing import TYPE_CHECKING, Any, Callable

import inflect
from typing_extensions import dataclass_transform, override

from .fields import (
    ClassPlaceholder,
    ExamplePosition,
    LLMArrayElementOutput,
    LLMArrayElementOutputInfo,
    LLMArrayOutput,
    LLMArrayOutputInfo,
    LLMFieldInfo,
    LLMOutput,
    LLMOutputInfo,
)
from .helper import ClassAttribute, generate_model_signature
from .utils.internal_types import _NoDefault

inflect_engine = inflect.engine()


class LLMMeta(ABCMeta):
    @staticmethod
    def _is_array_element() -> bool:
        return False

    def __new__(mcls: type, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs: Any) -> "LLMMeta":
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)

        annotations: dict[str, type] = namespace.get("__annotations__", {})

        fields: dict[str, LLMFieldInfo] = {}

        for field_name, field_type in annotations.items():
            from .utils.type_checker import is_array, is_optional, is_supported_output_type

            namespace_field = namespace.get(field_name, _NoDefault)
            is_array_type = is_array(field_type)

            if not is_supported_output_type(field_type):
                raise TypeError(f"Field {field_name} has unsupported type {field_type}")

            if isinstance(namespace_field, LLMOutputInfo):
                if is_array_type:
                    raise TypeError(f"Field {field_name} is not an array, but has `LLMArrayOutput` annotation. Use `LLMOutput` instead")
                if cls._is_array_element():
                    raise TypeError(
                        f"Field {field_name} is defined as `LLMOutput` in a class that subclasses `BaseLLMArrayElement`. Use `LLMArrayElementOutput` instead"
                    )
                displayed_name = LLMMeta.generate_field_name(field_name)
                namespace[field_name] = namespace_field.default
                fields[field_name] = LLMFieldInfo(name=displayed_name, key=field_name, type_=field_type, info=namespace_field)
            elif isinstance(namespace_field, LLMArrayOutputInfo):
                if not is_array_type:
                    raise TypeError(f"Field {field_name} is an array, but has normal `LLMOutput` annotation. Use `LLMArrayOutput` instead")
                displayed_name = LLMMeta.generate_field_name(field_name, is_array=True)
                namespace[field_name] = []
                fields[field_name] = LLMFieldInfo(name=displayed_name, key=field_name, type_=field_type, info=namespace_field)
            elif isinstance(namespace_field, LLMArrayElementOutputInfo):
                if not cls._is_array_element():
                    raise TypeError(
                        f"Field {field_name} is defined as `LLMArrayElementOutput` in a class that does not subclass `BaseLLMArrayElement`"
                    )
                displayed_name = LLMMeta.generate_field_name(field_name)
                namespace[field_name] = namespace_field.default
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

                    if cls._is_array_element():
                        field = LLMArrayElementOutputInfo(
                            instruction=LLMMeta.generate_default_array_element_instruction(displayed_name),
                            default=default,
                            required=(default is _NoDefault),
                            multiline=False,
                        )
                    else:
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
            if inspect.isclass(value):  # other classes defined in the same namespace is allowed
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
    def generate_default_array_element_instruction(name: str) -> Callable[[ExamplePosition], str]:
        return lambda position: f"Put the {position.ordinal} {name.lower()} here"

    @staticmethod
    def generate_default_array_instruction(name: str) -> Callable[[ExamplePosition], str]:
        # singular_word = inflect_engine.singular_noun(name.lower())
        singular_word = name.lower()
        return lambda position: f"Put the {position.ordinal} {singular_word} here"

    # - Verification

    @staticmethod
    def _verify_and_fix_field_info(field: LLMFieldInfo) -> LLMFieldInfo:
        """@throws ValueError, TypeError"""

        from .utils.type_checker import is_optional

        # TODO: check types and more

        if isinstance(field.info, LLMOutputInfo) or isinstance(field.info, LLMArrayElementOutputInfo):
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


# -


@dataclass_transform(kw_only_default=True, field_specifiers=(LLMArrayOutput, LLMOutput, ClassPlaceholder))
class LLMBaseMeta(LLMMeta):
    ...


@dataclass_transform(kw_only_default=True, field_specifiers=(LLMArrayOutput, LLMArrayElementOutput, ClassPlaceholder))
class LLMArrayElementMeta(LLMMeta):
    @override
    @staticmethod
    def _is_array_element() -> bool:
        return True
