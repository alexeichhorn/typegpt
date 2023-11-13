from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from .exceptions import LLMOutputFieldInvalidLength, LLMOutputFieldMissing, LLMOutputFieldWrongType
from .fields import ClassPlaceholder, LLMArrayElementOutputInfo, LLMArrayOutputInfo, LLMFieldInfo, LLMOutputInfo
from .meta import LLMArrayElementMeta, LLMBaseMeta
from .parser import Parser

if TYPE_CHECKING:
    from inspect import Signature


class _InternalBaseLLMResponse:
    if TYPE_CHECKING:
        # populated by the metaclass (ClassPlaceholder used to prevent showing up as type suggestion)
        __fields__: ClassVar[dict[str, LLMFieldInfo]] = ClassPlaceholder(init=False, value={})
        __signature__: ClassVar["Signature"] = ClassPlaceholder(init=False)

    def __init__(self, **data: Any):
        # print(data)
        data = self._prepare_and_validate_dict(data)
        self.__dict__.update(data)

    # don't allow setting of fields that aren't defined in IDE
    if not TYPE_CHECKING:

        def __setattr__(self, __name: str, __value: Any) -> None:
            # print(f"Setting {__name} to {__value}")

            if __name.startswith("__"):
                super().__setattr__(__name, __value)
                return

            __value = self._prepare_and_validate_field(__name, __value)  # throws error if invalid

            super().__setattr__(__name, __value)

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_"))
        return f"{self.__class__.__name__}({attrs})"

    def _prepare_field_value(self, value: Any, _type: type) -> Any:
        """Converts single values from string to their type, otherwise leaves as is"""

        from .utils.type_checker import if_optional

        # unwrap optional type
        if optional_type := if_optional(_type):
            _type = optional_type

        if _type == str:
            return value

        if isinstance(value, _type):
            return value

        if isinstance(value, str):
            value = value.strip()
            if _type == int:
                try:
                    return int(value)
                except ValueError:
                    raise LLMOutputFieldWrongType(f'"{value}" is not a valid integer value')
            elif _type == float:
                try:
                    return float(value)
                except ValueError:
                    raise LLMOutputFieldWrongType(f'"{value}" is not a valid float value')
            elif _type == bool:
                if value.lower() in ("true", "yes", "1"):
                    return True
                elif value.lower() in ("false", "no", "0"):
                    return False
                else:
                    raise LLMOutputFieldWrongType(f'"{value}" is not a valid boolean value')

        return value

    def _prepare_and_validate_field(self, __name: str, __value: Any) -> Any:
        if __name not in self.__fields__:
            raise ValueError(f'"{self.__class__.__name__}" object has no field "{__name}"')

        from .utils.type_checker import array_item_type

        field_info = self.__fields__[__name]
        if isinstance(field_info.info, LLMOutputInfo):
            __value = self._prepare_field_value(__value, field_info.type_)

            if __value is None and field_info.info.required:
                raise TypeError(f'"{self.__class__.__name__}" field "{__name}" is required')
            if not isinstance(__value, field_info.type_):
                raise LLMOutputFieldWrongType(f'"{self.__class__.__name__}" field "{__name}" must be of type {field_info.type_}')

        elif isinstance(field_info.info, LLMArrayOutputInfo):
            item_type = array_item_type(field_info.type_)

            if not isinstance(__value, list):
                raise LLMOutputFieldWrongType(f'"{self.__class__.__name__}" field "{__name}" must be a list')
            if field_info.info.min_count is not None and len(__value) < field_info.info.min_count:
                raise LLMOutputFieldInvalidLength(
                    f'"{self.__class__.__name__}" field "{__name}" must have at least {field_info.info.min_count} items'
                )
            if field_info.info.max_count is not None and len(__value) > field_info.info.max_count:
                raise LLMOutputFieldInvalidLength(
                    f'"{self.__class__.__name__}" field "{__name}" must have at most {field_info.info.max_count} items'
                )

            __value = [self._prepare_field_value(v, item_type) for v in __value]
            if not all(isinstance(v, item_type) for v in __value):
                raise LLMOutputFieldWrongType(f'"{self.__class__.__name__}" field "{__name}" must be a list of type {field_info.type_}')

        elif isinstance(field_info.info, LLMArrayElementOutputInfo):
            __value = self._prepare_field_value(__value, field_info.type_)

            if __value is None and field_info.info.required:
                raise TypeError(f'"{self.__class__.__name__}" field "{__name}" is required')
            if not isinstance(__value, field_info.type_):
                raise LLMOutputFieldWrongType(f'"{self.__class__.__name__}" field "{__name}" must be of type {field_info.type_}')

        return __value

    def _prepare_and_validate_dict(self, __values: dict[str, Any]) -> dict[str, Any]:
        # check every field if it's valid
        for k, v in __values.items():
            __values[k] = self._prepare_and_validate_field(k, v)

        # check every field not in dict, if it's required
        for field in self.__fields__.values():
            if field.key in __values:
                continue

            if isinstance(field.info, LLMOutputInfo):
                if field.info.required:
                    raise ValueError(f'"{self.__class__.__name__}" field "{field.key}" is required')
                else:
                    __values[field.key] = field.info.default

            elif isinstance(field.info, LLMArrayOutputInfo):
                if field.info.min_count > 0:
                    raise ValueError(f'"{self.__class__.__name__}" field "{field.key}" requires at least {field.info.min_count} items')
                else:
                    __values[field.key] = []

            elif isinstance(field.info, LLMArrayElementOutputInfo):
                if field.info.required:
                    raise ValueError(f'"{self.__class__.__name__}" field "{field.key}" is required')
                else:
                    __values[field.key] = field.info.default

        return __values


class BaseLLMResponse(_InternalBaseLLMResponse, metaclass=LLMBaseMeta):
    if TYPE_CHECKING:
        # populated by the metaclass (ClassPlaceholder used to prevent showing up as type suggestion)
        __raw_completion__: str = ClassPlaceholder(init=False, value="")

    def _set_raw_completion(self, completion: str):
        self.__raw_completion__ = completion

    # - Parsing

    _Self = TypeVar("_Self", bound="BaseLLMResponse")  # backward compatibility for pre-Python 3.12

    @classmethod
    def parse_response(cls: type[_Self], response: str) -> _Self:
        return Parser(cls).parse(response)


# -


class BaseLLMArrayElement(_InternalBaseLLMResponse, metaclass=LLMArrayElementMeta):
    if TYPE_CHECKING:
        # populated by the metaclass (ClassPlaceholder used to prevent showing up as type suggestion)
        __fields__: ClassVar[dict[str, LLMFieldInfo]] = ClassPlaceholder(init=False, value={})
        __signature__: ClassVar["Signature"] = ClassPlaceholder(init=False)

    def _set_raw_completion(self, completion: str):
        pass  # ignored for array elements

    # - Parsing

    _Self = TypeVar("_Self", bound="BaseLLMArrayElement")  # backward compatibility for pre-Python 3.12

    @classmethod
    def parse_response(cls: type[_Self], response: str) -> _Self:
        return Parser(cls).parse(response)
