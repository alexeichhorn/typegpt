import keyword
from itertools import islice
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from .fields import LLMFieldInfo, LLMOutputInfo

if TYPE_CHECKING:
    from inspect import Signature

T = TypeVar("T")


def is_valid_identifier(identifier: str) -> bool:
    """
    Checks that a string is a valid identifier and not a Python keyword. (from Pydantic)
    :param identifier: The identifier to test.
    :return: True if the identifier is valid.
    """
    return identifier.isidentifier() and not keyword.iskeyword(identifier)


def generate_model_signature(init: Callable[..., None], fields: dict[str, LLMFieldInfo]) -> "Signature":
    """
    Generate signature for model based on its fields (mostly from Pydantic)
    """
    from inspect import Parameter, Signature, signature

    present_params = signature(init).parameters.values()
    merged_params: dict[str, Parameter] = {}
    var_kw = None
    use_var_kw = False

    for param in islice(present_params, 1, None):  # skip self arg
        if param.kind is param.VAR_KEYWORD:
            var_kw = param
            continue
        merged_params[param.name] = param

    if var_kw:  # if custom init has no var_kw, fields which are not declared in it cannot be passed through
        for field_name, field in fields.items():
            if field_name in merged_params:
                continue
            elif not is_valid_identifier(field_name):
                if is_valid_identifier(field_name):
                    param_name = field_name
                else:
                    use_var_kw = True
                    continue

            # TODO: replace annotation with actual expected types once #1055 solved
            if isinstance(field.info, LLMOutputInfo) and field.info.required:
                kwargs = {"default": field.info.default}
            else:
                kwargs = {}
            merged_params[field_name] = Parameter(field_name, Parameter.KEYWORD_ONLY, annotation=field.type_, **kwargs)

    if var_kw and use_var_kw:
        # Make sure the parameter for extra kwargs
        # does not have the same name as a field
        default_model_signature = [
            ("self", Parameter.POSITIONAL_OR_KEYWORD),
            ("data", Parameter.VAR_KEYWORD),
        ]
        if [(p.name, p.kind) for p in present_params] == default_model_signature:
            # if this is the standard model signature, use extra_data as the extra args name
            var_kw_name = "extra_data"
        else:
            # else start from var_kw
            var_kw_name = var_kw.name

        # generate a name that's definitely unique
        while var_kw_name in fields:
            var_kw_name += "_"
        merged_params[var_kw_name] = var_kw.replace(name=var_kw_name)

    return Signature(parameters=list(merged_params.values()), return_annotation=None)


def is_valid_field_name(name: str) -> bool:
    return not name.startswith("_")


# region - Class Attribute (from Pydantic)

if TYPE_CHECKING:

    def ClassAttribute(name: str, value: T) -> T:
        ...

else:

    class ClassAttribute:
        """Hide class attribute from its instances."""

        __slots__ = "name", "value"

        def __init__(self, name: str, value: Any) -> None:
            self.name = name
            self.value = value

        def __get__(self, instance: Any, owner: type[Any]) -> None:
            if instance is None:
                return self.value
            raise AttributeError(f"{self.name!r} attribute of {owner.__name__!r} is class-only")


# endregion
