from types import UnionType
from typing import Optional, Union, get_args, get_origin


SupportedBaseTypes = str | int | float | bool


def is_optional(_type: type) -> bool:
    origin_type = get_origin(_type)
    if origin_type is Optional:
        return True
    elif origin_type is Union or origin_type is UnionType:
        return type(None) in get_args(_type)
    return False


def is_array(_type: type) -> bool:
    origin_type = get_origin(_type)
    if origin_type is list:
        return True
    elif origin_type is Union or origin_type is UnionType:
        return any([is_array(t) for t in get_args(_type)])
    return False


def is_supported_base_type(_type: type) -> bool:
    try:
        return issubclass(_type, SupportedBaseTypes)
    except TypeError:
        return False


def is_supported_output_type(_type: type) -> bool:
    origin_type = get_origin(_type)
    if origin_type is None:
        return issubclass(_type, SupportedBaseTypes)
    elif origin_type is Optional:
        return is_supported_base_type(get_args(_type)[0])
    elif origin_type is list:
        return is_supported_base_type(get_args(_type)[0])
    elif origin_type is Union or origin_type is UnionType:
        union_types: tuple[type, ...] = get_args(_type)
        if len(union_types) == 2 and type(None) in union_types:
            union_types = tuple(t for t in union_types if t is not type(None))  # remove None
            return len(union_types) == 1 and is_supported_base_type(union_types[0])

    return False


def array_item_type(_type: type) -> type:
    origin_type = get_origin(_type)
    if origin_type is list:
        return get_args(_type)[0]
    elif origin_type is Union or origin_type is UnionType:
        item_types: list[type] = []
        for t in get_args(_type):
            try:
                item_types.append(array_item_type(t))
            except TypeError:
                pass

        if not item_types:
            raise TypeError(f"Type {_type} is not an array type")

        return item_types[0]
    raise TypeError(f"Type {_type} is not an array type")
