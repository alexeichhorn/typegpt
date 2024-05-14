import re


def symmetric_strip(content: str, chars: list[str | tuple[str, str]]) -> str:
    """Strips the given chars from the beginning and end of the string, but only if they are present on both sides"""
    result = content
    did_strip = True
    while did_strip:
        did_strip = False

        for char in chars:
            start_char = char if isinstance(char, str) else char[0]
            end_char = char if isinstance(char, str) else char[1]
            if result.startswith(start_char) and result.endswith(end_char):
                result = result[len(start_char) : -len(end_char)]
                did_strip = True

    return result


def limit_newlines(content: str, max_newlines: int = 2) -> str:
    """Reduces the number of consecutive newline characters in a string to the given maximum"""
    pattern = re.compile(r"\n{" + str(max_newlines + 1) + r",}")
    return pattern.sub("\n" * max_newlines, content)
