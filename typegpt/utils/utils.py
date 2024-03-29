import re


def symmetric_strip(content: str, chars: list[str]) -> str:
    """Strips the given chars from the beginning and end of the string, but only if they are present on both sides"""
    result = content
    did_strip = True
    while did_strip:
        did_strip = False

        for char in chars:
            if result.startswith(char) and result.endswith(char):
                result = result[len(char) : -len(char)]
                did_strip = True

    return result


def limit_newlines(content: str, max_newlines: int = 2) -> str:
    """Reduces the number of consecutive newline characters in a string to the given maximum"""
    pattern = re.compile(r"\n{" + str(max_newlines + 1) + r",}")
    return pattern.sub("\n" * max_newlines, content)
