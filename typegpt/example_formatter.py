from typing import Any, Iterable


class LimitedExampleListFormatter:
    """
    The LimitedExampleListFormatter class converts a list of objects into a string representation,
    limiting the number of displayed elements. If the number of elements exceeds the limit,
    it includes a skip separator between the displayed and omitted elements.

    Attributes:
        max_examples (int): The maximum number of elements to display in the string.
        separator (str): The separator used between the displayed elements.
        skip_separator (str, optional): The separator used between the displayed and omitted elements. Defaults to "...".

    Examples:
        formatter = LimitedExampleListFormatter(3, ', ')
        print(formatter.format(['apple', 'banana', 'cherry', 'date', 'elderberry']))
        # Output: "apple, banana, ..., elderberry"
    """

    def __init__(self, max_examples: int, separator: str, skip_separator: str = "...", examples_after_separtor: int = 1):
        self.max_examples = max_examples
        self.separator = separator
        self.skip_separator = skip_separator
        self.examples_after_separtor = examples_after_separtor
        assert self.max_examples > self.examples_after_separtor, "max_examples must be greater than examples_after_separtor"

    def format(self, items: Iterable[Any]) -> str:
        str_items = [str(item) for item in items]

        if len(str_items) <= self.max_examples:
            return self.separator.join(str_items)

        # We want to show first n-1 examples, then skip separator, then last example
        return self.separator.join(
            str_items[: self.max_examples - self.examples_after_separtor]
            + [self.skip_separator]
            + str_items[-self.examples_after_separtor :]
        )
