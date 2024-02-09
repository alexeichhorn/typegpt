import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

import pytest

from typegpt import BaseLLMArrayElement, BaseLLMResponse, LLMArrayElementOutput, LLMArrayOutput, LLMOutput, PromptTemplate, FewShotExample


class TestFewShot:
    class SimplePrompt(PromptTemplate):
        class Output(BaseLLMResponse):
            title: str

        def system_prompt(self) -> str:
            return "Some system prompt"

        def user_prompt(self) -> str:
            return "Some user prompt"

    def test_simple_prompt(self):
        prompt = self.SimplePrompt()
        messages = prompt.generate_messages(token_limit=1000, token_counter=lambda x: 0)

        expected_system_prompt = f"""
Some system prompt

Always return the answer in the following format:
\"""
TITLE: <Put the title here>
\"""
""".strip()

        assert messages == [
            {"role": "system", "content": expected_system_prompt},
            {"role": "user", "content": "Some user prompt"},
        ]

    class SimplePromptWithFewShot(PromptTemplate):
        class Output(BaseLLMResponse):
            title: str

        def __init__(self, num_shots: int):
            self.num_shots = num_shots

        def system_prompt(self) -> str:
            return "Some system prompt"

        def user_prompt(self) -> str:
            return "Some user prompt"

        def few_shot_examples(self) -> list[FewShotExample[Output]]:
            return [FewShotExample(input=f"Some input {i}", output=self.Output(title=f"Some title {i}")) for i in range(self.num_shots)]

    def test_simple_few_shot_prompt(self):
        prompt_one = self.SimplePromptWithFewShot(num_shots=1)
        prompt_three = self.SimplePromptWithFewShot(num_shots=3)
        messages_one = prompt_one.generate_messages(token_limit=1000, token_counter=lambda x: 0)
        messages_three = prompt_three.generate_messages(token_limit=1000, token_counter=lambda x: 0)

        expected_system_prompt = f"""
Some system prompt

Always return the answer in the following format:
\"""
TITLE: <Put the title here>
\"""
""".strip()

        assert messages_one == [
            {"role": "system", "content": expected_system_prompt},
            {"role": "system", "name": "example_user", "content": "Some input 0"},
            {"role": "system", "name": "example_assistant", "content": "TITLE: Some title 0"},
            {"role": "user", "content": "Some user prompt"},
        ]

        assert messages_three == [
            {"role": "system", "content": expected_system_prompt},
            {"role": "system", "name": "example_user", "content": "Some input 0"},
            {"role": "system", "name": "example_assistant", "content": "TITLE: Some title 0"},
            {"role": "system", "name": "example_user", "content": "Some input 1"},
            {"role": "system", "name": "example_assistant", "content": "TITLE: Some title 1"},
            {"role": "system", "name": "example_user", "content": "Some input 2"},
            {"role": "system", "name": "example_assistant", "content": "TITLE: Some title 2"},
            {"role": "user", "content": "Some user prompt"},
        ]

    class ComplexPrompt(PromptTemplate):
        class Output(BaseLLMResponse):
            class Item(BaseLLMArrayElement):
                class InnerItem(BaseLLMResponse):
                    title: str
                    description: str

                class InnerElement(BaseLLMArrayElement):
                    value: float
                    is_accurate: bool

                subtitle: str
                description: str | None = LLMArrayElementOutput(lambda pos: f"Put the {pos.ordinal} item description here")
                abstract: str = LLMArrayElementOutput(lambda _: "...", multiline=True)
                inner_item: InnerItem
                inner_elements: list[InnerElement] = LLMArrayOutput((0, None), instruction=lambda _: "...")
                multiline_text: list[str] = LLMArrayOutput((0, None), lambda _: "...", multiline=True)

            class DirectItem(BaseLLMResponse):
                class InnerDirectElement(BaseLLMArrayElement):
                    subtitle: str

                title: str
                x: list[InnerDirectElement]

            title: str
            subitem: DirectItem
            middle_int: int
            items: list[Item]

        def system_prompt(self) -> str:
            return "Some system prompt"

        def user_prompt(self) -> str:
            return "Some user prompt"

        def few_shot_examples(self) -> list[FewShotExample[Output]]:
            return [
                FewShotExample(
                    input=f"Some input 1",
                    output=self.Output(
                        title=f"Some title 1", middle_int=13, subitem=self.Output.DirectItem(title=f"Subtitle 1", x=[]), items=[]
                    ),
                ),
                FewShotExample(
                    input=f"Some input 2",
                    output=self.Output(
                        title="TITLE 2\nSecond line of title",
                        middle_int=42,
                        subitem=self.Output.DirectItem(
                            title=f"Subtitle 2\nalso with a second line",
                            x=[
                                self.Output.DirectItem.InnerDirectElement(subtitle="Subtitle 2.1 :D"),
                                self.Output.DirectItem.InnerDirectElement(subtitle="Subtitle 2.2"),
                                self.Output.DirectItem.InnerDirectElement(subtitle="... and finally the last one"),
                            ],
                        ),
                        items=[
                            self.Output.Item(
                                subtitle="Subtitle in item 1",
                                description="Description in item 1",
                                abstract="Abstract in item 1",
                                inner_item=self.Output.Item.InnerItem(title="Inner item 1", description="Description in inner item 1"),
                                inner_elements=[],
                                multiline_text=["a", "b", "c\nd", "e"],
                            ),
                            self.Output.Item(
                                subtitle="Subtitle in item 2",
                                description=None,
                                abstract="Abstract in item 2",
                                inner_item=self.Output.Item.InnerItem(title="Inner item 2", description="Description in inner item 2"),
                                inner_elements=[
                                    self.Output.Item.InnerElement(value=1.0, is_accurate=True),
                                    self.Output.Item.InnerElement(value=0.5, is_accurate=False),
                                    self.Output.Item.InnerElement(value=3.14159, is_accurate=False),
                                ],
                                multiline_text=[],
                            ),
                        ],
                    ),
                ),
            ]

    def test_complex_prompt(self):
        prompt = self.ComplexPrompt()
        messages = prompt.generate_messages(token_limit=1000, token_counter=lambda x: 0)

        # remove system prompt (as it is too complicated and already tested in `test_parser.py`)
        messages = messages[1:]

        expected_output_1 = """
TITLE: Some title 1

SUBITEM TITLE: Subtitle 1
MIDDLE INT: 13
""".strip()

        expected_output_2 = """
TITLE: TITLE 2
Second line of title

SUBITEM TITLE: Subtitle 2
also with a second line

SUBITEM X 1 SUBTITLE: Subtitle 2.1 :D
SUBITEM X 2 SUBTITLE: Subtitle 2.2
SUBITEM X 3 SUBTITLE: ... and finally the last one
MIDDLE INT: 42

ITEM 1 SUBTITLE: Subtitle in item 1
ITEM 1 DESCRIPTION: Description in item 1
ITEM 1 ABSTRACT: Abstract in item 1

ITEM 1 INNER ITEM TITLE: Inner item 1
ITEM 1 INNER ITEM DESCRIPTION: Description in inner item 1

ITEM 1 MULTILINE TEXT 1: a
ITEM 1 MULTILINE TEXT 2: b
ITEM 1 MULTILINE TEXT 3: c
d
ITEM 1 MULTILINE TEXT 4: e

ITEM 2 SUBTITLE: Subtitle in item 2
ITEM 2 ABSTRACT: Abstract in item 2

ITEM 2 INNER ITEM TITLE: Inner item 2
ITEM 2 INNER ITEM DESCRIPTION: Description in inner item 2

ITEM 2 INNER ELEMENT 1 VALUE: 1.0
ITEM 2 INNER ELEMENT 1 IS ACCURATE: true

ITEM 2 INNER ELEMENT 2 VALUE: 0.5
ITEM 2 INNER ELEMENT 2 IS ACCURATE: false

ITEM 2 INNER ELEMENT 3 VALUE: 3.14159
ITEM 2 INNER ELEMENT 3 IS ACCURATE: false
""".strip()

        assert messages == [
            {"role": "system", "name": "example_user", "content": "Some input 1"},
            {"role": "system", "name": "example_assistant", "content": expected_output_1},
            {"role": "system", "name": "example_user", "content": "Some input 2"},
            {"role": "system", "name": "example_assistant", "content": expected_output_2},
            {"role": "user", "content": "Some user prompt"},
        ]
