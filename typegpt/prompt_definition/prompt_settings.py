from dataclasses import dataclass


@dataclass
class PromptSettings:
    """
    Prompt settings

    Args:
        disable_formatting_instructions: Disables added system instruction that instructs LLM how to format output. Only use this if you use few-shot prompting, a fine-tuned model, or instruct the model yourself (not recommended).
    """

    disable_formatting_instructions: bool = False
