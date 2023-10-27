from typing import Literal


class AzureContentFilterException(Exception):
    reason: Literal["prompt", "completion"]

    def __init__(self, reason: Literal["prompt", "completion"]):
        self.reason = reason
        super().__init__(f"Content filter blocked {reason}.")
