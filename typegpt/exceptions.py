class LLMException(Exception):

    def __init__(self, message: str, system_prompt: str | None = None, user_prompt: str | None = None, raw_completion: str | None = None):
        super().__init__(message)
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.raw_completion = raw_completion


class LLMTokenLimitExceeded(LLMException): ...


class LLMParseException(LLMException): ...


class LLMOutputFieldMissing(LLMParseException): ...


class LLMOutputFieldWrongType(LLMParseException): ...


class LLMOutputFieldInvalidLength(LLMParseException): ...
