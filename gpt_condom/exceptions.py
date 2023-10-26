class LLMException(Exception):
    ...


class LLMTokenLimitExceeded(LLMException):
    ...


class LLMOutputFieldMissing(LLMException):
    ...


class LLMOutputFieldWrongType(LLMException):
    ...
