class LLMException(Exception):
    ...


class LLMTokenLimitExceeded(LLMException):
    ...


class LLMParseException(LLMException):
    ...


class LLMOutputFieldMissing(LLMParseException):
    ...


class LLMOutputFieldWrongType(LLMParseException):
    ...


class LLMOutputFieldInvalidLength(LLMParseException):
    ...
