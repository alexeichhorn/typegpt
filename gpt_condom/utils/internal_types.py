from typing import final


@final
class _NoDefaultType:
    def __repr__(self):
        return "<no default>"


_NoDefault = _NoDefaultType()


# -


@final
class _UseDefaultType:
    def __repr__(self):
        return "<use default>"


_UseDefault = _UseDefaultType()
