from typing import final


@final
class _NoDefaultType:
    def __repr__(self):
        return "<no default>"


_NoDefault = _NoDefaultType()
