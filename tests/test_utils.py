import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

import pytest

from typegpt.utils.utils import symmetric_strip, limit_newlines


class TestUtils:

    def test_symmetric_strip(self):

        assert symmetric_strip("abc", ["a", "c"]) == "abc"
        assert symmetric_strip("abc", [("a", "c")]) == "b"
        assert symmetric_strip("abc", ["a", "b"]) == "abc"
        assert symmetric_strip("abc", [("a", "b")]) == "abc"

        assert symmetric_strip("<abc>", [("<", ">")]) == "abc"
        assert symmetric_strip("<ofo>", [("<o", "o>")]) == "f"

    def test_limit_newlines(self):

        assert limit_newlines("abc\n\n\n\n\n\n\n\ndef", 2) == "abc\n\ndef"
        assert limit_newlines("abc\ndef\n\nghi\n\n\njkl", 2) == "abc\ndef\n\nghi\n\njkl"
