# Copyright 2026 ConceptHierarchy Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for :meth:`LiteralValue.convert_to_value`, the step that turns a literal template argument's
*notation* into the JSON value it stands for.

These live at the unit level on purpose. A substituted literal is recognised only by
``defaultSerialization``, and that route keeps no parsed value tree -- so once substitution has happened
the resulting expression records the *type* and not the value, and an integration test cannot see whether
the conversion produced ``3`` or ``9``, ``True`` or ``False``. This is the only place the conversion's
correctness is observable.

Two mistakes it has already made, both silent:

* ``bool("false")`` is ``True`` -- every non-empty string is -- so converting the literal's text with
  ``bool()`` turned every boolean argument into ``True``;
* ``full_name`` re-adds the quotes a string literal is *written* with, so the value carried the quote
  characters and no longer matched anything expecting the bare string.
"""

from __future__ import annotations

import pytest

from concept_hierarchy.data.types.concept_hierarchy_types import LiteralValue


class TestBooleanLiterals:
    @pytest.mark.parametrize("text,expected", [("true", True), ("false", False)])
    def test_the_two_boolean_literals(self, text, expected):
        value = LiteralValue(text, "bool").convert_to_value()
        assert value is expected

    def test_false_is_not_truthy(self):
        """The regression: ``bool("false")`` is ``True``."""
        assert LiteralValue("false", "bool").convert_to_value() is False


class TestIntegerLiterals:
    @pytest.mark.parametrize("text,expected", [("0", 0), ("3", 3), ("-7", -7), ("1000000", 1000000)])
    def test_integers_convert_exactly(self, text, expected):
        value = LiteralValue(text, "int").convert_to_value()
        assert value == expected
        assert isinstance(value, int) and not isinstance(value, bool)


class TestNumberLiterals:
    @pytest.mark.parametrize("text,expected", [("3.5", 3.5), ("-0.25", -0.25), ("0.0", 0.0)])
    def test_numbers_convert_exactly(self, text, expected):
        value = LiteralValue(text, "float").convert_to_value()
        assert value == expected
        assert isinstance(value, float)


class TestStringLiterals:
    @pytest.mark.parametrize("text", ["s:hello", "hello", "", "one", "a.b.c", "true", "3"])
    def test_the_value_is_the_bare_string(self, text):
        """
        The quotes belong to the notation, not the value: ``LiteralValue.full_name`` re-adds them, and
        substituting *that* yields a string containing quote characters.
        """
        assert LiteralValue(text, "string").convert_to_value() == text

    def test_the_quoted_form_is_not_what_is_substituted(self):
        literal = LiteralValue("s:hello", "string")
        assert literal.full_name == '"s:hello"', "full_name is the written form"
        assert literal.convert_to_value() == "s:hello", "...and the value is not"


class TestAnUnknownSort:
    def test_it_is_rejected_rather_than_guessed(self):
        with pytest.raises(RuntimeError, match="invalid"):
            LiteralValue("3", "not-a-sort").convert_to_value()
