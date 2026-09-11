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
`split_function_interpretation_marker`: taking an ``fEval:`` / ``fComp:`` / ``fInst:`` prefix off a key.

The splitter lands before anything recognizes the markers
(`documentation/TODO_FUNCTION_INTERPRETATION_MARKER.md` §4.3), so it is tested here directly rather than
through a hierarchy. Two rules carry it, and both have a reason that is easy to get wrong later:

* **the first colon only** -- a type application may carry a *string literal* template argument, which is
  arbitrary text and may contain colons, including the marker words. ``Tagged<"fEval:x">`` is a valid
  application today (pinned end to end by
  `tests/integration/test_function_interpretation.py::TestTheGrammarAKeyPrefixMarkerWouldRelyOn`);
* **an unknown prefix is not a marker** -- the key is returned untouched, so ``s:``-prefixed values and the
  qualified-variable form ``Add:T`` pass through and fail, or not, exactly as they do today.
"""

from __future__ import annotations

import pytest

from concept_hierarchy.data.expressions.expression_utils import (
    FUNCTION_INTERPRETATION_MARKERS,
    FunctionInterpretation,
    split_function_interpretation_marker,
)


class TestAMarkedKeyIsSplit:
    @pytest.mark.parametrize("marker", sorted(FUNCTION_INTERPRETATION_MARKERS))
    def test_each_marker_is_recognised(self, marker):
        interpretation, key = split_function_interpretation_marker(f"{marker}:Nullary")
        assert interpretation is FUNCTION_INTERPRETATION_MARKERS[marker]
        assert key == "Nullary"

    def test_the_evaluation_marker_maps_to_the_evaluation_interpretation(self):
        assert split_function_interpretation_marker("fEval:Add")[0] is FunctionInterpretation.EVALUATION

    def test_a_template_application_keeps_its_arguments(self):
        assert split_function_interpretation_marker("fComp:Add<Integer>")[1] == "Add<Integer>"

    def test_a_template_dependent_application_is_unaffected(self):
        assert split_function_interpretation_marker("fInst:AddT<T>")[1] == "AddT<T>"

    def test_a_bare_template_variable_may_carry_a_marker(self):
        """
        The key is only substituted at grounding, so the interpretation has to be written on the variable
        -- there is no later spelling for the author to annotate.
        """
        interpretation, key = split_function_interpretation_marker("fInst:F")
        assert interpretation is not None and key == "F"


class TestOnlyTheFirstColonSplits:
    def test_a_string_literal_argument_may_contain_a_colon(self):
        assert split_function_interpretation_marker('fEval:Tagged<"a:b">')[1] == 'Tagged<"a:b">'

    def test_a_string_literal_argument_may_contain_a_marker_word(self):
        """The case that rules out splitting on the last colon."""
        interpretation, key = split_function_interpretation_marker('fEval:Tagged<"fComp:x">')
        assert interpretation is FunctionInterpretation.EVALUATION
        assert key == 'Tagged<"fComp:x">'

    def test_an_unmarked_key_with_a_colon_inside_is_untouched(self):
        assert split_function_interpretation_marker('Tagged<"fEval:x">') == (None, 'Tagged<"fEval:x">')


class TestWhatIsNotAMarker:
    def test_an_unmarked_key(self):
        assert split_function_interpretation_marker("Nullary") == (None, "Nullary")

    def test_an_unknown_prefix_is_left_alone(self):
        assert split_function_interpretation_marker("fOther:Nullary") == (None, "fOther:Nullary")

    def test_a_string_value_prefix_is_left_alone(self):
        """``s:`` is a `String` marker and must not be eaten here; `_parse_string` strips it separately."""
        assert split_function_interpretation_marker("s:Nullary") == (None, "s:Nullary")

    def test_the_qualified_variable_form_is_left_alone(self):
        """``Add:T`` is `templateContext` syntax, not a type application -- and not this splitter's business."""
        assert split_function_interpretation_marker("Add:T") == (None, "Add:T")

    def test_the_match_is_case_sensitive(self):
        assert split_function_interpretation_marker("feval:Nullary") == (None, "feval:Nullary")

    def test_a_marker_word_without_a_colon_is_a_plain_key(self):
        assert split_function_interpretation_marker("fEval") == (None, "fEval")

    def test_a_marker_word_that_is_the_whole_key_with_a_colon_leaves_an_empty_remainder(self):
        """Degenerate, and the caller's problem: an empty remainder is not a type and will be rejected."""
        assert split_function_interpretation_marker("fEval:") == (FunctionInterpretation.EVALUATION, "")

    def test_an_empty_key(self):
        assert split_function_interpretation_marker("") == (None, "")


class TestEachMarkerNamesADistinctInterpretation:
    """
    One marker per reading, and no two alike. This class replaced the reminder that ``fComp`` and
    ``fInst`` were indistinguishable -- which they were until the enum was split, and which is exactly
    what a marker had to exist to make possible.
    """

    def test_the_three_interpretations_are_distinct(self):
        interpretations = {FUNCTION_INTERPRETATION_MARKERS[m] for m in ("fEval", "fComp", "fInst")}
        assert len(interpretations) == 3, interpretations

    def test_each_maps_to_the_member_named_after_it(self):
        assert FUNCTION_INTERPRETATION_MARKERS["fEval"] is FunctionInterpretation.EVALUATION
        assert FUNCTION_INTERPRETATION_MARKERS["fComp"] is FunctionInterpretation.COMPOSITION
        assert FUNCTION_INTERPRETATION_MARKERS["fInst"] is FunctionInterpretation.INSTANTIATION

    def test_no_marker_says_only_not_an_evaluation(self):
        """
        `NOT_AN_EVALUATION` says only what a value is *not*, which no marker ever needs to say: a marker
        names one reading, and the three of them name three different ones.
        """
        assert FunctionInterpretation.NOT_AN_EVALUATION not in FUNCTION_INTERPRETATION_MARKERS.values()
