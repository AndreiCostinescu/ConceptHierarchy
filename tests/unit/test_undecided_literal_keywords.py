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
Unit tests for :meth:`CHSchemaNode.undecided_literal_keywords_for`.

It answers one question: *was this value actually checked against this node?* A keyword written as a literal
template variable is popped out of the schema by `jsonschema_parser` and parked in a ``*_def`` field, so
while it sits there the Draft-07 validator never sees it and a value sails through unchecked. An
`InstExpression` is supposed to mean "parsed against this schema, and it holds", so a value that reaches
such a node produces a `PossibleInstExpression` instead -- template dependent, to be decided when an
application binds the keyword.

The type family is the interesting half. A keyword constrains only its own JSON type, so an undecided
``minItems`` says nothing about a string, and treating it as undecidable would refuse to decide values that
*are* decided -- the same over-eagerness that would make an unmatched ``anyOf`` branch poison a value that
never entered it.
"""

from __future__ import annotations

import pytest

from concept_hierarchy.data.jsonschema.parsed_schema import (
    LITERAL_KEYWORD_FIELDS,
    CHSchemaNode,
    json_type_family_of,
)


def node(**parked: str) -> CHSchemaNode:
    """A schema node with the given ``*_def`` fields parked, as `jsonschema_parser` would leave them."""
    result = CHSchemaNode(location_id=[], raw={}, canonical={})
    for field_name, variable in parked.items():
        setattr(result, field_name, variable)
    return result


SAMPLE = {"object": {"a": 1}, "array": [1], "string": "s", "number": 3}


class TestJsonTypeFamily:
    @pytest.mark.parametrize(
        "value,expected",
        [({}, "object"), ([], "array"), ("", "string"), (0, "number"), (1.5, "number"), (-2, "number")],
    )
    def test_the_families(self, value, expected):
        assert json_type_family_of(value) == expected

    @pytest.mark.parametrize("value", [True, False])
    def test_a_boolean_is_not_a_number(self, value):
        """``bool`` is a subclass of ``int``; no numeric keyword applies to a boolean."""
        assert json_type_family_of(value) is None

    def test_null_has_no_family(self):
        assert json_type_family_of(None) is None


class TestAParkedKeywordIsUndecidedForItsOwnType:
    @pytest.mark.parametrize(
        "field_name,keyword,family",
        [(f, k, fam) for f, (k, fam) in LITERAL_KEYWORD_FIELDS.items()],
        ids=list(LITERAL_KEYWORD_FIELDS),
    )
    def test_every_keyword(self, field_name, keyword, family):
        subject = node(**{field_name: "N"})
        assert subject.undecided_literal_keywords_for(SAMPLE[family]) == (keyword,)

    @pytest.mark.parametrize(
        "field_name,family",
        [(f, fam) for f, (_k, fam) in LITERAL_KEYWORD_FIELDS.items()],
        ids=list(LITERAL_KEYWORD_FIELDS),
    )
    def test_it_says_nothing_about_the_other_types(self, field_name, family):
        subject = node(**{field_name: "N"})
        for other, value in SAMPLE.items():
            if other == family:
                continue
            assert subject.undecided_literal_keywords_for(value) == (), f"{field_name} fired on a {other}"

    @pytest.mark.parametrize("value", [True, False, None])
    def test_no_keyword_applies_to_a_boolean_or_null(self, value):
        subject = node(**{field_name: "N" for field_name in LITERAL_KEYWORD_FIELDS})
        assert subject.undecided_literal_keywords_for(value) == ()


class TestNothingParked:
    def test_a_node_with_no_literal_keywords_is_decided(self):
        for value in SAMPLE.values():
            assert node().undecided_literal_keywords_for(value) == ()

    def test_a_cleared_field_is_decided_again(self):
        """
        What `substitute_schema` does once the value is known: it writes the bound into `shallow_canonical`
        and clears the field. Clearing it is what makes the value decidable, so the two must go together.
        """
        subject = node(min_items_def="N")
        assert subject.undecided_literal_keywords_for([1]) == ("minItems",)
        subject.min_items_def = None
        assert subject.undecided_literal_keywords_for([1]) == ()


class TestSeveralAtOnce:
    def test_two_keywords_of_the_same_family(self):
        subject = node(min_items_def="N", max_items_def="M")
        assert set(subject.undecided_literal_keywords_for([1])) == {"minItems", "maxItems"}

    def test_only_the_applicable_ones_of_a_mixed_node(self):
        subject = node(min_items_def="N", min_length_def="M", minimum_def="K")
        assert subject.undecided_literal_keywords_for([1]) == ("minItems",)
        assert subject.undecided_literal_keywords_for("s") == ("minLength",)
        assert subject.undecided_literal_keywords_for(3) == ("minimum",)
        assert subject.undecided_literal_keywords_for({}) == ()
