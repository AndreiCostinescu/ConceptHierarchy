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
``"requireAllItems": <boolean>``: every position an ``items`` tuple names must be present in the value.

draft-07's tuple form constrains position *i* only when element *i* is *there*, so ``items: [A, B]``
accepts ``[a]``. ``additionalItems: false`` bounds such a tuple above and nothing bounds it below. This
keyword is that lower bound, and the two are **complementary** rather than overlapping -- each says one
thing, and they compose:

======================  ======================  =========================================
``requireAllItems``     ``additionalItems``     accepted for ``items: [A, B]``
======================  ======================  =========================================
absent / ``false``      absent / ``true``       any array whose present positions match
absent / ``false``      ``false``               at most 2
``true``                absent / ``true``       at least 2
``true``                ``false``               exactly 2
======================  ======================  =========================================

The motivating case is a tuple whose positions come from a **variadic expansion**
(`documentation/TODO_VARIADIC_SCHEMA_EXPANSION.md` 9.1): there the length is the arity of the group, so
the author cannot write it as a ``minItems`` at the declaration.
Nothing about the keyword is specific to that, which is why it is tested here on written-out tuples too.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from concept_hierarchy.errors import ConceptHierarchyError
from tests.integration.test_expression_parsing import build_hierarchy, check_hierarchy, error_messages


def vd(instantiation: object, name: str = "V") -> dict:
    return {name: {"directParents": ["ValueDomain"], "data": {"instantiation": instantiation}}}


def box(site_type: str) -> dict:
    return {
        "Box": {
            "directParents": ["ValueDomain"],
            "data": {
                "instantiation": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["v"],
                    "properties": {"v": {"type": site_type}},
                }
            },
        }
    }


def declare(concepts: dict) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        check_hierarchy(build_hierarchy(concepts))


def declaration_rejected(concepts: dict) -> str:
    with pytest.raises(ConceptHierarchyError) as excinfo:
        declare(concepts)
    return error_messages(excinfo.value)


def accepts(instantiation: object, value: object) -> None:
    concepts = {**vd(instantiation), **box("V")}
    with contextlib.redirect_stdout(io.StringIO()):
        check_hierarchy(build_hierarchy(concepts, instances={"p": {"Box": {"v": value}}}))


def refuses(instantiation: object, value: object) -> str:
    """The value is rejected while the schema itself is valid; see the variadic suite for why both."""
    declare({**vd(instantiation), **box("V")})
    with pytest.raises(ConceptHierarchyError) as excinfo:
        accepts(instantiation, value)
    return error_messages(excinfo.value)


TUPLE = {"type": "array", "items": ["Integer", "String"]}


# ==================================================================================================
# 1. Where the keyword may be written
# ==================================================================================================


class TestWhereTheKeywordIsAllowed:
    """
    It says something only about a *positional* list, so it is refused wherever there is none. Fail
    closed, as everywhere else in this area: a keyword that is silently ignored is the failure mode that
    `isFunctionEvaluation` was removed for.
    """

    @pytest.mark.parametrize("written", [True, False])
    def test_it_is_allowed_beside_an_items_tuple(self, written):
        declare(vd({**TUPLE, "requireAllItems": written}))

    def test_it_is_allowed_beside_additional_items(
        self,
    ):
        declare(vd({**TUPLE, "additionalItems": False, "requireAllItems": True}))

    @pytest.mark.parametrize(
        "schema",
        [
            pytest.param({"type": "array", "items": "Integer", "requireAllItems": True}, id="items-schema-form"),
            pytest.param({"type": "array", "requireAllItems": True}, id="no-items"),
            pytest.param({"type": "object", "requireAllItems": True}, id="not-an-array"),
        ],
    )
    def test_it_is_refused_where_items_is_not_an_array(self, schema):
        assert declaration_rejected(vd(schema))

    def test_the_rejection_names_the_keyword(self):
        messages = declaration_rejected(vd({"type": "array", "items": "Integer", "requireAllItems": True}))
        assert "requireAllItems" in messages and "items" in messages, messages

    @pytest.mark.parametrize("written", [2, "s:yes", None, []])
    def test_a_non_boolean_value_is_refused(self, written):
        assert declaration_rejected(vd({**TUPLE, "requireAllItems": written}))


# ==================================================================================================
# 2. What it does, on a written-out tuple
# ==================================================================================================


class TestTheLowerBoundOnAWrittenOutTuple:
    def test_a_short_value_is_refused(self):
        assert refuses({**TUPLE, "requireAllItems": True}, [1])

    def test_the_rejection_gives_both_counts(self):
        messages = refuses({**TUPLE, "requireAllItems": True}, [1])
        assert "2" in messages and "1" in messages, messages

    def test_a_full_value_is_accepted(self):
        accepts({**TUPLE, "requireAllItems": True}, [1, "s:x"])

    def test_an_empty_tuple_cannot_be_written_at_all(self):
        """
        ``items: []`` is invalid draft-07 and refused at declaration, so the empty tuple has no
        written-out form. It arises only from an expansion, where
        `TestTheLowerBoundOnAnExpandedTuple::test_an_empty_group_requires_nothing` covers it.
        """
        assert declaration_rejected(vd({"type": "array", "items": [], "requireAllItems": True}))

    def test_without_the_keyword_a_short_value_is_accepted(self):
        """The control, and the behavior the keyword exists to change. Passes today."""
        accepts(TUPLE, [1])


class TestItIsComplementaryToAdditionalItems:
    """
    Each bounds one end, and neither implies the other. All four combinations are pinned, because the
    point of choosing `requireAllItems` over an `exactItems` was that the two compose.
    """

    def test_the_lower_bound_alone_allows_extras(self):
        """What an `exactItems` could not have expressed: at least all of them, extras welcome."""
        accepts({**TUPLE, "requireAllItems": True}, [1, "s:x", 2])

    def test_the_upper_bound_alone_allows_a_short_value(self):
        """The mirror image, and true today."""
        accepts({**TUPLE, "additionalItems": False}, [1])

    def test_both_together_mean_exactly(self):
        schema = {**TUPLE, "additionalItems": False, "requireAllItems": True}
        accepts(schema, [1, "s:x"])
        assert refuses(schema, [1])
        assert refuses(schema, [1, "s:x", 2])

    def test_neither_constrains_the_length_at_all(self):
        accepts(TUPLE, [])
        accepts(TUPLE, [1, "s:x", 2])

    def test_writing_false_is_the_same_as_omitting_it(self):
        accepts({**TUPLE, "requireAllItems": False}, [1])


# ==================================================================================================
# 3. What it does on an expanded tuple -- the reason it exists
# ==================================================================================================


VARIADIC_TUPLE = {
    "Tuple": {
        "directParents": ["ValueDomain"],
        "data": {
            "templateContext": {"order": ["T..."], "T": "ValueDomain"},
            "instantiation": {"type": "array", "items": ["T..."], "requireAllItems": True},
        },
    }
}


class TestTheLowerBoundOnAnExpandedTuple:
    """
    The case no other keyword can express: the length is the arity of the group, which is not known where
    the schema is written. The bound is read off the *expanded* `items` list, so nothing has to count the group.
    """

    def _accepts(self, site_type: str, value: object) -> None:
        concepts = {**VARIADIC_TUPLE, **box(site_type)}
        with contextlib.redirect_stdout(io.StringIO()):
            check_hierarchy(build_hierarchy(concepts, instances={"p": {"Box": {"v": value}}}))

    def _refuses(self, site_type: str, value: object) -> None:
        declare({**VARIADIC_TUPLE, **box(site_type)})
        with pytest.raises(ConceptHierarchyError):
            self._accepts(site_type, value)

    def test_a_value_shorter_than_the_group_is_refused(self):
        self._refuses("Tuple<[Integer, String]>", [1])

    def test_a_value_as_long_as_the_group_is_accepted(self):
        self._accepts("Tuple<[Integer, String]>", [1, "s:x"])

    def test_the_bound_is_per_application(self):
        """One declaration, two arities: the same schema requires 1 here and 3 there."""
        self._accepts("Tuple<[Integer]>", [1])
        self._refuses("Tuple<[Integer, String, Integer]>", [1, "s:x"])

    def test_an_empty_group_requires_nothing(self):
        self._accepts("Tuple<[]>", [])
