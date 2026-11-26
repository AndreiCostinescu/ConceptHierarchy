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
Integration tests: a type written in an expression means what it means **where it is written**.

`ExpressionValidator.create_possibly_template_dependent_type` memoises by the type's *text*, and
`parse_convert_type_in_template_context` -- what it memoises -- reads the ambient template context. The
text ``Box<T>`` is therefore not one type: in ``A<T: ValueDomain>`` it is ``Box<A:T>`` and in
``B<T: Numeric>`` it is ``Box<B:T>``, and a cache keyed by the text alone hands the second concept the
first one's parse.

The leaked variable is caught eventually -- `substitute` compares its name against the mapping it is given
-- but by a bare ``assert``, so an **authoring mistake surfaces as an `AssertionError`** instead of a
diagnosed `CHSemanticError`. The same defect therefore gets two entirely different outcomes depending on
what else the hierarchy happens to contain:

* alone, ``B<S>`` writing ``Box<T>`` is rejected with *"ParsedType 'T' is not a template variable (in this
  context) nor a concept!"* -- a proper diagnosis, naming the text at fault;
* with an unrelated ``A<T>`` in the hierarchy, the identical mistake raises
  ``AssertionError: A:T is not a template variable of the context it is written in (['S'])`` -- an
  internal-invariant failure, naming a concept the author of ``B`` never mentioned.

An `AssertionError` is not a rejection an author can act on, and it is not one a caller can catch: it says
the checker's own invariant broke, when in truth the input was simply wrong. These tests pin that the same
mistake gets the same diagnosis whatever else is in the hierarchy.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from concept_hierarchy.errors import ConceptHierarchyError
from tests.integration.test_expression_parsing import check_concepts
from tests.integration.test_schema_substitution import obj, vd


def prelude() -> dict:
    """
    Built fresh per call, so that no test can depend on what another left behind.

    `check_concepts` does **not** mutate what it is handed -- ``test_definition_data_is_not_modified``
    pins that, and a direct deep-copy comparison over these fragments confirms it. Fresh per call is
    cheap insurance rather than a workaround.
    """
    return {
        **vd("Leaf", {"type": "object", "additionalProperties": False}),
        **vd("Box", obj({"b": {"type": "P"}}), {"order": ["P"], "P": "ValueDomain"}),
    }


def concept(name: str, variable: str, default_key: str, variable_constraint: str = "ValueDomain") -> dict:
    """``name<variable>`` whose only property defaults to an instantiation of ``default_key``."""
    return vd(
        name,
        obj({"p": {"type": f"Box<{variable}>", "default": {default_key: {"b": {}}}}}),
        {"order": [variable], variable: variable_constraint},
    )


def uses(*applications: str) -> dict:
    return vd("Uses", obj({f"u{i}": {"type": t, "default": {}} for i, t in enumerate(applications)}))


def rejection(concepts: dict) -> str:
    with pytest.raises(ConceptHierarchyError) as excinfo:
        with contextlib.redirect_stdout(io.StringIO()):
            check_concepts(concepts)
    return str(excinfo.value)


def rejected_properly(concepts: dict) -> bool:
    """
    Rejected, and rejected as a `ConceptHierarchyError`.

    Deliberately not a bare ``except Exception``: an `AssertionError` means the checker's own invariant
    broke, which is a different -- and worse -- outcome than telling the author their hierarchy is wrong.
    """
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            check_concepts(concepts)
        return False
    except ConceptHierarchyError:
        return True


def accepts(concepts: dict) -> bool:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            check_concepts(concepts)
        return True
    except ConceptHierarchyError as e:
        print(f"This will error: causing error is {e!r}")
        return False


# ``B<S>`` writes ``Box<T>``: ``T`` is not ``B``'s template variable, so this is wrong however it is
# reached. ``A<T>`` writes the same text legitimately, and is what warms the cache.
def wrong_alone() -> dict:
    return {**prelude(), **concept("B", "S", "Box<T>"), **uses("B<Leaf>")}


def wrong_beside_a() -> dict:
    return {
        **prelude(),
        **concept("A", "T", "Box<T>"),
        **concept("B", "S", "Box<T>"),
        **uses("A<Leaf>", "B<Leaf>"),
    }


def both_legitimate() -> dict:
    return {
        **prelude(),
        **concept("A", "T", "Box<T>"),
        **concept("B", "T", "Box<T>"),
        **uses("A<Leaf>", "B<Leaf>"),
    }


def key_based_same_type() -> dict:
    return {
        **prelude(),
        **concept("A", "T", "Box<T>", "Box"),
        **concept("B", "T", "Box<T>", "Not(Box)"),
        **uses("A<Box<Leaf>>", "B<Box<Leaf>>"),
    }


class TestATypeTextMeansWhatItMeansWhereItIsWritten:
    def test_an_undefined_template_variable_is_rejected(self):
        assert rejected_properly(wrong_alone())

    def test_it_is_still_rejected_as_a_diagnosis_when_another_concept_writes_the_same_text(self):
        """Today this raises `AssertionError` instead, which no caller catches and no author can act on."""
        assert rejected_properly(wrong_beside_a())

    def test_the_diagnosis_does_not_depend_on_what_else_the_hierarchy_contains(self):
        """
        The same mistake, explained the same way. Today the second one names ``A`` -- a concept the author
        of ``B`` never mentioned -- because ``Box<T>`` was resolved out of a cache keyed by its text.
        """
        alone = rejection(wrong_alone())
        beside = rejection(wrong_beside_a())
        assert "A:T" not in beside, f"the diagnosis names an unrelated concept's template variable: {beside[:400]}"
        assert "'T' is not a template variable" in alone
        assert "'T' is not a template variable" in beside

    def test_two_concepts_may_legitimately_write_the_same_text(self):
        """
        The passing guard. ``A<T>`` and ``B<T>`` both writing ``Box<T>`` is ordinary and must keep working
        -- the fix is to stop *sharing* the parse, not to stop allowing the text.
        """
        assert accepts(both_legitimate())

    def test_the_same_text_resolves_to_each_concepts_own_variable(self):
        """
        White-box, because the two parses are indistinguishable from outside once both are correct: the
        memo must not hold one entry for ``Box<T>`` that both concepts share.
        """
        with contextlib.redirect_stdout(io.StringIO()):
            context = check_concepts(both_legitimate())
        entries = {
            key: str(value)
            for key, value in context.expression_parser_validator.parsed_types.items()
            if "Box<" in str(key)
        }
        resolved = set(entries.values())
        assert {"Box<A:T>", "Box<B:T>"} <= resolved, (
            f"each concept's own variable must be memoised separately; got {entries}"
        )
