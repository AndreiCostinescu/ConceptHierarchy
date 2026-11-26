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
Integration tests: the static-semantic compatibility rules of ``documentation/[CH].md`` §10.3.

Whether an expression may stand at a site depends on three things -- the site's **provenance**, its
**access**, and whether the expression's type is the expected one **exactly** or a **strict subtype** --
and the rules are two independent axes, not one:

* **provenance**: `Addr` demands a *place*. Only a variable, an instance property chain, or a Function
  evaluation whose result is declared `Addr` is one; `Inst`, `Narrow`, a default serialization and a
  non-`Addr` evaluation all construct a fresh value, which has no address to give.
* **access**: `Modify` demands that what is written to is the exact expected type *and* may be written to.
  Both apply **only to a place**. A fresh value may be a strict subtype under `Modify`, because the
  modification lands on the value the expression just built.

Reading those as one axis is what the implementation used to do -- any strict subtype was rejected under
`Modify`, whatever kind it was -- and it rejected three shapes §10.3 permits. `Inst` compounded it by
declaring itself a strict subtype unconditionally, though it instantiates *exactly* the expected type.

The rules now live on `Expression.static_semantic_violation`, so every construction of an `Expression` is
held to them rather than only the one the parser makes.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from concept_hierarchy.errors import ConceptHierarchyError
from tests.integration.test_expression_parsing import check_concepts
from tests.integration.test_schema_substitution import obj, vd

# --------------------------------------------------------------------------------------------------
# Hierarchy fragments
# --------------------------------------------------------------------------------------------------

POINT = vd("Point", obj({"x": {"type": "Integer"}, "y": {"type": "Integer"}}))

GROUND_FUNCTION = {"order": [], "substitution": {"FunctionReturning:T": "Integer"}}


def taking(argument_type: str, provenance: str, access: str) -> dict:
    """``F(p: argument_type)`` with the given provenance and access on ``p``."""
    return {
        "F": {
            "directParents": ["FunctionReturning"],
            "data": {
                "templateContext": GROUND_FUNCTION,
                "interface": {"p": [argument_type, provenance, access], "res": "Integer"},
            },
        }
    }


def returning(result: list | str, name: str = "Src") -> dict:
    """A Function whose result carries the given provenance/access, for use as an argument value."""
    return {
        name: {
            "directParents": ["FunctionReturning"],
            "data": {"templateContext": GROUND_FUNCTION, "interface": {"res": result}},
        }
    }


def site(value: object) -> dict:
    """A ValueDomain whose default evaluates ``F`` with ``p`` set to ``value``."""
    return vd("Site", obj({"s": {"type": "Integer", "default": {"F": {"p": value}}}}))


def accepts(concepts: dict) -> bool:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            check_concepts(concepts)
        return True
    except ConceptHierarchyError:
        return False


def violation(concepts: dict) -> str:
    with pytest.raises(ConceptHierarchyError) as excinfo:
        with contextlib.redirect_stdout(io.StringIO()):
            check_concepts(concepts)
    text = str(excinfo.value)
    assert "violation" in text, f"expected a static-semantic diagnosis, got: {text[:400]}"
    return text


INST = {"x": 1, "y": 2}
"""An `Inst` of `Point`: instantiated *at* the expected type, so exact by construction."""

NARROW = {"Point": {"x": 1, "y": 2}}
"""A `Narrow` -- exact at a `Point` site, a strict subtype at a `ValueDomain` one."""


# ==================================================================================================
# 1. Provenance: Addr demands a place
# ==================================================================================================


class TestAddrProvenanceDemandsAPlace:
    @pytest.mark.parametrize(
        ("kind", "argument_type", "value"),
        [("Inst", "Point", INST), ("Narrow", "ValueDomain", NARROW), ("default serialization", "Integer", 1)],
    )
    def test_a_constructed_value_can_not_satisfy_addr(self, kind, argument_type, value):
        text = violation({**POINT, **taking(argument_type, "Addr", "Get"), **site(value)})
        assert "Provenance violation" in text

    def test_a_function_evaluation_without_an_addr_result_can_not_satisfy_addr(self):
        concepts = {
            **POINT,
            **taking("Integer", "Addr", "Get"),
            **returning("Integer"),
            **site({"Src": {}}),
        }
        assert "Provenance violation" in violation(concepts)

    def test_a_function_evaluation_with_an_addr_result_can(self):
        """The passing guard: the same shape, with the result declared `Addr`."""
        concepts = {
            **POINT,
            **taking("Integer", "Addr", "Get"),
            **returning(["Integer", "Addr"]),
            **site({"Src": {}}),
        }
        assert accepts(concepts)

    @pytest.mark.parametrize(
        ("kind", "argument_type", "value"),
        [("Inst", "Point", INST), ("Narrow", "ValueDomain", NARROW), ("default serialization", "Integer", 1)],
    )
    def test_the_same_values_are_fine_at_an_any_site(self, kind, argument_type, value):
        """`Any` asks for a value, and all three are values."""
        assert accepts({**POINT, **taking(argument_type, "Any", "Get"), **site(value)})


# ==================================================================================================
# 2. Access: Modify constrains a place, and only a place
# ==================================================================================================


class TestModifyConstrainsOnlyAPlace:
    """
    The half that was wrong. A fresh value under `Modify` is modified *in the copy the expression built*,
    so neither its exactness nor where it came from constrains it.
    """

    @pytest.mark.parametrize("access", ["Modify", "GetModify"])
    def test_an_exact_instantiation_may_be_modified(self, access):
        """
        Rejected before this rule was split in two, because `Inst` declared itself a strict subtype
        unconditionally -- while instantiating exactly the expected type.
        """
        assert accepts({**POINT, **taking("Point", "Any", access), **site(INST)})

    @pytest.mark.parametrize("access", ["Modify", "GetModify"])
    def test_a_strict_subtype_narrow_may_be_modified(self, access):
        """``{"Point": ...}`` at a `ValueDomain` site is a strict subtype, and §10.3 permits it."""
        assert accepts({**POINT, **taking("ValueDomain", "Any", access), **site(NARROW)})

    def test_an_exact_narrow_may_be_modified(self):
        assert accepts({**POINT, **taking("Point", "Any", "Modify"), **site(NARROW)})

    def test_a_strict_subtype_default_serialization_may_be_modified(self):
        """
        ``1`` at a `Number` site serializes to `Integer`, a strict subtype -- and modifying it is legal,
        because what is modified is the value that serialization just produced.
        """
        assert accepts({**POINT, **taking("Number", "Any", "Modify"), **site(1)})


class TestModifyThroughAPlaceRequiresTheExactType:
    """The other side of the same rule: writing through an alias typed as a supertype is what is refused."""

    def test_a_strict_subtype_function_result_may_not_be_modified_through(self):
        concepts = {
            **POINT,
            **taking("ValueDomain", "Addr", "Modify"),
            **returning(["Point", "Addr", "Modify"]),
            **site({"Src": {}}),
        }
        text = violation(concepts)
        assert "Access violation" in text and "strict subtype" in text

    def test_the_exact_type_is_accepted_through_the_same_place(self):
        concepts = {
            **POINT,
            **taking("Point", "Addr", "Modify"),
            **returning(["Point", "Addr", "Modify"]),
            **site({"Src": {}}),
        }
        assert accepts(concepts)


class TestModifyRequiresAModifiableResult:
    """
    The axis §10.3 did not have. `FunctionEvaluation.is_result_modifiable` was computed and stored and
    then read by nothing, so a `Get`-only result satisfied a `Modify` argument.
    """

    def test_a_get_only_addressable_result_can_not_satisfy_modify(self):
        concepts = {
            **POINT,
            **taking("Integer", "Addr", "Modify"),
            **returning(["Integer", "Addr"]),
            **site({"Src": {}}),
        }
        text = violation(concepts)
        assert "Access violation" in text and "not modifiable" in text

    def test_a_modifiable_result_can(self):
        concepts = {
            **POINT,
            **taking("Integer", "Addr", "Modify"),
            **returning(["Integer", "Addr", "Modify"]),
            **site({"Src": {}}),
        }
        assert accepts(concepts)

    def test_a_get_only_result_is_still_fine_where_only_reading_is_asked(self):
        concepts = {
            **POINT,
            **taking("Integer", "Addr", "Get"),
            **returning(["Integer", "Addr"]),
            **site({"Src": {}}),
        }
        assert accepts(concepts)


# ==================================================================================================
# 3. Both axes at once
# ==================================================================================================


class TestEveryViolationIsReported:
    """
    A value can fail both axes, and hearing about only one sends the reader looking in the wrong place --
    which the previous ``if``/``elif`` guaranteed.
    """

    def test_a_constructed_strict_subtype_at_an_addr_modify_site(self):
        text = violation({**POINT, **taking("ValueDomain", "Addr", "Modify"), **site(NARROW)})
        assert "Provenance violation" in text
