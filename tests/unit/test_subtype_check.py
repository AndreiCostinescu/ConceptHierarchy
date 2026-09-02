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
Unit tests for the template-aware subtype check.

``_general_subtype_check(validator, a, b, template_context, location_id)`` answers "is ``a`` a subtype of
``b``" where either side may be a template variable or a template-dependent type. It returns a
:class:`SubtypeCheckResult` with one of three verdicts:

``YES``
    holds under every instantiation permitted by the template context;
``MAYBE``
    holds under some instantiations -- ``result.template_context`` records the constraint on the template
    variables under which it holds;
``NO``
    holds under no instantiation.

``_check_if_subtype`` is the ``bool`` wrapper the parser uses, read existentially: only ``NO`` is falsy.

The tests below walk the nine combinations of ``{TemplateVariable, TemplateDependentType,
InstantiatedType}`` for ``a`` and ``b``.

Template arguments are matched **invariantly**: ``Box<Integer>`` is not a subtype of ``Box<Number>``.
"""

from __future__ import annotations

import pytest

from concept_hierarchy.data.contexts.template_context import TemplateContext
from concept_hierarchy.data.parsers.expression_parser import (
    SubtypeCheckResult,
    SubtypeVerdict,
    _check_if_subtype,
    _general_subtype_check,
)
from concept_hierarchy.data.types.concept_hierarchy_types import (
    ConceptHierarchyTemplateArgument,
    InstantiatedType,
    NonVariadicTemplateVariable,
    TemplateDependentType,
    TypeValue,
    VariadicTemplateVariable,
)
from concept_hierarchy.models import ConceptHierarchyModel
from concept_hierarchy.validator.checker import ConceptHierarchyChecker

MODEL_DATA = {
    "name": "SubtypeCheckHierarchy",
    "concepts": {
        "Concept": {},
        "ValueDomain": {"directParents": ["Concept"], "data": {}, "abstract": True},
        "Number": {"directParents": ["ValueDomain"], "data": {}},
        "Integer": {"directParents": ["Number"], "data": {}},
        "Str": {"directParents": ["ValueDomain"], "data": {}},
        "Box": {"directParents": ["ValueDomain"], "data": {"templateContext": {"order": ["T"], "T": "ValueDomain"}}},
        # NumBox extends Box<Number> -- a ground substitution
        "NumBox": {"directParents": ["Box"], "data": {"templateContext": {"substitution": {"Box:T": "Number"}}}},
        # TwinBox<E> extends Box<E> -- a template-dependent substitution.
        # `E` is deliberately *not* named `T`: a template variable that shares a name with the parent's own
        # template parameter is resolved as that parameter instead, which silently changes the answer.
        "TwinBox": {
            "directParents": ["Box"],
            "data": {"templateContext": {"order": ["E"], "E": "ValueDomain", "substitution": {"Box:T": "E"}}},
        },
        # N is constrained to Number; used to show a derived constraint that contradicts the declared one
        "NumOnly": {"directParents": ["ValueDomain"], "data": {"templateContext": {"order": ["N"], "N": "Number"}}},
        # a *literal* template variable (a number, not a type)
        "Vector": {"directParents": ["ValueDomain"], "data": {"templateContext": {"order": ["N"], "N": 3}}},
    },
}


# --------------------------------------------------------------------------------------------------
# Scaffold
# --------------------------------------------------------------------------------------------------


class Hierarchy:
    """A checked hierarchy plus the bits a subtype check needs."""

    def __init__(self, model_data: dict):
        model = ConceptHierarchyModel.create_from_data(model_data)
        checker = ConceptHierarchyChecker(model, lambda _concept, _instance: None)
        checker.check()
        self.context = checker.context
        self.validator = self.context.expression_parser_validator

    def template_context(self, concept_name: str) -> TemplateContext:
        """The declared template context of a concept, e.g. ``TwinBox``'s ``<E : ValueDomain>``."""
        return self.context.model.concepts[concept_name].template_context

    def check(self, a: TypeValue, b: TypeValue, in_context: str = "TwinBox") -> SubtypeCheckResult:
        """
        Check ``a <: b`` with `in_context`'s template context in scope.

        The ambient template context is set as well as passed, because the validator resolves literal
        template variables against the ambient one -- during a real parse the two always coincide.
        """
        template_context = self.template_context(in_context)
        self.context.set_template_context(template_context)
        return _general_subtype_check(self.validator, a, b, template_context, [])

    def check_bool(self, a: TypeValue, b: TypeValue, in_context: str = "TwinBox") -> bool:
        template_context = self.template_context(in_context)
        self.context.set_template_context(template_context)
        return _check_if_subtype(self.validator, a, b, template_context, [])


@pytest.fixture(scope="module")
def hierarchy() -> Hierarchy:
    """Checking a hierarchy is expensive and noisy, so build it once for the whole module."""
    return Hierarchy(MODEL_DATA)


def inst(name: str, *template_arguments: ConceptHierarchyTemplateArgument) -> InstantiatedType:
    """A fully instantiated type, e.g. ``inst("Box", inst("Number"))`` for ``Box<Number>``."""
    return InstantiatedType(name, template_arguments)


def tvar(name: str, defined_in: str) -> NonVariadicTemplateVariable:
    """A template variable, e.g. ``tvar("E", "TwinBox")`` for ``TwinBox``'s ``E``."""
    return NonVariadicTemplateVariable(name, defined_in)


def tdep(name: str, *template_arguments: ConceptHierarchyTemplateArgument) -> TemplateDependentType:
    """A template-dependent type, e.g. ``tdep("Box", tvar("E", "TwinBox"))`` for ``Box<E>``."""
    return TemplateDependentType(name, template_arguments)


def constraint_of(result: SubtypeCheckResult) -> str:
    """The ``repr`` of a MAYBE's derived constraint, for readable assertions."""
    assert result.verdict is SubtypeVerdict.MAYBE, f"expected MAYBE, got {result.verdict.name}"
    assert result.template_context is not None
    return repr(result.template_context.constraint)


# Types used across the tests. `E` is TwinBox's template variable.
INTEGER = inst("Integer")
NUMBER = inst("Number")
STR = inst("Str")
VALUE_DOMAIN = inst("ValueDomain")
BOX_NUMBER = inst("Box", NUMBER)
BOX_INTEGER = inst("Box", INTEGER)
NUMBOX = inst("NumBox")
E = tvar("E", "TwinBox")
BOX_E = tdep("Box", E)
TWINBOX_E = tdep("TwinBox", E)


# --------------------------------------------------------------------------------------------------
# Case 9: InstantiatedType x InstantiatedType -- the answer is always definite
# --------------------------------------------------------------------------------------------------


class TestGroundSubtyping:
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            (INTEGER, INTEGER, SubtypeVerdict.YES),
            (INTEGER, NUMBER, SubtypeVerdict.YES),
            (INTEGER, VALUE_DOMAIN, SubtypeVerdict.YES),
            (NUMBER, INTEGER, SubtypeVerdict.NO),
            (STR, NUMBER, SubtypeVerdict.NO),
            (NUMBOX, BOX_NUMBER, SubtypeVerdict.YES),
        ],
        ids=["reflexive", "direct-parent", "transitive-parent", "parent-not-child", "unrelated", "ground-substitution"],
    )
    def test_verdict(self, hierarchy: Hierarchy, a, b, expected):
        assert hierarchy.check(a, b).verdict is expected

    def test_never_maybe(self, hierarchy: Hierarchy):
        """With no template variables anywhere there is nothing to defer."""
        for a, b in [(INTEGER, NUMBER), (STR, NUMBER), (NUMBOX, BOX_NUMBER)]:
            assert hierarchy.check(a, b).verdict is not SubtypeVerdict.MAYBE

    def test_agrees_with_the_ground_validator(self, hierarchy: Hierarchy):
        """The template-aware check must not change the answer for fully instantiated types."""
        ground = hierarchy.context.type_application_constraints_validator
        for a, b in [(INTEGER, NUMBER), (NUMBER, INTEGER), (NUMBOX, BOX_NUMBER), (BOX_INTEGER, BOX_NUMBER)]:
            expected = ground.is_a_subtype_of_b(a, b, [])
            assert bool(hierarchy.check(a, b)) is expected, f"disagreement on {a} <: {b}"


class TestTemplateArgumentsAreInvariant:
    def test_a_subtype_argument_does_not_make_a_subtype(self, hierarchy: Hierarchy):
        """`Integer` is a subtype of `Number`, but `Box<Integer>` is *not* a subtype of `Box<Number>`."""
        assert hierarchy.check(INTEGER, NUMBER).verdict is SubtypeVerdict.YES
        assert hierarchy.check(BOX_INTEGER, BOX_NUMBER).verdict is SubtypeVerdict.NO

    def test_a_supertype_argument_does_not_make_a_subtype(self, hierarchy: Hierarchy):
        assert hierarchy.check(BOX_INTEGER, inst("Box", VALUE_DOMAIN)).verdict is SubtypeVerdict.NO

    def test_identical_arguments_do(self, hierarchy: Hierarchy):
        assert hierarchy.check(BOX_NUMBER, BOX_NUMBER).verdict is SubtypeVerdict.YES


# --------------------------------------------------------------------------------------------------
# Cases 7 and 8: b is fully instantiated, a is not
# --------------------------------------------------------------------------------------------------


class TestSubtypeOfInstantiatedType:
    def test_template_variable_is_deferred(self, hierarchy: Hierarchy):
        """Case 7. Whether `E <: Number` holds depends on what `E` becomes, so it is recorded."""
        result = hierarchy.check(E, NUMBER)
        assert result.verdict is SubtypeVerdict.MAYBE
        assert "Number" in constraint_of(result)

    def test_template_dependent_type_that_can_not_match(self, hierarchy: Hierarchy):
        """Case 8. No instantiation of `E` makes a `TwinBox` a `Str`."""
        assert hierarchy.check(TWINBOX_E, STR).verdict is SubtypeVerdict.NO

    def test_template_dependent_type_that_always_matches(self, hierarchy: Hierarchy):
        """Case 8. `Box<E>` is a `ValueDomain` whatever `E` is, so nothing needs constraining."""
        assert hierarchy.check(BOX_E, VALUE_DOMAIN).verdict is SubtypeVerdict.YES

    def test_template_dependent_type_that_constrains_the_variable(self, hierarchy: Hierarchy):
        """Case 8. `Box<E> <: Box<Number>` exactly when `E` is `Number` -- invariantly, so an exact match."""
        result = hierarchy.check(BOX_E, BOX_NUMBER)
        assert result.verdict is SubtypeVerdict.MAYBE
        assert "Number" in constraint_of(result)

    def test_constraint_is_derived_through_the_parent_substitution(self, hierarchy: Hierarchy):
        """`TwinBox<E>` extends `Box<E>`, so the constraint on `E` comes out of the substitution."""
        result = hierarchy.check(TWINBOX_E, BOX_NUMBER)
        assert result.verdict is SubtypeVerdict.MAYBE
        assert "Number" in constraint_of(result)


# --------------------------------------------------------------------------------------------------
# Cases 4, 5 and 6: b is a template-dependent type application
# --------------------------------------------------------------------------------------------------


class TestSubtypeOfTypeApplication:
    def test_instantiated_type_constrains_the_variable(self, hierarchy: Hierarchy):
        """Case 6. `Box<Number> <: Box<E>` exactly when `E` is `Number`."""
        result = hierarchy.check(BOX_NUMBER, BOX_E)
        assert result.verdict is SubtypeVerdict.MAYBE
        assert "Number" in constraint_of(result)

    def test_constraint_is_derived_through_a_ground_substitution(self, hierarchy: Hierarchy):
        """Case 6. `NumBox` extends `Box<Number>`, so `NumBox <: Box<E>` also pins `E` to `Number`."""
        result = hierarchy.check(NUMBOX, BOX_E)
        assert result.verdict is SubtypeVerdict.MAYBE
        assert "Number" in constraint_of(result)

    def test_unrelated_concept_is_rejected_outright(self, hierarchy: Hierarchy):
        """Case 6. `Integer` is not a `Box` under any instantiation of `E`."""
        assert hierarchy.check(INTEGER, BOX_E).verdict is SubtypeVerdict.NO

    def test_template_variable_against_a_type_application(self, hierarchy: Hierarchy):
        """Case 4. `E <: Box<E>` is deferred as a constraint on `E`."""
        result = hierarchy.check(E, BOX_E)
        assert result.verdict is SubtypeVerdict.MAYBE
        assert "Box" in constraint_of(result)

    def test_template_dependent_against_template_dependent(self, hierarchy: Hierarchy):
        """Case 5. `TwinBox<E> <: Box<E>` holds whenever `E` is itself -- a tautological constraint."""
        assert hierarchy.check(TWINBOX_E, BOX_E).verdict is SubtypeVerdict.MAYBE


# --------------------------------------------------------------------------------------------------
# Cases 1, 2 and 3: b is a bare template variable
# --------------------------------------------------------------------------------------------------


class TestSubtypeOfTemplateVariable:
    def test_the_same_variable_is_its_own_subtype(self, hierarchy: Hierarchy):
        """Case 1. Reflexivity, whatever `E` is instantiated to."""
        assert hierarchy.check(E, E).verdict is SubtypeVerdict.YES

    def test_reflexivity_uses_the_defining_location(self, hierarchy: Hierarchy):
        """Two variables that merely share a name are not the same variable."""
        other_e = tvar("E", "SomewhereElse")
        assert E.full_name != other_e.full_name
        assert hierarchy.check(E, other_e).verdict is not SubtypeVerdict.YES

    def test_instantiated_type_constrains_the_supertype_variable(self, hierarchy: Hierarchy):
        """Case 3. `Integer <: E` when `E` becomes `Integer` or one of its supertypes."""
        result = hierarchy.check(INTEGER, E)
        assert result.verdict is SubtypeVerdict.MAYBE
        assert "Integer" in constraint_of(result)

    def test_the_declared_constraint_is_kept(self, hierarchy: Hierarchy):
        """The derived constraint is conjoined with the variable's own declared constraint."""
        result = hierarchy.check(INTEGER, E)
        assert "ValueDomain" in constraint_of(result), (
            f"expected E's declared `ValueDomain` constraint to survive: {constraint_of(result)}"
        )

    def test_template_dependent_type_constrains_the_supertype_variable(self, hierarchy: Hierarchy):
        """Case 2. `Box<E> <: E` -- note this is an occurs check; see the xfail below."""
        result = hierarchy.check(BOX_E, E)
        assert result.verdict is SubtypeVerdict.MAYBE
        assert "Box" in constraint_of(result)


# --------------------------------------------------------------------------------------------------
# The bool wrapper and result invariants
# --------------------------------------------------------------------------------------------------

ALL_CASES = [
    pytest.param(E, E, id="1-tvar-x-tvar"),
    pytest.param(BOX_E, E, id="2-tdep-x-tvar"),
    pytest.param(INTEGER, E, id="3-inst-x-tvar"),
    pytest.param(E, BOX_E, id="4-tvar-x-tdep"),
    pytest.param(TWINBOX_E, BOX_E, id="5-tdep-x-tdep"),
    pytest.param(NUMBOX, BOX_E, id="6-inst-x-tdep"),
    pytest.param(INTEGER, BOX_E, id="6-inst-x-tdep-no"),
    pytest.param(E, NUMBER, id="7-tvar-x-inst"),
    pytest.param(TWINBOX_E, BOX_NUMBER, id="8-tdep-x-inst"),
    pytest.param(TWINBOX_E, STR, id="8-tdep-x-inst-no"),
    pytest.param(INTEGER, NUMBER, id="9-inst-x-inst"),
    pytest.param(STR, NUMBER, id="9-inst-x-inst-no"),
]
"""Every one of the nine combinations, so the invariants below are checked across the whole matrix."""


class TestAllCasesAreAnswered:
    @pytest.mark.parametrize(("a", "b"), ALL_CASES)
    def test_every_combination_produces_a_verdict(self, hierarchy: Hierarchy, a, b):
        assert hierarchy.check(a, b).verdict in set(SubtypeVerdict)

    @pytest.mark.parametrize(("a", "b"), ALL_CASES)
    def test_bool_wrapper_agrees_with_the_verdict(self, hierarchy: Hierarchy, a, b):
        """Existential reading: only NO is falsy."""
        result = hierarchy.check(a, b)
        assert hierarchy.check_bool(a, b) is (result.verdict is not SubtypeVerdict.NO)

    @pytest.mark.parametrize(("a", "b"), ALL_CASES)
    def test_a_constraint_is_carried_exactly_for_maybe(self, hierarchy: Hierarchy, a, b):
        result = hierarchy.check(a, b)
        if result.verdict is SubtypeVerdict.MAYBE:
            assert result.template_context is not None, "a MAYBE must say under which constraint it holds"
        else:
            assert result.template_context is None, f"{result.verdict.name} must not carry a constraint"

    @pytest.mark.parametrize(("a", "b"), ALL_CASES)
    def test_errors_are_reported_only_for_no(self, hierarchy: Hierarchy, a, b):
        result = hierarchy.check(a, b)
        if result.verdict is not SubtypeVerdict.NO:
            assert result.errors == ()


# --------------------------------------------------------------------------------------------------
# Operand guards
# --------------------------------------------------------------------------------------------------


class TestOperandGuards:
    def test_variadic_subtype_is_rejected(self, hierarchy: Hierarchy):
        variadic = VariadicTemplateVariable("E", "TwinBox")
        with pytest.raises(RuntimeError, match="checked as subtype is a variadic template variable"):
            hierarchy.check(variadic, NUMBER)

    def test_variadic_supertype_is_rejected(self, hierarchy: Hierarchy):
        variadic = VariadicTemplateVariable("E", "TwinBox")
        with pytest.raises(RuntimeError, match="checked as supertype is a variadic template variable"):
            hierarchy.check(NUMBER, variadic)

    def test_literal_template_variable_is_rejected(self, hierarchy: Hierarchy):
        """`Vector<N>` declares `N` as the literal `3`, which can never stand for a type."""
        literal_variable = tvar("N", "Vector")
        with pytest.raises(RuntimeError, match="is a literal template variable"):
            hierarchy.check(literal_variable, NUMBER, in_context="Vector")


# --------------------------------------------------------------------------------------------------
# Known limitations
# --------------------------------------------------------------------------------------------------


class TestKnownLimitations:
    """
    Non-strict xfails: each pins a limitation that is understood and deliberately not fixed yet, so
    closing one reports XPASS rather than breaking the suite.
    """

    @pytest.mark.xfail(
        reason="TemplateConstraintHierarchyOperator.is_empty is hardcoded False, so a derived constraint "
        "that contradicts the variable's declared constraint can not be detected -- see its FIXME",
        strict=False,
    )
    def test_a_contradicting_constraint_is_detected(self, hierarchy: Hierarchy):
        """`NumOnly`'s `N` must be a `Number`, so `N <: Str` is impossible -- but comes back as MAYBE."""
        n = tvar("N", "NumOnly")
        assert hierarchy.check(n, STR, in_context="NumOnly").verdict is SubtypeVerdict.NO

    @pytest.mark.xfail(
        reason="the occurs check is not implemented: `Box<E> <: E` constrains E in terms of E itself",
        strict=False,
    )
    def test_an_infinite_type_is_rejected(self, hierarchy: Hierarchy):
        assert hierarchy.check(BOX_E, E).verdict is SubtypeVerdict.NO

    @pytest.mark.xfail(
        reason="`TwinBox<E> <: Box<E>` holds for every E, but is reported as MAYBE under the "
        "tautological constraint `E is E` rather than recognised as YES",
        strict=False,
    )
    def test_a_tautological_constraint_is_recognised_as_yes(self, hierarchy: Hierarchy):
        assert hierarchy.check(TWINBOX_E, BOX_E).verdict is SubtypeVerdict.YES
