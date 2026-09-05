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
Integration tests: a global variable whose value can only be determined by determining it again.

Two cycle checks over references between *names* existed before this one, and each was blind to the
other's edges:

* the **alias graph** (`ConceptHierarchyChecker.order_aliases`) has an edge only where a variable's value
  *is* a bare name, so ``a = b``, ``b = a`` is rejected but ``v = {"Loopy": {}}`` is not an edge at all;
* the **sibling-argument graph** (`FunctionData.default_argument_dependencies`) has an edge only between
  arguments of one Function, so ``Loopy.x := v`` -- naming a global rather than a sibling -- is not one
  either.

A cycle that alternates between the two therefore passed both. It is a genuine infinite regress:
evaluating ``v`` needs ``Loopy()``, which needs ``x``, which is ``v``. Unlike an instantiation default,
which may legitimately be left undecided because the key can be supplied instead, a global variable's
value has no such alternative -- it must simply be decided -- so this is an **error**.

The one exception is a global of `CustomFunction` type: its value is a *procedure*, and a procedure that
names the variable holding it is ordinary recursion rather than a circular initialisation.
`animal_kingdom.json`'s ``factorial`` is exactly that.

What makes the check see these edges at all is `FunctionEvaluation.applied_defaults`: an evaluation
records what each unsupplied argument fell back on, and `expression_checks.expression_dependencies` walks
it alongside `get_subexpressions`. See ``documentation/TODO_GLOBAL_DEFAULT_CYCLES.md``.
"""

from __future__ import annotations

import pytest

from concept_hierarchy.data.expressions.expression import Expression
from concept_hierarchy.data.expressions.expression_utils import FunctionArgumentAccessor, FunctionArgumentProvenance
from concept_hierarchy.data.expressions.subexpressions import TemplateDependentExpression, Variable
from concept_hierarchy.data.types.concept_hierarchy_types import InstantiatedType
from concept_hierarchy.errors import ConceptHierarchyError
from concept_hierarchy.validator import expression_checks
from concept_hierarchy.validator.expression_checks import find_recursive_global_variable_initialisation
from tests.integration.test_expression_parsing import build_hierarchy, check_hierarchy
from tests.integration.test_function_default_arguments import GROUND, function
from tests.integration.test_schema_substitution import obj, vd

# --------------------------------------------------------------------------------------------------
# Hierarchy fragments
# --------------------------------------------------------------------------------------------------


def integer_function(name: str, argument: str, default: object) -> dict:
    """``name(argument: Integer) -> Integer``, with ``argument`` defaulting to ``default``.

    Ground on purpose: a template variable would make the *result* type undecided, and a global's type is
    read off its value expression, so the hierarchy would fail for a reason that is not the one under test.
    """
    return function(name, {argument: ["Integer"], "res": "Integer"}, {argument: default}, template=GROUND)


LEAF = vd("Leaf", {"type": "object", "additionalProperties": False})
"""A ValueDomain that instantiates from ``{}`` and forces no default."""

C1 = vd("C1", obj({"p": {"type": "Integer", "default": "v"}}))
"""One whose absent ``p`` materialises a default that names the global ``v``."""

MAKE_W = function(
    "MakeW",
    {"a": ["T"], "res": "Integer"},
    {"a": {}},
    template={"order": ["T"], "T": "ValueDomain", "substitution": {"FunctionReturning:T": "Integer"}},
)
"""``MakeW<T>(a: T)``: one declared default, ``{}``, whose meaning is settled only by the application."""

CUSTOM_FUNCTION = vd("CustomFunction", obj({"procedure": {"type": "CustomFunction"}}))
"""
Enough of a `CustomFunction` for the exception: a value whose ``procedure`` may name another one.

`CH_PRELUDE` does not define it -- it is a concept of the standard prelude rather than of the language --
and `is_a_custom_function_global_variable` answers ``False`` for a hierarchy that leaves it out, which is
why every other hierarchy in this file is unaffected by the exception.
"""


# --------------------------------------------------------------------------------------------------
# Assertions
# --------------------------------------------------------------------------------------------------


def assert_rejected_as_a_global_cycle(concepts: dict, instances: dict, *named: str) -> str:
    """
    The hierarchy must be rejected, and rejected *as a diagnosed recursive initialisation*.

    The distinction is the whole point, exactly as in `assert_rejected_as_a_cycle`: a runaway that hit the
    default-expansion depth bound, or the interpreter's stack, would satisfy a bare ``raises`` while being
    a far worse outcome reported far later. Each name in ``named`` must appear, so the report identifies
    the global at fault and the chain that closes on it.
    """
    with pytest.raises(ConceptHierarchyError) as excinfo:
        check_hierarchy(build_hierarchy(concepts, instances=instances))
    text = str(excinfo.value)
    assert "can never be initialised" in text, f"expected a recursive-initialisation diagnosis, got: {text[:600]}"
    assert "levels deep" not in text, "the cycle must be diagnosed, not left to the depth bound"
    assert "Ran out of stack" not in text, "the cycle must be diagnosed, not left to the interpreter"
    for name in named:
        assert name in text, f"expected {name!r} to be named in: {text[:600]}"
    return text


def global_value(context, name: str) -> Expression:
    return context.model.instances[name].value


# ==================================================================================================
# 1. The cycles that must be rejected
# ==================================================================================================


class TestGlobalVariableCyclesAreRejected:
    """
    Each of these is accepted by both existing graphs and is nonetheless a value that does not exist.
    """

    def test_a_global_evaluating_a_function_whose_default_names_it(self):
        """
        The motivating case: ``v = {"Loopy": {}}`` with ``Loopy.x`` defaulting to ``v``.

        Neither existing graph has this edge -- ``v``'s value is not a bare name, and ``x``'s default names
        a global rather than a sibling.
        """
        text = assert_rejected_as_a_global_cycle(integer_function("Loopy", "x", "v"), {"v": {"Loopy": {}}}, "v")
        assert "v -> v" in text, f"expected the chain to be reported, got: {text[:600]}"

    def test_the_reference_may_be_written_under_an_alias(self):
        """
        ``w = v`` makes ``w`` a second *name* for ``v``, not a second variable, so ``L2.x := w`` is the
        same edge. The path holds canonical names and the query canonicalises, which is what closes it.
        """
        assert_rejected_as_a_global_cycle(integer_function("L2", "x", "w"), {"v": {"L2": {}}, "w": "v"}, "v")

    def test_the_cycle_may_run_through_an_instantiation_default(self):
        """
        ``MakeW<C1>``'s unsupplied ``a`` defaults to ``{}``, which instantiates a `C1` whose absent ``p``
        materialises *its* default -- and that names ``v``. No extra code: it is the same walk.
        """
        assert_rejected_as_a_global_cycle({**LEAF, **C1, **MAKE_W}, {"v": {"MakeW<C1>": {}}}, "v")

    def test_the_cycle_may_run_two_functions_deep(self):
        """``F.x`` defaults to an evaluation of ``G``, whose own unsupplied ``y`` defaults to ``v``."""
        concepts = {**integer_function("F", "x", {"G": {}}), **integer_function("G", "y", "v")}
        assert_rejected_as_a_global_cycle(concepts, {"v": {"F": {}}}, "v")

    def test_the_reference_may_sit_in_a_supplied_argument_of_a_nested_evaluation(self):
        """
        ``H.x`` defaults to ``{"WrapF": {"w": v}}``. The nested evaluation *supplies* ``w``, so this edge
        is an ordinary sub-expression -- reached only because the walk got into ``H``'s applied default in
        the first place.
        """
        concepts = {
            **integer_function("H", "x", {"WrapF": {"w": "v"}}),
            **function("WrapF", {"w": ["Integer"], "res": "Integer"}, template=GROUND),
        }
        assert_rejected_as_a_global_cycle(concepts, {"v": {"H": {}}}, "v")

    @pytest.mark.parametrize("declaration_order", [("v", "u"), ("u", "v")])
    def test_a_cycle_spanning_two_globals_is_found_whichever_is_declared_first(self, declaration_order):
        """
        ``v`` needs ``u`` and ``u`` needs ``v``, with a Function default on each leg -- so neither leg is
        an alias edge and the two are not siblings.

        Order must not matter, and the walk does not assume it does not: the global reached first has no
        parsed value yet and is simply not descended into. The cycle is found from the *other* end, when
        that one's turn comes and both values exist.
        """
        concepts = {**integer_function("Fu", "x", "u"), **integer_function("Gv", "y", "v")}
        values = {"v": {"Fu": {}}, "u": {"Gv": {}}}
        instances = {name: values[name] for name in declaration_order}
        text = assert_rejected_as_a_global_cycle(concepts, instances, "v", "u")
        assert "-> u ->" in text or "-> v ->" in text, f"expected both legs in the chain, got: {text[:600]}"


# ==================================================================================================
# 2. The references that must still be accepted
# ==================================================================================================


class TestBenignGlobalReferencesAreAccepted:
    """
    The passing guards. A check that answered "cycle" for every global reference would satisfy every test
    above and be useless; these are what say it does not.
    """

    def test_supplying_the_argument_grounds_no_default(self):
        """``{"Loopy": {"x": 1}}`` never applies ``x``'s default, so the edge does not exist here."""
        context = check_hierarchy(
            build_hierarchy(integer_function("Loopy", "x", "v"), instances={"v": {"Loopy": {"x": 1}}})
        )
        assert global_value(context, "v").is_valid

    def test_a_global_reference_with_no_way_back(self):
        """``Loopy.x`` names ``u``, and ``u`` is a literal -- a dependency, not a cycle."""
        context = check_hierarchy(
            build_hierarchy(integer_function("Loopy", "x", "u"), instances={"v": {"Loopy": {}}, "u": 1})
        )
        assert global_value(context, "v").is_valid

    def test_a_later_global_may_name_an_earlier_one(self):
        """
        The path is per global and is cleared when its turn ends. ``u`` legitimately reaches ``v``, which
        was resolved first; a path left standing from ``v``'s turn would call this a cycle.
        """
        context = check_hierarchy(
            build_hierarchy(integer_function("Loopy", "x", "v"), instances={"v": 1, "u": {"Loopy": {}}})
        )
        assert global_value(context, "u").is_valid

    def test_two_globals_may_evaluate_the_same_function(self):
        """A shared referent is not a cycle: both reach ``u``, and neither reaches itself."""
        context = check_hierarchy(
            build_hierarchy(
                integer_function("Loopy", "x", "u"), instances={"v": {"Loopy": {}}, "u": 1, "z": {"Loopy": {}}}
            )
        )
        assert global_value(context, "v").is_valid and global_value(context, "z").is_valid

    def test_the_plain_alias_case_still_checks(self):
        """``w = v`` is what the alias graph is for, and adding this check must not disturb it."""
        context = check_hierarchy(build_hierarchy({}, instances={"v": 1, "w": "v"}))
        assert "w" not in context.model.instances, "an alias is a name for a variable, not a second variable"
        assert context.ch.canonical_variable_name("w") == "v"

    def test_the_edge_is_per_application(self):
        """
        ``MakeW<C1>`` is cyclic and ``MakeW<Leaf>`` is not, from the *same* declared default ``{}``: only
        the application decides what it instantiates, and only the application it is used at is walked.

        The cyclic ``C1`` is still in the hierarchy here -- what makes the difference is that no site
        applies ``MakeW<C1>``.
        """
        context = check_hierarchy(build_hierarchy({**LEAF, **C1, **MAKE_W}, instances={"v": {"MakeW<Leaf>": {}}}))
        assert global_value(context, "v").is_valid


# ==================================================================================================
# 3. The CustomFunction exception
# ==================================================================================================


class TestTheCustomFunctionException:
    """
    A `CustomFunction` global holds a procedure, and a procedure naming itself is recursion, not a value
    whose initialisation needs itself. Such a global is never put on the path and never descended into --
    so the check can only fail to fire on one, never fire wrongly.
    """

    def test_a_custom_function_global_may_name_itself(self):
        """`animal_kingdom.json`'s ``factorial``, reduced to the part that matters."""
        context = check_hierarchy(
            build_hierarchy(CUSTOM_FUNCTION, instances={"selfy": {"CustomFunction": {"procedure": "selfy"}}})
        )
        assert global_value(context, "selfy").is_valid

    @pytest.mark.parametrize("declaration_order", [("selfy", "g"), ("g", "selfy")])
    def test_a_custom_function_global_reached_from_an_ordinary_one(self, declaration_order):
        """
        ``g``'s evaluation applies a default that names ``selfy``. The walk stops there rather than
        descending into a procedure that names itself -- which, without the exception, would be reported
        as a cycle that is not one.
        """
        concepts = {
            **CUSTOM_FUNCTION,
            **function("UsesF", {"f": ["CustomFunction"], "res": "Integer"}, {"f": "selfy"}, template=GROUND),
        }
        values = {"selfy": {"CustomFunction": {"procedure": "selfy"}}, "g": {"UsesF": {}}}
        context = check_hierarchy(
            build_hierarchy(concepts, instances={name: values[name] for name in declaration_order})
        )
        assert global_value(context, "g").is_valid


# ==================================================================================================
# 4. What closes the cycle: where the name resolved, not the name
# ==================================================================================================


class TestTheWalkTestsWhereTheNameResolved:
    """
    A reference closes the cycle only when it resolved **at the global scope**. Today no hierarchy can
    shadow a global -- a Function argument, a concept property or a concept function sharing a global's
    name is rejected in `check_after_parsing_concepts` -- so the two halves are pinned on the walk itself,
    which is the only place that would notice if the guard elsewhere were ever relaxed.

    Calling it directly is also what says the resolution path is nothing but the ``chain`` argument: there
    is no scope to enter first and no state to reset afterwards.
    """

    @staticmethod
    def _reference_to(name: str, scope_index: int) -> Expression:
        return Expression(
            InstantiatedType("Integer", ()),
            FunctionArgumentProvenance.ANY,
            FunctionArgumentAccessor.GET,
            name,
            Variable(name, InstantiatedType("Integer", ()), scope_index=scope_index),
        )

    def _context_with_a_global(self):
        return check_hierarchy(build_hierarchy({}, instances={"v": 1}))

    def test_a_reference_that_resolved_at_the_global_scope_closes_the_cycle(self):
        found = find_recursive_global_variable_initialisation(
            self._context_with_a_global(), self._reference_to("v", scope_index=0), ["v"]
        )
        assert found == ["v", "v"]

    def test_the_same_name_resolved_in_a_local_frame_does_not(self):
        """A Function's argument or a nested call's introduces names above the globals and shadows them."""
        found = find_recursive_global_variable_initialisation(
            self._context_with_a_global(), self._reference_to("v", scope_index=1), ["v"]
        )
        assert found is None


# ==================================================================================================
# 5. The invariant: a global's value is decided
# ==================================================================================================


class TestGlobalValuesAreDecided:
    """
    A global variable's value must be neither template dependent nor partially parsed. There is no
    application still to come that could settle it, and no "supply the key instead" alternative of the kind
    that makes an undecided instantiation default legitimate.

    The assertion is vacuous on every hierarchy today, which is exactly why it needs a test that builds the
    undecided value directly -- otherwise removing it changes nothing that anything notices.
    """

    def test_every_global_of_a_checked_hierarchy_is_decided(self):
        context = check_hierarchy(
            build_hierarchy(integer_function("Loopy", "x", "u"), instances={"v": {"Loopy": {}}, "u": 1, "s": "s:x"})
        )
        for name in ("v", "u", "s"):
            value = global_value(context, name)
            assert value.is_fully_parsed and not value.is_value_template_dependent, name

    def test_an_undecided_value_is_rejected(self, monkeypatch):
        """
        No natural hierarchy produces one, so the parse of the global is replaced by an expression that
        was never walked into -- the shape a template-dependent site leaves behind.
        """
        real_parse_expression = expression_checks.parse_expression

        def parse_the_global_as_undecided(json_value, expression_type, provenance, accessor, *args, **kwargs):
            location_id = args[2]
            if [str(part) for part in location_id] == ["instances", "v"]:
                return Expression(
                    expression_type,
                    provenance,
                    accessor,
                    json_value,
                    TemplateDependentExpression(expression_type),
                )
            return real_parse_expression(json_value, expression_type, provenance, accessor, *args, **kwargs)

        monkeypatch.setattr(expression_checks, "parse_expression", parse_the_global_as_undecided)
        with pytest.raises(ConceptHierarchyError, match="is not decided"):
            check_hierarchy(build_hierarchy({}, instances={"v": 1}))
