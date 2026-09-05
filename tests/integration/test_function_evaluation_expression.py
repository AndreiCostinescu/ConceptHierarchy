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
Integration tests: `parse_function_evaluation_expression` called **directly**, as its second caller will.

The function was split out of `_parse_expression_of_json_object` so that the FunctionComposition
instantiation schema's ``"properties": "args"`` node can reuse it: a value like
``{"Add<Integer>": {"arg1": 1, "arg2": 2}}`` reaching `value_instantiation_parser` is a Function evaluation
and must be parsed and checked as one -- argument expressions, required arguments, default grounding and
the acyclicity of the remaining defaults -- rather than merely shape-checked.

Every other test in this suite reaches it through `parse_expression`, which supplies the whole
alternative-search apparatus around it: an ``expr_type``, an ``ensure_expression_invariant`` closure, a
shared ``attempts`` list, and a fallback to `Narrow`/`Inst` when the key turns out not to be a Function.
The schema caller has none of that. So these tests call it with the arguments *that* caller can supply,
and pin what it must be able to read back:

* **the returned triple**, which is a three-way signal, not a result plus two extras -- see
  `TestTheReturnedTriple`. The schema caller has to distinguish "not a type at all", "a type but not a
  Function" and "a Function" itself, because unlike the expression caller it has no other alternative to
  fall back to and must report the middle case as an error;
* **``expr_type=None``**, the schema case: the ``args`` node constrains the Function's *arguments* and says
  nothing about its return type, so there is no expected type to check against;
* **what the resulting `FunctionEvaluation` carries** -- in particular that its ``value_type`` is the
  return type *substituted for this application*, since a caller storing the expression on a parsed value
  has nothing else to recover it from.

See ``documentation/EXPRESSIONS_AND_INSTANTIATION_SCHEMAS.md`` for how the two parsers interlock.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from concept_hierarchy.data.contexts.context import ConceptHierarchyContext
from concept_hierarchy.data.contexts.template_context import TemplateContext
from concept_hierarchy.data.expressions.expression import Expression
from concept_hierarchy.data.expressions.expression_utils import FunctionArgumentAccessor, FunctionArgumentProvenance
from concept_hierarchy.data.expressions.subexpressions import (
    ExpressionAttempt,
    ExpressionKind,
    ExpressionValue,
    FunctionEvaluation,
    IllFormedExpression,
    VerifiedTemplateDependentExpression,
)
from concept_hierarchy.data.parsers.expression_parser import parse_expression, parse_function_evaluation_expression
from concept_hierarchy.data.types.concept_hierarchy_types import TypeValue
from concept_hierarchy.errors import ConceptHierarchyError
from concept_hierarchy.validator.expression_checks import ill_formed_parts, invalid_expression_error
from tests.integration.test_expression_parsing import build_hierarchy, check_concepts, check_hierarchy
from tests.integration.test_function_default_arguments import GROUND, add_like, function
from tests.integration.test_schema_substitution import holder, vd

# --------------------------------------------------------------------------------------------------
# Hierarchy fragments
# --------------------------------------------------------------------------------------------------

ADD = function("Add", {"arg1": ["Integer"], "arg2": ["Integer"], "res": "Integer"}, template=GROUND)
"""``Add(arg1: Integer, arg2: Integer) -> Integer``, ground, so nothing about it waits on an application."""

INCREMENT = function("Increment", {"arg1": ["Integer", "Any", "Modify"], "arg2": ["Integer"]}, template=GROUND)
"""``Increment(arg1: Integer, arg2: Integer)``, ground, so nothing about it waits on an application."""

DEFAULTED = function("Defaulted", {"arg1": ["Integer"], "arg2": ["Integer"], "res": "Integer"}, {"arg2": 3}, GROUND)
CYCLIC = function("Cyclic", {"a": ["Integer"], "b": ["Integer"], "res": "Integer"}, {"a": "b", "b": "a"}, GROUND)
ADD_T = add_like("AddT", {})
"""``AddT<T: Numeric>(arg1: T, arg2: T) -> T``: the return type is decided only by the application."""

BAD_DEFAULT = add_like("BadAdd", {"arg2": "s:x"})
"""``arg2``'s declared default is a String, which no ``BadAdd<Integer>`` can accept."""

MODIFIABLE = function("Modf", {"arg1": ["Integer"], "res": ["Integer", "Any", "Modify"]}, template=GROUND)
ADDRESSABLE = function("Addrf", {"arg1": ["Integer"], "res": ["Integer", "Addr"]}, template=GROUND)
LEAF = vd("Leaf", {"type": "object", "additionalProperties": False})
"""A ValueDomain, i.e. a key that names a type which is not a Function."""

ALL_CONCEPTS = {
    **ADD,
    **INCREMENT,
    **DEFAULTED,
    **CYCLIC,
    **ADD_T,
    **BAD_DEFAULT,
    **MODIFIABLE,
    **ADDRESSABLE,
    **LEAF,
}


# --------------------------------------------------------------------------------------------------
# Calling the function directly
# --------------------------------------------------------------------------------------------------


@dataclass
class Parsed:
    """The whole of what a caller gets back: the returned triple, plus the ``attempts`` it filled in."""

    expressions: list[ExpressionValue]
    key_type: TypeValue | None
    is_function_subtype: bool
    attempts: list[ExpressionAttempt]

    @property
    def evaluation(self) -> FunctionEvaluation:
        assert len(self.expressions) == 1, f"expected one expression, got {self.expressions}"
        value = self.expressions[0]
        assert isinstance(value, FunctionEvaluation), f"expected a FunctionEvaluation, got {value!r}"
        return value

    @property
    def ill_formed(self) -> IllFormedExpression:
        assert len(self.expressions) == 1, f"expected one expression, got {self.expressions}"
        value = self.expressions[0]
        assert isinstance(value, IllFormedExpression), f"expected an IllFormedExpression, got {value!r}"
        return value


_SITE_TYPE_UNSET = object()


@pytest.fixture(scope="module")
def context() -> ConceptHierarchyContext:
    """
    One checked hierarchy for the whole module; the calls below only read from it.

    `check_hierarchy` leaves no template context behind, and the validator reads the ambient one whenever it
    converts a written type -- so the empty context is restored here, which is also what the schema caller
    will be in for a non-templated ValueDomain.
    """
    checked = check_hierarchy(build_hierarchy(ALL_CONCEPTS))
    checked.set_template_context(TemplateContext())
    return checked


def parse_evaluation(
    context: ConceptHierarchyContext,
    key: str,
    value: object,
    *,
    site_type: object = _SITE_TYPE_UNSET,
    recursively_parse: bool = True,
    is_function_evaluation: bool = True,
) -> Parsed:
    """
    Call `parse_function_evaluation_expression` with the arguments the ``args`` schema node can supply.

    ``site_type`` defaults to ``Integer`` -- the return type of every ground Function here -- and is passed
    as ``None`` for the tests that exercise the schema case, where there is no expected type at all.

    The ``ensure_expression_invariant`` given here is the *caller's* half of the protocol, and this one is
    what the schema caller needs: "one expression means stop". The expression parser's version additionally
    asserts on ``expr_type``, which a caller passing ``None`` cannot use.
    """
    resolved_site_type = (
        context.expression_parser_validator.create_instantiated_type("Integer", [])
        if site_type is _SITE_TYPE_UNSET
        else site_type
    )
    attempts: list[ExpressionAttempt] = []
    expressions, key_type, is_function_subtype = parse_function_evaluation_expression(
        key,
        value,
        resolved_site_type,
        TemplateContext(),
        context.expression_parser_validator,
        [],
        recursively_parse,
        False,
        is_function_evaluation,
        lambda produced, _site_type: len(produced) == 1,
        attempts,
    )
    return Parsed(expressions, key_type, is_function_subtype, attempts)


# ==================================================================================================
# 1. The returned triple
# ==================================================================================================


class TestTheReturnedTriple:
    """
    ``(expressions, key_type, is_function_subtype)`` is a three-way signal about the *key*, and a caller
    that reads only the first element cannot tell the three apart.
    """

    def test_a_key_that_is_not_a_type_yields_no_type_and_no_expression(self, context):
        """
        Nothing was attempted, because neither an FEval nor a `Narrow` was ever possible. ``key_type`` is
        ``None``, which is the only case in which it is -- so it is the caller's test for "not a type".
        """
        parsed = parse_evaluation(context, "NoSuchConcept", {"arg1": 1})
        assert parsed.expressions == []
        assert parsed.key_type is None
        assert parsed.is_function_subtype is True, "no type was resolved, so nothing contradicts it"

    def test_a_key_that_is_not_a_type_records_both_alternatives_as_impossible(self, context):
        """
        Both `FEval` and `Narrow` are ruled out by the same fact, and both are recorded: the expression
        caller goes on to try `Inst`, and its diagnosis has to say why the other two never applied.
        """
        parsed = parse_evaluation(context, "NoSuchConcept", {"arg1": 1})
        assert [attempt.kind for attempt in parsed.attempts] == [
            ExpressionKind.FUNCTION_EVALUATION,
            ExpressionKind.NARROW,
        ]

    def test_a_type_that_is_not_a_function_yields_the_type_and_no_expression(self, context):
        """
        The middle case, and the one the schema caller must handle for itself. `Leaf` is a perfectly good
        type, so `Narrow` is still open and *nothing* is recorded as failed -- which is right for the
        expression caller and is exactly why the ``args`` node, where only an evaluation is legal, has to
        read ``is_function_subtype`` and report it.
        """
        parsed = parse_evaluation(context, "Leaf", {})
        assert parsed.expressions == []
        assert parsed.key_type is not None and parsed.key_type.clean_name == "Leaf"
        assert parsed.is_function_subtype is False
        assert parsed.attempts == [], "not being a Function is not a failed attempt"

    def test_a_function_key_yields_an_evaluation(self, context):
        parsed = parse_evaluation(context, "Add", {"arg1": 1, "arg2": 2})
        assert parsed.key_type is not None and parsed.key_type.clean_name == "Add"
        assert parsed.is_function_subtype is True
        assert sorted(parsed.evaluation.arguments) == ["arg1", "arg2"]

    def test_an_abstract_key_is_rejected_as_an_ill_formed_expression(self, context):
        """
        Returned, not raised -- and recorded against *both* alternatives, since the one fact rules out both.

        This is the type-level twin of the "not a concept" branch, and it is returned for the same reason
        every other rejection here is: a caller inside a trial branch must be able to fail that branch
        without the whole parse being torn down.
        """
        parsed = parse_evaluation(context, "ValueDomain", {})
        assert "is an abstract type" in parsed.ill_formed.reason
        assert [attempt.kind for attempt in parsed.attempts] == [
            ExpressionKind.FUNCTION_EVALUATION,
            ExpressionKind.NARROW,
        ]


# ==================================================================================================
# 2. The arguments
# ==================================================================================================


class TestArgumentsAreParsedAndChecked:
    """
    What ``recursively_parse`` buys, and it is the whole reason the ``args`` node calls this rather than
    checking the shape itself.
    """

    def test_each_supplied_argument_is_parsed_against_its_declared_type(self, context):
        arguments = parse_evaluation(context, "Add", {"arg1": 1, "arg2": 2}).evaluation.arguments
        assert sorted(arguments) == ["arg1", "arg2"]
        assert all(argument.is_valid and argument.is_fully_parsed for argument in arguments.values())

    def test_an_argument_the_function_does_not_have_is_rejected(self, context):
        parsed = parse_evaluation(context, "Add", {"arg1": 1, "arg2": 2, "nope": 3})
        assert 'does not have the argument "nope"' in parsed.ill_formed.reason
        assert parsed.attempts[-1].kind is ExpressionKind.FUNCTION_EVALUATION

    def test_a_missing_required_argument_is_rejected(self, context):
        parsed = parse_evaluation(context, "Add", {"arg1": 1})
        assert "are missing from the Function evaluation interface" in parsed.ill_formed.reason
        assert "arg2" in parsed.ill_formed.reason

    def test_an_argument_whose_value_does_not_parse_gives_an_ill_formed_expression(self, context):
        """
        Returned, not raised: the value might still be a `Narrow`, so the expression caller must be able to
        go on. The reason names the argument, and the attempt carries the argument's own failure as its
        ``cause`` so the trace reaches the leaf.
        """
        parsed = parse_evaluation(context, "Add", {"arg1": "s:x", "arg2": 2})
        assert "argument arg1" in parsed.ill_formed.reason
        assert parsed.attempts[-1].kind is ExpressionKind.FUNCTION_EVALUATION
        assert parsed.attempts[-1].cause is not None

    def test_a_non_object_value_gives_an_ill_formed_expression(self, context):
        parsed = parse_evaluation(context, "Add", 5)
        assert "should have the value of the json object an other json object" in parsed.ill_formed.reason
        assert parsed.key_type is not None, "the key was still resolved; only the value was wrong"

    def test_without_recursive_parsing_no_argument_is_looked_at(self, context):
        """
        ``recursively_parse=False`` still produces an evaluation, with no arguments and no checks at all --
        not even the required-argument one. A caller that wants the arguments validated must ask for it.
        """
        evaluation = parse_evaluation(context, "Add", {"arg1": 1}, recursively_parse=False).evaluation
        assert evaluation.arguments == {}
        assert evaluation.applied_defaults == {}


# ==================================================================================================
# 3. Defaults for the arguments the call site leaves out
# ==================================================================================================


class TestUnsuppliedArgumentDefaults:
    """
    Grounding and the acyclicity check run inside this function, so the ``args`` node gets them by calling
    it -- which is the substance of "only a Function Evaluation expression must be valid at that point".
    """

    def test_an_unsupplied_argument_is_grounded_and_recorded(self, context):
        evaluation = parse_evaluation(context, "Defaulted", {"arg1": 1}).evaluation
        assert sorted(evaluation.arguments) == ["arg1"], "still only what the call site wrote"
        assert sorted(evaluation.applied_defaults) == ["arg2"]
        assert evaluation.applied_defaults["arg2"].is_valid

    def test_a_default_that_cannot_hold_for_this_application_gives_an_ill_formed_expression(self, context):
        """``BadAdd<Integer>``'s ``arg2`` defaults to a String; only the application can notice."""
        parsed = parse_evaluation(context, "BadAdd<Integer>", {"arg1": 1})
        assert 'argument "arg2"' in parsed.ill_formed.reason

    def test_cyclic_dependencies_between_the_remaining_defaults_are_rejected(self, context):
        """``Cyclic.a`` defaults to ``b`` and ``b`` to ``a``; supplying neither is a circular dependency."""
        parsed = parse_evaluation(context, "Cyclic", {})
        assert "is not acyclic" in parsed.ill_formed.reason

    def test_supplying_one_of_the_cyclic_pair_breaks_the_cycle(self, context):
        """The check is per call site: with ``a`` written, ``b``'s default has somewhere to land."""
        evaluation = parse_evaluation(context, "Cyclic", {"a": 1}).evaluation
        assert sorted(evaluation.arguments) == ["a"]
        assert sorted(evaluation.applied_defaults) == ["b"]


# ==================================================================================================
# 4. The shape the schema caller passes: no expected type
# ==================================================================================================


class TestTheCallWithoutAnExpectedType:
    """
    At a ``"properties": "args"`` node there is no site type: `FunctionComposition`'s schema constrains the
    key to name a Function and the value to be its arguments, and says nothing about what it returns.
    """

    def test_an_evaluation_is_produced_with_no_expected_type(self, context):
        evaluation = parse_evaluation(context, "Add", {"arg1": 1, "arg2": 2}, site_type=None).evaluation
        assert sorted(evaluation.arguments) == ["arg1", "arg2"]
        assert evaluation.value_type is not None and evaluation.value_type.full_name == "Integer"

    def test_the_is_function_evaluation_flag_is_irrelevant_without_an_expected_type(self, context):
        """
        With no site type there is no `FunctionComposition` alternative to lose to, so the ``args`` caller
        need not decide what to pass -- both spellings produce the evaluation.
        """
        for flag in (True, False):
            parsed = parse_evaluation(
                context, "Add", {"arg1": 1, "arg2": 2}, site_type=None, is_function_evaluation=flag
            )
            assert sorted(parsed.evaluation.arguments) == ["arg1", "arg2"], flag

    def test_defaults_are_still_grounded_without_an_expected_type(self, context):
        """The checks the ``args`` node is being wired up for do not depend on there being a site type."""
        evaluation = parse_evaluation(context, "Defaulted", {"arg1": 1}, site_type=None).evaluation
        assert sorted(evaluation.applied_defaults) == ["arg2"]

    def test_a_cyclic_default_is_still_rejected_without_an_expected_type(self, context):
        parsed = parse_evaluation(context, "Cyclic", {}, site_type=None)
        assert "is not acyclic" in parsed.ill_formed.reason


# ==================================================================================================
# 5. What the produced evaluation carries
# ==================================================================================================


class TestTheEvaluationCarriesTheReturnInterface:
    """
    A caller that stores the expression -- on a `ParsedCustomValue`, say -- has only these fields left; the
    Function's interface is not reachable from the expression.
    """

    def test_the_value_type_is_the_return_type_substituted_for_this_application(self, context):
        """
        ``AddT<T>`` returns ``T``. The evaluation's type must be ``Integer``, not ``AddT:T``: this is read
        back by every consumer, and by the subtype check against the site.
        """
        evaluation = parse_evaluation(context, "AddT<Integer>", {"arg1": 1, "arg2": 2}).evaluation
        assert evaluation.value_type is not None and evaluation.value_type.full_name == "Integer"

    def test_a_result_type_that_does_not_fit_the_site_names_the_substituted_type(self, context):
        """The same substitution, in the diagnosis: ``AddT:T`` there would say nothing to a reader."""
        string_type = context.expression_parser_validator.create_instantiated_type("String", [])
        parsed = parse_evaluation(context, "AddT<Integer>", {"arg1": 1, "arg2": 2}, site_type=string_type)
        assert parsed.ill_formed.reason == "Function result type Integer is not a subtype of String"

    def test_the_result_accessor_and_provenance_are_recorded(self, context):
        """
        ``is_result_addressable`` and ``is_result_modifiable`` come off the return interface and are kept,
        because recomputing them needs the interface the expression no longer holds.
        """
        plain = parse_evaluation(context, "Add", {"arg1": 1, "arg2": 2}).evaluation
        assert (plain.is_result_addressable, plain.is_result_modifiable) == (False, False)

        modifiable = parse_evaluation(context, "Modf", {"arg1": 1}).evaluation
        assert (modifiable.is_result_addressable, modifiable.is_result_modifiable) == (False, True)

        addressable = parse_evaluation(context, "Addrf", {"arg1": 1}).evaluation
        assert (addressable.is_result_addressable, addressable.is_result_modifiable) == (True, False)


# ==================================================================================================
# 6. The four failures that used to be raised
# ==================================================================================================

# The value, and the phrase its diagnosis must contain. Each of these was a bare ``raise CHSemanticError``
# inside `parse_function_evaluation_expression` until they became `IllFormedExpression`\ s.
PREVIOUSLY_RAISED = [
    (
        "an argument the Function does not have",
        {"Add": {"arg1": 1, "arg2": 2, "zz": 3}},
        'does not have the argument "zz"',
    ),
    ("a missing required argument", {"Add": {"arg1": 1}}, "are missing from the Function evaluation interface"),
    ("cyclic dependencies between the remaining defaults", {"Cyclic": {}}, "is not acyclic"),
    ("an abstract key", {"ValueDomain": {}}, "is an abstract type"),
]


def parse_at_site(context: ConceptHierarchyContext, value: object, site_type: TypeValue):
    """
    A whole `parse_expression`, as an ordinary caller makes it -- not the evaluation parser on its own.

    What matters here is what comes *back*: before the conversion these four values left this call by
    raising, so there was no expression to inspect and no decision left to the caller.
    """
    return parse_expression(
        value,
        site_type,
        FunctionArgumentProvenance.ANY,
        FunctionArgumentAccessor.GET,
        TemplateContext(),
        context.expression_parser_validator,
        [],
    )


@pytest.mark.parametrize(("case", "value", "expected"), PREVIOUSLY_RAISED, ids=[c[0] for c in PREVIOUSLY_RAISED])
class TestFailuresAreReturnedAtAGroundSite:
    """
    Three separate claims, and the third is the one that would be silently catastrophic if it slipped:
    the parse must come *back*, it must come back **invalid**, and it must not come back looking parsed.
    """

    @staticmethod
    def _integer_site(context: ConceptHierarchyContext):
        return context.expression_parser_validator.create_instantiated_type("Integer", [])

    def test_the_parse_returns_instead_of_raising(self, context, case, value, expected):
        """No `pytest.raises` here on purpose: reaching the assertion at all is the claim."""
        parsed = parse_at_site(context, value, self._integer_site(context))
        assert parsed is not None

    def test_the_result_is_invalid(self, context, case, value, expected):
        assert parse_at_site(context, value, self._integer_site(context)).is_valid is False

    def test_the_result_is_not_reported_as_fully_parsed(self, context, case, value, expected):
        """
        The failure mode worth guarding: an expression that failed but reports itself as parsed is worse
        than one that raised, because nothing downstream looks at it again.
        """
        assert parse_at_site(context, value, self._integer_site(context)).is_fully_parsed is False

    def test_the_diagnosis_names_the_real_cause(self, context, case, value, expected):
        parsed = parse_at_site(context, value, self._integer_site(context))
        assert any(expected in part.reason for part in ill_formed_parts(parsed)), (
            f"expected {expected!r} among {[part.reason for part in ill_formed_parts(parsed)]}"
        )


class TestFailuresAtATemplateDependentSite:
    """
    A site whose type is still template dependent cannot settle on one alternative:
    ``ensure_expression_invariant`` answers ``None``, every applicable alternative is tried, and the results
    are collected into a `VerifiedTemplateDependentExpression`. So the failure arrives in a **different
    shape** -- invalid, but not itself an `IllFormedExpression` -- and everything that consumes it has to
    know both. `H<T>`'s default site, of type ``T``, is such a site.
    """

    @staticmethod
    def _hierarchy(default: object) -> dict:
        return {**ADD, **CYCLIC, **holder(default)}

    @pytest.mark.parametrize(("case", "value", "expected"), PREVIOUSLY_RAISED, ids=[c[0] for c in PREVIOUSLY_RAISED])
    def test_the_failure_is_diagnosed_and_not_asserted(self, case, value, expected):
        """
        `ConceptHierarchyError` and nothing else. An `AssertionError` would also stop the checker and also
        satisfy a bare ``raises``, and that is exactly what happened here until `ill_formed_parts` learned
        the second shape -- `invalid_expression_error` asserted that an invalid expression *is* an
        `IllFormedExpression`, which at this site it is not.
        """
        with pytest.raises(ConceptHierarchyError) as excinfo:
            check_concepts(self._hierarchy(value))
        assert expected in str(excinfo.value)

    def test_a_valid_evaluation_at_the_same_site_still_checks(self):
        """The passing guard: the shape above must not make every template-dependent site fail."""
        context = check_concepts(self._hierarchy({"Add": {"arg1": 1, "arg2": 2}}))
        assert "H" in context.model.value_domains


class TestTheInvalidExpressionErrorAcceptsBothShapes:
    """
    `ill_formed_parts` in isolation, because the shape it exists for is reachable only through a
    template-dependent site and would otherwise be pinned only indirectly.
    """

    @staticmethod
    def _expression(context: ConceptHierarchyContext, value):
        return Expression(
            context.expression_parser_validator.create_instantiated_type("Integer", []),
            FunctionArgumentProvenance.ANY,
            FunctionArgumentAccessor.GET,
            {"some": "value"},
            value,
        )

    def test_a_bare_ill_formed_expression(self, context):
        ill_formed = IllFormedExpression("boom")
        assert ill_formed_parts(self._expression(context, ill_formed)) == (ill_formed,)

    def test_one_collected_among_template_dependent_alternatives(self, context):
        ill_formed = IllFormedExpression("boom")
        collected = VerifiedTemplateDependentExpression([ill_formed])
        assert ill_formed_parts(self._expression(context, collected)) == (ill_formed,)

    def test_the_error_reports_the_collected_reason(self, context):
        collected = VerifiedTemplateDependentExpression([IllFormedExpression("boom")])
        error = invalid_expression_error(self._expression(context, collected), [])
        assert "boom" in str(error)

    def test_an_expression_with_no_alternative_at_all_still_reports(self, context):
        """``VerifiedTemplateDependentExpression`` with nothing in it is invalid with nothing to point at."""
        empty = VerifiedTemplateDependentExpression([])
        assert ill_formed_parts(self._expression(context, empty)) == ()
        assert "no alternative could be parsed" in str(invalid_expression_error(self._expression(context, empty), []))


class TestTheParseIsNotStoppedByTheFirstFailure:
    """
    What a ``raise`` cost that returning does not: everything after it.

    Two bad evaluations in one value produce two diagnoses. Under a raise the first one ended the parse, so
    the second could never be reported -- which is the difference between "this value has two problems" and
    "this value has a problem, and who knows what else".
    """

    def test_both_failures_in_one_value_are_reported(self):
        multiply = function("Multiply", {"x": ["Integer"], "y": ["Integer"], "res": "Integer"}, template=GROUND)
        pair = vd(
            "Pair",
            {
                "type": "object",
                "properties": {"first": {"type": "Integer"}, "second": {"type": "Integer"}},
                "required": ["first", "second"],
            },
        )
        hierarchy = build_hierarchy(
            {**ADD, **multiply, **pair},
            instances={
                "p": {
                    "Pair": {
                        "first": {"Add": {"arg1": 1, "arg2": 2, "zz": 3}},
                        "second": {"Multiply": {"x": 1, "y": 2, "qq": 4}},
                    }
                }
            },
        )
        with pytest.raises(ConceptHierarchyError) as excinfo:
            check_hierarchy(hierarchy)
        text = str(excinfo.value)
        assert 'does not have the argument "zz"' in text, "the first failure"
        assert 'does not have the argument "qq"' in text, "the second, which a raise would have pre-empted"
