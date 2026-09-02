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
Integration tests: expression parsing in a full Concept Hierarchy definition.

Scaffold
--------
A test writes a Concept Hierarchy the same way a user does -- as JSON data -- runs the real
:class:`ConceptHierarchyChecker` over it, and then asserts on the :class:`Expression` objects the parser
produced. :func:`check_hierarchy` builds the context, and the ``*_expressions`` accessors below pull the
parsed expressions back out of the places the checker stores them.

Expressions are parsed in two places today (``expression_checks.check_expressions_in_concept_hierarchy``):

- **Function default argument values** -- stored in ``FunctionData.evaluation_argument_default_value``;
  read with :func:`function_default_expressions`.
- **ValueDomain instantiation defaults** -- stored in ``CHSchemaNode.default_expr``, replaced in place;
  read with :func:`instantiation_default_expressions`.
- **Global variables** -- only their *type* is inferred, into ``GlobalVariableData.value_type``;
  read with :func:`global_variable_types`.

Everything else -- property defaults and constraints, Function procedures, computations, hooks,
variations, inversions -- is not parsed yet: ``check_expressions_in_domain_concept_definition``,
``check_expressions_in_value_domain_definition`` and ``check_expressions_in_function_definition`` are all
``pass`` stubs. Tests for those live in :class:`TestNotYetParsedExpressions` and are marked ``xfail``
(non-strict), so they document the gap without breaking the suite and will report XPASS once the
corresponding parsing lands.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from concept_hierarchy.data.contexts.context import ConceptHierarchyContext
from concept_hierarchy.data.expressions.expression import Expression, ExpressionValue
from concept_hierarchy.data.expressions.subexpressions import (
    FunctionEvaluation,
    IllFormedExpression,
    InstExpression,
    NarrowExpression,
    PossibleVariableExpression,
    Variable,
    VerifiedTemplateDependentExpression,
)
from concept_hierarchy.data.types.concept_hierarchy_types import TypeValue
from concept_hierarchy.errors import CHSemanticError
from concept_hierarchy.models import ConceptHierarchyModel
from concept_hierarchy.validator.checker import ConceptHierarchyChecker

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


# --------------------------------------------------------------------------------------------------
# Scaffold
# --------------------------------------------------------------------------------------------------

CH_PRELUDE: dict[str, dict] = {
    "Concept": {},
    "ValueDomain": {"directParents": ["Concept"], "data": {}, "abstract": True},
    "Numeric": {"directParents": ["ValueDomain"], "data": {}},
    "Number": {
        "directParents": ["Numeric"],
        "data": {"defaultSerialization": "number", "instantiation": [[[], "number"]]},
    },
    "Integer": {"directParents": ["Number"], "data": {"defaultSerialization": "integer", "instantiation": "integer"}},
    "String": {
        "directParents": ["ValueDomain"],
        "data": {"defaultSerialization": "string", "instantiation": {"type": "string", "pattern": "^s:"}},
    },
    "Function": {"directParents": ["ValueDomain"], "data": {}, "abstract": True},
    "FunctionReturning": {"directParents": ["Function"], "data": {"templateContext": ["T"]}, "abstract": True},
}
"""
The built-in concepts that essentially every Concept Hierarchy needs. Tests spread ``**CH_PRELUDE`` into
their own ``concepts`` dict and add only what the test is actually about.
"""


def build_hierarchy(concepts: dict[str, dict], instances: dict | None = None, name: str = "TestHierarchy") -> dict:
    """Assemble a Concept Hierarchy definition from `CH_PRELUDE` plus the given concepts."""
    model_data: dict = {"name": name, "concepts": {**CH_PRELUDE, **concepts}}
    if instances is not None:
        model_data["instances"] = instances
    return model_data


def check_hierarchy(model_data: dict, external_data: object = None) -> ConceptHierarchyContext:
    """
    Run the full checker over a Concept Hierarchy definition and return the checked context.

    The returned context exposes the parsed model at ``.model`` (``.functions``, ``.value_domains``,
    ``.instances``, ...) and the raw definitions at ``.ch.concepts``.

    :raises ConceptHierarchyError: if the hierarchy does not check, as for any invalid definition.
    """
    model = ConceptHierarchyModel.create_from_data(model_data)
    checker = ConceptHierarchyChecker(model, lambda _concept, _instance: external_data)
    checker.check()
    return checker.context


def check_concepts(concepts: dict[str, dict], **kwargs) -> ConceptHierarchyContext:
    """Shorthand for ``check_hierarchy(build_hierarchy(concepts), ...)``."""
    return check_hierarchy(build_hierarchy(concepts, instances=kwargs.pop("instances", None)), **kwargs)


# --------------------------------------------------------------------------------------------------
# Accessors: getting the parsed expressions back out
# --------------------------------------------------------------------------------------------------


def function_default_expressions(context: ConceptHierarchyContext, function_name: str) -> dict[str, Expression]:
    """
    The parsed default-argument expressions of a Function, keyed by argument name.

    Includes defaults inherited from parent Functions, exactly as the model stores them.
    """
    assert function_name in context.model.functions, (
        f"{function_name!r} is not a Function of this hierarchy; available: {sorted(context.model.functions)}"
    )
    return dict(context.model.functions[function_name].evaluation_argument_default_value)


_SCHEMA_STRUCTURE_KEYWORDS = frozenset({"instantiation", "properties", "items", "type", "default", "oneOf", "anyOf"})


def schema_field_path(location_id) -> str:
    """
    The dotted field path of a schema node, derived from its location by dropping the concept prefix and
    the JSON-Schema structure keywords: ``"Point": "data": "instantiation": "properties": "x": "type"``
    becomes ``"x"``. Nested fields keep their nesting, e.g. ``"outer.inner"``.
    """
    segments = [str(segment) for segment in location_id]
    if "instantiation" in segments:
        segments = segments[segments.index("instantiation") + 1 :]
    return ".".join(
        segment for segment in segments if segment not in _SCHEMA_STRUCTURE_KEYWORDS and not segment.isdigit()
    )


def instantiation_default_expressions(context: ConceptHierarchyContext, value_domain_name: str) -> dict[str, object]:
    """
    The parsed instantiation-schema default expressions of a ValueDomain, keyed by field path.

    A ValueDomain may declare several instantiation schemas (one per template-argument constraint); the
    defaults of all of them are returned together. Values are :class:`Expression` objects once the default
    has been parsed, and the raw JSON value if it has not.
    """
    assert value_domain_name in context.model.value_domains, (
        f"{value_domain_name!r} is not a ValueDomain of this hierarchy; "
        f"available: {sorted(context.model.value_domains)}"
    )
    defaults: dict[str, object] = {}
    for _instantiation_constraint, instantiation_schema in context.model.value_domains[value_domain_name].instantiation:
        for schema_node in instantiation_schema.walk():
            if schema_node.has_default:
                defaults[schema_field_path(schema_node.location_id)] = schema_node.default_expr
    return defaults


def global_variable_types(context: ConceptHierarchyContext) -> dict[str, TypeValue]:
    """The inferred types of the hierarchy's global variables (their *values* are not parsed yet)."""
    return {name: datum.value_type for name, datum in context.model.instances.items()}


# --------------------------------------------------------------------------------------------------
# Assertion helpers
# --------------------------------------------------------------------------------------------------


def describe(expression: Expression | ExpressionValue) -> str:
    """A readable one-liner for an expression, used in assertion messages."""
    value = expression.value if isinstance(expression, Expression) else expression
    description = f"{type(value).__name__}(type={value.value_type})"
    if isinstance(value, IllFormedExpression):
        description += f" -- {value.reason}"
    if isinstance(expression, Expression):
        description += f" from {expression.unparsed!r}"
    return description


def assert_expression(
    expression: Expression,
    *,
    kind: type[ExpressionValue] | None = None,
    value_type: str | None = None,
    is_valid: bool | None = None,
    is_fully_parsed: bool | None = None,
    is_template_dependent: bool | None = None,
) -> ExpressionValue:
    """
    Assert the shape of a parsed expression and return its :class:`ExpressionValue`.

    Only the given properties are checked, so a test states just what it cares about. ``value_type`` is
    compared against the type's ``full_name`` so tests can be written with plain strings.
    """
    assert isinstance(expression, Expression), f"expected a parsed Expression, got {expression!r}"
    value = expression.value
    if kind is not None:
        assert isinstance(value, kind), f"expected {kind.__name__}, got {describe(expression)}"
    if value_type is not None:
        actual = None if value.value_type is None else value.value_type.full_name
        assert actual == value_type, f"expected type {value_type!r}, got {actual!r} -- {describe(expression)}"
    if is_valid is not None:
        assert expression.is_valid is is_valid, f"expected is_valid={is_valid} -- {describe(expression)}"
    if is_fully_parsed is not None:
        assert expression.is_fully_parsed is is_fully_parsed, (
            f"expected is_fully_parsed={is_fully_parsed} -- {describe(expression)}"
        )
    if is_template_dependent is not None:
        assert expression.is_value_template_dependent is is_template_dependent, (
            f"expected is_value_template_dependent={is_template_dependent} -- {describe(expression)}"
        )
    return value


def subexpression_kinds(expression: Expression) -> list[type]:
    """The types of every sub-expression of `expression`, itself included, in traversal order."""
    return [type(sub.value) for sub in expression.all_subexpressions()]


# --------------------------------------------------------------------------------------------------
# Hierarchies under test
# --------------------------------------------------------------------------------------------------

ADD_FUNCTION = {
    "Add": {
        "directParents": ["FunctionReturning"],
        "data": {
            "templateContext": {"order": ["T"], "T": "Numeric", "substitution": {"FunctionReturning:T": "T"}},
            "interface": {
                "arg1": ["T"],
                "arg2": ["T"],
                "res": "T",
                # a literal default, and a default that refers to another argument
                "_defaultArgumentValues": {"arg1": "arg2", "arg2": 0},
            },
        },
    },
}

POINT_VALUE_DOMAIN = {
    "Point": {
        "directParents": ["ValueDomain"],
        "data": {
            "instantiation": {
                "type": "object",
                "properties": {
                    "x": {"type": "Integer", "default": 0},
                    "y": {"type": "Integer", "default": 1},
                    "label": {"type": "String", "default": "s:origin"},
                },
            }
        },
    },
}


# --------------------------------------------------------------------------------------------------
# Tests: the scaffold itself
# --------------------------------------------------------------------------------------------------


class TestScaffold:
    def test_prelude_alone_checks(self):
        context = check_concepts({})
        assert "Integer" in context.model.value_domains

    def test_context_exposes_model_and_definitions(self):
        context = check_concepts(ADD_FUNCTION)
        assert "Add" in context.model.functions
        assert "Add" in context.ch.concepts

    def test_unknown_function_accessor_gives_a_helpful_error(self):
        context = check_concepts({})
        with pytest.raises(AssertionError, match="is not a Function of this hierarchy"):
            function_default_expressions(context, "NoSuchFunction")

    def test_unknown_value_domain_accessor_gives_a_helpful_error(self):
        context = check_concepts({})
        with pytest.raises(AssertionError, match="is not a ValueDomain of this hierarchy"):
            instantiation_default_expressions(context, "NoSuchValueDomain")

    def test_invalid_hierarchy_still_raises(self):
        """The scaffold must not swallow definition errors."""
        with pytest.raises(CHSemanticError, match="is not defined in the hierarchy"):
            check_concepts({"Broken": {"directParents": ["DoesNotExist"], "data": {}}})


# --------------------------------------------------------------------------------------------------
# Tests: Function default argument expressions  (parsed today)
# --------------------------------------------------------------------------------------------------


class TestFunctionDefaultArgumentExpressions:
    def test_defaults_are_parsed_into_expressions(self):
        context = check_concepts(ADD_FUNCTION)
        defaults = function_default_expressions(context, "Add")
        assert set(defaults) == {"arg1", "arg2"}
        assert all(isinstance(expression, Expression) for expression in defaults.values())

    def test_literal_default_for_a_template_typed_argument_stays_deferred(self):
        """
        ``Add.arg2`` is declared as ``T``, so the literal can not be resolved to a single type yet: the
        parser keeps every possible reading and defers the choice to instantiation time.
        """
        context = check_concepts(ADD_FUNCTION)
        defaults = function_default_expressions(context, "Add")
        value = assert_expression(
            defaults["arg2"],
            kind=VerifiedTemplateDependentExpression,
            is_valid=True,
            is_fully_parsed=False,
            is_template_dependent=True,
        )
        assert {str(possible.value_type) for possible in value.possible_expressions} == {"None", "Integer"}

    def test_default_referring_to_another_argument_is_a_variable(self):
        """``Add.arg1`` defaults to ``arg2``; its declared type ``T`` keeps it a *possible* variable."""
        context = check_concepts(ADD_FUNCTION)
        defaults = function_default_expressions(context, "Add")
        value = assert_expression(
            defaults["arg1"],
            kind=PossibleVariableExpression,
            value_type="Add:T",
            is_valid=True,
            is_template_dependent=True,
        )
        assert value.variable_name == "arg2"

    def test_literal_default_for_an_instantiated_argument_is_an_inst_expression(self):
        """With a concrete declared type there is nothing to defer, so the literal resolves immediately."""
        context = check_concepts(
            {
                "Countdown": {
                    "directParents": ["FunctionReturning"],
                    "data": {
                        "templateContext": {
                            "order": ["T"],
                            "T": "Numeric",
                            "substitution": {"FunctionReturning:T": "T"},
                        },
                        "interface": {"from": "Integer", "res": "T", "_defaultArgumentValues": {"from": 10}},
                    },
                },
            }
        )
        defaults = function_default_expressions(context, "Countdown")
        assert_expression(
            defaults["from"],
            kind=InstExpression,
            value_type="Integer",
            is_valid=True,
            is_fully_parsed=True,
        )

    def test_default_expressions_keep_their_unparsed_json(self):
        context = check_concepts(ADD_FUNCTION)
        defaults = function_default_expressions(context, "Add")
        assert defaults["arg1"].unparsed == "arg2"
        assert defaults["arg2"].unparsed == 0

    def test_argument_dependencies_are_recorded(self):
        """``arg1``'s default reads ``arg2``, so the dependency graph must record that."""
        context = check_concepts(ADD_FUNCTION)
        dependencies = context.model.functions["Add"].default_argument_dependencies
        assert dependencies["arg1"] == frozenset({"arg2"})
        assert dependencies["arg2"] == frozenset()

    def test_defaults_are_inherited_from_a_parent_function(self):
        context = check_concepts(
            {
                **ADD_FUNCTION,
                "Increment": {
                    "directParents": ["Add"],
                    "data": {
                        "templateContext": {"order": ["T"], "substitution": {"T": "T"}},
                        "interface": {"_defaultArgumentValues": {"arg2": 1}},
                    },
                },
            }
        )
        defaults = function_default_expressions(context, "Increment")
        # `arg2` is overridden by Increment, `arg1` is inherited from Add
        assert defaults["arg2"].unparsed == 1
        assert defaults["arg1"].unparsed == "arg2"

    def test_default_of_the_wrong_type_is_rejected(self):
        with pytest.raises(CHSemanticError, match="Invalid expression: expected Integer"):
            check_concepts(
                {
                    "Stringify": {
                        "directParents": ["FunctionReturning"],
                        "data": {
                            "templateContext": {
                                "order": ["T"],
                                "T": "Numeric",
                                "substitution": {"FunctionReturning:T": "T"},
                            },
                            "interface": {"arg1": "Integer", "res": "T", "_defaultArgumentValues": {"arg1": "s:nope"}},
                        },
                    },
                }
            )


# --------------------------------------------------------------------------------------------------
# Tests: ValueDomain instantiation default expressions  (parsed today)
# --------------------------------------------------------------------------------------------------


class TestInstantiationDefaultExpressions:
    def test_defaults_are_parsed_into_expressions(self):
        context = check_concepts(POINT_VALUE_DOMAIN)
        defaults = instantiation_default_expressions(context, "Point")
        assert len(defaults) == 3, f"expected one default per property, got {list(defaults)}"
        assert all(isinstance(expression, Expression) for expression in defaults.values())

    def test_defaults_are_keyed_by_field(self):
        context = check_concepts(POINT_VALUE_DOMAIN)
        assert set(instantiation_default_expressions(context, "Point")) == {"x", "y", "label"}

    def test_defaults_are_typed_by_their_schema_node(self):
        context = check_concepts(POINT_VALUE_DOMAIN)
        defaults = instantiation_default_expressions(context, "Point")
        assert_expression(defaults["x"], kind=InstExpression, value_type="Integer", is_valid=True)
        assert_expression(defaults["label"], kind=InstExpression, value_type="String", is_valid=True)

    def test_defaults_keep_their_unparsed_json(self):
        context = check_concepts(POINT_VALUE_DOMAIN)
        defaults = instantiation_default_expressions(context, "Point")
        assert defaults["x"].unparsed == 0
        assert defaults["label"].unparsed == "s:origin"

    def test_default_of_the_wrong_type_is_rejected(self):
        with pytest.raises(CHSemanticError, match="Invalid expression: expected Integer"):
            check_concepts(
                {
                    "BadPoint": {
                        "directParents": ["ValueDomain"],
                        "data": {
                            "instantiation": {
                                "type": "object",
                                "properties": {"x": {"type": "Integer", "default": "s:not-an-integer"}},
                            }
                        },
                    },
                }
            )


# --------------------------------------------------------------------------------------------------
# Tests: the shipped examples  (regression guard over realistic definitions)
# --------------------------------------------------------------------------------------------------


class TestShippedExamples:
    def _animal_kingdom(self) -> ConceptHierarchyContext:
        model_data = json.loads((EXAMPLES_DIR / "animal_kingdom.json").read_text())
        external_data = json.loads((EXAMPLES_DIR / "external_animal_data.json").read_text())
        return check_hierarchy(model_data, external_data)

    def test_animal_kingdom_checks(self):
        context = self._animal_kingdom()
        assert "Add" in context.model.functions

    def test_nested_function_evaluation_default_is_parsed(self):
        """``IncrementByThreeTwice.howManyTimes`` defaults to a nested Function evaluation."""
        defaults = function_default_expressions(self._animal_kingdom(), "IncrementByThreeTwice")
        expression = defaults["howManyTimes"]
        assert_expression(expression, kind=FunctionEvaluation, is_valid=True)
        kinds = subexpression_kinds(expression)
        assert FunctionEvaluation in kinds
        assert Variable in kinds, f"expected the nested variable references to be parsed; got {kinds}"

    def test_narrow_expression_is_parsed(self):
        """``{"Integer": 1}`` inside that default narrows the literal to a specific type."""
        defaults = function_default_expressions(self._animal_kingdom(), "IncrementByThreeTwice")
        kinds = subexpression_kinds(defaults["howManyTimes"])
        assert NarrowExpression in kinds, f"expected a NarrowExpression among {kinds}"

    def test_literal_default_is_parsed(self):
        """``arg2`` is declared as ``T``, so the literal ``3`` stays a deferred, template-dependent value."""
        defaults = function_default_expressions(self._animal_kingdom(), "IncrementByThreeTwice")
        assert_expression(
            defaults["arg2"], kind=VerifiedTemplateDependentExpression, is_valid=True, is_template_dependent=True
        )
        assert defaults["arg2"].unparsed == 3

    def test_empty_list_instantiation_default_is_parsed(self):
        defaults = instantiation_default_expressions(self._animal_kingdom(), "InstanceBase")
        assert defaults, "expected InstanceBase to declare an instantiation default"
        for expression in defaults.values():
            assert_expression(expression, is_valid=True)

    def test_global_variable_types_are_inferred(self):
        types = global_variable_types(self._animal_kingdom())
        assert types, "expected the example to declare global variables"
        assert all(value_type is not None for value_type in types.values())


# --------------------------------------------------------------------------------------------------
# Tests: expressions that are not parsed yet
# --------------------------------------------------------------------------------------------------


class TestNotYetParsedExpressions:
    """
    Placeholders for the expression kinds whose checks are still ``pass`` stubs in
    ``concept_hierarchy.validator.expression_checks``. They are non-strict xfails: once the corresponding
    parsing lands, they report XPASS rather than failing the suite, and can then be fleshed out.
    """

    @pytest.mark.xfail(reason="check_expressions_in_domain_concept_definition is a stub", strict=False)
    def test_property_default_value_is_parsed(self):
        context = check_concepts(
            {
                "Animal": {
                    "directParents": ["Concept"],
                    "data": {"properties": {"legs": {"valueDomain": "Integer", "default": 4}}},
                },
            }
        )
        legs = context.model.domain_concepts["Animal"].properties["legs"]
        assert isinstance(legs.default, Expression), f"property default is still unparsed: {legs.default!r}"

    @pytest.mark.xfail(reason="check_expressions_in_domain_concept_definition is a stub", strict=False)
    def test_property_constraint_is_parsed(self):
        context = check_concepts(
            {
                "Animal": {
                    "directParents": ["Concept"],
                    "data": {
                        "properties": {"legs": {"valueDomain": "Integer", "constraint": {"Interval<Integer>": [0, 8]}}}
                    },
                },
            }
        )
        legs = context.model.domain_concepts["Animal"].properties["legs"]
        assert isinstance(legs.constraint, Expression), f"property constraint is still unparsed: {legs.constraint!r}"

    @pytest.mark.xfail(reason="check_expressions_in_function_definition is a stub", strict=False)
    def test_function_procedure_is_parsed(self):
        context = check_concepts(
            {
                **ADD_FUNCTION,
                "AddOne": {
                    "directParents": ["Add"],
                    "data": {
                        "templateContext": {"order": ["T"], "substitution": {"T": "T"}},
                        "procedure": {"Add<Integer>": {"arg1": "arg1", "arg2": 1}},
                    },
                },
            }
        )
        procedure = context.model.functions["AddOne"].procedure
        assert isinstance(procedure, Expression), f"Function procedure is still unparsed: {procedure!r}"

    @pytest.mark.xfail(reason="only the type of a global variable is computed, not its expression", strict=False)
    def test_global_variable_value_is_parsed(self):
        context = check_concepts({}, instances={"one": {"value": 1, "valueDomain": "Integer"}})
        assert isinstance(context.model.instances["one"].value, Expression)


class TestTemplateDependenceOfParsedValues:
    """
    ``InstExpression.is_template_dependent`` reports the value's own content as well as its
    subexpressions. The contract is unit-tested directly in ``tests/unit/test_expression_values.py``;
    these check it end-to-end, through a real parse.
    """

    def test_a_plain_literal_is_not_template_dependent(self):
        context = check_concepts(
            {
                "Countdown": {
                    "directParents": ["FunctionReturning"],
                    "data": {
                        "templateContext": {
                            "order": ["T"],
                            "T": "Numeric",
                            "substitution": {"FunctionReturning:T": "T"},
                        },
                        "interface": {"from": "Integer", "res": "T", "_defaultArgumentValues": {"from": 10}},
                    },
                },
            }
        )
        expression = function_default_expressions(context, "Countdown")["from"]
        assert expression.is_value_template_dependent is False, (
            f"a literal 10 of declared type Integer depends on no template: {describe(expression)}"
        )

    def test_an_instantiation_default_literal_is_not_template_dependent(self):
        context = check_concepts(POINT_VALUE_DOMAIN)
        for field, expression in instantiation_default_expressions(context, "Point").items():
            assert expression.is_value_template_dependent is False, (
                f"Point.{field} is a literal and depends on no template: {describe(expression)}"
            )

    def test_a_template_typed_default_is_template_dependent(self):
        """``Add.arg2`` is declared as ``T``, so its value can not be resolved without knowing ``T``."""
        context = check_concepts(ADD_FUNCTION)
        expression = function_default_expressions(context, "Add")["arg2"]
        assert expression.is_value_template_dependent is True, describe(expression)
