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

- **Function default argument values** -- stored in ``FunctionData.evaluation_argument_default_value_expressions``;
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
import re
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
from concept_hierarchy.definitions.concept_hierarchy import ConceptHierarchyDefinition
from concept_hierarchy.errors import CHSemanticError, ConceptHierarchyError
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
    "TypeValue": {
        "directParents": ["String"],
        "data": {"instantiation": {"type": "string", "pattern": "^s:", "format": "Type"}},
    },
    "Function": {"directParents": ["ValueDomain"], "data": {}, "abstract": True},
    "FunctionReturning": {"directParents": ["Function"], "data": {"templateContext": ["T"]}, "abstract": True},
    "FunctionComposition": {
        "directParents": ["ValueDomain"],
        "data": {
            "instantiation": {
                "type": "object",
                "minProperties": 1,
                "maxProperties": 1,
                "propertyNames": {"type": "string", "format": "Type", "constraint": "Function"},
                "additionalProperties": {"type": "object", "properties": "args", "additionalProperties": False},
            },
        },
    },
    "CustomFunction": {
        "directParents": ["ValueDomain"],
        "data": {
            "instantiation": {
                "oneOf": [
                    "FunctionComposition",
                    {
                        "type": "object",
                        "properties": {
                            "interface": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "_defaultArgumentValues": {
                                        "type": "object",
                                        "patternProperties": {"^[a-z][A-Za-z0-9_]*$": True},
                                    }
                                },
                                "patternProperties": {
                                    "^[a-z][A-Za-z0-9_]*$": {"$ref": "#/$defs/argumentTypeDefinition"}
                                },
                            },
                            "procedure": "FunctionComposition",
                        },
                        "additionalProperties": False,
                        "required": ["procedure"],
                    },
                ],
                "$defs": {
                    "argumentTypeDefinition": {
                        "oneOf": [
                            "TypeValue",
                            {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 3,
                                "items": ["TypeValue", {"$ref": "#/$defs/argAcc"}, {"$ref": "#/$defs/argProv"}],
                            },
                        ]
                    },
                    "argProv": {"type": "string", "enum": ["Any", "Addr", "ResetAddr"]},
                    "argAcc": {"type": "string", "enum": ["Get", "Modify", "GetModify"]},
                },
            }
        },
    },
    "Instance": {
        "directParents": ["ValueDomain"],
        "data": {
            "templateContext": {
                "order": ["AcceptConcepts...", "RejectConcepts..."],
                "AcceptConcepts": "And(Concept, Not(ValueDomain))",
                "RejectConcepts": "And(Concept, Not(ValueDomain))",
                "variadicGroupIdentifiers": {"AcceptConcepts": "", "RejectConcepts": "!"},
            }
        },
    },
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
    model = ConceptHierarchyDefinition.create_from_data(model_data)
    checker = ConceptHierarchyChecker(model, lambda _concept, _instance: external_data)
    checker.check()
    return checker.context


def check_concepts(concepts: dict[str, dict], **kwargs) -> ConceptHierarchyContext:
    """Shorthand for ``check_hierarchy(build_hierarchy(concepts), ...)``."""
    return check_hierarchy(build_hierarchy(concepts, instances=kwargs.pop("instances", None)), **kwargs)


def rejection(concepts: dict[str, dict]) -> ConceptHierarchyError:
    with pytest.raises(ConceptHierarchyError) as exec_info:
        check_concepts(concepts)
    return exec_info.value


def refuses_commitment(messages: str, reading: str) -> bool:
    """
    Whether ``messages`` refuses a value for not being ``reading``, in either of the two spellings.

    Two different checks produce such a refusal, and they are worded apart on purpose: the classification
    filter says the value **must be** that reading and this position cannot hold one, while the
    consumption check says the value **is committed to being** it and the schema matched without ever
    reading it as one. A test that only cares that the commitment was refused should not have to know
    which of the two fired, nor break when the wording is polished again.
    """
    return re.search(r"(?:must be|is committed to being) a " + re.escape(reading), messages) is not None


def error_messages(error: ConceptHierarchyError) -> str:
    collected: list[str] = []

    def walk(err: ConceptHierarchyError) -> None:
        collected.append(err.args[0] if err.args else "")
        for cause in err.causes:
            walk(cause)

    walk(error)
    return " | ".join(collected)


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
    return dict(context.model.functions[function_name].evaluation_argument_default_value_expressions)


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


def instantiation_default_expressions(
    context: ConceptHierarchyContext, value_domain_name: str
) -> dict[str, Expression | None]:
    """
    The parsed instantiation-schema default expressions of a ValueDomain, keyed by field path.

    A ValueDomain may declare several instantiation schemas (one per template-argument constraint); the
    defaults of all of them are returned together. Values are :class:`Expression` objects once the default
    has been parsed, and `None` if it has not been parsed.
    """
    assert value_domain_name in context.model.value_domains, (
        f"{value_domain_name!r} is not a ValueDomain of this hierarchy; "
        f"available: {sorted(context.model.value_domains)}"
    )
    defaults: dict[str, Expression | None] = {}
    for _instantiation_constraint, instantiation_schema in context.model.value_domains[value_domain_name].instantiation:
        for schema_node in instantiation_schema.walk():
            if schema_node.has_default:
                defaults[schema_field_path(schema_node.location_id)] = schema_node.parsed_default_expr
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
    def _animal_kingdom(
        self, concepts: dict[str, dict | str] | None = None, instances: dict[str, dict | str] | None = None
    ) -> ConceptHierarchyContext:
        model_data = json.loads((EXAMPLES_DIR / "animal_kingdom.json").read_text())
        if "instances" not in model_data and "concepts" not in model_data:
            model_data = {"concepts": model_data}
        assert "concepts" in model_data
        if "instances" not in model_data:
            model_data["instances"] = {}
        if concepts is not None:
            model_data["concepts"].update(concepts)
        if instances is not None:
            model_data["instances"].update(instances)
        external_data = json.loads((EXAMPLES_DIR / "external_animal_data.json").read_text())
        return check_hierarchy(model_data, external_data)

    test_concepts = {
        "TestNestedDefaultValueWithFunctionEvaluationExpression": {
            "directParents": ["Function"],
            "data": {
                "templateContext": ["T"],
                "interface": {
                    "arg1": "T",
                    "arg": "Integer",
                    "_defaultArgumentValues": {
                        "arg": {
                            "Add<Integer>": {
                                "arg1": {"fEval:Add<Integer>": {"arg1": {"Integer": 1}, "arg2": 1}},
                                "arg2": {
                                    "Add<Integer>": {
                                        "arg1": "oneRef",
                                        "arg2": {"Add<Integer>": {"arg1": "arg1", "arg2": 1}},
                                    }
                                },
                            }
                        },
                        "arg1": 3.0,
                    },
                },
            },
        }
    }

    test_instances = {"one": {"Integer": 1}, "oneRef": "one"}

    def test_animal_kingdom_checks(self):
        context = self._animal_kingdom()
        assert "Add" in context.model.functions

    def test_nested_function_evaluation_default_is_parsed(self):
        """``TestNestedDefaultValueWithFunctionEvaluationExpression.arg`` defaults to a nested Function evaluation."""
        defaults = function_default_expressions(
            self._animal_kingdom(
                concepts=TestShippedExamples.test_concepts, instances=TestShippedExamples.test_instances
            ),
            "TestNestedDefaultValueWithFunctionEvaluationExpression",
        )
        expression = defaults["arg"]
        assert_expression(expression, kind=FunctionEvaluation, is_valid=True)
        kinds = subexpression_kinds(expression)
        assert FunctionEvaluation in kinds
        assert Variable in kinds, f"expected the nested variable references to be parsed; got {kinds}"

    def test_narrow_expression_is_parsed(self):
        """``{"Integer": 1}`` inside that default narrows the literal to a specific type."""
        defaults = function_default_expressions(
            self._animal_kingdom(
                concepts=TestShippedExamples.test_concepts, instances=TestShippedExamples.test_instances
            ),
            "TestNestedDefaultValueWithFunctionEvaluationExpression",
        )
        kinds = subexpression_kinds(defaults["arg"])
        assert NarrowExpression in kinds, f"expected a NarrowExpression among {kinds}"

    def test_literal_default_is_parsed(self):
        """``arg2`` is declared as ``T``, so the literal ``3`` stays a deferred, template-dependent value."""
        defaults = function_default_expressions(
            self._animal_kingdom(
                concepts=TestShippedExamples.test_concepts, instances=TestShippedExamples.test_instances
            ),
            "TestNestedDefaultValueWithFunctionEvaluationExpression",
        )
        assert_expression(
            defaults["arg1"], kind=VerifiedTemplateDependentExpression, is_valid=True, is_template_dependent=True
        )
        assert defaults["arg1"].unparsed == 3.0

    def test_empty_list_instantiation_default_is_parsed(self):
        defaults = instantiation_default_expressions(self._animal_kingdom(), "InstanceBase")
        assert defaults, "expected InstanceBase to declare an instantiation default"
        for expression in defaults.values():
            assert_expression(expression, is_valid=True)

    def test_global_variable_types_are_inferred(self):
        types = global_variable_types(self._animal_kingdom())
        assert types, "expected the example to declare global variables"
        assert all(value_type is not None for value_type in types.values())

    # ----------------------------------------------------------------------------------------------
    # ``Dog.f2``: a Function-keyed object at a `CustomFunction` site
    #
    # The site the `fComp:` marker exists for, and the one place in the shipped data where all four
    # readings can be told apart. ``data.functions.<name>`` expects a `CustomFunction`, whose
    # instantiation is ``oneOf: ["FunctionComposition", {interface, procedure}]``; ``Add<Number>``
    # returns a `Number`, which is neither a `CustomFunction` nor a `FunctionComposition`, so only the
    # composition reading can stand. Each of the others fails for its own stated reason, which is what
    # makes this evidence about the markers rather than one accident that happens to fail.
    # ----------------------------------------------------------------------------------------------

    def _dog_f2_keyed(self, key: str) -> dict:
        """The shipped example with ``Dog.f2``'s single content key replaced by ``key``."""
        model_data = json.loads((EXAMPLES_DIR / "animal_kingdom.json").read_text())
        f2 = model_data["concepts"]["Dog"]["data"]["functions"]["f2"]
        (only_key,) = list(f2)
        f2[key] = f2.pop(only_key)
        return model_data

    def _check_dog_f2(self, key: str) -> ConceptHierarchyContext:
        external_data = json.loads((EXAMPLES_DIR / "external_animal_data.json").read_text())
        return check_hierarchy(self._dog_f2_keyed(key), external_data)

    def _dog_f2_rejection(self, key: str) -> str:
        with pytest.raises(ConceptHierarchyError) as exec_info:
            self._check_dog_f2(key)
        return error_messages(exec_info.value)

    def test_dog_f2_is_written_with_the_composition_marker(self):
        """The shipped spelling, so that the tests below are mutations of what is really in the file."""
        model_data = json.loads((EXAMPLES_DIR / "animal_kingdom.json").read_text())
        assert list(model_data["concepts"]["Dog"]["data"]["functions"]["f2"]) == ["fComp:Add<Number>"]

    def test_the_composition_marker_is_accepted_there(self):
        assert "Add" in self._check_dog_f2("fComp:Add<Number>").model.functions

    def test_the_instantiation_marker_is_refused_there(self):
        """
        The migration check for `fInst:`: ``Add<Number>`` is not a subtype of `CustomFunction`, so asking
        for the Function *value* here cannot be honored. If this ever passes, the marker is being ignored.
        """
        messages = self._dog_f2_rejection("fInst:Add<Number>")
        assert refuses_commitment(messages, "Function instantiation"), messages

    def test_the_evaluation_marker_is_refused_there(self):
        messages = self._dog_f2_rejection("fEval:Add<Number>")
        assert refuses_commitment(messages, "Function evaluation"), messages

    def test_the_refusal_names_the_type_expected_at_the_position(self):
        """
        Not the type written at the site. ``Dog.f2`` is a `CustomFunction`, but the reading is refused
        inside the `FunctionComposition` branch of its instantiation, and that branch is what was checked
        -- so the message says `FunctionComposition`, and says it is "at this position".
        """
        messages = self._dog_f2_rejection("fInst:Add<Number>")
        assert "FunctionComposition is expected at this position" in messages, messages

    def test_the_refusal_states_the_condition_that_failed(self):
        """ "Does not accept one" said only that something was refused; the rule is what an author acts on."""
        messages = self._dog_f2_rejection("fInst:Add<Number>")
        assert "stands here only if that Function is a subtype of FunctionComposition" in messages, messages

    def test_an_evaluation_at_a_composition_is_refused_as_unmeetable_not_merely_unmet(self):
        """
        The half of the message that answers "why would a FunctionComposition never accept an evaluation":
        Section 9.4 forbids a ``"res"`` that is a `FunctionComposition`, so the condition cannot be met by
        any Function at all -- a different thing from this particular Function failing it.
        """
        messages = self._dog_f2_rejection("fEval:Add<Number>")
        assert "no Function may return a FunctionComposition" in messages, messages

    def test_without_a_marker_the_evaluation_default_hard_fails(self):
        """Why the site needs a marker at all: `FEval` commits, and ``res(Add<Number>)`` does not fit."""
        messages = self._dog_f2_rejection("Add<Number>")
        assert "not a subtype of CustomFunction" in messages, messages

    def test_an_unknown_prefix_is_not_a_marker_there(self):
        """``fOther:`` is not in the vocabulary, so the whole string stays the key -- and names nothing."""
        messages = self._dog_f2_rejection("fOther:Add<Number>")
        assert "is not a concept or a template variable" in messages, messages

    def test_a_marked_key_naming_nothing_is_refused(self):
        messages = self._dog_f2_rejection("fInst:NoSuchFunction")
        assert messages

    def test_a_domain_concept_named_as_the_key_is_diagnosed_not_crashed(self):
        error_msg = self._dog_f2_rejection("Dog")
        assert "Dog is not a ValueDomain Type. It seems to be a DomainConcept." in error_msg, error_msg


# --------------------------------------------------------------------------------------------------
# Tests: expressions that are not parsed yet
# --------------------------------------------------------------------------------------------------


class TestNotYetParsedExpressions:
    """
    Placeholders for the expression kinds whose checks are still ``pass`` stubs in
    ``concept_hierarchy.validator.expression_checks``. They are non-strict xfails: once the corresponding
    parsing lands, they report XPASS rather than failing the suite, and can then be fleshed out.
    """

    def test_property_default_value_is_parsed(self):
        context = check_concepts(
            {
                "Animal": {
                    "directParents": ["Concept"],
                    "data": {"properties": {"legs": {"valueDomain": "Integer", "default": 4}}},
                },
            }
        )
        legs_default = context.model.domain_concepts["Animal"].property_expressions["legs"]["forSub"]["default"]
        assert isinstance(legs_default, Expression), f"property default is still unparsed: {legs_default!r}"

    def test_property_constraint_is_parsed(self):
        context = check_concepts(
            {
                "Animal": {
                    "directParents": ["Concept"],
                    "data": {
                        "properties": {"legs": {"valueDomain": "Integer", "constraint": {"Interval<Integer>": [0, 8]}}}
                    },
                },
                "Variation": {"directParents": ["ValueDomain"], "data": {"templateContext": ["T"]}},
                "Interval": {
                    "directParents": ["ValueDomain"],
                    "data": {
                        "templateContext": {"order": ["T"], "T": "Numeric"},
                        "instantiation": {"type": "array", "items": "T", "minItems": 2, "maxItems": 2},
                    },
                },
            }
        )
        legs_constraint = context.model.domain_concepts["Animal"].property_expressions["legs"]["forSub"]["constraint"]
        assert isinstance(legs_constraint, Expression), f"property constraint is still unparsed: {legs_constraint!r}"

    @pytest.mark.xfail(reason="check_expressions_in_function_definition is a stub", strict=False)
    def test_function_procedure_is_parsed(self):
        context = check_concepts(
            {
                **ADD_FUNCTION,
                "AddOne": {
                    "directParents": ["Add"],
                    "data": {
                        "templateContext": {"order": ["T"], "substitution": {"T": "T"}},
                        "interface": {},
                        "procedure": {"Add<Integer>": {"arg1": "arg1", "arg2": 1}},
                    },
                },
            }
        )
        procedure = context.model.functions["AddOne"].procedure
        assert isinstance(procedure, Expression), f"Function procedure is still unparsed: {procedure!r}"

    def test_a_domain_concept_function_body_is_parsed(self):
        error = rejection(
            {
                "Animal": {
                    "directParents": ["Concept"],
                    "data": {"functions": {"f": {"NoSuchFunction": {"nonsense": True}}}},
                },
            }
        )
        error_msg = error_messages(error)
        assert '"NoSuchFunction" is not a concept or a template variable of this' in error_msg, error_msg

    def test_global_variable_value_is_parsed(self):
        context = check_concepts({}, instances={"one": {"Integer": 1}})
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
