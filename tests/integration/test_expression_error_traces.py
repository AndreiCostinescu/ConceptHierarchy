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
Integration tests: what an expression-parsing *failure* tells you.

Parsing an expression is a search over a fixed set of alternatives -- ``FEval``, ``Narrow``, ``Inst``,
default-serialisation, ``Var``, an instance-property chain, a literal template variable -- and for
``Inst``/``Narrow`` a further search over the type's instantiation constraint groups. When the search comes
up empty, :class:`IllFormedExpression` records one :class:`ExpressionAttempt` per alternative that was
applicable, and the headline reason -- "Could not match a valid <Type> expression to value <value>" --
is only the first line of it. For ``Inst``/``Narrow`` the attempt also carries every constraint group
that was tried and the :class:`~instantiated_value.ParsedValue` errors of the one that matched, which is
what says *which property* was missing, additional, or wrongly typed.

These tests were written against the *old* behaviour, where all of this was lost, and are now the
regression guard on the trace that replaced it.

The asserts deliberately look for content (a property name, a type name, a constraint) rather than exact
phrasing, so that landing the trace does not require rewriting every expectation. ``str(error)`` renders
``ConceptHierarchyError.causes`` recursively and indented, so a trace threaded through ``causes`` is
matched by these assertions without further work.

See ``documentation/TODO_DEFAULT_EXPANSION_CYCLES.md`` §14 for the design these describe.
"""

from __future__ import annotations

import pytest

from concept_hierarchy.errors import ConceptHierarchyError
from tests.integration.test_expression_parsing import check_concepts

# --------------------------------------------------------------------------------------------------
# Hierarchy fragments
# --------------------------------------------------------------------------------------------------

ADD = {
    "Add": {
        "directParents": ["FunctionReturning"],
        "data": {
            "templateContext": {"order": ["T"], "T": "Numeric", "substitution": {"FunctionReturning:T": "T"}},
            "interface": {"arg1": ["T"], "arg2": ["T"], "res": "T"},
        },
    }
}

POINT = {
    "Point": {
        "directParents": ["ValueDomain"],
        "data": {
            "instantiation": {
                "type": "object",
                "properties": {"x": {"type": "Integer"}, "y": {"type": "Integer"}},
                "required": ["x", "y"],
                "additionalProperties": False,
            }
        },
    }
}

PICK = {
    "Leaf": {
        "directParents": ["ValueDomain"],
        "data": {"instantiation": {"type": "object", "additionalProperties": False}},
    },
    "Pick": {
        "directParents": ["ValueDomain"],
        "data": {
            "templateContext": ["Q"],
            "instantiation": [
                [
                    ["Integer"],
                    {
                        "type": "object",
                        "properties": {"i": {"type": "Integer"}},
                        "required": ["i"],
                        "additionalProperties": False,
                    },
                ],
                [
                    [""],
                    {
                        "type": "object",
                        "properties": {"o": {"type": "Leaf"}},
                        "required": ["o"],
                        "additionalProperties": False,
                    },
                ],
            ],
        },
    },
}


def site(expected_type: str, default: object, name: str = "Site", provenance: str | None = None) -> dict:
    """A ValueDomain with one defaulted property, so that `default` is parsed as an expression."""
    prop: dict = {"type": expected_type, "default": default}
    if provenance is not None:
        prop["provenance"] = provenance
    return {
        name: {
            "directParents": ["ValueDomain"],
            "data": {"instantiation": {"type": "object", "properties": {"p": prop}}},
        }
    }


def located_messages(text: str) -> list[tuple[str, str]]:
    """
    Every ``(location, message)`` pair of a rendered trace.

    ``ConceptHierarchyError.__str__`` prints a location on its own line and the message under it, so the
    two are separable and a test can assert *where* something was reported, not only *that* it was. The
    location is what a reader follows to the offending text, and it is the half that goes wrong silently:
    a message reported three levels above where it happened still reads perfectly.
    """
    pairs: list[tuple[str, str]] = []
    location: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            location = line
        elif line and location is not None:
            pairs.append((location, line))
    return pairs


def location_reporting(text: str, needle: str) -> str:
    """The location of the one message containing ``needle``."""
    matches = {location for location, message in located_messages(text) if needle in message}
    assert len(matches) == 1, f"expected exactly one message containing {needle!r}, got {sorted(matches)}"
    return matches.pop()


def path_of(*segments: str) -> str:
    """A rendered location, as `ConceptHierarchyError` prints it."""
    return "[" + ": ".join(f'"{segment}"' for segment in segments) + "]"


def site_path(*segments: str) -> str:
    """The location of `site`'s default expression, extended by ``segments``."""
    return path_of("concepts", "Site", "data", "instantiation", "properties", "p", "type", "default", *segments)


def expression_error(concepts: dict) -> str:
    """Check the hierarchy, require it to fail, and return the fully rendered error text."""
    with pytest.raises(ConceptHierarchyError) as excinfo:
        check_concepts(concepts)
    return str(excinfo.value)


def _echoed_forms(value: object) -> list[str]:
    """Every rendering of ``value`` and its sub-values that could appear quoted in an error message."""
    forms: list[str] = []
    if isinstance(value, (dict, list, str)):
        forms += [str(value), repr(value)]
    if isinstance(value, dict):
        for sub in value.values():
            forms += _echoed_forms(sub)
    elif isinstance(value, list):
        for sub in value:
            forms += _echoed_forms(sub)
    return forms


def explanation(concepts: dict, offending: object) -> str:
    """
    The rendered error with every echo of the offending value blanked out.

    The parser quotes the value it could not parse (``got {...}!``, and again inside the reason), so a
    plain substring assertion is satisfied by the message merely repeating the input -- which says
    nothing about whether anything was *explained*. Blanking the echoes first makes these assertions
    mean what they look like they mean.
    """
    text = expression_error(concepts)
    for form in sorted(set(_echoed_forms(offending)), key=len, reverse=True):
        text = text.replace(form, "<value>")
    return text


# --------------------------------------------------------------------------------------------------
# What already works
# --------------------------------------------------------------------------------------------------


class TestFailuresAreReportedAtTheRightPlace:
    """Whatever the trace ends up saying, it has to say it at the failing default's location."""

    @pytest.mark.parametrize(
        "expected_type,default,extra",
        [
            ("Integer", {"NotAType": 1}, {}),
            ("Integer", "s:not-a-variable", {}),
            ("Point", {"x": 1}, POINT),
            ("Point", "whatever", POINT),
        ],
    )
    def test_location_points_at_the_default(self, expected_type, default, extra):
        text = expression_error({**extra, **site(expected_type, default)})
        assert '"concepts": "Site": "data": "instantiation": "properties": "p": "type": "default"' in text

    def test_the_expected_type_is_named(self):
        assert "Point" in expression_error({**POINT, **site("Point", {"x": 1})})


class TestMessagesThatAreAlreadyGood:
    """These are raised directly by `_parse_expression_of_json_object` and are already specific."""

    def test_unknown_function_argument_names_it_and_the_alternatives(self):
        value = {"Add<Integer>": {"nope": 1}}
        text = explanation({**ADD, **site("Integer", value)}, value)
        assert "nope" in text
        assert "arg1" in text and "arg2" in text

    def test_missing_required_function_argument_names_it(self):
        value = {"Add<Integer>": {"arg1": 1}}
        text = explanation({**ADD, **site("Integer", value)}, value)
        assert "arg2" in text
        assert "missing" in text.lower()

    def test_the_path_through_nested_function_evaluations_survives(self):
        """
        ``FEval`` failures *do* nest today, by concatenating reasons: "Add<Integer> argument arg2's value
        ... is invalid: ...". The *path* therefore survives, and must keep surviving once the
        concatenation is replaced by a structured trace.
        """
        value = {"Add<Integer>": {"arg1": 1, "arg2": {"Add<Integer>": {"arg1": 1, "arg2": {"NotAType": 9}}}}}
        text = explanation({**ADD, **site("Integer", value)}, value)
        assert text.count("arg2") >= 2, "each level of the nesting must name the argument it failed at"
        assert "is invalid" in text


# --------------------------------------------------------------------------------------------------
# What is lost
# --------------------------------------------------------------------------------------------------


class TestTheInnermostCauseIsDescribed:
    """
    The nested `FEval` reason names the *path* (``arg2``, then ``arg2`` again) but never says what was
    actually wrong at the bottom -- the innermost value is visible only because the message quotes it
    back. Strip the quotes and nothing is left but "could not match".
    """

    def test_innermost_failure_is_explained_not_merely_quoted(self):
        value = {"Add<Integer>": {"arg1": 1, "arg2": {"NotAType": 9}}}
        text = explanation({**ADD, **site("Integer", value)}, value)
        assert "NotAType" in text, "the innermost offending key must be named, not only echoed"


class TestSchemaErrorsReachTheExpressionError:
    """
    ``_check_instantiation_schema`` builds a full ``ParsedValue`` whose errors say exactly what was wrong
    with the value, then discards it via a bare ``is_valid()`` test. Those errors are the single most
    useful thing the trace can carry.
    """

    def test_missing_required_property_is_named(self):
        value = {"x": 1}
        text = explanation({**POINT, **site("Point", value)}, value)
        assert "y" in text and "required" in text.lower()

    def test_additional_property_is_named(self):
        value = {"x": 1, "y": 2, "z": 3}
        text = explanation({**POINT, **site("Point", value)}, value)
        assert "z" in text
        assert "additional" in text.lower()

    def test_wrong_property_type_is_located(self):
        value = {"x": 1, "y": "s:nope"}
        text = explanation({**POINT, **site("Point", value)}, value)
        assert "y" in text, "the trace must say which property failed"
        assert "Integer" in text, "...and what it needed to be"


class TestTriedAlternativesAreRecorded:
    """
    A value that matches nothing should say what was attempted and why each attempt failed, rather than
    only that nothing matched.
    """

    def test_object_that_matches_nothing_lists_the_attempts(self):
        value = {"NotAType": 1}
        text = explanation({**site("Integer", value)}, value)
        assert "NotAType" in text, "the key that was tried must be named"
        assert "concept" in text.lower() or "type" in text.lower()

    def test_key_that_is_a_type_but_not_a_subtype_says_so(self):
        value = {"String": "s:x"}
        text = explanation({**site("Integer", value)}, value)
        assert "String" in text and "Integer" in text
        assert "subtype" in text.lower()

    def test_string_that_is_no_kind_of_variable_says_which_kinds_were_tried(self):
        text = explanation({**site("Integer", "s:not-a-variable")}, "s:not-a-variable")
        assert "variable" in text.lower(), "Var / instance-property-chain / literal-template-variable were tried"

    def test_default_serialization_miss_is_explained(self):
        """``"whatever"`` is a string; Point registers no string ``defaultSerialization``."""
        text = explanation({**POINT, **site("Point", "whatever")}, "whatever")
        assert "defaultSerialization" in text or "default serialization" in text.lower()


class TestTriedConstraintGroupsAreRecorded:
    """
    For an `Inst`/`Narrow` of a templated ValueDomain, the trace should name each instantiation constraint
    group that was tried -- both the ones whose constraint the type did not satisfy and the one that
    matched but whose schema then rejected the value.
    """

    def test_both_constraint_groups_are_reported(self):
        value = {"WRONG": 1}
        text = explanation({**PICK, **site("Pick<Leaf>", value)}, value)
        assert "Pick<Leaf>" in text
        assert "Integer" in text, "the constraint of the group that did not match must be named"
        assert "required" in text.lower(), "the group that did match must explain why its schema rejected"


class TestNestedExpressionErrors:
    """
    The deep cases. An error arbitrarily far inside a value must still reach the top, with enough path to
    locate it -- this is where the single-line reason is least adequate.
    """

    def test_error_inside_an_instantiated_property_survives(self):
        """
        ``Inst(Point) -> property y -> FEval(Add) -> argument arg2 -> not a type``. Today the whole inner
        story is replaced by "could not match a valid Point expression".
        """
        value = {"x": 1, "y": {"Add<Integer>": {"arg1": 1, "arg2": {"NotAType": 9}}}}
        text = explanation({**ADD, **POINT, **site("Point", value)}, value)
        assert "y" in text, "the failing property must be identified"
        assert "arg2" in text, "and the failing Function argument inside it"
        assert "NotAType" in text, "and the innermost offending key"

    def test_error_inside_a_narrowed_value_survives(self):
        """``Narrow`` to ``Point`` fails only because ``y`` is missing; today ``Point`` is not even named."""
        value = {"Point": {"x": 1}}
        text = explanation({**POINT, **site("ValueDomain", value)}, value)
        assert "Point" in text, "the narrowed type must be named"
        assert "y" in text, "and the reason its schema rejected the value"

    def test_two_levels_of_instantiation_nesting_survive(self):
        """``Inst(Outer) -> property inner -> Inst(Point) -> missing required y``."""
        outer = {
            "Outer": {
                "directParents": ["ValueDomain"],
                "data": {
                    "instantiation": {
                        "type": "object",
                        "properties": {"inner": {"type": "Point"}},
                        "required": ["inner"],
                        "additionalProperties": False,
                    }
                },
            }
        }
        value = {"inner": {"x": 1}}
        text = explanation({**POINT, **outer, **site("Outer", value)}, value)
        assert "inner" in text and "y" in text


# --------------------------------------------------------------------------------------------------
# Additional hierarchy fragments
# --------------------------------------------------------------------------------------------------

NO_RETURN = {"Shout": {"directParents": ["Function"], "data": {"interface": {"what": ["Integer"]}}}}
"""A Function that returns nothing, so evaluating it can not produce a value."""

ADDR_FUNCTION = {
    "AddrF": {
        "directParents": ["FunctionReturning"],
        "data": {
            "templateContext": {"order": [], "substitution": {"FunctionReturning:T": "Integer"}},
            "interface": {"res": ["Integer", "Addr"]},
        },
    }
}
"""A Function whose result is addressable, i.e. usable where `Addr` provenance is required."""

ABSTRACT = {"Abstr": {"directParents": ["ValueDomain"], "data": {}, "abstract": True}}


def target(instantiation: object) -> dict:
    """A ValueDomain named ``Target`` with the given instantiation schema."""
    return {"Target": {"directParents": ["ValueDomain"], "data": {"instantiation": instantiation}}}


# --------------------------------------------------------------------------------------------------
# Coverage: every JSON value shape
# --------------------------------------------------------------------------------------------------


class TestEveryJsonValueShapeIsExplained:
    """
    Whatever the JSON value is, a failure names the alternatives that were applicable to *that* shape.
    ``Inst`` and default-serialisation apply to every shape; the string-only alternatives apply to strings
    and nothing else.
    """

    @pytest.mark.parametrize(
        "label,value",
        [
            ("null", None),
            ("boolean", True),
            ("float", 1.5),
            ("array", [1, 2]),
            ("empty object", {}),
            ("multi-key object", {"a": 1, "b": 2}),
        ],
    )
    def test_non_string_shapes_report_inst_and_default_serialization(self, label, value):
        text = explanation({**POINT, **site("Point", value)}, value)
        assert "as Inst" in text, f"{label}: the Inst alternative must be reported"
        assert "as default serialization" in text, f"{label}: the DS alternative must be reported"

    @pytest.mark.parametrize("value", [None, True, 1.5, [1, 2], {}, {"a": 1, "b": 2}])
    def test_non_string_shapes_do_not_report_string_alternatives(self, value):
        """`Var`, an instance-property chain and a literal template variable are only reachable from a string."""
        text = explanation({**POINT, **site("Point", value)}, value)
        assert "as variable" not in text
        assert "as instance property chain" not in text
        assert "as literal template variable" not in text

    def test_a_string_reports_every_string_alternative(self):
        text = explanation({**site("Integer", "s:nope")}, "s:nope")
        assert "as variable" in text
        assert "as literal template variable" in text

    def test_a_dotted_string_also_reports_the_instance_property_chain(self):
        text = explanation({**site("Integer", "a.b.c")}, "a.b.c")
        assert "as instance property chain" in text
        assert "as variable" in text

    def test_an_undotted_string_does_not_report_a_property_chain(self):
        text = explanation({**site("Integer", "s:nope")}, "s:nope")
        assert "as instance property chain" not in text


# --------------------------------------------------------------------------------------------------
# Coverage: Function evaluation
# --------------------------------------------------------------------------------------------------


class TestFunctionEvaluationFailures:
    """Every way an `FEval` can be rejected, and what the reader is told."""

    def test_function_returns_nothing(self):
        value = {"Shout": {"what": 1}}
        text = explanation({**NO_RETURN, **site("Integer", value)}, value)
        assert "Shout" in text
        assert "does not return anything" in text
        assert "Integer" in text, "the type that was expected instead must be named"

    def test_function_evaluation_value_is_not_an_object(self):
        text = explanation({**ADD, **site("Integer", {"Add<Integer>": 5})}, {"Add<Integer>": 5})
        assert "as FEval" in text
        assert "json object" in text

    def test_result_type_is_not_a_subtype(self):
        value = {"Add<Integer>": {"arg1": 1, "arg2": 2}}
        text = explanation({**ADD, **site("String", value)}, value)
        assert "Function result type Integer is not a subtype of String" in text

    def test_unknown_argument_names_it_and_the_alternatives(self):
        value = {"Add<Integer>": {"nope": 1}}
        text = explanation({**ADD, **site("Integer", value)}, value)
        assert "nope" in text and "arg1" in text and "arg2" in text

    def test_missing_required_argument(self):
        value = {"Add<Integer>": {"arg1": 1}}
        text = explanation({**ADD, **site("Integer", value)}, value)
        assert "arg2" in text and "missing" in text.lower()

    def test_argument_value_is_ill_formed_reports_the_argument_and_its_own_trace(self):
        value = {"Add<Integer>": {"arg1": 1, "arg2": {"Nope": 2}}}
        text = explanation({**ADD, **site("Integer", value)}, value)
        assert 'argument "arg2" is not a valid Integer expression' in text
        assert "Nope" in text, "the argument's own failure must be nested underneath"

    def test_is_function_evaluation_on_a_non_function_key(self):
        value = {"Point": {"x": 1, "y": 2}, "isFunctionEvaluation": True}
        text = explanation({**POINT, **site("Point", value)}, value)
        assert "isFunctionEvaluation" in text
        assert "Point" in text


# --------------------------------------------------------------------------------------------------
# Coverage: Narrow, and abstract types
# --------------------------------------------------------------------------------------------------


class TestNarrowAndAbstractFailures:
    def test_narrowing_to_an_abstract_type_is_rejected(self):
        value = {"Abstr": {}}
        text = explanation({**ABSTRACT, **site("ValueDomain", value)}, value)
        assert "Abstr" in text and "abstract" in text.lower()

    def test_an_abstract_expected_type_reports_that_it_has_no_schema(self):
        text = explanation({**ABSTRACT, **site("Abstr", {})}, {})
        assert "as Inst" in text
        assert "abstract" in text.lower()
        assert "no instantiation schema" in text

    def test_narrow_key_that_is_not_a_subtype(self):
        value = {"Point": {"x": 1, "y": 2}}
        text = explanation({**POINT, **site("Integer", value)}, value)
        assert "as Narrow" in text
        assert "Point is not a subtype of Integer" in text

    def test_narrow_whose_schema_rejects_the_value(self):
        value = {"Point": {"x": 1}}
        text = explanation({**POINT, **site("ValueDomain", value)}, value)
        assert "as Narrow (Point)" in text
        assert 'Required property "y" is missing' in text


# --------------------------------------------------------------------------------------------------
# Coverage: every instantiation-schema keyword that can reject a value
# --------------------------------------------------------------------------------------------------

SCHEMA_KEYWORD_CASES = [
    (
        "required",
        {"type": "object", "properties": {"a": {"type": "Integer"}}, "required": ["a"]},
        {},
        ['Required property "a" is missing'],
    ),
    (
        "additionalProperties",
        {"type": "object", "properties": {}, "additionalProperties": False},
        {"z": 1},
        ["Additional property is not allowed"],
    ),
    (
        "property type",
        {"type": "object", "properties": {"a": {"type": "Integer"}}},
        {"a": "s:x"},
        ["is not of type 'integer'"],
    ),
    ("enum", {"type": "integer", "enum": [1, 2, 3]}, 9, ["is not one of", "[1, 2, 3]"]),
    ("const", {"type": "integer", "const": 7}, 8, ["was expected"]),
    ("minimum", {"type": "integer", "minimum": 5}, 1, ["is less than the minimum of 5"]),
    ("maximum", {"type": "integer", "maximum": 5}, 9, ["is greater than the maximum of 5"]),
    ("multipleOf", {"type": "integer", "multipleOf": 3}, 7, ["is not a multiple of 3"]),
    ("pattern", {"type": "string", "pattern": "^s:a"}, "s:b", ["does not match", "'^s:a'"]),
    ("minLength", {"type": "string", "minLength": 8}, "s:ab", ["is too short"]),
    ("minItems", {"type": "array", "items": "Integer", "minItems": 3}, [1], ["is too short"]),
    ("maxItems", {"type": "array", "items": "Integer", "maxItems": 1}, [1, 2], ["is too long"]),
    ("uniqueItems", {"type": "array", "items": "Integer", "uniqueItems": True}, [1, 1], ["has non-unique elements"]),
    ("items element", {"type": "array", "items": "Integer"}, [1, "s:x"], ["is not of type 'integer'"]),
    ("tuple items", {"type": "array", "items": ["Integer", "String"]}, [1, 2], ["is not of type 'string'"]),
    (
        "contains",
        {"type": "array", "items": "Integer", "contains": {"type": "integer", "minimum": 9}},
        [1, 2],
        ["does not contain any element matching the 'contains' schema"],
    ),
    ("anyOf", {"anyOf": [{"type": "integer"}, {"type": "string"}]}, [1], ["anyOf"]),
    (
        "oneOf",
        {"oneOf": [{"type": "integer", "minimum": 0}, {"type": "integer", "maximum": 10}]},
        5,
        ["matches 2 schemas in 'oneOf'"],
    ),
    (
        "allOf",
        {"allOf": [{"type": "integer"}, {"type": "integer", "minimum": 9}]},
        1,
        ["is less than the minimum of 9"],
    ),
    ("not", {"not": {"type": "integer"}}, 1, ["must not match the schema in 'not'"]),
    (
        "if/then",
        {"if": {"type": "integer"}, "then": {"type": "integer", "minimum": 9}},
        1,
        ["is less than the minimum of 9"],
    ),
    (
        "propertyNames",
        {"type": "object", "propertyNames": {"type": "string", "pattern": "^a"}, "additionalProperties": True},
        {"zz": 1},
        ["does not match", "'^a'"],
    ),
    (
        "patternProperties",
        {"type": "object", "patternProperties": {"^a": {"type": "Integer"}}},
        {"ab": "s:x"},
        ["is not of type 'integer'"],
    ),
    (
        "dependencies",
        {
            "type": "object",
            "properties": {"a": {"type": "Integer"}, "b": {"type": "Integer"}},
            "dependencies": {"a": ["b"]},
        },
        {"a": 1},
        ["is a dependency of"],
    ),
    (
        "minProperties",
        {"type": "object", "minProperties": 2, "additionalProperties": True},
        {"a": 1},
        ["does not have enough properties"],
    ),
    (
        "nested required",
        {
            "type": "object",
            "properties": {"o": {"type": "object", "properties": {"q": {"type": "Integer"}}, "required": ["q"]}},
            "required": ["o"],
        },
        {"o": {}},
        ['Required property "q" is missing'],
    ),
]


class TestInstantiationSchemaKeywordsAreExplained:
    """
    Every draft-07 keyword that can reject a value must have its complaint reach the top of the
    expression error. This is the bulk of what the trace exists to carry: before it, all 26 of these
    collapsed into one identical line.
    """

    @pytest.mark.parametrize(
        "label,schema,value,fragments", SCHEMA_KEYWORD_CASES, ids=[case[0] for case in SCHEMA_KEYWORD_CASES]
    )
    def test_keyword_rejection_reaches_the_expression_error(self, label, schema, value, fragments):
        text = explanation({**target(schema), **site("Target", value)}, value)
        for fragment in fragments:
            assert fragment in text, f"{label}: expected {fragment!r} in the trace"

    @pytest.mark.parametrize(
        "label,schema,value,fragments", SCHEMA_KEYWORD_CASES, ids=[case[0] for case in SCHEMA_KEYWORD_CASES]
    )
    def test_keyword_rejection_is_attributed_to_the_instantiation_attempt(self, label, schema, value, fragments):
        text = expression_error({**target(schema), **site("Target", value)})
        assert "as Inst (Target)" in text
        assert "does not satisfy the instantiation schema of Target" in text


# --------------------------------------------------------------------------------------------------
# Coverage: provenance and access
# --------------------------------------------------------------------------------------------------


class TestProvenanceAndAccessViolations:
    """
    These are checked in `parse_expression` *after* an alternative matched, so the expression is
    well-formed and merely unusable here. The message therefore has to say what it got, not just that
    something was wrong.
    """

    def test_addr_provenance_rejects_a_literal(self):
        text = expression_error(site("Integer", 1, provenance="Addr"))
        assert "Provenance violation" in text
        assert "addressable" in text
        assert "value domain instantiation" in text, "the message must name what the expression actually is"

    def test_addr_provenance_rejects_a_non_addressable_function_evaluation(self):
        value = {"Add<Integer>": {"arg1": 1, "arg2": 2}}
        text = expression_error({**ADD, **site("Integer", value, provenance="Addr")})
        assert "Provenance violation" in text
        assert "Function evaluation" in text

    def test_addr_provenance_accepts_an_addressable_function_evaluation(self):
        """The negative control: a Function whose result is `Addr` is fine, so nothing is reported."""
        context = check_concepts({**ADDR_FUNCTION, **site("Integer", {"AddrF": {}}, provenance="Addr")})
        assert "Site" in context.model.value_domains


# --------------------------------------------------------------------------------------------------
# Coverage: string expressions
# --------------------------------------------------------------------------------------------------


class TestStringExpressionFailures:
    def test_a_type_template_variable_used_as_a_value_is_rejected(self):
        text = expression_error(
            {
                "G": {
                    "directParents": ["ValueDomain"],
                    "data": {
                        "templateContext": ["T"],
                        "instantiation": {
                            "type": "object",
                            "properties": {"p": {"type": "Integer", "default": "T"}},
                        },
                    },
                }
            }
        )
        assert "template variable" in text
        assert "T" in text, "the offending template variable must be named"

    def test_an_unknown_string_reports_each_kind_of_name_that_was_looked_for(self):
        text = explanation({**site("Integer", "s:nope")}, "s:nope")
        assert "is not a variable of this Concept Hierarchy" in text
        assert "is not a template variable in this context" in text


# --------------------------------------------------------------------------------------------------
# Structural invariants of the trace
# --------------------------------------------------------------------------------------------------

FAILING_HIERARCHIES = {
    "object matching nothing": ({**site("Integer", {"NotAType": 1})}, {"NotAType": 1}),
    "missing required property": ({**POINT, **site("Point", {"x": 1})}, {"x": 1}),
    "unknown string": ({**site("Integer", "s:nope")}, "s:nope"),
    "bad function argument": (
        {**ADD, **site("Integer", {"Add<Integer>": {"arg1": 1, "arg2": {"Nope": 2}}})},
        {"Add<Integer>": {"arg1": 1, "arg2": {"Nope": 2}}},
    ),
    "narrow rejected": ({**POINT, **site("ValueDomain", {"Point": {"x": 1}})}, {"Point": {"x": 1}}),
}


class TestTraceStructuralInvariants:
    """Properties that must hold of every failure, whatever went wrong."""

    @pytest.mark.parametrize("label", sorted(FAILING_HIERARCHIES))
    def test_every_failure_has_a_headline_and_at_least_one_attempt(self, label):
        concepts, _value = FAILING_HIERARCHIES[label]
        text = expression_error(concepts)
        assert "Reason:" in text, f"{label}: the headline reason must be present"
        assert "as " in text, f"{label}: at least one attempt must be reported"

    @pytest.mark.parametrize("label", sorted(FAILING_HIERARCHIES))
    def test_every_failure_is_located_at_the_default(self, label):
        concepts, _value = FAILING_HIERARCHIES[label]
        text = expression_error(concepts)
        assert '"properties": "p": "type": "default"' in text

    def test_a_schema_error_is_not_reported_twice(self):
        """
        Regression guard. ``InstantiationSearch.errors`` and the matched ``ConstraintGroupAttempt.errors``
        are the same list; rendering both doubled every schema error in the first implementation.
        """
        text = expression_error({**POINT, **site("Point", {"x": 1})})
        assert text.count('Required property "y" is missing') == 1

    def test_nested_errors_extend_the_parent_location(self):
        value = {"x": 1, "y": "s:nope"}
        text = expression_error({**POINT, **site("Point", value)})
        assert '"properties": "p": "type": "default": "y"' in text, (
            "an error inside property y must be located inside the default, not at it"
        )

    def test_the_headline_reason_also_appears_for_nested_failures(self):
        """The trace supplements the reason; it never replaces it."""
        value = {"x": 1, "y": "s:nope"}
        text = expression_error({**POINT, **site("Point", value)})
        assert "Could not match a valid Point expression" in text
        assert "Could not match a valid Integer expression" in text, "the nested failure keeps its own reason"


# --------------------------------------------------------------------------------------------------
# Coverage: depth
# --------------------------------------------------------------------------------------------------


class TestDeeplyNestedFailures:
    """
    An error must survive an arbitrary alternation of Inst / Narrow / FEval on the way up. Each test
    below adds one more level between the site and the actual mistake.
    """

    def test_depth_two_inst_inst(self):
        outer = {
            "Outer": {
                "directParents": ["ValueDomain"],
                "data": {
                    "instantiation": {
                        "type": "object",
                        "properties": {"inner": {"type": "Point"}},
                        "required": ["inner"],
                    }
                },
            }
        }
        text = expression_error({**POINT, **outer, **site("Outer", {"inner": {"x": 1}})})
        assert 'Required property "y" is missing' in text
        assert '"default": "inner"' in text

    def test_depth_three_inst_inst_inst(self):
        levels = {
            "L2": {
                "directParents": ["ValueDomain"],
                "data": {
                    "instantiation": {
                        "type": "object",
                        "properties": {"b": {"type": "Point"}},
                        "required": ["b"],
                    }
                },
            },
            "L1": {
                "directParents": ["ValueDomain"],
                "data": {
                    "instantiation": {"type": "object", "properties": {"a": {"type": "L2"}}, "required": ["a"]},
                },
            },
        }
        text = expression_error({**POINT, **levels, **site("L1", {"a": {"b": {"x": 1}}})})
        assert 'Required property "y" is missing' in text
        assert '"default": "a": "b"' in text, "the full path to the mistake must be visible"

    def test_depth_three_inst_feval_feval(self):
        value = {
            "x": 1,
            "y": {"Add<Integer>": {"arg1": 1, "arg2": {"Add<Integer>": {"arg1": 1, "arg2": {"Nope": 9}}}}},
        }
        text = explanation({**ADD, **POINT, **site("Point", value)}, value)
        assert text.count('argument "arg2" is not a valid Integer expression') == 2, (
            "both Function evaluation levels must be named"
        )
        assert "Nope" in text, "the innermost offending key must survive three levels"

    def test_depth_three_narrow_inst_feval(self):
        holder = {
            "Holder": {
                "directParents": ["ValueDomain"],
                "data": {
                    "instantiation": {
                        "type": "object",
                        "properties": {"h": {"type": "Integer"}},
                        "required": ["h"],
                    }
                },
            }
        }
        value = {"Holder": {"h": {"Add<Integer>": {"arg1": 1, "arg2": {"Nope": 9}}}}}
        text = explanation({**ADD, **holder, **site("ValueDomain", value)}, value)
        assert "as Narrow (Holder)" in text
        assert 'argument "arg2" is not a valid Integer expression' in text
        assert "Nope" in text


# ==================================================================================================
# Where each part of a trace is reported
# ==================================================================================================


class TestTheTraceDescendsWithTheValue:
    """
    Every level of a trace is reported at the location of the value *that* level was parsing.

    It used not to be. `ExpressionAttempt.as_error` rendered a nested cause at the enclosing attempt's
    location, so an argument three levels down claimed the location of the outermost value: on
    `animal_kingdom.json`, ``"n" is not a variable`` -- which is inside
    ``Condition/condition/LessEqual<Integer>/arg1`` -- was reported at ``procedure/Condition``, and every
    alternative tried for it said the same. A trace whose locations do not move is worse than no locations
    at all, because it reads as though the failure really is at the top.

    These assert the whole path, not a substring of it. A containment check passes for a location that has
    merely stopped early, which is exactly the failure being guarded against.
    """

    def test_a_failing_function_argument_is_reported_at_that_argument(self):
        value = {"Add<Integer>": {"arg1": 1, "arg2": "s:x"}}
        text = expression_error({**ADD, **site("Integer", value)})
        assert location_reporting(text, 'as variable: "s:x"') == site_path("Add<Integer>", "arg2")

    def test_the_attempt_that_names_the_argument_stays_at_the_evaluation(self):
        """
        The attempt is about the *evaluation*: it is what was being parsed when the argument was found
        wanting. Only its cause belongs further in -- which is why the two are separate fields.
        """
        value = {"Add<Integer>": {"arg1": 1, "arg2": "s:x"}}
        text = expression_error({**ADD, **site("Integer", value)})
        assert location_reporting(text, 'as FEval (Add<Integer>): argument "arg2"') == site_path()

    def test_each_nesting_level_adds_its_own_segments(self):
        """Two evaluations deep: the innermost failure is four segments below the site, not one."""
        inner = {"Add<Integer>": {"arg1": 1, "arg2": "s:x"}}
        value = {"Add<Integer>": {"arg1": 1, "arg2": inner}}
        text = expression_error({**ADD, **site("Integer", value)})
        assert location_reporting(text, 'as variable: "s:x"') == site_path(
            "Add<Integer>", "arg2", "Add<Integer>", "arg2"
        )

    def test_a_failure_inside_an_instantiated_property_is_reported_inside_it(self):
        """``Inst(Point) -> y -> FEval(Add) -> arg2``: the schema error names the property and the argument."""
        value = {"x": 1, "y": {"Add<Integer>": {"arg1": 1, "arg2": {"NotAType": 9}}}}
        text = expression_error({**ADD, **POINT, **site("Point", value)})
        assert location_reporting(text, "is not of type 'integer'") == site_path("y", "Add<Integer>", "arg2")

    def test_every_alternative_tried_for_one_value_shares_that_value_s_location(self):
        """
        The alternatives are all attempts on the *same* text, so they must agree -- and they must agree on
        the innermost location, not on the outermost one.
        """
        value = {"Add<Integer>": {"arg1": 1, "arg2": "s:x"}}
        text = expression_error({**ADD, **site("Integer", value)})
        argument_location = site_path("Add<Integer>", "arg2")
        alternatives = {
            message.split(":")[0]
            for location, message in located_messages(text)
            if location == argument_location and message.startswith("as ")
        }
        assert len(alternatives) > 1, f"expected several alternatives at the argument, got {alternatives}"


class TestTheFunctionCompositionArgsNodeLocatesItsKeyOnce:
    """
    The ``"properties": "args"`` node stands *on* the Function name, and the evaluation parser appends the
    key itself. Handing the callee the location it was standing on spelled the key twice --
    ``procedure/Condition/Condition/condition`` -- which is a path that exists in no document.
    """

    COMPOSED = {
        "Holder": {
            "directParents": ["ValueDomain"],
            "data": {"instantiation": {"type": "object", "properties": {"p": {"type": "FunctionComposition"}}}},
        }
    }

    def test_the_function_name_appears_once_in_the_location(self):
        value = {"Add<Integer>": {"arg1": 1, "arg2": "s:x"}}
        text = expression_error({**ADD, **site("FunctionComposition", value)})
        location = location_reporting(text, 'as variable: "s:x"')
        assert location.count('"Add<Integer>"') == 1, f"the key is spelled twice in {location}"

    def test_the_argument_is_reached_through_the_key(self):
        value = {"Add<Integer>": {"arg1": 1, "arg2": "s:x"}}
        text = expression_error({**ADD, **site("FunctionComposition", value)})
        assert location_reporting(text, 'as variable: "s:x"') == site_path("Add<Integer>", "arg2")
