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
Integration tests: literal template variables used as expression values.

A template variable constrained to a *literal sort* rather than a type -- ``"N": "Literal:integer"`` --
stands for a value, not a type, so writing ``"N"` where an expression is expected means "whatever `N` is
bound to". That makes it the one substitution point that is **value**-level rather than type-level, and
the only one where substitution changes the expression's *class*:

- with ``N`` unbound, ``"N"`` parses to a :class:`LiteralTemplateVariableValue`, typed by whichever
  concept registers a ``defaultSerialization`` for the sort's JSON type (``integer`` -> ``Integer``);
- under a ground application such as ``H<3>``, the *string* ``"N"`` becomes the JSON value ``3`` before
  anything is interpreted, and the expression is an :class:`InstExpression` instead.

**A substituted literal is recognised only by ``defaultSerialization``.** Neither the name alternatives
(``Var``, an instance property chain, a literal template variable) nor the instantiation-schema route
applies to it: it is a value, so it is classified by what its JSON type serialises to and by nothing else.
That is what these tests assert -- an `InstExpression` whose ``value`` tree is ``None`` and whose
``value_type`` is the sort's serialisation concept.

One consequence worth knowing: the defaultSerialization route keeps no parsed value, so the *value* the
literal was substituted to is not recoverable from the expression -- ``H<3>`` and ``H<9>`` produce
identical expressions, and ``Expression.unparsed`` still holds the pre-substitution ``"N"``. The value
lives in the site's type application instead. The conversion itself is covered by
``tests/unit/test_literal_value_conversion.py``.

No rewrite of a parsed tree could do that -- the two are different classes reached through different
branches -- which is why substitution re-parses. See ``TODO_DEFAULT_EXPANSION_CYCLES.md`` §4.2.

The four sorts are ``boolean`` / ``integer`` / ``number`` / ``string``; their literal forms are ``true`` /
``false``, ``-7``, ``3.5`` and ``"quoted"``.
"""

from __future__ import annotations

import pytest

from concept_hierarchy.data.expressions.expression import Expression
from concept_hierarchy.data.expressions.instantiated_value import ParsedCustomValue, ParsedStructural
from concept_hierarchy.data.expressions.subexpressions import (
    DefaultSerializationExpression,
    FunctionEvaluation,
    InstExpression,
    LiteralTemplateVariableValue,
    Variable,
)
from concept_hierarchy.errors import ConceptHierarchyError
from tests.integration.test_expression_parsing import build_hierarchy, check_concepts, check_hierarchy

# --------------------------------------------------------------------------------------------------
# Hierarchy fragments
# --------------------------------------------------------------------------------------------------

BOOLEAN = {
    "Boolean": {
        "directParents": ["ValueDomain"],
        "data": {"defaultSerialization": "boolean", "instantiation": {"type": "boolean"}},
    }
}
"""``CH_PRELUDE`` registers a defaultSerialization for integer/number/string but not boolean."""

ADD = {
    "Add": {
        "directParents": ["FunctionReturning"],
        "data": {
            "templateContext": {"order": ["T"], "T": "Numeric", "substitution": {"FunctionReturning:T": "T"}},
            "interface": {"arg1": ["T"], "arg2": ["T"], "res": "T"},
        },
    }
}

SORT_SITE_TYPE = {"integer": "Integer", "number": "Number", "string": "String", "boolean": "Boolean"}
"""Each literal sort, and the concept its JSON type serialises to."""


def holder(sort: str, site_type: str, default: object, name: str = "H", variable: str = "N") -> dict:
    """A ValueDomain with one literal template variable and one defaulted property."""
    return {
        name: {
            "directParents": ["ValueDomain"],
            "data": {
                "templateContext": {"order": [variable], variable: f"Literal:{sort}"},
                "instantiation": {
                    "type": "object",
                    "properties": {"p": {"type": site_type, "default": default}},
                    "required": ["p"],
                },
            },
        }
    }


def uses(applied: str, name: str = "Uses", prop: str = "u") -> dict:
    return {
        name: {
            "directParents": ["ValueDomain"],
            "data": {
                "instantiation": {
                    "type": "object",
                    "properties": {prop: {"type": applied, "default": {}}},
                    "required": [prop],
                }
            },
        }
    }


# --------------------------------------------------------------------------------------------------
# Accessors
# --------------------------------------------------------------------------------------------------


def default_expression(context, value_domain: str, prop: str) -> Expression:
    for _constraint, schema in context.model.value_domains[value_domain].instantiation:
        for node in schema.walk():
            if node.has_default and str(node.location_id[-2]) == prop:
                return node.parsed_default_expr
    raise AssertionError(f"no default site {prop!r} on {value_domain}")


def materialised_leaves(expression: Expression) -> list[ParsedCustomValue]:
    """Every leaf of ``expression``'s value that was filled in from a default."""
    value = expression.value
    assert isinstance(value, InstExpression) and value.value is not None, (
        f"expected an Inst expression with a value tree, got {value!r}"
    )
    return [n for n in value.value.walk() if isinstance(n, ParsedCustomValue) and n.used_default]


def substituted_expression(context, applied_via: str = "Uses", prop: str = "u") -> Expression:
    """The expression the literal-variable site resolved to under a ground application."""
    leaves = materialised_leaves(default_expression(context, applied_via, prop))
    assert len(leaves) == 1, f"expected exactly one materialised leaf, got {leaves!r}"
    assert leaves[0].expression is not None, "the materialised default was never resolved"
    return leaves[0].expression


def is_default_serialized(expression: Expression) -> bool:
    """Whether this `Inst` matched by *defaultSerialization* rather than by an instantiation schema."""
    value = expression.value
    return isinstance(value, InstExpression) and value.value is None


def assert_is_a_literal(expression: Expression, value_type: str, json_value: object = ...) -> None:
    """
    Assert the shape every substituted literal must have: a `DefaultSerializationExpression`, carrying no
    parsed value tree, typed by the concept its JSON type serialises to, and holding the substituted value.

    Asserting the *absence* of the tree is the point, not an accident of the assertion: matching an
    instantiation schema instead would give the literal a structural reading, and would make the same
    literal mean different things at sites whose types happen to have different schemas.

    ``json_value`` is read from the expression, not from `Expression.unparsed` -- that holds the *declared*
    text and is never rewritten by substitution, so it still says ``"N"`` here.
    """
    value = expression.value
    assert isinstance(value, DefaultSerializationExpression), (
        f"a substituted literal must be recognised by defaultSerialization; got {type(value).__name__}"
    )
    assert value.value is None, (
        f"a defaultSerialization expression has no parsed value tree, but this one produced {value.value!r}"
    )
    assert value.value_type is not None and value.value_type.full_name == value_type, (
        f"expected the literal to be typed {value_type}, got {value.value_type}"
    )
    if json_value is not ...:
        assert value.json_value == json_value and type(value.json_value) is type(json_value), (
            f"expected the substituted value {json_value!r} ({type(json_value).__name__}), "
            f"got {value.json_value!r} ({type(value.json_value).__name__})"
        )


def parsed_value_of(expression: Expression) -> object:
    """
    The raw JSON value an `Inst` expression parsed.

    Only meaningful for an expression that matched an *instantiation schema*: the defaultSerialization
    route keeps no value tree, which is exactly the shape :func:`assert_is_a_literal` requires of every
    substituted literal. So this is for the ordinary, non-literal values used as contrast.
    """
    value = expression.value
    assert isinstance(value, InstExpression), f"expected an Inst expression, got {type(value).__name__}"
    assert value.value is not None, (
        f"this {value.value_type} expression matched by defaultSerialization, which keeps no parsed value "
        f"tree, so there is no value to read"
    )
    assert isinstance(value.value, ParsedStructural), f"expected a structural value, got {value.value!r}"
    return value.value.value


def site_type_of(context, value_domain: str, prop: str) -> str:
    """
    The type application at a default site -- where a literal argument's *value* actually lives.

    The expression does not carry it (see the module docstring), so this is what distinguishes ``H<3>``
    from ``H<9>``.
    """
    for _constraint, schema in context.model.value_domains[value_domain].instantiation:
        for node in schema.walk():
            if node.has_default and str(node.location_id[-2]) == prop:
                return node.custom_type.full_name
    raise AssertionError(f"no default site {prop!r} on {value_domain}")


# --------------------------------------------------------------------------------------------------
# A. Unbound: the variable stands for itself
# --------------------------------------------------------------------------------------------------


class TestAnUnboundLiteralVariable:
    """
    With nothing binding ``N``, ``"N"`` is still a well-formed expression: it is a literal template
    variable, typed by the concept that serialises its sort.
    """

    @pytest.mark.parametrize("sort,site_type", sorted(SORT_SITE_TYPE.items()))
    def test_it_parses_as_a_literal_template_variable(self, sort, site_type):
        context = check_concepts({**BOOLEAN, **holder(sort, site_type, "N")})
        expression = default_expression(context, "H", "p")
        assert isinstance(expression.value, LiteralTemplateVariableValue), (
            f"{sort}: got {type(expression.value).__name__}"
        )

    @pytest.mark.parametrize("sort,site_type", sorted(SORT_SITE_TYPE.items()))
    def test_its_type_is_the_default_serialization_of_its_sort(self, sort, site_type):
        context = check_concepts({**BOOLEAN, **holder(sort, site_type, "N")})
        expression = default_expression(context, "H", "p")
        assert expression.value.value_type.full_name == site_type

    @pytest.mark.parametrize("sort,site_type", sorted(SORT_SITE_TYPE.items()))
    def test_it_names_the_variable_it_stands_for(self, sort, site_type):
        context = check_concepts({**BOOLEAN, **holder(sort, site_type, "N")})
        assert default_expression(context, "H", "p").value.template_variable_name == "N"

    def test_it_is_reported_as_value_template_dependent(self):
        context = check_concepts({**holder("integer", "Integer", "N")})
        assert default_expression(context, "H", "p").is_value_template_dependent


# --------------------------------------------------------------------------------------------------
# B. Bound: the variable becomes its value, and the expression changes class
# --------------------------------------------------------------------------------------------------


class TestABoundLiteralVariable:
    """
    Under a ground application the string ``"N"`` is replaced by the JSON value before anything is
    interpreted, so the expression is no longer a literal template variable at all.
    """

    @pytest.mark.parametrize(
        "sort,site_type,argument,expected",
        [
            ("integer", "Integer", "3", 3),
            ("integer", "Integer", "-7", -7),
            ("integer", "Integer", "0", 0),
            ("integer", "Integer", "1000000", 1000000),
            ("number", "Number", "3.5", 3.5),
            ("number", "Number", "-0.25", -0.25),
            ("boolean", "Boolean", "true", True),
            ("boolean", "Boolean", "false", False),
            ("string", "String", '"s:hello"', "s:hello"),
            ("string", "String", '"hello"', "hello"),
        ],
    )
    def test_the_literal_is_recognised_by_default_serialization(self, sort, site_type, argument, expected):
        context = check_concepts({**BOOLEAN, **holder(sort, site_type, "N"), **uses(f"H<{argument}>")})
        expression = substituted_expression(context)
        assert not isinstance(expression.value, LiteralTemplateVariableValue), "substitution must change the class"
        assert_is_a_literal(expression, site_type, expected)

    def test_two_applications_keep_their_own_values(self):
        """The value is on the expression and the application is on the site; the two must agree."""
        context = check_concepts(
            {
                **holder("integer", "Integer", "N"),
                **uses("H<3>"),
                **uses("H<9>", name="UsesNine", prop="n"),
            }
        )
        assert_is_a_literal(substituted_expression(context), "Integer", 3)
        assert_is_a_literal(substituted_expression(context, "UsesNine", "n"), "Integer", 9)
        assert site_type_of(context, "Uses", "u") == "H<3>"
        assert site_type_of(context, "UsesNine", "n") == "H<9>"

    def test_the_declared_site_stays_a_literal_template_variable(self):
        """One declared site, one entry: the per-application result lives on the substituted copy."""
        context = check_concepts({**holder("integer", "Integer", "N"), **uses("H<3>")})
        assert isinstance(default_expression(context, "H", "p").value, LiteralTemplateVariableValue)


# --------------------------------------------------------------------------------------------------
# C. Nested positions
# --------------------------------------------------------------------------------------------------


class TestNestedUses:
    """The variable is a *value*, so it can appear anywhere a value can -- at any depth."""

    def test_inside_a_function_evaluation_argument(self):
        context = check_concepts(
            {
                **ADD,
                **holder("integer", "Integer", {"Add<Integer>": {"arg1": "N", "arg2": 1}}),
                **uses("H<3>"),
            }
        )
        feval = substituted_expression(context)
        assert isinstance(feval.value, FunctionEvaluation)
        assert_is_a_literal(feval.value.arguments["arg1"], "Integer", 3)

    def test_inside_both_arguments_of_a_function_evaluation(self):
        context = check_concepts(
            {
                **ADD,
                **holder("integer", "Integer", {"Add<Integer>": {"arg1": "N", "arg2": "N"}}),
                **uses("H<4>"),
            }
        )
        feval = substituted_expression(context)
        for argument in ("arg1", "arg2"):
            assert_is_a_literal(feval.value.arguments[argument], "Integer", 4)

    def test_inside_a_supplied_object_property(self):
        """This one only reaches the parser across the `ValueInstantiationContext` seam."""
        pair = {
            "Pair": {
                "directParents": ["ValueDomain"],
                "data": {
                    "instantiation": {
                        "type": "object",
                        "properties": {"a": {"type": "Integer"}},
                        "required": ["a"],
                    }
                },
            }
        }
        context = check_concepts({**pair, **holder("integer", "Pair", {"a": "N"}), **uses("H<5>")})
        inner = substituted_expression(context)
        assert isinstance(inner.value, InstExpression)
        leaves = [n for n in inner.value.value.walk() if isinstance(n, ParsedCustomValue)]
        assert len(leaves) == 1
        assert_is_a_literal(leaves[0].expression, "Integer", 5)

    def test_inside_an_array_item(self):
        array = {
            "Arr": {
                "directParents": ["ValueDomain"],
                "data": {"instantiation": {"type": "array", "items": "Integer"}},
            }
        }
        context = check_concepts({**array, **holder("integer", "Arr", ["N", 1]), **uses("H<6>")})
        inner = substituted_expression(context)
        leaves = [n for n in inner.value.value.walk() if isinstance(n, ParsedCustomValue)]
        assert len(leaves) == 2, "the substituted item and the written one"
        assert_is_a_literal(leaves[0].expression, "Integer", 6)

    def test_two_variables_in_one_schema(self):
        two = {
            "Two": {
                "directParents": ["ValueDomain"],
                "data": {
                    "templateContext": {"order": ["M", "N"], "M": "Literal:integer", "N": "Literal:integer"},
                    "instantiation": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "Integer", "default": "M"},
                            "b": {"type": "Integer", "default": "N"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        }
        context = check_concepts({**two, **uses("Two<1, 2>")})
        leaves = materialised_leaves(default_expression(context, "Uses", "u"))
        by_name = {str(leaf.location_id[-1]): leaf for leaf in leaves}
        assert set(by_name) == {"a", "b"}
        assert_is_a_literal(by_name["a"].expression, "Integer", 1)
        assert_is_a_literal(by_name["b"].expression, "Integer", 2)
        assert site_type_of(context, "Uses", "u") == "Two<1, 2>"


# --------------------------------------------------------------------------------------------------
# D. Used where it may not be
# --------------------------------------------------------------------------------------------------


class TestInvalidUses:
    def test_a_type_template_variable_is_not_a_value(self):
        """``T: Concept`` names a type, so it can not stand in for one."""
        with pytest.raises(ConceptHierarchyError, match="type template variable"):
            check_concepts(
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

    def test_a_sort_that_does_not_fit_the_site(self):
        """An ``integer`` variable at a ``String`` site: ``Integer`` is not a subtype of ``String``."""
        with pytest.raises(ConceptHierarchyError) as excinfo:
            check_concepts(holder("integer", "String", "N"))
        assert "String" in str(excinfo.value)

    def test_a_sort_with_no_registered_default_serialization(self):
        """Without a ``boolean`` defaultSerialization there is no type to give the variable."""
        with pytest.raises(ConceptHierarchyError) as excinfo:
            check_concepts(holder("boolean", "Integer", "N"))
        assert "Integer" in str(excinfo.value)

    def test_a_name_that_is_neither_a_variable_nor_anything_else(self):
        with pytest.raises(ConceptHierarchyError) as excinfo:
            check_concepts(holder("integer", "Integer", "M"))
        text = str(excinfo.value)
        assert "M" in text


class TestInvalidArguments:
    """The application has to satisfy the variable's sort."""

    @pytest.mark.parametrize(
        "sort,site_type,argument",
        [
            ("integer", "Integer", "true"),
            ("integer", "Integer", "3.5"),
            ("boolean", "Boolean", "3"),
            ("number", "Number", "true"),
        ],
    )
    def test_an_argument_of_the_wrong_sort_is_rejected(self, sort, site_type, argument):
        with pytest.raises(ConceptHierarchyError) as excinfo:
            check_concepts({**BOOLEAN, **holder(sort, site_type, "N"), **uses(f"H<{argument}>")})
        assert "Literal:" in str(excinfo.value), "the unsatisfied literal constraint must be named"

    def test_a_type_argument_is_rejected_for_a_literal_variable(self):
        with pytest.raises(ConceptHierarchyError) as excinfo:
            check_concepts({**holder("integer", "Integer", "N"), **uses("H<Integer>")})
        assert "non-type constraint" in str(excinfo.value)


# --------------------------------------------------------------------------------------------------
# E. The string sort
# --------------------------------------------------------------------------------------------------


class TestTheStringSort:
    """
    The string sort has one wrinkle the others do not: the literal is *written* quoted, and the quotes
    belong to the notation rather than to the value. The conversion itself is unit-tested; what matters
    here is that a string literal takes the same route as every other sort.
    """

    def test_declaring_and_using_it_unbound_works(self):
        context = check_concepts(holder("string", "String", "N"))
        assert isinstance(default_expression(context, "H", "p").value, LiteralTemplateVariableValue)

    @pytest.mark.parametrize(
        "argument,expected",
        [('"s:hello"', "s:hello"), ('"hello"', "hello"), ('"s:x"', "s:x"), ('""', "")],
    )
    def test_every_string_argument_takes_the_default_serialization_route(self, argument, expected):
        """
        Including ``"s:hello"``, which ``String``'s own instantiation schema would have matched. A literal
        must not be read structurally, so the schema is not consulted for it at all.
        """
        context = check_concepts({**holder("string", "String", "N"), **uses(f"H<{argument}>")})
        assert_is_a_literal(substituted_expression(context), "String", expected)


class TestDefaultSerializationIsNotLiteralSpecific:
    """
    Isolating the two oddities above from the literal feature: they show up for an ordinary string default
    with no template variable anywhere, so they belong to the defaultSerialization path.
    """

    def _string_default(self, default: str):
        concepts = {
            "S": {
                "directParents": ["ValueDomain"],
                "data": {
                    "instantiation": {
                        "type": "object",
                        "properties": {"p": {"type": "String", "default": default}},
                        "required": ["p"],
                    }
                },
            }
        }
        return default_expression(check_concepts(concepts), "S", "p")

    def test_a_prefixed_string_matches_the_instantiation_schema(self):
        assert parsed_value_of(self._string_default("s:hello")) == "s:hello"

    def test_a_plain_unprefixed_string_behaves_the_same(self):
        """
        Accepted, but through defaultSerialization -- so ``String``'s ``^s:`` pattern is not enforced on
        this route, and the value is discarded. Worth knowing before anything downstream needs to read it.
        """
        expression = self._string_default("hello")
        assert is_default_serialized(expression)
        assert expression.value.value_type.full_name == "String"


# --------------------------------------------------------------------------------------------------
# F. Where the substituted value has to land
# --------------------------------------------------------------------------------------------------


class TestTheSiteTheValueLandsIn:
    """
    Once substituted, the literal is an ordinary value and is subject to the ordinary subtype rules of the
    site it sits at -- nothing about it being a template argument changes that.
    """

    def test_a_site_of_a_supertype_accepts_it(self):
        """``Integer`` is a subtype of ``Number``, so an integer literal is fine at a ``Number`` site."""
        context = check_concepts({**holder("integer", "Number", "N"), **uses("H<3>")})
        assert_is_a_literal(substituted_expression(context), "Integer")

    def test_a_site_of_a_subtype_rejects_it(self):
        """The other direction does not hold: ``Number`` is not a subtype of ``Integer``."""
        with pytest.raises(ConceptHierarchyError, match="expected Integer"):
            check_concepts({**holder("number", "Integer", "N"), **uses("H<3.5>")})

    def test_an_abstract_site_accepts_it_too(self):
        """``ValueDomain`` has no instantiation schema at all, which changes nothing: the route is the same."""
        context = check_concepts({**holder("integer", "ValueDomain", "N"), **uses("H<3>")})
        assert_is_a_literal(substituted_expression(context), "Integer")


class TestApplicationsThatNestFurther:
    def test_a_literal_application_used_inside_another_templated_value_domain(self):
        outer = {
            "Outer": {
                "directParents": ["ValueDomain"],
                "data": {
                    "templateContext": {"order": ["M"], "M": "Literal:integer"},
                    "instantiation": {
                        "type": "object",
                        "properties": {"o": {"type": "H<7>", "default": {}}},
                        "required": ["o"],
                    },
                },
            }
        }
        context = check_concepts({**holder("integer", "Integer", "N"), **outer, **uses("Outer<1>")})
        # Uses.u -> Inst(Outer<1>) -> o -> Inst(H<7>) -> p -> the literal 7, not 1.
        outer_leaf = substituted_expression(context)
        inner_leaves = [n for n in outer_leaf.value.value.walk() if isinstance(n, ParsedCustomValue) and n.used_default]
        assert len(inner_leaves) == 1
        assert_is_a_literal(inner_leaves[0].expression, "Integer", 7)
        # The inner application binds N to 7 and Outer's own M must not leak into it -- visible only in
        # the type, since the expression carries no value.
        assert inner_leaves[0].schema_node.location_id[1] == "H"

    def test_a_function_evaluation_key_alongside_a_literal_variable(self):
        """
        The enclosing concept's variable (``N``, a literal) and the Function's (``T``, a type) are
        different names, which is what makes this a regression guard: substituting the Function's
        interface with the *enclosing* context in hand rejects every hierarchy where the two differ.
        """
        context = check_concepts(
            {
                **ADD,
                **holder("integer", "Integer", {"Add<Integer>": {"arg1": "N", "arg2": 1}}),
                **uses("H<3>"),
            }
        )
        feval = substituted_expression(context)
        assert isinstance(feval.value, FunctionEvaluation)
        assert feval.value.f_type.full_name == "Add<Integer>"
        assert_is_a_literal(feval.value.arguments["arg1"], "Integer", 3)


# --------------------------------------------------------------------------------------------------
# G. A substituted literal is a value, never a name
# --------------------------------------------------------------------------------------------------

STRING_HOLDER = {
    "H": {
        "directParents": ["ValueDomain"],
        "data": {
            "templateContext": {"order": ["N"], "N": "Literal:string"},
            "instantiation": {
                "type": "object",
                "properties": {"p": {"type": "String", "default": "N"}},
                "required": ["p"],
            },
        },
    }
}


def string_literal_hierarchy(argument: str, instances: dict | None = None) -> dict:
    return build_hierarchy({**STRING_HOLDER, **uses(f"H<{argument}>")}, instances=instances)


class TestASubstitutedLiteralIsAValueNotAName:
    """
    Every string alternative -- ``Var``, an instance property chain, a literal template variable --
    interprets a string as a **name**. A string literal is not a name: it is the string. So once a literal
    template variable has been substituted, those alternatives must not be tried at all, and the value is
    classified only by what it is.

    Without that, a literal collides with whatever happens to be in scope at the site it is used, and the
    collision changes both the expression's class and its type -- so the same ``H<"...">`` means different
    things in different hierarchies, for reasons the author of ``H`` cannot see.
    """

    def test_a_literal_that_collides_with_a_variable_stays_a_string(self):
        """
        With a global ``one`` in scope, ``H<"one">`` used to parse as *that variable* and be rejected with
        "Type Integer of variable one is not a subtype of String".
        """
        context = check_hierarchy(string_literal_hierarchy('"one"', instances={"one": 1}))
        expression = substituted_expression(context)
        assert isinstance(expression.value, InstExpression), (
            f"expected the literal to stay a value; got {type(expression.value).__name__}"
        )
        assert expression.value.value_type.full_name == "String"

    def test_a_dotted_literal_is_not_an_instance_property_chain(self):
        """``"one.x"`` used to be read as a property chain and crash in `get_type_of_instance_property`."""
        context = check_hierarchy(string_literal_hierarchy('"one.x"', instances={"one": 1}))
        expression = substituted_expression(context)
        assert isinstance(expression.value, InstExpression)
        assert expression.value.value_type.full_name == "String"

    def test_a_dotted_literal_with_an_unknown_root_is_also_fine(self):
        context = check_hierarchy(string_literal_hierarchy('"a.b.c"'))
        assert isinstance(substituted_expression(context).value, InstExpression)

    @pytest.mark.parametrize("argument", ['"s:x"', '"one"', '"one.x"', '"N"', '"Integer"', '"true"'])
    def test_the_class_does_not_depend_on_what_the_text_looks_like(self, argument):
        """
        A literal means the same thing whatever its text resembles -- a variable, a dotted chain, a
        template variable, a concept name, a boolean.
        """
        context = check_hierarchy(string_literal_hierarchy(argument, instances={"one": 1}))
        expression = substituted_expression(context)
        assert isinstance(expression.value, InstExpression)
        assert expression.value.value_type.full_name == "String"

    def test_a_real_variable_reference_is_still_a_variable(self):
        """
        The guard on the other side: skipping the name alternatives must apply *only* to substituted
        literals. An ordinary string written at an expression site is still resolved as a name.
        """
        concepts = {
            "V": {
                "directParents": ["ValueDomain"],
                "data": {
                    "instantiation": {
                        "type": "object",
                        "properties": {"p": {"type": "Integer", "default": "one"}},
                        "required": ["p"],
                    }
                },
            }
        }
        context = check_hierarchy(build_hierarchy(concepts, instances={"one": 1}))
        expression = default_expression(context, "V", "p")
        assert isinstance(expression.value, Variable), (
            f"a written name must still be a variable; got {type(expression.value).__name__}"
        )
        assert expression.value.variable_name == "one"

    def test_a_non_string_literal_is_unaffected(self):
        """Only strings could ever collide with a name, so the other sorts go through unchanged."""
        context = check_concepts({**holder("integer", "Integer", "N"), **uses("H<3>")})
        assert_is_a_literal(substituted_expression(context), "Integer")


class TestTheSubstitutedValueIsRecoverable:
    """
    `Expression.unparsed` is the *declared* source text and is never rewritten by substitution -- so it is
    not where the substituted value lives, and reading it would silently give the wrong answer.
    `DefaultSerializationExpression.json_value` is.
    """

    def test_unparsed_still_holds_the_variable_name(self):
        context = check_concepts({**holder("integer", "Integer", "N"), **uses("H<3>")})
        expression = substituted_expression(context)
        assert expression.unparsed == "N", "unparsed is the declared text"
        assert expression.value.json_value == 3, "...and json_value is the substituted value"

    def test_unparsed_still_holds_the_templated_function_key(self):
        """The same is true of a type-level substitution: ``Add<T>`` stays written as ``Add<T>``."""
        context = check_concepts(
            {
                **ADD,
                **{
                    "H": {
                        "directParents": ["ValueDomain"],
                        "data": {
                            "templateContext": {"order": ["T"], "T": "Numeric"},
                            "instantiation": {
                                "type": "object",
                                "properties": {"p": {"type": "T", "default": {"Add<T>": {"arg1": 1, "arg2": 2}}}},
                                "required": ["p"],
                            },
                        },
                    }
                },
                **uses("H<Integer>"),
            }
        )
        expression = substituted_expression(context)
        assert isinstance(expression.value, FunctionEvaluation)
        assert expression.value.f_type.full_name == "Add<Integer>", "the type is substituted"
        assert expression.unparsed == {"Add<T>": {"arg1": 1, "arg2": 2}}, "the source text is not"

    def test_an_ordinary_default_serialized_value_also_keeps_its_value(self):
        """Not literal-specific: every defaultSerialization expression now records what it matched."""
        concepts = {
            "S": {
                "directParents": ["ValueDomain"],
                "data": {
                    "instantiation": {
                        "type": "object",
                        "properties": {"p": {"type": "String", "default": "hello"}},
                        "required": ["p"],
                    }
                },
            }
        }
        expression = default_expression(check_concepts(concepts), "S", "p")
        assert isinstance(expression.value, DefaultSerializationExpression)
        assert expression.value.json_value == "hello"
