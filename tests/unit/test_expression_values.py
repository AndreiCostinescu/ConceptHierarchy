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
Unit tests for the properties of :class:`ExpressionValue`\\ s, built directly rather than parsed.

Building the parse tree by hand keeps these tests independent of what the parser currently produces, so
they pin the *contract* of ``is_template_dependent`` / ``is_fully_parsed`` / ``is_valid`` -- including the
cases the parser can not reach today.

``InstExpression.is_template_dependent`` in particular must report the value's **own content** as well as
its subexpressions: a custom-type leaf parsed against ``Box<T>`` makes the value template dependent even
when no expression was parsed at that leaf.
"""

from __future__ import annotations

import pytest

from concept_hierarchy.data.expressions.expression import Expression
from concept_hierarchy.data.expressions.expression_utils import (
    FunctionArgumentAccessor,
    FunctionArgumentProvenance,
    ValueDomainArgumentProvenance,
)
from concept_hierarchy.data.expressions.instantiated_value import ParsedCustomValue, ParsedStructural
from concept_hierarchy.data.expressions.subexpressions import (
    IllFormedExpression,
    InstExpression,
    NarrowExpression,
    Variable,
    VerifiedTemplateDependentExpression,
)
from concept_hierarchy.data.types.concept_hierarchy_types import (
    InstantiatedType,
    NonVariadicTemplateVariable,
    TemplateDependentType,
    TypeValue,
)

INTEGER = InstantiatedType("Integer", ())
T = NonVariadicTemplateVariable("T", "Box")
BOX_INTEGER = InstantiatedType("Box", (INTEGER,))
BOX_T = TemplateDependentType("Box", (T,))


# --------------------------------------------------------------------------------------------------
# Builders for a parse tree
# --------------------------------------------------------------------------------------------------


def expression(value, expression_type: TypeValue = INTEGER, unparsed: object = None) -> Expression:
    """Wrap an :class:`ExpressionValue` in the :class:`Expression` envelope the parser produces."""
    return Expression(
        expression_type,
        FunctionArgumentProvenance.ANY,
        FunctionArgumentAccessor.GET,
        unparsed,
        value,
    )


def custom_leaf(custom_type: TypeValue, inner: Expression | None = None) -> ParsedCustomValue:
    """A custom-type leaf of a parsed value, optionally carrying a parsed sub-expression."""
    return ParsedCustomValue(
        location_id=[],
        schema_node=None,
        errors=[],
        custom_type=custom_type,
        provenance=ValueDomainArgumentProvenance.ANY,
        used_default=False,
        default_expr=None,
        expression=inner,
    )


def structural(value: object = None, properties: dict[str, ParsedCustomValue] | None = None) -> ParsedStructural:
    """A structural node of a parsed value, holding the raw JSON value and any parsed properties."""
    return ParsedStructural(
        location_id=[],
        schema_node=None,
        errors=[],
        value=value,
        properties_parsed=properties or {},
    )


# --------------------------------------------------------------------------------------------------
# InstExpression.is_template_dependent
# --------------------------------------------------------------------------------------------------


class TestInstExpressionIsTemplateDependent:
    def test_a_plain_literal_does_not_depend_on_templates(self):
        """A literal with no custom-type leaves and no subexpressions depends on nothing."""
        assert InstExpression(structural(10), INTEGER).is_template_dependent is False

    def test_a_default_serialization_value_does_not_depend_on_templates(self):
        """``InstExpression(None, ...)`` is how a default-serialization expression is built."""
        assert InstExpression(None, INTEGER).is_template_dependent is False

    def test_a_ground_custom_leaf_does_not_depend_on_templates(self):
        value = structural(properties={"x": custom_leaf(BOX_INTEGER)})
        assert InstExpression(value, INTEGER).is_template_dependent is False

    def test_a_template_dependent_custom_leaf_does(self):
        value = structural(properties={"x": custom_leaf(BOX_T)})
        assert InstExpression(value, INTEGER).is_template_dependent is True

    def test_a_bare_template_variable_leaf_does(self):
        value = structural(properties={"x": custom_leaf(T)})
        assert InstExpression(value, INTEGER).is_template_dependent is True

    def test_a_template_dependent_subexpression_does(self):
        inner = expression(VerifiedTemplateDependentExpression([]), T)
        value = structural(properties={"x": custom_leaf(INTEGER, inner)})
        assert InstExpression(value, INTEGER).is_template_dependent is True

    def test_a_ground_subexpression_does_not(self):
        inner = expression(Variable("v", INTEGER), INTEGER)
        value = structural(properties={"x": custom_leaf(INTEGER, inner)})
        assert InstExpression(value, INTEGER).is_template_dependent is False

    def test_one_template_dependent_part_is_enough(self):
        """Aggregation is `any`, not `all`: a single dependent part makes the whole value dependent."""
        value = structural(
            properties={
                "ground": custom_leaf(INTEGER, expression(Variable("v", INTEGER), INTEGER)),
                "dependent": custom_leaf(BOX_T),
            }
        )
        assert InstExpression(value, INTEGER).is_template_dependent is True

    def test_narrow_expression_inherits_the_behaviour(self):
        assert NarrowExpression(structural(10), INTEGER).is_template_dependent is False
        assert NarrowExpression(structural(properties={"x": custom_leaf(BOX_T)}), INTEGER).is_template_dependent is True


# --------------------------------------------------------------------------------------------------
# InstExpression.is_fully_parsed
# --------------------------------------------------------------------------------------------------


class TestInstExpressionIsFullyParsed:
    def test_a_default_serialization_value_has_nothing_left_to_parse(self):
        """Regression: this used to raise ``AttributeError`` because ``None`` has no ``walk``."""
        assert InstExpression(None, INTEGER).is_fully_parsed is True

    def test_a_literal_is_fully_parsed(self):
        assert InstExpression(structural(10), INTEGER).is_fully_parsed is True

    def test_a_leaf_without_an_expression_is_fully_parsed(self):
        value = structural(properties={"x": custom_leaf(INTEGER)})
        assert InstExpression(value, INTEGER).is_fully_parsed is True

    def test_an_unparsed_subexpression_makes_it_not_fully_parsed(self):
        inner = expression(VerifiedTemplateDependentExpression([]), T)
        value = structural(properties={"x": custom_leaf(INTEGER, inner)})
        assert InstExpression(value, INTEGER).is_fully_parsed is False


# --------------------------------------------------------------------------------------------------
# is_valid
# --------------------------------------------------------------------------------------------------


class TestIsValid:
    def test_is_a_property_on_every_expression_value(self):
        """Regression: a missing ``@property`` returned a truthy bound method, so `not expr.is_valid`
        could never fire and an ill-formed expression passed validation silently."""
        values = [
            InstExpression(structural(10), INTEGER),
            NarrowExpression(structural(10), INTEGER),
            Variable("v", INTEGER),
            IllFormedExpression("nope"),
            VerifiedTemplateDependentExpression([]),
            VerifiedTemplateDependentExpression([InstExpression(structural(10), INTEGER)]),
        ]
        for value in values:
            assert isinstance(value.is_valid, bool), f"{type(value).__name__}.is_valid is not a bool"
            assert isinstance(expression(value).is_valid, bool)

    def test_an_ill_formed_expression_is_invalid(self):
        assert IllFormedExpression("nope").is_valid is False
        assert expression(IllFormedExpression("nope")).is_valid is False

    @pytest.mark.parametrize(
        ("possible_expressions", "expected"),
        [
            ([], False),
            ([InstExpression(structural(10), INTEGER)], True),
            ([IllFormedExpression("nope")], False),
            ([InstExpression(structural(10), INTEGER), IllFormedExpression("nope")], False),
        ],
        ids=["none-possible", "one-good", "one-ill-formed", "one-of-each"],
    )
    def test_a_verified_template_dependent_expression_needs_every_reading_to_be_well_formed(
        self, possible_expressions, expected
    ):
        assert VerifiedTemplateDependentExpression(possible_expressions).is_valid is expected
