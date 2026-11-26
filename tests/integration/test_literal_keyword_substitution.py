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
Integration tests: JSON Schema keywords written as literal template variables (``{"minItems": "N"}``).

A Draft-07 validator cannot be handed the name ``"N"`` where it expects a number, so `jsonschema_parser`
**pops** such a keyword out of the schema and parks it in the matching ``*_def`` field of the
`CHSchemaNode`. Until something puts it back, the keyword constrains nothing at all -- ``H<2>`` accepted an
empty array exactly as ``H<0>`` did.

`substitute_schema` is where it can be put back, because that is where the application is known: the
literal is substituted into `shallow_canonical`, which is the dict `parse_value` hands to the Draft-07
validator, and the ``*_def`` field is cleared.

Two things follow, and both are tested here: the keyword now *binds*, and it binds **per application** --
the same declared schema constrains differently under ``H<2>`` and under ``H<0>``, which is the whole
reason it could not be resolved where it was written.

A keyword the substitution does not cover stays parked. That is not an oversight: a partial substitution
has not decided it, and the field is what records that it is still waiting.
"""

from __future__ import annotations

import pytest

from concept_hierarchy.errors import ConceptHierarchyError
from tests.integration.test_expression_parsing import check_concepts
from tests.integration.test_schema_substitution import obj, vd


def holder(sort: str, value_schema: dict, variable: str = "N") -> dict:
    """``H<N: Literal:sort>`` whose single property ``v`` carries the keyword under test."""
    return {
        "H": {
            "directParents": ["ValueDomain"],
            "data": {
                "templateContext": {"order": [variable], variable: f"Literal:{sort}"},
                "instantiation": {"type": "object", "properties": {"v": value_schema}, "required": ["v"]},
            },
        }
    }


def uses(application: str, value: object) -> dict:
    """A ground ValueDomain whose only property defaults to ``value`` at type ``application``."""
    return vd("Uses", obj({"u": {"type": application, "default": value}}))


def accepts(sort: str, value_schema: dict, application: str, value: object) -> bool:
    try:
        check_concepts({**holder(sort, value_schema), **uses(application, value)})
        return True
    except ConceptHierarchyError:
        return False


# --------------------------------------------------------------------------------------------------
# Every keyword that can be written as a literal template variable
# --------------------------------------------------------------------------------------------------

ARRAY = {"type": "array", "items": {"type": "integer"}}

KEYWORD_CASES = [
    # keyword,            sort,      schema,                      application, rejected value, accepted value
    ("minItems", "integer", {**ARRAY, "minItems": "N"}, "H<2>", {"v": [1]}, {"v": [1, 2]}),
    ("maxItems", "integer", {**ARRAY, "maxItems": "N"}, "H<1>", {"v": [1, 2]}, {"v": [1]}),
    ("minLength", "integer", {"type": "string", "minLength": "N"}, "H<3>", {"v": "ab"}, {"v": "abc"}),
    ("maxLength", "integer", {"type": "string", "maxLength": "N"}, "H<2>", {"v": "abc"}, {"v": "ab"}),
    (
        "minProperties",
        "integer",
        {"type": "object", "minProperties": "N"},
        "H<2>",
        {"v": {"a": 1}},
        {"v": {"a": 1, "b": 2}},
    ),
    (
        "maxProperties",
        "integer",
        {"type": "object", "maxProperties": "N"},
        "H<1>",
        {"v": {"a": 1, "b": 2}},
        {"v": {"a": 1}},
    ),
    ("minimum", "number", {"type": "number", "minimum": "N"}, "H<5>", {"v": 4}, {"v": 5}),
    ("maximum", "number", {"type": "number", "maximum": "N"}, "H<5>", {"v": 6}, {"v": 5}),
    ("exclusiveMinimum", "number", {"type": "number", "exclusiveMinimum": "N"}, "H<5>", {"v": 5}, {"v": 6}),
    ("exclusiveMaximum", "number", {"type": "number", "exclusiveMaximum": "N"}, "H<5>", {"v": 5}, {"v": 4}),
    ("multipleOf", "number", {"type": "number", "multipleOf": "N"}, "H<3>", {"v": 4}, {"v": 6}),
]


class TestEveryLiteralKeywordBinds:
    @pytest.mark.parametrize(
        "keyword,sort,schema,application,rejected,accepted",
        KEYWORD_CASES,
        ids=[case[0] for case in KEYWORD_CASES],
    )
    def test_the_keyword_rejects_and_accepts(self, keyword, sort, schema, application, rejected, accepted):
        """
        Both halves matter. A keyword that is never put back accepts everything, so the *rejection* is what
        shows it bound; a keyword put back wrongly rejects everything, so the *acceptance* is what shows it
        bound to the right value.
        """
        assert not accepts(sort, schema, application, rejected), f'"{keyword}" did not constrain {rejected}'
        assert accepts(sort, schema, application, accepted), f'"{keyword}" wrongly rejected {accepted}'


class TestTheApplicationIsWhatDecides:
    def test_the_same_declared_schema_constrains_differently_per_application(self):
        """
        One declaration, two applications, opposite verdicts on the same value. Nothing about the schema
        text distinguishes them -- which is exactly why this cannot be resolved where it is written.
        """
        schema = {**ARRAY, "minItems": "N"}
        assert not accepts("integer", schema, "H<2>", {"v": [1]})
        assert accepts("integer", schema, "H<1>", {"v": [1]})

    def test_a_zero_bound_admits_everything(self):
        assert accepts("integer", {**ARRAY, "minItems": "N"}, "H<0>", {"v": []})

    def test_a_bound_that_no_value_can_meet_rejects(self):
        assert not accepts("integer", {**ARRAY, "minItems": "N"}, "H<2>", {"v": []})


class TestNestedPositions:
    def test_inside_an_items_subschema(self):
        schema = {"type": "array", "items": {"type": "string", "minLength": "N"}}
        assert not accepts("integer", schema, "H<3>", {"v": ["ab"]})
        assert accepts("integer", schema, "H<3>", {"v": ["abc"]})

    def test_inside_an_any_of_branch(self):
        schema = {"anyOf": [{"type": "string"}, {"type": "array", "minItems": "N"}]}
        assert not accepts("integer", schema, "H<2>", {"v": [1]})
        assert accepts("integer", schema, "H<2>", {"v": [1, 2]})
        assert accepts("integer", schema, "H<2>", {"v": "s"}), "the other branch is untouched"

    def test_two_keywords_on_one_node(self):
        schema = {**ARRAY, "minItems": "N", "maxItems": "N"}
        assert not accepts("integer", schema, "H<2>", {"v": [1]})
        assert not accepts("integer", schema, "H<2>", {"v": [1, 2, 3]})
        assert accepts("integer", schema, "H<2>", {"v": [1, 2]})

    def test_a_literal_and_a_written_bound_on_one_node(self):
        """The substituted keyword joins the ones that were always there, rather than replacing them."""
        schema = {**ARRAY, "minItems": "N", "maxItems": 3}
        assert not accepts("integer", schema, "H<2>", {"v": [1]})
        assert not accepts("integer", schema, "H<2>", {"v": [1, 2, 3, 4]})
        assert accepts("integer", schema, "H<2>", {"v": [1, 2]})


class TestTheDeclarationIsNotWrittenTo:
    def test_two_applications_of_one_declaration_do_not_interfere(self):
        """
        Substitution copies before it writes. If it did not, the first application built would leave its
        bound in the shared declaration and the second would inherit it -- so this checks the strict bound
        under `H<3>` does not leak into `H<1>`, whichever order they are built in.
        """
        two = vd(
            "Uses",
            obj(
                {
                    "strict": {"type": "H<3>", "default": {"v": [1, 2, 3]}},
                    "loose": {"type": "H<1>", "default": {"v": [1]}},
                }
            ),
        )
        check_concepts({**holder("integer", {**ARRAY, "minItems": "N"}), **two})


class TestTheSubstitutedSchemaNoLongerWaitsOnAnything:
    """
    The ``*_def`` field is not just a place to read the keyword from -- it is the record that the keyword is
    still *undecided*, which is what `CHSchemaNode.is_template_dependent` reports. Leaving it set after
    substitution would keep a fully ground schema looking template dependent for ever.
    """

    def _substituted(self, application: str):
        context = check_concepts(
            {**holder("integer", {**ARRAY, "minItems": "N"}), **uses(application, {"v": [1, 2, 3]})}
        )
        schemas = {
            name: schema
            for (name, _group), schema in context.expression_parser_validator.resolved_instantiation_schemas.items()
        }
        assert application in schemas, f"{application} was never built; have {sorted(schemas)}"
        return schemas[application]

    def test_the_keyword_field_is_cleared(self):
        schema = self._substituted("H<2>")
        parked = [node.min_items_def for node in schema.walk() if node.min_items_def is not None]
        assert parked == [], f"still waiting on {parked}"

    def test_the_value_is_where_the_validator_reads_it(self):
        schema = self._substituted("H<2>")
        bounds = [
            node.shallow_canonical["minItems"]
            for node in schema.walk()
            if isinstance(node.shallow_canonical, dict) and "minItems" in node.shallow_canonical
        ]
        assert bounds == [2], f"expected the substituted bound in shallow_canonical, found {bounds}"

    def test_the_substituted_schema_is_no_longer_template_dependent(self):
        """
        `is_template_dependent` could not be called at all before -- it read ``custom_type`` unguarded and
        raised on every structural node -- so nothing had ever asked a schema this question.
        """
        assert not self._substituted("H<2>").is_template_dependent

    def test_the_declaration_still_is(self):
        """The control: what was substituted away is genuinely there before substitution."""
        context = check_concepts({**holder("integer", {**ARRAY, "minItems": "N"}), **uses("H<2>", {"v": [1, 2, 3]})})
        declared = [schema for _constraint, schema in context.model.value_domains["H"].instantiation]
        assert declared and all(schema.is_template_dependent for schema in declared)


# ==================================================================================================
# Forwarding the variable: the keyword is bound by an application two concepts away
# ==================================================================================================

CELL = {
    "Cell": {
        "directParents": ["ValueDomain"],
        "data": {
            "templateContext": {"order": ["M"], "M": "Literal:integer"},
            "instantiation": {
                "type": "object",
                "properties": {"v": {"type": "array", "items": {"type": "integer"}, "minItems": "M"}},
                "required": ["v"],
            },
        },
    }
}
"""A ValueDomain whose bound is its own literal template variable."""


class TestAForwardedLiteralVariable:
    """
    ``Cell<N>`` written inside ``Holder<N>``: the keyword is not bound by `Cell`'s own application but by
    whatever binds `Holder`'s ``N``, one level out.

    This shape used to assert in `template_argument_constraints_validator` -- `has_template_variable` is a
    membership test over bare names and was handed the *qualified* variable (``Holder:N``) -- so nothing
    could reach it, and neither this nor the two classes below could be written at all.
    """

    HOLDER = {
        "Holder": {
            "directParents": ["ValueDomain"],
            "data": {
                "templateContext": {"order": ["N"], "N": "Literal:integer"},
                "instantiation": {
                    "type": "object",
                    "properties": {"c": {"type": "Cell<N>", "default": {"v": [1]}}},
                    "required": ["c"],
                },
            },
        }
    }

    def test_the_bound_is_met(self):
        check_concepts({**CELL, **self.HOLDER, **uses("Holder<1>", {})})

    def test_the_bound_is_violated(self):
        """The default ``{"v": [1]}`` cannot satisfy ``minItems: 2``, and only `Holder<2>` says so."""
        with pytest.raises(ConceptHierarchyError):
            check_concepts({**CELL, **self.HOLDER, **uses("Holder<2>", {})})

    def test_the_declared_default_stays_undecided(self):
        """
        At the declaration ``N`` is bound to nothing, so the default cannot be decided there -- and must not
        be *reported* as decided either, which is what would let a later check skip it.
        """
        context = check_concepts({**CELL, **self.HOLDER, **uses("Holder<1>", {})})
        declared = [
            node.parsed_default_expr
            for _constraint, schema in context.model.value_domains["Holder"].instantiation
            for node in schema.walk()
            if node.has_default and node.parsed_default_expr is not None
        ]
        assert declared, "the default site was not found"
        assert all(expression.is_value_template_dependent for expression in declared)


class TestAForwardedLiteralVariableInAFunctionArgument:
    """
    The same forwarding, at a Function's default argument -- the case the whole `_recheck_decided_default`
    caveat was about. ``a: Cell<M>`` defaults to ``{"v": [1]}``, which no `Cell<2>` admits.

    Nothing in the *expression* records that: it is a value with no custom-type leaves, so before
    `undecided_literal_keywords_for` it reported as decided and a subtype check would have waved it through.
    """

    F = {
        "F": {
            "directParents": ["FunctionReturning"],
            "data": {
                "templateContext": {
                    "order": ["M"],
                    "M": "Literal:integer",
                    "substitution": {"FunctionReturning:T": "Integer"},
                },
                "interface": {
                    "a": ["Cell<M>"],
                    "b": ["Integer"],
                    "res": "Integer",
                    "_defaultArgumentValues": {"a": {"v": [1]}},
                },
            },
        }
    }

    def _site(self, application: str) -> dict:
        return vd("Site", obj({"p": {"type": "Integer", "default": {application: {"b": 1}}}}))

    def test_the_default_holds_for_the_application(self):
        check_concepts({**CELL, **self.F, **self._site("F<1>")})

    def test_the_default_can_not_hold_for_the_application(self):
        with pytest.raises(ConceptHierarchyError):
            check_concepts({**CELL, **self.F, **self._site("F<2>")})

    def test_supplying_the_argument_leaves_the_default_alone(self):
        """As everywhere else: a default the call site does not need is never decided."""
        site = vd("Site", obj({"p": {"type": "Integer", "default": {"F<2>": {"a": {"v": [1, 2]}, "b": 1}}}}))
        check_concepts({**CELL, **self.F, **site})
