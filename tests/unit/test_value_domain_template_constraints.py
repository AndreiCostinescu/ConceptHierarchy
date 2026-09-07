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
Tests for the ValueDomain template-constraint formulae
(``checker.check_types`` step 1 -> ``check_value_domain_template_constraint_formulae``).

Each templated ValueDomain declares a constraint per template argument. This step parses those formulae
into a :class:`TemplateContext` on the ValueDomain's model data -- the context every later check consults
to decide what a template argument is allowed to be.

The *syntax* of the formulae is covered exhaustively by ``test_template_argument_constraint_parsing``;
what is checked here is the semantic step: which concepts a formula may name, what the resulting context
looks like, and which formulae are rejected.
"""

from __future__ import annotations

import pytest

from concept_hierarchy.errors import CHSemanticError
from tests.ch_support import check_concepts


def value_domain(template_context: dict | list, name: str = "Box") -> dict:
    return {name: {"directParents": ["ValueDomain"], "data": {"templateContext": template_context}}}


def template_context_of(context, name: str = "Box"):
    return context.model.value_domains[name].template_context


# --------------------------------------------------------------------------------------------------
# What the step produces
# --------------------------------------------------------------------------------------------------


class TestResultingTemplateContext:
    def test_a_non_templated_value_domain_has_an_empty_context(self):
        context = check_concepts({"Plain": {"directParents": ["ValueDomain"], "data": {}}})
        assert template_context_of(context, "Plain").empty is True

    def test_the_declared_variables_are_recorded_in_order(self):
        context = check_concepts(value_domain({"order": ["A", "B"], "A": "Number", "B": "Number"}, "Pair"))
        assert template_context_of(context, "Pair").variables == ("A", "B")

    def test_the_constraint_is_parsed_onto_the_context(self):
        context = check_concepts(value_domain({"order": ["T"], "T": "Number"}))
        template_context = template_context_of(context)
        assert template_context.empty is False
        assert "Number" in repr(template_context.constraint)

    def test_the_shorthand_list_form_declares_unconstrained_variables(self):
        """``"templateContext": ["T"]`` declares ``T`` with the default constraint."""
        context = check_concepts(value_domain(["T"]))
        assert template_context_of(context).variables == ("T",)

    def test_a_literal_template_argument_is_marked_as_literal(self):
        context = check_concepts(value_domain({"order": ["N"], "N": 3}, "Vector"))
        assert template_context_of(context, "Vector").is_literal_template_variable("N") is True

    def test_a_type_template_argument_is_not_marked_as_literal(self):
        context = check_concepts(value_domain({"order": ["T"], "T": "Number"}))
        assert template_context_of(context).is_literal_template_variable("T") is False


# --------------------------------------------------------------------------------------------------
# Which formulae are accepted
# --------------------------------------------------------------------------------------------------


class TestAcceptedFormulae:
    @pytest.mark.parametrize(
        "formula",
        ["Number", "And(Number, Not(Integer))", "Or(Number, String)", "Not(String)", "^Integer", "Number*"],
        ids=["concept", "and-not", "or", "not", "ascendants", "non-abstract-descendants"],
    )
    def test_formula_is_accepted(self, formula):
        check_concepts(value_domain({"order": ["T"], "T": formula}))

    def test_a_constraint_may_name_an_earlier_template_argument(self):
        check_concepts(value_domain({"order": ["A", "B"], "A": "Number", "B": "A"}, "Pair"))

    def test_a_constraint_may_currently_also_name_a_later_template_argument(self):
        """
        Characterisation test. ``check_value_domain_template_constraint_formulae`` registers *all* of the
        ValueDomain's template arguments before parsing any formula, so a forward reference resolves --
        even though the loop's own comment says arguments are iterated in definition order "because newer
        arguments have the older arguments as variables". If forward references should be rejected, this
        is the test that will catch the change.
        """
        check_concepts(value_domain({"order": ["A", "B"], "A": "B", "B": "Number"}, "Pair"))


# --------------------------------------------------------------------------------------------------
# Which formulae are rejected
# --------------------------------------------------------------------------------------------------


class TestRejectedFormulae:
    def test_a_formula_naming_an_unknown_concept(self):
        with pytest.raises(CHSemanticError, match="is not a concept and not a template variable"):
            check_concepts(value_domain({"order": ["T"], "T": "NoSuchConcept"}))

    def test_a_formula_naming_a_template_argument_of_another_value_domain(self):
        """``Other``'s ``U`` is not in scope while parsing ``Box``'s constraints."""
        concepts = {
            **value_domain({"order": ["U"], "U": "Number"}, "Other"),
            **value_domain({"order": ["T"], "T": "U"}),
        }
        with pytest.raises(CHSemanticError, match="is not a concept and not a template variable"):
            check_concepts(concepts)


# --------------------------------------------------------------------------------------------------
# Known limitation
# --------------------------------------------------------------------------------------------------


class TestKnownLimitations:
    @pytest.mark.xfail(
        reason="TemplateConstraintHierarchyOperator.is_empty is hardcoded False, so an unsatisfiable "
        "constraint is never recognised and the 'prevent any type-instantiation' error can not fire",
        strict=False,
    )
    def test_an_unsatisfiable_constraint_is_rejected(self):
        """``And(Number, String)`` is empty -- ``Number`` and ``String`` are disjoint."""
        with pytest.raises(CHSemanticError, match="prevent any type-instantiation"):
            check_concepts(value_domain({"order": ["T"], "T": "And(Number, String)"}))
