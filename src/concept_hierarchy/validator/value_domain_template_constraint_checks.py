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

from concept_hierarchy.data.contexts.context import ConceptHierarchyContext
from concept_hierarchy.data.contexts.template_context import TemplateContext
from concept_hierarchy.data.parsers.template_argument_constraint_parser import (
    TemplateConstraintFormulaValidator,
    parse_constraint_string,
)
from concept_hierarchy.data.template_argument_constraints.constraint_formula import (
    ConstraintGroup,
    NonStructureConstraintFormula,
)
from concept_hierarchy.definitions.concept_definition_hidden_implementation import HiddenImplementationDefinition
from concept_hierarchy.errors import CHSemanticError


class ConstraintFormulaValidator(TemplateConstraintFormulaValidator):
    def __init__(self, context: ConceptHierarchyContext, t_arg_context: set[str]):
        self.ch_context = context
        # contains all template arguments available in the ValueDomain concept that defines the constraints
        self.t_arg_context: set[str] = t_arg_context

    def full_type_name(self, name: str) -> str:
        if self.is_concept(name):
            type_def_data = self.ch_context.ch.concepts[name]
            if isinstance(type_def_data, HiddenImplementationDefinition):
                return type_def_data.name_with_template_variables()
        return name

    def get_nr_template_arguments(self, concept_name: str) -> int:
        if self.is_concept(concept_name):
            type_def_data = self.ch_context.ch.concepts[concept_name]
            if isinstance(type_def_data, HiddenImplementationDefinition):
                return len(type_def_data.template_argument_order)
        return 0

    def is_concept(self, name: str) -> bool:
        return self.ch_context.ch.is_concept(name)

    def is_template_variable(self, name: str):
        return name in self.t_arg_context


def check_value_domain_template_constraint_formulae(context: ConceptHierarchyContext):
    # value domain name -> template argument name -> template argument constraint formula
    for vd_name in context.ch.topo_sort_concept_parents:
        if not context.ch.is_value_domain(vd_name):
            continue
        vd = context.ch.concepts[vd_name]
        vd_data = context.model.value_domains[vd_name]
        assert isinstance(vd, HiddenImplementationDefinition)
        # parse template argument constraints;
        # iterate in definition order because newer arguments have the older arguments as variables
        validator = ConstraintFormulaValidator(context, set(vd.template_argument_order))
        constraint = None
        constraints: list[NonStructureConstraintFormula] = []
        for t_arg in vd.template_argument_order:
            t_arg_constraint_formula = vd.template_argument_constraints[t_arg]
            if not vd.has_location_of(t_arg):
                # this is the default constraint, which should not raise an error!
                location_id = []
            else:
                location_id = vd.location_of(t_arg)
            t_arg_constraint = parse_constraint_string(t_arg_constraint_formula, validator, location_id)
            if not isinstance(t_arg_constraint, NonStructureConstraintFormula):
                raise CHSemanticError(
                    f"Found a non structure constraint formula {t_arg_constraint} when defining the constraint of "
                    f"{t_arg}",
                    location_id=location_id,
                )
            constraints.append(t_arg_constraint)

        if vd.is_templatable():
            constraint = ConstraintGroup(
                vd.location_of(HiddenImplementationDefinition.hidden_template_arguments), tuple(constraints)
            )
        vd_data.template_context = TemplateContext(
            vd.template_argument_order, vd.variadic_template_arguments, constraint
        )
