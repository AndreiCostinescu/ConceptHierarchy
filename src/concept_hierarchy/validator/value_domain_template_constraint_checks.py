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
from concept_hierarchy.data.parsers.template_argument_constraint_parser import parse_constraint_definition
from concept_hierarchy.data.type_template_variables.constraint_formula import (
    ConstraintGroup,
    NonStructureConstraintFormula,
)
from concept_hierarchy.definitions.concept_definition_hidden_implementation import HiddenImplementationDefinition
from concept_hierarchy.errors import CHSemanticError, PathPart
from concept_hierarchy.validator.validators.constraint_formula_validator import ConstraintFormulaValidator


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
        validator = ConstraintFormulaValidator(context)
        validator.update_existing_template_variables(set(vd.template_argument_order))
        constraint = None
        constraints: list[NonStructureConstraintFormula] = []
        for t_arg in vd.template_argument_order:
            t_arg_constraint_formula = vd.template_argument_constraints[t_arg]
            if not vd.has_location_of(t_arg):
                # this is the default constraint, which should not raise an error!
                location_id = []
            else:
                location_id = vd.location_of(t_arg)
            t_arg_constraint = parse_constraint_definition(
                t_arg_constraint_formula, validator, location_id, allow_unconstrained=False
            )
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
        if vd_data.template_context.is_empty_constraint:
            raise CHSemanticError(
                f"The template constraints of {vd_name} prevent any type-instantiation!",
                location_id=vd.location_of("templateArguments"),
                part=PathPart.VALUE,
            )
