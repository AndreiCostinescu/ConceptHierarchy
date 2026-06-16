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

from frozendict import frozendict

from concept_hierarchy.data.contexts.context import ConceptHierarchyContext
from concept_hierarchy.data.parsers.template_argument_constraint_parser import (
    TemplateConstraintFormulaValidator,
    parse_constraint_string,
)
from concept_hierarchy.data.template_argument_constraints.constraint_formula import TemplateConstraintFormula
from concept_hierarchy.definitions.concept_definition_hidden_implementation import HiddenImplementationDefinition
from concept_hierarchy.errors import CHSemanticError, LocationId


class ConstraintFormulaValidator(TemplateConstraintFormulaValidator):
    def __init__(self, context: ConceptHierarchyContext):
        self.ch_context = context
        # built incrementally as template arguments are processed
        self.t_arg_context: dict[str, TemplateConstraintFormula] = {}

    def validate(
        self,
        ch_type_name: str,
        template_constraint_arguments: tuple[TemplateConstraintFormula, ...] | None,
        location_id: LocationId,
    ):
        has_specification_of_template_constraints = template_constraint_arguments is not None
        # don't allow constraints like "T<ValueDomain>" where T is a template variable!
        if ch_type_name in self.t_arg_context and has_specification_of_template_constraints:
            raise CHSemanticError(
                "Can not define a constraint literal value that is a template variable ({0}) and also "
                "specify constraints on template arguments: {0}<{1}>".format(
                    ch_type_name, ", ".join(str(t_constraint) for t_constraint in template_constraint_arguments)
                ),
                location_id=location_id,
            )
        # Check that either no template_constraint_formulae are specified
        #  or the same number of formulae as the literal has template arguments!
        elif ch_type_name not in self.t_arg_context and has_specification_of_template_constraints:
            assert self.ch_context.ch.is_concept(ch_type_name)
            if not self.ch_context.ch.is_value_domain(ch_type_name):
                raise CHSemanticError(
                    f"Defined template argument constraints {template_constraint_arguments!r} on a non-ValueDomain "
                    f"concept: {ch_type_name!r}!",
                    location_id=location_id,
                )
            v = self.ch_context.ch.concepts[ch_type_name]
            assert isinstance(v, HiddenImplementationDefinition)
            if not v.is_templatable() and has_specification_of_template_constraints:
                t_arg_constraints_str = ", ".join(str(t_constraint) for t_constraint in template_constraint_arguments)
                literal_str = ch_type_name + (("<" + t_arg_constraints_str + ">") if t_arg_constraints_str else "")
                raise CHSemanticError(
                    f"Can not define a constraint literal value {ch_type_name} that is a non-template "
                    f"ValueDomain with template arguments: {literal_str}!",
                    location_id=location_id,
                )
            if has_specification_of_template_constraints and len(template_constraint_arguments) != len(
                v.template_argument_order
            ):
                t_arg_constraints_str = ", ".join(str(t_constraint) for t_constraint in template_constraint_arguments)
                raise CHSemanticError(
                    f"The number {len(template_constraint_arguments)} of template argument constraints "
                    f"{t_arg_constraints_str} on literal {ch_type_name} does not match the number of template arguments"
                    f" in the ValueDomain's definition: {v.name_with_template_variables()}",
                    location_id=location_id,
                )

    def should_be_ch_type_or_template_variable(
        self, ch_type_name: str, location_id: LocationId
    ) -> TemplateConstraintFormula | None:
        if self.ch_context.ch.is_concept(ch_type_name):
            return None
        if ch_type_name in self.t_arg_context:
            return self.t_arg_context[ch_type_name]
        raise CHSemanticError(
            f"{ch_type_name}, that is part of the template constraint formula, is neither a concept nor a "
            f"template argument variable (available variables: {[x for x in self.t_arg_context]!r})",
            location_id=location_id,
        )


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
        constraints: dict[str, TemplateConstraintFormula] = {}
        for t_arg in vd.template_argument_order:
            t_arg_constraint_formula = vd.template_argument_constraints[t_arg]
            if not vd.has_location_of(t_arg):
                # this is the default constraint, which should not raise an error!
                location_id = []
            else:
                location_id = vd.location_of(t_arg)
            t_arg_constraint = parse_constraint_string(t_arg_constraint_formula, validator, location_id)
            constraints[t_arg] = t_arg_constraint
            validator.t_arg_context[t_arg] = t_arg_constraint

        vd_data.template_argument_constraints = frozendict(constraints)
