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
from concept_hierarchy.data.expressions.expression import Expression
from concept_hierarchy.data.expressions.expression_utils import (
    ExpressionProvenance,
    FunctionArgumentAccessor,
    FunctionArgumentProvenance,
)
from concept_hierarchy.data.expressions.subexpressions import IllFormedExpression
from concept_hierarchy.data.parsers.expression_parser import parse_expression
from concept_hierarchy.data.parsers.template_argument_constraint_parser import parse_constraint_definition
from concept_hierarchy.data.parsers.value_instantiation_parser import ValueInstantiationContext
from concept_hierarchy.data.type_template_variables.constraint_formula import (
    NonStructureConstraintFormula,
    TemplateConstraintFormula,
)
from concept_hierarchy.data.types.concept_hierarchy_types import TypeValue
from concept_hierarchy.data.validators.template_argument_constraints_validator import (
    TemplateContextDeterminator,
    validate_template_argument_value_against_constraint,
)
from concept_hierarchy.data.validators.type_validator import parse_convert_type_in_template_context
from concept_hierarchy.errors import CHSemanticError, ConceptHierarchyError, LocationId, PathPart


class ValueValidator(ValueInstantiationContext):
    def __init__(self, context: ConceptHierarchyContext):
        self.context = context

    def parse_value_against_custom_type_expression(
        self,
        custom_type: TypeValue,
        provenance: ExpressionProvenance,
        default_expr: object,
        value: object,
        location_id: LocationId,
    ) -> tuple[Expression | None, list[ConceptHierarchyError]]:
        expr = parse_expression(
            value,
            custom_type,
            FunctionArgumentProvenance.ADDR
            if provenance == ExpressionProvenance.ADDR
            else FunctionArgumentProvenance.ANY,
            FunctionArgumentAccessor.GET,
            self.context.template_context,
            self.context.expression_parser_validator,
            location_id,
        )
        if expr.is_valid:
            return expr, []
        assert isinstance(expr.value, IllFormedExpression)
        return None, [CHSemanticError(expr.value.reason, location_id=location_id, part=PathPart.VALUE)]

    def is_concept(self, concept_candidate: str) -> bool:
        return self.context.ch.is_concept(concept_candidate)

    def is_type(self, type_candidate: str, location_id: LocationId) -> bool:
        try:
            parse_convert_type_in_template_context(type_candidate, self.context.type_validator, location_id)
            return True
        except ConceptHierarchyError:
            return False

    def parse_constraint(self, constraint: str, location_id: LocationId) -> TemplateConstraintFormula:
        return parse_constraint_definition(
            constraint, self.context.template_constraint_formula_validator, location_id, allow_unconstrained=False
        )

    def validate_string_constraint(
        self, constraint: NonStructureConstraintFormula, value: str, location_id: LocationId
    ) -> bool:
        template_context = TemplateContextDeterminator(self.context.template_context)
        value = parse_convert_type_in_template_context(value, self.context.type_validator, location_id)
        errors = validate_template_argument_value_against_constraint(
            constraint, value, template_context, self.context.type_application_constraints_validator, {}, location_id
        )
        return errors == []
