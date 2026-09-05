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
from concept_hierarchy.data.expressions.expression import Expression
from concept_hierarchy.data.expressions.expression_utils import (
    ExpressionProvenance,
    FunctionArgumentAccessor,
    FunctionArgumentProvenance,
    ValueDomainArgumentProvenance,
)
from concept_hierarchy.data.expressions.instantiated_value import ParsedCustomValue, ParsedValue
from concept_hierarchy.data.expressions.subexpressions import ExpressionAttempt, IllFormedExpression
from concept_hierarchy.data.jsonschema import CHSchemaNode
from concept_hierarchy.data.parsers.expression_parser import parse_expression, parse_function_evaluation_expression
from concept_hierarchy.data.parsers.template_argument_constraint_parser import parse_constraint_definition
from concept_hierarchy.data.parsers.value_instantiation_parser import ValueInstantiationContext
from concept_hierarchy.data.type_template_variables.constraint_formula import (
    NonStructureConstraintFormula,
    TemplateConstraintFormula,
)
from concept_hierarchy.data.types.concept_hierarchy_types import TypeValue
from concept_hierarchy.data.utils import MISSING
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
        value: object,
        location_id: LocationId,
        template_context: TemplateContext,
        template_substitution: dict | None,
        expansion_depth: int,
    ) -> tuple[Expression | None, list[ConceptHierarchyError]]:
        expr = parse_expression(
            value,
            custom_type,
            provenance,
            FunctionArgumentAccessor.GET,
            template_context,
            self.context.expression_parser_validator,
            location_id,
            template_substitution=template_substitution,
            expansion_depth=expansion_depth,
        )
        if expr.is_valid:
            return expr, []
        assert isinstance(expr.value, IllFormedExpression)
        # Carry the explanation trace with the headline: this is a *nested* expression, so without the
        # trace the enclosing value's error would say only that some leaf did not parse.
        error = CHSemanticError(expr.value.reason, location_id=location_id, part=PathPart.VALUE)
        error.causes.extend(expr.value.explanation_causes(location_id))
        return None, [error]

    def parse_function_evaluation(
        self,
        function_name: str,
        arguments: object,
        schema_node: CHSchemaNode,
        location_id: LocationId,
        template_context: TemplateContext,
        template_substitution: dict | None,
        expansion_depth: int,
    ) -> tuple[ParsedValue | None, list[ConceptHierarchyError]]:
        attempts: list[ExpressionAttempt] = []
        # Where the arguments are, which is where everything this method reports belongs: the key error is
        # about that key, and the evaluation is the value under it.
        arguments_location_id = location_id + [function_name]
        expressions, function_type, is_function_subtype = parse_function_evaluation_expression(
            function_name,
            arguments,
            None,  # no expected result type!
            template_context,
            self.context.expression_parser_validator,
            location_id,
            # The whole point of the call: parse and check the argument expressions.
            recursively_parse=True,
            parse_template_expressions_without_type_checks=False,
            is_function_evaluation=True,
            # The caller's half of the protocol. The expression parser's version additionally asserts that
            # the expected type is a `TypeValue`, which is of no use to a caller that has none.
            ensure_expression_invariant=lambda produced, _expected: len(produced) == 1,
            attempts=attempts,
            template_substitution=template_substitution,
            expansion_depth=expansion_depth,
        )
        if function_type is None:
            return None, [
                CHSemanticError(
                    f"{function_name!r} is not a valid Function name!",
                    location_id=arguments_location_id,
                    part=PathPart.KEY,
                )
            ]
        if not is_function_subtype:
            # The one case the expression parser reports nothing for, because there it simply means "try
            # `Narrow` instead". Here there is no other alternative: only an evaluation may stand.
            return None, [
                CHSemanticError(
                    f"{function_name!r} names {function_type}, which is not a Function, so it can not be "
                    f"evaluated here!",
                    location_id=arguments_location_id,
                    part=PathPart.KEY,
                )
            ]
        assert len(expressions) == 1, expressions
        expression_value = expressions[0]
        if isinstance(expression_value, IllFormedExpression):
            # As in `parse_value_against_custom_type_expression`: without the trace the enclosing value's
            # error says only that something under it did not parse.
            error = CHSemanticError(expression_value.reason, location_id=arguments_location_id, part=PathPart.VALUE)
            error.causes.extend(expression_value.explanation_causes(arguments_location_id))
            return None, [error]
        expression = Expression(
            function_type,
            FunctionArgumentProvenance.ANY,
            FunctionArgumentAccessor.GET,
            {function_name: arguments},
            expression_value,
        )
        parsed = ParsedCustomValue(
            location_id=arguments_location_id,
            schema_node=schema_node,
            errors=[],
            custom_type=function_type,
            provenance=ValueDomainArgumentProvenance.ANY,
            used_default=False,
            default_expr=MISSING,
            expression=expression,
        )
        return parsed, []

    def resolve_default(self, schema_node: CHSchemaNode) -> Expression | None:
        return self.context.expression_parser_validator.resolve_default_site(schema_node)

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
