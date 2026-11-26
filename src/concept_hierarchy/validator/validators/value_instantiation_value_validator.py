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
from contextlib import contextmanager
from typing import Iterator

from concept_hierarchy.data.contexts.context import ConceptHierarchyContext, TemplateContext
from concept_hierarchy.data.contexts.variable_context import VariableContext, VariableStackFrame
from concept_hierarchy.data.expressions.expression import Expression
from concept_hierarchy.data.expressions.expression_utils import (
    ExpressionProvenance,
    FunctionArgumentAccessor,
    FunctionArgumentProvenance,
    FunctionInterpretation,
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
from concept_hierarchy.data.type_template_variables.template_substitution import substitute
from concept_hierarchy.data.types.concept_hierarchy_types import (
    TYPE_VALUE_IS_INSTANCE_CHECK,
    ConceptHierarchyTemplateArgument,
    Instantiated,
    InstantiatedType,
    TypeValue,
)
from concept_hierarchy.data.utils import MISSING
from concept_hierarchy.data.validators.template_argument_constraints_validator import (
    TemplateContextDeterminator,
    validate_template_argument_value_against_constraint,
)
from concept_hierarchy.data.validators.type_validator import parse_convert_type_in_template_context
from concept_hierarchy.definitions.concept_definition_domain_concept import ForPropertyOrFunction
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
        template_substitution: dict | None,
        expansion_depth: int,
        function_interpretation: FunctionInterpretation,
    ) -> tuple[Expression | None, list[ConceptHierarchyError]]:
        expr = parse_expression(
            value,
            custom_type,
            provenance,
            FunctionArgumentAccessor.GET,
            self.context.expression_parser_validator,
            location_id,
            template_substitution=template_substitution,
            expansion_depth=expansion_depth,
            function_interpretation=function_interpretation,
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
        template_substitution: dict | None,
        expansion_depth: int,
    ) -> tuple[ParsedValue | None, list[ConceptHierarchyError]]:
        attempts: list[ExpressionAttempt] = []
        # Where the arguments are, which is where everything this method reports belongs: the key error is
        # about that key, and the evaluation is the value under it.
        arguments_location_id = location_id + [function_name]
        expressions, function_type, is_function_subtype, _ = parse_function_evaluation_expression(
            function_name,
            arguments,
            None,  # no expected result type!
            self.context.expression_parser_validator,
            location_id,
            # The whole point of the call: parse and check the argument expressions.
            recursively_parse=True,
            parse_template_expressions_without_type_checks=False,
            # No enclosing site has decided anything about this value:
            # the interpretation, if any, is the marker on `function_name`, which is handled by the parser.
            force_function_evaluation_interpretation=None,
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
        if len(expressions) == 0:
            # This is the case where the type is not a Function; normally, the `Narrow` expression check would follow.
            # But here, there is no other alternative: only an evaluation may stand.
            return None, [
                CHSemanticError(
                    f"{function_type!r} (of concept {function_name}) is not a Function; it can not be evaluated here!",
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
        # create expression with permissive provenance and access characters
        expression = Expression(
            function_type,
            FunctionArgumentProvenance.ANY,
            FunctionArgumentAccessor.GET,
            {function_name: arguments},
            expression_value,
        )
        violation = expression.static_semantic_violation()
        if violation is not None:
            return None, [CHSemanticError(violation, location_id=arguments_location_id, part=PathPart.VALUE)]
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

    def parse_type(
        self,
        type_candidate: str,
        template_substitution: dict[str, ConceptHierarchyTemplateArgument],
        location_id: LocationId,
    ) -> tuple[InstantiatedType | None, list[ConceptHierarchyError]]:
        try:
            res = parse_convert_type_in_template_context(type_candidate, self.context.type_validator, location_id)
            if not isinstance(res, Instantiated):
                res = substitute(
                    res,
                    template_substitution,
                    self.context.template_context,
                    TemplateContext("global"),
                    self.context.type_application_constraints_validator,
                    location_id,
                )
                assert isinstance(res, TYPE_VALUE_IS_INSTANCE_CHECK)
            return res, []
        except ConceptHierarchyError as e:
            return None, [e]

    @contextmanager
    def replace_variable_scope_for_custom_function(self, new_variables: dict[str, TypeValue]) -> Iterator[None]:
        previous = self.context.variable_context
        assert previous is not None, "there is no variable context to scope"
        # Frame 0 is the global variables: those stay; every frame above them is dropped for the duration.
        globals_frame = previous.stack_frames[0] if previous.stack_frames else VariableStackFrame()
        self.context.set_variable_context(VariableContext([globals_frame, VariableStackFrame(new_variables)]))
        try:
            yield
        finally:
            self.context.set_variable_context(previous)

    def is_domain_concept_datum_in_concept_hierarchy(self, key: str) -> bool:
        return (
            key in self.context.ch.all_domain_concept_functions or key in self.context.ch.all_domain_concept_properties
        )

    def substitute_with_x(
        self,
        custom_type: TypeValue,
        substitution: dict[str, ConceptHierarchyTemplateArgument],
        schema_owner: str,
        key_location_id: LocationId,
    ) -> TypeValue:
        model_value_domain = self.context.model.value_domains[schema_owner]
        template_context_in_which_type_was_written = model_value_domain.template_context
        assert self.context.model.x_template_variable_constraint is not None
        self.context.type_validator.add_template_variable(
            "x", self.context.model.x_template_variable_constraint, key_location_id
        )
        try:
            res = substitute(
                custom_type,
                substitution,
                self.context.template_context,
                template_context_in_which_type_was_written,
                self.context.type_application_constraints_validator,
                key_location_id,
            )[0]
            self.context.type_validator.delete_template_variable("x", key_location_id)
            return res
        except ConceptHierarchyError:
            self.context.type_validator.delete_template_variable("x", key_location_id)
            raise

    def collect_data(
        self,
        concept_restriction: list[InstantiatedType] | None,
        for_properties_or_functions: ForPropertyOrFunction,
        include_parent_data: bool,
    ) -> dict[str, InstantiatedType]:
        res: dict[str, InstantiatedType] = {}
        if concept_restriction is None:
            if for_properties_or_functions is ForPropertyOrFunction.PROPERTY:
                for prop_name, concept_defining_prop in self.context.ch.all_domain_concept_properties.items():
                    res[prop_name] = self.context.model.domain_concepts[concept_defining_prop].property_types[prop_name]
            else:
                assert for_properties_or_functions is ForPropertyOrFunction.FUNCTION
                for func_name, concept_defining_func in self.context.ch.all_domain_concept_functions.items():
                    res[func_name] = self.context.model.domain_concepts[concept_defining_func].function_types[func_name]
        else:
            for domain_concept in concept_restriction:
                if for_properties_or_functions is ForPropertyOrFunction.PROPERTY:
                    if include_parent_data:
                        res.update(
                            self.context.model.domain_concepts[domain_concept.full_name].all_available_property_types
                        )
                    else:
                        res.update(self.context.model.domain_concepts[domain_concept.full_name].property_types)
                else:
                    assert for_properties_or_functions is ForPropertyOrFunction.FUNCTION
                    if include_parent_data:
                        res.update(
                            self.context.model.domain_concepts[domain_concept.full_name].all_available_function_types
                        )
                    else:
                        res.update(self.context.model.domain_concepts[domain_concept.full_name].function_types)
        return res
