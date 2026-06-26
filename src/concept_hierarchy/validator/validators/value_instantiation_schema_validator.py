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
from concept_hierarchy.data.parsers.jsonschema_parser import CHSchemaValidator
from concept_hierarchy.data.parsers.template_argument_constraint_parser import parse_constraint_definition
from concept_hierarchy.data.type_template_variables.constraint_formula import (
    NonStructureConstraintFormula,
    NonTypeTemplateConstraintFormula,
    TypeTemplateConstraintFormula,
)
from concept_hierarchy.data.types.concept_hierarchy_types import TypeValue
from concept_hierarchy.data.validators.type_validator import parse_convert_type_in_template_context
from concept_hierarchy.definitions.concept_definition_hidden_implementation import HiddenImplementationDefinition
from concept_hierarchy.errors import ConceptHierarchyError, LocationId


class SchemaValidator(CHSchemaValidator):
    x_template_variable_constraint: NonStructureConstraintFormula | None = None

    def __init__(self, context: ConceptHierarchyContext):
        self.context = context
        self.type_validator = self.context.type_validator
        if SchemaValidator.x_template_variable_constraint is None:
            res = parse_constraint_definition(
                HiddenImplementationDefinition.default_template_argument_constraint,
                self.context.template_constraint_formula_validator,
                LocationId(),
                allow_unconstrained=True,
            )
            assert isinstance(res, NonStructureConstraintFormula)
            SchemaValidator.x_template_variable_constraint = res

    def parse_custom_type(
        self, type_name: str, location_id: LocationId, allow_x_as_template_variable: bool
    ) -> TypeValue:
        if not allow_x_as_template_variable:
            return parse_convert_type_in_template_context(type_name, self.type_validator, location_id)

        # FIXME: add as identifier to the x template variable the location_id!
        #  Because in an instantiation schema, there can be multiple x template variables defined,
        #  and all of them must be uniquely identifiable.
        self.type_validator.add_template_variable("x", SchemaValidator.x_template_variable_constraint, location_id)
        try:
            res = parse_convert_type_in_template_context(type_name, self.type_validator, location_id)
            self.type_validator.delete_template_variable("x", location_id)
            return res
        except ConceptHierarchyError as e:
            self.type_validator.delete_template_variable("x", location_id)
            raise e

    def set_identifier_where_types_are_defined(self, identifier: str) -> None:
        self.type_validator.set_identifier_where_types_are_defined(identifier)

    def clear_identifier_where_types_are_defined(self) -> None:
        self.type_validator.clear_identifier_where_types_are_defined()

    def is_concept(self, concept_name: str) -> bool:
        return self.context.ch.is_concept(concept_name)

    def is_template_variable(self, template_variable_name: str) -> bool:
        return template_variable_name in self.context.template_context.variables

    def _get_constraint_type_of_template_variable(self, template_variable_candidate: str) -> str | None:
        if self.context.template_context.empty:
            return None
        try:
            index = self.context.template_context.variables.index(template_variable_candidate)
            return self.context.template_context.constraint.variable_constraint_types[index]
        except ValueError:
            return None

    def is_type_template_variable(self, template_variable_candidate: str) -> bool:
        return (
            self._get_constraint_type_of_template_variable(template_variable_candidate)
            == TypeTemplateConstraintFormula.TYPE
        )

    def is_boolean_template_variable(self, template_variable_candidate: str) -> bool:
        return (
            self._get_constraint_type_of_template_variable(template_variable_candidate)
            == NonTypeTemplateConstraintFormula.BOOLEAN
        )

    def is_integer_template_variable(self, template_variable_candidate: str) -> bool:
        return (
            self._get_constraint_type_of_template_variable(template_variable_candidate)
            == NonTypeTemplateConstraintFormula.INTEGER
        )

    def is_number_template_variable(self, template_variable_candidate: str) -> bool:
        return self._get_constraint_type_of_template_variable(template_variable_candidate) in {
            NonTypeTemplateConstraintFormula.INTEGER,
            NonTypeTemplateConstraintFormula.NUMBER,
        }

    def is_string_template_variable(self, template_variable_candidate: str) -> bool:
        return (
            self._get_constraint_type_of_template_variable(template_variable_candidate)
            == NonTypeTemplateConstraintFormula.STRING
        )
