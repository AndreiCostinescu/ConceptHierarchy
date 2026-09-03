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

from concept_hierarchy.data.concept_hierarchy import TypeData
from concept_hierarchy.data.contexts.context import ConceptHierarchyContext
from concept_hierarchy.data.contexts.template_context import TemplateContext
from concept_hierarchy.data.type_template_variables.constraint_formula import NonStructureConstraintFormula
from concept_hierarchy.data.types.concept_hierarchy_types import (
    ConceptHierarchyTemplateArgument,
)
from concept_hierarchy.data.validators.template_argument_constraints_validator import (
    validate_instantiation_constraints_in_template_argument_value,
)
from concept_hierarchy.data.validators.type_validator import (
    TypeTemplateData,
    TypeValidator,
)
from concept_hierarchy.definitions.concept_definition_hidden_implementation import HiddenImplementationDefinition
from concept_hierarchy.errors import CHSemanticError, LocationId


class ConceptHierarchyTypeValidator(TypeValidator):
    def __init__(self, context: ConceptHierarchyContext):
        self.context = context
        self.cached_template_data: dict[str, TypeTemplateData] = {}
        """The cache can persist between validations. This does not change: it is the (processed) definition data."""
        self.identifier_for_types: str | None = None
        self.empty_template_context_for_non_template_concepts = TemplateContext()

    def full_type_name(self, concept_name: str) -> str:
        if self.is_concept(concept_name):
            type_def_data = self.context.ch.concepts[self.canonical_concept_name(concept_name)]
            if isinstance(type_def_data, HiddenImplementationDefinition):
                return type_def_data.name_with_template_variables()
        return concept_name

    def get_template_data_of(self, concept_name: str) -> TypeTemplateData:
        # an alias and the concept it names share one entry, so also one cache entry
        concept_name = self.canonical_concept_name(concept_name)
        if concept_name not in self.context.ch.concepts:
            raise RuntimeError(f"Wrong concept name specified: {concept_name}")
        if concept_name not in self.cached_template_data:
            type_def_data = self.context.ch.concepts[concept_name]
            if not isinstance(type_def_data, HiddenImplementationDefinition):
                type_template_data = TypeTemplateData(
                    get_context=lambda: self.empty_template_context_for_non_template_concepts,
                    variadic_group_identifiers={},
                    defined_variadic_group_identifiers={},
                )
            else:
                type_model_data = self.context.model.concepts[concept_name]
                assert isinstance(type_model_data, TypeData)
                type_data: TypeData = type_model_data
                type_template_data = TypeTemplateData(
                    get_context=lambda: type_data.template_context,
                    variadic_group_identifiers=type_def_data.variadic_template_argument_group_identifiers,
                    defined_variadic_group_identifiers=type_def_data.defined_variadic_group_identifiers,
                )
            self.cached_template_data[concept_name] = type_template_data
        return self.cached_template_data[concept_name]

    def is_concept(self, concept_name: str) -> bool:
        return self.context.ch.is_concept(concept_name)

    def canonical_concept_name(self, concept_name: str) -> str:
        return self.context.ch.canonical_concept_name(concept_name)

    def is_template_variable(self, concept_name: str) -> bool:
        return self.context.template_context.has_template_variable(concept_name)

    def is_variadic_template_variable(self, concept_name: str) -> bool:
        return self.context.template_context.has_template_variable(
            concept_name
        ) and self.context.template_context.is_variadic(concept_name)

    def get_available_template_variables(self) -> list[str]:
        return list(self.context.template_context.variables)

    def set_identifier_where_types_are_defined(self, identifier: str):
        self.identifier_for_types = identifier

    def get_identifier_where_types_are_defined(self) -> str:
        return self.identifier_for_types

    def clear_identifier_where_types_are_defined(self):
        self.identifier_for_types = None

    def add_template_variable(
        self,
        template_variable_name: str,
        template_variable_constraint: NonStructureConstraintFormula,
        location_id: LocationId,
    ):
        self.context.template_context = self.context.template_context.add_template_variable(
            template_variable_name, False, template_variable_constraint, location_id
        )

    def delete_template_variable(self, template_variable_name: str, location_id: LocationId):
        self.context.template_context = self.context.template_context.delete_template_variable(
            template_variable_name, location_id
        )

    def validate_fully_instantiated_types_in_converted_value(
        self, value: ConceptHierarchyTemplateArgument, location_id: LocationId
    ) -> None:
        errors = validate_instantiation_constraints_in_template_argument_value(
            value, self.context.type_application_constraints_validator, location_id
        )
        if errors:
            raise CHSemanticError(
                f"Type validation failed for {value}! Template argument constraints of a defined type not satisfied!",
                causes=errors,
            )
