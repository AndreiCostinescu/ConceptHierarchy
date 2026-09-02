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
from concept_hierarchy.data.contexts.template_context import TemplateContext
from concept_hierarchy.data.expressions.expression_utils import (
    FunctionArgumentAccessor,
    FunctionArgumentProvenance,
    FunctionResultAccessor,
    ValueDomainArgumentProvenance,
)
from concept_hierarchy.data.expressions.instantiated_value import ParsedValue
from concept_hierarchy.data.jsonschema import CHSchemaNode
from concept_hierarchy.data.parsers.expression_parser import ExpressionParserValidator
from concept_hierarchy.data.parsers.value_instantiation_parser import parse_value
from concept_hierarchy.data.type_template_variables.constraint_formula import ConstraintGroup
from concept_hierarchy.data.types.concept_hierarchy_types import InstantiatedType, TemplateDependentType, TypeValue
from concept_hierarchy.data.validators.template_argument_constraints_validator import TypeTemplateInstantiationValidator
from concept_hierarchy.data.validators.type_validator import parse_convert_type, parse_convert_type_in_template_context
from concept_hierarchy.definitions.concept_definition_functions import FunctionDefinition
from concept_hierarchy.errors import LocationId


class ExpressionValidator(ExpressionParserValidator):
    def __init__(self, context: ConceptHierarchyContext):
        self.context = context
        self.instantiated_types: dict[str, InstantiatedType] = {}
        self.parsed_types: dict[str, TypeValue] = {}

    def is_concept(self, candidate_concept_name: str) -> bool:
        return self.context.ch.is_concept(candidate_concept_name)

    def is_template_variable(self, candidate_template_variable_name: str) -> bool:
        return self.context.template_context.has_template_variable(candidate_template_variable_name)

    def is_literal_template_variable(self, candidate_literal_template_variable_name: str) -> bool:
        return self.context.template_context.is_literal_template_variable(candidate_literal_template_variable_name)

    def get_literal_template_var_constraint(self, literal_template_variable_name: str) -> str:
        raise NotImplementedError

    def is_variable(self, candidate_variable_name: str) -> bool:
        return self.context.variable_context.has_variable(candidate_variable_name)

    def get_variable_type(self, variable_name: str) -> TypeValue:
        return self.context.variable_context.get(variable_name)

    def is_type_abstract(self, candidate_type: InstantiatedType | TemplateDependentType) -> bool:
        return not self.context.model.value_domains[candidate_type.clean_name].instantiable

    def is_a_subtype_of_b(self, a: InstantiatedType, b: InstantiatedType, location_id: LocationId) -> bool:
        return self.context.type_application_constraints_validator.is_a_subtype_of_b(a, b, location_id)

    def create_instantiated_type(self, instantiated_type_name: str, location_id: LocationId) -> InstantiatedType:
        if instantiated_type_name not in self.instantiated_types:
            # FixMe: parse_convert_type does not check if the instantiation's template arguments fulfill the constraints
            self.instantiated_types[instantiated_type_name] = parse_convert_type(
                instantiated_type_name, self.context.type_validator, location_id
            )
            self.parsed_types[instantiated_type_name] = self.instantiated_types[instantiated_type_name]
        return self.instantiated_types[instantiated_type_name]

    def create_possibly_template_dependent_type(self, type_name: str, location_id: LocationId) -> TypeValue:
        if type_name not in self.parsed_types:
            # FixMe: parse_convert_type does not check if the instantiation's template arguments fulfill the constraints
            self.parsed_types[type_name] = parse_convert_type_in_template_context(
                type_name, self.context.type_validator, location_id
            )
            if isinstance(self.parsed_types[type_name], InstantiatedType):
                self.instantiated_types[type_name] = self.parsed_types[type_name]
        return self.parsed_types[type_name]

    def get_substituted_value_domain_instantiation_schema(
        self, type_name: InstantiatedType | TemplateDependentType
    ) -> TypeValue:
        raise NotImplementedError

    def get_substituted_function_interface(
        self, ch_type: InstantiatedType
    ) -> tuple[tuple[str, ...], set[str], InstantiatedType | None, ValueDomainArgumentProvenance | None]:
        raise NotImplementedError

    def get_properties_of_concepts(self, concepts: list[str]) -> dict[str, InstantiatedType]:
        raise NotImplementedError

    def get_type_of_instance_property(self, instance_type: InstantiatedType, prop_name: str) -> InstantiatedType:
        raise NotImplementedError

    def get_if_has_instantiation_schema(
        self, type_data: TypeValue
    ) -> tuple[tuple[ConstraintGroup, CHSchemaNode], ...] | None:
        """
        If the type is fully instantiated, return the schema.
        If the type is a TemplateDependentType, return all instantiation schemas.
        Otherwise, return None

        :param type_data: the type to check for an instantiation schema
        :return:
        """
        if not isinstance(type_data, (InstantiatedType, TemplateDependentType)):
            return None

        assert self.is_concept(type_data.clean_name)
        concept_name = type_data.clean_name
        assert concept_name in self.context.model.value_domains
        value_domain_instantiation = self.context.model.value_domains[concept_name].instantiation
        return value_domain_instantiation

    def validate_value_against_schema(
        self, schema: CHSchemaNode, value: object, location_id: LocationId
    ) -> ParsedValue:
        return parse_value(value, schema, self.context.instantiation_values_validator, location_id)[0]

    def get_default_serialization_concept_name_for(self, json_value_type: str) -> str | None:
        return self.context.ch.default_serializations.get(json_value_type, None)

    def get_function_return_interface(
        self, f_name
    ) -> tuple[TypeValue, FunctionResultAccessor, ValueDomainArgumentProvenance] | None:
        assert f_name in self.context.model.functions
        if not self.context.model.functions[f_name].returns_something:
            return None
        f = self.context.model.functions[f_name]
        return f.evaluation_result_type, f.evaluation_result_access_type, f.evaluation_result_provenance_type

    def is_function_argument(self, f_name, f_arg_name) -> bool:
        assert f_name in self.context.model.functions
        return f_arg_name in self.context.model.functions[f_name].evaluation_argument_types

    def get_function_arguments(self, f_name) -> set[str]:
        assert f_name in self.context.model.functions
        return set(self.context.model.functions[f_name].evaluation_argument_types)

    def get_required_function_arguments(self, f_name) -> set[str]:
        assert f_name in self.context.model.functions
        f = self.context.model.functions[f_name]
        f_def = self.context.ch.concepts[f_name]
        assert isinstance(f_def, FunctionDefinition)
        return set(f.evaluation_argument_types) - set(f_def.evaluation_argument_default_values)

    def get_function_argument_interface(
        self, f_name, f_arg_name
    ) -> tuple[TypeValue, FunctionArgumentAccessor, FunctionArgumentProvenance]:
        assert f_name in self.context.model.functions
        f = self.context.model.functions[f_name]
        assert f_arg_name in f.evaluation_argument_types
        return (
            f.evaluation_argument_types[f_arg_name],
            f.evaluation_argument_access_type[f_arg_name],
            f.evaluation_argument_provenance_type[f_arg_name],
        )

    def get_default_argument_dependencies(self, f_name) -> frozendict[str, frozenset[str]] | None:
        assert f_name in self.context.model.functions
        f = self.context.model.functions[f_name]
        if f.is_default_argument_dependencies_initialized():
            return f.default_argument_dependencies
        return None

    def get_template_context(self, type_name_clean) -> TemplateContext:
        assert type_name_clean in self.context.model.value_domains
        return self.context.model.value_domains[type_name_clean].template_context

    def get_type_template_instantiation_validator(self) -> TypeTemplateInstantiationValidator:
        return self.context.type_application_constraints_validator
