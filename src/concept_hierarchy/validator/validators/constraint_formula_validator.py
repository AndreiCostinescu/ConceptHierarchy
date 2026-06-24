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
from concept_hierarchy.data.parsers.template_argument_constraint_parser import TemplateConstraintFormulaValidator
from concept_hierarchy.definitions.concept_definition_hidden_implementation import HiddenImplementationDefinition


class ConstraintFormulaValidator(TemplateConstraintFormulaValidator):
    def __init__(self, context: ConceptHierarchyContext):
        self.context = context
        # contains all template arguments available in the ValueDomain concept that defines the constraints
        self.t_arg_context: set[str] = set()

    def full_type_name(self, name: str) -> str:
        if self.is_concept(name):
            type_def_data = self.context.ch.concepts[name]
            if isinstance(type_def_data, HiddenImplementationDefinition):
                return type_def_data.name_with_template_variables()
        return name

    def get_nr_template_arguments(self, concept_name: str) -> int:
        if self.is_concept(concept_name):
            type_def_data = self.context.ch.concepts[concept_name]
            if isinstance(type_def_data, HiddenImplementationDefinition):
                return len(type_def_data.template_argument_order)
        return 0

    def is_concept(self, name: str) -> bool:
        return self.context.ch.is_concept(name)

    def is_template_variable(self, name: str):
        return name in self.t_arg_context

    def get_existing_template_variables(self) -> set[str]:
        return self.t_arg_context

    def update_existing_template_variables(self, new_template_variables: set[str]):
        self.t_arg_context = new_template_variables
