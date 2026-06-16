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

from __future__ import annotations

from concept_hierarchy.data.concept_hierarchy import ConceptHierarchy
from concept_hierarchy.data.contexts.template_context import TemplateContext
from concept_hierarchy.data.contexts.variable_context import VariableContext
from concept_hierarchy.data.parsers.type_parser import ParsedType


class ConceptHierarchyContext:
    def __init__(
        self,
        concept_hierarchy_model: ConceptHierarchy,
        template_context: TemplateContext,
        variable_context: VariableContext,
    ):
        self.model = concept_hierarchy_model
        self.template_context = template_context
        self.variable_context = variable_context

    @property
    def ch(self):
        return self.model.ch

    def add_new_template_variable(self, template_variable: str, is_variadic: bool) -> ConceptHierarchyContext:
        return ConceptHierarchyContext(
            self.model,
            self.template_context.add_template_variable(template_variable, is_variadic),
            self.variable_context,
        )

    def add_new_template_variables(self, template_variables: dict[str, bool]) -> ConceptHierarchyContext:
        return ConceptHierarchyContext(
            self.model, self.template_context.add_template_variables(template_variables), self.variable_context
        )

    def add_template_context(self, template_context: TemplateContext) -> ConceptHierarchyContext:
        return ConceptHierarchyContext(
            self.model, self.template_context.add_context(template_context), self.variable_context
        )

    def add_new_variable(self, variable: str, variable_type: ParsedType) -> ConceptHierarchyContext:
        return ConceptHierarchyContext(
            self.model, self.template_context, self.variable_context.add_variable(variable, variable_type)
        )

    def add_new_variables(self, variables: dict[str, ParsedType | dict]) -> ConceptHierarchyContext:
        return ConceptHierarchyContext(
            self.model, self.template_context, self.variable_context.add_variables(variables)
        )

    def add_variable_context(self, variable_context: VariableContext) -> ConceptHierarchyContext:
        return ConceptHierarchyContext(
            self.model, self.template_context, self.variable_context.add_context(variable_context)
        )
