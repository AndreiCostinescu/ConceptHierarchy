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
from concept_hierarchy.errors import LocationId
from concept_hierarchy.models import ConceptHierarchyModel


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
    def ch(self) -> ConceptHierarchyModel:
        return self.model.ch

    def set_template_context(self, template_context: TemplateContext) -> ConceptHierarchyContext:
        if self.template_context.empty:
            return ConceptHierarchyContext(self.model, template_context, self.variable_context)
        raise RuntimeError(
            f"Use the extend method to extend an existing template_context; "
            f"this one {self.template_context!r} is not empty, can't set!"
        )

    def template_context_extend(
        self, template_context: TemplateContext, location_id: LocationId
    ) -> ConceptHierarchyContext:
        return ConceptHierarchyContext(
            self.model, self.template_context.add_context(template_context, location_id), self.variable_context
        )

    def template_context_make_neg(self) -> ConceptHierarchyContext:
        return ConceptHierarchyContext(
            self.model,
            TemplateContext(
                self.template_context.variables,
                self.template_context.variadic_variables,
                self.template_context.make_constraint_neg(),
            ),
            self.variable_context,
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
