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
from concept_hierarchy.data.parsers.expression_parser import ExpressionParserValidator
from concept_hierarchy.data.parsers.jsonschema_parser import CHSchemaValidator
from concept_hierarchy.data.type_template_variables.constraint_formula import TemplateConstraintFormulaValidator
from concept_hierarchy.data.types.concept_hierarchy_types import TypeValue
from concept_hierarchy.data.validators.template_argument_constraints_validator import (
    TypeTemplateInstantiationValidator,
)
from concept_hierarchy.data.validators.type_validator import TypeValidator
from concept_hierarchy.data.validators.value_instantiation_validator import CHValueValidator
from concept_hierarchy.errors import LocationId
from concept_hierarchy.models import ConceptHierarchyModel


class ConceptHierarchyContext:
    """This class can not be copied, because the context of the validators will not be updated!"""

    def __init__(self, concept_hierarchy_model: ConceptHierarchy):
        self.model = concept_hierarchy_model
        self.template_context: TemplateContext | None = None
        self.variable_context: VariableContext | None = None
        # first the template constraint formulae are validated
        self.template_constraint_formula_validator: TemplateConstraintFormulaValidator | None = None
        # then types are parsed and validated
        self.type_validator: TypeValidator | None = None
        # then, during type validator, the template arguments are validated
        self.type_instantiation_constraints_validator: TypeTemplateInstantiationValidator | None = None
        # then value domain instantiation schemas must be parsed & validated
        self.instantiation_schema_validator: CHSchemaValidator | None = None
        # then value domain instantiation values must be parsed and validated
        self.instantiation_values_validator: CHValueValidator | None = None
        # and, finally, full expressions can be validated
        self.expression_parser_validator: ExpressionParserValidator | None = None

    @property
    def ch(self) -> ConceptHierarchyModel:
        return self.model.ch

    def __copy__(self):
        raise RuntimeError(
            "The ConceptHierarchyContext can not be copied because the context of the validators can not be updated "
            "from here!"
        )

    def __deepcopy__(self, memo):
        raise RuntimeError(
            "The ConceptHierarchyContext can not be copied because the context of the validators can not be updated "
            "from here!"
        )

    def set_template_context(self, template_context: TemplateContext) -> None:
        self.template_context = template_context

    def reset_template_context(self) -> None:
        self.template_context = None

    def set_variable_context(self, variable_context: VariableContext) -> None:
        self.variable_context = variable_context

    def reset_variable_context(self) -> None:
        self.variable_context = None

    def template_context_extend(self, template_context: TemplateContext, location_id: LocationId):
        self.set_template_context(self.template_context.add_context(template_context, location_id))

    def template_context_make_neg(self):
        self.set_template_context(
            TemplateContext(
                self.template_context.variables,
                self.template_context.variadic_variables,
                self.template_context.make_constraint_neg(),
            )
        )

    def add_new_variable(self, variable: str, variable_type: TypeValue):
        self.set_variable_context(self.variable_context.add_variable(variable, variable_type))

    def add_new_variables(self, variables: dict[str, TypeValue | dict]):
        self.set_variable_context(self.variable_context.add_variables(variables))

    def add_variable_context(self, variable_context: VariableContext):
        self.set_variable_context(self.variable_context.add_context(variable_context))
