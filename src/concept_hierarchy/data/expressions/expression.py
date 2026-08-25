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

from abc import ABC, abstractmethod

from concept_hierarchy.data.expressions.expression_utils import (
    FunctionArgumentAccessor,
    FunctionArgumentProvenance,
)
from concept_hierarchy.data.types.concept_hierarchy_types import TypeValue


class ExpressionValue(ABC):
    def __init__(self, value_type: TypeValue | None = None, is_strict_subtype: bool | None = None):
        self.value_type: TypeValue | None = value_type
        self.is_strict_subtype: bool | None = is_strict_subtype

    @property
    @abstractmethod
    def is_template_dependent(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_fully_parsed(self) -> bool:
        """
        An expression can be not fully parsed if its type is a template variable:
         - not a TemplateDependentType instance, a TemplateVariable instance
         - e.g. Sequence<T> can be parsed as a sequence, but its individual elements (of type T) can not be parsed
        IllFormedExpressions are considered fully parsed (even if

        :return:
        """

    @property
    def is_valid(self) -> bool:
        return True


class Expression:
    def __init__(
        self,
        expr_type: TypeValue,
        expr_provenance: FunctionArgumentProvenance,
        expr_access: FunctionArgumentAccessor,
        json_value: object,
        expr_value: ExpressionValue,
    ):
        self.required_expression_type: TypeValue = expr_type
        self.required_provenance_type: FunctionArgumentProvenance = expr_provenance
        self.required_access_type: FunctionArgumentAccessor = expr_access
        self.unparsed: object = json_value
        self.value: ExpressionValue | None = expr_value

    @property
    def value_type(self) -> TypeValue | None:
        return self.value.value_type

    @property
    def is_strict_subtype(self) -> bool | None:
        """
        Whether the resolved expression type is a *strict* subtype (i.e. not exactly required_expression_type itself).
        """
        return self.value.is_strict_subtype

    @property
    def is_type_template_dependent(self) -> bool:
        return self.required_expression_type.depends_on_templates

    @property
    def is_value_template_dependent(self) -> bool:
        return self.value.is_template_dependent

    @property
    def is_fully_parsed(self) -> bool:
        """
        An expression can be not fully parsed if its type is a template variable:
         - not a TemplateDependentType instance, a TemplateVariable instance
         - e.g. Sequence<T> can be parsed as a sequence, but its individual elements (of type T) can not be parsed
        IllFormedExpressions are considered fully parsed (even if

        :return:
        """
        return self.value.is_fully_parsed

    @property
    def is_valid(self) -> bool:
        return self.value.is_valid
