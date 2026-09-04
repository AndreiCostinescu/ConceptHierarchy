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
from typing import Callable, Iterator, Type

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
        Whether the parser walked through to every leaf of this expression and built one.

        That is the whole of it. **Template dependence is not recorded here at all** -- an expression can be
        walked to the end and still be waiting on a template argument, and this property says nothing about
        that. :attr:`is_template_dependent` is the property for that question, and neither implies the
        other:

        * a `FunctionEvaluation` of ``Add<T>`` whose arguments all parsed: every leaf was reached, and ``T``
          is still open -- fully parsed *and* template dependent;
        * a value holding an unresolved default site: nothing about it depends on a template, but a leaf has
          no expression -- template independent *and* not fully parsed.

        So anything deciding "can this be reused without reparsing?" has to ask both.

        ``False`` therefore means one thing only: somewhere a leaf was left without an expression. That
        happens when the parser could not decide an alternative and recorded a placeholder instead, and it
        is why an :class:`IllFormedExpression` is not fully parsed -- parsing is what failed.
        """

    @property
    def is_valid(self) -> bool:
        return True

    @abstractmethod
    def get_subexpressions(self) -> Iterator[Expression]:
        pass


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

    def all_subexpressions(
        self, filter_f: Type[ExpressionValue] | Callable[[Expression], bool] | None = None
    ) -> Iterator[Expression]:
        """Recursively yield all sub-expressions."""
        if isinstance(filter_f, type):
            filter_type = filter_f

            def filter_f(x: Expression) -> bool:
                return isinstance(x.value, filter_type)

        if filter_f is None or filter_f(self):
            yield self

        for sub_expr in self.value.get_subexpressions():
            yield from sub_expr.all_subexpressions(filter_f)
