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

    kind_name: str = ""
    """A reader-facing name for this kind of expression, used in diagnostics."""

    @property
    def is_valid(self) -> bool:
        return True

    @property
    def is_addressable(self) -> bool:
        """
        Whether this expression denotes a *place* rather than a fresh value.

        ``False`` for everything that constructs: an `Inst`, a `Narrow`, a default serialization, and a
        Function evaluation whose result is not declared `Addr`. Only a place can satisfy an argument whose
        provenance is `Addr`, and only a place makes the exactness rule bite -- writing through an alias
        typed as a supertype could store a value of the wrong type, whereas modifying a fresh value
        modifies a copy and harms nobody.
        """
        return False

    @property
    def is_modifiable(self) -> bool:
        """
        Whether what this expression denotes may be written to.

        Only meaningful together with :attr:`is_addressable`: a fresh value is always modifiable, because
        modifying it modifies the copy the expression just produced. Variables are modifiable
        unconditionally -- the exception would be a ``Constant<T>``, which this hierarchy does not have --
        and a Function evaluation is modifiable exactly when its result accessor says so.
        """
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

    def static_semantic_violation(self) -> str | None:
        """
        Why :attr:`value` can not satisfy the required provenance, access and type -- or ``None``.

        The rules of ``documentation/[CH].md`` §10.3 in one place, so that *every* construction of an
        `Expression` is checked rather than only the one the parser makes. There are two independent axes,
        and reading them as one is what made the previous implementation reject three legitimate shapes:

        * **provenance** -- `Addr` demands a *place*, so only a variable, an instance property chain, or a
          Function evaluation with an `Addr` result can satisfy it. Everything else constructs a fresh
          value, which has no address to give.
        * **access** -- `Modify` demands that what is written to is of the exact expected type, *and* that
          it may be written to at all. Both apply only to a place. An `Inst`, a `Narrow`, a default
          serialization or a non-`Addr` evaluation may be a strict subtype under `Modify`, because the
          modification lands on the value the expression just built. The default serialization is why that
          matters in practice: ``1`` written at a `Number` site serializes to `Integer`, a strict subtype,
          and is perfectly legal to modify.

        Every violation is reported, not the first: a value can fail both axes, and hearing about only one
        of them sends the reader looking in the wrong place.
        """
        violations: list[str] = []
        kind = self.value.kind_name or type(self.value).__name__
        if self.required_provenance_type != FunctionArgumentProvenance.ANY and not self.value.is_addressable:
            violations.append(
                f"Provenance violation: {self.required_provenance_type.value} provenance requires an "
                f"addressable expression (a variable, an instance property chain, or a Function evaluation "
                f"whose result is {FunctionArgumentProvenance.ADDR.value}); got a "
                f"{kind} of type {self.value.value_type}"
            )
        if self.required_access_type != FunctionArgumentAccessor.GET and self.value.is_addressable:
            if self.is_strict_subtype:
                violations.append(
                    f"Access violation: {self.required_access_type.value} access requires the exact type "
                    f"{self.required_expression_type}, but this {kind} has type {self.value.value_type}, "
                    f"which is a strict subtype"
                )
            if not self.value.is_modifiable:
                violations.append(
                    f"Access violation: {self.required_access_type.value} access requires an expression that "
                    f"may be written to; this {kind} of type {self.value.value_type} is not modifiable"
                )
        return "; ".join(violations) if violations else None

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
