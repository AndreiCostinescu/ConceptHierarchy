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

from typing import Iterator

from concept_hierarchy.data.expressions.expression import Expression, ExpressionValue
from concept_hierarchy.data.expressions.instantiated_value import ParsedCustomValue, ParsedValue
from concept_hierarchy.data.type_template_variables.constraint_formula import ConstraintGroup
from concept_hierarchy.data.types.concept_hierarchy_types import ConceptHierarchyType, InstantiatedType, TypeValue


class TemplateDependentExpression(ExpressionValue):
    """
    Represents an expression whose type and/or value is template dependent.
    For example:
    - an expression whose type is a TemplateVariable or a TemplateDependentInstantiation.
    -- in this case, the expression is unparsed
    -- if it comes to an instantiation, there can be multiple schemas that could match the expression's type
    - an expression whose value is template dependent
    -- a literal TemplateVariable?
    -- an evaluation of a non-ground Function-type-application (template-dependent)
    """

    def __init__(self, value_type: TypeValue | None = None, is_strict_subtype: bool | None = None):
        super().__init__(value_type=value_type, is_strict_subtype=is_strict_subtype)
        self.checked_template_instantiations: dict[ConstraintGroup, Expression] = {}

    @property
    def is_fully_parsed(self) -> bool:
        return False

    @property
    def is_template_dependent(self) -> bool:
        return True

    def get_subexpressions(self) -> Iterator[Expression]:
        """this is not a parsed expression... on its own, it does not contain subexpressions"""
        yield from []


class VerifiedTemplateDependentExpression(TemplateDependentExpression):
    """
    Represents an expression whose type and/or value is template dependent,
    and who had its value partially verified, but not parsed.
    """

    def __init__(
        self,
        possible_expressions: list[ExpressionValue],
        value_type: TypeValue | None = None,
        is_strict_subtype: bool | None = None,
    ):
        super().__init__(value_type=value_type, is_strict_subtype=is_strict_subtype)
        self.possible_expressions = possible_expressions

    @property
    def is_valid(self) -> bool:
        return len(self.possible_expressions) > 0 and all(
            not isinstance(expr_value, IllFormedExpression) for expr_value in self.possible_expressions
        )

    def get_subexpressions(self) -> Iterator[Expression]:
        """Yield the subexpressions located at any of its possible expressions."""
        for expr_value in self.possible_expressions:
            yield from expr_value.get_subexpressions()


class PossibleInstExpression(TemplateDependentExpression):
    def __init__(self, value_type: TypeValue | None = None):
        super().__init__(value_type)


class PossibleNarrowExpression(TemplateDependentExpression):
    def __init__(self, value_type: TypeValue, is_strict_subtype: bool | None = None):
        super().__init__(value_type, is_strict_subtype)


class PossibleFunctionEvaluationExpression(TemplateDependentExpression):
    def __init__(self, value_type: TypeValue | None = None, is_strict_subtype: bool | None = None):
        super().__init__(value_type, is_strict_subtype)


class LiteralTemplateVariableValue(TemplateDependentExpression):
    """The value of the expression is a TemplateVariable with a literal constraint."""

    def __init__(
        self, template_variable_name: str, template_variable_literal_constraint: TypeValue, is_strict_subtype: bool
    ):
        super().__init__(value_type=template_variable_literal_constraint, is_strict_subtype=is_strict_subtype)
        self.template_variable_name = template_variable_name

    @property
    def is_value_template_dependent(self) -> bool:
        return True

    @property
    def is_fully_parsed(self) -> bool:
        return True

    def get_subexpressions(self) -> Iterator[Expression]:
        """This does not contain subexpressions."""
        yield from []


class Variable(ExpressionValue):
    def __init__(self, variable_name: str, variable_type: TypeValue, is_strict_subtype: bool | None = None):
        super().__init__(value_type=variable_type, is_strict_subtype=is_strict_subtype)
        self.variable_name = variable_name

    @property
    def variable_type(self) -> TypeValue:
        return self.value_type

    @property
    def is_template_dependent(self) -> bool:
        return False

    @property
    def is_fully_parsed(self) -> bool:
        return True

    def get_subexpressions(self) -> Iterator[Expression]:
        """This does not contain subexpressions."""
        yield from []


class VariableWithTemplateType(Variable, TemplateDependentExpression):
    def __init__(self, variable_name: str, variable_type: TypeValue):
        super().__init__(variable_name, variable_type)

    @property
    def is_template_dependent(self) -> bool:
        return True

    @property
    def is_fully_parsed(self) -> bool:
        return False


class PossibleVariableExpression(Variable, TemplateDependentExpression):
    """
    The difference between PossibleVariableExpression and VariableWithTemplateType is:
     - for VariableWithTemplateType, the expression type is a ground type application
     - for PossibleVariableExpression, the expression type is not a ground type application
        a template variable or a template-dependent instantiation
    """

    def __init__(self, variable_name: str, variable_type: TypeValue):
        super().__init__(variable_name, variable_type)

    @property
    def is_template_dependent(self) -> bool:
        return True

    @property
    def is_fully_parsed(self) -> bool:
        return False


class InstancePropertyChain(Variable):
    def __init__(
        self,
        instance_prop_chain: list[str],
        instance_prop_chain_types: list[InstantiatedType],
        is_strict_subtype: bool | None = None,
    ):
        if len(instance_prop_chain) == 0:
            raise RuntimeError("Can not have an empty instance property chain!")
        elif len(instance_prop_chain) == 1:
            raise RuntimeError(
                "Tried to create a single-element property chain; please create a Variable expression instead!"
            )
        super().__init__(instance_prop_chain[0], instance_prop_chain_types[-1], is_strict_subtype)
        self.prop_chain = instance_prop_chain
        self.prop_chain_types = instance_prop_chain_types

    @property
    def is_template_dependent(self) -> bool:
        return False

    @property
    def is_fully_parsed(self) -> bool:
        return True


class FunctionEvaluation(ExpressionValue):
    def __init__(
        self,
        f_type: ConceptHierarchyType,
        f_res: TypeValue,
        arguments: dict[str, Expression],
        is_result_addressable: bool,
        is_strict_subtype: bool | None = None,
    ):
        super().__init__(value_type=f_res, is_strict_subtype=is_strict_subtype)
        self.f_type = f_type
        self.arguments = arguments
        self.is_result_addressable = is_result_addressable

    @property
    def is_fully_parsed(self) -> bool:
        return all(arg.is_fully_parsed for arg in self.arguments.values())

    @property
    def is_template_dependent(self) -> bool:
        return self.f_type.depends_on_templates or any(
            arg.is_value_template_dependent for arg in self.arguments.values()
        )

    def get_subexpressions(self) -> Iterator[Expression]:
        """The argument values are the sub-expressions."""
        yield from self.arguments.values()


class InstExpression(ExpressionValue):
    def __init__(
        self, value: ParsedValue = None, value_type: TypeValue | None = None, is_strict_subtype: bool | None = None
    ):
        super().__init__(value_type=value_type, is_strict_subtype=is_strict_subtype)
        self.value = value

    @property
    def is_template_dependent(self) -> bool:
        """
        The value is template dependent if its own content is, or if any of its subexpressions is.

        The content itself is template dependent when a custom-type leaf of the value was parsed against a
        type that still mentions a template variable (``Box<T>`` rather than ``Box<Integer>``); such a leaf
        contributes even when no expression could be parsed at it.

        Values that *denote* a type -- a TypeValue or a ConceptValue -- are deliberately not inspected: the
        schema does not substitute those, so a template variable written there is not a template argument
        of this value.

        A value of None (a default-serialization expression) carries no content and no subexpressions, and
        so never depends on templates.
        """
        if self.value is None:
            return False
        content_is_template_dependent = any(
            value_node.custom_type.depends_on_templates
            for value_node in self.value.walk()
            if isinstance(value_node, ParsedCustomValue)
        )
        return content_is_template_dependent or any(
            expression.is_value_template_dependent for _, expression in self.value.iter_expressions()
        )

    @property
    def is_fully_parsed(self) -> bool:
        """A value of None (a default-serialization expression) has nothing left to parse."""
        if self.value is None:
            return True
        for value_node in self.value.walk():
            if isinstance(value_node, ParsedCustomValue):
                if value_node.expression is not None and not value_node.expression.is_fully_parsed:
                    return False
        return True

    def get_subexpressions(self) -> Iterator[Expression]:
        """Check the self.value ParsedValue type for subexpressions"""
        if self.value is None:
            yield from []
        else:
            yield from (expr for _, expr in self.value.iter_expressions())


class NarrowExpression(InstExpression):
    def __init__(self, value: ParsedValue, value_type: TypeValue | None = None, is_strict_subtype: bool | None = None):
        super().__init__(value, value_type, is_strict_subtype)


class IllFormedExpression(ExpressionValue):
    def __init__(self, reason: str):
        super().__init__()
        self.reason = reason

    @property
    def is_template_dependent(self) -> bool:
        return False

    @property
    def is_fully_parsed(self) -> bool:
        return False

    @property
    def is_valid(self):
        return False

    def get_subexpressions(self) -> Iterator[Expression]:
        """This is not a (completely) parsed expression... It does not contain subexpressions"""
        yield from []
