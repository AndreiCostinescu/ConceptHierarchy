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

from concept_hierarchy.data.expressions.expression import Expression, ExpressionValue
from concept_hierarchy.data.expressions.instantiated_value import ParsedCustomValue, ParsedValue
from concept_hierarchy.data.type_template_variables.constraint_formula import (
    ConstraintGroup,
)
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

    def get_subexpressions(self):
        """this is not a parsed expression... on its own, it does not contain subexpressions"""
        yield from []


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

    def get_subexpressions(self):
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

    def get_subexpressions(self):
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

    def get_subexpressions(self):
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
        for value_node in self.value.walk():
            if isinstance(value_node, ParsedCustomValue):
                if value_node.expression is not None and not value_node.expression.is_value_template_dependent:
                    return False
        return True
        """
        # FIXME: this is not correct because it only checks subexpressions, not the value itself!
        return all(x[1].is_value_template_dependent for x in self.value.iter_expressions())

    @property
    def is_fully_parsed(self) -> bool:
        for value_node in self.value.walk():
            if isinstance(value_node, ParsedCustomValue):
                if value_node.expression is not None and not value_node.expression.is_fully_parsed:
                    return False
        return True

    def get_subexpressions(self):
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

    def get_subexpressions(self):
        """This is not a (completely) parsed expression... It does not contain subexpressions"""
        yield from []
