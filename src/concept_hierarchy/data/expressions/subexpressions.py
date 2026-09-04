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

from dataclasses import dataclass
from enum import Enum
from typing import Iterator

from concept_hierarchy.data.expressions.expression import Expression, ExpressionValue
from concept_hierarchy.data.expressions.instantiated_value import ParsedCustomValue, ParsedValue
from concept_hierarchy.data.type_template_variables.constraint_formula import ConstraintGroup
from concept_hierarchy.data.types.concept_hierarchy_types import ConceptHierarchyType, InstantiatedType, TypeValue
from concept_hierarchy.errors import CHSemanticError, ConceptHierarchyError, LocationId


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


class DefaultSerializationExpression(InstExpression):
    """
    A value recognised by its concept's ``defaultSerialization`` rather than by an instantiation schema.

    There is no :class:`ParsedValue` for such an expression -- the JSON literal *is* the value, and no
    schema was walked to produce it -- so the value is kept here instead.

    It has to be kept *somewhere*, because :attr:`Expression.unparsed` is the declared source text and is
    never rewritten by substitution: an evaluation written ``{"Add<T>": ...}`` still reports exactly that
    after ``T := Integer``, and a literal template variable still reports ``"N"`` after substituting ``N := 3``.
    This is the only place where the value, from which a default-serialized expression was built, survives.
    """

    def __init__(self, value_type: TypeValue, json_value: object, is_strict_subtype: bool | None = None):
        super().__init__(None, value_type, is_strict_subtype)
        self.json_value = json_value
        """The JSON value this was recognised from, *after* any substitution -- unlike ``unparsed``."""


class NarrowExpression(InstExpression):
    def __init__(self, value: ParsedValue, value_type: TypeValue | None = None, is_strict_subtype: bool | None = None):
        super().__init__(value, value_type, is_strict_subtype)


class ExpressionKind(Enum):
    """The alternatives ``_parse_syntax_of_expression_with_instantiated_type`` searches over."""

    FUNCTION_EVALUATION = "FEval"
    NARROW = "Narrow"
    INSTANTIATION = "Inst"
    DEFAULT_SERIALIZATION = "default serialization"
    VARIABLE = "variable"
    INSTANCE_PROPERTY_CHAIN = "instance property chain"
    LITERAL_TEMPLATE_VARIABLE = "literal template variable"


@dataclass(frozen=True)
class ConstraintGroupAttempt:
    """
    One ``(ConstraintGroup, CHSchemaNode)`` pair of a type's ``instantiation``, as tried by
    :func:`~expression_parser._check_instantiation_schema`.

    ``matched`` says whether the type application satisfied the group's constraint. ``errors`` holds the
    constraint-validation errors when it did not, and the *value* errors from parsing against that group's
    schema when it did.
    """

    constraint: ConstraintGroup | None
    matched: bool
    errors: tuple[ConceptHierarchyError, ...] = ()

    def describe(self) -> str:
        constraint = "the unconstrained fallback" if self.constraint is None else f"constraint {self.constraint}"
        if not self.matched:
            return f"{constraint}: the type application does not satisfy it"
        return f"{constraint}: matched, but the value does not satisfy its instantiation schema"


@dataclass(frozen=True)
class ExpressionAttempt:
    """One alternative that was applicable to the value, and why it was rejected."""

    kind: ExpressionKind
    reason: str
    tried_type: TypeValue | None = None
    schema_errors: tuple[ConceptHierarchyError, ...] = ()
    constraint_groups: tuple[ConstraintGroupAttempt, ...] = ()
    cause: IllFormedExpression | None = None

    def describe(self) -> str:
        if self.tried_type is None:
            return f"as {self.kind.value}: {self.reason}"
        return f"as {self.kind.value} ({self.tried_type}): {self.reason}"

    def as_error(self, location_id: LocationId) -> ConceptHierarchyError:
        """Render this attempt, and anything under it, as one nested :class:`ConceptHierarchyError`."""
        causes: list[ConceptHierarchyError] = []
        for group in self.constraint_groups:
            group_error = CHSemanticError(group.describe(), location_id=location_id)
            group_error.causes.extend(group.errors)
            causes.append(group_error)
        causes.extend(self.schema_errors)
        if self.cause is not None:
            causes.extend(self.cause.explanation_causes(location_id))
        error = CHSemanticError(self.describe(), location_id=location_id)
        error.causes.extend(causes)
        return error


class IllFormedExpression(ExpressionValue):
    """
    An expression that could not be parsed.

    ``reason`` is the headline. ``attempts`` is the explanation trace: one entry per alternative that was
    applicable to this value, in the order the parser tried them, carrying whatever that alternative
    learned before giving up -- the errors from an instantiation schema, the constraint groups that were
    tested, or a nested :class:`IllFormedExpression` for a sub-expression that itself failed.
    """

    def __init__(self, reason: str, attempts: tuple[ExpressionAttempt, ...] = ()):
        super().__init__()
        self.reason = reason
        self.attempts: tuple[ExpressionAttempt, ...] = tuple(attempts)

    @property
    def is_template_dependent(self) -> bool:
        return False

    @property
    def is_fully_parsed(self) -> bool:
        return False

    @property
    def is_valid(self):
        return False

    def explanation_causes(self, location_id: LocationId) -> list[ConceptHierarchyError]:
        """The trace as nested errors, ready to hang off a :attr:`ConceptHierarchyError.causes`."""
        return [attempt.as_error(location_id) for attempt in self.attempts]

    def get_subexpressions(self) -> Iterator[Expression]:
        """This is not a (completely) parsed expression... It does not contain subexpressions"""
        yield from []
