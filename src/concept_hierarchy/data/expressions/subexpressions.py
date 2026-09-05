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
from concept_hierarchy.data.expressions.instantiated_value import ParsedCustomValue, ParsedStructural, ParsedValue
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
        """
        Not because it is template dependent -- that is not what this property records -- but because
        nothing was built: this class stands in for an expression the parser declined to walk into.
        """
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
    kind_name = "literal template variable"
    """The value of the expression is a TemplateVariable with a literal constraint."""

    def __init__(
        self, template_variable_name: str, template_variable_literal_constraint: TypeValue, is_strict_subtype: bool
    ):
        super().__init__(value_type=template_variable_literal_constraint, is_strict_subtype=is_strict_subtype)
        self.template_variable_name = template_variable_name

    @property
    def is_fully_parsed(self) -> bool:
        """
        A leaf, and it was reached: the variable's name and its literal constraint are both known. What is
        not known is its *value*, which this property does not ask about -- `is_template_dependent` does,
        and is inherited as ``True`` here.
        """
        return True

    def get_subexpressions(self) -> Iterator[Expression]:
        """This does not contain subexpressions."""
        yield from []


class Variable(ExpressionValue):
    kind_name = "variable"

    def __init__(
        self,
        variable_name: str,
        variable_type: TypeValue,
        is_strict_subtype: bool | None = None,
        *,
        scope_index: int,
    ):
        super().__init__(value_type=variable_type, is_strict_subtype=is_strict_subtype)
        self.variable_name = variable_name

        self.scope_index = scope_index
        """
        The index of the variable stack frame this name resolved in, 0 being the global variables.

        Recorded at every usage because the name alone does not identify the variable: a Function's
        arguments, and a nested call's, introduce names into scopes above the globals and shadow them. Only
        a reference that resolved at `GLOBAL_VARIABLE_SCOPE_INDEX` is a reference to the global.
        """

    @property
    def is_global_variable(self) -> bool:
        """Whether this reference resolved to a global variable rather than to something more local."""
        return self.scope_index == 0

    @property
    def is_addressable(self) -> bool:
        """A variable names a place, which is the whole of what makes it addressable."""
        return True

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
    """A variable at a site of ground type, whose *own* type still mentions a template variable."""

    def __init__(self, variable_name: str, variable_type: TypeValue, *, scope_index: int):
        super().__init__(variable_name, variable_type, scope_index=scope_index)

    @property
    def is_template_dependent(self) -> bool:
        return True

    @property
    def is_fully_parsed(self) -> bool:
        """
        A variable reference is a leaf and this one was reached, its name resolved and its type known.
        Whether that type fits the site is open, but this property does not ask that.
        """
        return True


class PossibleVariableExpression(Variable, TemplateDependentExpression):
    """
    The difference between PossibleVariableExpression and VariableWithTemplateType is:
     - for VariableWithTemplateType, the expression type is a ground type application
     - for PossibleVariableExpression, the expression type is not a ground type application
        a template variable or a template-dependent instantiation
    """

    def __init__(self, variable_name: str, variable_type: TypeValue, *, scope_index: int):
        super().__init__(variable_name, variable_type, scope_index=scope_index)

    @property
    def is_template_dependent(self) -> bool:
        return True

    @property
    def is_fully_parsed(self) -> bool:
        """
        As for :class:`VariableWithTemplateType`: a leaf, reached, with its name and type known. What is
        undecided is the *site's* type, so the parser could not check the variable against it -- an open
        check, not an unbuilt leaf, and this property only reports the latter.
        """
        return True


class InstancePropertyChain(Variable):
    def __init__(
        self,
        instance_prop_chain: list[str],
        instance_prop_chain_types: list[InstantiatedType],
        is_strict_subtype: bool | None = None,
        *,
        scope_index: int,
    ):
        if len(instance_prop_chain) == 0:
            raise RuntimeError("Can not have an empty instance property chain!")
        elif len(instance_prop_chain) == 1:
            raise RuntimeError(
                "Tried to create a single-element property chain; please create a Variable expression instead!"
            )
        super().__init__(
            instance_prop_chain[0], instance_prop_chain_types[-1], is_strict_subtype, scope_index=scope_index
        )
        self.prop_chain = instance_prop_chain
        self.prop_chain_types = instance_prop_chain_types

    kind_name = "instance property chain"

    @property
    def is_template_dependent(self) -> bool:
        return False

    @property
    def is_fully_parsed(self) -> bool:
        return True


class FunctionEvaluation(ExpressionValue):
    kind_name = "Function evaluation"

    def __init__(
        self,
        f_type: ConceptHierarchyType,
        f_res: TypeValue,
        arguments: dict[str, Expression],
        is_result_addressable: bool,
        is_result_modifiable: bool | None = None,
        is_strict_subtype: bool | None = None,
        applied_defaults: dict[str, Expression] | None = None,
    ):
        super().__init__(value_type=f_res, is_strict_subtype=is_strict_subtype)
        self.f_type = f_type

        self.arguments = arguments
        """What this call site **wrote**. Nothing else may go in here -- see :attr:`applied_defaults`."""

        self.is_result_addressable = is_result_addressable

        self.is_result_modifiable = is_result_modifiable
        """
        Whether the Function's *result* may be modified, i.e. its `FunctionResultAccessor` is not ``Get``.

        ``None`` when the Function returns nothing, which is the same case in which `value_type` is ``None``.
        Read off the return interface beside `is_result_addressable`, and stored for the same reason: it is a
        property of the evaluation, and recomputing it needs the interface this expression no longer holds.
        """

        self.applied_defaults: dict[str, Expression] = applied_defaults or {}
        """
        What each argument the call site left out fell back on, grounded for *this* application.

        A **second** field rather than entries in :attr:`arguments`, and that is not tidiness.
        `_parse_expression_of_json_object` derives ``supplied_arguments = set(f_args)`` from `arguments`, so a
        default recorded there would make the acyclicity check believe every argument was supplied and skip
        the graph exactly where it is needed. Keeping the two apart lets an evaluation record what it depends
        on without lying about what was written.

        Deliberately **not** yielded from :meth:`get_subexpressions`. `init_expressions` computes
        `FunctionData.default_argument_dependencies` with `Expression.all_subexpressions(Variable)` and
        filters the names it finds against *this* Function's argument names; descending into a nested
        evaluation's applied defaults imports another Function's argument names, and any that collide become
        edges that do not exist. Measured, with `G.q := p` where `F` also has an argument `p`: `F`'s
        dependencies become ``{'p': ['b'], 'b': ['p']}`` instead of ``{'p': ['b'], 'b': []}``, and the call
        site ``{"F": {}}`` is then rejected as a cycle that is not there. A consumer that wants these has to
        walk them explicitly.
        """

    @property
    def is_addressable(self) -> bool:
        """Declared by the Function's result provenance: only an `Addr` result denotes a place."""
        return bool(self.is_result_addressable)

    @property
    def is_modifiable(self) -> bool:
        """Declared by the Function's result accessor. ``None`` -- a Function returning nothing -- is not."""
        return bool(self.is_result_modifiable)

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
    kind_name = "value domain instantiation"

    def __init__(
        self, value: ParsedValue = None, value_type: TypeValue | None = None, is_strict_subtype: bool | None = None
    ):
        super().__init__(value_type=value_type, is_strict_subtype=is_strict_subtype)
        self.value = value

    @property
    def is_template_dependent(self) -> bool:
        """
        The value is template dependent if its own content is, or if any of its subexpressions is.

        The content is template dependent in either of two ways, and both are read off the parsed value
        rather than encoded in the class:

        * a **custom-type leaf** was parsed against a type that still mentions a template variable
          (``Box<T>`` rather than ``Box<Integer>``); such a leaf contributes even when no expression could
          be parsed at it;
        * a **keyword** that would have constrained one of the nodes is still written as an unsubstituted
          literal template variable (``{"minItems": "N"}``). `jsonschema_parser` pops such a keyword out of
          the schema, so the Draft-07 validator never saw it and the value passed that node unchecked -- it
          is undecided in exactly the same sense as the leaf above, and invisible unless the schema node is
          asked.

        Only the nodes the parse actually visited are walked, so an ``anyOf`` branch the value never entered
        holds anything against it.

        Values that *denote* a type -- a TypeValue or a ConceptValue -- are deliberately not inspected: the
        schema does not substitute those, so a template variable written there is not a template argument
        of this value.

        A value of None (a default-serialization expression) carries no content and no subexpressions, and
        so never depends on templates.
        """
        if self.value is None:
            return False
        for value_node in self.value.walk():
            if isinstance(value_node, ParsedCustomValue):
                if value_node.custom_type.depends_on_templates:
                    return True
            else:
                assert isinstance(value_node, ParsedStructural)
                if value_node.schema_node is not None and value_node.schema_node.undecided_literal_keywords_for(
                    value_node.value
                ):
                    return True
        return any(expression.is_value_template_dependent for _, expression in self.value.iter_expressions())

    @property
    def is_fully_parsed(self) -> bool:
        """
        Whether every custom-type leaf of this value was walked through to an expression, recursively.

        A leaf with **no** expression counts against this. It used to be skipped, which read as "nothing to
        check here" but means the opposite: a leaf with no expression is precisely a leaf the walk did not
        finish, and it is exactly how an unresolved default site looks
        (see :meth:`ParsedValue.unresolved_default_sites`).

        Nothing here consults template dependence; :attr:`is_template_dependent` is the property for that.

        A value of ``None`` (a default-serialization expression) has no leaves to walk.
        """
        if self.value is None:
            return True
        for value_node in self.value.walk():
            if isinstance(value_node, ParsedCustomValue) and (
                value_node.expression is None or not value_node.expression.is_fully_parsed
            ):
                return False
        return True

    def get_subexpressions(self) -> Iterator[Expression]:
        """Check the self.value ParsedValue type for subexpressions"""
        if self.value is None:
            yield from []
        else:
            yield from (expr for _, expr in self.value.iter_expressions())


class DefaultSerializationExpression(InstExpression):
    kind_name = "default-serialized value"
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
    kind_name = "narrowed value domain instantiation"

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
    cause_location_id: LocationId | None = None
    """
    Where :attr:`cause` was parsed, when that is not where this attempt was made.

    An attempt is reported at the location of the value it tried to parse. Its cause is a *different*
    value -- a Function argument, or the default of one -- parsed at a location of its own, and rendering
    it at the parent's makes every level of a nested failure claim the same place. The chain then names
    the outermost value however deep the real failure is: ``LessEqual<Integer>``'s ``arg1`` reported at
    ``Condition``, three levels above where ``n`` actually is.

    ``None`` where the cause has no location of its own to give, which keeps the parent's.
    """

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
            # The cause's own location, not this attempt's -- see `cause_location_id`.
            causes.extend(self.cause.explanation_causes(self.cause_location_id or location_id))
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
