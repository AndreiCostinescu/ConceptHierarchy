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

"""Parser of Expressions"""

from abc import ABC, abstractmethod
from collections import deque
from contextlib import AbstractContextManager
from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from frozendict import frozendict

from concept_hierarchy.data.contexts.template_context import TemplateContext
from concept_hierarchy.data.expressions.expression import Expression, ExpressionValue
from concept_hierarchy.data.expressions.expression_utils import (
    FunctionArgumentAccessor,
    FunctionArgumentProvenance,
    FunctionResultAccessor,
    ValueDomainArgumentProvenance,
)
from concept_hierarchy.data.expressions.instantiated_value import ParsedValue
from concept_hierarchy.data.expressions.subexpressions import (
    ConstraintGroupAttempt,
    DefaultSerializationExpression,
    ExpressionAttempt,
    ExpressionKind,
    FunctionEvaluation,
    IllFormedExpression,
    InstancePropertyChain,
    InstExpression,
    LiteralTemplateVariableValue,
    NarrowExpression,
    PossibleInstExpression,
    PossibleVariableExpression,
    TemplateDependentExpression,
    Variable,
    VariableWithTemplateType,
    VerifiedTemplateDependentExpression,
)
from concept_hierarchy.data.jsonschema import CHSchemaNode
from concept_hierarchy.data.jsonschema.parsed_schema import LITERAL_KEYWORD_FIELDS
from concept_hierarchy.data.type_template_variables.constraint_formula import (
    ConstraintGroup,
    HierarchyCheckType,
    NonTypeTemplateConstraintFormula,
    TemplateConstraintOr,
)
from concept_hierarchy.data.type_template_variables.template_substitution import (
    substitute,
    substitute_template_variables_in_value,
)
from concept_hierarchy.data.types.concept_hierarchy_types import (
    TYPE_VALUE_IS_INSTANCE_CHECK,
    ConceptHierarchyTemplateArgument,
    ConceptHierarchyVariadicGroup,
    ExpandedVariadicTemplateVariable,
    InstantiatedType,
    LiteralValue,
    NonVariadicTemplateVariable,
    TemplateDependent,
    TemplateDependentType,
    TemplateVariable,
    TypeValue,
)
from concept_hierarchy.data.utils import MISSING
from concept_hierarchy.data.validators.template_argument_constraints_validator import (
    TemplateContextDeterminator,
    TypeTemplateInstantiationValidator,
    validate_template_argument_value_against_constraint,
    validate_type_against_constraint_formula,
)
from concept_hierarchy.definitions.concept_definition_value_domain import ValueDomainDefinition
from concept_hierarchy.definitions.concept_hierarchy import ConceptHierarchyDefinition
from concept_hierarchy.errors import CHSemanticError, ConceptHierarchyError, LocationId, PathPart
from concept_hierarchy.utils import get_items_of_single_entry_dict


@dataclass(frozen=True)
class GroundedArgumentDefault:
    """
    A Function argument's declared default, parsed for **one ground application**.

    Cached on the validator rather than on the declared `Expression`, because the declaration is shared by
    every application and the result is not: `F<Integer>`'s default and `F<String>`'s are different
    expressions from the same text. Keyed by the application, two call sites that evaluate `F<Integer>`
    share this entry, which an expression-local cache could not give them.
    """

    expression: Expression | None
    """``None`` marks an entry that is still being produced -- see `in_progress`."""

    sibling_dependencies: frozenset[str] = frozenset()
    """
    The Function's own arguments this default reads, **as seen once ground**.

    Not the same set as `FunctionData.default_argument_dependencies`, which is collected at definition time
    from a parse with the template variables unbound. That parse stops where the type stops being decidable,
    so a reference nested inside a template-dependent value never becomes a `Variable` and never enters the
    set: with ``arg2: Cell<T>`` defaulting to ``{"b": "arg1"}`` the edge is missing, and with
    ``arg2: Cell<Integer>`` it is there. Grounding is where the rest of the tree finally exists.
    """

    @property
    def in_progress(self) -> bool:
        """Whether this default is on the current grounding path, i.e. grounding it needs itself."""
        return self.expression is None


class ExpressionParserValidator(ABC):
    @abstractmethod
    def is_concept(self, candidate_concept_name: str) -> bool:
        pass

    @abstractmethod
    def is_template_variable(self, candidate_template_variable_name: str) -> bool:
        pass

    @abstractmethod
    def is_literal_template_variable(self, candidate_literal_template_variable_name: str) -> bool:
        pass

    @abstractmethod
    def get_literal_template_var_constraint(self, literal_template_variable_name: str) -> str:
        """Raises an error if the literal template variable name is not valid."""

    @abstractmethod
    def is_variable(self, candidate_variable_name: str) -> bool:
        pass

    @abstractmethod
    def get_variable_type(self, variable_name: str) -> TypeValue:
        pass

    @abstractmethod
    def get_variable_scope_index(self, variable_name: str) -> int:
        """
        The index of the variable stack frame ``variable_name`` resolves in; 0 is the global variables.

        Every `Variable` records it, because the name alone does not say which variable it is: a Function's
        arguments and a nested call's introduce names above the globals and shadow them.
        """

    @abstractmethod
    def is_type_abstract(self, candidate_type: InstantiatedType | TemplateDependentType) -> bool:
        pass

    @abstractmethod
    def is_a_subtype_of_b(self, a: InstantiatedType, b: InstantiatedType, location_id: LocationId) -> bool:
        pass

    @abstractmethod
    def create_instantiated_type(self, instantiated_type_name: str, location_id: LocationId) -> InstantiatedType:
        pass

    @abstractmethod
    def create_possibly_template_dependent_type(self, type_name: str, location_id: LocationId) -> TypeValue:
        pass

    @abstractmethod
    def get_substituted_value_domain_instantiation_schema(
        self, type_name: InstantiatedType | TemplateDependentType
    ) -> TypeValue:
        pass

    @abstractmethod
    def get_substituted_function_interface(
        self, ch_type: InstantiatedType
    ) -> tuple[tuple[str, ...], set[str], InstantiatedType | None, ValueDomainArgumentProvenance | None]:
        pass

    @abstractmethod
    def get_properties_of_concepts(self, concepts: list[str]) -> dict[str, InstantiatedType]:
        pass

    @abstractmethod
    def get_type_of_instance_property(self, instance_type: InstantiatedType, prop_name: str) -> InstantiatedType:
        """
        This must raise a CHSemanticError if:
            - instance_type is not a subtype of InstanceBase
            - prop_name is not a property of the type represented by instance_type

        :param instance_type: the type which is to-be-checked that it is an instance type that has the property
        :param prop_name: the name of the property whose type is to be determined by the function
        :return: the InstantiatedType type of the property `prop_name` of `instance_type`
        """

    @abstractmethod
    def get_if_has_instantiation_schema(
        self, type_data: TypeValue
    ) -> tuple[tuple[ConstraintGroup, CHSchemaNode], ...] | None:
        pass

    @abstractmethod
    def validate_value_against_schema(
        self,
        schema: CHSchemaNode,
        value: object,
        location_id: LocationId,
        template_substitution: dict[str, ConceptHierarchyTemplateArgument] | None,
        expansion_depth: int,
    ) -> tuple[ParsedValue, list[ConceptHierarchyError]]:
        """
        Parse ``value`` against ``schema``, returning the result tree **and** the authoritative error list.

        ``template_substitution`` is the *caller's* mapping, not this schema's: ``value`` is text from the
        enclosing expression, so it names the enclosing concept's template variables however many schemas
        deep it is nested. A schema's own defaults are grounded separately, by :func:`resolve_substituted_defaults`.

        Both are needed: the errors are what an :class:`IllFormedExpression` reports when this value turns
        out not to be a valid instantiation, and they cannot be recovered by walking the tree -- trial
        branches and failed ``allOf`` branches deliberately do not attach their errors to retained nodes.
        """

    @abstractmethod
    def get_default_serialization_concept_name_for(self, json_value_type: str) -> str | None:
        pass

    @abstractmethod
    def get_function_return_interface(
        self, f_name
    ) -> tuple[TypeValue, FunctionResultAccessor, ValueDomainArgumentProvenance] | None:
        pass

    @abstractmethod
    def is_function_argument(self, f_name, f_arg_name) -> bool:
        pass

    @abstractmethod
    def get_function_arguments(self, f_name) -> set[str]:
        pass

    @abstractmethod
    def get_required_function_arguments(self, f_name) -> set[str]:
        pass

    @abstractmethod
    def get_function_argument_interface(
        self, f_name, f_arg_name
    ) -> tuple[TypeValue, FunctionArgumentAccessor, FunctionArgumentProvenance]:
        pass

    @abstractmethod
    def get_default_argument_dependencies(self, f_name) -> frozendict[str, frozenset[str]] | None:
        pass

    @abstractmethod
    def get_function_argument_default_source(self, f_name: str, f_arg_name: str) -> object:
        """
        The JSON a Function argument's default was written as, or `MISSING` when it has none.

        Read from the **definitions**, including inherited ones, rather than from the parsed model, because
        it has to be answerable before the model has it. Parsing a Function's own default can reach an
        evaluation of that same Function -- `MakeT1`'s default instantiating a `T1` whose schema evaluates
        `MakeT1` again -- and at that moment its parsed defaults are empty by construction. Answering
        "no default" there would drop the one edge that closes such a cycle, at the one moment the
        resolution path still holds it.
        """

    @abstractmethod
    def get_parsed_function_argument_default(self, f_name: str, f_arg_name: str) -> Expression | None:
        """
        The *parsed* form of that default, if it has been parsed yet; ``None`` if it has not.

        ``None`` means "not yet", never "no default" -- `get_function_argument_default_source` answers that.
        Without a parsed form there is nothing to shortcut against, so the call site simply reparses.

        A parse recorded by `put_parsed_function_argument_default` answers this too: it is the declaration's
        own parse, produced early, and there is no sense in which it is less parsed than the one the
        declaring Function's pass will store.
        """

    @abstractmethod
    def put_parsed_function_argument_default(self, f_name: str, f_arg_name: str, expression: Expression) -> None:
        """
        Record ``expression`` as the parse of that declared default, for the declaration itself to reuse.

        Only for an expression that **is** what parsing the declaration produces -- see the conditions at
        the one call site. The declaring Function's own pass then has nothing left to do for that argument,
        which is the point: without this it reparses the same source into an identical tree.

        Whether to keep the entry is the implementation's to decide; a Function that only *inherits* the
        default is not the one whose pass will look for it.
        """

    @abstractmethod
    def function_argument_scope(self, arguments: dict[str, TypeValue]) -> AbstractContextManager[None]:
        """
        Run with the variable scope **replaced** by the global variables plus these Function arguments.

        Replaced, not extended. A default may name a sibling, so the Function's arguments have to be in
        scope -- but everything *between* the global frame and them must be out of it, because the parser
        classifies a bare string as a variable before it considers anything else. Leaving an enclosing
        Function's arguments visible lets one of its names capture a string that is meant to be a value:
        `Inner`'s default ``"leak"`` stops being a (mistyped) `String` and silently becomes `Outer`'s
        `leak` argument.
        """

    @abstractmethod
    def get_grounded_function_default(self, application: str, argument: str) -> GroundedArgumentDefault | None:
        """
        The cached grounding of one argument's default for one application, if there is one.

        An entry whose `GroundedArgumentDefault.in_progress` is set means this default is on the current
        grounding path -- a default that evaluates its own Function and leaves the same argument unsupplied
        again -- which is the on-path check that instantiation defaults use, keyed by the ground
        application for the reason node identity is (D2): `F<Integer>` may terminate where `F<Box<Integer>>`
        does not.
        """

    @abstractmethod
    def put_grounded_function_default(self, application: str, argument: str, grounded: GroundedArgumentDefault) -> None:
        """
        Record a grounding, or (with `GroundedArgumentDefault.in_progress`) that one has begun.
        Overwrites existing content.
        """

    @abstractmethod
    def get_template_context(self, type_name_clean) -> TemplateContext:
        pass

    @abstractmethod
    def get_type_template_instantiation_validator(self) -> TypeTemplateInstantiationValidator:
        pass

    @abstractmethod
    def register_default_site(
        self,
        node: CHSchemaNode,
        template_substitution: dict[str, ConceptHierarchyTemplateArgument] | None,
        expansion_depth: int,
    ) -> None:
        """
        Record how ``node``'s default is to be parsed, without parsing it: the substitution of the
        application whose schema ``node`` belongs to, and how many *applications* deep that schema is.

        The depth deliberately counts application builds rather than the length of the resolution path,
        because only one of those can run away. Resolution visits each site at most once -- it is memoized
        as soon as it is parsed, and re-entering one still on the path is reported as a cycle -- so a chain
        *within* the schemas already built is bounded by the number of their default sites, and is finite
        however long it gets. What is not bounded is a path that keeps *generating* applications, each
        with a schema and default sites of its own, and that is what the depth limit exists to stop.
        """

    @abstractmethod
    def resolve_default_site(self, node: CHSchemaNode) -> Expression | None:
        """
        ``node``'s default expression, parsing it now if it has not been parsed yet.

        Returns ``None`` for a node that was never registered, which is any node of a *declared* schema
        rather than of one resolved for a ground application.

        A node reached while it is **already being resolved** is a genuine expansion cycle, and gets an
        `IllFormedExpression` saying so. That is the whole cycle check: being *unresolved* is not being
        cyclic, and the two are only distinguishable by whether the node is on the current resolution path.
        """

    @abstractmethod
    def get_resolved_instantiation_schema(self, cache_key: tuple[str, int]) -> CHSchemaNode | None:
        """
        The instantiation schema already substituted and resolved for one ground application, if it has
        been built. ``cache_key`` is ``(application.full_name, constraint group index)``.

        There is only **one** cache here, not two. The per-application `parsed_default_expr` lives on the
        nodes of the cached copy, so memoizing the schema memoizes the resolved defaults with it.
        """

    @abstractmethod
    def put_resolved_instantiation_schema(self, cache_key: tuple[str, int], schema: CHSchemaNode) -> None:
        """
        Record a resolved schema, **before** its defaults are resolved.

        Publishing it early is deliberate: resolving a default re-enters the parser, and a default that
        comes back round to this same application has to find this entry rather than build a second one.
        """

    @abstractmethod
    def get_default_expansion_depth_limit(self) -> int:
        """
        How many nested *default* expansions are allowed before the parse is rejected.

        Materialising one instantiation default can force materialising another, and that can grow without
        repeating -- a cycle is not the only way for it to fail to terminate -- so the recursion needs a
        bound that is not cycle detection. See ``TODO_DEFAULT_EXPANSION_CYCLES.md`` §9a.
        """


def build_template_substitution(
    template_context_of_concept: TemplateContext, expr_type: TypeValue
) -> dict[str, ConceptHierarchyTemplateArgument]:
    """The concept's template variables bound to the arguments of one of its ground applications."""
    return {
        t_arg_name: t_arg_value
        for t_arg_name, t_arg_value in zip(template_context_of_concept.variables, expr_type.template_arguments)
    }


def _substitute_literal_keywords(
    node: CHSchemaNode, template_substitution: dict[str, ConceptHierarchyTemplateArgument]
) -> CHSchemaNode:
    """
    Put a keyword that was written as a literal template variable back into the schema, now that its value is known.

    ``node`` must already be a copy -- the caller hands one over -- so the declaration is never written to.

    A keyword the mapping does not cover is left as it is: a partial substitution has not decided it yet,
    and the field is what records that it is still waiting. Only the value being *known* moves it into
    `shallow_canonical`, which is what `parse_value` hands to the Draft-07 validator.
    """
    for field_name, (keyword, _applies_to) in LITERAL_KEYWORD_FIELDS.items():
        declared = getattr(node, field_name)
        if declared is None:
            continue
        substituted = template_substitution.get(declared)
        if not isinstance(substituted, LiteralValue):
            continue
        value = substituted.convert_to_value()
        setattr(node, field_name, None)
        node.extra_keywords[keyword] = value
        # Only a custom-type node or a boolean schema gets a non-dict `shallow_canonical`, and neither can
        # carry one of these keywords -- `jsonschema_parser` returns before the keyword loop for both.
        assert isinstance(node.shallow_canonical, dict), f"{node.location_id} has no keyword dict to write to"
        node.shallow_canonical[keyword] = value
    return node


def substitute_schema(
    instantiation_schema: CHSchemaNode,
    template_context_of_concept: TemplateContext,
    expr_type: TypeValue,
    constraint_validator: TypeTemplateInstantiationValidator,
    location_id: LocationId,
) -> CHSchemaNode:
    """
    The instantiation schema of ``expr_type``'s concept, with ``expr_type``'s template arguments
    substituted into the type of every custom-type node **and** into every keyword that was written as a
    literal template variable (``{"minItems": "N"}``).

    Types and literal keywords only. Grounding the ``default`` *expressions* of the result is
    :func:`resolve_substituted_defaults`, which is a separate step because it re-enters the expression
    parser -- it must be driven from `_check_instantiation_schema`, where it can be bounded and (in
    stage 2) memoized per application, rather than from the middle of a schema walk.

    Returns ``instantiation_schema`` itself when there is nothing to substitute, so callers must not
    mutate the result without checking that a substitution actually happened.
    """
    template_substitution = build_template_substitution(template_context_of_concept, expr_type)
    if not template_substitution:
        return instantiation_schema

    def _parse_and_substitute(node: CHSchemaNode) -> CHSchemaNode:
        if node.is_boolean_schema:
            return node
        if not node.is_custom_type:
            # `apply` returns a copy with copied containers, so writing the keywords into it is safe.
            return _substitute_literal_keywords(node.apply(_parse_and_substitute), template_substitution)
        assert node.custom_type is not None
        # No literal keywords here: `jsonschema_parser` builds a custom-type node in
        # `_finish_custom_type_node`, which never reaches the keyword loop in `_finish_builtin_node`, so
        # such a node carries no `*_def` field to substitute.
        res = copy(node)
        # substitute
        subst_res = substitute_template_variables_in_value(
            node.custom_type,
            template_substitution,
            template_context_of_concept,
            TemplateContext(),
            constraint_validator,
            location_id + node.location_id,
        )
        assert isinstance(subst_res, TYPE_VALUE_IS_INSTANCE_CHECK)

        res.custom_type = subst_res
        assert res.custom_type is not None
        return res

    return instantiation_schema.apply(_parse_and_substitute)


def _copy_schema(node: CHSchemaNode) -> CHSchemaNode:
    """A fresh tree with the same content, so that resolving defaults never writes to the declaration."""
    if node.is_boolean_schema:
        return node
    return node.apply(_copy_schema)


def expansion_cycle_expression(node: CHSchemaNode) -> Expression:
    """
    What a default site resolves to when expanding it comes back round to needing itself.

    Reported by `_parse_custom` at the point of *materialisation*, which is exactly right: a value that
    **supplies** the key never reads it and stays legal (D1). The message names the site and its type
    rather than only the concept, because one declared site is reached under several ground applications
    and may be cyclic under only some of them.
    """
    site = node.location_id[-2] if len(node.location_id) >= 2 else node.location_id[-1]
    return Expression(
        node.custom_type,
        node.provenance,
        FunctionArgumentAccessor.GET,
        node.default_expr,
        IllFormedExpression(
            f'the default of "{site}" ({node.custom_type}) can never be applied -- materialising it '
            f"requires materialising it again, so no finite value satisfies it"
        ),
    )


def build_resolved_instantiation_schema(
    declared_schema: CHSchemaNode,
    template_context_of_concept: TemplateContext,
    expr_type: InstantiatedType,
    constraint_validator: TypeTemplateInstantiationValidator,
    validator: ExpressionParserValidator,
    location_id: LocationId,
    cache_key: tuple[str, int],
    expansion_depth: int,
) -> CHSchemaNode:
    """
    The instantiation schema of ``expr_type``, substituted and with every ``default`` resolved, memoized
    per ``cache_key``.

    Templated and non-templated applications go through the same path -- the non-templated case merely has
    an empty substitution, and is copied rather than substituted. Before this existed, a non-templated
    ValueDomain's defaults were only ever parsed at definition time, and were therefore never checked for
    the expansion cycles below.
    """
    own_substitution = build_template_substitution(template_context_of_concept, expr_type)
    if own_substitution:
        resolved = substitute_schema(
            declared_schema, template_context_of_concept, expr_type, constraint_validator, location_id
        )
    else:
        resolved = _copy_schema(declared_schema)

    # Published before the defaults are resolved: see `put_resolved_instantiation_schema`.
    validator.put_resolved_instantiation_schema(cache_key, resolved)

    # Registered, not resolved. A default is parsed the first time something materialises it, which is
    # what makes "already being resolved" (a cycle) distinguishable from "not resolved yet" (merely not
    # reached). Resolving everything up front cannot tell those apart, and every site would then need
    # re-checking once its siblings were done.
    for node in resolved.walk():
        if node.has_default and isinstance(node.custom_type, InstantiatedType):
            # The copy inherited whatever the *declared* node was parsed to, which was parsed without this
            # application's substitution. Drop it so it can not be mistaken for a resolved value.
            node.parsed_default_expr = None
            validator.register_default_site(node, own_substitution or None, expansion_depth + 1)
    return resolved


def parse_expression(
    json_value: object,
    expr_type: TypeValue,
    expr_provenance: FunctionArgumentProvenance | ValueDomainArgumentProvenance,
    expr_access: FunctionArgumentAccessor | FunctionResultAccessor,
    expr_template_context: TemplateContext,
    validator: ExpressionParserValidator,
    location_id: LocationId,
    parse_template_expressions_without_type_checks: bool = False,
    template_substitution: dict[str, ConceptHierarchyTemplateArgument] | None = None,
    expansion_depth: int = 0,
) -> Expression:
    """
    ``template_substitution`` is ``None`` in the ordinary case. It is set when this expression is being
    reparsed under a ground type application, and maps the enclosing concept's template variables to that
    application's arguments; it is applied wherever a type or a literal is created *from source text*.

    ``expansion_depth`` counts nested *default* expansions only -- supplied values are bounded by the
    finite JSON they came from, defaults are not.
    """
    depth_limit = validator.get_default_expansion_depth_limit()
    if expansion_depth > depth_limit:
        raise CHSemanticError(
            f"Default expansion is more than {depth_limit} levels deep at {expr_type}, and is still "
            f"producing new values. Either it does not terminate, or the limit is too low -- raise "
            f'"{ConceptHierarchyDefinition.metadata_expansion_depth_limit_for_default_instantiation_expressions}" in '
            f"the Concept Hierarchy metadata.",
            location_id=location_id,
            part=PathPart.VALUE,
        )
    if isinstance(expr_provenance, ValueDomainArgumentProvenance):
        expr_provenance = (
            FunctionArgumentProvenance.ADDR
            if expr_provenance == ValueDomainArgumentProvenance.ADDR
            else FunctionArgumentProvenance.ANY
        )
    if isinstance(expr_access, FunctionResultAccessor):
        expr_access = (
            FunctionArgumentAccessor.GET if expr_access == FunctionResultAccessor.GET else FunctionArgumentAccessor.MOD
        )
    try:
        expr_candidate_value = _parse_syntax_of_expression(
            json_value,
            expr_type,
            expr_template_context,
            validator,
            location_id,
            True,
            parse_template_expressions_without_type_checks,
            template_substitution,
            expansion_depth,
        )
    except RecursionError:
        # The interpreter's own stack runs out long before `depth_limit` does -- one expansion level costs
        # a dozen-odd Python frames -- and a bare RecursionError says nothing about the hierarchy. Convert
        # it here rather than where the depth is counted: by the time it reaches an outermost call the
        # frames below have unwound, so building this message is safe, which it would not be deeper down.
        if expansion_depth != 0:
            raise
        raise CHSemanticError(
            f"Ran out of stack while parsing this expression of type {expr_type}. Either a value is nested "
            f"extremely deeply, or its instantiation defaults expand without terminating; the configured "
            f'"{ConceptHierarchyDefinition.metadata_expansion_depth_limit_for_default_instantiation_expressions}" of '
            f"{depth_limit} was never reached, so it is above what the interpreter can support.",
            location_id=location_id,
            part=PathPart.VALUE,
        ) from None
    is_strict_subtype = expr_candidate_value.is_strict_subtype
    is_addressable = isinstance(expr_candidate_value, Variable)
    if not is_addressable and isinstance(expr_candidate_value, FunctionEvaluation):
        is_addressable = expr_candidate_value.is_result_addressable
    if expr_provenance != FunctionArgumentProvenance.ANY and not is_addressable:
        # The expression parsed, but not into something that can be addressed. Say what it *is*: a bare
        # "got False" leaves the reader to work out which of the alternatives matched.
        expr_candidate_value = IllFormedExpression(
            f"Provenance violation: {expr_provenance.value} provenance requires an addressable expression "
            f"(a variable, an instance property chain, or a Function evaluation whose result is "
            f"{ValueDomainArgumentProvenance.ADDR.value}); got a "
            f"{_describe_expression_kind(expr_candidate_value)} of type {expr_candidate_value.value_type}"
        )
    elif expr_access != FunctionArgumentAccessor.GET and is_strict_subtype:
        expr_candidate_value = IllFormedExpression(
            f"Access violation: {expr_access.value} access requires the exact type {expr_type}, but this "
            f"{_describe_expression_kind(expr_candidate_value)} has type {expr_candidate_value.value_type}, "
            f"which is a strict subtype"
        )

    expression = Expression(expr_type, expr_provenance, expr_access, json_value, expr_candidate_value)
    return expression

    # TODO: check the types // semantic of the expression:
    #  - check expr_ref, expr_mod, is_strict_subtype!
    #  - check that "isFunctionEvaluation" is used correctly
    #  - check that Function result types match the expected expression type
    #  -- subexpressions (i.e. the evaluation arguments) should already be sub-checked in the syntax-above!)
    #  - check that ValueDomain instantiations are subtypes of the expected expression type
    #  -- subexpressions thereof were already checked when they were parsed
    #  - check that the variables (literal template variables, code variables, and instance prop chains)
    #    match the expected expression type
    #  -- for literal template variables check that the literal constraint type is a registered defaultSerialization
    #     somewhere; if it is not registered, then it can't be used!
    #     if it is registered, interpret the template variable value as the type that registers
    raise NotImplementedError


_EXPRESSION_KIND_NAMES: tuple[tuple[type, str], ...] = (
    # Most specific first: NarrowExpression and DefaultSerializationExpression subclass InstExpression,
    # InstancePropertyChain subclasses Variable, so a plain isinstance sweep in the wrong order reports the base class.
    (NarrowExpression, "narrowed value domain instantiation"),
    (DefaultSerializationExpression, "default-serialized value"),
    (InstExpression, "value domain instantiation"),
    (FunctionEvaluation, "Function evaluation"),
    (InstancePropertyChain, "instance property chain"),
    (LiteralTemplateVariableValue, "literal template variable"),
    (Variable, "variable"),
)


def _describe_expression_kind(expr_value: ExpressionValue) -> str:
    """A reader-facing name for what an expression turned out to be, for provenance/access messages."""
    for kind, name in _EXPRESSION_KIND_NAMES:
        if isinstance(expr_value, kind):
            return name
    return type(expr_value).__name__


def get_expression_type(
    json_value: object,
    expr_type: TypeValue,
    expr_template_context: TemplateContext,
    validator: ExpressionParserValidator,
    location_id: LocationId,
) -> TypeValue | None:
    return _parse_syntax_of_expression(
        json_value, expr_type, expr_template_context, validator, location_id, recursively_parse=False
    ).value_type


def _parse_syntax_of_expression(
    json_value: object,
    expr_type: TypeValue,
    expr_template_context: TemplateContext,
    validator: ExpressionParserValidator,
    location_id: LocationId,
    recursively_parse: bool = True,
    parse_template_expressions_without_type_checks: bool = False,
    template_substitution: dict[str, ConceptHierarchyTemplateArgument] | None = None,
    expansion_depth: int = 0,
) -> ExpressionValue:
    # A literal template variable stands for a value, not a type, so it is substituted in the JSON itself
    # and then interpreted from scratch -- under `N := 3` the string "N" *becomes* the literal 3, and the
    # expression changes class from LiteralTemplateVariableValue to InstExpression.
    value_is_a_substituted_literal = False
    if template_substitution is not None and isinstance(json_value, str):
        substituted_literal: ConceptHierarchyTemplateArgument | None = template_substitution.get(json_value)
        if isinstance(substituted_literal, LiteralValue):
            json_value = substituted_literal.convert_to_value()
            value_is_a_substituted_literal = True

    assert isinstance(expr_type, TYPE_VALUE_IS_INSTANCE_CHECK)
    if isinstance(expr_type, (ConceptHierarchyVariadicGroup, LiteralValue, ExpandedVariadicTemplateVariable)):
        raise RuntimeError(f"Can't parse an expression of type {expr_type}")

    is_expr_type_template_variable = isinstance(expr_type, TemplateVariable)
    if is_expr_type_template_variable:
        assert expr_template_context.has_template_variable(expr_type.clean_name), expr_type.full_name
    is_expr_type_template_containing = isinstance(expr_type, TemplateDependentType)
    is_expr_type_ground = isinstance(expr_type, InstantiatedType)
    assert is_expr_type_template_variable + is_expr_type_template_containing + is_expr_type_ground == 1

    if not parse_template_expressions_without_type_checks and not is_expr_type_ground:
        return TemplateDependentExpression()
        # TODO: how do I select the correct instantiation formula from the list of template-constraints?
        #   - if the expression type is fully instantiated => verify constraints => check first matching instantiation
        #   - if the expression type depends on templates:
        #       - if there is a single instantiation (i.e. no template-dependent instantiation) => validate against it
        #       - if there are multiple instantiations
        #           - if at least one instantiation matches the constraints => do not verify the constraints yet
        #             (add the list of possible instantiation schemas, the
        #           - if no instantiation could ever match the constraints => proceed as if there is no instantiation
        #             defined, but signal that there are instantiation schemas that do not match the expected type
        # TODO: validate literal formula; if formula is not validated -> raise CHSemanticError

    is_function_evaluation, is_function_evaluation_present, len_content_keys = True, False, None
    # compute the amount of **content keys** in the JSON object
    if isinstance(json_value, dict):
        len_content_keys = len(json_value)
        is_function_evaluation_present = "isFunctionEvaluation" in json_value
        if is_function_evaluation_present:
            len_content_keys -= 1
            # remove the `"isFunctionEvaluation"` key from `json_value` ONLY in a Narrow/FEval expression type!
            is_function_evaluation = json_value["isFunctionEvaluation"]

    def ensure_unmodified_json_value(
        _value: object, _is_function_evaluation_present: bool, _is_function_evaluation_value: bool
    ) -> None:
        if isinstance(_value, dict) and _is_function_evaluation_present and "isFunctionEvaluation" not in _value:
            _value["isFunctionEvaluation"] = _is_function_evaluation_value

    try:
        expr_value_res = _parse_syntax_of_expression_with_instantiated_type(
            json_value,
            expr_type,
            expr_template_context,
            validator,
            location_id,
            recursively_parse,
            parse_template_expressions_without_type_checks,
            is_function_evaluation,
            is_function_evaluation_present,
            len_content_keys,
            ensure_unmodified_json_value,
            template_substitution,
            expansion_depth,
            value_is_a_substituted_literal,
        )
        ensure_unmodified_json_value(json_value, is_function_evaluation_present, is_function_evaluation)
        assert isinstance(expr_type, TemplateVariable) or len(expr_value_res) == 1
        if isinstance(expr_type, TemplateVariable):
            if len(expr_value_res) == 1:
                expr_res = expr_value_res[0]
            else:
                expr_res = VerifiedTemplateDependentExpression(expr_value_res)
        else:
            expr_res = expr_value_res[0]
        return expr_res
    except ConceptHierarchyError as e:
        ensure_unmodified_json_value(json_value, is_function_evaluation_present, is_function_evaluation)
        raise e


def _can_be_subtype_of_instantiated(
    type_to_be_checked: TypeValue,
    instantiated_type: InstantiatedType,
    validator: ExpressionParserValidator,
    template_context: TemplateContext,
    location_id: LocationId,
) -> bool:
    """
    Whether ``type_to_be_checked`` -- possibly a template variable or a template-dependent type such as
    ``Add<T>`` -- can be a subtype of the instantiated ``instantiated_type``.

    Examples: ``Add<T>`` is a subtype of ``Function``, and ``Increment<T>`` is a subtype of ``Add<Number>``
    if and only if ``T`` is ``Number``.
    """
    return _check_if_subtype(validator, type_to_be_checked, instantiated_type, template_context, location_id)


@dataclass(frozen=True)
class InstantiationSearch:
    """
    The outcome of searching a type's ``instantiation`` for a group that accepts a value.

    ``parsed`` is ``None`` when the type declares no instantiation schema at all (an abstract type).
    ``groups`` records every group that was tried, matched or not, so that a failure can say *why* --
    which constraints the type application did not satisfy, and how the one it did satisfy rejected the
    value.
    """

    parsed: ParsedValue | None
    errors: tuple[ConceptHierarchyError, ...] = ()
    groups: tuple[ConstraintGroupAttempt, ...] = ()


def _parse_syntax_of_expression_with_instantiated_type(
    json_value: object,
    expr_type: TypeValue,
    expr_template_context: TemplateContext,
    validator: ExpressionParserValidator,
    location_id: LocationId,
    recursively_parse: bool,
    parse_template_expressions_without_type_checks: bool,
    is_function_evaluation: bool,
    is_function_evaluation_present: bool,
    len_content_keys: int,
    ensure_unmodified_json_value: Callable[[object, bool, bool], None],
    template_substitution: dict[str, ConceptHierarchyTemplateArgument] | None = None,
    expansion_depth: int = 0,
    value_is_a_substituted_literal: bool = False,
) -> list[ExpressionValue]:
    expressions_res: list[ExpressionValue] = []
    attempts: list[ExpressionAttempt] = []
    """Every alternative that was applicable to this value and was rejected; see IllFormedExpression."""

    def ensure_expression_invariant(_expressions_res: list[ExpressionValue], _expr_type: TypeValue) -> bool | None:
        assert isinstance(_expr_type, TYPE_VALUE_IS_INSTANCE_CHECK)
        # FIXME: Can TemplateDependentTypes contain multiple expression results?
        if isinstance(_expr_type, InstantiatedType):
            assert len(_expressions_res) <= 1
            return len(_expressions_res) == 1
        return None

    # check Narrow and FEval expressions
    if len_content_keys == 1:
        assert isinstance(json_value, dict)
        if is_function_evaluation_present:
            json_value.pop("isFunctionEvaluation")
        key, value = get_items_of_single_entry_dict(json_value)
        # TODO: verify if key is a Concept Hierarchy-specific type:
        #  - a (non-literal) Template Variable and
        #  - a Function or ValueDomain type application
        is_concept_hierarchy_expression = not key.startswith("s:")
        if validator.is_template_variable(key) and validator.is_literal_template_variable(key):
            is_concept_hierarchy_expression = False

        if is_concept_hierarchy_expression:
            expressions_res.extend(
                parse_expression_of_json_object(
                    key,
                    value,
                    expr_type,
                    expr_template_context,
                    validator,
                    location_id,
                    recursively_parse,
                    parse_template_expressions_without_type_checks,
                    is_function_evaluation,
                    is_function_evaluation_present,
                    ensure_expression_invariant,
                    attempts,
                    template_substitution,
                    expansion_depth,
                )
            )
            if ensure_expression_invariant(expressions_res, expr_type):
                return expressions_res

        ensure_unmodified_json_value(json_value, is_function_evaluation_present, is_function_evaluation)
    # check Var expression
    #
    # A string that came from substituting a literal template variable is skipped here: it is a *value*,
    # and the alternatives below all interpret a string as a *name*. Letting them run means a string
    # literal whose text happens to match something in scope stops being that string -- `H<"one">` with a
    # variable `one` around parses as that variable, taking its type with it, and a dotted literal like
    # `"a.b"` is read as an instance property chain. A literal is classified by what it *is*, so only the
    # instantiation and defaultSerialization alternatives below apply to it.
    if isinstance(json_value, str) and not value_is_a_substituted_literal:
        assert expressions_res == []
        # Prioritize variables over template variables if there is a name clash!
        if validator.is_variable(json_value):
            if validator.is_template_variable(json_value):
                print(
                    f'[CH Warning] Prioritize variable "{json_value}" over template variable "{json_value}" in '
                    f"expression {json_value!r}"
                )
            var_type = validator.get_variable_type(json_value)
            var_scope = validator.get_variable_scope_index(json_value)
            if not isinstance(expr_type, InstantiatedType):
                return [PossibleVariableExpression(json_value, var_type, scope_index=var_scope)]
            elif isinstance(var_type, TemplateDependent):
                return [VariableWithTemplateType(json_value, var_type, scope_index=var_scope)]
            assert isinstance(var_type, InstantiatedType), f"{var_type} of type {str(type(var_type))}"
            if _check_if_subtype(validator, var_type, expr_type, expr_template_context, location_id):
                return [Variable(json_value, var_type, var_type != expr_type, scope_index=var_scope)]
            else:
                reason = f"Type {var_type} of variable {json_value} is not a subtype of {expr_type}!"
                attempts.append(ExpressionAttempt(ExpressionKind.VARIABLE, reason, tried_type=var_type))
                return [IllFormedExpression(reason, tuple(attempts))]
        else:
            possible_instance_property_chain = json_value.split(".")
            if len(possible_instance_property_chain) <= 1 or not validator.is_variable(
                possible_instance_property_chain[0]
            ):
                # Not a variable, and not a chain rooted at one. Record both, so that a string matching
                # nothing says which kinds of name were looked for rather than only that it matched none.
                attempts.append(
                    ExpressionAttempt(
                        ExpressionKind.VARIABLE, f'"{json_value}" is not a variable of this Concept Hierarchy'
                    )
                )
                if len(possible_instance_property_chain) > 1:
                    attempts.append(
                        ExpressionAttempt(
                            ExpressionKind.INSTANCE_PROPERTY_CHAIN,
                            f'"{possible_instance_property_chain[0]}" is not a variable, so "{json_value}" is not '
                            f"an instance property chain",
                        )
                    )
                if not validator.is_template_variable(json_value):
                    attempts.append(
                        ExpressionAttempt(
                            ExpressionKind.LITERAL_TEMPLATE_VARIABLE,
                            f'"{json_value}" is not a template variable in this context',
                        )
                    )
            if len(possible_instance_property_chain) > 1 and validator.is_variable(possible_instance_property_chain[0]):
                # Validate that the instance property chain is actually a property chain.
                types_in_property_chain = [validator.get_variable_type(possible_instance_property_chain[0])]
                # The chain is rooted at its first name, so that is the reference whose scope it has.
                var_scope = validator.get_variable_scope_index(possible_instance_property_chain[0])
                for prop in possible_instance_property_chain[1:]:
                    # The below raises a CHSemanticError if:
                    #  - instance_type is not a subtype of InstanceBase
                    #  - prop_name is not a property of the type represented by instance_typ
                    prop_type = validator.get_type_of_instance_property(types_in_property_chain[-1], prop)
                    types_in_property_chain.append(prop_type)
                var_type = types_in_property_chain[-1]
                assert isinstance(var_type, InstantiatedType)
                if isinstance(expr_type, TemplateDependent):
                    return [PossibleVariableExpression(json_value, var_type, scope_index=var_scope)]
                if _check_if_subtype(validator, var_type, expr_type, expr_template_context, location_id):
                    return [
                        InstancePropertyChain(
                            possible_instance_property_chain,
                            types_in_property_chain,
                            var_type != expr_type,
                            scope_index=var_scope,
                        )
                    ]
                else:
                    reason = f"Type {var_type} of instance property chain {json_value} is not a subtype of {expr_type}!"
                    attempts.append(
                        ExpressionAttempt(ExpressionKind.INSTANCE_PROPERTY_CHAIN, reason, tried_type=var_type)
                    )
                    return [IllFormedExpression(reason, tuple(attempts))]
            elif validator.is_template_variable(json_value):
                if not validator.is_literal_template_variable(json_value):
                    raise CHSemanticError(
                        f"Can not use a type template variable in an expression as a variable. Found {json_value}",
                        location_id=location_id,
                    )
                # check to see whether the template parameter's literal type is a registered defaultSerialization!
                literal_constraint_type = validator.get_literal_template_var_constraint(json_value)
                assert literal_constraint_type in NonTypeTemplateConstraintFormula.ALL_CONSTRAINT_TYPES
                match literal_constraint_type:
                    case NonTypeTemplateConstraintFormula.BOOLEAN:
                        value_type_str = "boolean"
                    case NonTypeTemplateConstraintFormula.INTEGER:
                        value_type_str = "integer"
                    case NonTypeTemplateConstraintFormula.NUMBER:
                        value_type_str = "number"
                    case NonTypeTemplateConstraintFormula.STRING:
                        value_type_str = "string"
                    case _:
                        raise RuntimeError(
                            f'Unknown constraint type "{literal_constraint_type}" of template variable "{json_value}"'
                        )
                type_name_str = validator.get_default_serialization_concept_name_for(value_type_str)
                if type_name_str is not None:
                    ch_value_type = validator.create_instantiated_type(type_name_str, location_id)
                    assert ch_value_type is not None
                    if _check_if_subtype(validator, ch_value_type, expr_type, expr_template_context, location_id):
                        return [LiteralTemplateVariableValue(json_value, ch_value_type, ch_value_type != expr_type)]
                    else:
                        reason = (
                            f"The type of the literal template variable {json_value} (matched via default "
                            f"serialization to {ch_value_type}) is not a subtype of {expr_type}!"
                        )
                        attempts.append(
                            ExpressionAttempt(
                                ExpressionKind.LITERAL_TEMPLATE_VARIABLE, reason, tried_type=ch_value_type
                            )
                        )
                        return [IllFormedExpression(reason, tuple(attempts))]
                else:
                    reason = (
                        f'The literal constraint "{literal_constraint_type}" of "{json_value}" does not have a '
                        f'matching registered "{ValueDomainDefinition.value_domain_default_serialization}" '
                        f"({value_type_str})!"
                    )
                    attempts.append(ExpressionAttempt(ExpressionKind.LITERAL_TEMPLATE_VARIABLE, reason))
                    return [IllFormedExpression(reason, tuple(attempts))]

    # check Inst expression (abstract Types do not have instantiation schemas)
    #
    # Skipped for a substituted literal, like the name alternatives above: a literal template argument is
    # recognised *only* by defaultSerialization. Letting it match an instantiation schema as well would
    # make the same literal mean different things at different sites -- `"s:hello"` would be a `String`
    # built from String's schema at one site and a defaultSerialized `String` at another -- and would give
    # a literal a structural reading it was never meant to have.
    if value_is_a_substituted_literal:
        pass
    elif isinstance(expr_type, TemplateVariable):
        expressions_res.append(PossibleInstExpression())
    else:
        inst_res = _check_instantiation_schema(
            json_value,
            expr_type,
            expr_template_context,
            validator,
            location_id,
            template_substitution,
            expansion_depth,
        )
        if inst_res.parsed is not None and inst_res.parsed.is_valid():
            # Whether this instantiation is *decided* (= fully parsed and not template-dependent) is not settled here.
            # A value, that met a node whose keyword is still an unbound literal template variable, passed that node
            # unchecked, and `InstExpression.is_template_dependent` reads that off the parsed value.
            # Template dependence is a property of what was parsed, not of which class was chosen.
            # Saving this value as `InstExpression` also keeps the parse tree, which `PossibleInstExpression` discards.
            expressions_res.append(InstExpression(inst_res.parsed, expr_type, True))
            if ensure_expression_invariant(expressions_res, expr_type):
                return expressions_res
        attempts.append(_instantiation_attempt(ExpressionKind.INSTANTIATION, expr_type, inst_res))

    # check DS (default serialization) expression (abstract Types do not have a defaultSerialization)
    value_type_str = get_json_type_as_string(json_value, location_id)
    type_name_str = validator.get_default_serialization_concept_name_for(value_type_str)
    if type_name_str is None:
        attempts.append(
            ExpressionAttempt(
                ExpressionKind.DEFAULT_SERIALIZATION,
                f'no concept of this Concept Hierarchy registers a "'
                f'{ValueDomainDefinition.value_domain_default_serialization}" for the JSON type '
                f'"{value_type_str}"',
            )
        )
    else:
        ch_value_type = validator.create_instantiated_type(type_name_str, location_id)
        assert ch_value_type is not None
        if isinstance(expr_type, TemplateVariable):
            expressions_res.append(PossibleInstExpression(ch_value_type))
        elif isinstance(expr_type, InstantiatedType) and _check_if_subtype(
            validator, ch_value_type, expr_type, expr_template_context, location_id
        ):
            expressions_res.append(
                DefaultSerializationExpression(ch_value_type, json_value, ch_value_type != expr_type)
            )
            if ensure_expression_invariant(expressions_res, expr_type):
                return expressions_res
        else:
            attempts.append(
                ExpressionAttempt(
                    ExpressionKind.DEFAULT_SERIALIZATION,
                    f'the JSON type "{value_type_str}" serializes to {ch_value_type}, which is not a subtype '
                    f"of {expr_type}",
                    tried_type=ch_value_type,
                )
            )

    if isinstance(expr_type, InstantiatedType) and expressions_res == []:
        expressions_res.append(
            IllFormedExpression(
                f"Could not match a valid {expr_type} expression to value {json_value}", tuple(attempts)
            )
        )
    return expressions_res


def _instantiation_attempt(
    kind: ExpressionKind, tried_type: TypeValue, search: InstantiationSearch
) -> ExpressionAttempt:
    """Turn a failed :func:`_check_instantiation_schema` search into one attempt of the explanation trace."""
    if search.parsed is None:
        return ExpressionAttempt(kind, f"{tried_type} is abstract: it declares no instantiation schema", tried_type)
    # `search.errors` is the matched group's error list, which its own ConstraintGroupAttempt already
    # carries -- passing it as `schema_errors` too would render every schema error twice.
    return ExpressionAttempt(
        kind,
        f"the value does not satisfy the instantiation schema of {tried_type}",
        tried_type=tried_type,
        constraint_groups=search.groups,
    )


def _check_instantiation_schema(
    expr_value: object,
    expr_type: InstantiatedType,
    expr_template_context: TemplateContext,
    validator: ExpressionParserValidator,
    location_id: LocationId,
    template_substitution: dict[str, ConceptHierarchyTemplateArgument] | None = None,
    expansion_depth: int = 0,
) -> InstantiationSearch:
    instantiation_schema = validator.get_if_has_instantiation_schema(expr_type)
    if instantiation_schema is None or len(instantiation_schema) == 0:
        assert validator.is_type_abstract(expr_type), (
            f'It can\'t be that there is no instantiation schema defined for a non-abstract ValueDomain "{expr_type}"!'
        )
        return InstantiationSearch(None)
    groups: list[ConstraintGroupAttempt] = []
    for i, (type_application_constraint, schema_to_match) in enumerate(instantiation_schema):
        type_template_instantiation_validator = validator.get_type_template_instantiation_validator()
        found_matching_schema = type_application_constraint is None
        constraint_errors: list[ConceptHierarchyError] = []
        if not found_matching_schema:
            constraint_errors = validate_type_against_constraint_formula(
                type_application_constraint,
                expr_type,
                TemplateContextDeterminator(expr_template_context),
                type_template_instantiation_validator,
                location_id,
            )
            found_matching_schema = len(constraint_errors) == 0
        if not found_matching_schema:
            groups.append(
                ConstraintGroupAttempt(type_application_constraint, matched=False, errors=tuple(constraint_errors))
            )
            continue
        # Substitute this application's template arguments and resolve the schema's own defaults, once.
        expr_type_template_context = validator.get_template_context(expr_type.clean_name)
        cache_key = (expr_type.full_name, i)
        substituted_schema_to_match = validator.get_resolved_instantiation_schema(cache_key)
        if substituted_schema_to_match is None:
            substituted_schema_to_match = build_resolved_instantiation_schema(
                schema_to_match,
                expr_type_template_context,
                expr_type,
                type_template_instantiation_validator,
                validator,
                location_id,
                cache_key,
                expansion_depth,
            )
        # The *caller's* mapping, not `own_substitution`: `expr_value` is text from the enclosing
        # expression and names the enclosing concept's variables.
        parsed, errors = validator.validate_value_against_schema(
            substituted_schema_to_match, expr_value, location_id, template_substitution, expansion_depth
        )
        groups.append(ConstraintGroupAttempt(type_application_constraint, matched=True, errors=tuple(errors)))
        return InstantiationSearch(parsed, tuple(errors), tuple(groups))
    raise RuntimeError(f"There should always be a fallback matching schema... This was not reached at {expr_type}!")


def get_json_type_as_string(json_value: object, location_id: LocationId) -> str:
    if json_value is None:
        return "null"
    if isinstance(json_value, bool):
        return "boolean"
    if isinstance(json_value, int):
        return "integer"
    if isinstance(json_value, float):
        return "number"
    if isinstance(json_value, str):
        return "string"
    if isinstance(json_value, dict):
        return "object"
    if isinstance(json_value, list):
        return "array"
    raise RuntimeError(
        "Impossible case that the json deserialization of a value produced a non-standard Python type ("
        f"{str(type(json_value))}); got {json_value} at {location_id}!"
    )


def _ground_unsupplied_argument_defaults_in_instantiated_context(
    f_name: str,
    f_type: InstantiatedType,
    all_arguments: set[str],
    unsupplied_arguments: set[str],
    f_substitution_mapping: dict[str, ConceptHierarchyTemplateArgument],
    f_template_context: TemplateContext,
    expr_template_context: TemplateContext,
    validator: ExpressionParserValidator,
    location_id: LocationId,
    attempts: list[ExpressionAttempt],
    expansion_depth: int,
) -> tuple[IllFormedExpression | None, dict[str, frozenset[str]], dict[str, Expression]]:
    """
    Check the defaults this call site leaves unsupplied, under *this* application's template arguments.

    A Function's declared defaults are in the position instantiation defaults were in before stage 1: they
    are parsed once, in the Function's own template context, with `T` standing for nothing in particular --
    so an argument of type `T` defaulting to something no `T` could ever be was accepted and never looked
    at again. Only an application decides it, and only a call site produces one.

    Grounding produces two things, and the second is not a by-product: the verdict, and the set of sibling
    arguments each default actually reads. That second set is what makes the acyclicity check complete --
    see `GroundedArgumentDefault.sibling_dependencies`.

    The parsed expression itself is deliberately **not** written into `FunctionEvaluation.arguments`. That
    dict means "what the call site wrote", and the acyclicity check derives ``supplied_arguments`` from it
    -- materialising defaults into it would tell that check every argument was supplied, and switch it off
    exactly where it is needed. It is cached on the validator instead, per application.

    Returns the failure to report (or ``None``), the grounded dependencies of every default it decided,
    and the expression each unsupplied argument fell back on -- which the caller keeps on the evaluation
    as `FunctionEvaluation.applied_defaults`. Every branch below records one, including the two that
    decide without reparsing: a default that needed no work is still a default this site applied.
    """
    dependencies: dict[str, frozenset[str]] = {}
    applied: dict[str, Expression] = {}
    to_ground: dict[str, object] = {}
    declaration_unparsed: set[str] = set()
    """The arguments whose *declaration* has no parse yet -- the only ones this pass can hand one back."""
    for argument in sorted(unsupplied_arguments):
        # A missing default here is not an omission: a *required* argument left unsupplied was already
        # reported above, and an optional one with no default has nothing to ground.
        declared_source = validator.get_function_argument_default_source(f_type.clean_name, argument)
        assert declared_source is not MISSING  # unsupplied arguments must be default; verified before this _ground call
        declared_default = validator.get_parsed_function_argument_default(f_type.clean_name, argument)
        declared_type, _, _ = validator.get_function_argument_interface(f_type.clean_name, argument)
        # Two *independent* things can still be waiting on the application, and they are settled
        # differently -- which is why this is two questions rather than one condition:
        #
        #   1. the default's own contents, which takes *both* of the two properties that describe an
        #      expression, because neither implies the other. `is_fully_parsed` asks whether every part was
        #      built at all -- a custom-type leaf with no expression was not -- and
        #      `is_value_template_dependent` asks whether any part is still waiting on a template argument.
        #      A `FunctionEvaluation` of `Add<T>` whose arguments all parsed is fully parsed *and* template
        #      dependent (measured: 24 of them); a value holding an unresolved default is the other way
        #      round. Only an expression that is both can never be reparsed into a different tree.
        #   2. how the expression's type relates to the *site's* type. That was left open whenever the site
        #      mentioned a template variable, however decided the expression itself is -- an expression
        #      reports its own dependence, never its type's.
        contents_are_decided = (
            declared_default is not None
            and declared_default.is_fully_parsed
            and not declared_default.is_value_template_dependent
        )
        site_type_is_decided = isinstance(declared_type, InstantiatedType)
        if contents_are_decided and site_type_is_decided:
            # Nothing was left open, so the declaration already holds the whole verdict -- which is
            # therefore also the expression this site applies.
            applied[argument] = declared_default
            continue
        if contents_are_decided:
            # Only (2). One subtype check settles it; reparsing would rebuild an identical tree to ask it.
            failure = _recheck_decided_default(
                f_name,
                f_type,
                argument,
                declared_default,
                declared_type,
                f_substitution_mapping,
                f_template_context,
                expr_template_context,
                validator,
                location_id,
                attempts,
            )
            if failure is not None:
                return failure, dependencies, applied
            # No entry is added to `dependencies`: the declaration's own scan of this default was complete
            # (a decided tree has no unreached parts), so the declared edges already say everything. The
            # expression itself is unchanged by the subtype check, so the declaration is what was applied.
            applied[argument] = declared_default
            continue
        # From here on, a reparsing of the argument is needed; because the default-expression is not decided (or parsed)
        cached: GroundedArgumentDefault | None = validator.get_grounded_function_default(f_type.full_name, argument)
        if cached is None:
            # then the default argument of that Function was not parsed yet! Schedule it for parsing!
            to_ground[argument] = declared_source
            if declared_default is None:
                declaration_unparsed.add(argument)
            continue
        if cached.in_progress:
            # this is what represents a (possibly-nested) dependency cycle
            reason = (
                f'the default of argument "{argument}" of {f_type} can never be applied: grounding it '
                f"requires grounding it again"
            )
            attempts.append(ExpressionAttempt(ExpressionKind.FUNCTION_EVALUATION, reason, tried_type=f_type))
            return IllFormedExpression(reason, tuple(attempts)), dependencies, applied
        # A cached failure is re-reported rather than passed over: the entry is the verdict for this
        # application, and a second call site reaching it is in exactly the position the first one was.
        assert cached.expression is not None  # equivalent to `not cached.in_progress`
        if not cached.expression.is_valid:
            return _rejected_default(f_name, f_type, argument, cached.expression, attempts), dependencies, applied
        dependencies[argument] = cached.sibling_dependencies
        applied[argument] = cached.expression
    # if there's nothing to parse, finish
    if not to_ground:
        return None, dependencies, applied

    # Every argument goes into scope, not just the unsupplied ones: a default may name a sibling the call
    # site *did* supply, and what it sees there is the argument's declared type, not the supplied value.
    argument_types: dict[str, TypeValue] = {}
    for argument in sorted(all_arguments):
        argument_type, _, _ = validator.get_function_argument_interface(f_type.clean_name, argument)
        argument_type, _ = substitute(
            argument_type,
            f_substitution_mapping,
            f_template_context,
            expr_template_context,
            validator.get_type_template_instantiation_validator(),
            location_id,
        )
        argument_types[argument] = argument_type

    with validator.function_argument_scope(argument_types):
        for argument, declared_source in to_ground.items():
            argument_type = argument_types[argument]
            assert isinstance(argument_type, TYPE_VALUE_IS_INSTANCE_CHECK)
            # Published before the parse, not after, so that a default which reaches itself finds the
            # in-progress entry instead of recursing. (An exception escaping the parse leaves the marker
            # behind, which is harmless: it aborts the whole check.)
            validator.put_grounded_function_default(f_type.full_name, argument, GroundedArgumentDefault(None))
            grounded = parse_expression(
                declared_source,
                argument_type,
                # ANY/GET, matching how the default was parsed where it was declared. Whether a default
                # may satisfy an ADDR or MOD argument is a question about defaults, not about
                # substitution, and answering it differently here would reject hierarchies for a reason
                # this pass did not set out to find.
                FunctionArgumentProvenance.ANY,
                FunctionArgumentAccessor.GET,
                # The default's text lives in the Function, so the Function's mapping -- *replacing* the
                # call site's, not merged with it -- is what grounds it. The context passed alongside is
                # the Function's for the same reason, though while `key_type` is ground nothing can tell
                # the two apart: a ground application makes `f_substitution_mapping` ground, so every
                # substitution below lands in the empty context either way.
                f_template_context,
                validator,
                location_id + [f_name, argument],
                template_substitution=f_substitution_mapping,
                # Counted like an instantiation default, and for the same reason: each level here is a
                # *new application*, because a level that repeated one is the in-progress case above. So a
                # default that keeps growing its own application -- `F<T>`'s argument defaulting to
                # `F<Box<T>>` -- never repeats and is stopped only by the limit.
                expansion_depth=expansion_depth + 1,
            )
            grounded_dependencies = frozenset(
                subexpression.value.variable_name
                for subexpression in grounded.all_subexpressions(Variable)
                if isinstance(subexpression.value, Variable) and subexpression.value.variable_name in all_arguments
            )
            validator.put_grounded_function_default(
                f_type.full_name, argument, GroundedArgumentDefault(grounded, grounded_dependencies)
            )
            _offer_grounding_as_the_declarations_parse(
                f_type, argument, grounded, f_template_context, declaration_unparsed, validator
            )
            if not grounded.is_valid:
                return _rejected_default(f_name, f_type, argument, grounded, attempts), dependencies, applied
            dependencies[argument] = grounded_dependencies
            applied[argument] = grounded
    return None, dependencies, applied


def _offer_grounding_as_the_declarations_parse(
    key_type: InstantiatedType,
    argument: str,
    grounded: Expression,
    f_template_context: TemplateContext,
    declaration_unparsed: set[str],
    validator: ExpressionParserValidator,
) -> None:
    """
    Hand a grounding back to the declaration, when the two parses cannot differ.

    Grounding reaches a default the declaration has not parsed yet in exactly one window: `init_expressions`
    fills the Functions' parsed defaults one at a time, and parsing one Function's default can evaluate a
    Function later in that loop (§7). The default is then parsed here -- and parsed *again*, from the same
    source, when the loop reaches its Function. Sometimes those two parses cannot come out differently, and
    then the second one is pure waste; this is where that is noticed.

    They cannot differ when **the declaring Function has no template variables**. Everything grounding does
    over and above the declaration's parse is substitution, and with no variables to substitute every one of
    them is the identity: the argument type is the declared type, the sibling types put back in scope are
    the declared ones, the Function's template context is the declaration's, and the mapping is empty. What
    is left is the same source parsed against the same type in the same scope.

    That leaves two differences, and neither can change the tree:

    * ``parse_template_expressions_without_type_checks`` is set at the declaration and not here. It decides
      one thing only -- whether a *non-ground* site is walked into or cut off with a placeholder -- so it
      can only matter where something is template dependent, and such a parse is refused below.
    * The enclosing concept's template context and identifier are still the current ones here, so a bare
      name that happens to be an *enclosing* Function's template variable is read as one, which the
      declaration's pass would not do. It cannot slip through: with this Function's mapping empty that
      variable is not substituted either, so the expression is template dependent, and is refused below.

    Hence the conditions: the expression must be valid, fully parsed and template independent -- the same
    pair of questions the decided-default shortcut asks, for the same reason (`Expression.is_fully_parsed`).
    Anything less is an expression the declaration's parse could still build differently, and it is dropped.

    What the declaration's pass does with the parse -- record the sibling edges it names, store it on the
    Function -- it still does; only the reparse is skipped.
    """
    if argument not in declaration_unparsed or f_template_context.variables:
        return
    if not grounded.is_valid or not grounded.is_fully_parsed or grounded.is_value_template_dependent:
        return
    validator.put_parsed_function_argument_default(key_type.clean_name, argument, grounded)


def _recheck_decided_default(
    key: str,
    key_type: InstantiatedType,
    argument: str,
    declared_default: Expression,
    declared_type: TypeValue,
    f_substitution_mapping: dict[str, ConceptHierarchyTemplateArgument],
    f_template_context: TemplateContext,
    expr_template_context: TemplateContext,
    validator: ExpressionParserValidator,
    location_id: LocationId,
    attempts: list[ExpressionAttempt],
) -> IllFormedExpression | None:
    """
    Settle a default that is decided everywhere except in its relation to the site's type.

    An expression reports its *own* template dependence, never its type's, so
    ``{"Mk<Integer>": {"v": 1}}`` at an argument declared `Cell<T>` is fully decided -- a ground evaluation
    with ground arguments -- while the one thing nobody could check yet is whether `Cell<Integer>` is a
    `Cell<T>`. Under `W<String>` it is not.

    So the application decides one question here, and it is answered with one subtype check. Reparsing would
    rebuild an identical tree to ask it.

    **This is narrower than a reparse, and one open defect is what keeps it sound.** A reparse would
    revalidate an `Inst` value against the *substituted* schema, and this does not. That is safe only
    because a value whose schema substitution could change it has a custom-type leaf that depends on
    templates, which makes the expression template dependent and sends it down the reparse path instead --
    with one exception: `substitute_schema` does not substitute `min_items_def` / `minimum_def` / ... (see
    the defect table in ``EXPRESSIONS_AND_INSTANTIATION_SCHEMAS.md``), so a schema whose only
    template-dependence sits in one of *those* keywords is not enforced either way today. If that defect is
    fixed, this shortcut has to be revisited: such a value would be decided, would come through here, and
    would never meet the substituted keyword.
    """

    def ground(type_value: TypeValue) -> TypeValue:
        substituted, _ = substitute(
            type_value,
            f_substitution_mapping,
            f_template_context,
            expr_template_context,
            validator.get_type_template_instantiation_validator(),
            location_id,
        )
        return substituted

    argument_type = ground(declared_type)
    value_type = declared_default.value.value_type
    assert value_type is not None, f"a valid expression has a type; {declared_default.unparsed} has none"
    # The expression's own type may mention the Function's variables too -- an `Inst` of `Cell<T>` is typed
    # `Cell<T>` -- so it is grounded by the same mapping before the two are compared.
    # Both sides are grounded before they are compared so the answer is definite. `_check_if_subtype` reads
    # existentially -- with a template variable left in, "no instantiation could make this hold" is the only
    # thing it can report `False` for, and a call site needs "this does not hold". No test distinguishes it,
    # because the expressions that reach here are typed by the declared type itself and so compare equal
    # either way; it is two lines, and it removes the need to rely on that.
    if value_type.depends_on_templates:
        value_type = ground(value_type)
    if _check_if_subtype(validator, value_type, argument_type, expr_template_context, location_id):
        return None
    reason = (
        f'the default {declared_default.unparsed} of {key} argument "{argument}" has type {value_type}, '
        f"which is not a subtype of {argument_type} under this application"
    )
    attempts.append(
        ExpressionAttempt(
            ExpressionKind.FUNCTION_EVALUATION,
            f'the default of the unsupplied argument "{argument}" does not hold for {key_type}',
            tried_type=key_type,
        )
    )
    return IllFormedExpression(reason, tuple(attempts))


def _rejected_default(
    key: str,
    key_type: InstantiatedType,
    argument: str,
    grounded: Expression,
    attempts: list[ExpressionAttempt],
) -> IllFormedExpression:
    """Report one default that does not hold for this application, keeping the parse's own explanation."""
    assert isinstance(grounded.value, IllFormedExpression)
    reason = (
        f'the default {grounded.unparsed} of {key} argument "{argument}" is not a valid '
        f"{grounded.required_expression_type} expression under this application: {grounded.value.reason}"
    )
    attempts.append(
        ExpressionAttempt(
            ExpressionKind.FUNCTION_EVALUATION,
            f'the default of the unsupplied argument "{argument}" does not hold for {key_type}',
            tried_type=key_type,
            cause=grounded.value,
        )
    )
    return IllFormedExpression(reason, tuple(attempts))


def parse_expression_of_json_object(
    key: str,
    value: object,
    expr_type: TypeValue,
    expr_template_context: TemplateContext,
    validator: ExpressionParserValidator,
    location_id: LocationId,
    recursively_parse: bool,
    parse_template_expressions_without_type_checks: bool,
    is_function_evaluation: bool,
    is_function_evaluation_present: bool,
    ensure_expression_invariant: Callable[[list[ExpressionValue], TypeValue], bool | None],
    attempts: list[ExpressionAttempt],
    template_substitution: dict[str, ConceptHierarchyTemplateArgument] | None = None,
    expansion_depth: int = 0,
) -> list[ExpressionValue]:
    expressions_res = []

    function_composition_type = validator.create_instantiated_type("FunctionComposition", location_id)
    function_type = validator.create_instantiated_type("Function", location_id)

    try:
        key_type = validator.create_possibly_template_dependent_type(key, location_id)
        if template_substitution:
            # The key is written in source text, so it can name the enclosing concept's template
            # variables (`{"Add<T>": ...}`); ground them before anything is decided from the type.
            key_type = substitute_template_variables_in_value(
                key_type,
                template_substitution,
                expr_template_context,
                TemplateContext(),
                validator.get_type_template_instantiation_validator(),
                location_id,
            )
    except CHSemanticError as e:
        if e.args[0] == f"ParsedType '{key}' is not a template variable (in this context) nor a concept!":
            # The single key is not a type at all, so neither an FEval nor a Narrow was ever possible.
            reason = f'"{key}" is not a concept or a template variable of this Concept Hierarchy'
            attempts.append(ExpressionAttempt(ExpressionKind.FUNCTION_EVALUATION, reason))
            attempts.append(ExpressionAttempt(ExpressionKind.NARROW, reason))
            return expressions_res
        raise e
    if not isinstance(key_type, TemplateVariable) and validator.is_type_abstract(key_type):
        raise CHSemanticError(
            f"{key_type} is an abstract type! Thus, it can not be used in expression values "
            f"(neither as FEval nor as Narrow expressions)!",
            location_id=location_id + [key],
            part=PathPart.KEY,
        )
    function_evaluation = isinstance(expr_type, TemplateVariable) or (
        not _check_if_subtype(validator, expr_type, function_composition_type, expr_template_context, location_id)
        and is_function_evaluation
    )
    is_function_subtype = _check_if_subtype(validator, key_type, function_type, expr_template_context, location_id)
    if function_evaluation and is_function_subtype:
        function_return = validator.get_function_return_interface(key_type.clean_name)
        if function_return is None:
            raise CHSemanticError(
                f"Function {key} does not return anything; expected a return type of {expr_type}!",
                location_id=location_id + [key],
                part=PathPart.KEY,
            )
        if not isinstance(value, dict):
            reason = (
                f"Function evaluation expression should have the value of the json object an other json object, "
                f"not {value}!"
            )
            attempts.append(ExpressionAttempt(ExpressionKind.FUNCTION_EVALUATION, reason, tried_type=key_type))
            expressions_res.append(IllFormedExpression(reason, tuple(attempts)))
            if ensure_expression_invariant(expressions_res, expr_type):
                return expressions_res
        else:
            function_return_type, function_return_access, function_return_provenance = function_return
            is_result_addressable = function_return_provenance == ValueDomainArgumentProvenance.ADDR

            # create substitution mapping
            f_substitution_mapping: dict[str, ConceptHierarchyTemplateArgument] = {}
            f_template_context: TemplateContext = validator.get_template_context(key_type.clean_name)
            for t_arg_name, t_arg_val in zip(f_template_context.variables, key_type.template_arguments):
                f_substitution_mapping[t_arg_name] = t_arg_val
            # substitute `function_return_type` with template instantiation of Function
            function_return_type, _ = substitute(
                function_return_type,
                f_substitution_mapping,
                # `function_return_type` is written in the *Function's* context and `f_substitution_mapping`
                # maps the Function's variables to values written in the enclosing one -- so the Function's
                # context is `template_context_of_value`, not the other way round. Swapping them only shows
                # when the two use different variable names, because the guard in `substitute` compares
                # names against the mapping's keys.
                f_template_context,
                expr_template_context,
                validator.get_type_template_instantiation_validator(),
                location_id,
            )
            assert isinstance(function_return_type, TYPE_VALUE_IS_INSTANCE_CHECK)
            if not isinstance(expr_type, TemplateVariable) and not _check_if_subtype(
                validator, function_return_type, expr_type, expr_template_context, location_id
            ):
                reason = f"Function result type {function_return_type} is not a subtype of {expr_type}"
                attempts.append(ExpressionAttempt(ExpressionKind.FUNCTION_EVALUATION, reason, tried_type=key_type))
                expressions_res.append(IllFormedExpression(reason, tuple(attempts)))
                if ensure_expression_invariant(expressions_res, expr_type):
                    return expressions_res
            f_args: dict[str, Expression] = {}
            applied_defaults: dict[str, Expression] = {}
            if recursively_parse:
                # verify sub-expressions + make sure that the Function arguments are actually correct ones
                all_arguments = validator.get_function_arguments(key_type.clean_name)
                for f_arg_name, f_arg_expr_val in value.items():
                    if not validator.is_function_argument(key_type.clean_name, f_arg_name):
                        raise CHSemanticError(
                            f'Function {key} does not have the argument "{f_arg_name}"; only {all_arguments}',
                            location_id=location_id,
                            part=PathPart.KEY,
                        )
                    f_arg_type, f_arg_access, f_arg_prov = validator.get_function_argument_interface(
                        key_type.clean_name, f_arg_name
                    )
                    # substitute `f_arg_type` with template instantiation of Function
                    f_arg_type, _ = substitute(
                        f_arg_type,
                        f_substitution_mapping,
                        f_template_context,
                        expr_template_context,
                        validator.get_type_template_instantiation_validator(),
                        location_id,
                    )
                    assert isinstance(f_arg_type, TYPE_VALUE_IS_INSTANCE_CHECK)
                    arg_expr = parse_expression(
                        f_arg_expr_val,
                        f_arg_type,
                        f_arg_prov,
                        f_arg_access,
                        expr_template_context,
                        validator,
                        location_id + [key, f_arg_name],
                        parse_template_expressions_without_type_checks,
                        template_substitution,
                        expansion_depth,
                    )
                    if not arg_expr.is_valid:
                        assert isinstance(arg_expr.value, IllFormedExpression)
                        reason = (
                            f"{key} argument {f_arg_name}'s value {f_arg_expr_val} is invalid: {arg_expr.value.reason}"
                        )
                        attempts.append(
                            ExpressionAttempt(
                                ExpressionKind.FUNCTION_EVALUATION,
                                f'argument "{f_arg_name}" is not a valid {f_arg_type} expression',
                                tried_type=key_type,
                                cause=arg_expr.value,
                            )
                        )
                        expressions_res.append(IllFormedExpression(reason, tuple(attempts)))
                        if ensure_expression_invariant(expressions_res, expr_type):
                            return expressions_res
                    f_args[f_arg_name] = arg_expr
                # verify required arguments are present
                required_arguments: set[str] = validator.get_required_function_arguments(key_type.clean_name)
                missing_arguments: set[str] = set(
                    required_arg for required_arg in required_arguments if required_arg not in f_args
                )
                if missing_arguments:
                    raise CHSemanticError(
                        f"Argument(s) {missing_arguments} are missing from the Function evaluation interface of {key}!",
                        location_id=location_id + [key],
                        part=PathPart.VALUE,
                    )
                # verify that the dependencies between the remaining default arguments are not cyclic
                supplied_arguments = set(f_args)
                unsupplied_arguments: set[str] = all_arguments - supplied_arguments
                default_argument_dependencies = validator.get_default_argument_dependencies(key_type.clean_name)
                # Ground first, then check the graph. Once the application is ground, each unsupplied
                # default has to be a valid expression *for it* -- and grounding is also the only place the
                # complete dependency set exists, because the definition-time scan stops wherever the type
                # stopped being decidable. Only `InstantiatedType` is ground; a call site still inside a
                # template gets its turn when the enclosing schema is built for an application (stage 1).
                if isinstance(key_type, InstantiatedType):
                    grounding_res = _ground_unsupplied_argument_defaults_in_instantiated_context(
                        key,
                        key_type,
                        all_arguments,
                        unsupplied_arguments,
                        f_substitution_mapping,
                        f_template_context,
                        expr_template_context,
                        validator,
                        location_id,
                        attempts,
                        expansion_depth,
                    )
                    default_failure, grounded_dependencies, applied_defaults = grounding_res
                    if default_failure is not None:
                        expressions_res.append(default_failure)
                        if ensure_expression_invariant(expressions_res, expr_type):
                            return expressions_res
                    elif default_argument_dependencies is not None and grounded_dependencies:
                        # Union rather than replacement. Grounding sees strictly more than the
                        # definition-time scan did -- the only way it could see *less* is if substitution
                        # turned a sibling reference into something else, which needs an argument and a
                        # template variable of the same name, and those cannot collide (arguments start
                        # lowercase, template variables uppercase). So the two agree today and no test can
                        # tell the union from a replacement; it is kept because losing an edge is the
                        # failure that matters, and the union cannot lose one if that ever changes.
                        merged = dict(default_argument_dependencies)
                        for argument, argument_dependencies in grounded_dependencies.items():
                            merged[argument] = merged.get(argument, frozenset()) | argument_dependencies
                        default_argument_dependencies = frozendict(merged)
                if default_argument_dependencies is not None and not _validate_acyclic_default_argument_dependencies(
                    default_argument_dependencies, supplied_arguments
                ):
                    raise CHSemanticError(
                        f"The dependency graph between the remaining default arguments {unsupplied_arguments} of "
                        f"the Function evaluation of {key} is not acyclic!",
                        location_id=location_id + [key],
                        part=PathPart.VALUE,
                    )
            expressions_res.append(
                FunctionEvaluation(
                    key_type,
                    function_return_type,
                    f_args,
                    is_result_addressable,
                    function_return_type != expr_type,
                    applied_defaults,
                )
            )
            if ensure_expression_invariant(expressions_res, expr_type):
                return expressions_res

    if _check_if_subtype(validator, key_type, expr_type, expr_template_context, location_id):
        if is_function_evaluation_present and not is_function_subtype:
            raise CHSemanticError(
                f'Invalid use of the "isFunctionEvaluation" keyword at single-content-key object "{key_type}"!',
                location_id=location_id + ["isFunctionEvaluation"],
                part=PathPart.KEY,
            )
        if not recursively_parse:
            expressions_res.append(NarrowExpression(None, key_type, key_type != expr_type))
            if ensure_expression_invariant(expressions_res, expr_type):
                return expressions_res
        else:
            # abstract Types do not have instantiation schemas
            narrow_res = _check_instantiation_schema(
                value,
                key_type,
                expr_template_context,
                validator,
                location_id,
                template_substitution,
                expansion_depth,
            )
            if narrow_res.parsed is not None and narrow_res.parsed.is_valid():
                expressions_res.append(NarrowExpression(narrow_res.parsed, key_type, key_type != expr_type))
                if ensure_expression_invariant(expressions_res, expr_type):
                    return expressions_res
            attempts.append(_instantiation_attempt(ExpressionKind.NARROW, key_type, narrow_res))
    else:
        attempts.append(
            ExpressionAttempt(
                ExpressionKind.NARROW,
                f"{key_type} is not a subtype of {expr_type}",
                tried_type=key_type,
            )
        )

    # if expr_type is InstantiatedTypes, this expression is neither a `FEval` nor a `Narrow`
    return expressions_res


class SubtypeVerdict(Enum):
    """
    The outcome of a subtype check whose operands may be template-dependent.

    NO
        There is no instantiation of the template variables under which the check holds.
    MAYBE
        The check holds under some, but not necessarily all, instantiations. The accompanying
        ``TemplateContext`` records the constraint on the template variables under which it holds.
    YES
        The check holds under every instantiation permitted by the template context.
    """

    NO = 0
    MAYBE = 1
    YES = 2


@dataclass(frozen=True)
class SubtypeCheckResult:
    """
    The result of :func:`check_if_subtype`.

    ``template_context`` is only set for a ``MAYBE`` verdict; it holds the constraint on the template
    variables under which ``a`` is a subtype of ``b``, and is what a later instantiation-time check must
    re-verify. ``errors`` is only set for a ``NO`` verdict and explains why the check failed.
    """

    verdict: SubtypeVerdict
    template_context: TemplateContext | None = None
    errors: tuple[ConceptHierarchyError, ...] = ()

    def __bool__(self) -> bool:
        """Existential reading: only a definite ``NO`` is falsy; a ``MAYBE`` is a "not yet ruled out"."""
        return self.verdict is not SubtypeVerdict.NO


def _general_subtype_check(
    validator: ExpressionParserValidator,
    a: TypeValue,
    b: TypeValue,
    template_context: TemplateContext,
    location_id: LocationId,
) -> SubtypeCheckResult:
    """
    Check whether ``a`` is a subtype of ``b``, where either may be a template variable or a
    template-dependent type.

    All template variables occurring in ``a`` or ``b`` must be template parameters of the ValueDomain or
    Function that the expression lies in, i.e. they must be variables of ``template_context``; a reference
    to any other variable is an invalid type application and is reported as such.

    Template arguments are matched **invariantly**: ``Box<Integer>`` is not a subtype of ``Box<Number>``.

    :return: a :class:`SubtypeCheckResult`; a ``MAYBE`` carries the template-variable constraint under which
        the subtype relation holds, which the (still to be written) instantiation-time check must verify.
    """
    _verify_subtype_check_operands(validator, a, b)

    if isinstance(a, InstantiatedType) and isinstance(b, InstantiatedType):
        # Neither side depends on template variables, so the answer is definite.
        if validator.is_a_subtype_of_b(a, b, location_id):
            return SubtypeCheckResult(SubtypeVerdict.YES)
        return SubtypeCheckResult(SubtypeVerdict.NO)

    type_template_instantiation_validator = validator.get_type_template_instantiation_validator()
    if isinstance(b, TemplateVariable):
        return _check_if_subtype_of_template_variable(
            type_template_instantiation_validator, a, b, template_context, location_id
        )
    return _check_if_subtype_of_type_application(
        type_template_instantiation_validator, a, b, template_context, location_id
    )


def _verify_subtype_check_operands(validator: ExpressionParserValidator, a: TypeValue, b: TypeValue) -> None:
    """Neither operand of a subtype check may be a variadic or a literal template variable."""
    for type_value, role in ((a, "subtype"), (b, "supertype")):
        if not isinstance(type_value, TemplateVariable):
            continue
        if not isinstance(type_value, NonVariadicTemplateVariable):
            raise RuntimeError(
                f'It can not be that the type "{type_value}" to be checked as {role} is a variadic template variable!'
            )
        if validator.is_literal_template_variable(type_value.clean_name):
            raise RuntimeError(f'It can not be that a type "{type_value}" is a literal template variable!')


def _check_if_subtype_of_template_variable(
    validator: TypeTemplateInstantiationValidator,
    a: TypeValue,
    b: NonVariadicTemplateVariable,
    template_context: TemplateContext,
    location_id: LocationId,
) -> SubtypeCheckResult:
    """
    ``b`` is an uninstantiated template variable, so ``a`` is a subtype of it exactly when ``b`` is
    instantiated to ``a`` itself or to one of ``a``'s supertypes. That can not be decided while ``b`` is
    uninstantiated, so it is recorded as an additional constraint on ``b`` instead.
    """
    if a.full_name == b.full_name:
        # The very same template variable; whatever it is instantiated to, it is a subtype of itself.
        return SubtypeCheckResult(SubtypeVerdict.YES)
    # `ASCENDANTS_OF` always excludes the literal itself, so the reflexive case is added explicitly.
    supertype_or_equal = TemplateConstraintOr(
        location_id,
        (
            validator.create_type_constraint_from_value(a, location_id, HierarchyCheckType.SELF, template_context),
            validator.create_type_constraint_from_value(
                a, location_id, HierarchyCheckType.ASCENDANTS_OF, template_context
            ),
        ),
    )
    narrowed = TemplateContext(
        template_context.variables,
        template_context.variadic_variables,
        template_context.add_and_constraint_to(b.clean_name, supertype_or_equal, location_id),
    )
    if narrowed.is_empty_constraint:
        return SubtypeCheckResult(SubtypeVerdict.NO)
    return SubtypeCheckResult(SubtypeVerdict.MAYBE, narrowed)


def _check_if_subtype_of_type_application(
    validator: TypeTemplateInstantiationValidator,
    a: TypeValue,
    b: InstantiatedType | TemplateDependentType,
    template_context: TemplateContext,
    location_id: LocationId,
) -> SubtypeCheckResult:
    """
    ``b`` is a type application, so "being a subtype of ``b``" is expressible as the constraint formula
    "a descendant of ``b``, whose template arguments match ``b``'s exactly", and ``a`` can be validated
    against it. Where that validation meets a template variable -- on either side -- it does not decide the
    check but accumulates the constraint under which it would hold.
    """
    b_constraint = validator.create_type_constraint_from_value(
        b, location_id, HierarchyCheckType.DESCENDANTS_OF, template_context
    )
    determinator = TemplateContextDeterminator(template_context)
    errors = validate_template_argument_value_against_constraint(
        b_constraint, a, determinator, validator, {}, location_id, collect_all_errors=True
    )
    if errors:
        return SubtypeCheckResult(SubtypeVerdict.NO, errors=tuple(errors))
    if determinator.determined is None:
        # Nothing had to be constrained, so the check holds for every instantiation.
        return SubtypeCheckResult(SubtypeVerdict.YES)
    if determinator.determined.is_empty_constraint:
        return SubtypeCheckResult(SubtypeVerdict.NO)
    return SubtypeCheckResult(SubtypeVerdict.MAYBE, determinator.determined)


def _check_if_subtype(
    validator: ExpressionParserValidator,
    a: TypeValue,
    b: TypeValue,
    template_context: TemplateContext,
    location_id: LocationId,
) -> bool:
    """
    Whether ``a`` can be a subtype of ``b``, read existentially: ``False`` means that no instantiation of the
    template variables can make ``a`` a subtype of ``b``, and the expression is therefore ill-formed.

    Use :func:`_general_subtype_check` directly to also obtain the constraint that a ``MAYBE`` result depends on.
    """
    return bool(_general_subtype_check(validator, a, b, template_context, location_id))


def _validate_acyclic_default_argument_dependencies(
    default_argument_dependencies: frozendict[str, frozenset[str]], supplied_arguments: set[str]
) -> bool:
    # Default arguments whose value must actually be evaluated at this call site:
    # supplied defaults are terminal (their expressions are never evaluated) and
    # are therefore excluded from the dependency graph entirely.
    unsupplied_arguments = default_argument_dependencies.keys() - supplied_arguments
    if not unsupplied_arguments:
        return True

    # in_degree[node]: number of not-yet-resolved unresolved dependencies of `node`.
    # successors[node]: unresolved default arguments that depend on `node`
    # (i.e., the reverse adjacency list, needed to propagate resolution in Kahn's algorithm).
    in_degree: dict[str, int] = {}
    successors: dict[str, list[str]] = {node: [] for node in unsupplied_arguments}

    for node in unsupplied_arguments:
        # Restrict this node's declared dependencies to the relevant subgraph:
        # non-default arguments and already-supplied defaults are always resolved,
        # so they contribute no edge and are dropped here.
        deps = default_argument_dependencies[node] & unsupplied_arguments
        in_degree[node] = len(deps)
        for dep in deps:
            successors[dep].append(node)

    # Nodes with no unresolved dependencies can be evaluated immediately.
    queue = deque(node for node, degree in in_degree.items() if degree == 0)
    resolved_count = 0

    # Standard Kahn's algorithm: repeatedly resolve nodes with in-degree 0 and
    # decrement the in-degree of their dependents. A node stuck with in-degree > 0
    # forever (never enqueued) is part of, or depends on, a cycle.
    while queue:
        node = queue.popleft()
        resolved_count += 1
        for successor in successors[node]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue.append(successor)

    # Acyclic iff every unresolved node was eventually resolved.
    # A self-dependency (node depends on itself) leaves in_degree >= 1 permanently,
    # so it is correctly caught here without special-casing.
    return resolved_count == len(unsupplied_arguments)
