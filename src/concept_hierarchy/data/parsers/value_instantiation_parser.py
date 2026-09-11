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

"""Parse a Python (JSON-decoded) value against a :class:`~parsed_schema.CHSchemaNode`, producing a
:class:`~instantiated_value.ParsedValue` tree and an error list.

This module supersedes the former ``value_instantiation_validator``: the two modules performed the *same* traversal,
one discarding its results.  There is now a single traversal.  Validation is the error projection of parsing, not a
separate pass; :func:`validate_value` is retained only as a thin wrapper for call sites that discard the result tree.

Recursion strategy
------------------
Every structural keyword is handled explicitly; ``jsonschema`` is used only to check a node's *own* leaf-level
constraints via :attr:`~parsed_schema.CHSchemaNode.shallow_canonical`.  This is what routes custom-type leaves to
:meth:`ValueInstantiationContext.parse_value_against_custom_type_expression` wherever they occur, including inside
``anyOf``/``oneOf``/``allOf``.

One consequence is worth knowing here: a keyword written as a literal template variable (``{"minItems": "N"}``) is
**not** in ``shallow_canonical`` until an application binds it, so a value passes such a node *unchecked* and this
module reports no error for it.  Nothing is wrong with the value -- it simply has not been checked yet.
:meth:`~parsed_schema.CHSchemaNode.undecided_literal_keywords_for` is what reports that a node is in that state, and
:attr:`~subexpressions.InstExpression.is_template_dependent` consults it so the resulting expression is treated as
still undecided rather than as parsed-and-holding.

Relation to the expression parser
---------------------------------
:class:`ValueInstantiationContext` is not a second validation algorithm.  It is the abstract seam that breaks the
import cycle between this module and the expression parser: at a custom-type leaf this module calls the context, whose
concrete implementation invokes the expression parser, which on ``Inst``/``Narrow`` selects an instantiation schema and
re-enters :func:`parse_value`.

Termination of that mutual recursion is *not* guaranteed by the value shrinking: the value is unchanged across the
re-entry, and the schema-depth measure is reset at each new schema root.  Descent through ``properties``, ``items``,
``patternProperties``, ``additionalProperties``, ``additionalItems``, ``contains`` and ``propertyNames`` does consume
value structure; descent through ``anyOf``/``allOf``/``oneOf``/``not``/``if``-``then``-``else``/``$ref`` and the schema
root does not.  The composite recursion therefore terminates iff no cycle exists in the graph whose edges are
``T -> T'`` for every custom-type node of type ``T'`` reachable from the root of ``T``'s instantiation schema without
crossing a value-consuming keyword.  This is a static property of the schema set and belongs in schema well-formedness
checking, not here.

Absent-value / default semantics
--------------------------------
An absent optional property is materialised only when its schema (or an ``anyOf`` branch of it) is a custom-type node
carrying ``default_expr``.  In that case :attr:`~instantiated_value.ParsedCustomValue.used_default` is ``True`` and the
property counts as present for the purposes of ``required``.

This module never parses ``default_expr`` itself, and does not hold the parsed result either.  It asks the context for
it -- :meth:`ValueInstantiationContext.resolve_default`, keyed by the *schema node*, not by the expression text -- and
the context resolves it on demand and memoises it per ground application.  Two things follow, and neither is visible
from ``default_expr`` alone:

* the same declared default is a different expression under different applications of the same concept, since the node
  it hangs off belongs to a substituted schema;
* a site reached while it is *already being resolved* is a genuine expansion cycle, and the context reports it as one
  rather than recursing.

``resolve_default`` is called whenever the node **has** a default, not only when the value is absent -- but an invalid
default is reported only where it is *materialised* (``used_default``).  Supplying the key instead stays legal, so a
default that cannot hold for this application must not reject a value that never asks for it.

``allOf`` is all-or-nothing: if any branch fails, the errors of all failing branches are reported on the parent and
``all_of_parsed`` is left empty.  ``anyOf`` retains every matching branch.  ``if_parsed`` is metadata only and is
excluded from :meth:`~instantiated_value.ParsedValue.iter_children` and
:meth:`~instantiated_value.ParsedValue.is_valid`.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from typing import Callable

from jsonschema import Draft7Validator

from concept_hierarchy.data.expressions.expression import Expression
from concept_hierarchy.data.expressions.expression_utils import (
    ExpressionProvenance,
    FunctionInterpretation,
    split_function_interpretation_marker,
)
from concept_hierarchy.data.expressions.instantiated_value import ParsedCustomValue, ParsedStructural, ParsedValue
from concept_hierarchy.data.jsonschema.parsed_schema import CHSchemaNode
from concept_hierarchy.data.type_template_variables.constraint_formula import (
    NonStructureConstraintFormula,
    TemplateConstraintFormula,
)
from concept_hierarchy.data.types.concept_hierarchy_types import (
    ConceptHierarchyTemplateArgument,
    InstantiatedType,
    TypeValue,
)
from concept_hierarchy.data.utils import MISSING, StopValidation, record
from concept_hierarchy.definitions.concept_definition_domain_concept import (
    DomainConceptDefinition,
    ForPropertyOrFunction,
)
from concept_hierarchy.errors import CHSemanticError, ConceptHierarchyError, LocationId, PathPart

# ===========================================================================================================
# Context seam
# ===========================================================================================================


class ValueInstantiationContext(ABC):
    """Supplies the knowledge that is not part of the schema text: how to turn a raw JSON value at a custom-type node
    into a typed :class:`~expression.Expression`.

    The concrete implementation lives with the expression parser.  This module depends only on the abstract interface,
    so the two parsers can be mutually recursive without a module-level import cycle.
    """

    @abstractmethod
    def parse_value_against_custom_type_expression(
        self,
        custom_type: TypeValue,
        provenance: ExpressionProvenance,
        value: object,
        location_id: LocationId,
        template_substitution: dict | None,
        expansion_depth: int,
        function_interpretation: FunctionInterpretation,
    ) -> tuple[Expression | None, list[ConceptHierarchyError]]:
        """Parse and validate ``value`` as an expression of ``custom_type``.

        ``template_substitution`` and ``expansion_depth`` are carried through from the expression this
        value belongs to.  The value is text from the *same source expression*, so it can still name the
        enclosing concept's template variables even several schemas deep, and its own default expansions
        count against the same depth bound.

        Returns:
            ``(expression, errors)``.  ``expression`` is ``None`` when parsing failed; ``errors`` lists *every* problem
            found, not just the first.
        """

    @abstractmethod
    def parse_function_evaluation(
        self,
        function_name: str,
        arguments: object,
        schema_node: CHSchemaNode,
        location_id: LocationId,
        template_substitution: dict | None,
        expansion_depth: int,
    ) -> tuple[ParsedValue | None, list[ConceptHierarchyError]]:
        """Parse ``{function_name: arguments}`` as a **Function evaluation**, which is the only thing it
        may be.

        ``location_id`` is the location of the **object**, i.e. of ``{function_name: arguments}`` -- not of
        the arguments. That is the convention the expression parser uses wherever a single-key object is
        parsed, and it is what makes ``location_id + [function_name]`` the arguments and
        ``location_id + [function_name, <argument>]`` one of them. Passing the arguments' own location
        instead spells the key twice.

        This is what ``"properties": "args"`` means: the legal keys of ``arguments`` are the Function's
        arguments, which only the Function's interface knows, so the schema cannot spell them and this
        module cannot check them.  Everything that makes an evaluation well-formed is decided here --
        the argument expressions, the required arguments, the grounding of the defaults the site leaves
        unsupplied, and the acyclicity of the dependencies between those defaults.

        A single method rather than the pieces to rebuild it with: the alternative is exporting the
        expression parser's type resolution and template substitution through this interface and
        reimplementing the evaluation logic on top of them, which is the same coupling with more of it.

        Returns:
            ``(parsed, errors)``, in the shape of `parse_value_against_custom_type_expression`.  ``parsed``
            is ``None`` when the value is not a valid evaluation -- including when ``function_name`` does
            not name a Function at all, which is an error *here* though it is merely another alternative
            to the expression parser.  **Errors are returned, never raised**: this runs inside ``anyOf`` /
            ``oneOf`` trial branches, whose errors must be able to be discarded with the branch.
        """

    @abstractmethod
    def resolve_default(self, schema_node: CHSchemaNode) -> Expression | None:
        """The expression of ``schema_node``'s ``default``, parsed now if that has not happened yet.

        Defaults are resolved on demand rather than when their schema is built, so that a site reached
        while it is *already being resolved* -- a genuine expansion cycle -- is distinguishable from one
        that simply has not been reached yet.  Returns ``None`` when the node has no resolvable default.
        """

    @abstractmethod
    def is_concept(self, concept_candidate: str) -> bool:
        pass

    @abstractmethod
    def is_type(self, type_candidate: str, location_id: LocationId) -> bool:
        pass

    @abstractmethod
    def parse_constraint(self, constraint: str, location_id: LocationId) -> TemplateConstraintFormula:
        pass

    @abstractmethod
    def validate_string_constraint(
        self, constraint: NonStructureConstraintFormula, value: str, location_id: LocationId
    ) -> bool:
        pass

    @abstractmethod
    def parse_type(
        self,
        type_candidate: str,
        template_substitution: dict[str, ConceptHierarchyTemplateArgument],
        location_id: LocationId,
    ) -> tuple[InstantiatedType, list[ConceptHierarchyError]]:
        pass

    @abstractmethod
    def replace_variable_scope_for_custom_function(
        self, new_variables: dict[str, TypeValue]
    ) -> AbstractContextManager[None]:
        pass

    def is_domain_concept_datum_in_concept_hierarchy(self, key: str) -> bool:
        pass

    def substitute_with_x(
        self,
        custom_type: TypeValue,
        substitution: dict[str, ConceptHierarchyTemplateArgument],
        schema_owner: str,
        key_location_id: LocationId,
    ) -> TypeValue:
        pass

    def collect_data(
        self,
        concept_restriction: list[InstantiatedType] | None,
        for_properties_or_functions: ForPropertyOrFunction,
        include_parent_data: bool,
    ) -> dict[str, InstantiatedType]:
        pass


# ===========================================================================================================
# Traversal state
# ===========================================================================================================


@dataclass
class _State:
    """Accumulator threaded through the traversal.

    ``errors`` is the global error list for the current (sub-)parse; a *silent* substate is used for trial branches so
    that their errors do not escape.
    """

    context: ValueInstantiationContext
    errors: list[ConceptHierarchyError] = field(default_factory=list)
    collect_all_errors: bool = True
    template_substitution: dict | None = None
    """Carried to every custom-type leaf; see `ValueInstantiationContext.parse_value_against_custom_type_expression`."""
    expansion_depth: int = 0
    root_location_id: LocationId | None = None
    """
    Where this ``parse_value`` started, so that a node can ask whether it *is* the value being parsed.

    One question needs it: `CustomFunction`'s instantiation offers a bare `FunctionComposition` as a
    shorthand for the whole function, and also declares a `FunctionComposition` at its ``procedure`` property.
    Both are custom-type leaves owned by `CustomFunction` and are otherwise indistinguishable --
    but the shorthand *is* the value, so it sits at this location, while ``procedure`` sits at the "properties" key.
    A full trace of the walk would answer the same question; but the root alone is enough for it.
    """
    function_interpretation_at: LocationId | None = None
    """Where `function_interpretation` applies; a different location is a different value."""
    function_interpretation: FunctionInterpretation = FunctionInterpretation.UNSPECIFIED
    upper_level_object_key: str | None = None

    def record(self, err: ConceptHierarchyError) -> None:
        """Record ``err`` globally; raises :class:`StopValidation` in fail-fast mode."""
        record(self.errors, self.collect_all_errors, err)

    def silent(self) -> _State:
        """A substate whose errors are collected in full and do not escape."""
        return replace(self, errors=[], collect_all_errors=True)

    def new_state(self, upper_level_object_key: str | None = None) -> _State:
        return replace(self, upper_level_object_key=upper_level_object_key)

    def reading(self, location_id: LocationId):
        return (
            self.function_interpretation
            if location_id == self.function_interpretation_at
            else FunctionInterpretation.UNSPECIFIED
        )


# ===========================================================================================================
# Entry points
# ===========================================================================================================


def parse_value(
    value: object,
    node: CHSchemaNode,
    context: ValueInstantiationContext,
    location_id: LocationId | None = None,
    template_substitution: dict | None = None,
    expansion_depth: int = 0,
    function_interpretation: FunctionInterpretation = FunctionInterpretation.UNSPECIFIED,
    collect_all_errors: bool = True,
) -> tuple[ParsedValue, list[ConceptHierarchyError]]:
    """Parse ``value`` against ``node``.

    Args:
        value: The Python object to parse (JSON-decoded).
        node: Schema AST from ``parse_schema``, which should itself have parsed without errors.
        context: Handles custom-type leaves.
            is to be interpreted/parsed.
        location_id: Starting location in the Concept Hierarchy (``[]`` at the root).
        template_substitution: If not ``None``, stores the mapping of template parameters that could have been used
            in this value and which must be substituted in the value to do a complete check of the value.
        expansion_depth: how many default-instantiation-expressions were triggered.
            This detects a possibly infinite expansion cycle.
        function_interpretation: What an enclosing expression site already decided about reading this
            value as a Function evaluation; honored at the custom-type leaf sitting at ``location_id``.
        collect_all_errors: ``True`` to collect every error, ``False`` to stop at the first one.

    Returns:
        ``(result, errors)``.  ``result`` is always a tree, even on failure.  ``errors`` is the authoritative error
        list; do not reconstruct it by walking ``result``, because trial branches and failed ``allOf`` branches
        deliberately do not attach their errors to retained nodes.
    """
    if location_id is None:
        location_id = []

    state = _State(
        context,
        [],
        collect_all_errors,
        template_substitution,
        expansion_depth,
        location_id,
        None if function_interpretation is FunctionInterpretation.UNSPECIFIED else location_id,
        function_interpretation,
    )
    result: ParsedValue | None = None
    try:
        result = _parse(node, value, True, location_id, state)
    except StopValidation:
        pass

    # Guarantee a non-None return even when parsing was cut short.
    if result is None:
        # Parsing was cut short, or the root schema admitted an absent value.
        result = ParsedStructural(location_id=location_id, schema_node=node, errors=list(state.errors), value=value)
    return result, state.errors


def validate_value(
    value: object,
    node: CHSchemaNode,
    context: ValueInstantiationContext,
    location_id: LocationId | None = None,
    template_substitution: dict | None = None,
    expansion_depth: int = 0,
    function_interpretation: FunctionInterpretation = FunctionInterpretation.UNSPECIFIED,
    collect_all_errors: bool = True,
) -> list[ConceptHierarchyError]:
    """Error projection of :func:`parse_value`, for call sites that discard the result tree.

    This performs the full parse; it is not cheaper.
    """
    _, errors = parse_value(
        value,
        node,
        context,
        location_id,
        template_substitution,
        expansion_depth,
        function_interpretation,
        collect_all_errors,
    )
    return errors


# ===========================================================================================================
# Core dispatcher
# ===========================================================================================================


def _parse(
    node: CHSchemaNode, value: object, present: bool, location_id: LocationId, state: _State
) -> ParsedValue | None:
    """Recursively parse ``value`` against ``node``.

    Returns ``None`` when ``present`` is ``False`` and no default fills the gap;
    the property is then omitted from the parent's ``properties_parsed``.
    """
    if not present:
        return _parse_absent(node, location_id, state)

    # $ref is transparent; draft-07 ignores siblings of $ref.
    if node.ref_resolved is not None:
        return _parse(node.ref_resolved, value, True, location_id, state)

    # Boolean schema.
    if node.is_boolean_schema:
        local: list[ConceptHierarchyError] = []
        if node.canonical is False:
            err = CHSemanticError("No value is allowed here (schema is `false`)", location_id)
            local.append(err)
            state.record(err)
        return ParsedStructural(location_id=location_id, schema_node=node, errors=local, value=value)

    # Custom-type leaf.
    if node.is_custom_type:
        return _parse_custom(node, value, location_id, state)

    # Builtin / structural / composite.
    return _parse_structural(node, value, location_id, state)


# ===========================================================================================================
# Absent value
# ===========================================================================================================


def _parse_absent(node: CHSchemaNode, location_id: LocationId, state: _State) -> ParsedValue | None:
    """Handle an absent optional property.

    Returns a :class:`ParsedCustomValue` with ``used_default=True`` when the node (or the first accepting ``anyOf``
    branch) is a custom-type node carrying a default; ``None`` otherwise.
    """
    if node.ref_resolved is not None:
        return _parse_absent(node.ref_resolved, location_id, state)

    # Boolean schemas never carry a default.
    if node.is_boolean_schema:
        return None

    # Custom-type: only call the validator when a default is defined.
    if node.is_custom_type:
        if not node.has_default:
            return None
        return _parse_custom(node, MISSING, location_id, state)

    # Composite: take the first anyOf branch that accepts MISSING, i.e. a custom-type branch with a default.
    if node.any_of:
        for branch in node.any_of:
            trial = state.silent()
            result = _parse_absent(branch, location_id, trial)
            if result is not None and not trial.errors:
                return result

    return None


# ===========================================================================================================
# Custom-type leaf
# ===========================================================================================================


def _parse_custom(node: CHSchemaNode, value: object, location_id: LocationId, state: _State) -> ParsedCustomValue:
    """Parse a custom-type leaf.  ``value`` is :data:`MISSING` when the node's default is being applied instead."""
    local: list[ConceptHierarchyError] = []
    used_default = value is MISSING
    expression, default_expr = None, node.default_expr if node.has_default else MISSING
    if node.has_default:
        expression = state.context.resolve_default(node)
    if expression is not None:
        if used_default and not expression.is_valid:
            # The default is being *applied* here, and it does not parse -- which for a substituted schema
            # means it does not type-check under this ground application even though it may under another.
            # Report it only on materialisation: supplying the key instead is still perfectly valid, so
            # failing at substitution time would reject applications that are entirely usable.
            reason = getattr(expression.value, "reason", "the default expression is not valid")
            err = CHSemanticError(
                f"The default of this {node.custom_type} can not be used here: {reason}",
                location_id,
                part=PathPart.VALUE,
            )
            if hasattr(expression.value, "explanation_causes"):
                err.causes.extend(expression.value.explanation_causes(location_id))
            local.append(err)
            state.record(err)
    if not used_default:

        def _parse_expr():
            return state.context.parse_value_against_custom_type_expression(
                node.custom_type,
                node.provenance,
                value,
                location_id,
                state.template_substitution,
                state.expansion_depth,
                state.reading(location_id),
            )

        if _is_the_custom_function_shorthand(node, location_id, state):
            with state.context.replace_variable_scope_for_custom_function({}):
                expression, errs = _parse_expr()
        else:
            expression, errs = _parse_expr()
        for err in errs:
            local.append(err)
            state.record(err)

    return ParsedCustomValue(
        location_id=location_id,
        schema_node=node,
        errors=local,
        custom_type=node.custom_type,
        provenance=node.provenance,
        used_default=used_default,
        default_expr=default_expr,
        expression=expression,
    )


# ===========================================================================================================
# Structural / composite node
# ===========================================================================================================


def _is_the_custom_function_shorthand(node: CHSchemaNode, location_id: LocationId, state: _State) -> bool:
    """
    Whether ``node`` is `CustomFunction`'s bare-`FunctionComposition` shorthand rather than its ``procedure``.

    Both are custom-type leaves declared by `CustomFunction`; what separates them is position.
    The shorthand is the whole value, so it stands at the root of this parse;
    ``procedure`` is a property of the object form and stands one key below it.
    """
    return (
        node.schema_owner == DomainConceptDefinition.default_value_domain_type_of_domain_concept_functions
        and state.root_location_id is not None
        and location_id == state.root_location_id
    )


def _parse_structural(node: CHSchemaNode, value: object, location_id: LocationId, state: _State) -> ParsedStructural:
    local: list[ConceptHierarchyError] = []

    def rec(err: ConceptHierarchyError) -> None:
        """Record an error on this node and in the global accumulator."""
        local.append(err)
        state.record(err)

    def child_p(
        schema: CHSchemaNode, val: object, child_loc: LocationId, new_state: _State | None = None
    ) -> ParsedValue | None:
        """Parse a *present* value."""
        return _parse(schema, val, True, child_loc, state.new_state() if new_state is None else new_state)

    def child_a(schema: CHSchemaNode, child_loc: LocationId, new_state: _State | None = None) -> ParsedValue | None:
        """Parse an *absent* value; may return ``None``."""
        return _parse_absent(schema, child_loc, state.new_state() if new_state is None else new_state)

    def child_silent(
        schema: CHSchemaNode, val: object, child_loc: LocationId
    ) -> tuple[ParsedValue | None, list[ConceptHierarchyError]]:
        """Trial parse; errors are collected but do not escape."""
        trial = state.silent()
        result = _parse(schema, val, True, child_loc, trial)
        return result, trial.errors

    structural = ParsedStructural(location_id=location_id, schema_node=node, errors=local, value=value)

    # --- this node's own keywords (type, enum, const, min/max, pattern, ...) ---
    own_validator = Draft7Validator(node.shallow_canonical)
    for e in own_validator.iter_errors(value):
        rec(CHSemanticError(e.message, location_id + list(e.absolute_path)))

    # --- object structure ------------------------------------------------
    if isinstance(value, dict):
        _parse_object(node, value, location_id, structural, rec, child_p, child_a, state)

    # --- array structure -------------------------------------------------
    if isinstance(value, list):
        _parse_array(node, value, location_id, structural, rec, child_p, child_silent)

    if isinstance(value, str):
        _parse_string(node, value, location_id, rec, state)

    # --- allOf (all-or-nothing) ------------------------------------------
    if node.all_of:
        _parse_all_of(node, value, location_id, structural, rec, state)

    # --- anyOf (keep all matching branches) ------------------------------
    if node.any_of:
        _parse_any_of(node, value, location_id, structural, rec, child_silent)

    # --- oneOf (exactly one branch) --------------------------------------
    if node.one_of:
        _parse_one_of(node, value, location_id, structural, rec, child_silent, state)

    # --- not -------------------------------------------------------------
    if node.not_ is not None:
        _, trial_errors = child_silent(node.not_, value, location_id)
        if not trial_errors:
            rec(CHSemanticError("Value must not match the schema in 'not'", location_id))

    # --- if / then / else ------------------------------------------------
    if node.if_ is not None:
        _parse_if_then_else(node, value, location_id, structural, child_p, child_silent)

    return structural


# ===========================================================================================================
# Object structure
# ===========================================================================================================


def _parse_object(
    node: CHSchemaNode,
    value: dict,
    location_id: LocationId,
    structural: ParsedStructural,
    rec: Callable[[ConceptHierarchyError], None],
    child_p,
    child_a,
    state: _State,
) -> None:
    matched_keys: set[str] = set()

    # process "properties": [("props", "x"), ("funcs+(Dog)", "x")]
    if node.custom_concept_data_constraints:
        _parse_custom_concept_data(node, value, location_id, structural, matched_keys, rec, state)

    # process "properties": "args"
    if node.custom_object_properties is not None:
        if node.custom_object_properties == "args":
            _parse_evaluation_arguments_of_function(node, value, location_id, structural, matched_keys, rec, state)
        else:
            raise RuntimeError(
                f'Unknown custom_object_properties "{node.custom_object_properties}" that was correctly parsed?!'
            )

    # --- properties -----------------------------------------------------------------------------------
    if node.schema_owner == DomainConceptDefinition.default_value_domain_type_of_domain_concept_functions and all(
        x in node.properties for x in ["interface", "procedure"]
    ):
        new_vars: dict[str, InstantiatedType] = {}
        # First process the `interface` (the prerequisite for the `procedure`'s variable scope) and then the `procedure`
        key = "interface"
        child_schema = node.properties[key]
        if key in value:
            matched_keys.add(key)
            result = child_p(child_schema, value[key], location_id + [key], state.new_state(key))
            if result is not None:
                # Transform interface definition into variables
                assert isinstance(result, ParsedStructural)
                assert isinstance(result.value, dict)
                for new_var, new_var_def in result.value.items():
                    if isinstance(new_var_def, str):
                        new_var_type = new_var_def
                    elif isinstance(new_var_def, list):
                        new_var_type = new_var_def[0]
                    else:
                        raise RuntimeError(
                            "Expected schema does not match the written interface schema of CustomFunction!"
                        )
                    new_var_instantiated_type, errors = state.context.parse_type(
                        new_var_type[2:] if new_var_type.startswith("s:") else new_var_type,
                        state.template_substitution,
                        location_id,
                    )
                    if errors:
                        for err in errors:
                            state.record(err)
                    else:
                        assert new_var_instantiated_type is not None
                        new_vars[new_var] = new_var_instantiated_type
        else:
            result = child_a(child_schema, location_id + [key], state.new_state(key))
        if result is not None:
            structural.properties_parsed[key] = result
        # process procedure
        key = "procedure"
        child_schema = node.properties[key]
        if key in value:
            matched_keys.add(key)
            # replace the variable scope in which the procedure of the CustomFunction is parsed!
            with state.context.replace_variable_scope_for_custom_function(new_vars):
                result = child_p(child_schema, value[key], location_id + [key], state.new_state(key))
        else:
            result = child_a(child_schema, location_id + [key], state.new_state(key))
        if result is not None:
            structural.properties_parsed[key] = result
    else:
        for key, child_schema in node.properties.items():
            matched_keys.add(key)
            if key in value:
                result = child_p(child_schema, value[key], location_id + [key], state.new_state(key))
            else:
                result = child_a(child_schema, location_id + [key], state.new_state(key))
            if result is not None:
                structural.properties_parsed[key] = result

    # --- required -------------------------------------------------------------------------------------
    # A required key is satisfied either by being present or by having been materialised from a default
    # (and hence appearing in properties_parsed).
    for key in node.required:
        if key not in value and key not in structural.properties_parsed:
            rec(CHSemanticError(f'Required property "{key}" is missing', location_id, part=PathPart.VALUE))

    # --- patternProperties ----------------------------------------------------------------------------
    for pattern, child_schema in node.pattern_properties.items():
        regex = re.compile(pattern)
        for key in value:
            if regex.search(key):
                matched_keys.add(key)
                result = child_p(child_schema, value[key], location_id + [key], state.new_state(key))
                if result is not None:
                    structural.pattern_properties_parsed.setdefault(key, []).append((pattern, result))

    # --- additionalProperties -------------------------------------------------------------------------
    if node.additional_properties is not None:
        for key in value:
            if key in matched_keys:
                continue
            if node.additional_properties is False:
                rec(CHSemanticError("Additional property is not allowed", location_id + [key], part=PathPart.KEY))
            elif node.additional_properties is True:
                pass
            else:
                result = child_p(node.additional_properties, value[key], location_id + [key], state.new_state(key))
                if result is not None:
                    structural.additional_properties_parsed[key] = result

    # --- propertyNames (checked only; results are intentionally discarded) -----------------------------
    if node.property_names is not None:
        pn = node.property_names
        if pn.is_custom_type:
            # Default values are not applicable here because it is the appearing/existing/available names of the JSON
            # object's keys that are checked; there is no MISSING case for which a default value/expression can be used.
            for key in value:
                _, errs = state.context.parse_value_against_custom_type_expression(
                    pn.custom_type,
                    pn.provenance,
                    key,
                    location_id + [key],
                    state.template_substitution,
                    state.expansion_depth,
                    state.reading(location_id),
                )
                for err in errs:
                    err.part = PathPart.KEY
                    rec(err)
        else:
            pn_validator = Draft7Validator(pn.shallow_canonical)
            for key in value:
                for e in pn_validator.iter_errors(key):
                    rec(CHSemanticError(e.message, location_id + [key], part=PathPart.KEY))

    # --- dependent schemas ----------------------------------------------------------------------------
    for key, dep_schema in node.dependent_schemas.items():
        if key in value:
            # don't overwrite the state here to keep the upper-level upper_level_object_key in the state!
            result = child_p(dep_schema, value, location_id, state)
            if result is not None:
                structural.dependent_schemas_parsed[key] = result


def _parse_custom_concept_data(
    node: CHSchemaNode,
    concept_data: dict,
    location_id: LocationId,
    structural: ParsedStructural,
    matched_keys: set[str],
    rec: Callable[[ConceptHierarchyError], None],
    state: _State,
):
    constraints_to_satisfy: list[dict[str, InstantiatedType]] = []
    for index, concept_data_constraint in enumerate(node.custom_concept_data_constraints):
        if concept_data_constraint.concept_restriction is not None and any(
            not isinstance(x, InstantiatedType) for x in concept_data_constraint.concept_restriction
        ):
            constraints_to_satisfy.append({})
            continue
        constraints_to_satisfy.append(
            state.context.collect_data(
                concept_data_constraint.concept_restriction,
                concept_data_constraint.for_properties_or_functions,
                concept_data_constraint.include_parent_data,
            )
        )

    local_matches: set[str] = set()
    satisfied_parsed_values: list[dict[str, ParsedCustomValue]] = [{} for _ in node.custom_concept_data_constraints]
    for key, value in concept_data.items():
        if not state.context.is_domain_concept_datum_in_concept_hierarchy(key):
            continue
        key_location_id = location_id + [key]
        constraint_index, value_type, node_value_type = None, None, None
        # Check if the key is contained in one of the constraints;
        #  if the constraint is template dependent, it contains no data to satisfy
        for constraint_index, (constraint_to_satisfy, concept_data_constraint) in enumerate(
            zip(constraints_to_satisfy, node.custom_concept_data_constraints)
        ):
            if key in constraint_to_satisfy:
                value_type, node_value_type = constraint_to_satisfy[key], concept_data_constraint.value
                break
        if value_type is None:
            continue
        local_matches.add(key)
        assert constraint_index is not None
        assert node_value_type is not None
        # create the x-substitution
        substitution: dict[str, ConceptHierarchyTemplateArgument] = {"x": value_type}
        # update the substitution with the template_substitution of the context
        if state.template_substitution is not None:
            substitution.update(state.template_substitution)
        # do the substitution
        assert node_value_type.custom_type is not None
        assert state.context.is_concept(node_value_type.schema_owner)
        substituted_value = state.context.substitute_with_x(
            node_value_type.custom_type, substitution, node_value_type.schema_owner, key_location_id
        )
        # check the expression at that site
        expr_res, errors = state.context.parse_value_against_custom_type_expression(
            substituted_value,
            node_value_type.provenance,
            value,
            key_location_id,
            state.template_substitution,
            state.expansion_depth,
            state.reading(key_location_id),
        )
        for err in errors:
            rec(err)

        # How to convert the Expression value into a ParsedValue?
        parsed = ParsedCustomValue(
            location_id=key_location_id,
            schema_node=node_value_type,
            errors=errors,
            custom_type=substituted_value,
            provenance=node_value_type.provenance,  # TODO: is this the provenance of the schema or of the expression?
            used_default=False,
            default_expr=MISSING,
            expression=expr_res,
        )
        satisfied_parsed_values[constraint_index][key] = parsed

    matched_keys.update(local_matches)
    for constraint, concept_data_constraint in zip(constraints_to_satisfy, node.custom_concept_data_constraints):
        if not concept_data_constraint.require_all_keys:
            continue
        required_keys = set(constraint.keys())
        if not (required_keys <= local_matches):
            rec(
                CHSemanticError(
                    f"The custom concept data constraint {concept_data_constraint}\nwas not satisfied because not all "
                    f"keys were present; missing keys are: {sorted(required_keys - local_matches)}",
                    location_id=location_id,
                )
            )

    structural.custom_concept_data = {}
    for constraint_index, parse_res in enumerate(satisfied_parsed_values):
        for key, parsed_value in parse_res.items():
            structural.custom_concept_data[key] = (parsed_value, constraint_index)


def _parse_evaluation_arguments_of_function(
    node: CHSchemaNode,
    value: dict,
    location_id: LocationId,
    structural: ParsedStructural,
    matched_keys: set[str],
    rec: Callable[[ConceptHierarchyError], None],
    state: _State,
):
    """
    Handle ``"properties": "args"``: the object being parsed is a Function evaluation's argument list.

    **Every key is claimed, whatever the outcome.** ``"properties": "args"`` stands in for a ``properties``
    block that only the Function's interface could have written, so it owes `_parse_object` the same thing a
    ``properties`` block owes it -- the set of keys it accounts for -- or the ``additionalProperties: false``
    beside it rejects each argument in turn. Claiming them on failure too is deliberate: the evaluation
    parser has already said what is wrong, and it names the Function and the argument, which
    "Additional property is not allowed" does not.

    Claiming *all* of them is exact rather than approximate: an argument the Function does not have makes
    the evaluation ill-formed, so a value that parses has none, and a value, that does not, has an error of its own.
    """
    if state.upper_level_object_key is None:
        raise RuntimeError('Can\'t happen that an "args" argument is specified in a non-object key')
    matched_keys.update(value)
    # The location of the *object* the key belongs to, not of the arguments. This node is reached through
    # `additionalProperties`, so `location_id` already ends with the Function name -- and the evaluation
    # parser appends the key itself, to the convention every expression caller uses. Handing it the
    # location it is standing on spelled the key twice: `.../procedure/Condition/Condition/condition`.
    assert location_id and str(location_id[-1]) == state.upper_level_object_key, (location_id, state)
    # The key names the Function whose arguments these are, so an interpretation marker on it is not part of the name.
    # The *location* keeps the marker -- it is what the author wrote.
    _, function_name = split_function_interpretation_marker(state.upper_level_object_key)
    parsed, errors = state.context.parse_function_evaluation(
        function_name,
        value,
        node,
        location_id[:-1],
        state.template_substitution,
        state.expansion_depth,
    )
    for error in errors:
        rec(error)  # may raise StopValidation in fail-fast mode
    if parsed is not None:
        structural.custom_args_evaluation = (parsed, state.upper_level_object_key)


# ===========================================================================================================
# Array structure
# ===========================================================================================================


def _parse_array(
    node: CHSchemaNode,
    value: list,
    location_id: LocationId,
    structural: ParsedStructural,
    rec,
    child_p,
    child_silent,
) -> None:
    # items_parsed is index-correlated with value; None means no schema applied.
    items_parsed: list[ParsedValue | None] = [None] * len(value)

    if isinstance(node.items, list):
        if node.require_all_items and len(value) < len(node.items):
            rec(
                CHSemanticError(
                    f"expected {len(node.items)} items, got {len(value)}: "
                    f'"requireAllItems" requires every position of the tuple to be present',
                    location_id=location_id,
                )
            )
        # Tuple validation.
        for i, item in enumerate(value):
            if i < len(node.items):
                items_parsed[i] = child_p(node.items[i], item, location_id + [i])
            elif node.additional_items is not None:
                if node.additional_items is False:
                    rec(CHSemanticError("Additional item is not allowed", location_id + [i]))
                elif node.additional_items is True:
                    pass  # items_parsed[i] stays None
                else:
                    items_parsed[i] = child_p(node.additional_items, item, location_id + [i])
    elif node.items is not None:
        # Single-schema: all items validated against the same schema.
        for i, item in enumerate(value):
            items_parsed[i] = child_p(node.items, item, location_id + [i])

    structural.items_parsed = items_parsed

    # --- contains --------------------------------------------------------
    if node.contains is not None:
        found = False
        for i, item in enumerate(value):
            result, trial_errors = child_silent(node.contains, item, location_id + [i])
            if not trial_errors and result is not None:
                structural.contains_parsed = result
                found = True
                break
        if not found:
            rec(CHSemanticError("Array does not contain any element matching the 'contains' schema", location_id))


# ===========================================================================================================
# String structure
# ===========================================================================================================


def _parse_string(node: CHSchemaNode, value: str, location_id: LocationId, rec, state: _State) -> None:
    # node.custom_string_format and node.custom_string_constraint must be checked.
    if node.custom_string_format is not None:
        assert node.custom_string_format in {"Concept", "Type"}
        # A `format: "Type"` string may be an expression key.
        # `FunctionComposition` constrains its `propertyNames` to a Type,
        # so the possible marker must be stripper to check whether the remaining key is a Type.
        marker, value_to_check = split_function_interpretation_marker(value)
        if marker is None and value_to_check.startswith("s:"):
            value_to_check = value_to_check[2:]
        if node.custom_string_format == "Concept":
            if not state.context.is_concept(value_to_check):
                rec(
                    CHSemanticError(
                        f'JSON string value "{value}" is not a concept in this Concept Hierarchy',
                        location_id=location_id,
                        part=PathPart.VALUE,
                    )
                )
        else:
            assert node.custom_string_format == "Type"
            if not state.context.is_type(value_to_check, location_id):
                rec(
                    CHSemanticError(
                        f'JSON string value "{value}" is not a Type in this Concept Hierarchy',
                        location_id=location_id,
                        part=PathPart.VALUE,
                    )
                )
        if node.custom_string_constraint is not None:
            # interpret the constraint with the template argument constraint syntax!
            string_constraint_formula = state.context.parse_constraint(node.custom_string_constraint, location_id)
            assert isinstance(string_constraint_formula, NonStructureConstraintFormula)
            if not state.context.validate_string_constraint(string_constraint_formula, value_to_check, location_id):
                rec(
                    CHSemanticError(
                        f'JSON string value "{value}" satisfies the format "{node.custom_string_format}", but does not '
                        f'satisfy the constraint "{node.custom_string_constraint}"',
                        location_id=location_id,
                        part=PathPart.VALUE,
                    )
                )


# ===========================================================================================================
# Composition
# ===========================================================================================================


def _parse_all_of(
    node: CHSchemaNode,
    value: object,
    location_id: LocationId,
    structural: ParsedStructural,
    rec,
    state: _State,
) -> None:
    """All-or-nothing: populate ``all_of_parsed`` only when every branch succeeds."""
    branch_results: list[ParsedValue] = []
    failed_errors: list[ConceptHierarchyError] = []
    all_valid = True

    for branch in node.all_of:
        # Branch errors are always collected in full so that they can be reported together, independently of the
        # parent's fail-fast mode.
        trial = state.silent()
        result = _parse(branch, value, True, location_id, trial)
        if trial.errors or result is None:
            all_valid = False
            failed_errors.extend(trial.errors)
        else:
            branch_results.append(result)

    if all_valid:
        structural.all_of_parsed = branch_results
    else:
        for err in failed_errors:
            rec(err)  # may raise StopValidation in fail-fast mode


def _parse_any_of(
    node: CHSchemaNode,
    value: object,
    location_id: LocationId,
    structural: ParsedStructural,
    rec,
    child_silent,
) -> None:
    """Retain every matching branch."""
    matching: list[ParsedValue] = []
    branch_errors: list[ConceptHierarchyError] = []

    for branch in node.any_of:
        result, errs = child_silent(branch, value, location_id)
        if not errs and result is not None:
            matching.append(result)
        else:
            branch_errors.extend(errs)

    if matching:
        structural.any_of_parsed = matching
    else:
        err = CHSemanticError("Value does not match any schema in 'anyOf'", location_id)
        err.causes.extend(branch_errors)
        rec(err)


def _parse_one_of(
    node: CHSchemaNode,
    value: object,
    location_id: LocationId,
    structural: ParsedStructural,
    rec,
    child_silent,
    state: _State,
) -> None:
    """Exactly one branch must match."""
    matching: list[ParsedValue] = []
    failed_branch_errors: list[ConceptHierarchyError] = []

    for branch in node.one_of:
        result, errs = child_silent(branch, value, location_id)
        if not errs and result is not None:
            matching.append(result)
        else:
            failed_branch_errors.extend(errs)

    # the state.template_substitution is None guards against the cases in which the schema is template dependent
    # but processing of the value/schema is not being done with a ground instantiation!
    if len(matching) == 1 or (len(matching) > 1 and node.is_template_dependent and state.template_substitution is None):
        structural.one_of_parsed = matching[0]
        return

    # The ambiguity the fComp, fEval, and fInst markers exist for: a one-key object keyed by a Function is ambiguous.
    # It could mean a Function composition, evaluation, or instantiation; and depending on the schemas and the
    # expression parser, a schema defined with `oneOf` may match both branches (when the marker is not present).
    #
    # The advice is based on the value's **shape**, not the owning ValueDomain. E.g. `FunctionCompositionRes` is merely
    # the declaration in the schema that has it; any `oneOf` written the same in any ValueDomain has the same issue.
    # A key that already carries a marker is not offered again to the next branch:
    # there the branches are ambiguous for some other reason, and naming the markers would misdirect.
    #
    # Which of the three is *acceptable* depends on the branch types, so the hint does not predict that;
    # all three are selectable, which is what makes the hint worth saying at all.
    addition = ""
    if isinstance(value, dict) and len(value) == 1:
        only_key = next(iter(value))
        assert isinstance(only_key, str)  # this comes from the JSON object, whose keys are always strings
        split_res = split_function_interpretation_marker(only_key)

        if split_res[0] is None and split_res[1] != "" and split_res[1][0].isupper():
            addition = (
                f".\nIf {only_key} names a Function, you may want to say which reading is meant by writing the key as "
                f'"fEval:{only_key}" (a Function evaluation), "fComp:{only_key}" (a FunctionComposition), or '
                f'"fInst:{only_key}" (a Function instantiation).'
            )

    message = (
        "Value does not match any schema in 'oneOf'"
        if not matching
        else f"Value matches {len(matching)} schemas in 'oneOf' (expected exactly 1)" + addition
    )
    err = CHSemanticError(message, location_id)
    err.causes.extend(failed_branch_errors)
    rec(err)


# ===========================================================================================================
# Conditional
# ===========================================================================================================


def _parse_if_then_else(
    node: CHSchemaNode,
    value: object,
    location_id: LocationId,
    structural: ParsedStructural,
    child_p,
    child_silent,
) -> None:
    """Evaluate if/then/else.

    The ``if`` result is retained as metadata; the branch actually taken is retained as a structural child.
    """
    if_result, trial_errors = child_silent(node.if_, value, location_id)
    structural.if_parsed = if_result

    if not trial_errors:
        # if matched → take then branch
        if node.then_ is not None:
            structural.then_else_parsed = child_p(node.then_, value, location_id)
    else:
        # if did not match → take else branch
        if node.else_ is not None:
            structural.then_else_parsed = child_p(node.else_, value, location_id)
