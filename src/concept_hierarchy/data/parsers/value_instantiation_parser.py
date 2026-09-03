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
carrying ``default_expr``.  In that case :attr:`~instantiated_value.ParsedCustomValue.used_default` is ``True``, the
context is asked to parse ``default_expr``, and the property counts as present for the purposes of ``required``.

``allOf`` is all-or-nothing: if any branch fails, the errors of all failing branches are reported on the parent and
``all_of_parsed`` is left empty.  ``anyOf`` retains every matching branch.  ``if_parsed`` is metadata only and is
excluded from :meth:`~instantiated_value.ParsedValue.iter_children` and
:meth:`~instantiated_value.ParsedValue.is_valid`.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from jsonschema import Draft7Validator

from concept_hierarchy.data.expressions.expression import Expression
from concept_hierarchy.data.expressions.expression_utils import ExpressionProvenance
from concept_hierarchy.data.expressions.instantiated_value import ParsedCustomValue, ParsedStructural, ParsedValue
from concept_hierarchy.data.jsonschema.parsed_schema import CHSchemaNode
from concept_hierarchy.data.type_template_variables.constraint_formula import (
    NonStructureConstraintFormula,
    TemplateConstraintFormula,
)
from concept_hierarchy.data.types.concept_hierarchy_types import TypeValue
from concept_hierarchy.data.utils import MISSING, StopValidation, record
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
    ) -> tuple[Expression | None, list[ConceptHierarchyError]]:
        """Parse and validate ``value`` as an expression of ``custom_type``.

        Returns:
            ``(expression, errors)``.  ``expression`` is ``None`` when parsing failed; ``errors`` lists *every* problem
            found, not just the first.
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


# ===========================================================================================================
# Traversal state
# ===========================================================================================================


@dataclass
class _State:
    """Accumulator threaded through the traversal.

    ``errors`` is the global error list for the current (sub-)parse; a *silent* sub-state is used for trial branches so
    that their errors do not escape.
    """

    context: ValueInstantiationContext
    errors: list[ConceptHierarchyError] = field(default_factory=list)
    collect_all_errors: bool = True

    def record(self, err: ConceptHierarchyError) -> None:
        """Record ``err`` globally; raises :class:`StopValidation` in fail-fast mode."""
        record(self.errors, self.collect_all_errors, err)

    def silent(self) -> _State:
        """A sub-state whose errors are collected in full and do not escape."""
        return _State(self.context, [], True)


# ===========================================================================================================
# Entry points
# ===========================================================================================================


def parse_value(
    value: object,
    node: CHSchemaNode,
    context: ValueInstantiationContext,
    location_id: LocationId | None = None,
    collect_all_errors: bool = True,
) -> tuple[ParsedValue, list[ConceptHierarchyError]]:
    """Parse ``value`` against ``node``.

    Args:
        value: The Python object to parse (JSON-decoded).
        node: Schema AST from ``parse_schema``, which should itself have parsed without errors.
        context: Handles custom-type leaves.
        location_id: Starting location in the Concept Hierarchy (``[]`` at the root).
        collect_all_errors: ``True`` to collect every error, ``False`` to stop at the first one.

    Returns:
        ``(result, errors)``.  ``result`` is always a tree, even on failure.  ``errors`` is the authoritative error
        list; do not reconstruct it by walking ``result``, because trial branches and failed ``allOf`` branches
        deliberately do not attach their errors to retained nodes.
    """
    if location_id is None:
        location_id = []

    state = _State(context, [], collect_all_errors)
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
    collect_all_errors: bool = True,
) -> list[ConceptHierarchyError]:
    """Error projection of :func:`parse_value`, for call sites that discard the result tree.

    This performs the full parse; it is not cheaper.
    """
    _, errors = parse_value(value, node, context, location_id, collect_all_errors)
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
    if node.has_default and node.parsed_default_expr is not None:
        expression = node.parsed_default_expr
    if not used_default:
        expression, errs = state.context.parse_value_against_custom_type_expression(
            node.custom_type, node.provenance, value, location_id
        )
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


def _parse_structural(node: CHSchemaNode, value: object, location_id: LocationId, state: _State) -> ParsedStructural:
    local: list[ConceptHierarchyError] = []

    def rec(err: ConceptHierarchyError) -> None:
        """Record an error on this node and in the global accumulator."""
        local.append(err)
        state.record(err)

    def child_p(schema: CHSchemaNode, val: object, child_loc: LocationId) -> ParsedValue | None:
        """Parse a *present* value."""
        return _parse(schema, val, True, child_loc, state)

    def child_a(schema: CHSchemaNode, child_loc: LocationId) -> ParsedValue | None:
        """Parse an *absent* value; may return ``None``."""
        return _parse_absent(schema, child_loc, state)

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
        _parse_one_of(node, value, location_id, structural, rec, child_silent)

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
    rec,
    child_p,
    child_a,
    state: _State,
) -> None:
    matched_keys: set[str] = set()

    # --- properties -----------------------------------------------------------------------------------
    for key, child_schema in node.properties.items():
        matched_keys.add(key)
        if key in value:
            result = child_p(child_schema, value[key], location_id + [key])
        else:
            result = child_a(child_schema, location_id + [key])
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
                result = child_p(child_schema, value[key], location_id + [key])
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
                result = child_p(node.additional_properties, value[key], location_id + [key])
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
                    pn.custom_type, pn.provenance, key, location_id + [key]
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
            result = child_p(dep_schema, value, location_id)
            if result is not None:
                structural.dependent_schemas_parsed[key] = result


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
        value_to_check = value[2:] if value.startswith("s:") else value
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

    if len(matching) == 1:
        structural.one_of_parsed = matching[0]
        return

    message = (
        "Value does not match any schema in 'oneOf'"
        if not matching
        else f"Value matches {len(matching)} schemas in 'oneOf' (expected exactly 1)"
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
