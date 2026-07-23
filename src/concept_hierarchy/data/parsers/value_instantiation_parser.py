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

"""Parse and validate a Python (JSON-decoded) value against a
:class:`~parsed_schema.CHSchemaNode`, producing a typed
:class:`~parsed_value.ParsedValue` tree.

The parser simultaneously validates (collecting
:class:`~errors.ConceptHierarchyError` instances) and constructs a rich
result tree that downstream code can walk to:

* check overall validity (:meth:`~parsed_value.ParsedValue.is_valid`),
* traverse sub-expressions (:meth:`~parsed_value.ParsedValue.walk`),
* replace template variables in expression leaves, and
* type-check individual :class:`~expression.Expression` objects.

Recursion strategy
------------------
Like :mod:`value_instantiation_validator`, every structural keyword is
handled manually — ``jsonschema`` is only used to check a node's *own*
leaf-level constraints via :attr:`~parsed_schema.CHSchemaNode.shallow_canonical`.
This ensures custom-type leaves are routed to
:meth:`~value_instantiation_validator.CHValueValidator.parse_custom_type`
regardless of where they appear in the schema.

Absent-value / default semantics
---------------------------------
An absent optional property is only materialized in the result tree when its
schema (or an ``anyOf`` branch of its schema) is a custom-type node with a
``default_expr``.  In that case :attr:`~parsed_value.ParsedCustomValue.used_default`
is ``True`` and the context is asked to parse ``default_expr``.  Absent
properties with no applicable default are simply omitted from
:attr:`~parsed_value.ParsedStructural.properties_parsed`.

``allOf`` is **all-or-nothing**: if any branch fails, errors from *all*
failed branches are reported on the parent node and
:attr:`~parsed_value.ParsedStructural.all_of_parsed` is left empty.

``anyOf`` keeps **all** matching branches in
:attr:`~parsed_value.ParsedStructural.any_of_parsed`.

``if_parsed`` on :class:`~parsed_value.ParsedStructural` is metadata only
and is **not** included in :meth:`~parsed_value.ParsedValue.iter_children`
or :meth:`~parsed_value.ParsedValue.is_valid`.
"""

from __future__ import annotations

import re

from jsonschema import Draft7Validator

from concept_hierarchy.data.expressions.instantiated_value import ParsedCustomValue, ParsedStructural, ParsedValue
from concept_hierarchy.data.jsonschema.parsed_schema import CHSchemaNode
from concept_hierarchy.data.utils import StopValidation, record
from concept_hierarchy.data.validators.value_instantiation_validator import MISSING, CHValueValidator
from concept_hierarchy.errors import CHSemanticError, ConceptHierarchyError, LocationId, PathPart


def parse_value(
    value: object,
    node: CHSchemaNode,
    context: CHValueValidator,
    location_id: LocationId = None,
    collect_all_errors: bool = True,
) -> ParsedValue:
    """Parse ``value`` against ``node``, returning a fully annotated result tree.

    Args:
        value: The Python object to parse (JSON-decoded).
        node: Schema AST from ``parse_schema``.  Should have no schema-level
            errors (i.e. come from a successful :func:`~jsonschema_parser.parse_schema`
            call).
        context: Provides :meth:`~CHValueValidator.parse_custom_type` for
            custom-type leaves.
        location_id: Starting location in the Concept Hierarchy (``[]`` if
            parsing from the root).
        collect_all_errors: ``True`` (default) to collect every error;
            ``False`` to stop at the first error.

    Returns:
        A :class:`~parsed_value.ParsedValue` tree — always, even on errors.
        Inspect :meth:`~parsed_value.ParsedValue.is_valid` and
        :meth:`~parsed_value.ParsedValue.walk` to find all problems.
    """
    if location_id is None:
        location_id = []

    global_errors: list[ConceptHierarchyError] = []
    result: ParsedValue | None = None
    try:
        result = _parse(node, value, True, location_id, context, global_errors, collect_all_errors)
    except StopValidation:
        pass

    # Guarantee a non-None return even when parsing was cut short.
    if result is None:
        result = ParsedStructural(location_id=location_id, schema_node=node, errors=global_errors, value=value)
    return result


# ---------------------------------------------------------------------------
# Core dispatcher
# ---------------------------------------------------------------------------


def _parse(
    node: CHSchemaNode,
    value: object,
    present: bool,
    location_id: LocationId,
    context: CHValueValidator,
    global_errors: list[ConceptHierarchyError],
    collect_all_errors: bool,
) -> ParsedValue | None:
    """Recursively parse ``value`` against ``node``.

    Returns ``None`` when ``present=False`` and no default fills the gap —
    the property should be omitted from the parent's ``properties_parsed``.
    """
    if not present:
        return _parse_absent(node, location_id, context, global_errors, collect_all_errors)

    # Follow $ref transparently (draft-07: siblings of $ref are ignored).
    if node.ref_resolved is not None:
        return _parse(node.ref_resolved, value, True, location_id, context, global_errors, collect_all_errors)

    # Boolean schema.
    if node.is_boolean_schema:
        local: list[ConceptHierarchyError] = []
        if node.canonical is False:
            err = CHSemanticError("No value is allowed here (schema is `false`)", location_id)
            local.append(err)
            record(global_errors, collect_all_errors, err)
        return ParsedStructural(location_id=location_id, schema_node=node, errors=local, value=value)

    # Custom-type leaf.
    if node.is_custom_type:
        return _parse_custom(node, value, location_id, context, global_errors, collect_all_errors)

    # Builtin / structural / composite.
    return _parse_structural(node, value, location_id, context, global_errors, collect_all_errors)


# ---------------------------------------------------------------------------
# Absent-value handling
# ---------------------------------------------------------------------------


def _parse_absent(
    node: CHSchemaNode,
    location_id: LocationId,
    context: CHValueValidator,
    global_errors: list[ConceptHierarchyError],
    collect_all_errors: bool,
) -> ParsedValue | None:
    """Handle an absent (not present) optional property.

    Returns a :class:`~parsed_value.ParsedCustomValue` with
    ``used_default=True`` if the node (or the first accepting ``anyOf``
    branch) is a custom-type node with a ``default_expr``.
    Returns ``None`` otherwise — the property is simply omitted.
    """
    if node.ref_resolved is not None:
        return _parse_absent(node.ref_resolved, location_id, context, global_errors, collect_all_errors)

    # Boolean schemas never carry a default.
    if node.is_boolean_schema:
        return None

    # Custom-type: only call the context when a default is defined.
    if node.is_custom_type:
        if not node.has_default:
            return None
        return _parse_custom(node, MISSING, location_id, context, global_errors, collect_all_errors)

    # Composite node: try each anyOf branch silently for the first one
    # that accepts MISSING (i.e. a custom-type branch with a default).
    if node.any_of:
        for branch in node.any_of:
            silent: list[ConceptHierarchyError] = []
            result = _parse_absent(branch, location_id, context, silent, True)
            if result is not None and not silent:
                return result

    return None


# ---------------------------------------------------------------------------
# Custom-type leaf
# ---------------------------------------------------------------------------


def _parse_custom(
    node: CHSchemaNode,
    value: object,  # may be MISSING when used_default=True
    location_id: LocationId,
    context: CHValueValidator,
    global_errors: list[ConceptHierarchyError],
    collect_all_errors: bool,
) -> ParsedCustomValue:
    local: list[ConceptHierarchyError] = []
    used_default = value is MISSING
    default_expr = node.default_expr if node.has_default else MISSING

    expression, errs = context.parse_custom_type(node.custom_type, node.provenance, default_expr, value, location_id)
    for err in errs:
        local.append(err)
        record(global_errors, collect_all_errors, err)

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


# ---------------------------------------------------------------------------
# Structural / composite node
# ---------------------------------------------------------------------------


def _parse_structural(
    node: CHSchemaNode,
    value: object,
    location_id: LocationId,
    context: CHValueValidator,
    global_errors: list[ConceptHierarchyError],
    collect_all_errors: bool,
) -> ParsedStructural:
    local: list[ConceptHierarchyError] = []

    def rec(err: ConceptHierarchyError) -> None:
        """Record an error on this node and in the global accumulator."""
        local.append(err)
        record(global_errors, collect_all_errors, err)

    def child_p(schema: CHSchemaNode, val: object, child_loc: LocationId) -> ParsedValue | None:
        """Parse a *present* value at child_loc."""
        return _parse(schema, val, True, child_loc, context, global_errors, collect_all_errors)

    def child_a(schema: CHSchemaNode, child_loc: LocationId) -> ParsedValue | None:
        """Parse an *absent* value at child_loc (may return None)."""
        return _parse_absent(schema, child_loc, context, global_errors, collect_all_errors)

    def child_silent(
        schema: CHSchemaNode, val: object, child_loc: LocationId
    ) -> tuple[ParsedValue | None, list[ConceptHierarchyError]]:
        """Parse silently — errors do not propagate to global_errors."""
        silent: list[ConceptHierarchyError] = []
        result = _parse(schema, val, True, child_loc, context, silent, True)
        return result, silent

    structural = ParsedStructural(location_id=location_id, schema_node=node, errors=local, value=value)

    # --- own keyword constraints (type, enum, const, min/max, pattern, …) ---
    own_validator = Draft7Validator(node.shallow_canonical)
    for e in own_validator.iter_errors(value):
        rec(CHSemanticError(e.message, location_id + list(e.absolute_path)))

    # --- object structure ------------------------------------------------
    if isinstance(value, dict):
        _parse_object(node, value, location_id, structural, rec, child_p, child_a, context)

    # --- array structure -------------------------------------------------
    if isinstance(value, list):
        _parse_array(node, value, location_id, structural, rec, child_p, child_silent)

    # --- allOf (all-or-nothing) ------------------------------------------
    if node.all_of:
        _parse_all_of(node, value, location_id, structural, rec, context, global_errors, collect_all_errors)

    # --- anyOf (keep all matching branches) ------------------------------
    if node.any_of:
        _parse_any_of(node, value, location_id, structural, rec, child_silent)

    # --- oneOf (exactly one branch) --------------------------------------
    if node.one_of:
        _parse_one_of(node, value, location_id, structural, rec, child_silent)

    # --- not -------------------------------------------------------------
    if node.not_ is not None:
        _, silent = child_silent(node.not_, value, location_id)
        if not silent:
            rec(CHSemanticError("Value must not match the schema in 'not'", location_id))

    # --- if / then / else ------------------------------------------------
    if node.if_ is not None:
        _parse_if_then_else(node, value, location_id, structural, child_p, child_silent)

    return structural


# ---------------------------------------------------------------------------
# Object structure
# ---------------------------------------------------------------------------


def _parse_object(
    node: CHSchemaNode,
    value: dict,
    location_id: LocationId,
    structural: ParsedStructural,
    rec,
    child_p,
    child_a,
    context: CHValueValidator,
) -> None:
    matched_keys: set[str] = set()

    # --- properties (and require_all_properties) -------------------------
    for key, child_schema in node.properties.items():
        matched_keys.add(key)
        if key in value:
            result = child_p(child_schema, value[key], location_id + [key])
        else:
            result = child_a(child_schema, location_id + [key])

        if result is not None:
            structural.properties_parsed[key] = result
        elif node.require_all_properties:
            # require_all_properties: absent and no default fills it in.
            rec(CHSemanticError(f'Required property "{key}" is missing', location_id, part=PathPart.VALUE))

    # --- required (explicit list) ----------------------------------------
    # A required key is satisfied if it is present in the value OR was filled
    # in from a default (and therefore appears in properties_parsed).
    for key in node.required:
        if key not in value and key not in structural.properties_parsed:
            rec(CHSemanticError(f'Required property "{key}" is missing', location_id, part=PathPart.VALUE))

    # --- patternProperties -----------------------------------------------
    for pattern, child_schema in node.pattern_properties.items():
        regex = re.compile(pattern)
        for key in value:
            if regex.search(key):
                matched_keys.add(key)
                result = child_p(child_schema, value[key], location_id + [key])
                if result is not None:
                    structural.pattern_properties_parsed.setdefault(key, []).append((pattern, result))

    # --- additionalProperties --------------------------------------------
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

    # --- propertyNames (validate only, result discarded per design) ------
    if node.property_names is not None:
        pn = node.property_names
        if pn.is_custom_type:
            for key in value:
                default_expr = pn.default_expr if pn.has_default else MISSING
                _, errs = context.parse_custom_type(
                    pn.custom_type, pn.provenance, default_expr, key, location_id + [key]
                )
                for err in errs:
                    err.part = PathPart.KEY
                    rec(err)
        else:
            pn_validator = Draft7Validator(pn.shallow_canonical)
            for key in value:
                for e in pn_validator.iter_errors(key):
                    rec(CHSemanticError(e.message, location_id + [key], part=PathPart.KEY))

    # --- dependent schemas -----------------------------------------------
    for key, dep_schema in node.dependent_schemas.items():
        if key in value:
            result = child_p(dep_schema, value, location_id)
            if result is not None:
                structural.dependent_schemas_parsed[key] = result


# ---------------------------------------------------------------------------
# Array structure
# ---------------------------------------------------------------------------


def _parse_array(
    node: CHSchemaNode,
    value: list,
    location_id: LocationId,
    structural: ParsedStructural,
    rec,
    child_p,
    child_silent,
) -> None:
    # items_parsed is index-correlated with value: items_parsed[i] for value[i].
    # None means no schema applied to value[i].
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
            result, silent = child_silent(node.contains, item, location_id + [i])
            if not silent and result is not None:
                structural.contains_parsed = result
                found = True
                break
        if not found:
            rec(CHSemanticError("Array does not contain any element matching the 'contains' schema", location_id))


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def _parse_all_of(
    node: CHSchemaNode,
    value: object,
    location_id: LocationId,
    structural: ParsedStructural,
    rec,
    context: CHValueValidator,
    global_errors: list[ConceptHierarchyError],
    collect_all_errors: bool,
) -> None:
    """All-or-nothing: populate all_of_parsed only when every branch succeeds."""
    branch_results: list[ParsedValue] = []
    failed_errors: list[ConceptHierarchyError] = []
    all_valid = True

    for branch in node.all_of:
        temp: list[ConceptHierarchyError] = []
        # Always collect all branch errors regardless of the parent's mode,
        # so we can report them together when the overall allOf fails.
        result = _parse(branch, value, True, location_id, context, temp, True)
        if temp or result is None:
            all_valid = False
            failed_errors.extend(temp)
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
    """Keep all matching anyOf branches."""
    matching: list[ParsedValue] = []
    all_branch_errors: list[ConceptHierarchyError] = []

    for branch in node.any_of:
        result, silent = child_silent(branch, value, location_id)
        if not silent and result is not None:
            matching.append(result)
        else:
            all_branch_errors.extend(silent)

    if matching:
        structural.any_of_parsed = matching
    else:
        err = CHSemanticError("Value does not match any schema in 'anyOf'", location_id)
        err.causes.extend(all_branch_errors)
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
        result, silent = child_silent(branch, value, location_id)
        if not silent and result is not None:
            matching.append(result)
        else:
            failed_branch_errors.extend(silent)

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


# ---------------------------------------------------------------------------
# Conditional
# ---------------------------------------------------------------------------


def _parse_if_then_else(
    node: CHSchemaNode,
    value: object,
    location_id: LocationId,
    structural: ParsedStructural,
    child_p,
    child_silent,
) -> None:
    """Evaluate if/then/else.  Store the if result as metadata and the taken
    branch result as a structural child."""
    if_result, silent = child_silent(node.if_, value, location_id)
    # Store the if evaluation result as metadata regardless of whether it
    # matched — callers can inspect it, but it does not affect is_valid().
    structural.if_parsed = if_result

    if not silent:
        # if matched → take then branch
        if node.then_ is not None:
            structural.then_else_parsed = child_p(node.then_, value, location_id)
    else:
        # if did not match → take else branch
        if node.else_ is not None:
            structural.then_else_parsed = child_p(node.else_, value, location_id)
