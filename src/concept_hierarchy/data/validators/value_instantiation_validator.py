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

"""Validate a value against a parsed :class:`~ch_schema.ast_nodes.CHSchemaNode`.

The recursion is performed by *this* module, not delegated wholesale to ``jsonschema``: at each node,
``jsonschema`` is only used to check that node's *own* keywords (``type`` for builtins, ``enum``, ``const``,
``minimum``/``maximum``, ``pattern``, ``format``, ``minItems``, ...) via :attr:`CHSchemaNode.shallow_canonical`
(which has every subschema-bearing keyword replaced by ``True``).
Recursion into ``properties``, ``items``, ``allOf``/``anyOf``/``oneOf``/``not``/``if``-``then``-``else``, etc. is done
explicitly, so that:

* every reported error carries a precise path into the *value*, and
* custom-type nodes -- which may appear anywhere, including inside ``anyOf``/``oneOf``/``allOf`` -- are routed to
  :meth:`~ch_schema.context.CHValueContext.check_value` instead of being treated as plain JSON values.

``required`` is also checked explicitly (rather than via ``jsonschema``) so that a missing required property is reported
with ``path`` pointing at the missing key itself (``part=KEY``), not at the containing object.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from jsonschema import Draft7Validator

from concept_hierarchy.data.jsonschema.ast_nodes import CHSchemaNode
from concept_hierarchy.data.jsonschema.internal import StopValidation, record
from concept_hierarchy.errors import CHSemanticError, ConceptHierarchyError, LocationId, PathPart


class _Missing:
    """
    Sentinel to signal "no value is present" / "no default is specified".
    To distinguish from a legitimate JSON ``null``.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<MISSING>"

    def __bool__(self) -> bool:  # pragma: no cover - defensive
        return False


MISSING = _Missing()


class CHValueContext(ABC):
    """
    Context protocol.

    A "context" provides the knowledge that is *not* part of the schema text itself:

    * :class:`CHValueContext` is consulted while a **value** is being checked against a (possibly custom-typed) schema
      node, to decide whether the value (together with its reference-kind and recorded default expression) is acceptable
      for a given custom type.

    Provides the rules needed while checking a *value* against a (possibly custom-typed) schema node.
    """

    @abstractmethod
    def check_value(
        self,
        type_name: str,
        ref: str,
        default_expr: object,
        value: object,
        location_id: LocationId,
    ) -> ConceptHierarchyError | None:
        """
        Check whether ``value`` is acceptable for a custom-typed schema node.

        Args:
            type_name: The custom type's name.
            ref: Either ``"Reference"`` or ``"NoRef"``.
            default_expr: The raw (unvalidated) default-value expression recorded for this node, or :data:`MISSING`
                if none was given.
            value: The value found at ``location_id``, or :data:`MISSING` if no value was present at all
                (e.g. an optional property that was omitted).
            location_id: Location of ``value`` within the value/instance being validated.

        Returns:
            ``None`` if ``value`` is acceptable, otherwise a
            :class:`CHSyntaxError` or :class:`CHSemanticError` describing the problem.
        """


def validate_value(
    value: object, node: CHSchemaNode, context: CHValueContext, location_id: LocationId = None, collect_all: bool = True
) -> list[ConceptHierarchyError]:
    """Validate ``value`` against the schema represented by ``node``.

    Args:
        value: The value to check (e.g. parsed from JSON).
        node: A schema AST produced by :func:`ch_schema.schema_validator.parse_schema`. (Typically you should only call
            this on a schema that came back with no errors from ``parse_schema``.)
        context: Used to validate values found at custom-type nodes; see :class:`~ch_schema.context.CHValueContext`.
        location_id: The starting location in a Concept Hierarchy where the check starts
        collect_all: If ``True`` (default), collect every error found. If ``False``, stop at the first error.

    Returns:
        A list of :class:`~ch_schema.errors.CHSemanticError` (and, in principle,
        :class:`~ch_schema.errors.CHSyntaxError` if ``context.check_value`` returns one), each with ``path`` pointing
        at the offending location *within ``value``*. Empty if ``value`` is valid.
    """
    if location_id is None:
        location_id = []
    errors: list[ConceptHierarchyError] = []
    try:
        _validate(node, value, True, location_id, context, errors, collect_all)
    except StopValidation:
        pass
    return errors


# ---------------------------------------------------------------------------
def _validate(
    node: CHSchemaNode,
    value: object,
    present: bool,
    value_path: LocationId,
    context: CHValueContext,
    errors: list[ConceptHierarchyError],
    collect_all: bool,
) -> None:
    # An absent optional value has nothing to check here.
    # (`required` is handled by the parent object node, which knows the property name.)
    if not present:
        return

    # --- $ref: delegate entirely (draft-07 ignores siblings of $ref) ----
    if node.ref_resolved is not None:
        _validate(node.ref_resolved, value, present, value_path, context, errors, collect_all)
        return

    # --- boolean schema --------------------------------------------------
    if node.is_boolean_schema():
        if node.canonical is False:
            record(errors, collect_all, CHSemanticError("no value is allowed here (schema is `false`)", value_path))
        return

    # --- custom type: delegate to the value context ----------------------
    if node.is_custom_type:
        default_expr = node.default_expr if node.has_default else MISSING
        err = context.check_value(node.custom_type_name, node.ref, default_expr, value, value_path)
        if err is not None:
            record(errors, collect_all, err)
        return

    # --- this node's own keywords (type, enum, const, numeric/string/array
    # size constraints, format, ...) -------------------------------------
    validator = Draft7Validator(node.shallow_canonical)
    for e in validator.iter_errors(value):
        record(errors, collect_all, CHSemanticError(e.message, value_path + list(e.absolute_path)))

    # --- required ----------------------------------------------------------
    if node.required and isinstance(value, dict):
        for key in node.required:
            if key not in value:
                record(
                    errors,
                    collect_all,
                    CHSemanticError("required property is missing", value_path + [key], part=PathPart.KEY),
                )

    # --- object structure --------------------------------------------------
    if isinstance(value, dict):
        _validate_object(node, value, value_path, context, errors, collect_all)

    # --- array structure --------------------------------------------------
    if isinstance(value, list):
        _validate_array(node, value, value_path, context, errors, collect_all)

    # --- composition --------------------------------------------------------
    if node.all_of:
        for sub in node.all_of:
            _validate(sub, value, True, value_path, context, errors, collect_all)

    if node.any_of:
        _validate_any_of(node, value, value_path, context, errors, collect_all)

    if node.one_of:
        _validate_one_of(node, value, value_path, context, errors, collect_all)

    if node.not_ is not None:
        tmp: list[ConceptHierarchyError] = []
        _validate(node.not_, value, True, value_path, context, tmp, True)
        if not tmp:
            record(errors, collect_all, CHSemanticError("Value must not match the schema in 'not'", value_path))

    if node.if_ is not None:
        tmp = []
        _validate(node.if_, value, True, value_path, context, tmp, True)
        branch = node.then_ if not tmp else node.else_
        if branch is not None:
            _validate(branch, value, True, value_path, context, errors, collect_all)


# ---------------------------------------------------------------------------
def _validate_object(
    node: CHSchemaNode,
    value: dict,
    value_path: LocationId,
    context: CHValueContext,
    errors: list[ConceptHierarchyError],
    collect_all: bool,
) -> None:
    matched_keys = set()

    for key, child in node.properties.items():
        matched_keys.add(key)
        _validate(child, value.get(key), key in value, value_path + [key], context, errors, collect_all)

    for pattern, child in node.pattern_properties.items():
        regex = re.compile(pattern)
        for key in value:
            if regex.search(key):
                matched_keys.add(key)
                _validate(child, value[key], True, value_path + [key], context, errors, collect_all)

    if node.additional_properties is not None:
        for key in value:
            if key in matched_keys:
                continue
            if node.additional_properties is False:
                record(
                    errors,
                    collect_all,
                    CHSemanticError("Additional property is not allowed", value_path + [key], part=PathPart.KEY),
                )
            elif node.additional_properties is True:
                pass
            else:
                _validate(
                    node.additional_properties, value[key], True, value_path + [key], context, errors, collect_all
                )

    if node.property_names is not None:
        # property *names* are validated, not their values; report errors
        # against the key itself.
        pn_validator = Draft7Validator(node.property_names.shallow_canonical)
        for key in value:
            if node.property_names.is_custom_type:
                default_expr = node.property_names.default_expr if node.property_names.has_default else MISSING
                err = context.check_value(
                    node.property_names.custom_type_name, node.property_names.ref, default_expr, key, value_path + [key]
                )
                if err is not None:
                    err.part = PathPart.KEY
                    record(errors, collect_all, err)
            else:
                for e in pn_validator.iter_errors(key):
                    record(errors, collect_all, CHSemanticError(e.message, value_path + [key], part=PathPart.KEY))

    for key, child in node.dependent_schemas.items():
        if key in value:
            _validate(child, value, True, value_path, context, errors, collect_all)


def _validate_array(
    node: CHSchemaNode,
    value: list,
    value_path: LocationId,
    context: CHValueContext,
    errors: list[ConceptHierarchyError],
    collect_all: bool,
) -> None:
    if isinstance(node.items, list):
        for i, item in enumerate(value):
            if i < len(node.items):
                _validate(node.items[i], item, True, value_path + [i], context, errors, collect_all)
            elif node.additional_items is not None:
                if node.additional_items is False:
                    record(errors, collect_all, CHSemanticError("Additional item is not allowed", value_path + [i]))
                elif node.additional_items is True:
                    pass
                else:
                    _validate(node.additional_items, item, True, value_path + [i], context, errors, collect_all)
    elif node.items is not None:
        for i, item in enumerate(value):
            _validate(node.items, item, True, value_path + [i], context, errors, collect_all)

    if node.contains is not None:
        found = False
        for i, item in enumerate(value):
            tmp: list[ConceptHierarchyError] = []
            _validate(node.contains, item, True, value_path + [i], context, tmp, True)
            if not tmp:
                found = True
                break
        if not found:
            record(
                errors,
                collect_all,
                CHSemanticError("Array does not contain any element matching the 'contains' schema", value_path),
            )


def _validate_any_of(
    node: CHSchemaNode,
    value: object,
    value_path: LocationId,
    context: CHValueContext,
    errors: list[ConceptHierarchyError],
    collect_all: bool,
) -> None:
    branch_errors: list[list[ConceptHierarchyError]] = []
    for sub in node.any_of:
        tmp: list[ConceptHierarchyError] = []
        _validate(sub, value, True, value_path, context, tmp, True)
        if not tmp:
            return  # at least one branch matched -> success
        branch_errors.append(tmp)

    err = CHSemanticError("Value does not match any schema in 'anyOf'", value_path)
    for be in branch_errors:
        err.causes.extend(be)
    record(errors, collect_all, err)


def _validate_one_of(
    node: CHSchemaNode,
    value: object,
    value_path: LocationId,
    context: CHValueContext,
    errors: list[ConceptHierarchyError],
    collect_all: bool,
) -> None:
    branch_errors: list[list[ConceptHierarchyError]] = []
    matches = 0
    for sub in node.one_of:
        tmp: list[ConceptHierarchyError] = []
        _validate(sub, value, True, value_path, context, tmp, True)
        if not tmp:
            matches += 1
        else:
            branch_errors.append(tmp)

    if matches == 1:
        return

    if matches == 0:
        message = "Value does not match any schema in 'oneOf'"
    else:
        message = f"Value matches {matches} schemas in 'oneOf' (expected exactly 1)"
    err = CHSemanticError(message, value_path)
    for be in branch_errors:
        err.causes.extend(be)
    record(errors, collect_all, err)
