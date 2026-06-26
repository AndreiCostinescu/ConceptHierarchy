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

"""
Parse and validate a CH schema definition.

This module is responsible for:

1. Expanding the two shorthand notations:

   * ``"x"`` -> ``{"type": "x"}`` for builtin types, or
     ``{"type": "x", "referenceType": "NoRef"}`` for a bare custom type name.
   * ``["MyType", "Reference"|"NoRef"]`` -> ``{"type": "MyType", "referenceType": ...}``
     (only valid when ``MyType`` is a custom type).

2. Recursively building a :class:`~ch_schema.ast_nodes.CHSchemaNode` tree that mirrors the (expanded) schema,
    full draft-07 structure included (``properties``, ``items``, ``anyOf``, ``$ref``, ...).

3. For every custom-type node:

   * asking the supplied :class:`~ch_schema.context.CHSchemaContext` whether the custom type name is valid
     (-> :class:`CHSemanticError` if not);
   * validating and extracting the ``referenceType`` extra (default ``"NoRef"``);
   * recording the ``default`` extra *without* validating it (that is a later pass, once all formulae are known);
   * rejecting any other keyword on a custom-type node.

4. Running the draft-07 *meta-schema* over the "safe" canonical form of the schema
   (custom-type subtrees replaced by ``True``) to catch ordinary draft-07 structural mistakes
   (-> :class:`CHSyntaxError`).

All errors carry a ``path`` that points at the corresponding location in the *original* (shorthand) schema
-- e.g. if ``properties.foo`` was written using the string shorthand ``"MyType"``, an "unknown custom type" error has
``path == ["properties", "foo", "type"]`` even though ``"type"`` does not literally appear in the user's input
at that point.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from jsonschema import Draft7Validator

from concept_hierarchy.data.expressions.expression_utils import ValueDomainArgumentReference
from concept_hierarchy.data.jsonschema.parsed_schema import CHSchemaNode, CustomConceptDataConstraint
from concept_hierarchy.data.parsers.string_parser import StringParser
from concept_hierarchy.data.types.concept_hierarchy_types import TypeValue
from concept_hierarchy.data.utils import StopValidation, record
from concept_hierarchy.definitions.concept_definition_domain_concept import ForPropertyOrFunction
from concept_hierarchy.errors import (
    CHSemanticError,
    CHSyntaxError,
    ConceptHierarchyError,
    LocationId,
    PathPart,
    PathSegment,
)

BUILTIN_TYPES = {"null", "boolean", "integer", "number", "string", "array", "object"}

# Keys allowed on a schema node whose "type" is a single custom type name.
# "title"/"description"/"$comment" are plain draft-07 annotation keywords and are harmless to allow through.
CUSTOM_TYPE_EXTRA_KEYS = {"type", "referenceType", "default", "title", "description", "$comment"}

_REF_PATTERN = re.compile(r"^#/(definitions|\$defs)/([^/]+)$")

_MISSING = object()


class CHSchemaValidator(ABC):
    """
    Context protocol.

    A "context" provides the knowledge that is *not* part of the schema text itself:

    * :class:`CHSchemaContext` is consulted while a schema is being **parsed and validated**,
      to decide whether a custom type name refers to a valid template type / instantiated value domain.

    Provides knowledge about which custom types (template types / instantiated value domains) are valid
    while a schema is being *defined*.
    """

    argument_reference_types: set[str] = {
        ValueDomainArgumentReference.NO_REF.value,
        ValueDomainArgumentReference.REF.value,
    }

    @abstractmethod
    def parse_custom_type(
        self, type_name: str, location_id: LocationId, allow_x_as_template_variable: bool
    ) -> TypeValue:
        """
        Parse ``type_name`` into a valid type value in this context.

        Args:
            type_name: The custom type name as written in the schema (i.e. the value of the ``"type"`` keyword).
            location_id: Location of the ``"type"`` keyword in the *original* (shorthand) schema,
                for use in the returned error's ``path``.
            allow_x_as_template_variable: whether to accept x as a template variable or not

        Returns:
            ``None`` if ``type_name`` is invalid, otherwise a :class:`CHSemanticError` explaining why it isn't.

        Raises:
            CHSyntaxError or CHSemanticError upon failure in parsing
        """

    @abstractmethod
    def is_concept(self, concept_name: str) -> bool:
        pass

    @abstractmethod
    def is_boolean_template_variable(self, template_variable_candidate: str) -> bool:
        pass

    @abstractmethod
    def is_integer_template_variable(self, template_variable_candidate: str) -> bool:
        pass

    @abstractmethod
    def is_number_template_variable(self, template_variable_candidate: str) -> bool:
        pass

    @abstractmethod
    def is_string_template_variable(self, template_variable_candidate: str) -> bool:
        pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def parse_schema(
    schema: object, validator: CHSchemaValidator, location_id: LocationId = None, collect_all_errors: bool = True
) -> tuple[CHSchemaNode | None, list[ConceptHierarchyError]]:
    """Parse and validate ``schema``.

    Args:
        schema: The raw schema definition
            (using the shorthand notations described in the module docstring, plus full draft-07).
        validator: Used to validate custom type names (see :class:`~ch_schema.context.CHSchemaContext`).
        location_id: The location in a Concept Hierarchy where the schema is defined (and parsed)
        collect_all_errors: If ``True`` (default), collect every error found. If ``False``, stop at the first error.

    Returns:
        A tuple ``(node, errors)``. ``node`` is the parsed AST (or ``None`` if parsing failed catastrophically before
        any node could be built -- this should be rare). ``errors`` is the (possibly empty) list of
        :class:`~ch_schema.errors.CHSyntaxError` / :class:`~ch_schema.errors.CHSemanticError`.
        If ``errors`` is non-empty, the schema is invalid; ``node`` may still be returned on a best-effort basis
        (with the offending parts treated as "anything goes" placeholders) so that, e.g., editor tooling can keep
        working with a partially-valid schema.
    """
    errors: list[ConceptHierarchyError] = []
    node: CHSchemaNode | None = None
    try:
        if location_id is None:
            location_id = []
        node = _build_node(
            schema, location_id, validator, errors, collect_all_errors, allow_x_as_template_variable=False
        )
        _resolve_local_refs(node, errors, collect_all_errors)
        _check_meta_schema(node, errors, collect_all_errors)
    except StopValidation:
        pass
    return node, errors


# ---------------------------------------------------------------------------
# Shorthand expansion
# ---------------------------------------------------------------------------
def _expand_string_shorthand(raw: str) -> dict:
    if raw in BUILTIN_TYPES:
        return {"type": raw}
    # Bare custom type name -> no default, NoRef.
    return {"type": raw, "referenceType": ValueDomainArgumentReference.NO_REF.value}


def _expand_array_shorthand(
    raw: list,
    location_id: LocationId,
    validator: CHSchemaValidator,
    errors: list[ConceptHierarchyError],
    collect_all_errors: bool,
) -> dict | bool:
    valid_shape = (
        len(raw) == 2
        and isinstance(raw[0], str)
        and isinstance(raw[1], str)
        and raw[1] in validator.argument_reference_types
    )
    if not valid_shape:
        record(
            errors,
            collect_all_errors,
            CHSyntaxError(
                f"Array shorthand must be of the form [<Concept Hierarchy type name>, "
                f'"{ValueDomainArgumentReference.REF.value}"|"{ValueDomainArgumentReference.NO_REF.value}"]',
                location_id,
            ),
        )
        if isinstance(raw[0], list):
            record(
                errors,
                collect_all_errors,
                CHSyntaxError(
                    f'Concept Hierarchy types are not allowed inside a multi-type "type" array (got {raw[0]!r}); use '
                    f"the single-type shorthand or object form for a Concept Hierarchy type\n(+ define a Variant<T...> "
                    f"ValueDomain that has a generic template argument and represents a value of one of the types "
                    f"given as template argument values)",
                    location_id,
                ),
            )
        # Fall back to "anything goes" so callers can keep walking.
        return True

    type_name, ref_kind = raw
    if type_name in BUILTIN_TYPES:
        record(
            errors,
            collect_all_errors,
            CHSyntaxError(
                f"A reference kind ({ref_kind!r}) can only be given for Concept Hierarchy types, not for the builtin "
                f"type {type_name!r}",
                location_id,
            ),
        )
        return {"type": type_name}

    return {"type": type_name, "referenceType": ref_kind}


# ---------------------------------------------------------------------------
# Recursive AST construction
# ---------------------------------------------------------------------------
def _build_node(
    raw: object,
    location_id: LocationId,
    validator: CHSchemaValidator,
    errors: list[ConceptHierarchyError],
    collect_all_errors: bool,
    allow_x_as_template_variable: bool,
) -> CHSchemaNode:
    if isinstance(raw, bool):
        node = CHSchemaNode(location_id=location_id, raw=raw, canonical=raw)
        node.safe_canonical = raw
        node.shallow_canonical = raw
        return node

    if isinstance(raw, str):
        canonical: dict | bool = _expand_string_shorthand(raw)
    elif isinstance(raw, list):
        canonical = _expand_array_shorthand(raw, location_id, validator, errors, collect_all_errors)
    elif isinstance(raw, dict):
        canonical = raw
    else:
        record(
            errors,
            collect_all_errors,
            CHSyntaxError(
                f"A schema must be a boolean (accepts all or nothing), a string (type shorthand), an array "
                f"(Concept Hierarchy type shorthand), or an object; got {type(raw).__name__} at {raw!r}",
                location_id,
            ),
        )
        canonical = True

    if isinstance(canonical, bool):
        node = CHSchemaNode(location_id=location_id, raw=raw, canonical=canonical)
        node.safe_canonical = canonical
        node.shallow_canonical = canonical
        return node

    # ------ BUILD OBJECT NODE ----------------------------------------------

    node = CHSchemaNode(location_id=location_id, raw=raw, canonical=canonical)
    work = dict(canonical)  # local working copy we can pop() from

    type_value = work.get("type", _MISSING)
    node.type_value = None if type_value is _MISSING else type_value

    is_custom = isinstance(type_value, str) and type_value not in BUILTIN_TYPES

    if isinstance(type_value, list):
        valid_types = []
        any_invalid = False
        for i, t in enumerate(type_value):
            if isinstance(t, str) and t in BUILTIN_TYPES:
                valid_types.append(t)
            else:
                any_invalid = True
                record(
                    errors,
                    collect_all_errors,
                    CHSyntaxError(
                        f'Concept Hierarchy types are not allowed inside a multi-type "type" array (got {t!r}); use the'
                        f" single-type shorthand or object form for a Concept Hierarchy type\n(+ define a Variant<T...>"
                        f" ValueDomain that has a generic template argument and represents a value of one of the types "
                        f"given as template argument values)",
                        location_id + ["type", i],
                    ),
                )
        if any_invalid:
            # Drop the invalid entries so the draft-07 meta-schema check below doesn't also complain about the same
            # "type" array (with a less specific error / path).
            if valid_types:
                work["type"] = valid_types
            else:
                work.pop("type", None)

    if is_custom:
        _finish_custom_type_node(
            node, type_value, work, location_id, validator, errors, collect_all_errors, allow_x_as_template_variable
        )
        return node

    return _finish_builtin_node(node, work, location_id, validator, errors, collect_all_errors)


def _finish_custom_type_node(
    node: CHSchemaNode,
    type_name: str,
    work: dict,
    location_id: LocationId,
    validator: CHSchemaValidator,
    errors: list[ConceptHierarchyError],
    collect_all_errors: bool,
    allow_x_as_template_variable: bool,
) -> None:
    node.is_custom_type = True

    try:
        node.custom_type = validator.parse_custom_type(type_name, location_id + ["type"], allow_x_as_template_variable)
    except ConceptHierarchyError as e:
        record(errors, collect_all_errors, e)

    ref = work.get("referenceType", ValueDomainArgumentReference.NO_REF.value)
    if "referenceType" in work and work["referenceType"] not in validator.argument_reference_types:
        record(
            errors,
            collect_all_errors,
            CHSyntaxError(
                f'"referenceType" must be "{ValueDomainArgumentReference.REF.value}" or '
                f'"{ValueDomainArgumentReference.NO_REF.value}", got {work["referenceType"]!r}',
                location_id + ["referenceType"],
            ),
        )
        ref = ValueDomainArgumentReference.NO_REF.value
    node.ref = ref

    if "default" in work:
        node.has_default = True
        node.default_expr = work["default"]
        # NOTE: the default expression is *not* validated here. A later pass, once all Concept Hierarchy instantiation
        # formulae are known, must check `default_expr` against this type's own schema.

    for key in work:
        if key not in CUSTOM_TYPE_EXTRA_KEYS:
            record(
                errors,
                collect_all_errors,
                CHSyntaxError(
                    f"Key {key!r} is not allowed on a Concept Hierarchy type schema "
                    f"(only {sorted(CUSTOM_TYPE_EXTRA_KEYS)} are allowed)",
                    location_id + [key],
                    part=PathPart.KEY,
                ),
            )

    # A custom-type node has no draft-07 structure of its own -- treat it as "anything goes" for both meta-schema
    # checking and (later) shallow per-node value checks (the actual check is delegated to CHValueContext.check_value).
    node.safe_canonical = True
    node.shallow_canonical = True


def _parse_custom_concept_data_constraint(
    constraint: str,
    value_schema: CHSchemaNode,
    require_all_keys: bool,
    validator: CHSchemaValidator,
    location_id: LocationId,
) -> CustomConceptDataConstraint:
    custom_concept_data_parser = StringParser(constraint, location_id)
    if custom_concept_data_parser.try_consume("props"):
        for_properties_of_functions = ForPropertyOrFunction.PROPERTY
    elif custom_concept_data_parser.try_consume("funcs"):
        for_properties_of_functions = ForPropertyOrFunction.FUNCTION
    else:
        raise CHSyntaxError(f'Expected either "props" or "funcs" at the beginning of {constraint}')
    include_parent_data = custom_concept_data_parser.try_consume("+")
    if include_parent_data:
        custom_concept_data_parser.consume("(")
        has_concept_restriction = True
    else:
        has_concept_restriction = custom_concept_data_parser.try_consume("(")
    concept_restriction = []
    if has_concept_restriction:
        # process list of uppercase names separated by ', '
        concept_name = custom_concept_data_parser.parse_upper_case_name()
        if not validator.is_concept(concept_name):
            # FIXME: possibly extend to allow an expanded variadic template argument here
            raise CHSemanticError(
                f"{concept_name} is not a valid concept in this Concept Hierarchy!",
                location_id=location_id,
            )
        concept_restriction.append(concept_name)
        while custom_concept_data_parser.try_consume(", "):
            concept_name = custom_concept_data_parser.parse_upper_case_name()
            if not validator.is_concept(concept_name):
                # FIXME: possibly extend to allow an expanded variadic template argument here
                raise CHSemanticError(
                    f"{concept_name} is not a valid concept in this Concept Hierarchy!",
                    location_id=location_id,
                )
            concept_restriction.append(concept_name)
        custom_concept_data_parser.consume(")")
    return CustomConceptDataConstraint(
        for_properties_of_functions, include_parent_data, concept_restriction, value_schema, require_all_keys
    )


def _finish_builtin_node(
    node: CHSchemaNode,
    work: dict,
    location_id: LocationId,
    validator: CHSchemaValidator,
    errors: list[ConceptHierarchyError],
    collect_all_errors: bool,
) -> CHSchemaNode:
    if "referenceType" in work:
        record(
            errors,
            collect_all_errors,
            CHSyntaxError(
                '"referenceType" is only allowed on custom-type schemas (i.e. when "type" is a single Concept Hierarchy'
                " type)",
                location_id + ["referenceType"],
                part=PathPart.KEY,
            ),
        )
        work.pop("referenceType", None)
    # "default" is a normal draft-07 annotation keyword here; leave it in extra_keywords untouched.

    def child(value: object, *suffix: PathSegment) -> CHSchemaNode:
        return _build_node(value, location_id + list(suffix), validator, errors, collect_all_errors, False)

    def child_with_x(value: object, *suffix: PathSegment) -> CHSchemaNode:
        return _build_node(value, location_id + list(suffix), validator, errors, collect_all_errors, True)

    # --- values that can be literal template variables --------------------
    possibly_literal_template_variable_key_mapping: dict[str, tuple[str, str]] = {
        "minProperties": ("min_properties_def", "integer"),
        "maxProperties": ("max_properties_def", "integer"),
        "minLength": ("min_length_def", "integer"),
        "maxLength": ("max_length_def", "integer"),
        "minItems": ("min_items_def", "integer"),
        "maxItems": ("max_items_def", "integer"),
        "minimum": ("minimum_def", "number"),
        "maximum": ("maximum_def", "number"),
        "exclusiveMinimum": ("exclusive_minimum_def", "number"),
        "exclusiveMaximum": ("exclusive_maximum_def", "number"),
        "multipleOf": ("multiple_of_def", "number"),
    }
    for key, (node_field_key, literal_type) in possibly_literal_template_variable_key_mapping.items():
        # if the data at key is not a string, then let it be processed by the normal validator of the json schema!
        if key in work and isinstance(work[key], str):
            template_variable_candidate = work.pop(key)
            template_type_check_success = True
            if literal_type == "number" and not validator.is_number_template_variable(template_variable_candidate):
                template_type_check_success = False
            elif literal_type == "integer" and not validator.is_integer_template_variable(template_variable_candidate):
                template_type_check_success = False
            elif literal_type == "string" and not validator.is_string_template_variable(template_variable_candidate):
                template_type_check_success = False
            elif literal_type == "boolean" and not validator.is_boolean_template_variable(template_variable_candidate):
                template_type_check_success = False
            if not template_type_check_success:
                record(
                    errors,
                    collect_all_errors,
                    CHSemanticError(
                        f"Can not use a non {literal_type} template variable literal {template_variable_candidate} as a"
                        f' value for "{key}"!',
                        location_id=location_id + [key],
                    ),
                )
            else:
                object.__setattr__(node, node_field_key, template_variable_candidate)

    # --- object structure -------------------------------------------------
    if "properties" in work:
        props = work.pop("properties")
        if isinstance(props, dict):
            for key, sub in props.items():
                node.properties[key] = child(sub, "properties", key)
        elif isinstance(props, list):
            # parse properties+/functions+ formula
            def check_structure(_x):
                if not isinstance(_x, list) or not (2 <= len(_x) <= 3) or not isinstance(props[0], str):
                    return False
                if len(_x) == 2:
                    return True
                return isinstance(_x[2], bool)

            if not (check_structure(props)) and not all(check_structure(x) for x in props):
                record(
                    errors,
                    collect_all_errors,
                    CHSyntaxError(
                        f'"properties" must be an object, a 2- or 3-elem array specifying concept-related data, or an '
                        f"array of 2- or 3-elem arrays that is interpreted as a union of concept-related data!\nGot "
                        f"{props!r}",
                        location_id=location_id + ["properties"],
                    ),
                )
            elif check_structure(props):
                require_all_keys = False if len(props) == 2 else props[2]
                try:
                    node.custom_concept_data_constraints.append(
                        _parse_custom_concept_data_constraint(
                            props[0], child_with_x(props[1], "properties", 1), require_all_keys, validator, location_id
                        )
                    )
                except ConceptHierarchyError as e:
                    record(errors, collect_all_errors, e)
            else:
                for custom_entry_index, custom_constraint_entry in enumerate(props):
                    try:
                        require_all_keys = False if len(custom_constraint_entry) == 2 else custom_constraint_entry[2]
                        new_custom_concept_data_constraint = _parse_custom_concept_data_constraint(
                            custom_constraint_entry[0],
                            child_with_x(custom_constraint_entry[1], "properties", custom_entry_index, 1),
                            require_all_keys,
                            validator,
                            location_id,
                        )
                        for existing_custom_constraint in node.custom_concept_data_constraints:
                            if (
                                existing_custom_constraint.for_properties_or_functions
                                == new_custom_concept_data_constraint.for_properties_or_functions
                            ):
                                custom_constraint_type = (
                                    "props"
                                    if existing_custom_constraint.for_properties_or_functions
                                    == ForPropertyOrFunction.PROPERTY
                                    else "funcs"
                                )
                                # FIXME: allow OR-constraint: allow specifying props(A, B, C) OR props(D, E)
                                raise CHSemanticError(
                                    f"There already is a custom concept data constraint on {custom_constraint_type} in "
                                    f"the list. Currently not allowed to have an OR-constraint on the same custom "
                                    f"constraint type: {custom_constraint_type}. Merge them together!",
                                    location_id=node.location_id + ["properties", custom_entry_index, 0],
                                )
                        node.custom_concept_data_constraints.append(new_custom_concept_data_constraint)
                    except ConceptHierarchyError as e:
                        record(errors, collect_all_errors, e)
        else:
            record(
                errors,
                collect_all_errors,
                CHSyntaxError(
                    f'"properties" must be an object, a 2-elem array specifying concept-related data, or an array of '
                    f"2-elem arrays that is interpreted as a union of concept-related data!\nGot {props!r}",
                    location_id=location_id + ["properties"],
                ),
            )

    if "patternProperties" in work:
        pprops = work.pop("patternProperties")
        if isinstance(pprops, dict):
            for key, sub in pprops.items():
                node.pattern_properties[key] = child(sub, "patternProperties", key)
        else:
            record(
                errors,
                collect_all_errors,
                CHSyntaxError('"patternProperties" must be an object', location_id + ["patternProperties"]),
            )

    if "additionalProperties" in work:
        ap = work.pop("additionalProperties")
        if isinstance(ap, bool):
            node.additional_properties = ap
        else:
            node.additional_properties = child(ap, "additionalProperties")

    if "propertyNames" in work:
        node.property_names = child(work.pop("propertyNames"), "propertyNames")

    if "required" in work:
        req = work.pop("required")
        if isinstance(req, list) and all(isinstance(x, str) for x in req):
            node.required = req
        else:
            record(
                errors,
                collect_all_errors,
                CHSyntaxError('"required" must be an array of strings', location_id=location_id + ["required"]),
            )

    if "dependencies" in work:
        deps = work.pop("dependencies")
        if isinstance(deps, dict):
            residual = {}
            for key, val in deps.items():
                if isinstance(val, list) and all(isinstance(x, str) for x in val):
                    residual[key] = val  # property-dependency, no subschema
                else:
                    node.dependent_schemas[key] = child(val, "dependencies", key)
            if residual:
                work["dependencies"] = residual
        else:
            record(
                errors,
                collect_all_errors,
                CHSyntaxError('"dependencies" must be an object', location_id + ["dependencies"]),
            )

    # --- array structure -------------------------------------------------
    if "items" in work:
        items = work.pop("items")
        if isinstance(items, list):
            node.items = [child(sub, "items", i) for i, sub in enumerate(items)]
        else:
            node.items = child(items, "items")

    if "additionalItems" in work:
        ai = work.pop("additionalItems")
        if isinstance(ai, bool):
            node.additional_items = ai
        else:
            node.additional_items = child(ai, "additionalItems")

    if "contains" in work:
        node.contains = child(work.pop("contains"), "contains")

    # --- composition -------------------------------------------------
    for keyword, attr in (("allOf", "all_of"), ("anyOf", "any_of"), ("oneOf", "one_of")):
        if keyword in work:
            arr = work.pop(keyword)
            if isinstance(arr, list):
                setattr(node, attr, [child(sub, keyword, i) for i, sub in enumerate(arr)])
            else:
                record(
                    errors,
                    collect_all_errors,
                    CHSyntaxError(f'"{keyword}" must be an array of schemas', location_id + [keyword]),
                )

    if "not" in work:
        node.not_ = child(work.pop("not"), "not")

    for keyword, attr in (("if", "if_"), ("then", "then_"), ("else", "else_")):
        if keyword in work:
            setattr(node, attr, child(work.pop(keyword), keyword))

    # --- $defs / definitions / $ref -------------------------------------
    for keyword in ("definitions", "$defs"):
        if keyword in work:
            defs = work.pop(keyword)
            if isinstance(defs, dict):
                for key, sub in defs.items():
                    node.definitions[key] = child(sub, keyword, key)
            else:
                record(
                    errors, collect_all_errors, CHSyntaxError(f'"{keyword}" must be an object', location_id + [keyword])
                )

    if "$ref" in work:
        ref_val = work.pop("$ref")
        if isinstance(ref_val, str):
            node.ref_string = ref_val
        else:
            record(
                errors,
                collect_all_errors,
                CHSyntaxError('"$ref" must be a string', location_id + ["$ref"]),
            )

    # Everything left over (type, enum, const, minimum, pattern, format, title, description, default, ...)
    # is a "plain" draft-07 keyword.
    node.extra_keywords = work

    node.safe_canonical = _build_safe_canonical(node, work)
    # shallow_canonical only ever needs to check THIS node's own, non-structural keywords (type for builtins, enum,
    # const, minimum, pattern, format, minItems/maxItems/uniqueItems, minProperties/maxProperties, multipleOf,
    # property-list "dependencies", ...).
    # All structural/composition keywords (properties, items, additionalProperties, required, allOf/anyOf/oneOf/not/
    # if-then-else, contains, propertyNames, $ref, ...) are handled explicitly by value_validator's manual recursion
    # -- including them here (even as `True` placeholders) would either duplicate errors (e.g. "required",
    # "additionalProperties": false) or actively produce *wrong* results (e.g. "not": true always fails, since every
    # value matches the placeholder schema `true`).
    node.shallow_canonical = dict(work)  # shallow-copy
    return node


# ---------------------------------------------------------------------------
# Derived (jsonschema-compatible) canonical forms
# ---------------------------------------------------------------------------
def _safe_child(child: bool | CHSchemaNode) -> dict | bool:
    """Render ``child`` for inclusion in ``safe_canonical``: booleans pass through unchanged,
    custom-type nodes collapse to ``True`` (their ``safe_canonical`` is already ``True``), everything else recurses."""
    if isinstance(child, bool):
        return child
    return child.safe_canonical


def _build_safe_canonical(node: CHSchemaNode, remaining_keywords: dict) -> dict:
    """Build the full, recursive draft-07-equivalent of ``node``, with every custom-type subtree replaced by ``True``.
    Used only for the structural (meta-schema) check of the schema definition."""
    out = dict(remaining_keywords)  # shallow-copy

    if node.properties:
        out["properties"] = {k: _safe_child(v) for k, v in node.properties.items()}
    if node.pattern_properties:
        out["patternProperties"] = {k: _safe_child(v) for k, v in node.pattern_properties.items()}
    if node.additional_properties is not None:
        out["additionalProperties"] = _safe_child(node.additional_properties)
    if node.property_names is not None:
        out["propertyNames"] = _safe_child(node.property_names)
    if node.required:
        out["required"] = node.required

    if node.items is not None:
        if isinstance(node.items, list):
            out["items"] = [_safe_child(c) for c in node.items]
        else:
            out["items"] = _safe_child(node.items)
    if node.additional_items is not None:
        out["additionalItems"] = _safe_child(node.additional_items)
    if node.contains is not None:
        out["contains"] = _safe_child(node.contains)

    for keyword, branches in (("allOf", node.all_of), ("anyOf", node.any_of), ("oneOf", node.one_of)):
        if branches:
            out[keyword] = [_safe_child(c) for c in branches]
    if node.not_ is not None:
        out["not"] = _safe_child(node.not_)
    for keyword, child in (("if", node.if_), ("then", node.then_), ("else", node.else_)):
        if child is not None:
            out[keyword] = _safe_child(child)

    if node.dependent_schemas:
        deps_out = dict(out.get("dependencies", {}))
        for key, child in node.dependent_schemas.items():
            deps_out[key] = _safe_child(child)
        out["dependencies"] = deps_out

    if node.definitions:
        out["definitions"] = {k: _safe_child(v) for k, v in node.definitions.items()}
    if node.ref_string is not None:
        out["$ref"] = node.ref_string

    return out


# ---------------------------------------------------------------------------
# $ref resolution (local "#/definitions/..." and "#/$defs/..." only)
# ---------------------------------------------------------------------------
def _resolve_local_refs(root: CHSchemaNode, errors: list[ConceptHierarchyError], collect_all_errors: bool) -> None:
    defs_by_name = {}
    for n in root.walk():
        for key, child in n.definitions.items():
            defs_by_name.setdefault(key, child)

    for n in root.walk():
        if n.ref_string is None:
            continue
        m = _REF_PATTERN.match(n.ref_string)
        if m is not None and m.group(2) in defs_by_name:
            n.ref_resolved = defs_by_name[m.group(2)]
        else:
            record(
                errors,
                collect_all_errors,
                CHSyntaxError(
                    f"Cannot resolve $ref {n.ref_string!r} (only local references of the form "
                    f'"#/definitions/<name>" or "#/$defs/<name>" are supported)',
                    n.location_id + ["$ref"],
                ),
            )


# ---------------------------------------------------------------------------
# draft-07 meta-schema check
# ---------------------------------------------------------------------------
def _check_meta_schema(root: CHSchemaNode, errors: list[ConceptHierarchyError], collect_all_errors: bool) -> None:
    """Validate ``root.safe_canonical`` against the draft-07 meta-schema.

    Because shorthand expansion never changes the *path* of a node (it only ever turns the value found at a given path
    into an equivalent object form), and custom-type subtrees collapse to ``True``
    (always valid against the meta-schema), any error path reported here corresponds 1:1 with the path conventions
    used for ``CHSyntaxError`` elsewhere.
    """
    meta_validator = Draft7Validator(Draft7Validator.META_SCHEMA)
    for err in meta_validator.iter_errors(root.safe_canonical):
        record(errors, collect_all_errors, CHSyntaxError(err.message, root.location_id + list(err.path)))
