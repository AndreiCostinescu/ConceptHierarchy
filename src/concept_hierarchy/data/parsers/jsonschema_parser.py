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
     ``{"type": "x", "provenance": "Any"}`` for a bare custom type name.
   * ``["MyType", "Addr"|"Any"]`` -> ``{"type": "MyType", "provenance": ...}``
     (only valid when ``MyType`` is a custom type).

2. Recursively building a :class:`~ch_schema.ast_nodes.CHSchemaNode` tree that mirrors the (expanded) schema,
    full draft-07 structure included (``properties``, ``items``, ``anyOf``, ``$ref``, ...).

3. For every custom-type node:

   * asking the supplied :class:`~ch_schema.context.CHSchemaContext` whether the custom type name is valid
     (-> :class:`CHSemanticError` if not);
   * validating and extracting the ``provenance`` extra (default ``"Any"``);
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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Sequence

from jsonschema import Draft7Validator

from concept_hierarchy.data.expressions.expression_utils import ValueDomainArgumentProvenance
from concept_hierarchy.data.jsonschema.parsed_schema import (
    CHSchemaNode,
    CrossSchemaReference,
    CustomConceptDataConstraint,
)
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

JSONSCHEMA_STRUCTURE_KEYS = {"allOf", "anyOf", "oneOf", "not", "if", "then", "else"}

# Keys allowed on a schema node whose "type" is a single custom type name.
# "title"/"description"/"$comment" are plain draft-07 annotation keywords and are harmless to allow through.
CUSTOM_TYPE_EXTRA_KEYS = {"type", "provenance", "default", "title", "description", "$comment"}

BUILTIN_NODE_EXTRA_KEYS = {
    "type",
    "const",
    "enum",
    "default",
    "title",
    "description",
    "examples",
    "readOnly",
    "writeOnly",
    # number-specific keys
    "multipleOf",
    "maximum",
    "exclusiveMaximum",
    "minimum",
    "exclusiveMinimum",
    # string-specific keys
    "maxLength",
    "minLength",
    "pattern",
    "format",
    "contentMediaType",
    "contentEncoding",
    # array-specific keys
    "items",
    "additionalItems",
    "maxItems",
    "minItems",
    "uniqueItems",
    "contains",
    # object-specific keys
    "maxProperties",
    "minProperties",
    "required",
    "properties",
    "patternProperties",
    "additionalProperties",
    "propertyNames",
    "dependencies",
    # reference-specific keys
    "definitions",
    "$defs",
}

_DEFINITION_KEYWORDS = ("definitions", "$defs")
"""The two spellings of the same container; `CHSchemaNode` keeps both in :attr:`definitions`."""

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

    argument_provenance_types: set[str] = {
        ValueDomainArgumentProvenance.ANY.value,
        ValueDomainArgumentProvenance.ADDR.value,
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
    def set_identifier_where_types_are_defined(self, identifier: str) -> None:
        pass

    @abstractmethod
    def clear_identifier_where_types_are_defined(self) -> None:
        pass

    @abstractmethod
    def is_concept(self, concept_name: str) -> bool:
        pass

    @abstractmethod
    def is_template_variable(self, template_variable_name: str) -> bool:
        pass

    @abstractmethod
    def is_type_template_variable(self, template_variable_candidate: str) -> bool:
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


@dataclass
class _State:
    """Accumulator threaded through the traversal."""

    context: CHSchemaValidator
    errors: list[ConceptHierarchyError] = field(default_factory=list)
    collect_all_errors: bool = True
    called_from_object_schema: bool = False

    def record(self, err: ConceptHierarchyError) -> None:
        """Record ``err`` globally; raises :class:`StopValidation` in fail-fast mode."""
        record(self.errors, self.collect_all_errors, err)

    def new_state(self, called_from_object_schema: bool) -> _State:
        return replace(self, called_from_object_schema=called_from_object_schema)


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
    state: _State = _State(validator, errors, collect_all_errors)
    node: CHSchemaNode | None = None
    try:
        if location_id is None:
            location_id = []
        node = _build_node(schema, location_id, state, allow_x_as_template_variable=False)
        _resolve_local_refs(node, state)
        _check_meta_schema(node, state)
    except StopValidation:
        pass
    return node, errors


# ---------------------------------------------------------------------------
# Shorthand expansion
# ---------------------------------------------------------------------------
def _expand_string_shorthand(raw: str) -> dict:
    if raw in BUILTIN_TYPES:
        return {"type": raw}
    # Bare custom type name -> no default, ANY.
    return {"type": raw, "provenance": ValueDomainArgumentProvenance.ANY.value}


def _expand_array_shorthand(raw: list, location_id: LocationId, state: _State) -> dict | bool:
    valid_shape = (
        len(raw) == 2
        and isinstance(raw[0], str)
        and isinstance(raw[1], str)
        and raw[1] in state.context.argument_provenance_types
    )
    if not valid_shape:
        state.record(
            CHSyntaxError(
                f"Array shorthand must be of the form [<Concept Hierarchy type name>, "
                f'"{ValueDomainArgumentProvenance.ADDR.value}"|"{ValueDomainArgumentProvenance.ANY.value}"]',
                location_id,
            )
        )
        if isinstance(raw[0], list):
            state.record(
                CHSyntaxError(
                    f'Concept Hierarchy types are not allowed inside a multi-type "type" array (got {raw[0]!r}); use '
                    f"the single-type shorthand or object form for a Concept Hierarchy type\n(+ define a Variant<T...> "
                    f"ValueDomain that has a generic template argument and represents a value of one of the types "
                    f"given as template argument values)",
                    location_id,
                )
            )
        # Fall back to "anything goes" so callers can keep walking.
        return True

    type_name, provenance_type = raw
    if type_name in BUILTIN_TYPES:
        state.record(
            CHSyntaxError(
                f"A provenance character ({provenance_type!r}) can only be given for Concept Hierarchy types, not for "
                f"the builtin type {type_name!r}",
                location_id,
            )
        )
        return {"type": type_name}

    return {"type": type_name, "provenance": provenance_type}


# ---------------------------------------------------------------------------
# Recursive AST construction
# ---------------------------------------------------------------------------
def _build_node(
    raw: object, location_id: LocationId, state: _State, allow_x_as_template_variable: bool
) -> CHSchemaNode:
    if isinstance(raw, bool):
        node = CHSchemaNode(location_id=location_id, raw=raw, canonical=raw)
        node.safe_canonical = raw
        node.shallow_canonical = raw
        return node

    if isinstance(raw, str):
        canonical: dict | bool = _expand_string_shorthand(raw)
    elif isinstance(raw, list):
        canonical = _expand_array_shorthand(raw, location_id, state)
    elif isinstance(raw, dict):
        canonical = raw
    else:
        state.record(
            CHSyntaxError(
                f"A schema must be a boolean (accepts all or nothing), a string (type shorthand), an array "
                f"(Concept Hierarchy type shorthand), or an object; got {type(raw).__name__} at {raw!r}",
                location_id,
            )
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
                state.record(
                    CHSyntaxError(
                        f'Concept Hierarchy types are not allowed inside a multi-type "type" array (got {t!r}); use the'
                        f" single-type shorthand or object form for a Concept Hierarchy type\n(+ define a Variant<T...>"
                        f" ValueDomain that has a generic template argument and represents a value of one of the types "
                        f"given as template argument values)",
                        location_id + ["type", i],
                    )
                )
        if any_invalid:
            # Drop the invalid entries so the draft-07 meta-schema check below doesn't also complain about the same
            # "type" array (with a less specific error / path).
            if valid_types:
                work["type"] = valid_types
            else:
                work.pop("type", None)

    if is_custom:
        _finish_custom_type_node(node, type_value, work, location_id, state, allow_x_as_template_variable)
        return node

    return _finish_builtin_node(node, work, location_id, state)


def _finish_custom_type_node(
    node: CHSchemaNode,
    type_name: str,
    work: dict,
    location_id: LocationId,
    state: _State,
    allow_x_as_template_variable: bool,
) -> None:
    node.is_custom_type = True

    type_location_id = location_id
    if "type" in node.raw:
        type_location_id += ["type"]

    if state.context.is_template_variable(type_name) and not state.context.is_type_template_variable(type_name):
        state.record(
            CHSemanticError(
                f'Used the non-type template variable "{type_name}" where a type name was expected!',
                location_id=type_location_id,
            )
        )

    try:
        node.custom_type = state.context.parse_custom_type(type_name, type_location_id, allow_x_as_template_variable)
    except ConceptHierarchyError as e:
        state.record(e)

    provenance = ValueDomainArgumentProvenance(work.get("provenance", ValueDomainArgumentProvenance.ANY.value))
    if "provenance" in work and work["provenance"] not in state.context.argument_provenance_types:
        state.record(
            CHSyntaxError(
                f'"provenance" must be "{ValueDomainArgumentProvenance.ADDR.value}" or '
                f'"{ValueDomainArgumentProvenance.ANY.value}", got {work["provenance"]!r}',
                location_id=location_id + ["provenance"],
            )
        )
        provenance = ValueDomainArgumentProvenance.ANY
    node.provenance = provenance

    if "default" in work:
        node.has_default = True
        node.default_expr = work["default"]
        # NOTE: the default expression is *not* validated here. A later pass, once all Concept Hierarchy instantiation
        # formulae are known, must check `default_expr` against this type's own schema.

    for key in work:
        if key not in CUSTOM_TYPE_EXTRA_KEYS:
            state.record(
                CHSyntaxError(
                    f"Key {key!r} is not allowed on a Concept Hierarchy type schema "
                    f"(only {sorted(CUSTOM_TYPE_EXTRA_KEYS)} are allowed)",
                    location_id=location_id + [key],
                    part=PathPart.KEY,
                )
            )

    # A custom-type node has no draft-07 structure of its own -- treat it as "anything goes" for both meta-schema
    # checking and (later) shallow per-node value checks
    # (actual check is delegated to ValueInstantiationContext.parse_value_against_custom_value_expression).
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


def _finish_builtin_node(node: CHSchemaNode, work: dict, location_id: LocationId, state: _State) -> CHSchemaNode:
    if "provenance" in work:
        state.record(
            CHSyntaxError(
                '"provenance" is only allowed on custom-type schemas (i.e. when "type" is a single Concept Hierarchy'
                " type)",
                location_id=location_id + ["provenance"],
                part=PathPart.KEY,
            )
        )
        work.pop("provenance", None)
    # "default" is a normal draft-07 annotation keyword here; leave it in extra_keywords untouched.

    def child(value: object, *suffix: PathSegment) -> CHSchemaNode:
        return _build_node(value, location_id + list(suffix), state.new_state(False), False)

    def child_from_object_properties(value: object, *suffix: PathSegment) -> CHSchemaNode:
        return _build_node(value, location_id + list(suffix), state.new_state(True), False)

    def child_with_x(value: object, *suffix: PathSegment) -> CHSchemaNode:
        return _build_node(value, location_id + list(suffix), state.new_state(False), True)

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
            if literal_type == "number" and not state.context.is_number_template_variable(template_variable_candidate):
                template_type_check_success = False
            elif literal_type == "integer" and not state.context.is_integer_template_variable(
                template_variable_candidate
            ):
                template_type_check_success = False
            elif literal_type == "string" and not state.context.is_string_template_variable(
                template_variable_candidate
            ):
                template_type_check_success = False
            elif literal_type == "boolean" and not state.context.is_boolean_template_variable(
                template_variable_candidate
            ):
                template_type_check_success = False
            if not template_type_check_success:
                state.record(
                    CHSemanticError(
                        f"Can not use a non {literal_type} template variable literal {template_variable_candidate} as a"
                        f' value for "{key}"!',
                        location_id=location_id + [key],
                    )
                )
            else:
                object.__setattr__(node, node_field_key, template_variable_candidate)

    # --- object structure -------------------------------------------------
    if "properties" in work:
        props = work.pop("properties")
        if isinstance(props, dict):
            for key, sub in props.items():
                node.properties[key] = child_from_object_properties(sub, "properties", key)
        elif isinstance(props, list):
            # parse properties+/functions+ formula
            def check_structure(_x):
                if not isinstance(_x, list) or not (2 <= len(_x) <= 3) or not isinstance(_x[0], str):
                    return False
                if len(_x) == 2:
                    return True
                return isinstance(_x[2], bool)

            if not (check_structure(props)) and not all(check_structure(x) for x in props):
                state.record(
                    CHSyntaxError(
                        f'"properties" must be an object, a 2- or 3-elem array specifying concept-related data, or an '
                        f"array of 2- or 3-elem arrays that is interpreted as a union of concept-related data!\nGot "
                        f"{props!r}",
                        location_id=location_id + ["properties"],
                    )
                )
            elif check_structure(props):
                require_all_keys = False if len(props) == 2 else props[2]
                try:
                    node.custom_concept_data_constraints.append(
                        _parse_custom_concept_data_constraint(
                            props[0],
                            child_with_x(props[1], "properties", 1),
                            require_all_keys,
                            state.context,
                            location_id,
                        )
                    )
                except ConceptHierarchyError as e:
                    state.record(e)
            else:
                for custom_entry_index, custom_constraint_entry in enumerate(props):
                    try:
                        require_all_keys = False if len(custom_constraint_entry) == 2 else custom_constraint_entry[2]
                        new_custom_concept_data_constraint = _parse_custom_concept_data_constraint(
                            custom_constraint_entry[0],
                            child_with_x(custom_constraint_entry[1], "properties", custom_entry_index, 1),
                            require_all_keys,
                            state.context,
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
                        state.record(e)
        elif isinstance(props, str) and props == "args":
            if not state.called_from_object_schema:
                state.record(
                    CHSemanticError(
                        '"properties": "args" may only be set from within a JSON object schema; which is not the case '
                        "at this location!",
                        location_id=location_id + ["properties"],
                        part=PathPart.VALUE,
                    )
                )
            else:
                node.custom_object_properties = props
        else:
            state.record(
                CHSyntaxError(
                    f'"properties" must be the "args" string, an object, a 2-elem array specifying concept-related data'
                    f", or an array of 2-elem arrays that is interpreted as a union of concept-related data!\n"
                    f"Got {props!r}",
                    location_id=location_id + ["properties"],
                )
            )

    if "patternProperties" in work:
        pprops = work.pop("patternProperties")
        if isinstance(pprops, dict):
            for key, sub in pprops.items():
                node.pattern_properties[key] = child_from_object_properties(sub, "patternProperties", key)
        else:
            state.record(CHSyntaxError('"patternProperties" must be an object', location_id + ["patternProperties"]))

    if "additionalProperties" in work:
        ap = work.pop("additionalProperties")
        if isinstance(ap, bool):
            node.additional_properties = ap
        else:
            node.additional_properties = child_from_object_properties(ap, "additionalProperties")

    if "propertyNames" in work:
        node.property_names = child(work.pop("propertyNames"), "propertyNames")

    if "required" in work:
        req = work.pop("required")
        if isinstance(req, list) and all(isinstance(x, str) for x in req):
            node.required = req
        else:
            state.record(
                CHSyntaxError('"required" must be an array of strings', location_id=location_id + ["required"])
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
            state.record(CHSyntaxError('"dependencies" must be an object', location_id + ["dependencies"]))

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

    # --- string structure -----------------------------------------------
    if "constraint" in work:
        node.custom_string_constraint = work.pop("constraint")

    if "format" in work and isinstance(work["format"], str) and work["format"] in {"Concept", "Type"}:
        node.custom_string_format = work.pop("format")

    # --- composition -------------------------------------------------
    for keyword, attr in (("allOf", "all_of"), ("anyOf", "any_of"), ("oneOf", "one_of")):
        if keyword in work:
            arr = work.pop(keyword)
            if isinstance(arr, list):
                setattr(node, attr, [child(sub, keyword, i) for i, sub in enumerate(arr)])
            else:
                state.record(CHSyntaxError(f'"{keyword}" must be an array of schemas', location_id + [keyword]))

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
                state.record(CHSyntaxError(f'"{keyword}" must be an object', location_id + [keyword]))

    if "$ref" in work:
        ref_val = work.pop("$ref")
        if isinstance(ref_val, str):
            node.ref_string = ref_val
        else:
            state.record(CHSyntaxError('"$ref" must be a string', location_id + ["$ref"]))

    for key in work:
        if key not in BUILTIN_NODE_EXTRA_KEYS:
            state.record(
                CHSyntaxError(
                    f"Key {key!r} is not allowed on a Draft07 schema "
                    f"(only {sorted(BUILTIN_NODE_EXTRA_KEYS)} are allowed)",
                    location_id=location_id + [key],
                    part=PathPart.KEY,
                )
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
# $ref resolution: a JSON Pointer from the schema root
# ---------------------------------------------------------------------------
def unescape_pointer_segment(segment: str) -> str:
    """RFC 6901: ``~1`` is a literal ``/`` and ``~0`` a literal ``~``, in that order."""
    return segment.replace("~1", "/").replace("~0", "~")


def split_json_pointer(pointer: str) -> list[str]:
    """The segments of ``pointer``, which must start at the document root (``#`` or ``#/...``)."""
    assert pointer.startswith("#"), pointer
    body = pointer[1:]
    if not body:
        return []
    assert body.startswith("/"), pointer
    return [unescape_pointer_segment(segment) for segment in body[1:].split("/")]


def resolve_schema_pointer(root: CHSchemaNode, segments: Sequence[str]) -> tuple[CHSchemaNode | None, str | None]:
    """
    Navigate a JSON Pointer through a parsed schema, one keyword at a time.

    A pointer rather than a name, because a name is not an address: the previous resolver flattened every
    ``$defs`` in the tree into one map and took the first entry for a name, so a nested definition was
    reachable *by accident*, two definitions sharing a name silently resolved to whichever came first, and
    ``#/$defs/a/$defs/b`` could not be written at all.

    Navigation is written out per keyword rather than driven by :meth:`CHSchemaNode.iter_children`, because
    the two answer different questions: ``iter_children`` enumerates children for a traversal and folds
    ``$defs`` and the custom concept-data constraints into shapes that suit that, while a pointer has to
    address exactly what the *source text* wrote.

    :return: ``(node, None)`` on success, or ``(None, reason)`` naming the segment that could not be taken.
    """
    node = root
    index = 0
    while index < len(segments):
        keyword = segments[index]
        index += 1
        if node.is_boolean_schema:
            return None, f"{keyword!r} can not be taken: the schema at that point is a boolean schema"

        def named(container: dict, what: str) -> tuple[CHSchemaNode | None, str | None]:
            if index >= len(segments):
                return None, f"{keyword!r} needs a name after it"
            name = segments[index]
            if name not in container:
                return None, f"there is no {what} named {name!r} (have: {sorted(container)})"
            return container[name], None

        if keyword in _DEFINITION_KEYWORDS:
            child, failure = named(node.definitions, "definition")
            index += 1
        elif keyword == "properties":
            child, failure = named(node.properties, "property")
            index += 1
        elif keyword == "patternProperties":
            child, failure = named(node.pattern_properties, "pattern property")
            index += 1
        elif keyword == "dependencies":
            child, failure = named(node.dependent_schemas, "dependent schema")
            index += 1
        elif keyword in ("allOf", "anyOf", "oneOf"):
            branches = {"allOf": node.all_of, "anyOf": node.any_of, "oneOf": node.one_of}[keyword]
            child, failure = _indexed(branches, segments, index, keyword)
            index += 1
        elif keyword == "items" and isinstance(node.items, list):
            child, failure = _indexed(node.items, segments, index, keyword)
            index += 1
        elif keyword == "items":
            child, failure = node.items, (None if node.items is not None else 'there is no "items" here')
        elif keyword == "additionalProperties":
            child = node.additional_properties if isinstance(node.additional_properties, CHSchemaNode) else None
            failure = None if child is not None else 'there is no schema at "additionalProperties" here'
        elif keyword == "additionalItems":
            child = node.additional_items if isinstance(node.additional_items, CHSchemaNode) else None
            failure = None if child is not None else 'there is no schema at "additionalItems" here'
        elif keyword in ("propertyNames", "contains", "not", "if", "then", "else"):
            attribute = {"not": "not_", "if": "if_", "then": "then_", "else": "else_"}.get(keyword, keyword)
            child = getattr(node, attribute if attribute != "propertyNames" else "property_names")
            failure = None if child is not None else f"there is no schema at {keyword!r} here"
        else:
            return None, f"{keyword!r} is not a schema keyword that can be pointed into"

        if failure is not None:
            return None, failure
        assert child is not None
        node = child
    return node, None


def _indexed(
    branches: list, segments: Sequence[str], index: int, keyword: str
) -> tuple[CHSchemaNode | None, str | None]:
    """One step into a keyword whose value is an array of schemas."""
    if index >= len(segments):
        return None, f"{keyword!r} needs an index after it"
    written = segments[index]
    if not written.isdigit():
        return None, f"{keyword!r} needs a numeric index, not {written!r}"
    position = int(written)
    if position >= len(branches):
        return None, f"{keyword!r} has only {len(branches)} branch(es), so index {position} does not exist"
    return branches[position], None


CROSS_SCHEMA_REFERENCE_PREFIX = "#ch#/"
"""What marks a ``$ref`` as naming *another* ValueDomain's schema -- see :class:`CrossSchemaReference`."""

POINTER_START_MISSING = (
    'a cross-schema reference needs a "#" where the pointer into the target schema starts, as in '
    '"#ch#/<Type>/#/$defs/<name>"'
)


def parse_cross_schema_reference(written: str) -> tuple[CrossSchemaReference | None, str | None]:
    """
    Parse ``#ch#/<Type>[/<index>]/#/<pointer>`` left to right.

    **Not by splitting on** ``/``, and not by searching for the ``#`` that starts the pointer. A type may
    contain a *literal template variable*, and a literal may contain any character at all -- so
    ``#ch#/Sequence<"a/b">/#/$defs/x`` has a ``/`` that separates nothing and ``#ch#/Sequence<"a#b">/#`` a
    ``#`` that starts nothing. Either would be cut in the wrong place by a search. Reading the grammar in
    order removes the question: the type ends at the first ``/`` that is neither quoted nor inside a
    template argument list (`StringParser.consume_until_unnested`), and what follows is whatever the
    grammar says can follow, never whatever a search happens to find.

    The pointer *after* the ``#`` is split on ``/`` -- correctly, because a JSON Pointer segment escapes a
    literal ``/`` as ``~1`` (RFC 6901), so a raw ``/`` there is always a separator.

    :return: ``(reference, None)``, or ``(None, reason)`` when the text is not of this shape.
    """
    assert written.startswith(CROSS_SCHEMA_REFERENCE_PREFIX), written
    parser = StringParser(written)
    parser.consume(CROSS_SCHEMA_REFERENCE_PREFIX)

    try:
        type_name = parser.consume_until_unnested("/")
    except CHSyntaxError as e:
        return None, f"the type is not well-formed ({e.args[0].splitlines()[0]})"
    if not type_name:
        return None, "a cross-schema reference must name the type whose schema it points into"
    if not parser.try_consume("/"):
        return None, POINTER_START_MISSING

    schema_index: int | None = None
    if not parser.starts_with("#"):
        # Only an index may stand between the type and the pointer. Splitting on "/" is safe *here*, which
        # it was not before the type was consumed: what remains is an index and a JSON Pointer, and a
        # pointer segment escapes a literal "/" as "~1".
        segment = parser.remaining().split("/")[0]
        if segment.isdigit():
            schema_index = int(segment)
            parser.pos += len(segment)
            if not parser.try_consume("/"):
                return None, 'the instantiation schema index must be followed by "/#" and the pointer'
        elif parser.remaining()[len(segment) :].startswith("/#"):
            # It stands where an index stands, so it was meant as one; saying the "#" is missing would
            # send the author looking at the wrong end of the reference.
            return None, f"the instantiation schema index must be a non-negative integer, not {segment!r}"
        else:
            return None, POINTER_START_MISSING
    if not parser.try_consume("#"):
        if schema_index is not None:
            return None, (
                'only "<Type>" or "<Type>/<index>" may stand before the pointer, so the index must be followed by "#"'
            )
        return None, POINTER_START_MISSING

    pointer: tuple[str, ...] = ()
    if not parser.eof():
        if not parser.try_consume("/"):
            return None, f'expected the pointer to continue with "/" after the "#", got {parser.remaining()!r}'
        pointer = tuple(unescape_pointer_segment(segment) for segment in parser.remaining().split("/"))
    return (
        CrossSchemaReference(
            type_name=type_name,
            schema_index=schema_index,
            pointer=pointer,
            written=written,
        ),
        None,
    )


def _resolve_local_refs(root: CHSchemaNode, state: _State) -> None:
    for n in root.walk():
        if n.ref_string is None:
            continue
        if n.ref_string.startswith(CROSS_SCHEMA_REFERENCE_PREFIX):
            # Parsed now, bound later: the target application's schema does not exist yet, and whether the
            # target is even a ValueDomain is not a question this module can ask.
            reference, failure = parse_cross_schema_reference(n.ref_string)
            if reference is None:
                state.record(
                    CHSyntaxError(
                        f"Cannot resolve $ref {n.ref_string!r}: {failure}",
                        n.location_id + ["$ref"],
                    )
                )
                continue
            n.cross_reference = reference
            continue
        if not n.ref_string.startswith("#/") and n.ref_string != "#":
            state.record(
                CHSyntaxError(
                    f"Cannot resolve $ref {n.ref_string!r}: a reference must be a JSON Pointer from the "
                    f'schema root ("#" or "#/<keyword>/..."), or a cross-schema reference '
                    f'("{CROSS_SCHEMA_REFERENCE_PREFIX}<Type>/#/...").',
                    n.location_id + ["$ref"],
                )
            )
            continue
        resolved, failure = resolve_schema_pointer(root, split_json_pointer(n.ref_string))
        if resolved is None:
            state.record(
                CHSyntaxError(
                    f"Cannot resolve $ref {n.ref_string!r}: {failure}",
                    n.location_id + ["$ref"],
                )
            )
            continue
        n.ref_resolved = resolved


# ---------------------------------------------------------------------------
# draft-07 meta-schema check
# ---------------------------------------------------------------------------
def _check_meta_schema(root: CHSchemaNode, state: _State) -> None:
    """Validate ``root.safe_canonical`` against the draft-07 meta-schema.

    Because shorthand expansion never changes the *path* of a node (it only ever turns the value found at a given path
    into an equivalent object form), and custom-type subtrees collapse to ``True``
    (always valid against the meta-schema), any error path reported here corresponds 1:1 with the path conventions
    used for ``CHSyntaxError`` elsewhere.
    """
    meta_validator = Draft7Validator(Draft7Validator.META_SCHEMA)
    for err in meta_validator.iter_errors(root.safe_canonical):
        state.record(CHSyntaxError(err.message, root.location_id + list(err.path)))
