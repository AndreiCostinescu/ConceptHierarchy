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
The AST produced by :func:`ch_schema.schema_validator.parse_schema`.

Every node corresponds to one schema location (after shorthand expansion).
A node either:

* is a **boolean schema** (``True``/``False``, as in draft-07) -- a leaf, with no further structure;
* is a **custom-type node** (``is_custom_type=True``) -- a leaf as far as JSON-Schema structure is concerned
  (it carries no ``properties``/``items`` etc. of its own), but carries ``custom_type_name``, ``referenceType`` and an
  optional, *unvalidated* ``default_expr``;
* is a **builtin/composite node** -- carries the (recursively parsed) draft-07 structure:
  ``properties``, ``items``, ``allOf``/``anyOf``/..., etc., plus any other draft-07 keywords
  (``enum``, ``minimum``, ``pattern``, ...) in :attr:`extra_keywords`.

:meth:`CHSchemaNode.iter_children` / :meth:`CHSchemaNode.walk` give a uniform way to traverse the tree
(e.g. for code generation), independent of which particular keywords were used in the original schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

from concept_hierarchy.errors import LocationId, PathSegment


@dataclass
class CHSchemaNode:
    # --- provenance -------------------------------------------------
    location_id: LocationId
    """Location of this node in the *original* (shorthand) schema."""

    raw: object
    """The original, un-expanded value at :attr:`path` (a bool, string, list, or dict)."""

    canonical: dict | bool
    """
    ``raw`` after shorthand expansion: 
    either a bool (draft-07 boolean schema) or a dict with an explicit ``"type"`` (if any).
    """

    # --- custom types -------------------------------------------------
    is_custom_type: bool = False
    custom_type_name: str | None = None
    ref: str | None = None  # "Reference" | "NoRef", only set if is_custom_type
    has_default: bool = False
    default_expr: object = None
    """
    Raw, *unvalidated* default-value expression (only meaningful if :attr:`has_default` is True). 
    Validating this expression against the custom type's own schema is a separate, later pass.
    """

    # --- "type" keyword (builtin / multi-type only) -------------------
    type_value: str | list[str] | None = None

    # --- object structure ---------------------------------------------
    properties: dict[str, CHSchemaNode] = field(default_factory=dict)
    pattern_properties: dict[str, CHSchemaNode] = field(default_factory=dict)
    additional_properties: CHSchemaNode | bool | None = None
    property_names: CHSchemaNode | None = None
    required: list[str] = field(default_factory=list)
    require_all_properties: bool = False
    dependent_schemas: dict[str, CHSchemaNode] = field(default_factory=dict)

    # --- array structure ----------------------------------------------
    items: CHSchemaNode | list[CHSchemaNode] | None = None
    additional_items: CHSchemaNode | bool | None = None
    contains: CHSchemaNode | None = None

    # --- composition --------------------------------------------------
    all_of: list[CHSchemaNode] = field(default_factory=list)
    any_of: list[CHSchemaNode] = field(default_factory=list)
    one_of: list[CHSchemaNode] = field(default_factory=list)
    not_: CHSchemaNode | None = None
    if_: CHSchemaNode | None = None
    then_: CHSchemaNode | None = None
    else_: CHSchemaNode | None = None

    # --- $defs / $ref -------------------------------------------------
    definitions: dict[str, CHSchemaNode] = field(default_factory=dict)
    ref_string: str | None = None
    ref_resolved: CHSchemaNode | None = None

    # --- everything else (enum, const, minimum, pattern, format, ...) -
    extra_keywords: dict = field(default_factory=dict)

    # --- derived, jsonschema-Draft7-compatible forms ------------------
    safe_canonical: dict | bool = True
    """
    Full recursive draft-07 schema equivalent to this node, with every *custom-type* subtree replaced by ``True``. 
    Used for the structural (meta-schema) check of the schema definition.
    """

    shallow_canonical: dict | bool = True
    """
    This node's *own* keywords only (``enum``, ``type`` if builtin, ``minimum``, ...), 
    with **every** subschema-bearing keyword replaced by ``True``. 
    Used to check this node's own keyword constraints against a value, while recursion into children is performed 
    separately by :mod:`ch_schema.value_validator`.
    """

    # ------------------------------------------------------------------
    @property
    def is_boolean_schema(self) -> bool:
        return isinstance(self.canonical, bool)

    def iter_children(self) -> Iterator[tuple[tuple[PathSegment, ...], CHSchemaNode]]:
        """
        Yield ``(relative_path, child_node)`` for every direct child schema node of this node,
        regardless of which keyword it came from.
        ``relative_path`` is a tuple of path segments to append to this node's :attr:`path` to get the child's path.
        """
        if self.is_boolean_schema:
            return

        for key, child in self.properties.items():
            yield ("properties", key), child
        for key, child in self.pattern_properties.items():
            yield ("patternProperties", key), child
        if isinstance(self.additional_properties, CHSchemaNode):
            yield ("additionalProperties",), self.additional_properties
        if self.property_names is not None:
            yield ("propertyNames",), self.property_names

        if isinstance(self.items, list):
            for i, child in enumerate(self.items):
                yield ("items", i), child
        elif isinstance(self.items, CHSchemaNode):
            yield ("items",), self.items
        if isinstance(self.additional_items, CHSchemaNode):
            yield ("additionalItems",), self.additional_items
        if self.contains is not None:
            yield ("contains",), self.contains

        for keyword, branches in (
            ("allOf", self.all_of),
            ("anyOf", self.any_of),
            ("oneOf", self.one_of),
        ):
            for i, child in enumerate(branches):
                yield (keyword, i), child
        if self.not_ is not None:
            yield ("not",), self.not_
        for keyword, child in (("if", self.if_), ("then", self.then_), ("else", self.else_)):
            if child is not None:
                yield (keyword,), child

        for key, child in self.definitions.items():
            yield ("definitions", key), child
        for key, child in self.dependent_schemas.items():
            yield ("dependencies", key), child

    def walk(self) -> Iterator[CHSchemaNode]:
        """
        Depth-first iterator over this node and all of its descendants (custom-type and boolean-schema leaves included).
        """
        yield self
        for _, child in self.iter_children():
            yield from child.walk()

    def apply(self, f: Callable[[CHSchemaNode], CHSchemaNode]) -> CHSchemaNode:
        if self.is_boolean_schema:
            return self

        res = self

        for key, child in self.properties.items():
            res.properties[key] = f(child)
        for key, child in self.pattern_properties.items():
            res.pattern_properties[key] = f(child)
        if isinstance(self.additional_properties, CHSchemaNode):
            res.additional_properties = f(self.additional_properties)
        if self.property_names is not None:
            res.property_names = f(self.property_names)

        if isinstance(self.items, list):
            for i, child in enumerate(self.items):
                res.items[i] = f(child)
        elif isinstance(self.items, CHSchemaNode):
            res.items = f(self.items)
        if isinstance(self.additional_items, CHSchemaNode):
            res.additional_items = f(self.additional_items)
        if self.contains is not None:
            res.contains = f(self.contains)

        for res_list, branches in ((res.all_of, self.all_of), (res.any_of, self.any_of), (res.one_of, self.one_of)):
            for i, child in enumerate(branches):
                res_list[i] = f(child)
        if self.not_ is not None:
            res.not_ = f(self.not_)
        if self.if_ is not None:
            res.if_ = f(self.if_)
        if self.then_ is not None:
            res.then_ = f(self.then_)
        if self.else_ is not None:
            res.else_ = f(self.else_)

        for key, child in self.definitions.items():
            res.definitions[key] = f(child)
        for key, child in self.dependent_schemas.items():
            res.dependent_schemas[key] = f(child)

        return res
