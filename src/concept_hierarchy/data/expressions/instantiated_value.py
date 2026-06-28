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
Result tree produced by :func:`value_parser.parse_value`.

Every node mirrors one schema position and carries:

* the errors that originated *directly at that node* (not in its children),
* the successfully parsed subtrees for every structural keyword
  (``properties``, ``items``, ``allOf``, …), and
* for custom-type leaves, the :class:`Expression` parsed by the context.

:meth:`ParsedValue.walk` / :meth:`ParsedValue.is_valid` traverse the tree
bottom-up so callers can find all errors without re-implementing the
structural walk.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field

from concept_hierarchy.data.expressions.expression import Expression
from concept_hierarchy.data.jsonschema.parsed_schema import CHSchemaNode
from concept_hierarchy.data.types.concept_hierarchy_types import TypeValue
from concept_hierarchy.errors import ConceptHierarchyError, LocationId, PathSegment


class ParsedValue(ABC):
    """Abstract base for all nodes in the parse-result tree.

    Concrete subclasses:

    * :class:`ParsedCustomValue` — a custom-type leaf validated and parsed
      by :meth:`~value_instantiation_validator.CHValueValidator.parse_custom_type`.
    * :class:`ParsedStructural` — a boolean schema, a builtin-type schema,
      or any composite draft-07 schema (``properties``, ``allOf``, …).

    Attributes exposed by all subclasses (as plain dataclass fields):
        location_id: The value path for this node (matches the schema's
            ``location_id`` at the same position).
        schema_node: The :class:`~parsed_schema.CHSchemaNode` that was used
            to parse this node.
        errors: Errors that originated *directly* at this node.  Errors
            from child nodes live on those children.
    """

    location_id: LocationId
    schema_node: CHSchemaNode
    errors: list[ConceptHierarchyError]

    @abstractmethod
    def iter_children(self) -> Iterator[tuple[tuple[PathSegment, ...], ParsedValue]]:
        """Yield ``(relative_path_tuple, child)`` for every direct child node.

        The relative path is a tuple of path segments that, when appended to
        :attr:`location_id`, gives the child's :attr:`location_id`.
        """

    @abstractmethod
    def iter_expressions(self) -> Iterator[tuple[tuple[PathSegment, ...], Expression]]:
        """Yield ``(relative_path_tuple, expression)`` for every expression in the value.

        The relative path is a tuple of path segments that, when appended to
        :attr:`location_id`, gives the expression's :attr:`location_id`.
        """

    def walk(self) -> Iterator[ParsedValue]:
        """Depth-first walk over ``self`` and all descendants."""
        yield self
        for _, child in self.iter_children():
            yield from child.walk()

    def is_valid(self) -> bool:
        """``True`` iff this node has no direct errors *and* all children are valid."""
        if self.errors:
            return False
        return all(child.is_valid() for _, child in self.iter_children())


@dataclass
class ParsedCustomValue(ParsedValue):
    """A custom-type leaf node parsed by
    :meth:`~value_instantiation_validator.CHValueValidator.parse_custom_type`.

    Attributes:
        custom_type: The resolved :class:`~concept_hierarchy_types.TypeValue`
            from the schema node.
        ref: ``"Reference"`` or ``"NoRef"``.
        used_default: ``True`` when the value was absent (``MISSING``) and
            the schema's ``default_expr`` was used instead.
        default_expr: The raw default-value expression from the schema, or
            ``MISSING`` if no default was defined.
        expression: The parsed :class:`~expression.Expression`, or ``None``
            when parsing/validation failed.
    """

    location_id: LocationId
    schema_node: CHSchemaNode
    errors: list[ConceptHierarchyError]
    custom_type: TypeValue
    ref: str
    used_default: bool
    default_expr: object  # MISSING sentinel or raw expression
    expression: Expression | None

    def iter_children(self) -> Iterator[tuple[tuple[PathSegment, ...], ParsedValue]]:
        """No structural children — expression traversal is the caller's concern."""
        yield from []

    def iter_expressions(self) -> Iterator[tuple[tuple[PathSegment, ...], Expression]]:
        if self.expression:
            yield (), self.expression


@dataclass
class ParsedStructural(ParsedValue):
    """A boolean, builtin-type, or composite schema node.

    Fields are populated only for the keywords present in the schema.  An
    absent field (empty dict / empty list / ``None``) means that keyword
    either wasn't present or produced no parse-able results (e.g. an absent
    optional property without a default).

    Attributes:
        value: The raw Python (JSON-decoded) value at this position.

        properties_parsed: Keyed by property name.  Only properties that
            were present in the value *or* filled in from a default appear
            here; absent optional properties are omitted.
        pattern_properties_parsed: ``{value_key: [(pattern, ParsedValue)]}``
            — one entry per pattern that matched the key.
        additional_properties_parsed: Keys that fell through both
            ``properties`` and ``patternProperties``, parsed against
            ``additionalProperties`` schema.

        items_parsed: Index-correlated with the value array (``len ==
            len(value)``).  ``None`` at position ``i`` means no schema
            applied to ``value[i]`` (items beyond a tuple schema with no
            ``additionalItems``).
        contains_parsed: The first array item that matched the ``contains``
            schema, or ``None`` if none did (in which case an error is
            present on this node).

        all_of_parsed: One entry per ``allOf`` branch.  Empty when *any*
            branch failed (all-or-nothing: errors from the failed branch(es)
            appear on this node instead).
        any_of_parsed: One entry per *matching* ``anyOf`` branch (all
            matching branches are kept).
        one_of_parsed: The single matching ``oneOf`` branch, or ``None`` if
            the constraint failed.

        if_parsed: Result of evaluating the ``if`` schema — metadata only,
            **not** included in :meth:`iter_children` / :meth:`is_valid`.
        then_else_parsed: Result of parsing the branch that was taken
            (``then`` when ``if`` matched, ``else`` otherwise).

        dependent_schemas_parsed: Keyed by the triggering property name.
    """

    location_id: LocationId
    schema_node: CHSchemaNode
    errors: list[ConceptHierarchyError]
    value: object

    # object structure
    properties_parsed: dict[str, ParsedValue] = field(default_factory=dict)
    pattern_properties_parsed: dict[str, list[tuple[str, ParsedValue]]] = field(default_factory=dict)
    additional_properties_parsed: dict[str, ParsedValue] = field(default_factory=dict)

    # array structure
    items_parsed: list[ParsedValue | None] = field(default_factory=list)
    contains_parsed: ParsedValue | None = None

    # composition
    all_of_parsed: list[ParsedValue] = field(default_factory=list)
    any_of_parsed: list[ParsedValue] = field(default_factory=list)
    one_of_parsed: ParsedValue | None = None

    # conditional — if_parsed is metadata only; then_else_parsed is structural
    if_parsed: ParsedValue | None = None
    then_else_parsed: ParsedValue | None = None

    # dependencies
    dependent_schemas_parsed: dict[str, ParsedValue] = field(default_factory=dict)

    def iter_children(self) -> Iterator[tuple[tuple[PathSegment, ...], ParsedValue]]:
        for key, child in self.properties_parsed.items():
            yield ("properties", key), child
        for key, matches in self.pattern_properties_parsed.items():
            for pattern, child in matches:
                yield ("patternProperties", key, pattern), child
        for key, child in self.additional_properties_parsed.items():
            yield ("additionalProperties", key), child
        for i, child in enumerate(self.items_parsed):
            if child is not None:
                yield ("items", i), child
        if self.contains_parsed is not None:
            yield ("contains",), self.contains_parsed
        for i, child in enumerate(self.all_of_parsed):
            yield ("allOf", i), child
        for i, child in enumerate(self.any_of_parsed):
            yield ("anyOf", i), child
        if self.one_of_parsed is not None:
            yield ("oneOf",), self.one_of_parsed
        # if_parsed intentionally omitted — it is metadata, not part of the validity tree
        if self.then_else_parsed is not None:
            yield ("then_else",), self.then_else_parsed
        for key, child in self.dependent_schemas_parsed.items():
            yield ("dependencies", key), child

    def iter_expressions(self) -> Iterator[tuple[tuple[PathSegment, ...], Expression]]:
        for key, child in self.properties_parsed.items():
            for expr_location, expr in child.iter_expressions():
                yield ("properties", key) + expr_location, expr
        for key, matches in self.pattern_properties_parsed.items():
            for pattern, child in matches:
                for expr_location, expr in child.iter_expressions():
                    yield ("patternProperties", key, pattern) + expr_location, expr
        for key, child in self.additional_properties_parsed.items():
            for expr_location, expr in child.iter_expressions():
                yield ("additionalProperties", key) + expr_location, expr
        for i, child in enumerate(self.items_parsed):
            if child is not None:
                for expr_location, expr in child.iter_expressions():
                    yield ("items", i) + expr_location, expr
        if self.contains_parsed is not None:
            for expr_location, expr in self.contains_parsed.iter_expressions():
                yield ("contains",) + expr_location, expr
        for i, child in enumerate(self.all_of_parsed):
            for expr_location, expr in child.iter_expressions():
                yield ("allOf", i) + expr_location, expr
        for i, child in enumerate(self.any_of_parsed):
            for expr_location, expr in child.iter_expressions():
                yield ("anyOf", i) + expr_location, expr
        if self.one_of_parsed is not None:
            for expr_location, expr in self.one_of_parsed.iter_expressions():
                yield ("onfOf",) + expr_location, expr
        if self.then_else_parsed is not None:
            for expr_location, expr in self.then_else_parsed.iter_expressions():
                yield ("then_else",) + expr_location, expr
        for key, child in self.dependent_schemas_parsed.items():
            for expr_location, expr in child.iter_expressions():
                yield ("dependencies", key) + expr_location, expr
