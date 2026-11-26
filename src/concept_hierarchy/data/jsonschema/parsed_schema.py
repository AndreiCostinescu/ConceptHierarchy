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
  (it carries no ``properties``/``items`` etc. of its own), but carries ``custom_type``, ``provenance`` and an
  optional, *unvalidated* ``default_expr``;
* is a **builtin/composite node** -- carries the (recursively parsed) draft-07 structure:
  ``properties``, ``items``, ``allOf``/``anyOf``/..., etc., plus any other draft-07 keywords
  (``enum``, ``minimum``, ``pattern``, ...) in :attr:`extra_keywords`.

:meth:`CHSchemaNode.iter_children` / :meth:`CHSchemaNode.walk` give a uniform way to traverse the tree
(e.g. for code generation), independent of which particular keywords were used in the original schema.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import Callable, Iterator

from concept_hierarchy.data.expressions.expression import Expression
from concept_hierarchy.data.expressions.expression_utils import ExpressionProvenance
from concept_hierarchy.data.types.concept_hierarchy_types import (
    ExpandedVariadicTemplateVariable,
    TemplateDependent,
    TypeValue,
)
from concept_hierarchy.definitions.concept_definition_domain_concept import ForPropertyOrFunction
from concept_hierarchy.errors import LocationId, PathSegment


@dataclass(frozen=True)
class CrossSchemaReference:
    """
    A ``$ref`` into another ValueDomain's instantiation schema: ``#ch#/<Type>[/<index>]/#/<pointer>``.

    The target is a **type**, not a concept name -- ``#ch#/Box<Integer>/#/$defs/x``. That is what makes the
    referenced fragment meaningful: it carries the template arguments the target schema is substituted
    with, so a ``$defs`` entry of type ``Box:T`` arrives as ``Integer`` rather than as a variable nothing in
    the referring schema could bind.

    ``schema_index`` selects one of the target's ``instantiation`` entries. It is absent exactly when the
    target did not write a template-dependent instantiation -- there is one schema then, and naming its
    index would be a fiction -- and required, and in range, when the target did.
    """

    type_name: str
    schema_index: int | None
    pointer: tuple[str, ...]
    written: str
    """The reference exactly as written, for diagnostics."""


class CustomConceptDataConstraint:
    def __init__(
        self,
        for_properties_or_functions: ForPropertyOrFunction,
        include_parent_data: bool,
        concept_restriction: list[TypeValue] | None,
        value: CHSchemaNode,
        require_all_keys: bool,
    ):
        self.for_properties_or_functions = for_properties_or_functions
        self.include_parent_data = include_parent_data
        self.concept_restriction = concept_restriction
        self.value = value
        self.require_all_keys = require_all_keys

    def __repr__(self):
        kind = "funcs" if self.for_properties_or_functions is ForPropertyOrFunction.FUNCTION else "props"
        constraint = (
            "(" + ", ".join(x.full_name for x in self.concept_restriction) + ")"
            if self.concept_restriction is not None
            else ""
        )
        require_str = "true" if self.require_all_keys else "false"
        return f"[{kind}{'+' if self.include_parent_data else ''}{constraint}, {self.value}, {require_str}]"

    @property
    def is_template_dependent(self):
        return self.value.is_template_dependent or any(
            isinstance(x, TemplateDependent) for x in self.concept_restriction or ()
        )


LITERAL_KEYWORD_FIELDS: dict[str, tuple[str, str]] = {
    "min_properties_def": ("minProperties", "object"),
    "max_properties_def": ("maxProperties", "object"),
    "min_length_def": ("minLength", "string"),
    "max_length_def": ("maxLength", "string"),
    "min_items_def": ("minItems", "array"),
    "max_items_def": ("maxItems", "array"),
    "minimum_def": ("minimum", "number"),
    "maximum_def": ("maximum", "number"),
    "exclusive_minimum_def": ("exclusiveMinimum", "number"),
    "exclusive_maximum_def": ("exclusiveMaximum", "number"),
    "multiple_of_def": ("multipleOf", "number"),
}
"""
Every keyword that may be written as a literal template variable: the `CHSchemaNode` field it is parked in,
the JSON Schema keyword it came from, and the JSON type it constrains.

The type matters because a keyword only applies to values of its own type -- an undecided ``minItems`` says
nothing about a string -- so it is what separates "this value could not be checked yet" from "this value was
checked completely, and a keyword that does not apply to it happens to be undecided".
"""


def json_type_family_of(value: object) -> str | None:
    """The JSON Schema type family ``value`` belongs to, or ``None`` for null and booleans."""
    if isinstance(value, bool):
        # Before the int check: `bool` is a subclass of `int`, and no numeric keyword applies to a boolean.
        return None
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    return None


EXPANDABLE_KEYWORDS = frozenset({"allOf", "anyOf", "oneOf", "items"})
"""
The keywords whose *list* form is an expandable container -- one a ``T...`` may be replicated into.

``items`` is in the set but only counts when it holds a list: ``"items": <schema>`` constrains every
element with one schema and has no positions to splice into. The fifth container of
`documentation/TODO_VARIADIC_SCHEMA_EXPANSION.md` 2.1, a ``props``/``funcs`` argument list, is not a
keyword and so is not here; `CHSchemaNode.variadic_selectors_of_this_unit` is what reaches it.
"""


NON_VALUE_CONSUMING_KEYWORDS = frozenset({"allOf", "anyOf", "oneOf", "not", "if", "then", "else", "dependencies"})
"""
The keywords whose subschemas are applied to the **same** value the parent was applied to.

`value_instantiation_parser` descends into these without moving through the value, so a cycle among them
never terminates. Every other subschema-bearing keyword -- ``properties``, ``patternProperties``,
``additionalProperties``, ``propertyNames``, ``items``, ``additionalItems``, ``contains`` -- consumes a
step of value structure, which is finite, so recursion through those is ordinary and legitimate.

``definitions``/``$defs`` is absent on purpose, and is not merely "consuming": the parser never descends
into it at all. A definition is reached only by a ``$ref``, and that edge is followed explicitly below.
"""


def non_value_consuming_children(node: CHSchemaNode) -> Iterator[CHSchemaNode]:
    """
    The schemas ``node`` hands the same value on to, which is the edge relation of :func:`find_reference_cycle`.

    Mirrors `value_instantiation_parser._parse` exactly, including that a ``$ref`` node has **only** the
    one edge: that function returns the reference's result outright and never looks at the node's other
    keywords, so modelling them here would invent edges the parser cannot take.
    """
    if node.ref_resolved is not None:
        yield node.ref_resolved
        return
    for path, child in node.iter_children():
        if path[0] in NON_VALUE_CONSUMING_KEYWORDS:
            yield child


def find_reference_cycle(root: CHSchemaNode) -> list[CHSchemaNode] | None:
    """
    A cycle among the schemas that are applied to one value without any of it being consumed, or ``None``.

    Such a cycle describes no value and cannot be parsed against: the parser re-enters the same node with
    the same value forever. Left undetected it surfaces as "Ran out of stack", which arrives far from the
    schema at fault and says nothing about it.

    Every node is an entry point, not only the root, so a cycle that sits inside an unreferenced ``$defs``
    entry is still reported -- it is ill-formed whether or not anything happens to point at it yet.

    :return: the nodes of the cycle, the first repeated one appearing at both ends, or ``None``.
    """
    IN_PROGRESS, DONE = 0, 1
    status: dict[int, int] = {}
    path: list[CHSchemaNode] = []

    def visit(node: CHSchemaNode) -> list[CHSchemaNode] | None:
        state = status.get(id(node))
        if state == DONE:
            return None
        if state == IN_PROGRESS:
            start = next(index for index, on_path in enumerate(path) if on_path is node)
            return path[start:] + [node]
        status[id(node)] = IN_PROGRESS
        path.append(node)
        for child in non_value_consuming_children(node):
            found = visit(child)
            if found is not None:
                return found
        path.pop()
        status[id(node)] = DONE
        return None

    for node in root.walk():
        found = visit(node)
        if found is not None:
            return found
    return None


@dataclass
class CHSchemaNode:
    # --- provenance -------------------------------------------------
    schema_owner: str
    """The name of the ValueDomain concept that defined this instantiation schema."""

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
    custom_type: TypeValue | None = None
    expands_variadic: bool = False
    """
    This custom-type node is a ``T...`` placeholder rather than a type of its own.

    Stage B replicates the *element* of the enclosing expandable container that holds it -- once per argument of the 
    group -- so the node is not replaced in place and nothing here records the unit: 
    it is found by walking down from the container, which is the walk that has to copy it anyway. 
    See `documentation/TODO_VARIADIC_SCHEMA_EXPANSION.md` 2.7.
    """
    provenance: ExpressionProvenance | None = None  # "Addr" | "Any", only set if is_custom_type
    has_default: bool = False
    """
    Differentiates `default_expr` being `None` because there is no value, 
    versus an actual JSON `null` value supplied as default expression.
    """
    default_expr: object = None
    """
    Raw, *unvalidated* default-value expression (only meaningful if :attr:`has_default` is True). 
    Validating this expression against the custom type's own schema is a separate, later pass.
    """
    parsed_default_expr: Expression | None = None
    """ Validated expression representing the value's default expression. `None` only when has_default is False. """

    # --- custom keywords -------------------
    custom_string_format: str | None = None
    custom_string_constraint: str | None = None
    custom_object_properties: str | None = None

    # --- "type" keyword (builtin / multi-type only) -------------------
    type_value: str | list[str] | None = None

    # --- object structure ---------------------------------------------
    properties: dict[str, CHSchemaNode] = field(default_factory=dict)
    custom_concept_data_constraints: list[CustomConceptDataConstraint] = field(default_factory=list)
    pattern_properties: dict[str, CHSchemaNode] = field(default_factory=dict)
    additional_properties: CHSchemaNode | bool | None = None
    property_names: CHSchemaNode | None = None
    required: list[str] = field(default_factory=list)
    dependent_schemas: dict[str, CHSchemaNode] = field(default_factory=dict)

    # --- array structure ----------------------------------------------
    items: CHSchemaNode | list[CHSchemaNode] | None = None
    additional_items: CHSchemaNode | bool | None = None
    require_all_items: bool = False
    """
    Whether every position of the ``items`` tuple must be **present** in the value.

    draft-07 constrains position *i* only when element *i* is there, so ``items: [A, B]`` accepts ``[a]``.
    This is the missing lower bound, and it is complementary to `additional_items` rather than overlapping it:
    one says "no more than these" (additional_items), the other "no fewer" (require_all_items).
    Together they mean an exact length, which is what a tuple whose positions come from a variadic expansion needs.
    There, the length is the arity of the group, and the author cannot write it as a ``minItems`` at the declaration.
    """
    contains: CHSchemaNode | None = None

    # --- values that could be literal template variables --------------
    # Once a complete substitution is done, these values will be set to None
    #  and the substituted values will be set in the extra_keywords and
    #  shallow_canonical to be validated against the regular JSON Schema V7 validator.
    min_items_def: str | None = None  # must be non-negative integer
    max_items_def: str | None = None  # must be non-negative integer
    min_properties_def: str | None = None  # must be non-negative integer
    max_properties_def: str | None = None  # must be non-negative integer
    min_length_def: str | None = None  # must be non-negative integer
    max_length_def: str | None = None  # must be non-negative integer
    minimum_def: str | None = None
    maximum_def: str | None = None
    exclusive_minimum_def: str | None = None
    exclusive_maximum_def: str | None = None
    multiple_of_def: str | None = None  # must be positive number

    # --- composition --------------------------------------------------
    all_of: list[CHSchemaNode] = field(default_factory=list)  # the Draft7Validator forbids empty arrays for these
    any_of: list[CHSchemaNode] = field(default_factory=list)  # the Draft7Validator forbids empty arrays for these
    one_of: list[CHSchemaNode] = field(default_factory=list)  # the Draft7Validator forbids empty arrays for these
    not_: CHSchemaNode | None = None
    if_: CHSchemaNode | None = None
    then_: CHSchemaNode | None = None
    else_: CHSchemaNode | None = None

    # --- $defs / $ref -------------------------------------------------
    definitions: dict[str, CHSchemaNode] = field(default_factory=dict)
    ref_string: str | None = None
    cross_reference: CrossSchemaReference | None = None
    """
    Set instead of resolving locally when ``ref_string`` names *another* ValueDomain's schema.

    Kept apart from `ref_resolved` because the two are bound at different times and for different
    reasons: a local reference addresses a node of this same tree and is rebuilt with it, while a
    cross-schema one addresses a node of the *target application's* resolved schema, which only exists
    once that application has been built. See `bind_cross_schema_references`.
    """
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
    def __repr__(self):
        non_empty_fields = [f"defined_in={self.schema_owner}", f"location={self.location_id}"]
        if self.is_custom_type:
            if self.custom_type is not None:
                non_empty_fields.append(f"type={self.custom_type}")
            if self.provenance is not None:
                non_empty_fields.append(f"provenance={self.provenance}")
            if self.has_default:
                if self.parsed_default_expr is None:
                    non_empty_fields.append(f"default={self.default_expr}")
                else:
                    non_empty_fields.append(f"default={self.parsed_default_expr}")
        elif self.type_value is not None:
            non_empty_fields.append(f"type={self.type_value}")
            types = self.type_value
            if not isinstance(types, list):
                types = [types]
            if "object" in types:
                if self.properties:
                    non_empty_fields.append(f"properties={self.properties}")
                if self.custom_concept_data_constraints:
                    non_empty_fields.append(f"custom concept data constraints={self.custom_concept_data_constraints}")
                if self.property_names is not None:
                    non_empty_fields.append(f"property_names={self.property_names}")
                if self.pattern_properties:
                    non_empty_fields.append(f"patternProperties={self.pattern_properties}")
                if self.additional_properties is not None:
                    non_empty_fields.append(f"additionalProperties={self.additional_properties}")
                non_empty_fields.append(f"required={self.required}")
                if self.dependent_schemas:
                    non_empty_fields.append(f"dependent={self.dependent_schemas}")
            if "array" in types:
                if self.items is not None:
                    non_empty_fields.append(f"items={self.items}")
                if self.additional_items is not None:
                    non_empty_fields.append(f"additional_items={self.additional_items}")
                if self.contains is not None:
                    non_empty_fields.append(f"contains={self.contains}")
        else:
            if self.any_of:
                non_empty_fields.append(f"any_of={self.any_of}")
            if self.all_of:
                non_empty_fields.append(f"all_of={self.all_of}")
            if self.one_of:
                non_empty_fields.append(f"one_of={self.one_of}")
            if self.not_ is not None:
                non_empty_fields.append(f"not_={self.not_}")
            if self.if_ is not None:
                non_empty_fields.append(f"if={self.if_}")
            if self.then_ is not None:
                non_empty_fields.append(f"then_={self.then_}")
            if self.else_ is not None:
                non_empty_fields.append(f"else_={self.else_}")
        joined = ", ".join(non_empty_fields)
        return f"CHSchemaNode({joined})"

    def __copy__(self):
        """Shallow copy: containers (dict, list) are copied, but their elements are not."""
        cls = self.__class__
        result = cls.__new__(cls)

        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, (dict, list)):
                # Copy the container, but elements remain references
                setattr(result, field_name, field_value.copy())
            else:
                # Everything else: just copy the reference
                setattr(result, field_name, field_value)

        return result

    @property
    def is_boolean_schema(self) -> bool:
        return isinstance(self.canonical, bool)

    def undecided_literal_keywords_for(self, value: object) -> tuple[str, ...]:
        """
        The keywords of this node that would constrain ``value`` but are still written as an unsubstituted
        literal template variable.

        `jsonschema_parser` pops such a keyword out of the schema -- a Draft-07 validator cannot be handed
        the name ``"N"`` where it expects a number -- so while it is parked here, ``value`` has *not* been
        fully checked against this node, however clean the result looks.

        Only the keywords that apply to ``value``'s own JSON type count. An undecided ``minItems`` on a node
        reached with a string says nothing about that string, and deferring on it would refuse to decide
        values that are in fact decided.
        """
        family = json_type_family_of(value)
        if family is None:
            return ()
        return tuple(
            keyword
            for field_name, (keyword, applies_to) in LITERAL_KEYWORD_FIELDS.items()
            if applies_to == family and getattr(self, field_name) is not None
        )

    @property
    def is_template_dependent(self):
        """
        Whether anything in this subtree is still waiting on a template argument.
        """
        # Check `self.parsed_default_expr.is_value_template_dependent` for the parsed default expression
        # Don't check `self.parsed_default_expr.is_type_template_dependent`
        #  because that is implied by `self.custom_type.depends_on_templates`
        if (
            (self.custom_type is not None and self.custom_type.depends_on_templates)
            or (self.parsed_default_expr is not None and self.parsed_default_expr.is_value_template_dependent)
            or self.min_items_def is not None
            or self.max_items_def is not None
            or self.min_properties_def is not None
            or self.max_properties_def is not None
            or self.min_length_def is not None
            or self.max_length_def is not None
            or self.minimum_def is not None
            or self.maximum_def is not None
            or self.exclusive_minimum_def is not None
            or self.exclusive_maximum_def is not None
            or self.multiple_of_def is not None
        ):
            return True
        return any(child.is_template_dependent for _, child in self.iter_children())

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

        for index, custom_concept_data_constraint in enumerate(self.custom_concept_data_constraints):
            yield ("properties", index), custom_concept_data_constraint.value

    def is_expandable_container_child(self, relative_path: tuple[PathSegment, ...]) -> bool:
        """Whether the child at ``relative_path`` is an *element* of an expandable container of this node."""
        if not relative_path:
            return False
        keyword = relative_path[0]
        if keyword == "items":
            return isinstance(self.items, list)
        return keyword in EXPANDABLE_KEYWORDS

    def nodes_of_this_unit(self) -> Iterator[CHSchemaNode]:
        """
        Every node of the replication unit rooted at this one, this node included.

        A unit is one element of an expandable container, and stops where the next container begins:
        a ``T...`` below a nested ``anyOf`` belongs to *that* container, not to this one, which is why
        `documentation/TODO_VARIADIC_SCHEMA_EXPANSION.md` 2.3 counts ``"items": [{"anyOf": ["T1...", "T2..."]}]``
        as two units rather than one.
        ``$defs`` is skipped because its subtree is shared, not replicated (i.e. why it closes the context when parsing)
        """
        yield self
        for relative_path, child in self.iter_children():
            if relative_path[0] == "definitions" or self.is_expandable_container_child(relative_path):
                continue
            yield from child.nodes_of_this_unit()

    def variadic_templates_of_this_unit(self) -> Iterator[CHSchemaNode]:
        """The ``T...`` *schema nodes* of the replication unit rooted at this node."""
        return (node for node in self.nodes_of_this_unit() if node.expands_variadic)

    def variadic_selectors_of_this_unit(self) -> Iterator[tuple[CustomConceptDataConstraint, int]]:
        """
        The ``T...`` written in a `props`/`funcs` argument list of this unit, as ``(constraint, index)``.

        A selector argument is not a `CHSchemaNode`, so no walk over the children ever reaches it -- but it
        names a group exactly as a subschema does, and every rule about groups is about *the unit*, not
        about the kind of thing that names one. Yielding the position rather than the type is what lets a
        caller rewrite it in place, which is what binding one to a replica's argument needs.
        """
        for node in self.nodes_of_this_unit():
            for constraint in node.custom_concept_data_constraints:
                for index, restriction in enumerate(constraint.concept_restriction or ()):
                    if isinstance(restriction, ExpandedVariadicTemplateVariable):
                        yield constraint, index

    def variadic_template_parameters_of_this_unit(self) -> list[str]:
        """
        Every variadic template parameter in this unit, in the order first written and without repeats.
        """
        names = dict.fromkeys(node.custom_type.clean_name for node in self.variadic_templates_of_this_unit())
        names.update(
            dict.fromkeys(
                constraint.concept_restriction[index].clean_name
                for constraint, index in self.variadic_selectors_of_this_unit()
            )
        )
        return list(names)

    def apply_expanding(self, f: Callable[[CHSchemaNode, bool], list[CHSchemaNode]]) -> CHSchemaNode:
        """
        ``f`` is called as ``f(child, is_container_element)``.
        The flag matters twice over: only a list-valued container can absorb more than one node,
        and only an element of one roots a replication unit -- a ``T...`` under, say, ``contains`` belongs to whichever
        unit encloses *this* node, so treating that child as a unit of its own would replicate the wrong subtree.

        Everywhere but a container, a result of any length but 1 is a programming error.
        `apply` cannot express replication at all, because it maps every child position one-to-one.
        """

        def one(_child: CHSchemaNode) -> CHSchemaNode:
            _produced = f(_child, False)
            assert len(_produced) == 1, f"{len(_produced)} nodes produced at a position that holds only one."
            return _produced[0]

        if self.is_boolean_schema:
            return self

        res = copy(self)  # copies the containers, not their contents

        for key, child in res.properties.items():
            res.properties[key] = one(child)
        for key, child in res.pattern_properties.items():
            res.pattern_properties[key] = one(child)
        if isinstance(res.additional_properties, CHSchemaNode):
            res.additional_properties = one(res.additional_properties)
        if res.property_names is not None:
            res.property_names = one(res.property_names)

        if isinstance(res.items, list):
            res.items = [produced for child in res.items for produced in f(child, True)]
        elif isinstance(res.items, CHSchemaNode):
            res.items = one(res.items)
        if isinstance(res.additional_items, CHSchemaNode):
            res.additional_items = one(res.additional_items)
        if res.contains is not None:
            res.contains = one(res.contains)

        res.all_of = [produced for child in res.all_of for produced in f(child, True)]
        res.any_of = [produced for child in res.any_of for produced in f(child, True)]
        res.one_of = [produced for child in res.one_of for produced in f(child, True)]
        if res.not_ is not None:
            res.not_ = one(res.not_)
        if res.if_ is not None:
            res.if_ = one(res.if_)
        if res.then_ is not None:
            res.then_ = one(res.then_)
        if res.else_ is not None:
            res.else_ = one(res.else_)

        for key, child in res.definitions.items():
            res.definitions[key] = one(child)
        for key, child in res.dependent_schemas.items():
            res.dependent_schemas[key] = one(child)

        for index, constraint in enumerate(res.custom_concept_data_constraints):
            res.custom_concept_data_constraints[index] = copy(constraint)
            res.custom_concept_data_constraints[index].value = one(constraint.value)
        return res

    def walk(self) -> Iterator[CHSchemaNode]:
        """
        Depth-first iterator over this node and all of its descendants (custom-type and boolean-schema leaves included).
        """
        yield self
        for _, child in self.iter_children():
            yield from child.walk()

    def apply(self, f: Callable[[CHSchemaNode], CHSchemaNode]) -> CHSchemaNode:
        return self.apply_expanding(lambda _child, _is_container_element: [f(_child)])

    def short_repr(self) -> str:
        if self.is_boolean_schema:
            kind = f"bool({self.canonical})"
        elif self.is_custom_type:
            kind = f"custom:{self.custom_type.full_name} ({self.provenance.value})"
        else:
            kind = f"type={self.type_value!r}" if self.type_value is not None else "composite"
        location_id_str = ", ".join(f'"{x}"' for x in self.location_id)
        return f"  [{location_id_str}]: {kind}"
