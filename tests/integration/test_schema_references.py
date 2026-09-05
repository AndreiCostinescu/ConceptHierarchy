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
Integration tests: ``$ref`` inside an instantiation schema.

A reference is a **JSON Pointer from the schema root**, as it is in JSON Schema itself: ``#``,
``#/$defs/a``, ``#/$defs/a/$defs/b``, ``#/$defs/d/oneOf/1/properties/x``.

It did not used to be. The previous resolver matched ``^#/(definitions|$defs)/([^/]+)$`` and looked the
single name up in a map built by flattening *every* ``$defs`` in the tree with ``setdefault``. Three things
followed, and all three are pinned as changed behaviour below:

* a nested definition resolved **by accident**, addressed by a name that says nothing about where it is;
* two definitions sharing a name resolved to whichever the walk reached first, silently;
* a pointer with more than one step could not be written at all.

Two defects were fixed alongside, and both were invisible because the only reference anyone wrote was a
one-step one into a template-free ``$defs``:

* **substitution did not reach through a reference.** `CHSchemaNode.apply` rebuilds a node's children, but
  ``ref_resolved`` is a pointer *across* the tree rather than a child, so the rebuilt tree's references
  still addressed the declaration. `TestReferencesUnderSubstitution` is the measurement: before the fix,
  ``W<Integer>`` rejected an `Integer` *and* a `String`, because both were checked against ``W:T``.
* **a `default` on a referenced node was never registered** for the application, so it was never resolved.
  `TestADefaultReachedThroughAReference` pins that it now behaves exactly as an inline default does.

See ``documentation/TODO_FUNCTION_COMPOSITION_ARGS_SCHEMA.md`` for the value/expression parser interlock
this sits inside.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from concept_hierarchy.data.expressions.instantiated_value import ParsedCustomValue
from concept_hierarchy.data.jsonschema.parsed_schema import CHSchemaNode
from concept_hierarchy.data.parsers.jsonschema_parser import (
    parse_cross_schema_reference,
    parse_schema,
    resolve_schema_pointer,
    split_json_pointer,
    unescape_pointer_segment,
)
from concept_hierarchy.data.parsers.string_parser import StringParser
from concept_hierarchy.errors import ConceptHierarchyError
from tests.integration.test_default_expansion_cycles import LEAF
from tests.integration.test_expression_parsing import build_hierarchy, check_hierarchy
from tests.integration.test_schema_substitution import vd

# --------------------------------------------------------------------------------------------------
# Building schemas and hierarchies
# --------------------------------------------------------------------------------------------------

ANY_T = {"order": ["T"], "T": "ValueDomain"}


def referencing(pointer: str, defs: dict, template: object = None) -> dict:
    """A ValueDomain whose single property ``w`` is a ``$ref`` to ``pointer``."""
    return vd(
        "V",
        {"type": "object", "properties": {"w": {"$ref": pointer}}, "required": ["w"], "$defs": defs},
        template,
    )


def check_quietly(concepts: dict, instances: dict | None = None):
    """`check_hierarchy` without the parser's progress output, which is several screens per call."""
    with contextlib.redirect_stdout(io.StringIO()):
        return check_hierarchy(build_hierarchy(concepts, instances=instances))


def rejection(concepts: dict, instances: dict | None = None) -> str:
    with pytest.raises(ConceptHierarchyError) as excinfo:
        check_quietly(concepts, instances)
    return str(excinfo.value)


# --------------------------------------------------------------------------------------------------
# A parsed schema to navigate, built once
# --------------------------------------------------------------------------------------------------

NAVIGABLE_SCHEMA = {
    "type": "object",
    "properties": {"p1": {"type": "integer"}},
    "patternProperties": {"^x": {"type": "integer"}},
    "additionalProperties": {"type": "integer"},
    "propertyNames": {"type": "string"},
    "dependencies": {"p1": {"type": "object"}},
    "$defs": {
        "d": {"type": "integer"},
        "nested": {"$defs": {"deep": {"type": "integer"}}},
        "arr": {
            "type": "array",
            "items": [{"type": "integer"}, {"type": "string"}],
            "additionalItems": {"type": "integer"},
            "contains": {"type": "integer"},
        },
        "one": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
        "all": {"allOf": [{"type": "integer"}]},
        "any": {"anyOf": [{"type": "integer"}]},
        "single": {"type": "array", "items": {"type": "integer"}},
        "cond": {"if": {"type": "integer"}, "then": {"type": "integer"}, "else": {"type": "string"}},
        "neg": {"not": {"type": "string"}},
        "esc~0a": {"type": "integer"},
        "esc/b": {"type": "integer"},
    },
}


@pytest.fixture(scope="module")
def navigable() -> CHSchemaNode:
    context = check_quietly({})
    context.set_template_context(context.model.value_domains["Integer"].template_context)
    context.instantiation_schema_validator.set_identifier_where_types_are_defined("Integer")
    node, errors = parse_schema(NAVIGABLE_SCHEMA, context.instantiation_schema_validator, [])
    assert not errors, errors
    return node


def at(navigable: CHSchemaNode, pointer: str) -> CHSchemaNode:
    resolved, failure = resolve_schema_pointer(navigable, split_json_pointer(pointer))
    assert resolved is not None, f"{pointer} did not resolve: {failure}"
    return resolved


def failure_at(navigable: CHSchemaNode, pointer: str) -> str:
    resolved, failure = resolve_schema_pointer(navigable, split_json_pointer(pointer))
    assert resolved is None, f"{pointer} unexpectedly resolved"
    assert failure is not None
    return failure


# ==================================================================================================
# 1. Every keyword a pointer can step through
# ==================================================================================================


NAVIGABLE_POINTERS = [
    "#",
    "#/properties/p1",
    "#/patternProperties/^x",
    "#/additionalProperties",
    "#/propertyNames",
    "#/dependencies/p1",
    "#/$defs/d",
    "#/definitions/d",
    "#/$defs/arr/items/0",
    "#/$defs/arr/items/1",
    "#/$defs/arr/additionalItems",
    "#/$defs/arr/contains",
    "#/$defs/one/oneOf/0",
    "#/$defs/one/oneOf/1",
    "#/$defs/all/allOf/0",
    "#/$defs/any/anyOf/0",
    "#/$defs/single/items",
    "#/$defs/cond/if",
    "#/$defs/cond/then",
    "#/$defs/cond/else",
    "#/$defs/neg/not",
]


class TestPointerNavigation:
    """
    Every subschema-bearing keyword, because a pointer addresses what the *source text* wrote and each
    keyword spells its children differently -- by name, by index, or not at all.
    """

    @pytest.mark.parametrize("pointer", NAVIGABLE_POINTERS)
    def test_the_pointer_resolves(self, navigable, pointer):
        assert at(navigable, pointer) is not None

    def test_the_empty_pointer_is_the_root(self, navigable):
        assert at(navigable, "#") is navigable

    def test_definitions_and_defs_are_the_same_container(self, navigable):
        """`CHSchemaNode` keeps both spellings in one dict, so either addresses the other's entries."""
        assert at(navigable, "#/$defs/d") is at(navigable, "#/definitions/d")

    def test_a_pointer_steps_through_nested_definitions(self, navigable):
        assert at(navigable, "#/$defs/nested/$defs/deep") is not None

    def test_a_tuple_items_index_selects_the_right_branch(self, navigable):
        """``items`` as an array is positional, and the two positions must not be interchangeable."""
        assert at(navigable, "#/$defs/arr/items/0") is not at(navigable, "#/$defs/arr/items/1")

    def test_a_one_of_index_selects_the_right_branch(self, navigable):
        assert at(navigable, "#/$defs/one/oneOf/0") is not at(navigable, "#/$defs/one/oneOf/1")


class TestPointerEscaping:
    """RFC 6901: ``~1`` is a literal ``/`` and ``~0`` a literal ``~``, unescaped in that order."""

    def test_a_tilde_in_a_definition_name(self, navigable):
        assert at(navigable, "#/$defs/esc~00a") is not None

    def test_a_slash_in_a_definition_name(self, navigable):
        assert at(navigable, "#/$defs/esc~1b") is not None

    def test_the_order_of_unescaping(self):
        """``~01`` is ``~1`` and not ``/``: unescaping ``~1`` first would produce the wrong name."""
        assert unescape_pointer_segment("~01") == "~1"

    def test_a_slash_in_a_name_is_one_segment(self):
        assert split_json_pointer("#/$defs/esc~1b") == ["$defs", "esc/b"]


# ==================================================================================================
# 2. Pointers that must not resolve
# ==================================================================================================


class TestPointersThatDoNotResolve:
    """
    Each reports *which step* could not be taken. A resolver that answered only "cannot resolve" would
    leave the author of a five-step pointer to find the wrong step themselves.
    """

    def test_a_bare_name_no_longer_finds_a_nested_definition(self, navigable):
        """
        The behaviour change. ``deep`` lives under ``nested``; the flattened lookup found it anyway, and
        would have found an unrelated ``deep`` in a sibling subtree just as happily.
        """
        assert "there is no definition named 'deep'" in failure_at(navigable, "#/$defs/deep")

    def test_an_unknown_definition_lists_what_there_is(self, navigable):
        failure = failure_at(navigable, "#/$defs/zzz")
        assert "there is no definition named 'zzz'" in failure
        assert "'d'" in failure, "the available names help more than the absent one"

    def test_a_keyword_that_can_not_be_pointed_into(self, navigable):
        assert "not a schema keyword" in failure_at(navigable, "#/$defs/d/nope")

    def test_an_index_beyond_the_end(self, navigable):
        assert "index 9 does not exist" in failure_at(navigable, "#/$defs/one/oneOf/9")

    def test_an_index_that_is_not_a_number(self, navigable):
        assert "needs a numeric index" in failure_at(navigable, "#/$defs/one/oneOf/x")

    def test_a_named_keyword_with_no_name_after_it(self, navigable):
        assert "needs a name after it" in failure_at(navigable, "#/$defs")

    def test_an_indexed_keyword_with_no_index_after_it(self, navigable):
        assert "needs an index after it" in failure_at(navigable, "#/$defs/one/oneOf")


class TestBadReferencesAreReported:
    """The same failures, as they reach an author: a rejected hierarchy naming the reference."""

    def test_a_pointer_that_does_not_resolve_rejects_the_hierarchy(self):
        text = rejection({**LEAF, **referencing("#/$defs/nope", {"d": {"type": "Leaf"}})})
        assert "$ref" in text and "nope" in text

    def test_a_reference_that_is_not_a_local_pointer_is_rejected(self):
        text = rejection({**LEAF, **referencing("http://example.com/schema", {"d": {"type": "Leaf"}})})
        assert "JSON Pointer from the" in text

    def test_a_bare_nested_name_is_rejected_rather_than_guessed(self):
        """What used to work by accident now fails loudly, which is the point of the change."""
        text = rejection({**LEAF, **referencing("#/$defs/deep", {"outer": {"$defs": {"deep": {"type": "Leaf"}}}})})
        assert "there is no definition named 'deep'" in text


# ==================================================================================================
# 3. References that are actually used to check a value
# ==================================================================================================


class TestAReferencedSchemaChecksTheValue:
    """Resolution is not the point on its own -- the referenced schema has to be the one applied."""

    def test_a_value_matching_the_referenced_schema_is_accepted(self):
        check_quietly({**LEAF, **referencing("#/$defs/d", {"d": {"type": "Leaf"}})}, {"p": {"V": {"w": {}}}})

    def test_a_value_not_matching_the_referenced_schema_is_rejected(self):
        text = rejection({**LEAF, **referencing("#/$defs/d", {"d": {"type": "Integer"}})}, {"p": {"V": {"w": {}}}})
        assert "Invalid expression" in text

    def test_a_deep_pointer_applies_the_schema_it_points_at(self):
        """``oneOf/1`` is the `Integer` branch; the `String` branch beside it must not be what is applied."""
        defs = {"d": {"oneOf": [{"type": "String"}, {"type": "Integer"}]}}
        check_quietly({**LEAF, **referencing("#/$defs/d/oneOf/1", defs)}, {"p": {"V": {"w": 1}}})
        assert "Invalid expression" in rejection(
            {**LEAF, **referencing("#/$defs/d/oneOf/1", defs)}, {"p": {"V": {"w": "s:x"}}}
        )

    def test_a_pointer_into_a_nested_definition_applies_that_one(self):
        defs = {"outer": {"$defs": {"deep": {"type": "Integer"}}}}
        check_quietly({**LEAF, **referencing("#/$defs/outer/$defs/deep", defs)}, {"p": {"V": {"w": 1}}})


# ==================================================================================================
# 4. References under template substitution
# ==================================================================================================


class TestReferencesUnderSubstitution:
    """
    The defect this file exists for. ``V<T>``'s ``w`` references a ``$defs`` entry of type ``T``, so the
    reference has to be followed into *this application's* copy of the schema.

    Before `rewire_resolved_refs`, every one of these four was rejected -- including the two that are
    correct -- with a complaint about the unsubstituted ``V:T``. Both halves matter: rejecting the wrong
    value is not evidence of anything if the right value is rejected too.
    """

    CONCEPTS = {**LEAF, "_": {}}

    @staticmethod
    def _hierarchy() -> dict:
        return {**LEAF, **referencing("#/$defs/d", {"d": {"type": "T"}}, ANY_T)}

    @pytest.mark.parametrize(
        ("application", "value"),
        [("V<Integer>", 1), ("V<String>", "s:x"), ("V<Leaf>", {})],
    )
    def test_the_value_of_the_applied_argument_is_accepted(self, application, value):
        check_quietly(self._hierarchy(), {"p": {application: {"w": value}}})

    @pytest.mark.parametrize(
        ("application", "value"),
        [("V<Integer>", "s:x"), ("V<String>", 1), ("V<Leaf>", 1)],
    )
    def test_the_value_of_another_argument_is_rejected(self, application, value):
        assert "Invalid expression" in rejection(self._hierarchy(), {"p": {application: {"w": value}}})

    def test_the_reference_does_not_leak_the_unsubstituted_template_variable(self):
        """The symptom that named the bug: the diagnosis mentioned ``V:T``, which no value can ever be."""
        text = rejection(self._hierarchy(), {"p": {"V<Integer>": {"w": "s:x"}}})
        assert "V:T" not in text, f"the template variable survived substitution: {text[:400]}"


class TestADefaultReachedThroughAReference:
    """
    A ``default`` on a referenced node, which is the other half of the same defect.

    `build_resolved_instantiation_schema` registers default sites by walking the rebuilt tree; a reference
    pointing back at the declaration reached a node that walk never registered, so the default was never
    resolved. It behaves as an inline default now, and the inline one is carried alongside as the control
    -- an assertion that only says "the referenced one did something" proves nothing without it.
    """

    INLINE = vd("U", {"type": "object", "properties": {"w": {"type": "Leaf", "default": {}}}})
    VIA_REF = referencing("#/$defs/d", {"d": {"type": "Leaf", "default": {}}})

    @staticmethod
    def _materialised(context, global_variable: str) -> list[tuple[bool, bool]]:
        value = context.model.instances[global_variable].value.value.value
        return [
            (node.used_default, node.expression is not None)
            for node in value.walk()
            if isinstance(node, ParsedCustomValue)
        ]

    def test_an_inline_default_materialises_and_resolves(self):
        context = check_quietly({**LEAF, **self.INLINE}, {"p": {"U": {}}})
        assert self._materialised(context, "p") == [(True, True)]

    def test_a_default_behind_a_reference_does_the_same(self):
        context = check_quietly({**LEAF, **self.VIA_REF}, {"p": {"V": {}}})
        assert self._materialised(context, "p") == [(True, True)]

    def test_the_same_in_a_templated_value_domain(self):
        """Where the schema is genuinely rebuilt, rather than merely copied."""
        templated = referencing("#/$defs/d", {"d": {"type": "T", "default": {}}}, ANY_T)
        context = check_quietly({**LEAF, **templated}, {"p": {"V<Leaf>": {}}})
        assert self._materialised(context, "p") == [(True, True)]


# ==================================================================================================
# 5. References into another ValueDomain's schema
# ==================================================================================================

TARGET = vd(
    "Target",
    {
        "type": "object",
        "properties": {"t": {"type": "Integer"}},
        "$defs": {"num": {"type": "Integer"}, "leaf": {"type": "Leaf"}},
    },
)
"""An ordinary ValueDomain with something worth pointing at."""

BOX = vd(
    "Box",
    {"type": "object", "properties": {"b": {"type": "T"}}, "$defs": {"inner": {"type": "T"}}},
    ANY_T,
)
"""A templated one, so a reference has to name an *application* for its ``$defs`` to mean anything."""

MULTI = {
    "Multi": {
        "directParents": ["ValueDomain"],
        "data": {
            "templateContext": ANY_T,
            "instantiation": [
                [
                    ["Integer"],
                    {"type": "object", "properties": {"m": {"type": "Integer"}}, "$defs": {"x": {"type": "Integer"}}},
                ],
                [[""], {"type": "object", "properties": {"m": {"type": "T"}}, "$defs": {"x": {"type": "Leaf"}}}],
            ],
        },
    }
}
"""Two template-dependent instantiation schemas, so a reference must say which one it means."""

NO_INSTANTIATION = {"Bare": {"directParents": ["ValueDomain"], "data": {}}}
A_DOMAIN_CONCEPT = {"Animal": {"directParents": ["Concept"], "data": {"properties": {}}}}


def cross_referencing(pointer: str, name: str = "V") -> dict:
    """A ValueDomain whose property ``w`` is a ``$ref`` into another ValueDomain's schema."""
    return vd(name, {"type": "object", "properties": {"w": {"$ref": pointer}}, "required": ["w"]})


def reference_failure(concepts: dict, pointer: str) -> str:
    """The ``Cannot resolve $ref`` line, which is the whole diagnosis for a rejected reference."""
    text = rejection({**LEAF, **concepts, **cross_referencing(pointer)}, {"p": {"V": {"w": 1}}})
    lines = [line.strip() for line in text.splitlines() if "Cannot resolve $ref" in line]
    assert lines, f"expected a reference diagnosis, got: {text[:400]}"
    return lines[0]


class TestCrossSchemaReferences:
    """
    ``#ch#/<Type>[/<index>]/#/<pointer>``: the referenced fragment is applied exactly as if it had been
    written in place.
    """

    def test_a_reference_to_another_value_domains_definition(self):
        check_quietly({**LEAF, **TARGET, **cross_referencing("#ch#/Target/#/$defs/num")}, {"p": {"V": {"w": 1}}})

    def test_the_referenced_schema_is_the_one_applied(self):
        """Resolving is not the claim; being *checked against* is."""
        assert "Invalid expression" in rejection(
            {**LEAF, **TARGET, **cross_referencing("#ch#/Target/#/$defs/num")}, {"p": {"V": {"w": "s:x"}}}
        )

    def test_a_reference_to_the_whole_target_schema(self):
        """``#`` on its own is the root of the target, which is what a bare ``#`` means locally too."""
        check_quietly({**LEAF, **TARGET, **cross_referencing("#ch#/Target/#")}, {"p": {"V": {"w": {"t": 1}}}})

    def test_a_reference_into_the_targets_properties(self):
        check_quietly({**LEAF, **TARGET, **cross_referencing("#ch#/Target/#/properties/t")}, {"p": {"V": {"w": 1}}})


class TestTheReferenceNamesAnApplication:
    """
    The reason the target is a *type* and not a concept name: ``Box``'s ``$defs/inner`` is of type ``T``,
    and only an application says what that is. Pointing at the declaration would hand the referring schema
    a variable it cannot bind -- which is exactly the bug `rewire_resolved_refs` fixed one level down.
    """

    def test_the_targets_template_argument_grounds_the_fragment(self):
        check_quietly({**LEAF, **BOX, **cross_referencing("#ch#/Box<Integer>/#/$defs/inner")}, {"p": {"V": {"w": 1}}})

    def test_a_value_of_another_type_is_rejected(self):
        assert "Invalid expression" in rejection(
            {**LEAF, **BOX, **cross_referencing("#ch#/Box<Integer>/#/$defs/inner")}, {"p": {"V": {"w": "s:x"}}}
        )

    def test_a_different_application_is_a_different_fragment(self):
        """The same declared ``$defs`` entry, two applications, two types -- and both must hold."""
        check_quietly(
            {**LEAF, **BOX, **cross_referencing("#ch#/Box<String>/#/$defs/inner")}, {"p": {"V": {"w": "s:x"}}}
        )
        assert "Invalid expression" in rejection(
            {**LEAF, **BOX, **cross_referencing("#ch#/Box<String>/#/$defs/inner")}, {"p": {"V": {"w": 1}}}
        )

    def test_a_template_variable_may_not_be_written_in_the_reference(self):
        """
        ``#ch#/Box<T>`` has no meaning: the reference is resolved once, for every application of the
        referring schema, so there is no ``T`` in scope to bind.
        """
        assert "is not a ground type" in reference_failure(BOX, "#ch#/Box<T>/#/$defs/inner")


class TestTheSchemaIndex:
    """
    Present exactly when the target wrote a template-dependent instantiation. An index on a target that has
    one schema would suggest a choice that does not exist; an absent index on a target that has several
    would leave the reader to guess which.
    """

    def test_an_index_selects_the_schema(self):
        check_quietly({**LEAF, **MULTI, **cross_referencing("#ch#/Multi<Integer>/0/#/$defs/x")}, {"p": {"V": {"w": 1}}})

    def test_a_different_index_selects_a_different_schema(self):
        check_quietly(
            {**LEAF, **MULTI, **cross_referencing("#ch#/Multi<Integer>/1/#/$defs/x")}, {"p": {"V": {"w": {}}}}
        )

    def test_an_absent_index_on_a_template_dependent_target_is_rejected(self):
        failure = reference_failure(MULTI, "#ch#/Multi<Integer>/#/$defs/x")
        assert "must say which one" in failure

    def test_an_index_beyond_the_declared_schemas_is_rejected(self):
        assert "no schema at index 9" in reference_failure(MULTI, "#ch#/Multi<Integer>/9/#/$defs/x")

    def test_an_index_on_a_target_that_has_only_one_schema_is_rejected(self):
        """Including ``/0``, which is the only value it could have taken."""
        failure = reference_failure(TARGET, "#ch#/Target/0/#/$defs/num")
        assert "an index must not be written" in failure


class TestCrossSchemaReferenceValidation:
    """Everything that can be decided without an application, reported before any value is parsed."""

    def test_a_type_that_does_not_exist(self):
        assert "is not a ground type" in reference_failure(TARGET, "#ch#/NoSuchType/#/$defs/num")

    def test_a_target_that_is_not_a_value_domain(self):
        assert "is not a ValueDomain" in reference_failure(A_DOMAIN_CONCEPT, "#ch#/Animal/#/$defs/x")

    def test_an_abstract_target(self):
        assert "is abstract" in reference_failure(TARGET, "#ch#/ValueDomain/#/$defs/num")

    def test_a_target_that_declares_no_instantiation(self):
        """Its schema is the unconstrained one, which has no definitions to point at."""
        assert "declares no instantiation" in reference_failure(NO_INSTANTIATION, "#ch#/Bare/#/$defs/x")

    def test_a_pointer_that_does_not_resolve_in_the_target(self):
        failure = reference_failure(TARGET, "#ch#/Target/#/$defs/zzz")
        assert "in the instantiation schema of Target" in failure
        assert "there is no definition named 'zzz'" in failure

    def test_a_reference_with_no_pointer_separator(self):
        assert 'needs a "#" where the pointer' in reference_failure(TARGET, "#ch#/Target/$defs/num")

    def test_a_reference_with_no_type(self):
        assert "must name the type" in reference_failure(TARGET, "#ch#//#/$defs/num")

    def test_a_non_numeric_index(self):
        assert "must be a non-negative integer" in reference_failure(TARGET, "#ch#/Target/x/#/$defs/num")

    def test_more_than_a_type_and_an_index(self):
        assert "may stand before the pointer" in reference_failure(TARGET, "#ch#/Target/0/1/#/$defs/num")


class TestCrossSchemaReferenceCycles:
    """
    A reference chain that comes back round terminates, because `build_resolved_instantiation_schema`
    publishes the schema to the memo *before* binding its references -- the same entry that already makes
    a default-expansion cycle diagnosable rather than infinite. A reference is satisfied by the structure
    of the target, which is complete at that point; only its defaults are still outstanding.
    """

    def test_two_schemas_referencing_each_other(self):
        a = vd(
            "A",
            {"type": "object", "properties": {"a": {"$ref": "#ch#/B/#/$defs/x"}}, "$defs": {"y": {"type": "Integer"}}},
        )
        b = vd(
            "B",
            {"type": "object", "properties": {"b": {"$ref": "#ch#/A/#/$defs/y"}}, "$defs": {"x": {"type": "Integer"}}},
        )
        check_quietly({**LEAF, **a, **b}, {"p": {"A": {"a": 1}}})

    def test_a_schema_referencing_itself(self):
        own = vd(
            "S",
            {"type": "object", "properties": {"s": {"$ref": "#ch#/S/#/$defs/z"}}, "$defs": {"z": {"type": "Integer"}}},
        )
        check_quietly({**LEAF, **own}, {"p": {"S": {"s": 1}}})


class TestADefaultBehindACrossReference:
    """
    The identity question, for the case where the target really is another tree.

    A local reference is safe because its target is rebuilt with it. A cross-schema one is safe for a
    different reason: it lands on a node of the *target application's own* resolved schema, which that
    application built and registered. Pointing at the target's declaration instead would reach a site
    nothing ever registered -- so this asserts the resolved default, not merely that the value is accepted.
    """

    @staticmethod
    def _materialised(context) -> list[tuple[bool, bool]]:
        value = context.model.instances["p"].value.value.value
        return [
            (node.used_default, node.expression is not None)
            for node in value.walk()
            if isinstance(node, ParsedCustomValue)
        ]

    def test_the_default_materialises_and_resolves(self):
        target = vd(
            "Tgt",
            {
                "type": "object",
                "properties": {"t": {"type": "Integer"}},
                "$defs": {"d": {"type": "Leaf", "default": {}}},
            },
        )
        user = vd("U", {"type": "object", "properties": {"w": {"$ref": "#ch#/Tgt/#/$defs/d"}}})
        assert self._materialised(check_quietly({**LEAF, **target, **user}, {"p": {"U": {}}})) == [(True, True)]

    def test_a_default_on_a_templated_target_is_grounded_for_the_referenced_application(self):
        """
        The discriminating case, and the reason the test above is not enough on its own.

        A non-templated target's declaration already carries a parsed default -- `init_expressions` parses
        every declared instantiation default -- so binding to the declaration *looks* right there. Only a
        templated target separates the two: the declaration's default sits at ``Tgt:T``, and just
        materialising it is not the claim. The claim is that it is materialised at ``Leaf``, which is what
        binding to ``Tgt<Leaf>``'s own resolved schema gets.
        """
        target = vd(
            "Tgt",
            {"type": "object", "properties": {"t": {"type": "T"}}, "$defs": {"d": {"type": "T", "default": {}}}},
            ANY_T,
        )
        user = vd("U", {"type": "object", "properties": {"w": {"$ref": "#ch#/Tgt<Leaf>/#/$defs/d"}}})
        context = check_quietly({**LEAF, **target, **user}, {"p": {"U": {}}})
        assert self._materialised(context) == [(True, True)]

        value = context.model.instances["p"].value.value.value
        materialised_types = [str(node.custom_type) for node in value.walk() if isinstance(node, ParsedCustomValue)]
        assert materialised_types == ["Leaf"], "the referenced fragment must arrive ground, not as Tgt:T"


# ==================================================================================================
# 9. Reading the reference, rather than searching it
# ==================================================================================================


def reference(written: str):
    parsed, failure = parse_cross_schema_reference(written)
    assert parsed is not None, f"{written} did not parse: {failure}"
    return parsed


def reference_reason(written: str) -> str:
    parsed, failure = parse_cross_schema_reference(written)
    assert parsed is None, f"{written} unexpectedly parsed to {parsed}"
    assert failure is not None
    return failure


class TestTheReferenceIsParsedNotSplit:
    """
    A type may hold a **literal template variable**, and a literal may hold any character -- so neither
    ``/`` nor ``#`` can be found by searching.

    Splitting on ``/`` cuts ``Seq<"a/b">`` in half. Searching for the ``#`` that starts the pointer finds
    the one inside ``Seq<"a#b">``. Both produce a type name that is not a type and a pointer that is not a
    pointer, and both were what the first version of this parser did. Reading the grammar in order is what
    removes the question -- the type ends at the first ``/`` that is neither quoted nor nested, and
    everything after it is whatever the grammar allows, never whatever a search happens to land on.
    """

    def test_a_slash_inside_a_literal_does_not_end_the_type(self):
        parsed = reference('#ch#/Seq<"a/b">/#/$defs/x')
        assert parsed.type_name == 'Seq<"a/b">'
        assert list(parsed.pointer) == ["$defs", "x"]

    def test_a_hash_inside_a_literal_does_not_start_the_pointer(self):
        parsed = reference('#ch#/Seq<"a#b">/#/$defs/x')
        assert parsed.type_name == 'Seq<"a#b">'
        assert list(parsed.pointer) == ["$defs", "x"]

    def test_both_at_once_with_an_index(self):
        parsed = reference('#ch#/Seq<"a/b#c">/2/#/$defs/x')
        assert parsed.type_name == 'Seq<"a/b#c">'
        assert parsed.schema_index == 2
        assert list(parsed.pointer) == ["$defs", "x"]

    def test_an_escaped_quote_inside_a_literal(self):
        """The literal ends at the quote that is *not* escaped, which `parse_string_literal` decides."""
        parsed = reference('#ch#/Seq<"a\\"b/c">/#/$defs/x')
        assert parsed.type_name == 'Seq<"a\\"b/c">'
        assert list(parsed.pointer) == ["$defs", "x"]

    def test_a_nested_template_argument_list(self):
        parsed = reference('#ch#/Map<String, Seq<"x/y">>/#/$defs/x')
        assert parsed.type_name == 'Map<String, Seq<"x/y">>'

    def test_an_unterminated_literal_is_reported_as_a_malformed_type(self):
        assert "not well-formed" in reference_reason('#ch#/Seq<"unterminated/#/$defs/x')


class TestTheReferenceGrammar:
    """The three fields, and each way of getting them wrong."""

    def test_a_plain_type_and_pointer(self):
        parsed = reference("#ch#/Target/#/$defs/x")
        assert (parsed.type_name, parsed.schema_index, list(parsed.pointer)) == ("Target", None, ["$defs", "x"])

    def test_an_index_between_them(self):
        parsed = reference("#ch#/Target/0/#/$defs/x")
        assert (parsed.type_name, parsed.schema_index, list(parsed.pointer)) == ("Target", 0, ["$defs", "x"])

    def test_a_pointer_to_the_root_of_the_target(self):
        assert reference("#ch#/Target/#").pointer == ()

    def test_the_pointer_segments_are_unescaped(self):
        """After the ``#`` it *is* a JSON Pointer, where a literal ``/`` is written ``~1``."""
        assert list(reference("#ch#/Target/#/$defs/a~1b").pointer) == ["$defs", "a/b"]

    def test_a_missing_pointer_start(self):
        assert 'needs a "#"' in reference_reason("#ch#/Target/$defs/x")

    def test_an_index_that_is_not_a_number(self):
        """It stands where an index stands, so it is reported as a bad index, not as a missing ``#``."""
        assert "must be a non-negative integer, not 'x'" in reference_reason("#ch#/Target/x/#/$defs/x")

    def test_more_than_a_type_and_an_index(self):
        assert "may stand before the pointer" in reference_reason("#ch#/Target/0/1/#/$defs/x")

    def test_no_type_at_all(self):
        assert "must name the type" in reference_reason("#ch#//#/$defs/x")


class TestConsumeUntilUnnested:
    """
    The scanner itself, since the cases it exists for are easier to state on it than through a hierarchy.
    """

    @staticmethod
    def _scan(text: str, delimiters: str = "/") -> tuple[str, str]:
        parser = StringParser(text)
        return parser.consume_until_unnested(delimiters), parser.remaining()

    def test_it_stops_at_a_plain_delimiter(self):
        assert self._scan("Target/rest") == ("Target", "/rest")

    def test_it_passes_over_a_delimiter_inside_a_literal(self):
        assert self._scan('Seq<"a/b">/rest') == ('Seq<"a/b">', "/rest")

    def test_it_passes_over_a_delimiter_inside_a_template_argument_list(self):
        """``<>`` is tracked by depth, so a delimiter that is nested is not a delimiter."""
        assert self._scan("Map<A/B>/rest") == ("Map<A/B>", "/rest")

    def test_it_tracks_nesting_depth(self):
        assert self._scan("Map<A, Seq<B/C>>/rest") == ("Map<A, Seq<B/C>>", "/rest")

    def test_it_handles_the_other_bracket_kinds(self):
        assert self._scan("F(a/b)[c/d]/rest") == ("F(a/b)[c/d]", "/rest")

    def test_it_stops_at_end_of_input_when_there_is_no_delimiter(self):
        assert self._scan("Target") == ("Target", "")

    def test_an_unbalanced_closer_does_not_make_the_depth_negative(self):
        """A malformed type is the type parser's to report; the scanner must not compound it."""
        assert self._scan("A>B/rest") == ("A>B", "/rest")


# ==================================================================================================
# 10. A schema that applies to a value by applying to the same value again
# ==================================================================================================


def cycle_rejection(concepts: dict, instances: dict | None = None) -> str:
    """
    The hierarchy must be rejected, and rejected *as a diagnosed cycle*.

    The distinction is the point. Every one of these used to arrive as "Ran out of stack while parsing
    this expression", which names the value being parsed and says nothing about the schema that cannot be
    satisfied -- and which a bare ``raises`` would accept just as happily.
    """
    text = rejection(concepts, instances)
    assert "can never be applied" in text, f"expected a diagnosed cycle, got: {text[:400]}"
    assert "Ran out of stack" not in text, "the cycle must be diagnosed, not left to the interpreter"
    assert "levels deep" not in text, "the cycle must be diagnosed, not left to the depth bound"
    return text


def with_defs(defs: dict, pointer: str = "#/$defs/a", name: str = "V") -> dict:
    return vd(name, {"type": "object", "properties": {"w": {"$ref": pointer}}, "required": ["w"], "$defs": defs})


class TestNonConsumingReferenceCyclesAreRejected:
    """
    A cycle among schemas applied to the *same* value describes no value at all.

    The edge relation is the one `value_instantiation_parser` documents: the keywords that hand the value
    straight on -- ``allOf``, ``anyOf``, ``oneOf``, ``not``, ``if``/``then``/``else``, ``dependencies`` --
    plus ``$ref`` itself. A cycle among those re-enters the same node with the same value forever.
    """

    def test_a_reference_to_itself(self):
        cycle_rejection({**LEAF, **with_defs({"a": {"$ref": "#/$defs/a"}})}, {"p": {"V": {"w": 1}}})

    def test_a_two_step_reference_cycle(self):
        defs = {"a": {"$ref": "#/$defs/b"}, "b": {"$ref": "#/$defs/a"}}
        cycle_rejection({**LEAF, **with_defs(defs)}, {"p": {"V": {"w": 1}}})

    @pytest.mark.parametrize("keyword", ["allOf", "anyOf", "oneOf"])
    def test_a_cycle_through_a_composition_keyword(self, keyword):
        cycle_rejection({**LEAF, **with_defs({"a": {keyword: [{"$ref": "#/$defs/a"}]}})}, {"p": {"V": {"w": 1}}})

    def test_a_cycle_through_if_then(self):
        defs = {"a": {"if": {"type": "object"}, "then": {"$ref": "#/$defs/a"}}}
        cycle_rejection({**LEAF, **with_defs(defs)}, {"p": {"V": {"w": 1}}})

    def test_the_diagnosis_names_the_schema_and_the_chain(self):
        text = cycle_rejection({**LEAF, **with_defs({"a": {"$ref": "#/$defs/a"}})}, {"p": {"V": {"w": 1}}})
        assert '"$defs": "a"' in text, "the schema at fault must be located"
        assert "The cycle runs" in text, "and the chain that closes on it named"

    def test_it_is_reported_even_when_nothing_points_at_it(self):
        """
        A definition is ill-formed whether or not anything happens to reference it yet, and reporting it
        only on use would let it sit until some later hierarchy reached it.
        """
        defs = {"a": {"type": "Leaf"}, "unused": {"$ref": "#/$defs/unused"}}
        cycle_rejection({**LEAF, **with_defs(defs)}, {"p": {"V": {"w": {}}}})


class TestRecursionThatConsumesValueIsAccepted:
    """
    The passing guards, and they are what make the rule a rule rather than a ban on recursion.

    A reference under ``properties`` or ``items`` steps through the value before it comes back round, and
    the value is finite -- so the recursion terminates. Rejecting these would outlaw every recursive
    schema, which is most of the point of ``$ref``.
    """

    def test_a_reference_under_properties(self):
        defs = {"a": {"type": "object", "properties": {"c": {"$ref": "#/$defs/a"}}}}
        check_quietly({**LEAF, **with_defs(defs)}, {"p": {"V": {"w": {}}}})

    def test_a_reference_under_items(self):
        defs = {"a": {"type": "array", "items": {"$ref": "#/$defs/a"}}}
        check_quietly({**LEAF, **with_defs(defs)}, {"p": {"V": {"w": []}}})

    def test_a_nested_recursive_value_still_parses(self):
        """Not merely accepted at definition time: a value that actually recurses must parse."""
        defs = {"a": {"type": "object", "properties": {"c": {"$ref": "#/$defs/a"}}}}
        check_quietly({**LEAF, **with_defs(defs)}, {"p": {"V": {"w": {"c": {"c": {}}}}}})

    def test_a_chain_of_references_that_ends(self):
        defs = {"a": {"$ref": "#/$defs/b"}, "b": {"type": "Leaf"}}
        check_quietly({**LEAF, **with_defs(defs)}, {"p": {"V": {"w": {}}}})


class TestCrossSchemaCyclesAreRejectedToo:
    """
    The same rule once ``#ch#`` references are bound, which `parse_schema` cannot see: a cross-schema
    reference has no target until the application it names is resolved.

    Distinct from the reference-*resolution* cycle the memo already handles -- that one is about building
    the schemas, this one about applying them.
    """

    def test_two_schemas_whose_all_of_reach_each_other(self):
        a = vd(
            "A",
            {
                "type": "object",
                "properties": {"a": {"$ref": "#/$defs/x"}},
                "$defs": {"x": {"allOf": [{"$ref": "#ch#/B/#/$defs/y"}]}},
            },
        )
        b = vd(
            "B",
            {
                "type": "object",
                "properties": {"b": {"type": "Integer"}},
                "$defs": {"y": {"allOf": [{"$ref": "#ch#/A/#/$defs/x"}]}},
            },
        )
        cycle_rejection({**LEAF, **a, **b}, {"p": {"A": {"a": 1}}})

    def test_two_schemas_referencing_each_other_directly(self):
        c = vd(
            "C",
            {
                "type": "object",
                "properties": {"a": {"$ref": "#/$defs/x"}},
                "$defs": {"x": {"$ref": "#ch#/D/#/$defs/y"}},
            },
        )
        d = vd(
            "D",
            {
                "type": "object",
                "properties": {"b": {"type": "Integer"}},
                "$defs": {"y": {"$ref": "#ch#/C/#/$defs/x"}},
            },
        )
        cycle_rejection({**LEAF, **c, **d}, {"p": {"C": {"a": 1}}})
