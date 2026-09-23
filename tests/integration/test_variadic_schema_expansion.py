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
`T...` in an instantiation schema: where it may be written, and what it expands to.

A specification suite, written before the implementation
(`documentation/TODO_VARIADIC_SCHEMA_EXPANSION.md`) and kept as the record of what that plan decided. It
is all passing; a case that stops passing is a decision being reversed, not a test that needs relaxing.

Four things are asserted separately, because passing one of them proves little:

* **validity** -- whether the schema is accepted at declaration time. Parse-time only, no value involved;
* **acceptance** -- whether a value passes at a site of the applied type;
* **rejection of the complement** -- that the schema did not collapse to "accept everything". An empty
  expansion has two failure modes and they are opposites, so a test that only checks acceptance cannot
  tell a working expansion from a schema that stopped constraining anything;
* **shape** -- what the rebuilt schema actually looks like, through `resolved_schema`. Two different wrong
  expansions can accept and refuse the same values: a nested container given one branch instead of N still
  checks values, a `$ref` left pointing at the declaration still resolves. Where the difference is only
  structural, the structure is what is asserted.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from concept_hierarchy.data.expressions.expression_utils import ValueDomainArgumentProvenance
from concept_hierarchy.data.jsonschema.parsed_schema import CHSchemaNode
from concept_hierarchy.definitions.concept_definition_domain_concept import ForPropertyOrFunction
from concept_hierarchy.definitions.concept_hierarchy import ConceptHierarchyDefinition
from concept_hierarchy.errors import ConceptHierarchyError
from concept_hierarchy.validator.checker import ConceptHierarchyChecker
from tests.integration.test_expression_parsing import build_hierarchy, check_hierarchy, error_messages

# --------------------------------------------------------------------------------------------------
# Fixtures
#
# One ValueDomain per array-like container of the plan's table, each with a single variadic parameter,
# plus `Zip` for two groups in one container. A group is applied with the bracket syntax --
# ``Variant<[Integer, String]>`` -- and two groups need `variadicGroupIdentifiers` to tell them apart.
# --------------------------------------------------------------------------------------------------


def variadic_vd(
    name: str,
    instantiation: object,
    params: tuple[str, ...] = ("T",),
    constrained_to: str = "ValueDomain",
    variadic: bool = True,
) -> dict:
    """
    A ValueDomain with one template parameter per entry of ``params``.

    ``constrained_to`` is what the parameters range over: `ValueDomain` for the containers, `Concept` for
    the selectors, whose arguments must be DomainConcepts. ``variadic=False`` and ``params=()`` are the
    two controls -- a non-variadic parameter and no `templateContext` at all -- which is how a test can
    show that what it pinned is a property of the *group* rather than of templating in general.
    """
    if not params:
        return {name: {"directParents": ["ValueDomain"], "data": {"instantiation": instantiation}}}
    template: dict = {"order": [f"{p}..." if variadic else p for p in params]}
    for p in params:
        template[p] = constrained_to
    if variadic and len(params) > 1:
        template["variadicGroupIdentifiers"] = {p: "" if i == 0 else "!" * i for i, p in enumerate(params)}
    return {
        name: {"directParents": ["ValueDomain"], "data": {"templateContext": template, "instantiation": instantiation}}
    }


def selector_vd(
    name: str, selector: str, params: tuple[str, ...] = ("T",), variadic: bool = True, value: object = "x"
) -> dict:
    """
    A ValueDomain whose whole instantiation is one closed object constrained by ``selector``.

    Closed -- ``additionalProperties: false`` -- because with it open every "this key is not selected"
    assertion is vacuous: an unclaimed key would be accepted anyway.

    ``value`` is the schema each selected key is checked against, and carries the provenance: bare ``"x"``
    is ``Any``, and ``["x", "Addr"]`` is the addressable form. Both are available to `props` and to
    `funcs` alike -- a DomainConcept function is **not** required to be addressable
    (`examples/animal_kingdom.json` writes ``["funcs(AcceptConcepts...)", "x", true]``, which is ``Any``).
    """
    return variadic_vd(
        name,
        {"type": "object", "additionalProperties": False, "properties": [[selector, value]]},
        params=params,
        constrained_to="Concept",
        variadic=variadic,
    )


VARIANT = variadic_vd("Variant", {"anyOf": ["T..."]})
"""Accepts a value of any one of its arguments. The `anyOf` row of the table."""

TUPLE = variadic_vd("Tuple", {"type": "array", "items": ["T..."], "additionalItems": False, "requireAllItems": True})
"""
Positional, and bounded at both ends by the two complementary keywords.

`additionalItems: false` forbids anything past the tuple; `requireAllItems: true` requires every position
of it to be present. The second is the only way to bound an *expanded* tuple below: its length is the
arity of the group, which the author cannot write at the declaration.
"""

INTERSECTION = variadic_vd("Intersection", {"allOf": ["T..."]})
"""The `allOf` row, whose empty identity is the opposite of `Variant`'s."""

ZIP = variadic_vd("Zip", {"type": "array", "items": ["T1...", "T2..."]}, params=("T1", "T2"))
"""Two groups in one container: the case that fixes concatenation over cross product."""

PET = {"Pet": {"directParents": ["Concept"], "data": {"properties": {"age": {"valueDomain": "Integer"}}}}}
"""A DomainConcept with one property, for the `props(...)` selector."""

CAR = {"Car": {"directParents": ["Concept"], "data": {"properties": {"speed": {"valueDomain": "Integer"}}}}}
"""A second, *unrelated* DomainConcept: unrelated so that "this key is not selected" is observable at all."""

VOCABULARY = {**PET, **CAR}

SELECTOR = {**VOCABULARY, **selector_vd("Sel", "props(T...)")}
"""The `props(...)` row of the table: the one expandable container whose elements are bare types."""


def box(site_type: str) -> dict:
    """A ValueDomain with one required property of ``site_type``, so a value can be put at that site."""
    return {
        "Box": {
            "directParents": ["ValueDomain"],
            "data": {
                "instantiation": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["v"],
                    "properties": {"v": {"type": site_type}},
                }
            },
        }
    }


# --------------------------------------------------------------------------------------------------
# Accessors
# --------------------------------------------------------------------------------------------------


def declare(concepts: dict) -> None:
    """Check a hierarchy with no instances: validity of the *schema*, with no value in play."""
    with contextlib.redirect_stdout(io.StringIO()):
        check_hierarchy(build_hierarchy(concepts))


def declaration_rejected(concepts: dict) -> str:
    with pytest.raises(ConceptHierarchyError) as excinfo:
        declare(concepts)
    return error_messages(excinfo.value)


def accepts(concepts: dict, site_type: str, value: object) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        check_hierarchy(build_hierarchy({**concepts, **box(site_type)}, instances={"p": {"Box": {"v": value}}}))


def refuses(concepts: dict, site_type: str, value: object) -> str:
    """
    The *value* is rejected while the schema itself is valid.

    The two halves are separate on purpose. A rejection test that only asserts "something raised" passes
    for free while the feature is missing, because the declaration is what raises -- which is how the
    first draft of this suite reported fifteen false XPASSes. Declaring first turns that into a failure.
    """
    declare({**concepts, **box(site_type)})
    with pytest.raises(ConceptHierarchyError) as excinfo:
        accepts(concepts, site_type, value)
    return error_messages(excinfo.value)


def resolved_schema(concepts: dict, site_type: str, value: object = None) -> CHSchemaNode:
    """
    The schema ``site_type`` was actually checked against, as stage B rebuilt it.

    Behavioral assertions are the primary ones here, but some of what stage B must get right is only
    *shape*: a replica's location, how many branches a nested container produced, where a ``$ref`` points.
    Those are invisible from acceptance alone -- two different wrong expansions can accept the same value
    -- so the resolved tree is read back out of the same memo the checker used.

    The check is allowed to fail: a schema is resolved before a value is matched against it, so a refused
    value still leaves its expansion behind, and a test about the shape need not find a value that passes.

    ``site_type`` is matched on its **clean name** rather than in full, because the memo is keyed by the
    canonical form and an application need not be written in it -- ``Inner<Integer>`` is remembered as
    ``Inner<[Integer]>``. One application per hierarchy is what makes that unambiguous, and the assertion
    below is what keeps it so.
    """
    clean_name = site_type.split("<", 1)[0]
    model = ConceptHierarchyDefinition.create_from_data(
        build_hierarchy({**concepts, **box(site_type)}, instances={"p": {"Box": {"v": value}}})
    )
    checker = ConceptHierarchyChecker(model, lambda _concept, _instance: None)
    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.suppress(ConceptHierarchyError):
            checker.check()
    validator = checker.context.expression_parser_validator if checker.context is not None else None
    assert validator is not None, f"{site_type} never got as far as resolving a schema; the declaration is rejected"
    schemas = validator.resolved_instantiation_schemas
    matching = [node for (full_name, _index), node in schemas.items() if full_name.split("<", 1)[0] == clean_name]
    assert len(matching) == 1, f"{site_type} resolved {len(matching)} times; have {sorted(k for k, _ in schemas)}"
    return matching[0]


def custom_type_names(nodes: list[CHSchemaNode]) -> list[str]:
    """The type each of ``nodes`` was expanded to, for comparing a container against a declared order."""
    return [node.custom_type.full_name if node.is_custom_type else repr(node.canonical) for node in nodes]


# ==================================================================================================
# 1. Where the expansion operator may be written
# ==================================================================================================


class TestWhatEnclosesTheOperator:
    """
    Section 2.1: validity is one question -- does an **expandable container** enclose the `T...`?

    There is no table of allowed positions. `T...` at `type`, at `contains`, under `properties` or nested
    arbitrarily deep is fine as long as a container is somewhere above it; the same schema written without
    one is invalid. Every pair below is the same inner schema, once wrapped and once not, so what is
    being tested is the enclosure and not the position.

    Parse-time only: no value is instantiated, so these say nothing about what the expansion produces.
    """

    INNER = [
        pytest.param({"type": "T..."}, id="type"),
        pytest.param({"contains": "T..."}, id="contains"),
        pytest.param({"not": "T..."}, id="not"),
        pytest.param({"type": "array", "items": "T..."}, id="items-schema-form"),
        pytest.param({"type": "array", "additionalItems": "T..."}, id="additionalItems"),
        pytest.param({"type": "object", "additionalProperties": "T..."}, id="additionalProperties"),
        pytest.param({"type": "object", "propertyNames": "T..."}, id="propertyNames"),
        pytest.param({"type": "object", "properties": {"k": "T..."}}, id="a-properties-value"),
        pytest.param(
            {"type": "object", "properties": {"k": {"type": "object", "properties": {"j": {"type": "T..."}}}}},
            id="deeply-nested",
        ),
        pytest.param("T...", id="a-bare-type"),
    ]

    @pytest.mark.parametrize("inner", INNER)
    @pytest.mark.parametrize("container", ["anyOf", "oneOf", "allOf"])
    def test_a_container_above_it_makes_it_valid(self, container, inner):
        declare(variadic_vd("V", {container: [inner]}))

    @pytest.mark.parametrize("inner", INNER)
    def test_an_items_tuple_above_it_makes_it_valid(self, inner):
        declare(variadic_vd("V", {"type": "array", "items": [inner]}))

    @pytest.mark.parametrize("inner", INNER)
    def test_the_same_schema_without_a_container_is_invalid(self, inner):
        """The other half of each pair, and what "fail closed" means: today these already fail."""
        assert declaration_rejected(variadic_vd("V", inner))

    def test_the_rejection_says_no_container_encloses_it(self):
        messages = declaration_rejected(variadic_vd("V", {"type": "T..."}))
        assert "anyOf" in messages and "items" in messages, messages

    def test_the_documented_example_is_valid(self):
        """Section 2.1's own example: the operator is two levels below its container."""
        declare(variadic_vd("X", {"type": "array", "items": [{"type": "object", "properties": {"entry": "T..."}}]}))

    @pytest.mark.parametrize("selector", ["props", "props+", "funcs", "funcs+"])
    def test_a_selector_argument_list_is_a_container(self, selector):
        declare({**PET, **variadic_vd("V", {"type": "object", "properties": [[f"{selector}(T...)", "x"]]})})

    @pytest.mark.parametrize("scope", ["$defs", "definitions"])
    def test_a_ref_target_is_not_a_replication_site(self, scope):
        """
        Section 2.4. Enclosure is lexical, so walking outward from this `T...` reaches the root without
        meeting a container -- and that is the wanted answer, not an oversight: a shared target cannot be
        replicated for one referrer and not another.
        """
        schema = {
            scope: {"e": {"type": "object", "properties": {"entry": "T..."}}},
            "type": "array",
            "items": [{"$ref": f"#/{scope}/e"}],
        }
        assert declaration_rejected(variadic_vd("V", schema))

    def test_the_ref_rejection_explains_itself(self):
        schema = {
            "$defs": {"e": {"type": "object", "properties": {"entry": "T..."}}},
            "type": "array",
            "items": [{"$ref": "#/$defs/e"}],
        }
        assert "replication site" in declaration_rejected(variadic_vd("V", schema))


class TestOneGroupPerReplicationUnit:
    """
    Section 2.3. Two occurrences of the *same* group in one unit are zipped -- copy *i* substitutes
    argument *i* everywhere in it. Two *different* groups in one unit have no defensible answer, because
    zipping across groups of unequal length is undefined, so they are refused; written as separate
    elements they need no zipping and are fine.
    """

    def test_the_same_group_twice_in_one_unit_is_allowed(self):
        declare(variadic_vd("V", {"type": "array", "items": [{"properties": {"a": "T...", "b": "T..."}}]}))

    def test_two_groups_in_one_unit_are_refused(self):
        """
        The message is asserted, not merely the rejection: while enclosed operators are refused outright
        this schema is rejected anyway, so "something raised" would pass for free and keep passing if the
        check were never written.
        """
        schema = {"type": "array", "items": [{"properties": {"a": "T1...", "b": "T2..."}}]}
        messages = declaration_rejected(variadic_vd("V", schema, params=("T1", "T2")))
        assert "T1" in messages and "T2" in messages, messages
        assert "different variadic groups" in messages, messages

    def test_two_groups_in_separate_elements_are_allowed(self):
        schema = {"type": "array", "items": [{"properties": {"a": "T1..."}}, {"properties": {"b": "T2..."}}]}
        declare(variadic_vd("V", schema, params=("T1", "T2")))

    def test_a_nested_container_separates_two_groups_again(self):
        """Units are assigned by *nearest* container, so the inner `anyOf` takes both."""
        schema = {"type": "array", "items": [{"anyOf": ["T1...", "T2..."]}]}
        declare(variadic_vd("V", schema, params=("T1", "T2")))


class TestAnUnexpandedVariadicIsRejected:
    """
    Section 2.6, second row -- the fail-closed half, and a behavior change on its own. A bare `T` naming
    a variadic parameter used to be accepted in a schema, which was the opposite of the rule: a variadic
    parameter names a *group*, so it cannot stand where one type is expected. Step 7.1, landed.
    """

    @pytest.mark.parametrize(
        "schema",
        [
            pytest.param({"anyOf": ["T"]}, id="anyOf"),
            pytest.param({"type": "T"}, id="type"),
            pytest.param({"type": "array", "items": ["T"]}, id="items"),
            pytest.param({"type": "object", "properties": {"k": {"type": "T"}}}, id="a-properties-value"),
        ],
    )
    def test_a_variadic_parameter_without_the_operator_is_rejected(self, schema):
        assert declaration_rejected(variadic_vd("V", schema))

    def test_a_non_variadic_parameter_is_unaffected(self):
        """The control: `T` is only wrong here because the parameter is variadic."""
        declare(
            {
                "V": {
                    "directParents": ["ValueDomain"],
                    "data": {
                        "templateContext": {"order": ["T"], "T": "ValueDomain"},
                        "instantiation": {"anyOf": ["T"]},
                    },
                }
            }
        )


class TestATypeApplicationShieldsItsVariadic:
    """
    Section 2.1: `{"anyOf": ["List<T...>"]}` is **one** branch. The variadic inside ``<...>`` belongs to
    the type application, which the existing machinery already handles, and this feature must not also
    expand it into the enclosing list.
    """

    LIST = {
        "List": {
            "directParents": ["ValueDomain"],
            "data": {"templateContext": {"order": ["E..."], "E": "ValueDomain"}, "instantiation": True},
        }
    }

    def test_a_shielded_variadic_is_accepted(self):
        """Passes today: the type parser already allows the operator inside ``<...>``."""
        declare({**self.LIST, **variadic_vd("V", {"anyOf": ["List<[T...]>"]})})

    def test_a_shielded_variadic_produces_one_branch(self):
        """
        With two arguments, expanding would give two branches of `List` rather than one `List` of two.
        The observable difference: the value is a `List`, not a member of it.
        """
        concepts = {**self.LIST, **variadic_vd("V", {"anyOf": ["List<[T...]>"]})}
        accepts(concepts, "V<[Integer, String]>", {"List<[Integer, String]>": []})


# ==================================================================================================
# 2. What the expansion produces
# ==================================================================================================


class TestExpansion:
    @pytest.mark.parametrize("value", [1, "s:x"])
    def test_a_variant_accepts_any_of_its_arguments(self, value):
        accepts(VARIANT, "Variant<[Integer, String]>", value)

    def test_a_variant_refuses_a_type_it_was_not_given(self):
        """The half that shows the schema still constrains: `Boolean` is not among the arguments."""
        assert refuses(VARIANT, "Variant<[Integer, String]>", True)

    def test_a_single_argument_variant_is_that_argument(self):
        accepts(VARIANT, "Variant<[Integer]>", 1)
        assert refuses(VARIANT, "Variant<[Integer]>", "s:x")

    def test_a_tuple_matches_its_arguments_positionally(self):
        accepts(TUPLE, "Tuple<[Integer, String]>", [1, "s:x"])

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param([1], id="too-short"),
            pytest.param([1, "s:x", 2], id="too-long"),
            pytest.param(["s:x", 1], id="wrong-order"),
        ],
    )
    def test_a_tuple_is_positional_and_bounded(self, value):
        """``additionalItems: false`` bounds it above, ``requireAllItems: true`` below."""
        assert refuses(TUPLE, "Tuple<[Integer, String]>", value)


class TestTwoGroupsConcatenateInOrder:
    """
    Section 3.2. `Zip<[Number, Integer, String], [Boolean]>` means ``items: [Number, Integer, String,
    Boolean]`` -- every argument of the first group before every argument of the second.

    Concatenation rather than cross product, because `...` splices a sequence into a sequence everywhere
    else in the language, and because a cross product is not even well formed for a positional `items`.
    """

    def test_the_groups_concatenate(self):
        accepts(ZIP, "Zip<[Integer, String], [Number]>", [1, "s:x", 2.5])

    def test_the_first_group_comes_first(self):
        """The assertion that distinguishes concatenation from any other ordering."""
        assert refuses(ZIP, "Zip<[Integer, String], [Number]>", [2.5, 1, "s:x"])

    def test_an_empty_group_contributes_nothing_rather_than_annihilating(self):
        """Under a cross product this would be the empty schema; under concatenation it is just `T1`."""
        accepts(ZIP, "Zip<[Integer, String], []>", [1, "s:x"])


# ==================================================================================================
# 3. The empty group
# ==================================================================================================


class TestAnEmptyGroupCollapses:
    """
    Section 4. The collapse is determined by the **keyword**, never by the ValueDomain: `anyOf` is a
    disjunction whose identity is false, `allOf` a conjunction whose identity is true.

    The expander may not simply write ``[]``: draft-07 requires ``minItems: 1`` on these keywords, and
    ``{"items": [], "additionalItems": false}`` fails to parse at all today.
    """

    @pytest.mark.parametrize("value", [1, "s:x", True, None, [], {}])
    def test_an_empty_variant_accepts_nothing(self, value):
        assert refuses(VARIANT, "Variant<[]>", value)

    @pytest.mark.parametrize("value", [1, "s:x", 2.5])
    def test_an_empty_intersection_accepts_everything(self, value):
        """The opposite identity, and intended: the empty conjunction constrains nothing."""
        accepts(INTERSECTION, "Intersection<[]>", value)

    def test_an_empty_tuple_accepts_only_the_empty_array(self):
        accepts(TUPLE, "Tuple<[]>", [])

    def test_an_empty_tuple_refuses_a_non_empty_array(self):
        """
        The trap this pins: draft-07 ignores `additionalItems` unless `items` is an array, so collapsing
        by dropping the keyword would silently make `Tuple<>` accept any array.
        """
        assert refuses(TUPLE, "Tuple<[]>", [1])


# ==================================================================================================
# 4. Not ground is not empty
# ==================================================================================================


class TestNotGroundIsNotEmpty:
    """
    Section 5, and the one failure here that would not be loud: a site whose arguments are not yet known
    must be left **unexpanded**, not collapsed. Collapsing it turns "not known yet" into `false`, which
    rejects every value and is a perfectly valid schema, so nothing downstream flags it.

    Both directions are asserted. Acceptance alone cannot distinguish a correctly deferred schema from
    one that collapsed to "accept everything".
    """

    OUTER = {
        "Outer": {
            "directParents": ["ValueDomain"],
            "data": {
                "templateContext": {"order": ["S"], "S": "ValueDomain"},
                "instantiation": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["v"],
                    "properties": {"v": {"type": "Variant<[S]>"}},
                },
            },
        }
    }

    def test_a_deferred_variant_still_accepts_a_valid_value(self):
        """If the non-ground site collapsed to `false`, this is rejected."""
        accepts({**VARIANT, **self.OUTER}, "Outer<Integer>", {"v": 1})

    def test_a_deferred_variant_still_refuses_an_invalid_value(self):
        """And if it collapsed to `true`, or was dropped, this is accepted."""
        assert refuses({**VARIANT, **self.OUTER}, "Outer<Integer>", {"v": "s:x"})

    def test_the_same_variant_is_ground_at_a_direct_site(self):
        """The control: the identical `Variant<[Integer]>` written where it *is* ground behaves normally."""
        accepts(VARIANT, "Variant<[Integer]>", 1)
        assert refuses(VARIANT, "Variant<[Integer]>", "s:x")


# ==================================================================================================
# 5. The resolved schema is per application
# ==================================================================================================


class TestTheResolvedSchemaIsPerApplication:
    """
    Section 6. `_check_instantiation_schema` keys the resolved schema on ``(expr_type.full_name, i)``,
    and `full_name` carries the variadic arguments -- so two applications of the same ValueDomain must
    not share an expansion. Written as a behavioral test rather than an inspection of the cache: if the
    entries were shared, whichever was built first would answer for both.
    """

    TWO = {
        "Two": {
            "directParents": ["ValueDomain"],
            "data": {
                "instantiation": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "narrow": {"type": "Variant<[Integer]>"},
                        "wide": {"type": "Variant<[Integer, String]>"},
                    },
                }
            },
        }
    }

    def test_two_applications_of_one_value_domain_do_not_share_an_expansion(self):
        concepts = {**VARIANT, **self.TWO}
        with contextlib.redirect_stdout(io.StringIO()):
            check_hierarchy(build_hierarchy(concepts, instances={"p": {"Two": {"narrow": 1, "wide": "s:x"}}}))

    def test_the_narrower_application_does_not_inherit_the_wider_one(self):
        concepts = {**VARIANT, **self.TWO}
        declare(concepts)
        with pytest.raises(ConceptHierarchyError):
            with contextlib.redirect_stdout(io.StringIO()):
                check_hierarchy(build_hierarchy(concepts, instances={"p": {"Two": {"narrow": "s:x", "wide": "s:x"}}}))


# ==================================================================================================
# 6. A nested container is its own unit when expanding, not only when parsing
# ==================================================================================================


class TestANestedContainerIsItsOwnUnitWhenExpanding:
    """
    Section 2.3's *nearest container* rule decides what stage B copies, not only what stage A accepts.

    `variadic_templates_of_this_unit` stops where the next container begins, so a `T...` below a nested
    `anyOf` belongs to **that** container. Replicating the outer element must therefore leave it alone and
    let it expand on its own -- once per argument, inside every copy. A replication that recurses past the
    boundary binds it to the argument of the copy it is in instead, silently turning N branches into one.

    Only the *same* group shows it: a different group is already left alone because the names differ, so
    the two-group cases below are the control rather than the test.
    """

    NESTED = variadic_vd(
        "V",
        {
            "type": "array",
            "additionalItems": False,
            "items": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["k", "j"],
                    "properties": {"k": "T...", "j": {"anyOf": ["T..."]}},
                }
            ],
        },
    )
    """
    One unit with a `T...` of its own (at ``k``) **and** a nested `anyOf` naming the same group (at ``j``).

    ``k`` belongs to the outer unit, so it is bound: copy *i* requires ``k`` to be argument *i*. The
    `anyOf` at ``j`` is a unit of its own, so it is **not** bound -- every copy admits every argument there.
    """

    def test_the_outer_element_still_replicates_once_per_argument(self):
        node = resolved_schema(self.NESTED, "V<[Integer, String]>", [])
        assert [custom_type_names([element.properties["k"]])[0] for element in node.items] == ["Integer", "String"]

    def test_the_nested_container_expands_inside_every_copy(self):
        """Shape, not acceptance: one branch where two belong is still a schema that checks values."""
        node = resolved_schema(self.NESTED, "V<[Integer, String]>", [])
        assert [custom_type_names(element.properties["j"].any_of) for element in node.items] == [
            ["Integer", "String"],
            ["Integer", "String"],
        ]

    def test_a_copy_admits_every_argument_at_the_nested_container(self):
        """The observable half: ``j`` is free to be either argument while ``k`` is pinned to this copy's."""
        accepts(self.NESTED, "V<[Integer, String]>", [{"k": 1, "j": "s:x"}, {"k": "s:y", "j": 1}])

    def test_the_outer_position_is_still_pinned(self):
        """And the control, so that expanding everything everywhere cannot pass this class either."""
        assert refuses(self.NESTED, "V<[Integer, String]>", [{"k": "s:x", "j": 1}, {"k": "s:y", "j": 1}])

    TWO_GROUPS = variadic_vd(
        "V",
        {"type": "array", "items": [{"properties": {"k": "T1...", "j": {"anyOf": ["T2..."]}}}]},
        params=("T1", "T2"),
    )
    """The same shape with two groups, which already worked: the names differ, so nothing binds by accident."""

    def test_a_nested_container_of_another_group_expands_independently(self):
        node = resolved_schema(self.TWO_GROUPS, "V<[Integer], [Number, String]>", [])
        assert len(node.items) == 1
        assert custom_type_names(node.items[0].properties["j"].any_of) == ["Number", "String"]

    def test_a_nested_items_tuple_is_a_unit_boundary_too(self):
        """`items` in its array form is a container, so the rule is not special to the branch keywords."""
        concepts = variadic_vd(
            "V", {"anyOf": [{"type": "object", "properties": {"k": "T...", "j": {"type": "array", "items": ["T..."]}}}]}
        )
        node = resolved_schema(concepts, "V<[Integer, String]>", {})
        assert [custom_type_names(branch.properties["j"].items) for branch in node.any_of] == [
            ["Integer", "String"],
            ["Integer", "String"],
        ]


# ==================================================================================================
# 7. Where a `$ref` points after the schema is rebuilt
# ==================================================================================================


class TestEveryRefPointsIntoTheRebuiltTree:
    """
    `ref_resolved` is a pointer *across* the tree rather than a child, so a shallow copy carries the
    address of the declaration over and `rewire_resolved_refs` has to map it. Stage B replicates, so the
    two trees no longer line up positionally and the mapping has to be built by the transform itself --
    which is exactly where a node can go missing.

    A reference left pointing at the declaration reaches a subtree that was never substituted, so its
    template variables are still variables: the failure is an internal `AssertionError`, not a refusal.
    """

    SELF_REF = {
        "V": {
            "directParents": ["ValueDomain"],
            "data": {
                "templateContext": {"order": ["T..."], "T": "ValueDomain"},
                "instantiation": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["k"],
                    "properties": {"nested": {"$ref": "#"}, "k": {"anyOf": ["T..."]}},
                },
            },
        }
    }
    """``"$ref": "#"`` is the schema root -- an explicitly supported pointer, and the one node that no walk
    over the *children* of the tree ever reaches."""

    def test_the_root_is_rewired_to_the_rebuilt_root(self):
        node = resolved_schema(self.SELF_REF, "V<[Integer]>", {"k": 1})
        assert node.properties["nested"].ref_resolved is node

    def test_a_value_reached_through_the_root_reference_still_checks(self):
        accepts(self.SELF_REF, "V<[Integer]>", {"k": 1, "nested": {"k": 2}})

    REF_IN_A_UNIT = variadic_vd(
        "V",
        {
            "$defs": {"inner": {"type": "object", "additionalProperties": False, "properties": {"n": "Integer"}}},
            "type": "array",
            "additionalItems": False,
            "items": [{"type": "object", "properties": {"k": "T...", "shared": {"$ref": "#/$defs/inner"}}}],
        },
    )
    """
    2.4: a `$ref` **inside** a unit is fine -- every copy shares the one target, which is correct. But every
    copy must reach the *rebuilt* target, and the copies are the nodes a positional pairing cannot line up.
    """

    def test_every_copy_reaches_the_rebuilt_target(self):
        node = resolved_schema(self.REF_IN_A_UNIT, "V<[Integer, String]>", [])
        assert len(node.items) == 2
        for element in node.items:
            assert element.properties["shared"].ref_resolved is node.definitions["inner"]


# ==================================================================================================
# 8. The collapse composes with everything else the substitution does
# ==================================================================================================


class TestCollapseComposesWithLiteralKeywords:
    """
    A node can carry both an expandable container and a keyword written as a literal template variable.
    Collapsing it to the boolean `false` leaves nowhere to write that keyword, so the collapse has to be
    the **last** thing the substitution does -- and a node that collapsed has no keyword worth writing.

    Ordered the other way round this is an internal `AssertionError` ("has no keyword dict to write to"),
    which `refuses` does not catch: it is not a `ConceptHierarchyError`, so the test fails rather than
    passing for the wrong reason.
    """

    BOTH = {
        "V": {
            "directParents": ["ValueDomain"],
            "data": {
                "templateContext": {"order": ["T...", "N"], "T": "ValueDomain", "N": "Literal:integer"},
                "instantiation": {"anyOf": ["T..."], "minItems": "N"},
            },
        }
    }

    @pytest.mark.parametrize("value", [1, "s:x", []])
    def test_an_empty_group_still_collapses_to_false(self, value):
        assert refuses(self.BOTH, "V<[], 2>", value)

    def test_a_non_empty_group_keeps_both_the_branch_and_the_keyword(self):
        """The control: with something left in the container, the keyword is substituted as ever."""
        accepts(self.BOTH, "V<[Integer], 2>", 1)


# ==================================================================================================
# 9. The rejection has to name the cause it actually had
# ==================================================================================================


class TestTheRejectionNamesTheRealCause:
    """
    2.6's message is for an author who wrote a *variadic parameter* where no container could take it. Any
    other name ending in ``...`` failed for some other reason, and saying "needs an enclosing anyOf" about
    it sends the reader looking for a container that would not have helped.
    """

    def test_a_variadic_parameter_gets_the_schema_message(self):
        assert "needs an enclosing" in declaration_rejected(variadic_vd("V", {"type": "T..."}))

    def test_the_reason_of_the_type_parser_is_kept_as_a_cause(self):
        """Replacing the message must not discard the original: it is the one that says *what* failed."""
        assert "template expansion operator" in declaration_rejected(variadic_vd("V", {"type": "T..."}))

    @pytest.mark.parametrize(
        "written",
        [
            pytest.param("Nope...", id="an-unknown-name"),
            pytest.param("Integer...", id="a-concept-that-is-not-a-parameter"),
            pytest.param("...", id="not-a-name-at-all"),
            pytest.param("T....", id="one-dot-too-many"),
        ],
    )
    def test_an_unrelated_failure_keeps_its_own_message(self, written):
        messages = declaration_rejected(variadic_vd("V", {"type": written}))
        assert "needs an enclosing" not in messages, messages


# ==================================================================================================
# 10. The selector lists
# ==================================================================================================


class TestASelectorArgumentListExpands:
    """
    Section 3.4. `props(T...)` is the fourth kind of expandable container, and the only one whose elements
    are bare types rather than subschemas -- so where it differs from an `items` tuple is worth pinning,
    and where it does *not* differ is worth pinning twice.

    Declaring one is not evidence that it expands: an unexpanded selector collects no concept at all, and
    a schema that selects nothing is perfectly valid. Every case here asserts the complement as well.
    """

    def test_a_selected_property_is_admitted(self):
        accepts(SELECTOR, "Sel<[Pet]>", {"age": 3})

    def test_a_property_of_another_concept_is_not(self):
        """The half that fails if the group was never spliced: with nothing selected, nothing is admitted."""
        assert refuses(SELECTOR, "Sel<[Pet]>", {"speed": 5})

    def test_every_argument_of_the_group_contributes(self):
        accepts(SELECTOR, "Sel<[Pet, Car]>", {"age": 3, "speed": 5})

    def test_a_selected_value_is_still_checked_against_its_own_type(self):
        """Splicing the right concepts in is not enough if the value under them stops being checked."""
        assert refuses(SELECTOR, "Sel<[Pet]>", {"age": "s:old"})

    def test_an_empty_group_permits_no_property(self):
        """
        Section 4: the identity of an empty selector is the opposite of a bare `props`. `props` with no
        parentheses means unconstrained, so the parenthesised empty form is free to mean "nothing", and
        must -- an empty group that fell back to "unconstrained" would fail open.
        """
        accepts(SELECTOR, "Sel<[]>", {})
        assert refuses(SELECTOR, "Sel<[]>", {"age": 3})

    def test_two_groups_in_one_list_concatenate(self):
        """No unit machinery needed: each element of the list is its own unit, as in an `items` tuple."""
        concepts = {**VOCABULARY, **selector_vd("Sel", "props(T1..., T2...)", params=("T1", "T2"))}
        accepts(concepts, "Sel<[Pet], [Car]>", {"age": 3, "speed": 5})

    def test_a_plain_concept_beside_a_group_is_kept(self):
        concepts = {**VOCABULARY, **selector_vd("Sel", "props(Car, T...)")}
        accepts(concepts, "Sel<[Pet]>", {"age": 3, "speed": 5})


class TestASelectorSubstitutesNonVariadicVariablesToo:
    """
    `props(C)` names one concept through a **non-variadic** template variable, and the substitution has to
    reach it for the same reason it reaches `props(C...)`: until it does, the argument list holds a
    variable, `collect_data` is handed something that is not an `InstantiatedType`, and the selector
    quietly collects nothing.

    It fails *closed*, which is why it needs a test rather than a review: a selector that selects nothing
    looks exactly like a selector whose concept has no properties.
    """

    ONE = {**VOCABULARY, **selector_vd("Sel", "props(C)", params=("C",), variadic=False)}

    def test_the_bound_concept_is_selected(self):
        accepts(self.ONE, "Sel<Pet>", {"age": 3})

    def test_a_property_of_another_concept_is_still_refused(self):
        assert refuses(self.ONE, "Sel<Pet>", {"speed": 5})

    def test_a_different_argument_selects_differently(self):
        """The control: the same declaration, and what it admits follows the argument."""
        accepts(self.ONE, "Sel<Car>", {"speed": 5})
        assert refuses(self.ONE, "Sel<Car>", {"age": 3})


class TestAnUnrestrictedSelectorSurvivesSubstitution:
    """
    A bare `props` -- no parentheses -- has **no** argument list at all: `concept_restriction` is ``None``,
    not an empty list, and the two mean opposite things (``None`` is unconstrained, ``[]`` selects
    nothing). Every pass over a selector therefore has to tell them apart before iterating.

    Nothing about this involves a variadic, but it only breaks once a `templateContext` is present, which
    is why the plain form below is the control: the substitution pass is skipped entirely without one.
    """

    def test_a_bare_selector_in_a_templated_schema_still_works(self):
        concepts = {**VOCABULARY, **selector_vd("Sel", "props", params=("C",), variadic=False)}
        accepts(concepts, "Sel<Pet>", {"age": 3})

    def test_a_bare_selector_in_a_plain_schema_still_works(self):
        concepts = {**VOCABULARY, **selector_vd("Sel", "props", params=())}
        accepts(concepts, "Sel", {"age": 3})

    def test_a_bare_selector_beside_an_expanding_one(self):
        """Both kinds in one schema: the unrestricted one must not be mistaken for an emptied group."""
        concepts = {**VOCABULARY, **selector_vd("Sel", "props")}
        accepts(concepts, "Sel<[Pet]>", {"age": 3, "speed": 5})


class TestASelectorInsideAReplicatedUnitIsZipped:
    """
    Section 2.3 says every `T...` of one unit is zipped -- copy *i* substitutes argument *i* **everywhere**
    in it. A selector written in a unit that replicates is in that unit, so it is bound like any other
    occurrence rather than splicing the whole group into each copy.

    Spliced instead of bound, `Sel<[Pet, Car]>` would give two copies that each select both concepts --
    N copies x N concepts -- and the per-copy distinction the author wrote would be gone.
    """

    ZIPPED = {
        **VOCABULARY,
        "Sel": {
            "directParents": ["ValueDomain"],
            "data": {
                "templateContext": {"order": ["T..."], "T": "Concept"},
                "instantiation": {
                    "type": "array",
                    "additionalItems": False,
                    "items": [{"type": "object", "additionalProperties": False, "properties": [["props(T...)", "x"]]}],
                },
            },
        },
    }

    def test_each_copy_selects_its_own_argument(self):
        accepts(self.ZIPPED, "Sel<[Pet, Car]>", [{"age": 3}, {"speed": 5}])

    def test_a_copy_does_not_select_the_other_copy_s_argument(self):
        """The assertion that separates zipping from splicing: position 0 is `Pet`, not `Pet` and `Car`."""
        assert refuses(self.ZIPPED, "Sel<[Pet, Car]>", [{"speed": 5}, {"age": 3}])

    def test_the_unit_replicated_once_per_argument(self):
        node = resolved_schema(self.ZIPPED, "Sel<[Pet, Car]>", [])
        assert len(node.items) == 2
        selected = [
            [x.full_name for x in element.custom_concept_data_constraints[0].concept_restriction]
            for element in node.items
        ]
        assert selected == [["Pet"], ["Car"]]


class TestTheOneGroupRuleSeesSelectorsToo:
    """
    2.3 refuses two different groups in one unit because zipping across groups of unequal length has no
    answer. That reason does not care whether an occurrence is a subschema or a selector argument, so a
    check that only walks the schema nodes lets the very case it exists for through -- and then stage B
    zips one group and splices the other.
    """

    def test_a_selector_group_beside_a_schema_group_is_refused(self):
        schema = {
            "type": "array",
            "items": [
                {
                    "type": "object",
                    "properties": [["props(T2...)", "x"]],
                    "propertyNames": "T1...",
                }
            ],
        }
        messages = declaration_rejected({**VOCABULARY, **variadic_vd("V", schema, params=("T1", "T2"))})
        assert "different variadic groups" in messages, messages

    def test_two_selector_groups_in_one_unit_are_refused(self):
        schema = {
            "type": "array",
            "items": [
                {
                    "type": "object",
                    "properties": [["props(T1...)", "x"], ["funcs(T2...)", "x"]],
                }
            ],
        }
        messages = declaration_rejected({**VOCABULARY, **variadic_vd("V", schema, params=("T1", "T2"))})
        assert "different variadic groups" in messages, messages

    def test_two_groups_in_one_selector_list_are_still_allowed(self):
        """Each element of the list is its own unit, so this one needs no zipping and stays legal."""
        declare({**VOCABULARY, **selector_vd("Sel", "props(T1..., T2...)", params=("T1", "T2"))})

    def test_the_same_group_twice_across_a_selector_and_a_schema_is_allowed(self):
        """One group, so it zips -- which is the whole point of allowing repeated occurrences."""
        schema = {
            "type": "array",
            "items": [{"type": "object", "properties": [["props(T...)", "x"]], "propertyNames": "T..."}],
        }
        declare({**VOCABULARY, **variadic_vd("V", schema, params=("T",))})


# ==================================================================================================
# 11. What a replica is called
# ==================================================================================================


class TestTheLocationOfAReplica:
    """
    Section 4.1. The copies are real elements of the rebuilt container and already have distinct indices
    there, so uniqueness is not what the discriminator buys -- traceability is: an error inside copy 2 can
    still be pointed back at the one declared subtree it was written as.

    It has to reach every *descendant* as well, not only the element: default sites are registered by
    location, so two copies sharing one would overwrite each other.
    """

    DEEP = variadic_vd(
        "V",
        {
            "type": "array",
            "items": [{"type": "object", "properties": {"entry": {"type": "object", "properties": {"deep": "T..."}}}}],
        },
    )
    """The operator two levels below the element, so that "every descendant" has descendants to speak of."""

    def test_the_index_and_the_argument_are_both_named(self):
        """The index is part of it because arguments repeat: ``Variant<[Integer, Integer]>`` is legal."""
        node = resolved_schema(self.DEEP, "V<[Integer, Integer]>", [])
        assert [element.location_id[-1] for element in node.items] == ["T...:0:Integer", "T...:1:Integer"]

    def test_the_discriminator_sits_where_the_declared_element_sat(self):
        node = resolved_schema(self.DEEP, "V<[String]>", [])
        assert list(node.items[0].location_id)[-3:] == ["items", 0, "T...:0:String"]

    def test_every_descendant_of_a_copy_is_rewritten_too(self):
        node = resolved_schema(self.DEEP, "V<[Integer, String]>", [])
        descendants = [list(child.location_id) for element in node.items for child in element.walk()]
        assert all("T...:0:Integer" in loc or "T...:1:String" in loc for loc in descendants), descendants

    def test_no_two_copies_share_a_location(self):
        node = resolved_schema(self.DEEP, "V<[Integer, Integer]>", [])
        locations = [tuple(child.location_id) for element in node.items for child in element.walk()]
        assert len(locations) == len(set(locations)), locations


# ==================================================================================================
# 12. The remaining corners
# ==================================================================================================


DOG = {"Dog": {"directParents": ["Pet"], "data": {"properties": {"breed": {"valueDomain": "String"}}}}}
"""Inherits `Pet`'s property and adds one, which is what separates `props` from `props+`."""

OWNER = {"Owner": {"directParents": ["Concept"], "data": {"functions": {"greet": {}}}}}
"""One *function* and no property, so `funcs` can be shown not to collect what `props` does."""


class TestEveryKindOfSelectorExpands:
    """
    2.1 lists four selector spellings, and all four reach `parse_custom_type` by the same path -- which is
    an argument that they behave alike, not evidence. `props+` and `funcs` each change what is collected,
    so a splice that worked for `props` could still be spliced into the wrong collection.
    """

    PROVENANCES = [
        pytest.param("x", ValueDomainArgumentProvenance.ANY, id="any-implicit"),
        pytest.param(["x", "Any"], ValueDomainArgumentProvenance.ANY, id="any-explicit"),
        pytest.param(["x", "Addr"], ValueDomainArgumentProvenance.ADDR, id="addr"),
    ]
    """
    A DomainConcept function may be written at **either** provenance -- it is not required to be
    addressable, and the shipped `InstanceData` uses the ``Any`` form. Every `funcs` case below is
    parametrized over all three spellings, so the expansion is pinned as provenance-independent rather
    than tested at whichever one happened to get written first.
    """

    @staticmethod
    def funcs(value: object) -> dict:
        return {**VOCABULARY, **OWNER, **selector_vd("Sel", "funcs(T...)", value=value)}

    @pytest.mark.parametrize("value,expected", PROVENANCES)
    def test_funcs_splices_the_group_into_its_argument_list(self, value, expected):
        """
        Asserted on the resolved schema rather than on a value: what a `funcs` *value* must look like is
        a separate question from what the expansion does to the argument list, and no test in this suite
        has established one. The argument list is what the expansion owes, and it is read back directly.
        """
        node = resolved_schema(self.funcs(value), "Sel<[Owner, Pet]>", {})
        constraint = node.custom_concept_data_constraints[0]
        assert [x.full_name for x in constraint.concept_restriction] == ["Owner", "Pet"]
        assert constraint.value.provenance is expected

    @pytest.mark.parametrize("value,expected", PROVENANCES)
    def test_funcs_with_an_empty_group_selects_nothing(self, value, expected):
        node = resolved_schema(self.funcs(value), "Sel<[]>", {})
        constraint = node.custom_concept_data_constraints[0]
        assert constraint.concept_restriction == []
        assert constraint.value.provenance is expected

    @pytest.mark.parametrize("value,expected", PROVENANCES)
    def test_the_selector_is_still_a_funcs_one_after_expanding(self, value, expected):
        """Splicing into the argument list must not disturb which collection it is a list *of*."""
        node = resolved_schema(self.funcs(value), "Sel<[Owner]>", {})
        constraint = node.custom_concept_data_constraints[0]
        assert constraint.for_properties_or_functions is ForPropertyOrFunction.FUNCTION
        assert constraint.value.provenance is expected

    @pytest.mark.parametrize("value,_expected", PROVENANCES)
    def test_props_expands_at_either_provenance_too(self, value, _expected):
        """The same freedom on the `props` side, so neither selector is pinned to one spelling."""
        concepts = {**VOCABULARY, **selector_vd("Sel", "props(T...)", value=value)}
        node = resolved_schema(concepts, "Sel<[Pet]>", {})
        assert [x.full_name for x in node.custom_concept_data_constraints[0].concept_restriction] == ["Pet"]

    def test_props_plus_expands_and_collects_the_inherited_set(self):
        concepts = {**VOCABULARY, **DOG, **selector_vd("Sel", "props+(T...)")}
        accepts(concepts, "Sel<[Dog]>", {"age": 3, "breed": "s:collie"})

    def test_plain_props_expands_and_does_not_inherit(self):
        """The control that makes the previous one mean something: the ``+`` is what added `age`."""
        concepts = {**VOCABULARY, **DOG, **selector_vd("Sel", "props(T...)")}
        accepts(concepts, "Sel<[Dog]>", {"breed": "s:collie"})
        assert refuses(concepts, "Sel<[Dog]>", {"age": 3})


class TestANonGroundSelectorIsLeftAsWritten:
    """
    Section 5 again, at the selector. A `props(T...)` whose group is not yet known must be left alone, for
    the same reason an unexpanded container is: an empty argument list selects *nothing*, so collapsing a
    deferred site into one rejects every value at it.
    """

    OUTER = {
        "Outer": {
            "directParents": ["ValueDomain"],
            "data": {
                "templateContext": {"order": ["C"], "C": "Concept"},
                "instantiation": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["v"],
                    "properties": {"v": {"type": "Sel<[C]>"}},
                },
            },
        }
    }

    def test_a_deferred_selector_still_admits_its_property(self):
        accepts({**SELECTOR, **self.OUTER}, "Outer<Pet>", {"v": {"age": 3}})

    def test_a_deferred_selector_still_refuses_an_unselected_key(self):
        """And the other direction, so "left alone" cannot have meant "left unconstrained"."""
        assert refuses({**SELECTOR, **self.OUTER}, "Outer<Pet>", {"v": {"speed": 5}})


class TestTheArgumentsGoInAsWritten:
    """
    3.2: duplicates are not removed and contradictions are not detected -- that is the schema author's
    business. What the expander owes is the declared order and the declared count, nothing else.
    """

    def test_a_repeated_argument_is_expanded_twice(self):
        node = resolved_schema(VARIANT, "Variant<[Integer, Integer]>", 1)
        assert custom_type_names(node.any_of) == ["Integer", "Integer"]

    def test_a_repeated_argument_still_checks(self):
        accepts(VARIANT, "Variant<[Integer, Integer]>", 1)

    def test_a_tuple_keeps_a_repeated_argument_positional(self):
        accepts(TUPLE, "Tuple<[Integer, Integer]>", [1, 2])
        assert refuses(TUPLE, "Tuple<[Integer, Integer]>", [1])

    def test_an_empty_first_group_does_not_shift_the_second(self):
        """The mirror of the empty-second-group case: concatenation has to be right at both ends."""
        accepts(ZIP, "Zip<[], [Integer, String]>", [1, "s:x"])
        assert refuses(ZIP, "Zip<[], [Integer, String]>", ["s:x", 1])

    def test_the_declared_order_survives_three_groups_worth_of_arguments(self):
        node = resolved_schema(ZIP, "Zip<[Number, Integer, String], [Integer]>", [])
        assert custom_type_names(node.items) == ["Number", "Integer", "String", "Integer"]


class TestTheRootOfTheSchemaIsSubstitutedToo:
    """
    The substitution starts *at* the root rather than at its children, so a schema that is nothing but a
    custom type is substituted like any other node. Written as a walk over the children it would be the
    one node no rewrite reached, and `"instantiation": "T"` would keep its variable.

    Nothing variadic about it -- it is the same entry point the expansion uses, and the case that shows it
    covers the root.
    """

    ROOT = variadic_vd("V", "T", variadic=False)

    def test_a_bare_custom_type_at_the_root_is_substituted(self):
        accepts(self.ROOT, "V<Integer>", 1)

    def test_and_still_constrains(self):
        assert refuses(self.ROOT, "V<Integer>", "s:x")

    def test_the_argument_is_what_decides(self):
        accepts(self.ROOT, "V<String>", "s:x")
        assert refuses(self.ROOT, "V<String>", 1)


# ==================================================================================================
# 13. Naming a whole group as a template argument
# ==================================================================================================


class TestAGroupCanBeForwardedByName:
    """
    ``Inner<T>``, where `Inner`'s parameter **and** ``T`` are both variadic, means *forward the whole
    group*: it is the same application ``Inner<[T...]>`` spells out. A bare variadic template variable
    already denotes a group, so "one member of a group" is not a reading it has -- forwarding is the only
    one that type-checks, and it is what an author writing the shorthand means.

    Inside an **explicit** group the operator stays required: ``Inner<[T]>`` writes brackets, so it is
    building a group, and putting a group where a member belongs means nothing. That asymmetry is the
    point -- no brackets forwards, brackets construct.

    `examples/animal_kingdom.json` is the case this comes from: `Instance` hands its own
    ``AcceptConcepts`` group to `InstanceData`.
    """

    INNER = {
        "Inner": {
            "directParents": ["ValueDomain"],
            "data": {
                "templateContext": {"order": ["E..."], "E": "ValueDomain"},
                "instantiation": {"anyOf": ["E..."]},
            },
        }
    }
    """A `Variant` by another name. No `variadicGroupIdentifiers`: with one group there is nothing to
    disambiguate, and the bare argument form must not depend on declaring one."""

    @staticmethod
    def outer(argument: str) -> dict:
        """`Outer` passes its own group on to `Inner`, written as ``argument``."""
        return {
            **TestAGroupCanBeForwardedByName.INNER,
            "Outer": {
                "directParents": ["ValueDomain"],
                "data": {
                    "templateContext": {"order": ["T..."], "T": "ValueDomain"},
                    "instantiation": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["v"],
                        "properties": {"v": {"type": f"Inner<{argument}>"}},
                    },
                },
            },
        }

    @pytest.mark.parametrize("argument", ["T", "[T...]"])
    def test_both_spellings_declare(self, argument):
        declare(self.outer(argument))

    @pytest.mark.parametrize("argument", ["T", "[T...]"])
    def test_the_forwarded_group_reaches_the_inner_expansion(self, argument):
        """Both arguments of the outer group are branches of the inner `anyOf`, whichever way it is written."""
        accepts(self.outer(argument), "Outer<[Integer, String]>", {"v": 1})
        accepts(self.outer(argument), "Outer<[Integer, String]>", {"v": "s:x"})

    @pytest.mark.parametrize("argument", ["T", "[T...]"])
    def test_the_forwarded_group_still_constrains(self, argument):
        """The half that fails if forwarding degraded into "accept anything"."""
        assert refuses(self.outer(argument), "Outer<[Integer, String]>", {"v": True})

    @pytest.mark.parametrize("argument", ["T", "[T...]"])
    def test_an_empty_forwarded_group_collapses(self, argument):
        """Forwarding an empty group is forwarding, not failing to forward: `Inner<>` rejects everything."""
        assert refuses(self.outer(argument), "Outer<[]>", {"v": 1})

    def test_an_unexpanded_variable_inside_an_explicit_group_is_still_refused(self):
        """Brackets construct a group, and a group is not a member of one. The operator says which is meant."""
        messages = declaration_rejected(self.outer("[T]"))
        assert "without the expansion operator" in messages, messages

    def test_a_non_variadic_variable_is_not_forwarded(self):
        """The control: only a *variadic* variable denotes a group, so only it can be handed over as one."""
        concepts = {
            **self.INNER,
            "Outer": {
                "directParents": ["ValueDomain"],
                "data": {
                    "templateContext": {"order": ["S"], "S": "ValueDomain"},
                    "instantiation": {"type": "object", "properties": {"v": {"type": "Inner<S>"}}},
                },
            },
        }
        accepts(concepts, "Outer<Integer>", {"v": 1})
        assert refuses(concepts, "Outer<Integer>", {"v": "s:x"})


# ==================================================================================================
# 14. `variadicGroupIdentifiers` is a disambiguator, not a prerequisite
# ==================================================================================================


class TestTheShorthandNeedsNoDeclaredIdentifiers:
    """
    Declaring `variadicGroupIdentifiers` is what lets an author write the **bare** argument form when
    several groups could claim an argument. It is not a prerequisite for having a variadic parameter, and
    nothing should need it to work: with exactly one variadic group there is nothing to disambiguate, so
    ``Inner<Integer>`` means ``Inner<[Integer]>`` whether or not an identifier was ever written.

    Two groups and no identifiers is the case that genuinely has no answer -- which group does a bare
    argument join? -- and stays refused. That is the disambiguation the declaration buys, and the only
    thing it buys.
    """

    @staticmethod
    def inner(order: list[str], identifiers: dict | None = None, instantiation: object = None) -> dict:
        context: dict = {"order": order}
        for name in order:
            context[name.removesuffix("...")] = "ValueDomain"
        if identifiers is not None:
            context["variadicGroupIdentifiers"] = identifiers
        return {
            "Inner": {
                "directParents": ["ValueDomain"],
                "data": {
                    "templateContext": context,
                    "instantiation": instantiation or {"anyOf": [n.removesuffix("...") + "..." for n in order]},
                },
            }
        }

    ONE = ["E..."]
    """One variadic parameter and nothing else: the shorthand is unambiguous by construction."""

    @pytest.mark.parametrize("identifiers", [None, {"E": ""}], ids=["undeclared", "declared"])
    @pytest.mark.parametrize(
        "application,value",
        [
            ("Inner<Integer>", 1),
            ("Inner<[Integer]>", 1),
            ("Inner<Integer, String>", "s:x"),
            ("Inner<[Integer, String]>", "s:x"),
        ],
    )
    def test_one_group_takes_either_form(self, identifiers, application, value):
        """Declaring the identifier may not be what decides whether a well-formed application parses."""
        accepts(self.inner(self.ONE, identifiers), application, value)

    @pytest.mark.parametrize("identifiers", [None, {"E": ""}], ids=["undeclared", "declared"])
    @pytest.mark.parametrize("application", ["Inner<Integer>", "Inner<[Integer]>"])
    def test_and_still_constrains_either_way(self, identifiers, application):
        assert refuses(self.inner(self.ONE, identifiers), application, "s:x")

    @pytest.mark.parametrize("identifiers", [None, {"E": ""}], ids=["undeclared", "declared"])
    def test_the_bare_form_means_the_bracketed_one(self, identifiers):
        """Not merely "both parse": they have to expand to the same branches, in the same order."""
        bare = resolved_schema(self.inner(self.ONE, identifiers), "Inner<Integer, String>", 1)
        bracketed = resolved_schema(self.inner(self.ONE, identifiers), "Inner<[Integer, String]>", 1)
        assert custom_type_names(bare.any_of) == custom_type_names(bracketed.any_of) == ["Integer", "String"]

    MIXED = ["S", "E..."]
    """A plain parameter before a variadic one: the bare arguments have to fill `S` first, then the group."""

    def test_a_plain_parameter_beside_a_variadic_one_needs_no_identifiers(self):
        concepts = self.inner(self.MIXED, instantiation={"anyOf": ["S", "E..."]})
        accepts(concepts, "Inner<Integer>", 1)
        assert refuses(concepts, "Inner<Integer>", "s:x")

    def test_the_group_beside_a_plain_parameter_still_takes_arguments(self):
        concepts = self.inner(self.MIXED, instantiation={"anyOf": ["S", "E..."]})
        accepts(concepts, "Inner<Integer, [String]>", "s:x")

    TWO = ["A...", "B..."]
    """Two groups: a bare argument could join either, which is what the identifiers exist to settle."""

    def test_two_groups_without_identifiers_still_take_the_bracketed_form(self):
        concepts = self.inner(self.TWO)
        accepts(concepts, "Inner<[Integer], [String]>", 1)
        assert refuses(concepts, "Inner<[Integer], [String]>", True)

    def test_a_bare_argument_across_two_groups_is_refused(self):
        """The one case a declaration is actually needed for, and the only one that may be refused."""
        assert declaration_rejected({**self.inner(self.TWO), **box("Inner<Integer, String>")})

    def test_the_refusal_names_both_ways_out(self):
        """
        It is the only thing that check still fires for, so it may as well say what to do: write the
        groups out, which needs no declaration, or declare the empty identifier for one of them.
        """
        messages = declaration_rejected({**self.inner(self.TWO), **box("Inner<Integer, String>")})
        assert "does not say which variadic group" in messages, messages
        assert "variadicGroupIdentifiers" in messages, messages
