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
``"properties": [[<selector>, <valueSchema>, <requireAllKeys>?], ...]`` -- a `properties` block whose keys
are **selected** from the hierarchy's DomainConcepts rather than written out.

```json
{"type": "object", "additionalProperties": true,
 "properties": [["props(Pet)", "x"], ["funcs(Owner)", ["x", "Addr"]]]}
```

`x` is the *declared type of the selected datum*, substituted per key, so one entry checks `age` against
`Integer` and `name` against `String`. `InstanceDataBase` and `InstanceData` in
``examples/animal_kingdom.json`` are the shipped users.

The plan is `documentation/TODO_CUSTOM_CONCEPT_DATA_CONSTRAINTS_TESTS.md`. Its own vocabulary is built
here rather than loading `animal_kingdom`, which costs ~75 ms per check *(measured)* against ~6 ms for a
hierarchy this size -- at ~70 tests that is the difference between one second and seven.

Two things are worth knowing before reading the assertions:

* **`additionalProperties` decides what a selector does *not* claim.** With ``true`` an unclaimed key is
  accepted anyway, so every "this key is not selected" assertion is vacuous unless the schema closes the
  object. The fixtures come in open and closed forms for exactly that reason;
* **a selected value's type is checked per key**, so a test that only asserts acceptance proves very
  little. Where it matters, the ill-typed case is asserted alongside.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from concept_hierarchy.data.expressions.instantiated_value import ParsedCustomValue, ParsedStructural
from concept_hierarchy.errors import ConceptHierarchyError
from tests.integration.test_expression_parsing import build_hierarchy, check_hierarchy, error_messages

# --------------------------------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------------------------------

PET = {
    "Pet": {
        "directParents": ["Concept"],
        "data": {"properties": {"age": {"valueDomain": "Integer"}, "name": {"valueDomain": "String"}}},
    }
}
"""Two properties of **different** types, so "checked against its own declared type" is observable."""

DOG = {"Dog": {"directParents": ["Pet"], "data": {"properties": {"breed": {"valueDomain": "String"}}}}}
"""Inherits `Pet`'s two and adds one, which is what separates `props` from `props+`."""

CAT = {"Cat": {"directParents": ["Pet"], "data": {"properties": {"lives": {"valueDomain": "Integer"}}}}}

EMPTY = {"Empty": {"directParents": ["Concept"], "data": {}}}

OWNER = {
    "Owner": {
        "directParents": ["Concept"],
        "data": {"properties": {"pets": {"valueDomain": "Integer"}}, "functions": {"greet": {}}},
    }
}
"""One property and one function, so `props` and `funcs` can be shown not to collect each other."""

VOCABULARY = {**PET, **DOG, **CAT, **EMPTY, **OWNER}


def data_vd(properties: object, additional: bool = True, name: str = "Data") -> dict:
    """A ValueDomain whose instantiation is one selector block."""
    return {
        name: {
            "directParents": ["ValueDomain"],
            "data": {"instantiation": {"type": "object", "additionalProperties": additional, "properties": properties}},
        }
    }


# --------------------------------------------------------------------------------------------------
# Accessors
# --------------------------------------------------------------------------------------------------


def declare(concepts: dict) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        check_hierarchy(build_hierarchy({**VOCABULARY, **concepts}))


def declaration_rejected(concepts: dict) -> str:
    with pytest.raises(ConceptHierarchyError) as excinfo:
        declare(concepts)
    return error_messages(excinfo.value)


BOX = {
    "Box": {
        "directParents": ["ValueDomain"],
        "data": {
            "instantiation": {
                "type": "object",
                "additionalProperties": False,
                "required": ["v"],
                "properties": {"v": {"type": "Data"}},
            }
        },
    }
}
"""
The value is put at a **typed** property rather than written as ``{"Data": ...}`` at a `ValueDomain` site.

That shorthand is ambiguous, and the ambiguity is not academic: an *open* `Data` accepts an object whose
single key happens to be named ``"Data"``, so ``{"Data": {"age": "s:x"}}`` parses as a `Data` with one
additional property -- and every assertion about the selected value is then vacuous. Measured: with that
harness an ill-typed value looked accepted under ``additionalProperties: true`` and refused under
``false``, which says nothing about the selector at all.
"""


def accepts(properties: object, value: dict, additional: bool = True) -> None:
    concepts = {**VOCABULARY, **data_vd(properties, additional), **BOX}
    with contextlib.redirect_stdout(io.StringIO()):
        check_hierarchy(build_hierarchy(concepts, instances={"p": {"Box": {"v": value}}}))


def refuses(properties: object, value: dict, additional: bool = True) -> str:
    """The value is rejected while the schema itself is valid -- both halves, so neither passes for free."""
    declare({**data_vd(properties, additional), **BOX})
    with pytest.raises(ConceptHierarchyError) as excinfo:
        accepts(properties, value, additional)
    return error_messages(excinfo.value)


# ==================================================================================================
# 1. Selector syntax, at declaration
# ==================================================================================================


class TestTheSelectorIsParsed:
    @pytest.mark.parametrize(
        "selector",
        ["props", "funcs", "props(Pet)", "props(Pet, Dog)", "props+(Pet)", "funcs(Owner)", "funcs+(Owner)"],
    )
    def test_a_well_formed_selector_is_accepted(self, selector):
        declare(data_vd([[selector, "x"]]))

    def test_a_bare_plus_needs_parentheses(self):
        """``+`` selects inherited data *of named concepts*, so it has nothing to qualify on its own."""
        assert declaration_rejected(data_vd([["props+", "x"]]))

    def test_a_concept_list_needs_a_space_after_the_comma(self):
        """
        The parser advances on ``", "``. Worth pinning either way -- ``props(Pet,Dog)`` is an easy thing
        to write, and if this should be accepted the test says where to change it.
        """
        assert declaration_rejected(data_vd([["props(Pet,Dog)", "x"]]))

    @pytest.mark.parametrize(
        "selector",
        [
            pytest.param("props()", id="empty-list"),
            pytest.param("props(pet)", id="lowercase"),
            pytest.param("props(Nope)", id="unknown-concept"),
            pytest.param("props(Integer)", id="a-value-domain-not-a-domain-concept"),
            pytest.param("Props(Pet)", id="wrong-case-keyword"),
            pytest.param("prop(Pet)", id="misspelled-keyword"),
            pytest.param("props(Pet", id="unclosed"),
        ],
    )
    def test_a_malformed_selector_is_rejected(self, selector):
        assert declaration_rejected(data_vd([[selector, "x"]]))

    def test_the_rejection_names_what_was_wrong(self):
        messages = declaration_rejected(data_vd([["props(Integer)", "x"]]))
        assert "Integer" in messages and "DomainConcept" in messages, messages


class TestTheValueSchemaAndTheRequireFlag:
    @pytest.mark.parametrize("value_schema", ["x", ["x", "Addr"], ["x", "Any"]])
    def test_a_well_formed_value_schema_is_accepted(self, value_schema):
        declare(data_vd([["props(Pet)", value_schema]]))

    @pytest.mark.parametrize("value_schema", [["x", "Nope"], "y", 3, None])
    def test_a_malformed_value_schema_is_rejected(self, value_schema):
        assert declaration_rejected(data_vd([["props(Pet)", value_schema]]))

    @pytest.mark.parametrize("flag", [True, False])
    def test_the_third_element_may_be_a_boolean(self, flag):
        declare(data_vd([["props(Pet)", "x", flag]]))

    @pytest.mark.parametrize("flag", [3, "s:yes", None])
    def test_a_non_boolean_third_element_is_rejected(self, flag):
        assert declaration_rejected(data_vd([["props(Pet)", "x", flag]]))

    def test_a_single_triple_and_a_list_of_triples_mean_the_same(self):
        """``properties`` may be one entry or several; one entry need not be wrapped."""
        accepts(["props(Pet)", "x"], {"age": 3})
        accepts([["props(Pet)", "x"]], {"age": 3})


# ==================================================================================================
# 2. Which data the selector collects
# ==================================================================================================


class TestWhatIsCollected:
    def test_an_unrestricted_selector_collects_every_concepts_properties(self):
        """``props`` with no parentheses is unconstrained: `Pet`'s, `Dog`'s and `Owner`'s alike."""
        accepts([["props", "x"]], {"age": 3, "breed": "s:collie", "pets": 2}, additional=False)

    def test_a_restricted_selector_collects_only_the_named_concepts(self):
        assert refuses([["props(Pet)", "x"]], {"breed": "s:collie"}, additional=False)

    def test_several_concepts_may_be_named(self):
        accepts([["props(Pet, Cat)", "x"]], {"age": 3, "lives": 9}, additional=False)

    def test_without_plus_only_the_concepts_own_data_is_collected(self):
        """`Dog` declares `breed`; `age` it merely inherits."""
        accepts([["props(Dog)", "x"]], {"breed": "s:collie"}, additional=False)
        assert refuses([["props(Dog)", "x"]], {"age": 3}, additional=False)

    def test_with_plus_inherited_data_is_collected_too(self):
        accepts([["props+(Dog)", "x"]], {"breed": "s:collie", "age": 3, "name": "s:Rex"}, additional=False)

    def test_a_concept_with_no_data_contributes_nothing(self):
        """Not an error -- it simply selects no keys, which `additionalProperties: false` then refuses."""
        declare(data_vd([["props(Empty)", "x"]]))
        assert refuses([["props(Empty)", "x"]], {"age": 3}, additional=False)

    def test_properties_and_functions_are_kept_apart(self):
        """Asserted both ways on `Owner`, which declares one of each."""
        accepts([["props(Owner)", "x"]], {"pets": 2}, additional=False)
        assert refuses([["props(Owner)", "x"]], {"greet": None}, additional=False)

    def test_a_funcs_selector_collects_the_function_not_the_property(self):
        assert refuses([["funcs(Owner)", ["x", "Addr"]]], {"pets": 2}, additional=False)


class TestANameCannotBeDeclaredByTwoConcepts:
    """
    The plan asked which type wins when two concepts declare the same name, since `collect_data` does
    ``res.update(...)`` per concept and the last would silently overwrite.

    It cannot arise: property names are **globally unique** across the hierarchy, enforced before any of
    this runs. The overwrite is unreachable, and this is the test that says so -- if the uniqueness rule
    is ever relaxed, it fails and the question becomes live again.
    """

    def test_two_concepts_may_not_declare_the_same_property_name(self):
        clash = {"Vehicle": {"directParents": ["Concept"], "data": {"properties": {"age": {"valueDomain": "String"}}}}}
        messages = declaration_rejected({**clash, **data_vd([["props(Pet)", "x"]])})
        assert "defined in multiple places" in messages, messages


# ==================================================================================================
# 3. Which keys a selector claims
# ==================================================================================================


class TestWhichKeysAreClaimed:
    """
    A selected key is *claimed*, so ``additionalProperties: false`` does not reject it. Everything below
    is stated against a closed object, because an open one accepts unclaimed keys anyway and would make
    each of these vacuous.
    """

    def test_a_selected_key_is_claimed(self):
        accepts([["props(Pet)", "x"]], {"age": 3}, additional=False)

    def test_a_key_that_is_not_a_concept_datum_is_not_claimed(self):
        assert refuses([["props(Pet)", "x"]], {"whatever": 1}, additional=False)

    def test_such_a_key_is_accepted_when_the_object_is_open(self):
        """The contrast that shows the previous test is about claiming, not about the selector failing."""
        accepts([["props(Pet)", "x"]], {"whatever": 1}, additional=True)

    def test_a_datum_of_another_concept_is_not_claimed(self):
        assert refuses([["props(Pet)", "x"]], {"lives": 9}, additional=False)

    def test_a_props_and_a_funcs_constraint_claim_the_union(self):
        """
        A `funcs` value must be *addressable* (`["x", "Addr"]`), so ``{}`` fails -- but on the provenance
        rule, which only the `funcs` constraint could have applied. Its presence is what says the key
        reached that constraint at all. (The trace also carries the whole cascade's other attempts, so a
        negative assertion about them would say nothing.)
        """
        messages = refuses([["props(Pet)", "x"], ["funcs(Owner)", ["x", "Addr"]]], {"age": 3, "greet": {}}, False)
        assert "Provenance violation" in messages, messages

    def test_two_selectors_of_the_same_kind_are_refused(self):
        """
        Currently at most one `props` and one `funcs` entry per schema -- an OR of two `props` selectors
        is not supported, and the parser says so rather than silently keeping one. Merge the concept lists
        instead: `props(Pet, Cat)`.
        """
        messages = declaration_rejected(data_vd([["props(Pet)", "x"], ["props(Cat)", "x"]]))
        assert "OR-constraint" in messages, messages


class TestTheFirstMatchingConstraintWins:
    """
    `_parse_custom_concept_data` stops at the first constraint that collects a key. That ordering is
    currently **unobservable**: at most one `props` and one `funcs` entry may be written, and those two
    never collect the same name -- a name is either a property or a function, never both.

    Kept as a statement of that, so the day OR-constraints are allowed there is a test here to extend.
    """

    def test_a_name_is_either_a_property_or_a_function(self):
        """Both entries claim their own kind; `greet` fails on provenance, never as an unexpected key."""
        messages = refuses([["props(Owner)", "x"], ["funcs(Owner)", ["x", "Addr"]]], {"pets": 2, "greet": {}}, False)
        assert "Provenance violation" in messages, messages

    def test_so_the_two_kinds_never_compete_for_one_key(self):
        assert refuses([["funcs(Owner)", ["x", "Addr"]]], {"pets": 2}, additional=False)
        assert refuses([["props(Owner)", "x"]], {"greet": {}}, additional=False)


# ==================================================================================================
# 4. The value check
# ==================================================================================================


class TestTheSelectedValueIsChecked:
    """
    The point of the feature: a selected key's value is checked against the type its concept declared.

    A selected key is claimed, so `additionalProperties` plays no part in it either way -- see
    `TestTheValueCheckDoesNotDependOnAdditionalProperties`.
    """

    def test_a_well_typed_value_is_accepted(self):
        accepts([["props(Pet)", "x"]], {"age": 3, "name": "s:Rex"}, additional=False)

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param({"age": "s:x"}, id="string-at-an-integer"),
            pytest.param({"age": {"nonsense": 1}}, id="object-at-an-integer"),
            pytest.param({"age": [1, 2]}, id="array-at-an-integer"),
            pytest.param({"age": None}, id="null-at-an-integer"),
            pytest.param({"name": 7}, id="integer-at-a-string"),
        ],
    )
    def test_an_ill_typed_value_is_refused(self, value):
        assert refuses([["props(Pet)", "x"]], value, additional=False)

    def test_the_refusal_names_the_key(self):
        assert "age" in refuses([["props(Pet)", "x"]], {"age": "s:x"}, additional=False)

    def test_each_key_is_checked_against_its_own_type(self):
        """
        One constraint, two properties of different types, only the second wrong. If `x` were bound once
        rather than per key, either both would pass or both would fail.
        """
        assert refuses([["props(Pet)", "x"]], {"age": 3, "name": 7}, additional=False)

    def test_the_mirror_case_is_accepted(self):
        accepts([["props(Pet)", "x"]], {"age": 3, "name": "s:Rex"}, additional=False)

    def test_a_key_from_an_unrestricted_selector_is_checked_too(self):
        assert refuses([["props", "x"]], {"breed": 7}, additional=False)

    def test_an_inherited_key_is_checked_against_the_declaring_concepts_type(self):
        """`props+(Dog)` reaches `Pet.age`; the type comes from `Pet`, not from `Dog`."""
        accepts([["props+(Dog)", "x"]], {"age": 3}, additional=False)
        assert refuses([["props+(Dog)", "x"]], {"age": "s:x"}, additional=False)


class TestTheValueCheckDoesNotDependOnAdditionalProperties:
    """
    Whether the object admits extra keys says nothing about a **selected** key: it was claimed, so
    `additionalProperties` never sees it.

    Worth pinning because the obvious harness suggests otherwise. Writing the value as
    ``{"Data": {...}}`` at a `ValueDomain` site makes an *open* `Data` accept it as an object with one
    unknown key named ``"Data"``, and the ill-typed value then looks accepted -- a property of that
    shorthand, not of the selector. `BOX` exists to avoid it.
    """

    @pytest.mark.parametrize("additional", [True, False])
    @pytest.mark.parametrize(
        "value",
        [
            pytest.param({"age": "s:x"}, id="string-at-an-integer"),
            pytest.param({"age": {"nonsense": 1}}, id="object-at-an-integer"),
            pytest.param({"age": [1, 2]}, id="array-at-an-integer"),
        ],
    )
    def test_an_ill_typed_selected_value_is_refused_either_way(self, additional, value):
        assert refuses([["props(Pet)", "x"]], value, additional=additional)

    @pytest.mark.parametrize("additional", [True, False])
    def test_a_well_typed_selected_value_is_accepted_either_way(self, additional):
        accepts([["props(Pet)", "x"]], {"age": 3}, additional=additional)


class TestTheParsedExpressionIsRetained:
    """
    The value is parsed into an `Expression`, not merely validated, and kept in
    ``ParsedStructural.custom_concept_data`` for later passes -- one entry per selected key, as
    ``(ParsedValue, constraint_index)``. Asserted on what was recorded, because acceptance alone does not
    say anything was built.
    """

    def _parsed_value(self, properties: object, value: dict) -> ParsedStructural:
        concepts = {**VOCABULARY, **data_vd(properties), **BOX}
        with contextlib.redirect_stdout(io.StringIO()):
            context = check_hierarchy(build_hierarchy(concepts, instances={"p": {"Box": {"v": value}}}))
        box = context.model.instances["p"].value.value.value
        assert isinstance(box, ParsedStructural), type(box).__name__
        leaf = box.properties_parsed["v"]
        assert isinstance(leaf, ParsedCustomValue), type(leaf).__name__
        parsed = leaf.expression.value.value
        assert isinstance(parsed, ParsedStructural), type(parsed).__name__
        return parsed

    def _concept_data(self, properties: object, value: dict) -> dict:
        return self._parsed_value(properties, value).custom_concept_data

    def test_a_selected_value_is_recorded(self):
        recorded = self._concept_data([["props(Pet)", "x"]], {"age": 3})
        assert "age" in recorded, recorded

    def test_the_recorded_entry_holds_a_parsed_expression(self):
        """The name of this class: *retained*, not merely checked and thrown away."""
        parsed, _index = self._concept_data([["props(Pet)", "x"]], {"age": 3})["age"]
        assert isinstance(parsed, ParsedCustomValue), type(parsed).__name__
        assert parsed.expression is not None

    def test_the_recording_carries_the_constraint_it_matched(self):
        """
        Always ``0`` in anything this suite writes, and still worth asserting: the index is the position
        in ``custom_concept_data_constraints``, so a second entry needs a second selector, and two of a
        kind are refused. The `funcs` one would be index 1, but no test here supplies a function value.
        """
        recorded = self._concept_data([["props(Pet)", "x"]], {"age": 3, "name": "s:Rex"})
        assert recorded["age"][1] == 0, recorded
        assert recorded["name"][1] == 0, recorded

    def test_an_unselected_key_is_not_recorded(self):
        assert "whatever" not in self._concept_data([["props(Pet)", "x"]], {"whatever": 1})

    def test_the_recorded_entries_are_ordinary_children_of_the_node(self):
        """
        `ParsedStructural.iter_children` yields them, so `walk` and `is_valid` reach them like any other
        child. While it did not, the per-key errors were *produced and stored and never read*: a value
        that failed its selector looked valid, which is the one way this can be wrong and stay quiet.
        """
        parsed = self._parsed_value([["props(Pet)", "x"]], {"age": 3})
        recorded = parsed.custom_concept_data["age"][0]
        assert any(node is recorded for node in parsed.walk()), "the selected value is not reached by walk"


# ==================================================================================================
# 5. requireAllKeys
# ==================================================================================================


class TestRequireAllKeys:
    """
    The third element of an entry. It was reported as unsatisfied even when every key was present -- the
    check read ``required_keys < local_matches``, a *strict* subset, so equality failed and the message
    named an empty list of missing keys. Fixed; these keep it fixed.
    """

    REQUIRED = [["props(Pet)", "x", True]]

    def test_every_key_present_is_accepted(self):
        accepts(self.REQUIRED, {"age": 3, "name": "s:Rex"})

    def test_a_missing_key_is_refused(self):
        assert "name" in refuses(self.REQUIRED, {"age": 3})

    def test_the_message_names_only_what_is_missing(self):
        messages = refuses(self.REQUIRED, {"age": 3})
        assert "missing keys are: ['name']" in messages, messages

    def test_an_empty_value_is_refused(self):
        messages = refuses(self.REQUIRED, {})
        assert "age" in messages and "name" in messages, messages

    def test_extra_unselected_keys_do_not_affect_it(self):
        accepts(self.REQUIRED, {"age": 3, "name": "s:Rex", "whatever": 1})

    def test_omitting_the_flag_requires_nothing(self):
        accepts([["props(Pet)", "x"]], {"age": 3})

    def test_writing_false_requires_nothing(self):
        accepts([["props(Pet)", "x", False]], {"age": 3})

    def test_only_the_marked_constraint_is_enforced(self):
        """`Owner`'s function is required, `Pet`'s properties are not: a value with neither names only it."""
        messages = refuses([["props(Pet)", "x"], ["funcs(Owner)", ["x", "Addr"], True]], {"age": 3})
        assert "greet" in messages, messages
        assert "missing keys are: ['age', 'name']" not in messages, messages

    def test_it_applies_to_the_inherited_set_when_plus_is_used(self):
        accepts([["props+(Dog)", "x", True]], {"breed": "s:collie", "age": 3, "name": "s:Rex"})
        assert "breed" in refuses([["props+(Dog)", "x", True]], {"age": 3, "name": "s:Rex"})


# ==================================================================================================
# 6. Template-dependent restrictions
# ==================================================================================================


HOLDER = {
    "Holder": {
        "directParents": ["ValueDomain"],
        "data": {
            "templateContext": {"order": ["T..."], "T": "Concept"},
            "instantiation": {
                "type": "object",
                "additionalProperties": False,
                "properties": [["props(T...)", "x"]],
            },
        },
    }
}


class TestARestrictionNamingTemplateVariables:
    """
    ``props(T...)`` is expanded per application (`TODO_VARIADIC_SCHEMA_EXPANSION.md` 7.6), so one
    declaration selects different keys at different applications.
    """

    def _box(self, site: str) -> dict:
        return {
            "Box": {
                "directParents": ["ValueDomain"],
                "data": {
                    "instantiation": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["v"],
                        "properties": {"v": {"type": site}},
                    }
                },
            }
        }

    def _accepts(self, site: str, value: dict) -> None:
        concepts = {**VOCABULARY, **HOLDER, **self._box(site)}
        with contextlib.redirect_stdout(io.StringIO()):
            check_hierarchy(build_hierarchy(concepts, instances={"p": {"Box": {"v": value}}}))

    def _refuses(self, site: str, value: dict) -> None:
        with pytest.raises(ConceptHierarchyError):
            self._accepts(site, value)

    def test_the_declaration_is_accepted(self):
        declare(HOLDER)

    def test_a_ground_group_selects_that_concepts_data(self):
        self._accepts("Holder<[Pet]>", {"age": 3, "name": "s:Rex"})

    def test_the_selection_follows_the_application(self):
        """One declaration, two applications, different keys claimed."""
        self._accepts("Holder<[Cat]>", {"lives": 9})
        self._refuses("Holder<[Cat]>", {"breed": "s:collie"})

    def test_two_concepts_in_the_group_select_both(self):
        self._accepts("Holder<[Pet, Cat]>", {"age": 3, "lives": 9})

    def test_an_empty_group_selects_nothing(self):
        self._accepts("Holder<[]>", {})
        self._refuses("Holder<[]>", {"age": 3})


class TestANonGroundRestriction:
    """
    Section 0.3 of the plan. The guard meant to defer a not-yet-ground restriction tests
    ``isinstance(x, InstantiatedType)`` over the *constraint objects*, which are never that, so it never
    fires. A non-ground restriction therefore reaches `collect_data`, which indexes
    ``model.domain_concepts`` by the unsubstituted name.
    """

    NESTED = {
        "Nested": {
            "directParents": ["ValueDomain"],
            "data": {
                "templateContext": {"order": ["S..."], "S": "Concept"},
                "instantiation": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["inner"],
                    "properties": {"inner": {"type": "Holder<[S...]>"}},
                },
            },
        }
    }

    def test_a_restriction_that_is_not_yet_ground_is_not_a_crash(self):
        """
        Passes: the group is substituted before `collect_data` sees it, so the dead guard is never needed
        on this path. It is the guard against a *regression* if that ordering ever changes.
        """
        concepts = {**VOCABULARY, **HOLDER, **self.NESTED}
        with contextlib.redirect_stdout(io.StringIO()):
            check_hierarchy(build_hierarchy(concepts, instances={"p": {"Nested<[Pet]>": {"inner": {"age": 3}}}}))
