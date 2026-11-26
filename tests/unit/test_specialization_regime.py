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
Tests for the DomainConcept property/function specialization regime
(``checker.check_specializations`` -> ``process_specialization_for_domain_concepts``).

Every property keyword has two independent slots at each concept: the *for-this* value used when
instantiating that concept, and the *for-subconcepts* value passed down. ``_specializations`` fills the
for-subconcepts slot, ``_specializations._forThis`` the for-this slot, and a keyword written in a
property's own definition fills the for-subconcepts slot at the definition site.

Each slot holds one of four states, and the JSON spellings collide dangerously::

    SET VALUE              "keyword": <value>
    NO VALUE               "keyword": "inheritFrom:"          -- cancels the inherited value
    GET VALUE FROM PARENT  "keyword": "inheritFrom:<Parent>"  -- picks which parent to inherit from
    IMPLICIT GET           keyword absent

``"inheritFrom:"`` and ``"inheritFrom:<Name>"`` differ by a name suffix and mean opposite things, so the
tests below pin both.
"""

from __future__ import annotations

import pytest

from concept_hierarchy.errors import CHSemanticError, CHSyntaxError, CHWarning
from tests.ch_support import check_concepts

# ``prop`` is defined once on Base, so only the *specializable* keywords can ever be ambiguous.
BASE = {"Base": {"directParents": ["Concept"], "data": {"properties": {"prop": "String"}}}}
BASE_WITH_DEFAULT = {
    "Base": {
        "directParents": ["Concept"],
        "data": {"properties": {"prop": {"valueDomain": "String", "default": "s:base"}}},
    }
}


def specializing(parents: list[str], for_subconcepts: dict | None = None, for_this: dict | None = None) -> dict:
    """A DomainConcept that specializes ``prop`` in the for-subconcepts and/or the for-this slot."""
    specializations: dict = {}
    if for_subconcepts is not None:
        specializations["prop"] = for_subconcepts
    if for_this is not None:
        specializations["_forThis"] = {"prop": for_this}
    return {"directParents": parents, "data": {"properties": {"_specializations": specializations}}}


def diamond(tip: dict | None = None) -> dict:
    """``Base`` defines ``prop``; ``Spec1`` and ``Spec2`` both set a default; ``Diamond`` inherits both."""
    return {
        **BASE,
        "Spec1": specializing(["Base"], {"default": "s:one"}),
        "Spec2": specializing(["Base"], {"default": "s:two"}),
        "Diamond": specializing(["Spec1", "Spec2"], tip)
        if tip is not None
        else {
            "directParents": ["Spec1", "Spec2"],
            "data": {},
        },
    }


# --------------------------------------------------------------------------------------------------
# The four value states
# --------------------------------------------------------------------------------------------------


class TestValueStates:
    def test_set_value_in_a_property_definition(self):
        check_concepts({"Animal": {"directParents": ["Concept"], "data": {"properties": {"legs": "Integer"}}}})

    def test_set_value_for_subconcepts(self):
        check_concepts({**BASE, "Sub": specializing(["Base"], {"default": "s:sub"})})

    def test_set_value_for_this_only(self):
        """``_forThis`` fills this concept's slot without changing what subconcepts inherit."""
        check_concepts({**BASE, "Sub": specializing(["Base"], for_this={"default": "s:mine"})})

    def test_both_slots_independently(self):
        check_concepts({**BASE, "Sub": specializing(["Base"], {"default": "s:for-subs"}, {"default": "s:for-this"})})

    def test_no_value_cancels_an_inherited_value(self):
        """``"inheritFrom:"`` with no name is NO VALUE -- it cancels, it does not inherit."""
        check_concepts(
            {
                **BASE,
                "Mid": specializing(["Base"], {"default": "s:mid"}),
                "Sub": specializing(["Mid"], {"default": "inheritFrom:"}),
            }
        )

    def test_implicit_get_with_a_single_parent(self):
        """An absent keyword is IMPLICIT GET, which is unambiguous when only one parent contributes."""
        check_concepts({**BASE, "Mid": specializing(["Base"], {"default": "s:mid"}), "Sub": specializing(["Mid"], {})})

    def test_inherit_from_parents_marks_a_value_as_only_for_subconcepts(self):
        """``"inheritFrom:parents"`` in ``_forThis`` is how the onlyForSubconcepts mode is expressed."""
        check_concepts(
            {
                **BASE,
                "Mid": specializing(["Base"], {"default": "s:mid"}),
                "Sub": specializing(["Mid"], {"default": "s:sub"}, {"default": "inheritFrom:parents"}),
            }
        )


# --------------------------------------------------------------------------------------------------
# Multi-parent ambiguity
# --------------------------------------------------------------------------------------------------


class TestMultiParentAmbiguity:
    def test_two_parents_with_a_value_need_disambiguation(self):
        with pytest.raises(CHSemanticError, match="A disambiguation from multiple inherited values is needed"):
            with pytest.warns(CHWarning, match="Found a domain concept with no data defined: 'Diamond'"):
                check_concepts(diamond())

    def test_setting_a_value_resolves_the_ambiguity(self):
        check_concepts(diamond({"default": "s:own"}))

    def test_naming_a_parent_resolves_the_ambiguity(self):
        check_concepts(diamond({"default": "inheritFrom:Spec1"}))

    def test_the_error_names_the_property_and_the_keyword(self):
        with pytest.raises(CHSemanticError) as raised:
            with pytest.warns(CHWarning, match="Found a domain concept with no data defined: 'Diamond'"):
                check_concepts(diamond())
        message = str(raised.value)
        assert '"prop"' in message
        assert '"default"' in message

    def test_only_one_parent_with_a_value_is_unambiguous(self):
        """No ambiguity when a single parent contributes -- the other leaves the slot empty."""
        with pytest.warns(CHWarning, match="Found a domain concept with no data defined: 'Diamond'"):
            check_concepts(
                {
                    **BASE,
                    "Spec1": specializing(["Base"], {"default": "s:one"}),
                    "Spec2": specializing(["Base"], {}),
                    "Diamond": {"directParents": ["Spec1", "Spec2"], "data": {}},
                }
            )


# --------------------------------------------------------------------------------------------------
# Rejected specializations
# --------------------------------------------------------------------------------------------------


class TestRejectedSpecializations:
    def test_specializing_an_unavailable_property(self):
        with pytest.raises(CHSemanticError, match="is not an available property"):
            check_concepts(
                {
                    **BASE,
                    "Sub": {
                        "directParents": ["Base"],
                        "data": {"properties": {"_specializations": {"missing": {"default": "s:x"}}}},
                    },
                }
            )

    def test_unknown_definition_keyword(self):
        with pytest.raises(CHSyntaxError, match="Unknown definition key"):
            check_concepts({**BASE, "Sub": specializing(["Base"], {"notAKeyword": 1})})

    def test_a_conjunctive_keyword_can_not_be_cancelled(self):
        """``constraint`` is conjunctive: subconcepts narrow it, so cancelling it is meaningless."""
        with pytest.raises(CHSemanticError, match="Conjunctive .* definition keywords can not be cancelled"):
            check_concepts({**BASE, "Sub": specializing(["Base"], {"constraint": "inheritFrom:"})})

    def test_inherit_from_a_concept_that_is_not_a_direct_parent(self):
        with pytest.raises(CHSemanticError, match="is not a direct parent of"):
            check_concepts({**BASE, "Sub": specializing(["Base"], {"default": "inheritFrom:Concept"})})

    def test_inherit_from_parents_outside_for_this(self):
        with pytest.raises(CHSemanticError, match='"inheritFrom:parents" specialization value is only available'):
            check_concepts({**BASE, "Sub": specializing(["Base"], {"default": "inheritFrom:parents"})})

    def test_a_property_defined_here_can_not_be_specialized_at_the_top_level(self):
        """A defined property's for-subconcepts slot is already filled at the definition site."""
        with pytest.raises(CHSemanticError, match="can not be specialized at the top-level"):
            check_concepts(
                {
                    "Animal": {
                        "directParents": ["Concept"],
                        "data": {
                            "properties": {
                                "legs": {"valueDomain": "Integer", "default": 4},
                                "_specializations": {"legs": {"default": 2}},
                            }
                        },
                    },
                }
            )

    def test_a_property_defined_here_may_still_be_specialized_for_this(self):
        """...but ``_forThis`` is exactly how a defined property gets a different value for this concept."""
        check_concepts(
            {
                "Animal": {
                    "directParents": ["Concept"],
                    "data": {
                        "properties": {
                            "legs": {"valueDomain": "Integer", "default": 4},
                            "_specializations": {"_forThis": {"legs": {"default": 2}}},
                        }
                    },
                },
            }
        )


# --------------------------------------------------------------------------------------------------
# Functions specialize too, with a smaller keyword set
# --------------------------------------------------------------------------------------------------

FUNCTION_BASE = {
    "FBase": {"directParents": ["Concept"], "data": {"functions": {"f": {"valueDomain": "CustomFunction"}}}}
}


def function_specializing(parents: list[str], for_subconcepts=None, for_this=None) -> dict:
    """A DomainConcept specializing the function ``f`` in the for-subconcepts and/or the for-this slot."""
    specializations: dict = {}
    if for_subconcepts is not None:
        specializations["f"] = for_subconcepts
    if for_this is not None:
        specializations["_forThis"] = {"f": for_this}
    return {"directParents": parents, "data": {"functions": {"_specializations": specializations}}}


def function_silent(parents: list[str]) -> dict:
    """A DomainConcept that neither defines nor specializes any function."""
    return {"directParents": parents, "data": {"functions": {}}}


def available_functions(context, concept_name: str) -> dict:
    """The *derived* view: everything available to the concept, inherited included."""
    return {
        name: dict(data) for name, data in context.model.domain_concepts[concept_name].available_function_data.items()
    }


def declared_functions(context, concept_name: str) -> dict:
    """The *definition* view: only what this concept's own JSON declares and specializes."""
    return {name: dict(data) for name, data in context.ch.concepts[concept_name].available_function_data.items()}


class TestFunctionSpecialization:
    def test_unknown_function_is_rejected(self):
        with pytest.raises(CHSemanticError, match="is not an available function"):
            check_concepts(
                {
                    **FUNCTION_BASE,
                    "Sub": {
                        "directParents": ["FBase"],
                        "data": {"functions": {"_specializations": {"missing": {"default": None}}}},
                    },
                }
            )

    def test_the_specialization_body_is_the_default_itself(self):
        """
        Functions specialize only ``default``, so there is no keyword layer: whatever body is written is
        taken *as* the default. That means no key can ever be "unknown" here -- unlike for properties.
        """
        context = check_concepts({**FUNCTION_BASE, "Sub": function_specializing(["FBase"], {"whatever": 1})})
        assert context.ch.concepts["Sub"].function_specializations_for_sub == {"f": {"default": {"whatever": 1}}}

    def test_writing_the_default_keyword_explicitly_nests_it(self):
        """A trap worth pinning: ``{"default": X}`` becomes the default ``{"default": X}``, not ``X``."""
        context = check_concepts({**FUNCTION_BASE, "Sub": function_specializing(["FBase"], {"default": "X"})})
        assert context.ch.concepts["Sub"].function_specializations_for_sub == {"f": {"default": {"default": "X"}}}

    def test_a_function_default_can_still_be_cancelled(self):
        """
        The cancellation is recorded in the available-data map, not by editing the definition: the concept
        definition keeps saying what its JSON said, and the keyword is simply not passed on.
        """
        context = check_concepts({**FUNCTION_BASE, "Sub": function_specializing(["FBase"], "inheritFrom:")})
        assert "default" not in available_functions(context, "Sub")["f"]
        assert context.ch.concepts["Sub"].function_specializations_for_sub == {"f": {"default": "inheritFrom:"}}

    def test_a_function_default_can_be_taken_from_a_named_parent(self):
        context = check_concepts({**FUNCTION_BASE, "Sub": function_specializing(["FBase"], "inheritFrom:FBase")})
        assert available_functions(context, "Sub")["f"]["default"] == "FBase"

    def test_the_for_this_slot_works_for_functions_too(self):
        context = check_concepts({**FUNCTION_BASE, "Sub": function_specializing(["FBase"], for_this={"x": 1})})
        assert context.ch.concepts["Sub"].function_specializations_for_this == {"f": {"default": ({"x": 1}, False)}}

    def test_a_function_may_be_declared_without_a_value_domain(self):
        """It then takes the default value domain of DomainConcept functions, ``CustomFunction``."""
        context = check_concepts({"FBase": {"directParents": ["Concept"], "data": {"functions": {"f": {}}}}})
        assert available_functions(context, "FBase") == {"f": {"valueDomain": "FBase"}}

    def test_inherit_from_parents_outside_for_this_is_rejected(self):
        with pytest.raises(CHSemanticError, match='"inheritFrom:parents" specialization value is only available'):
            check_concepts({**FUNCTION_BASE, "Sub": function_specializing(["FBase"], "inheritFrom:parents")})

    def test_inherit_from_a_concept_that_is_not_a_direct_parent_is_rejected(self):
        with pytest.raises(CHSemanticError, match="is not a direct parent of"):
            check_concepts({**FUNCTION_BASE, "Sub": function_specializing(["FBase"], "inheritFrom:Concept")})


class TestFunctionMultiParentAmbiguity:
    """Functions inherit the same ambiguity rules as properties."""

    def _diamond(self, tip: dict | None) -> dict:
        return {
            **FUNCTION_BASE,
            "Left": function_specializing(["FBase"], {"x": 1}),
            "Right": function_specializing(["FBase"], {"x": 2}),
            "Diamond": function_specializing(["Left", "Right"], tip)
            if tip is not None
            else {"directParents": ["Left", "Right"], "data": {}},
        }

    def test_two_parents_with_a_default_need_disambiguation(self):
        with pytest.raises(CHSemanticError, match="A disambiguation from multiple inherited values is needed"):
            with pytest.warns(CHWarning, match="Found a domain concept with no data defined: 'Diamond'"):
                check_concepts(self._diamond(None))

    def test_the_error_names_the_function(self):
        with pytest.raises(CHSemanticError) as raised:
            with pytest.warns(CHWarning, match="Found a domain concept with no data defined: 'Diamond'"):
                check_concepts(self._diamond(None))
        assert 'of function "f"' in str(raised.value)

    def test_setting_a_default_resolves_the_ambiguity(self):
        check_concepts(self._diamond({"x": 3}))

    def test_naming_a_parent_resolves_the_ambiguity(self):
        check_concepts(
            {
                **FUNCTION_BASE,
                "Left": function_specializing(["FBase"], {"x": 1}),
                "Right": function_specializing(["FBase"], {"x": 2}),
                "Diamond": function_specializing(["Left", "Right"], "inheritFrom:Left"),
            }
        )


class TestInheritedFunctionPropagation:
    """The mirror of :class:`TestInheritedPropertyPropagation` -- both share the same seeding step."""

    def test_a_direct_child_can_specialize_an_inherited_function(self):
        check_concepts({**FUNCTION_BASE, "Sub": function_specializing(["FBase"], {"x": 1})})

    def test_a_grandchild_can_specialize_through_a_silent_parent(self):
        """Regression: the same one-generation cut-off applied to functions."""
        check_concepts(
            {**FUNCTION_BASE, "Mid": function_silent(["FBase"]), "Sub": function_specializing(["Mid"], {"x": 1})}
        )

    def test_a_function_survives_several_silent_generations(self):
        check_concepts(
            {
                **FUNCTION_BASE,
                "G1": function_silent(["FBase"]),
                "G2": function_silent(["G1"]),
                "G3": function_silent(["G2"]),
                "Sub": function_specializing(["G3"], {"x": 1}),
            }
        )

    def test_the_recorded_origin_is_the_defining_ancestor(self):
        context = check_concepts(
            {**FUNCTION_BASE, "Mid": function_silent(["FBase"]), "Sub": function_specializing(["Mid"], {"x": 1})}
        )
        assert available_functions(context, "Mid") == {"f": {"valueDomain": "FBase"}}
        assert available_functions(context, "Sub") == {"f": {"valueDomain": "FBase", "default": "Sub"}}

    def test_inherited_data_does_not_leak_into_the_concept_definition(self):
        context = check_concepts(
            {**FUNCTION_BASE, "Mid": function_silent(["FBase"]), "Sub": function_specializing(["Mid"], {"x": 1})}
        )
        assert declared_functions(context, "Mid") == {}
        assert declared_functions(context, "Sub") == {"f": {"default": "Sub"}}

    def test_one_origin_reached_by_two_paths_is_not_ambiguous(self):
        check_concepts(
            {
                **FUNCTION_BASE,
                "Left": function_silent(["FBase"]),
                "Right": function_silent(["FBase"]),
                "Diamond": function_specializing(["Left", "Right"], {"x": 1}),
            }
        )


# --------------------------------------------------------------------------------------------------
# How far down the hierarchy an inherited property stays specializable
# --------------------------------------------------------------------------------------------------


def silent(parents: list[str]) -> dict:
    """A DomainConcept that neither defines nor specializes anything."""
    return {"directParents": parents, "data": {"properties": {}}}


def available(context, concept_name: str) -> dict:
    """The *derived* view: everything available to the concept, inherited included."""
    return {
        name: dict(data) for name, data in context.model.domain_concepts[concept_name].available_property_data.items()
    }


def declared(context, concept_name: str) -> dict:
    """The *definition* view: only what this concept's own JSON declares and specializes."""
    return {name: dict(data) for name, data in context.ch.concepts[concept_name].available_property_data.items()}


class TestInheritedPropertyPropagation:
    """
    A property must stay available to *every* descendant, not just to direct children. Each concept's
    available-property map is seeded from its parents' before its own specializations are applied, so a
    concept that says nothing about a property still passes it on.
    """

    def test_a_direct_child_can_specialize_an_inherited_property(self):
        check_concepts({**BASE, "Sub": specializing(["Base"], {"default": "s:sub"})})

    def test_a_grandchild_can_specialize_through_a_silent_parent(self):
        """Regression: a concept that did not mention the property used to hide it from everything below."""
        check_concepts({**BASE, "Mid": silent(["Base"]), "Sub": specializing(["Mid"], {"default": "s:sub"})})

    def test_a_property_survives_several_silent_generations(self):
        check_concepts(
            {
                **BASE,
                "G1": silent(["Base"]),
                "G2": silent(["G1"]),
                "G3": silent(["G2"]),
                "Sub": specializing(["G3"], {"default": "s:deep"}),
            }
        )

    def test_the_recorded_origin_is_the_defining_ancestor(self):
        """The map records where a value came from, not which parent it was reached through."""
        context = check_concepts({**BASE, "Mid": silent(["Base"]), "Sub": specializing(["Mid"], {"default": "s:sub"})})
        assert available(context, "Mid") == {"prop": {"valueDomain": "Base"}}
        assert available(context, "Sub") == {"prop": {"valueDomain": "Base", "default": "Sub"}}

    def test_inherited_data_does_not_leak_into_the_concept_definition(self):
        """
        The definition describes what its own JSON says; only the model carries the inherited view. A
        concept that declares nothing must therefore have an empty definition map, however much it inherits.
        """
        context = check_concepts({**BASE, "Mid": silent(["Base"]), "Sub": specializing(["Mid"], {"default": "s:sub"})})
        assert declared(context, "Base") == {"prop": {"valueDomain": "Base"}}
        assert declared(context, "Mid") == {}
        assert declared(context, "Sub") == {"prop": {"default": "Sub"}}

    def test_a_named_parent_is_recorded_even_when_it_only_inherited_the_value(self):
        """
        ``inheritFrom:<Parent>`` names a *direct parent*, and that is what gets recorded -- deliberately.
        Here ``Mid`` only inherited the default from ``Base``, yet ``Sub`` records ``Mid``: the entry says
        which parent the value was taken from, not which concept originally provided it.
        """
        context = check_concepts(
            {
                "Base": {
                    "directParents": ["Concept"],
                    "data": {"properties": {"prop": {"valueDomain": "String", "default": "s:base"}}},
                },
                "Mid": silent(["Base"]),
                "Sub": specializing(["Mid"], {"default": "inheritFrom:Mid"}),
            }
        )
        assert available(context, "Mid")["prop"]["default"] == "Base"
        assert available(context, "Sub")["prop"]["default"] == "Mid"

    def test_one_origin_reached_by_two_paths_is_not_ambiguous(self):
        """
        Both branches of the diamond inherit ``prop`` from the same ``Base``, so there is a single value
        and nothing to disambiguate -- even though it arrives through two parents.
        """
        check_concepts(
            {
                **BASE,
                "Left": silent(["Base"]),
                "Right": silent(["Base"]),
                "Diamond": specializing(["Left", "Right"], {"default": "s:own"}),
            }
        )

    def test_two_different_origins_are_still_ambiguous(self):
        with pytest.raises(CHSemanticError, match="A disambiguation from multiple inherited values is needed"):
            with pytest.warns(CHWarning, match="Found a domain concept with no data defined: 'Diamond'"):
                check_concepts(diamond())

    def test_a_deep_specialization_can_be_cancelled(self):
        check_concepts(
            {
                **BASE,
                "Mid": specializing(["Base"], {"default": "s:mid"}),
                "Deep": silent(["Mid"]),
                "Sub": specializing(["Deep"], {"default": "inheritFrom:"}),
            }
        )


class TestCancellationIsPropagated:
    """
    A cancellation must be visible to the concepts below. ``verify_specializations`` records ``CANCELLED``
    for a keyword removed with ``"inheritFrom:"``, and the merge onto the model drops the keyword entirely
    -- otherwise the value inherited from further up would survive the cancel as the data is passed down.
    """

    def test_a_cancelled_keyword_is_dropped_from_the_available_data(self):
        context = check_concepts({**BASE_WITH_DEFAULT, "L": specializing(["Base"], {"default": "inheritFrom:"})})
        assert "default" not in available(context, "L")["prop"]

    def test_cancelling_in_one_parent_leaves_the_other_unambiguous(self):
        """Only ``M`` supplies a default -- ``L`` cancelled it -- so there is one value and no ambiguity."""
        check_concepts(
            {
                **BASE_WITH_DEFAULT,
                "L": specializing(["Base"], {"default": "inheritFrom:"}),
                "M": specializing(["Base"], {"default": "s:m"}),
                "D": silent(["L", "M"]),
            }
        )

    def test_a_cancelled_keyword_stays_cancelled_further_down(self):
        """The keyword must not reappear for a grandchild of the concept that cancelled it."""
        context = check_concepts(
            {
                **BASE_WITH_DEFAULT,
                "L": specializing(["Base"], {"default": "inheritFrom:"}),
                "Deep": silent(["L"]),
            }
        )
        assert "default" not in available(context, "Deep")["prop"]

    def test_a_concept_may_set_a_value_again_after_an_ancestor_cancelled_it(self):
        context = check_concepts(
            {
                **BASE_WITH_DEFAULT,
                "L": specializing(["Base"], {"default": "inheritFrom:"}),
                "Deep": specializing(["L"], {"default": "s:again"}),
            }
        )
        assert available(context, "Deep")["prop"]["default"] == "Deep"

    def test_the_cancellation_sentinel_never_reaches_the_model(self):
        """``CANCELLED`` is an internal marker on the definition; the model view only holds concept names."""
        context = check_concepts({**BASE_WITH_DEFAULT, "L": specializing(["Base"], {"default": "inheritFrom:"})})
        for data in context.model.domain_concepts["L"].available_property_data.values():
            for provider in data.values():
                assert isinstance(provider, str), f"expected a concept name, got {provider!r}"


# --------------------------------------------------------------------------------------------------
# The full matrix: every pair of parent states for one keyword
# --------------------------------------------------------------------------------------------------

BASE_WITH_DEFAULT_ONLY = {
    "Base": {
        "directParents": ["Concept"],
        "data": {"properties": {"prop": {"valueDomain": "String", "default": "s:base"}}},
    }
}

PARENT_STATES: dict[str, tuple[dict | str | None, str | None]] = {
    # state name -> (specialization body for `prop`, which concept then provides `default`)
    "silent": (None, "Base"),
    "set_own": ({"default": "s:own"}, "SELF"),
    "set_same": ({"default": "s:same"}, "SELF"),
    "get_from_base": ({"default": "inheritFrom:Base"}, "Base"),
    "cancel": ({"default": "inheritFrom:"}, None),
}
"""
The states a parent can be in for a single definition keyword, with the concept that ends up providing the
value. ``set_own`` and ``set_same`` differ only in the value written: both are SET VALUE, so both provide
the value themselves -- specializing is what counts, not what was written.
"""


def _parent(body) -> dict:
    return silent(["Base"]) if body is None else specializing(["Base"], body)


def _providers(left_state: str, right_state: str) -> set[str]:
    """The distinct concepts providing ``default`` to a concept inheriting both parents."""
    provided_by = set()
    for side, state in (("L", left_state), ("R", right_state)):
        provider = PARENT_STATES[state][1]
        if provider is not None:
            provided_by.add(side if provider == "SELF" else provider)
    return provided_by


@pytest.mark.parametrize("right_state", list(PARENT_STATES))
@pytest.mark.parametrize("left_state", list(PARENT_STATES))
def test_multi_parent_ambiguity_matrix(left_state: str, right_state: str):
    """
    A keyword is ambiguous exactly when two or more *distinct concepts* provide a value for it.

    Values are never compared: two parents that each specialize the keyword are ambiguous even when they
    wrote the same value, because each of them is its own provider. Conversely, one value reached through
    several paths -- both parents silent, or both naming the same parent -- is a single provider and needs
    no disambiguation, and a parent that cancelled the keyword provides nothing at all.
    """
    concepts = {
        **BASE_WITH_DEFAULT_ONLY,
        "L": _parent(PARENT_STATES[left_state][0]),
        "R": _parent(PARENT_STATES[right_state][0]),
        "D": silent(["L", "R"]),
    }
    should_be_ambiguous = len(_providers(left_state, right_state)) > 1
    if should_be_ambiguous:
        with pytest.raises(CHSemanticError, match="A disambiguation from multiple inherited values is needed"):
            check_concepts(concepts)
    else:
        check_concepts(concepts)


class TestSameValueIsStillTwoSpecializations:
    """Values are not compared: specializing in both parents is ambiguous however equal the values are."""

    def test_two_parents_writing_the_same_value_are_ambiguous(self):
        with pytest.raises(CHSemanticError, match="A disambiguation from multiple inherited values is needed"):
            check_concepts(
                {
                    **BASE_WITH_DEFAULT_ONLY,
                    "L": specializing(["Base"], {"default": "s:same"}),
                    "R": specializing(["Base"], {"default": "s:same"}),
                    "D": silent(["L", "R"]),
                }
            )

    def test_two_parents_naming_the_same_parent_are_not(self):
        """``inheritFrom:Base`` on both sides is not a specialization of the value -- it is one provider."""
        check_concepts(
            {
                **BASE_WITH_DEFAULT_ONLY,
                "L": specializing(["Base"], {"default": "inheritFrom:Base"}),
                "R": specializing(["Base"], {"default": "inheritFrom:Base"}),
                "D": silent(["L", "R"]),
            }
        )


AMBIGUOUS_PARENTS = {
    **BASE_WITH_DEFAULT_ONLY,
    "L": specializing(["Base"], {"default": "s:left"}),
    "R": specializing(["Base"], {"default": "s:right"}),
}


class TestResolvingAnAmbiguity:
    """Every way a concept can settle a keyword its parents disagree about."""

    def test_staying_silent_is_rejected(self):
        with pytest.raises(CHSemanticError, match="A disambiguation from multiple inherited values is needed"):
            check_concepts({**AMBIGUOUS_PARENTS, "D": silent(["L", "R"])})

    def test_setting_its_own_value(self):
        context = check_concepts({**AMBIGUOUS_PARENTS, "D": specializing(["L", "R"], {"default": "s:own"})})
        assert available(context, "D")["prop"]["default"] == "D"

    def test_naming_one_of_the_parents(self):
        context = check_concepts({**AMBIGUOUS_PARENTS, "D": specializing(["L", "R"], {"default": "inheritFrom:L"})})
        assert available(context, "D")["prop"]["default"] == "L"

    def test_cancelling_the_keyword(self):
        context = check_concepts({**AMBIGUOUS_PARENTS, "D": specializing(["L", "R"], {"default": "inheritFrom:"})})
        assert "default" not in available(context, "D")["prop"]

    def test_the_resolution_is_inherited_further_down(self):
        context = check_concepts(
            {
                **AMBIGUOUS_PARENTS,
                "D": specializing(["L", "R"], {"default": "s:own"}),
                "Deep": silent(["D"]),
            }
        )
        assert available(context, "Deep")["prop"]["default"] == "D"
