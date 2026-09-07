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

from typing import Callable

from frozendict import frozendict

from concept_hierarchy.data.contexts.context import ConceptHierarchyContext
from concept_hierarchy.definitions.concept_definition_domain_concept import (
    CANCELLED,
    INHERIT_FROM_KEYWORD,
    DomainConceptDefinition,
    ForPropertyOrFunction,
    FunctionDefinitionKeywords,
    PropertyDefinitionKeywords,
)
from concept_hierarchy.errors import CHSemanticError, CHSyntaxError, LocationId, PathPart


def verify_specializations(
    c: DomainConceptDefinition,
    for_either_properties_or_functions: ForPropertyOrFunction,
    subconcepts_spec_data: dict[str, set[str]] | None,
    available_parent_data: dict[str, dict[str, list[str]]],
    location_id: LocationId,
    canonical_concept_name: Callable[[str], str] = lambda concept_name: concept_name,
    verbose: bool = False,
) -> dict[str, set[str]]:
    verify_for_subconcepts = subconcepts_spec_data is None
    if for_either_properties_or_functions.value:
        data_type, data_type_plural, available_data = "property", "properties", c.available_property_data
        specialization_keys, concept_data = PropertyDefinitionKeywords.SPECIALIZATION_KEYWORDS, c.properties
        conjunctive_specialization_keys = PropertyDefinitionKeywords.CONJUNCTIVE_SPECIALIZATION_KEYWORDS
        if verify_for_subconcepts:
            specialization_content = c.property_specializations_for_sub
        else:
            specialization_content = c.property_specializations_for_this
        check_valid_types_at_specialization = DomainConceptDefinition.check_property_data_types
    else:
        data_type, data_type_plural, available_data = "function", "functions", c.available_function_data
        specialization_keys, concept_data = FunctionDefinitionKeywords.SPECIALIZATION_KEYWORDS, c.functions
        conjunctive_specialization_keys = FunctionDefinitionKeywords.CONJUNCTIVE_SPECIALIZATION_KEYWORDS
        if verify_for_subconcepts:
            specialization_content = c.function_specializations_for_sub
        else:
            specialization_content = c.function_specializations_for_this
        check_valid_types_at_specialization = DomainConceptDefinition.check_function_data_types

    # { name: { def_keys } } contains all the def_keys for prop/func name which was SET, GET_FROM_PARENT, or CANCELLED
    specialized_data: dict[str, set[str]] = {}
    data_not_from_subconcepts_specialization: set[tuple[str, str]] = set()
    # `name` is the name of the DomainConcept property or function
    for name, spec_data in specialization_content.items():
        if name not in concept_data and name not in available_parent_data:
            raise CHSemanticError(
                f"{name!r} is not an available {data_type} (neither defined nor inherited) for concept {c.name}! "
                f"Can't define specialization data for it!",
                location_id=location_id + [name],
                part=PathPart.KEY,
            )

        specialized_data[name] = set()
        if not isinstance(spec_data, dict):
            raise RuntimeError(
                f"{spec_data!r} should have been processed into a dictionary in domain concepts! "
                f"At {c.name}, {data_type_plural}, forSub={verify_for_subconcepts}"
            )
        check_valid_types_at_specialization(c, name, spec_data, True, verify_for_subconcepts)
        cancelled_keys = []
        for def_key, def_data in spec_data.items():
            # assertion, not check because this is a key of a JSON object
            assert isinstance(def_key, str)
            if def_key not in specialization_keys:
                raise CHSyntaxError(
                    f"Unknown definition key {def_key!r} for specialization of {data_type} {name} at {c.name}! "
                    f"Available definition keys to specialize are {specialization_keys!r}",
                    location_id=location_id + [name, def_key],
                    part=PathPart.KEY,
                )
            if not verify_for_subconcepts:
                assert isinstance(def_data, tuple)
                def_data = def_data[0]
            if isinstance(def_data, str) and def_data.startswith(INHERIT_FROM_KEYWORD):
                if def_data == INHERIT_FROM_KEYWORD:
                    if verbose:
                        print("CANCEL", name, def_key, "at", c.name, def_data)
                    if def_key in conjunctive_specialization_keys:
                        raise CHSemanticError(
                            f"Conjunctive ({data_type}) definition keywords can not be cancelled in specializations! "
                            f'Change the specialization value of "{def_key}"',
                            location_id=location_id + [name, def_key],
                            part=PathPart.VALUE,
                        )
                    # CANCEL
                    cancelled_keys.append(def_key)
                    specialized_data[name].add(def_key)
                    # Record the cancellation so that it is visible to the subconcepts.
                    # Without it, the keyword simply stays absent here, and the value inherited from further up survives
                    # the cancel when this concept's available data is passed down.
                    available_data.setdefault(name, {})[def_key] = CANCELLED
                    continue
                if def_data == INHERIT_FROM_KEYWORD + "parents":
                    # this keyword is only allowed in a "_forThis" specialization
                    if verify_for_subconcepts:
                        raise CHSemanticError(
                            f'The "{INHERIT_FROM_KEYWORD}parents" specialization value is only available under the '
                            f'"_forThis" specialization data!\nThis value is how the onlyForSubconcepts specialization '
                            f"mode is implemented.",
                            location_id=location_id + [name, def_key],
                            part=PathPart.VALUE,
                        )
                    data_not_from_subconcepts_specialization.add((name, def_key))
                    continue
                # GET VALUE FROM PARENT disambiguation
                res = def_data.split(INHERIT_FROM_KEYWORD)
                # the named parent is a concept-name position, so it may be written as an alias, while
                # `c.parents` is canonical -- record the parent under the name the hierarchy is keyed by,
                # and annotate the use site with the alias so a message naming the canonical parent can
                # still be traced back to what was written here
                inherit_from_location = location_id + [name, def_key]
                if len(res) == 2 and res[0] == "":
                    canonical_parent = canonical_concept_name(res[1])
                    if canonical_parent != res[1]:
                        inherit_from_location = inherit_from_location + ["ref:" + res[1]]
                    res[1] = canonical_parent
                if len(res) != 2 or res[0] != "" or res[1] not in c.parents:
                    not_in_parents = res[1] not in c.parents
                    if not_in_parents:
                        raise CHSemanticError(
                            f"Specified parent {res[1]} is not a direct parent of {c.name}! Only allowed to specify the"
                            f" inherited value from the direct parents; in this case, only from {c.parents!r}",
                            location_id=inherit_from_location,
                            part=PathPart.VALUE,
                        )
                    raise CHSyntaxError(
                        f"Wrong syntax for defining the specialization by getting the value from parent.\nUse "
                        f'"{INHERIT_FROM_KEYWORD}<NameOfDirectParentFromWhichToInheritTheData>", not {def_data}.'
                        f"\n\tEncountered at {c.name}",
                        location_id=location_id + [name, def_key],
                        part=PathPart.VALUE,
                    )
                # check whether the INHERIT_FROM_KEYWORD is actually allowed for this property
                #  (i.e. check if the value was not defined in this concept!)
                if name in concept_data:
                    raise CHSemanticError(
                        f"Can't define specialization for {name} as an inherited value for a {data_type} that is "
                        f"defined in this concept ({c.name})\nGot {def_data!r}",
                        location_id=location_id + [name, def_key],
                        part=PathPart.VALUE,
                    )
                if verbose:
                    print("GET VALUE FROM PARENT", name, def_key, "at", c.name, def_data)
                if name not in available_data:
                    available_data[name] = {}
                available_data[name][def_key] = res[1]
                # don't update specialization_content;
                # not needed because this is data already defined in specialization_content
                specialized_data[name].add(def_key)
            else:
                # SET VALUE
                if verbose:
                    print("SET VALUE", name, def_key, "at", c.name, def_data)
                if name not in available_data:
                    available_data[name] = {}
                available_data[name][def_key] = c.name
                specialized_data[name].add(def_key)
        # A cancelled keyword is deliberately left in place here:
        # the concept definition keeps describing what its JSON says.
        # The cancellation is recorded as CANCELLED in the available-data map instead,
        # which is what the concepts below read -- so nothing needs the definition data to be edited.

    # check that there is no data in available_parent_data that has two parents and is not set or disambiguated
    for name, available_parent_data_at_name in available_parent_data.items():
        for def_key, parents_defining in available_parent_data_at_name.items():
            if name in specialized_data and def_key in specialized_data[name]:
                continue
            if (
                not verify_for_subconcepts
                and (name, def_key) not in data_not_from_subconcepts_specialization
                and name in subconcepts_spec_data
                and def_key in subconcepts_spec_data[name]
            ):
                continue
            if len(parents_defining) > 1:
                raise CHSemanticError(
                    f"A disambiguation from multiple inherited values is needed in specialization of {data_type} "
                    f'"{name}" at definition key "{def_key}"!',
                    location_id=location_id,
                    part=PathPart.VALUE,
                )

    return specialized_data


def _merge_available_data(
    inherited_data: dict[str, dict[str, list[str]]], own_data: dict[str, dict[str, str]]
) -> frozendict[str, frozendict[str, str]]:
    """
    Combine what a concept inherits with what its own definition provides, into the derived per-concept view.

    The entries are per definition keyword, and each names the concept the *value of that keyword* comes
    from -- not the single concept that defines the property. One property routinely spans several: a
    concept can define ``prop`` while a subconcept sets its ``default``. Keeping it per keyword is what lets
    ``verify_specializations`` tell "two parents provide different defaults" from "one value, two paths".
    For a ``inheritFrom:<Parent>`` the recorded concept is that named direct parent, by design, even when
    the parent itself only inherited the value.

    What the concept provides itself always wins. Where only the parents provide a keyword there is normally
    exactly one provider -- ``verify_specializations`` has already rejected any unresolved ambiguity -- but
    the ``_forThis`` slot has exemptions that let several through, and the first is kept for those.
    """
    merged: dict[str, dict[str, str]] = {}
    for name, inherited_data_at_name in inherited_data.items():
        merged[name] = {def_key: providers[0] for def_key, providers in inherited_data_at_name.items()}
    for name, own_data_at_name in own_data.items():
        merged_at_name = merged.setdefault(name, {})
        for def_key, provider in own_data_at_name.items():
            if provider is CANCELLED:
                # NO VALUE: the concept removed the keyword, so nothing is available to pass on
                merged_at_name.pop(def_key, None)
            else:
                merged_at_name[def_key] = provider
    return frozendict({name: frozendict(data) for name, data in merged.items()})


def process_specialization_for_domain_concepts(context: ConceptHierarchyContext):
    """
    Property Data Specialization
    =============================

    Two Slots Per Keyword
    ----------------------
    Every property keyword (``constraint``, ``default``, ``computations``, ``hooks``, …)
    has two independent slots at each concept node:

        for-this            the value applied when instantiating *this* concept
        for-subconcepts     the value passed down and inherited by subconcepts

    These slots can be filled independently and may hold different values.
    The JSON structure maps directly onto this split:

        _specializations            fills the *for-subconcepts* slot for inherited properties
        _specializations._forThis   fills the *for-this* slot (for any property)

    If a keyword is written in ``_specializations`` but not overridden in ``_forThis``,
    the same value is used for both slots.


    Value States
    -------------
    Each slot can hold one of the following states:

        SET VALUE               an explicit value is provided
        NO VALUE                the slot is intentionally empty
        IMPLICIT GET            the slot is unspecified; resolved from parent(s) at compile time
        GET VALUE FROM PARENT   an explicit reference to a named parent's value

    Notes on applicability:

        - For a *defined* property: only SET VALUE and NO VALUE are possible.
        - For an *inherited* property: all four states are available.
        - IMPLICIT GET and GET VALUE FROM PARENT are inheritance mechanisms and therefore
          only meaningful for inherited properties.

    JSON syntax:

        SET VALUE               "keyword": <value>
        NO VALUE                "keyword": "inheritFrom:"           (in specializations)
                                keyword absent                      (in definition body)
        GET VALUE FROM PARENT   "keyword": "inheritFrom:<Name>"
        IMPLICIT GET            keyword absent                      (in specializations)

    WARNING: ``"inheritFrom:"`` and ``"inheritFrom:<Name>"`` differ by only a name suffix
             but mean opposite things:

                 "inheritFrom:"          NO VALUE — explicitly cancels the inherited value
                 "inheritFrom:<Name>"    GET VALUE FROM PARENT — selects which parent to inherit from


    IMPLICIT GET Hazard (Multi-Parent Ambiguity)
    ---------------------------------------------
    When a slot is left unspecified in ``_specializations``, IMPLICIT GET is assumed.
    Resolution depends on how many parent concepts carry a non-NO-VALUE for that keyword:

        One parent has a value      that value is used (unambiguous)
        No parent has a value       NO VALUE is used (safe)
        Multiple parents have       ERROR: the framework cannot choose a winner;
          a value                     use ``"inheritFrom:<Name>"`` to disambiguate

    IMPLICIT GET is only safe when at most one parent contributes a value for that keyword.


    Where to Write Specializations
    --------------------------------
    The correct location depends on whether the property is *defined* at this concept
    or *inherited* from a parent.

    Defined properties (``valueDomain`` is declared on this concept):

        - The keyword in the definition body fills the *for-subconcepts* slot.
        - To give *this* concept a different value, write the keyword in ``_forThis``.
        - To set *this* concept's slot to NO VALUE while keeping the definition's value
          for subconcepts, write ``"keyword": "inheritFrom:"`` in ``_forThis``.
        - Defined property keywords must NOT appear in the top-level ``_specializations``
          block (outside ``_forThis``); their *for-subconcepts* slot is already filled
          at the definition site.

    Inherited properties (defined on a parent concept):

        - Write the keyword in the top-level ``_specializations`` block to fill the
          *for-subconcepts* slot (and implicitly *for-this*, unless ``_forThis`` overrides it).
        - Write the keyword in ``_forThis`` to fill or override the *for-this* slot
          independently.

    In all cases, a concept fills only its own slots. It cannot overwrite a parent's slots.
    Each concept in the chain independently decides what value it passes to its subconcepts
    and what value it uses for itself.

    Summary:

        Property type       for-subconcepts slot             for-this slot
        ─────────────────   ──────────────────────────────   ──────────────────────────────
        Defined             definition body                  _forThis  (optional override)
        Inherited           _specializations                 _forThis  (optional override)


    Example
    --------
    ::

        {
          "X": {
            "data": {
              "properties": {
                "p": {
                  "valueDomain": "Integer",          // property definition (cannot be specialized)
                  "constraint": {
                    "Interval<Integer>": [0, 5]
                  },                                 // SET VALUE — same for this concept and subconcepts
                  "default": 1,                      // SET VALUE (for subconcepts);
                                                     //   overridden for this concept in _forThis below
                                                     // "hooks" absent → NO VALUE for this concept
                                                     //   and for subconcepts
                },
                "_specializations": {
                  "_forThis": {
                    "p": {
                      "default": 4                   // SET VALUE for this concept only;
                                                     //   instances of X use 4, instances of Y use 1
                    }
                  }
                }
              }
            }
          },
          "Y": {
            "directParents": ["X"],
            "data": {
              "properties": {
                "_specializations": {
                  "p": {
                    "constraint": {
                      "Interval<Integer>": "inheritFrom:"
                    },                               // NO VALUE for subconcepts of Y;
                                                     //   overridden for this concept in _forThis below
                    "default": "inheritFrom:X",      // GET VALUE FROM PARENT X (→ 1) for subconcepts of Y;
                                                     //   also applies to this concept (no _forThis override)
                    "computations": { ... },         // SET VALUE — same for this concept and for subconcepts
                                                     // all other keywords: IMPLICIT GET
                                                     //   (safe: Y has exactly one parent, X)
                  },
                  "_forThis": {
                    "p": {
                      "constraint": {
                        "Interval<Integer>": [1, 5]
                      }                              // SET VALUE for this concept only;
                                                     //   subconcepts of Y see NO VALUE (set above)
                    }
                  }
                }
              }
            }
          }
        }
    """
    for c_name in context.ch.concept_topo_sort:
        if not context.ch.is_domain_concept(c_name):
            continue
        c = context.ch.concepts[c_name]
        c_model = context.model.domain_concepts[c_name]
        assert isinstance(c, DomainConceptDefinition)
        concept_location = c.location_id()
        property_location = concept_location + [c.domain_concept_properties, c.domain_concept_specialization]
        function_location = concept_location + [c.domain_concept_functions, c.domain_concept_specialization]
        # Compile the data available from parents: { name: { def key: [ direct parent concept setting the value ] } }.
        # This reads each parent's *model* data, which already carries everything that parent inherited, so a
        # property is available all the way down instead of only one generation below where it is defined.
        # The concept definitions are not consulted here: those describe only what their own JSON declares.
        # Entries record the providing concept rather than the parent it was reached through, so that one
        # value inherited along several paths is recognised as one value and not as an ambiguity.
        available_parent_data_for_properties: dict[str, dict[str, list[str]]] = {}
        available_parent_data_for_functions: dict[str, dict[str, list[str]]] = {}
        for parent_name in c.parents:
            parent_datum = context.model.domain_concepts[parent_name]
            for collection, parent_data in [
                (available_parent_data_for_properties, parent_datum.available_property_data),
                (available_parent_data_for_functions, parent_datum.available_function_data),
            ]:
                for name, available_parent_data in parent_data.items():
                    if name not in collection:
                        collection[name] = {}
                    for available_parent_def_key, providing_concept in available_parent_data.items():
                        if available_parent_def_key not in collection[name]:
                            collection[name][available_parent_def_key] = []
                        if providing_concept not in collection[name][available_parent_def_key]:
                            collection[name][available_parent_def_key].append(providing_concept)

        prop_spec_data = verify_specializations(
            c,
            ForPropertyOrFunction.PROPERTY,
            None,
            available_parent_data_for_properties,
            location_id=property_location,
            canonical_concept_name=context.ch.canonical_concept_name,
        )
        verify_specializations(
            c,
            ForPropertyOrFunction.PROPERTY,
            prop_spec_data,
            available_parent_data_for_properties,
            location_id=property_location + [DomainConceptDefinition.domain_concept_specialization_for_this],
            canonical_concept_name=context.ch.canonical_concept_name,
        )
        func_spec_data = verify_specializations(
            c,
            ForPropertyOrFunction.FUNCTION,
            None,
            available_parent_data_for_functions,
            location_id=function_location,
            canonical_concept_name=context.ch.canonical_concept_name,
        )
        verify_specializations(
            c,
            ForPropertyOrFunction.FUNCTION,
            func_spec_data,
            available_parent_data_for_functions,
            location_id=function_location + [DomainConceptDefinition.domain_concept_specialization_for_this],
            canonical_concept_name=context.ch.canonical_concept_name,
        )

        # Record the derived view on the model: everything inherited from the parents, overlaid with what
        # this concept's own definition provides (`verify_specializations` has just finished filling that in).
        # Subconcepts read this, so the data keeps flowing down through concepts that say nothing themselves.
        c_model.available_property_data = _merge_available_data(
            available_parent_data_for_properties, c.available_property_data
        )
        c_model.available_function_data = _merge_available_data(
            available_parent_data_for_functions, c.available_function_data
        )
