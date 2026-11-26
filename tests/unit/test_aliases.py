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
Tests for aliases -- see ``documentation/TODO_ALIASES_IMPLEMENTATION.md`` for the design.

An alias is a **name, not an entity**: one definition reached under several names. All three kinds are
written the same way, as an entry whose definition is a bare JSON string naming its target, and which kind
an entry is follows from parsing that target::

    "Beast": "Animal"          a bare concept name  -> concept alias  -> ch.concept_aliases
    "IntBox": "Box<Integer>"   an applied type      -> type alias     -> ch.type_aliases
    "start": "origin"          under "instances"    -> variable alias -> ch.variable_aliases

A concept alias is usable wherever a **concept name** is expected -- and, because a concept name is also a
ground type, in type positions too; aliasing a templatable ValueDomain therefore aliases a *type
constructor*, so ``"MyBox": "Box"`` makes ``MyBox<Integer>`` legal and equal to ``Box<Integer>``. A type
alias holds a saturated, ground :class:`InstantiatedType` and is usable only where a **type** is expected:
it can neither be parameterised further nor stand in for a concept name. A variable alias is usable
wherever a global variable is.

The consequences the whole design turns on, and which most of these tests pin:

* ``ch.concepts`` and ``ch.instances`` hold **canonical entries only**. An alias never appears there, nor
  in the topological sort, the subconcept relation, or the "defining concept" maps -- which is what retires
  today's "The property X is defined in multiple places!" for an alias of a concept that defines
  properties.
* Aliases are resolved at the **lookup boundary**, never by rewriting the definition data.
* ``is_concept`` / ``is_variable`` and friends accept an alias, so a user-written name checks out wherever
  it is written.
* A chain of aliases resolves to the **canonical** target, and a cycle in it is rejected -- across kinds as
  well as within one kind.

**Concept and variable aliases are implemented; type aliases are not**, so the type-alias tests fail. They
are left failing rather than ``xfail``-ed, in the same spirit as ``tests/unit/test_name_uniqueness.py``, so
the gap stays visible. The alias containers live on :class:`ConceptHierarchyDefinition` (reached as
``context.ch``) next to ``concepts`` / ``instances``, because the code-generation backend reads that model
(§3.2, §8).
"""

from __future__ import annotations

import pytest

from concept_hierarchy.data.expressions.subexpressions import Variable
from concept_hierarchy.data.types.concept_hierarchy_types import InstantiatedType
from concept_hierarchy.errors import CHSemanticError, ConceptHierarchyError, LocationId
from tests.ch_support import check_concepts

# --------------------------------------------------------------------------------------------------
# Fixtures and accessors
# --------------------------------------------------------------------------------------------------

ANIMAL: dict[str, dict] = {"Animal": {"directParents": ["Concept"], "data": {"properties": {"age": "Integer"}}}}
"""A DomainConcept that *defines a property*: aliasing it is the case that breaks today (§2.1)."""

BOX: dict[str, dict] = {
    "Box": {"directParents": ["ValueDomain"], "data": {"templateContext": {"order": ["T"], "T": "ValueDomain"}}}
}
"""A templatable ValueDomain -- the type constructor the type-alias tests apply."""

NUMBER_BOX: dict[str, dict] = {
    "NumberBox": {"directParents": ["ValueDomain"], "data": {"templateContext": {"order": ["T"], "T": "Number"}}}
}
"""Like :data:`BOX`, but its template argument is constrained, so the constraint literal can be aliased."""

PAIR: dict[str, dict] = {
    "Pair": {
        "directParents": ["ValueDomain"],
        "data": {"templateContext": {"order": ["A", "B"], "A": "ValueDomain", "B": "ValueDomain"}},
    }
}
"""A ValueDomain with *two* template arguments: aliases have to land in the right positions, in order."""


COUNTDOWN: dict[str, dict] = {
    "Countdown": {
        "directParents": ["FunctionReturning"],
        "data": {
            "templateContext": {"order": ["T"], "T": "Numeric", "substitution": {"FunctionReturning:T": "T"}},
            # the default value of `from` is a *global variable reference* -- an expression position where a
            # variable name is expected, parsed after the global variables are in scope
            "interface": {"from": "Integer", "res": "T", "_defaultArgumentValues": {"from": "start"}},
        },
    }
}
"""A Function whose default argument value names the global variable ``start``."""


def holder(property_type: str, name: str = "Holder", prop: str = "p") -> dict:
    """A DomainConcept with a single property of the given type -- one type position, written by the user."""
    return {name: {"directParents": ["Concept"], "data": {"properties": {prop: property_type}}}}


def concept_aliases(context) -> dict[str, str]:
    """The concept-alias container: alias name -> the canonical concept name it denotes."""
    return dict(context.ch.concept_aliases)


def type_aliases(context) -> dict[str, str]:
    """The type-alias container, with each :class:`InstantiatedType` rendered as its full name."""
    return {name: str(alias_type) for name, alias_type in context.ch.type_aliases.items()}


def variable_aliases(context) -> dict[str, str]:
    """The variable-alias container: alias name -> the canonical global variable name it denotes."""
    return dict(context.ch.variable_aliases)


def property_type(context, concept: str = "Holder", prop: str = "p") -> str:
    """The resolved type of a DomainConcept property, as its full name."""
    return str(context.model.domain_concepts[concept].property_types[prop])


def instantiation_types(context, value_domain: str) -> list[str]:
    """The custom (Concept Hierarchy) types appearing in a ValueDomain's instantiation schemas."""
    return [
        str(node.custom_type)
        for _constraint, schema in context.model.value_domains[value_domain].instantiation
        for node in schema.walk()
        if getattr(node, "custom_type", None) is not None
    ]


# --------------------------------------------------------------------------------------------------
# Which container a string entry lands in
# --------------------------------------------------------------------------------------------------


class TestAliasClassification:
    """The kind of an alias is decided by parsing its target, not by new syntax."""

    def test_a_bare_concept_name_target_is_a_concept_alias(self):
        context = check_concepts({**ANIMAL, "Beast": "Animal"})
        assert concept_aliases(context) == {"Beast": "Animal"}
        assert type_aliases(context) == {}

    def test_an_applied_type_target_is_a_type_alias(self):
        context = check_concepts({**BOX, "IntBox": "Box<Integer>"})
        assert type_aliases(context) == {"IntBox": "Box<Integer>"}
        assert concept_aliases(context) == {}

    def test_a_type_alias_holds_a_saturated_ground_type(self):
        context = check_concepts({**BOX, "IntBox": "Box<Integer>"})
        alias_type = context.ch.type_aliases["IntBox"]
        assert isinstance(alias_type, InstantiatedType)
        assert alias_type.depends_on_templates is False

    def test_an_instances_entry_naming_another_variable_is_a_variable_alias(self):
        context = check_concepts({}, instances={"origin": 0, "start": "origin"})
        assert variable_aliases(context) == {"start": "origin"}

    def test_an_instances_entry_naming_no_variable_stays_a_value(self):
        """A global variable's value may itself be a string expression; only a *variable name* aliases."""
        context = check_concepts({}, instances={"greeting": "s:hello"})
        assert variable_aliases(context) == {}
        assert "greeting" in context.ch.instances

    def test_an_alias_target_that_does_not_exist_is_rejected(self):
        with pytest.raises(CHSemanticError, match="does not exist"):
            check_concepts({**ANIMAL, "Beast": "NoSuchConcept"})

    def test_a_type_alias_over_an_undefined_head_is_rejected(self):
        with pytest.raises(ConceptHierarchyError):
            check_concepts({**BOX, "IntBox": "NoSuchBox<Integer>"})

    def test_a_type_alias_with_an_undefined_template_argument_is_rejected(self):
        with pytest.raises(ConceptHierarchyError):
            check_concepts({**BOX, "IntBox": "Box<NoSuchConcept>"})


# --------------------------------------------------------------------------------------------------
# Aliases are names, not entities
# --------------------------------------------------------------------------------------------------


class TestAnAliasIsNotAnEntity:
    """
    The whole point of the design: one definition, several names. Nothing that enumerates the hierarchy
    may see an alias, and nothing that records *where* something is defined may name one.
    """

    def test_a_concept_alias_is_not_a_concept_entry(self):
        context = check_concepts({**ANIMAL, "Beast": "Animal"})
        assert "Beast" not in context.ch.concepts
        assert "Beast" not in context.model.concepts
        assert "Beast" not in context.model.domain_concepts

    def test_a_concept_alias_is_not_in_the_topological_sort(self):
        context = check_concepts({**ANIMAL, "Beast": "Animal"})
        assert "Beast" not in context.ch.concept_topo_sort
        assert "Beast" not in context.ch.all_concept_parents

    def test_a_type_alias_is_not_a_concept_entry(self):
        context = check_concepts({**BOX, "IntBox": "Box<Integer>"})
        assert "IntBox" not in context.ch.concepts
        assert "IntBox" not in context.ch.concept_topo_sort
        assert "IntBox" not in context.model.value_domains

    def test_a_variable_alias_is_not_an_instance_entry(self):
        context = check_concepts({}, instances={"origin": 0, "start": "origin"})
        assert "start" not in context.ch.instances
        assert "start" not in context.model.instances

    def test_a_variable_alias_does_not_create_a_second_variable(self):
        """§3.5: the alias and its target are the one ``GlobalVariableData``, so there is only one entry."""
        context = check_concepts({}, instances={"origin": 0, "start": "origin"})
        assert list(context.ch.instances) == ["origin"]

    def test_an_alias_of_a_concept_with_properties_does_not_duplicate_the_property(self):
        """The regression this whole design exists for -- today this raises "defined in multiple places"."""
        context = check_concepts({**ANIMAL, "Beast": "Animal"})
        assert context.ch.all_domain_concept_properties["age"] == "Animal"

    def test_an_alias_of_a_concept_with_functions_does_not_duplicate_the_function(self):
        context = check_concepts(
            {
                "Animal": {"directParents": ["Concept"], "data": {"functions": {"speak": {}}}},
                "Beast": "Animal",
            }
        )
        assert context.ch.all_domain_concept_functions["speak"] == "Animal"

    def test_an_alias_of_a_value_domain_does_not_duplicate_its_default_serialization(self):
        """``defaultSerialization`` must be unique across concepts; an alias is not a second concept."""
        context = check_concepts({"Text": "String"})
        assert context.ch.default_serializations["string"] == "String"

    def test_an_alias_is_not_a_direct_child_of_its_targets_parents(self):
        context = check_concepts({**ANIMAL, "Beast": "Animal"})
        assert "Beast" not in context.ch.defined_direct_children["Concept"]

    def test_a_child_of_an_alias_is_recorded_under_the_canonical_parent(self):
        context = check_concepts(
            {**ANIMAL, "Beast": "Animal", "Dog": {"directParents": ["Beast"], "data": {"properties": {}}}}
        )
        assert context.ch.defined_direct_children["Animal"] == {"Dog"}
        assert "Beast" not in context.ch.defined_direct_children

    def test_a_concepts_parents_are_recorded_canonically(self):
        """``parents`` is derived data, so canonicalising it there makes every consumer alias-safe (§5c)."""
        context = check_concepts(
            {**ANIMAL, "Beast": "Animal", "Dog": {"directParents": ["Beast"], "data": {"properties": {}}}}
        )
        assert context.ch.concepts["Dog"].parents == ("Animal",)


# --------------------------------------------------------------------------------------------------
# Alias-aware membership predicates
# --------------------------------------------------------------------------------------------------


class TestAliasAwarePredicates:
    """
    Every check written against a user-supplied name goes through these, so they must accept an alias --
    while the containers behind them stay canonical (see :class:`TestAnAliasIsNotAnEntity`).
    """

    def test_is_concept_accepts_a_concept_alias(self):
        context = check_concepts({**ANIMAL, "Beast": "Animal"})
        assert context.ch.is_concept("Beast") is True

    def test_is_concept_rejects_an_unknown_name(self):
        context = check_concepts({**ANIMAL, "Beast": "Animal"})
        assert context.ch.is_concept("NoSuchConcept") is False

    def test_is_value_domain_follows_a_concept_alias(self):
        context = check_concepts({"Text": "String"})
        assert context.ch.is_value_domain("Text") is True
        assert context.ch.is_domain_concept("Text") is False

    def test_is_domain_concept_follows_a_concept_alias(self):
        context = check_concepts({**ANIMAL, "Beast": "Animal"})
        assert context.ch.is_domain_concept("Beast") is True

    def test_is_a_subconcept_of_b_accepts_an_alias_as_the_subconcept(self):
        context = check_concepts({**ANIMAL, "Beast": "Animal"})
        assert context.ch.is_a_subconcept_of_b("Beast", "Concept", include_self=False) is True

    def test_is_a_subconcept_of_b_accepts_an_alias_as_the_superconcept(self):
        context = check_concepts(
            {**ANIMAL, "Beast": "Animal", "Dog": {"directParents": ["Animal"], "data": {"properties": {}}}}
        )
        assert context.ch.is_a_subconcept_of_b("Dog", "Beast", include_self=False) is True

    def test_an_alias_and_its_target_are_the_same_concept(self):
        context = check_concepts({**ANIMAL, "Beast": "Animal"})
        assert context.ch.is_a_subconcept_of_b("Beast", "Animal", include_self=True) is True

    def test_is_variable_accepts_a_variable_alias(self):
        context = check_concepts({}, instances={"origin": 0, "start": "origin"})
        assert context.ch.is_variable("start") is True


# --------------------------------------------------------------------------------------------------
# A concept alias wherever a concept name is expected
# --------------------------------------------------------------------------------------------------


class TestConceptAliasInConceptPositions:
    """§1: a concept alias is usable wherever a concept *name* is expected."""

    def test_in_direct_parents(self):
        check_concepts(
            {
                **ANIMAL,
                "Beast": "Animal",
                "Dog": {"directParents": ["Beast"], "data": {"properties": {"breed": "String"}}},
            }
        )

    def test_in_direct_children(self):
        """``directChildren`` must exhaustively enumerate the children, so an alias must count as its target."""
        check_concepts(
            {
                "Animal": {
                    "directParents": ["Concept"],
                    "data": {"properties": {"age": "Integer"}},
                    "directChildren": ["Puppy"],
                },
                "Dog": {"directParents": ["Animal"], "data": {"properties": {"breed": "String"}}},
                "Puppy": "Dog",
            }
        )

    def test_in_distinct_from(self):
        check_concepts(
            {
                **ANIMAL,
                "Beast": "Animal",
                "Plant": {
                    "directParents": ["Concept"],
                    "data": {"properties": {"height": "Integer"}},
                    "distinctFrom": ["Beast"],
                },
            }
        )

    def test_distinct_from_still_rejects_a_self_reference_through_an_alias(self):
        """An alias of the concept itself is the concept itself, so it can not be distinct from it."""
        with pytest.raises(CHSemanticError, match="must be distinct from itself"):
            check_concepts(
                {
                    "Animal": {
                        "directParents": ["Concept"],
                        "data": {"properties": {"age": "Integer"}},
                        "distinctFrom": ["Beast"],
                    },
                    "Beast": "Animal",
                }
            )

    def test_in_distinct_group(self):
        check_concepts(
            {
                "Animal": {
                    "directParents": ["Concept"],
                    "data": {"properties": {"age": "Integer"}},
                    "distinctGroup": ["Puppy", "Cat"],
                },
                "Dog": {"directParents": ["Animal"], "data": {"properties": {"breed": "String"}}},
                "Cat": {"directParents": ["Animal"], "data": {"properties": {"lives": "Integer"}}},
                "Puppy": "Dog",
            }
        )

    def test_in_a_template_argument_constraint_literal(self):
        """The constraint of ``T`` is written as ``Num``, an alias of ``Number``; ``Integer`` satisfies it."""
        context = check_concepts(
            {
                "Num": "Number",
                "NumberBox": {
                    "directParents": ["ValueDomain"],
                    "data": {"templateContext": {"order": ["T"], "T": "Num"}},
                },
                **holder("NumberBox<Integer>"),
            }
        )
        assert property_type(context) == "NumberBox<Integer>"

    def test_an_aliased_constraint_literal_still_rejects_a_violating_argument(self):
        with pytest.raises(CHSemanticError, match="into a type failed"):
            check_concepts(
                {
                    "Num": "Number",
                    "NumberBox": {
                        "directParents": ["ValueDomain"],
                        "data": {"templateContext": {"order": ["T"], "T": "Num"}},
                    },
                    **holder("NumberBox<String>"),
                }
            )

    def test_as_the_parent_named_by_a_template_substitution_key(self):
        """The ``<Parent>:<TemplateArgument>`` substitution key names a concept, so it accepts an alias."""
        check_concepts(
            {
                **BOX,
                "Container": "Box",
                "IntBox": {
                    "directParents": ["Box"],
                    "data": {"templateContext": {"substitution": {"Container:T": "Integer"}}},
                },
            }
        )

    def test_as_an_aliased_parent_with_a_canonical_substitution_key(self):
        check_concepts(
            {
                **BOX,
                "Container": "Box",
                "IntBox": {
                    "directParents": ["Container"],
                    "data": {"templateContext": {"substitution": {"Box:T": "Integer"}}},
                },
            }
        )

    def test_as_the_parent_named_by_an_inherit_from_specialization(self):
        """``inheritFrom:<Parent>`` names a direct parent; the canonical name is what gets recorded."""
        context = check_concepts(
            {
                "Base": {
                    "directParents": ["Concept"],
                    "data": {"properties": {"prop": {"valueDomain": "String", "default": "s:base"}}},
                },
                "Mid": "Base",
                "Sub": {
                    "directParents": ["Mid"],
                    "data": {"properties": {"_specializations": {"prop": {"default": "inheritFrom:Mid"}}}},
                },
            }
        )
        assert context.model.domain_concepts["Sub"].available_property_data["prop"]["default"] == "Base"

    def test_as_a_concept_value(self):
        """A ``format: Concept`` string names a concept, so an alias is a valid value for one."""
        check_concepts(
            {
                **ANIMAL,
                "Beast": "Animal",
                "ConceptValue": {
                    "directParents": ["String"],
                    "data": {"instantiation": {"type": "string", "pattern": "^s:", "format": "Concept"}},
                },
                "Wrap": {
                    "directParents": ["ValueDomain"],
                    "data": {
                        "instantiation": {
                            "type": "object",
                            "properties": {"c": {"type": "ConceptValue", "default": "s:Beast"}},
                        }
                    },
                },
            }
        )


# --------------------------------------------------------------------------------------------------
# A concept alias in type positions -- a concept name is a ground type
# --------------------------------------------------------------------------------------------------


class TestConceptAliasAsAType:
    def test_as_a_property_type(self):
        context = check_concepts({"Text": "String", **holder("Text")})
        assert property_type(context) == "String", "the resolved type is the canonical concept"

    def test_as_a_template_argument(self):
        context = check_concepts({**BOX, "Int": "Integer", **holder("Box<Int>")})
        assert property_type(context) == "Box<Integer>"

    def test_as_a_function_argument_and_result_type(self):
        context = check_concepts(
            {
                "Int": "Integer",
                "Sum": {
                    "directParents": ["FunctionReturning"],
                    "data": {
                        "templateContext": {"substitution": {"FunctionReturning:T": "Int"}},
                        "interface": {"arg": "Int", "res": "Int"},
                    },
                },
            }
        )
        assert str(context.model.functions["Sum"].evaluation_argument_types["arg"]) == "Integer"
        assert str(context.model.functions["Sum"].evaluation_result_type) == "Integer"

    def test_as_an_instantiation_schema_custom_type(self):
        context = check_concepts(
            {
                "Int": "Integer",
                "Wrap": {
                    "directParents": ["ValueDomain"],
                    "data": {"instantiation": {"type": "object", "properties": {"y": "Int"}}},
                },
            }
        )
        assert instantiation_types(context, "Wrap") == ["Integer"]


class TestConceptAliasAsATypeConstructor:
    """
    Aliasing a templatable ValueDomain aliases the *constructor*, not a type: the alias may still be
    applied, and the application is equal to the canonical one.
    """

    def test_an_aliased_constructor_can_be_applied(self):
        context = check_concepts({**BOX, "MyBox": "Box", **holder("MyBox<Integer>")})
        assert property_type(context) == "Box<Integer>"

    def test_an_applied_alias_equals_the_canonical_application(self):
        context = check_concepts(
            {
                **BOX,
                "MyBox": "Box",
                **holder("MyBox<Integer>", name="Left", prop="left"),
                **holder("Box<Integer>", name="Right", prop="right"),
            }
        )
        assert property_type(context, "Left", "left") == property_type(context, "Right", "right")
        assert (
            context.model.domain_concepts["Left"].property_types["left"]
            == context.model.domain_concepts["Right"].property_types["right"]
        )

    def test_an_aliased_constructor_still_checks_its_template_constraints(self):
        with pytest.raises(CHSemanticError, match="into a type failed"):
            check_concepts({**NUMBER_BOX, "MyBox": "NumberBox", **holder("MyBox<String>")})

    def test_an_aliased_constructor_still_checks_its_arity(self):
        with pytest.raises(CHSemanticError, match="into a type failed"):
            check_concepts({**BOX, "MyBox": "Box", **holder("MyBox<Integer, Integer>")})

    def test_an_unapplied_constructor_alias_is_not_a_type(self):
        """``Box`` needs an argument, and so does ``MyBox``."""
        with pytest.raises(CHSemanticError, match="into a type failed"):
            check_concepts({**BOX, "MyBox": "Box", **holder("MyBox")})


# --------------------------------------------------------------------------------------------------
# A type alias wherever a type is expected -- and nowhere else
# --------------------------------------------------------------------------------------------------


class TestTypeAliasInTypePositions:
    def test_as_a_property_value_domain(self):
        context = check_concepts({**BOX, "IntBox": "Box<Integer>", **holder("IntBox")})
        assert property_type(context) == "Box<Integer>"

    def test_as_a_function_argument_and_result_type(self):
        context = check_concepts(
            {
                **BOX,
                "IntBox": "Box<Integer>",
                "Unwrap": {
                    "directParents": ["FunctionReturning"],
                    "data": {
                        "templateContext": {"substitution": {"FunctionReturning:T": "Integer"}},
                        "interface": {"arg": "IntBox", "res": "Integer"},
                    },
                },
            }
        )
        assert str(context.model.functions["Unwrap"].evaluation_argument_types["arg"]) == "Box<Integer>"

    def test_as_an_instantiation_schema_custom_type(self):
        context = check_concepts(
            {
                **BOX,
                "IntBox": "Box<Integer>",
                "Wrap": {
                    "directParents": ["ValueDomain"],
                    "data": {"instantiation": {"type": "object", "properties": {"y": "IntBox"}}},
                },
            }
        )
        assert instantiation_types(context, "Wrap") == ["Box<Integer>"]

    def test_as_a_template_argument(self):
        context = check_concepts({**BOX, "IntBox": "Box<Integer>", **holder("Box<IntBox>")})
        assert property_type(context) == "Box<Box<Integer>>"

    def test_as_a_parent_template_substitution_value(self):
        check_concepts(
            {
                **BOX,
                "IntBox": "Box<Integer>",
                "BoxedIntBox": {
                    "directParents": ["Box"],
                    "data": {"templateContext": {"substitution": {"Box:T": "IntBox"}}},
                },
            }
        )

    def test_a_type_alias_equals_the_type_it_names(self):
        context = check_concepts({**BOX, "IntBox": "Box<Integer>", **holder("Box<Integer>")})
        assert context.ch.type_aliases["IntBox"] == context.model.domain_concepts["Holder"].property_types["p"]

    def test_as_a_type_value(self):
        """A ``format: Type`` string names a Type, so a type alias is a valid value for one."""
        check_concepts(
            {
                **BOX,
                "Int": "Integer",
                "TmpBox": "Box<Int>",
                "TypeValue": {
                    "directParents": ["String"],
                    "data": {"instantiation": {"type": "string", "pattern": "^s:", "format": "Type"}},
                },
                "Wrap": {
                    "directParents": ["ValueDomain"],
                    "data": {
                        "instantiation": {
                            "type": "object",
                            "properties": {
                                "c1": {"type": "TypeValue", "default": "s:TmpBox"},
                                "c2": {"type": "TypeValue", "default": "s:Box<Number>"},
                                "c3": {"type": "TypeValue", "default": "s:Int"},
                                "c4": {"type": "TypeValue", "default": "s:Integer"},
                                "c5": {"type": "TypeValue", "default": "s:Box<TmpBox>"},
                            },
                        }
                    },
                },
            }
        )


class TestTypeAliasRestrictions:
    """A type alias is saturated and is not a concept name -- both boundaries must be enforced."""

    def test_a_type_alias_can_not_be_parameterised_further(self):
        with pytest.raises(ConceptHierarchyError):
            check_concepts({**BOX, "IntBox": "Box<Integer>", **holder("IntBox<Integer>")})

    def test_a_type_alias_can_not_be_a_direct_parent(self):
        with pytest.raises(ConceptHierarchyError):
            check_concepts({**BOX, "IntBox": "Box<Integer>", "Sub": {"directParents": ["IntBox"], "data": {}}})

    def test_a_type_alias_can_not_appear_in_distinct_from(self):
        with pytest.raises(ConceptHierarchyError):
            check_concepts(
                {
                    **BOX,
                    "IntBox": "Box<Integer>",
                    "Plant": {
                        "directParents": ["Concept"],
                        "data": {"properties": {"height": "Integer"}},
                        "distinctFrom": ["IntBox"],
                    },
                }
            )

    def test_a_type_alias_is_not_a_concept(self):
        context = check_concepts({**BOX, "IntBox": "Box<Integer>"})
        assert context.ch.is_concept("IntBox") is False


# --------------------------------------------------------------------------------------------------
# Variable aliases
# --------------------------------------------------------------------------------------------------


class TestVariableAliases:
    def test_a_variable_alias_denotes_the_canonical_variable(self):
        context = check_concepts({}, instances={"origin": 0, "start": "origin"})
        assert variable_aliases(context) == {"start": "origin"}
        assert str(context.model.instances["origin"].value_type) == "Integer"

    def test_a_variable_alias_may_be_declared_before_its_target(self):
        """JSON object order is not resolution order."""
        context = check_concepts({}, instances={"start": "origin", "origin": 0})
        assert variable_aliases(context) == {"start": "origin"}

    def test_a_variable_alias_may_not_shadow_a_concept_name(self):
        with pytest.raises(CHSemanticError, match="is also a concept name"):
            check_concepts({**ANIMAL, "Beast": "Animal"}, instances={"origin": 0, "Beast": "origin"})

    def test_a_member_may_not_share_a_variable_aliass_name(self):
        """The global-variable name check must see the alias, exactly as it sees a real variable."""
        with pytest.raises(CHSemanticError, match="is also the name of a defined global variable"):
            check_concepts(
                {"A": {"directParents": ["Concept"], "data": {"properties": {"start": "Integer"}}}},
                instances={"origin": 0, "start": "origin"},
            )

    def test_an_alias_is_usable_in_an_expression(self):
        """
        The point of the kind: an alias stands wherever the variable does. The parsed expression keeps the
        name that was written and takes the canonical variable's type.
        """
        context = check_concepts(COUNTDOWN, instances={"origin": 5, "start": "origin"})
        expression = context.model.functions["Countdown"].evaluation_argument_default_value_expressions["from"]
        assert isinstance(expression.value, Variable)
        assert expression.unparsed == "start"
        assert str(expression.value.value_type) == "Integer"

    def test_an_expression_still_rejects_a_name_that_is_neither_a_variable_nor_an_alias(self):
        with pytest.raises(CHSemanticError):
            check_concepts(COUNTDOWN, instances={"origin": 5})


# --------------------------------------------------------------------------------------------------
# Chains
# --------------------------------------------------------------------------------------------------


class TestAliasChains:
    """
    An alias may name another alias. The containers record the **canonical** target, so a lookup never has
    to walk a chain, and a chain that crosses alias kinds still resolves.
    """

    def test_a_concept_alias_chain_resolves_to_the_canonical_concept(self):
        context = check_concepts({**ANIMAL, "Beast": "Animal", "Critter": "Beast"})
        assert concept_aliases(context) == {"Beast": "Animal", "Critter": "Animal"}

    def test_the_far_end_of_a_concept_alias_chain_is_usable(self):
        context = check_concepts(
            {
                **ANIMAL,
                "Beast": "Animal",
                "Critter": "Beast",
                "Dog": {"directParents": ["Critter"], "data": {"properties": {"breed": "String"}}},
            }
        )
        assert context.ch.concepts["Dog"].parents == ("Animal",)

    def test_a_variable_alias_chain_resolves_to_the_canonical_variable(self):
        context = check_concepts({}, instances={"origin": 0, "start": "origin", "begin": "start"})
        assert variable_aliases(context) == {"start": "origin", "begin": "origin"}
        assert list(context.ch.instances) == ["origin"]

    def test_a_type_alias_chain_resolves_to_the_same_type(self):
        """
        ``Boxed`` names a bare name, yet it is a **type** alias: the kind follows what the chain bottoms
        out in, which is settled by resolving in the alias graph's topological order.
        """
        context = check_concepts({**BOX, "IntBox": "Box<Integer>", "Boxed": "IntBox", **holder("Boxed")})
        assert property_type(context) == "Box<Integer>"
        assert type_aliases(context) == {"IntBox": "Box<Integer>", "Boxed": "Box<Integer>"}
        assert concept_aliases(context) == {}

    def test_a_type_alias_over_an_aliased_constructor_resolves(self):
        """The chain crosses kinds: ``IntBox`` is a type alias whose head ``MyBox`` is a concept alias."""
        context = check_concepts({**BOX, "MyBox": "Box", "IntBox": "MyBox<Integer>", **holder("IntBox")})
        assert type_aliases(context) == {"IntBox": "Box<Integer>"}
        assert property_type(context) == "Box<Integer>"

    def test_a_type_alias_over_an_aliased_template_argument_resolves(self):
        context = check_concepts({**BOX, "Int": "Integer", "IntBox": "Box<Int>", **holder("IntBox")})
        assert type_aliases(context) == {"IntBox": "Box<Integer>"}
        assert property_type(context) == "Box<Integer>"


class TestAliasesOverSeveralTemplateArguments:
    """
    A type alias resolves a whole type, not just a head: with more than one template argument, every
    position has to be resolved, independently and in order. ``Pair<A, B>`` is the smallest thing that can
    tell "resolved both arguments" apart from "resolved one and copied it".
    """

    def test_a_type_alias_over_two_arguments(self):
        context = check_concepts({**PAIR, "IntStr": "Pair<Integer, String>", **holder("IntStr")})
        assert type_aliases(context) == {"IntStr": "Pair<Integer, String>"}
        assert property_type(context) == "Pair<Integer, String>"

    def test_an_alias_in_each_argument_position(self):
        """Both arguments are concept aliases, and they must not be swapped or collapsed."""
        context = check_concepts(
            {**PAIR, "Int": "Integer", "Text": "String", "Mixed": "Pair<Int, Text>", **holder("Mixed")}
        )
        assert concept_aliases(context) == {"Int": "Integer", "Text": "String"}
        assert type_aliases(context) == {"Mixed": "Pair<Integer, String>"}

    def test_an_alias_in_only_one_argument_position(self):
        context = check_concepts({**PAIR, "Text": "String", "IntStr": "Pair<Integer, Text>", **holder("IntStr")})
        assert property_type(context) == "Pair<Integer, String>"

    def test_argument_order_is_preserved(self):
        """The two arguments differ, so a resolution that lost their order would show up here."""
        context = check_concepts(
            {
                **PAIR,
                "Int": "Integer",
                "Text": "String",
                "IntFirst": "Pair<Int, Text>",
                "TextFirst": "Pair<Text, Int>",
            }
        )
        assert type_aliases(context) == {"IntFirst": "Pair<Integer, String>", "TextFirst": "Pair<String, Integer>"}

    def test_a_two_argument_constructor_can_be_aliased(self):
        context = check_concepts({**PAIR, "MyPair": "Pair", **holder("MyPair<Integer, String>")})
        assert concept_aliases(context) == {"MyPair": "Pair"}
        assert property_type(context) == "Pair<Integer, String>"

    def test_a_two_argument_constructor_alias_still_checks_arity(self):
        with pytest.raises(CHSemanticError, match="into a type failed"):
            check_concepts({**PAIR, "MyPair": "Pair", **holder("MyPair<Integer>")})

    def test_a_type_alias_nested_several_levels_deep(self):
        """Every argument at every depth is resolved, including an alias naming another type alias."""
        context = check_concepts(
            {
                **PAIR,
                **BOX,
                "IntStr": "Pair<Integer, String>",
                "Deep": "Box<Pair<Box<Integer>, IntStr>>",
                **holder("Deep"),
            }
        )
        assert property_type(context) == "Box<Pair<Box<Integer>, Pair<Integer, String>>>"

    def test_a_type_alias_is_usable_in_one_argument_of_an_application(self):
        context = check_concepts({**PAIR, **BOX, "IntBox": "Box<Integer>", **holder("Pair<IntBox, String>")})
        assert property_type(context) == "Pair<Box<Integer>, String>"


class TestAliasCycles:
    def test_a_concept_alias_of_itself_is_rejected(self):
        with pytest.raises(CHSemanticError, match="cycle"):
            check_concepts({**ANIMAL, "Beast": "Beast"})

    def test_a_two_step_concept_alias_cycle_is_rejected(self):
        with pytest.raises(CHSemanticError, match="cycle"):
            check_concepts({**ANIMAL, "Beast": "Critter", "Critter": "Beast"})

    def test_a_three_step_concept_alias_cycle_is_rejected(self):
        with pytest.raises(CHSemanticError, match="cycle"):
            check_concepts({**ANIMAL, "Beast": "Critter", "Critter": "Fauna", "Fauna": "Beast"})

    def test_a_variable_alias_cycle_is_rejected(self):
        with pytest.raises(CHSemanticError, match="cycle"):
            check_concepts({}, instances={"start": "begin", "begin": "start"})

    def test_a_type_alias_cycle_through_the_head_is_rejected(self):
        with pytest.raises(CHSemanticError, match="cycle"):
            check_concepts({**BOX, "Boxed": "IntBox<Integer>", "IntBox": "Boxed"})

    def test_a_type_alias_cycle_through_a_template_argument_is_rejected(self):
        """``Boxed`` is defined in terms of itself, one level down in the argument list."""
        with pytest.raises(CHSemanticError, match="cycle"):
            check_concepts({**BOX, "Boxed": "Box<Boxed>"})

    def test_a_cycle_crossing_alias_kinds_is_rejected(self):
        """§4: cycle checking spans concept and type aliases together."""
        with pytest.raises(CHSemanticError, match="cycle"):
            check_concepts({**BOX, "MyBox": "IntBox", "IntBox": "MyBox<Integer>"})

    def test_a_cycle_through_a_nested_template_argument_is_rejected(self):
        """The self-reference is two levels down, not directly in the argument list."""
        with pytest.raises(CHSemanticError, match="cycle"):
            check_concepts({**BOX, "Boxed": "Box<Box<Boxed>>"})

    def test_a_three_step_cycle_through_template_arguments_is_rejected(self):
        with pytest.raises(CHSemanticError, match="cycle") as raised:
            check_concepts({**BOX, "A": "Box<B>", "B": "Box<C>", "C": "Box<A>"})
        message = str(raised.value)
        assert "'A'" in message and "'B'" in message and "'C'" in message, (
            f"the cycle should name its members, got: {message}"
        )

    def test_a_three_step_cycle_at_mixed_depths_is_rejected(self):
        """Each hop sits at a different depth and argument position; it is still one cycle."""
        with pytest.raises(CHSemanticError, match="cycle"):
            check_concepts(
                {
                    **PAIR,
                    **BOX,
                    "A": "Box<Pair<Integer, B>>",
                    "B": "Box<Box<C>>",
                    "C": "Pair<A, Integer>",
                }
            )

    def test_an_alias_outside_a_cycle_is_not_blamed_for_it(self):
        """Only the members of the cycle are named -- the old resolver listed every alias in the file."""
        with pytest.raises(CHSemanticError, match="cycle") as raised:
            check_concepts({**BOX, "Innocent": "Integer", "A": "Box<B>", "B": "Box<A>"})
        assert "Innocent" not in str(raised.value)


# --------------------------------------------------------------------------------------------------
# Error locations: `ref:` moves from the definition to the use site
# --------------------------------------------------------------------------------------------------


class TestErrorLocationsThroughAnAlias:
    """
    §6. There is no alias *definition* anymore, so the ``ref:<alias>`` annotation that used to sit on the
    cloned definition's location becomes a **use-site** annotation, appended by the resolving accessor.

    These pin the accessor contract ``ch.get_concept_definition(name, location_id) -> (definition, annotated_location)``
    proposed in §6; if that shape changes during implementation, these move with it.
    """

    def test_the_canonical_definition_location_carries_no_annotation(self):
        animal = check_concepts({**ANIMAL, "Beast": "Animal"}).ch.concepts["Animal"]
        assert not any(str(segment).startswith("ref:") for segment in animal.definition_location())
        assert animal.location_of("concepts", "Animal") == ["concepts", "Animal"]

    def test_a_canonical_lookup_returns_the_location_unchanged(self):
        context = check_concepts({**ANIMAL, "Beast": "Animal"})
        definition, location = context.ch.get_concept_definition("Animal", ["concepts", "Dog", "directParents", 0])
        assert definition is context.ch.concepts["Animal"]
        assert location == ["concepts", "Dog", "directParents", 0]

    def test_an_alias_lookup_annotates_the_use_site(self):
        context = check_concepts({**ANIMAL, "Beast": "Animal"})
        definition, location = context.ch.get_concept_definition("Beast", ["concepts", "Dog", "directParents", 0])
        assert definition is context.ch.concepts["Animal"]
        assert location == ["concepts", "Dog", "directParents", 0, "ref:Beast"]

    def test_an_alias_lookup_does_not_mutate_the_location_it_was_given(self):
        context = check_concepts({**ANIMAL, "Beast": "Animal"})
        use_site = LocationId(["concepts", "Dog", "directParents", 0])  # test with location as LocationId
        context.ch.get_concept_definition("Beast", use_site)
        assert use_site == ["concepts", "Dog", "directParents", 0]
        use_site = ["concepts", "Dog", "directParents", 0]  # test with location as list
        context.ch.get_concept_definition("Beast", use_site)
        assert use_site == ["concepts", "Dog", "directParents", 0]

    def test_a_chain_annotates_with_the_name_that_was_written(self):
        """The user wrote ``Critter``; that -- not the intermediate ``Beast`` -- is what the message shows."""
        context = check_concepts({**ANIMAL, "Beast": "Animal", "Critter": "Beast"})
        _definition, location = context.ch.get_concept_definition("Critter", ["concepts"])
        assert location == ["concepts", "ref:Critter"]


class TestAliasesInRaisedErrors:
    """
    §6, end to end: a real error raised against a name the user wrote as an alias must say so.

    The message names the *canonical* concept -- that is what the hierarchy is keyed by -- while the JSON
    at the reported path names the alias. Without the ``ref:<alias>`` segment nothing connects the two, and
    the reader is left looking at ``"Beast"`` while being told about ``Animal``.
    """

    @staticmethod
    def location_of_raised(raised) -> str:
        """The rendered location prefix of the raised error."""
        return str(raised.value).split("\n")[0]

    def test_a_parent_written_as_an_alias(self):
        with pytest.raises(CHSemanticError) as raised:
            check_concepts(
                {**ANIMAL, "Beast": "Animal", "Weird": {"directParents": ["Beast", "ValueDomain"], "data": {}}}
            )
        assert '"ref:Beast"' in self.location_of_raised(raised)

    def test_a_distinct_from_entry_written_as_an_alias(self):
        with pytest.raises(CHSemanticError, match="must be distinct from itself") as raised:
            check_concepts(
                {
                    "Animal": {
                        "directParents": ["Concept"],
                        "data": {"properties": {"age": "Integer"}},
                        "distinctFrom": ["Beast"],
                    },
                    "Beast": "Animal",
                }
            )
        assert '"ref:Beast"' in self.location_of_raised(raised)

    def test_a_distinct_group_entry_written_as_an_alias(self):
        with pytest.raises(CHSemanticError, match="is not a direct child") as raised:
            check_concepts(
                {
                    "A": {
                        "directParents": ["Concept"],
                        "data": {"properties": {"a": "Integer"}},
                        "distinctGroup": ["Pup", "C"],
                    },
                    "B": {"directParents": ["Concept"], "data": {"properties": {"b": "Integer"}}},
                    "C": {"directParents": ["A"], "data": {"properties": {"c": "Integer"}}},
                    "Pup": "B",
                }
            )
        assert '"ref:Pup"' in self.location_of_raised(raised)

    def test_a_direct_children_entry_written_as_an_alias(self):
        with pytest.raises(CHSemanticError, match="is not a direct child") as raised:
            check_concepts(
                {
                    "A": {
                        "directParents": ["Concept"],
                        "data": {"properties": {"a": "Integer"}},
                        "directChildren": ["Pup"],
                    },
                    "B": {"directParents": ["Concept"], "data": {"properties": {"b": "Integer"}}},
                    "Pup": "B",
                }
            )
        assert '"ref:Pup"' in self.location_of_raised(raised)

    def test_an_inherit_from_parent_written_as_an_alias(self):
        with pytest.raises(CHSemanticError, match="is not a direct parent of") as raised:
            check_concepts(
                {
                    "Base": {
                        "directParents": ["Concept"],
                        "data": {"properties": {"prop": {"valueDomain": "String", "default": "s:base"}}},
                    },
                    "Other": {"directParents": ["Concept"], "data": {"properties": {"o": "Integer"}}},
                    "OtherAlias": "Other",
                    "Sub": {
                        "directParents": ["Base"],
                        "data": {"properties": {"_specializations": {"prop": {"default": "inheritFrom:OtherAlias"}}}},
                    },
                }
            )
        assert '"ref:OtherAlias"' in self.location_of_raised(raised)

    def test_a_canonical_name_is_not_annotated(self):
        """The annotation marks an alias; writing the concept's own name must not add noise."""
        with pytest.raises(CHSemanticError) as raised:
            check_concepts({**ANIMAL, "Weird": {"directParents": ["Animal", "ValueDomain"], "data": {}}})
        assert "ref:" not in str(raised.value)
