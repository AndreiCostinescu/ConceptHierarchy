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
`CHSchemaNode.apply_expanding` and `variadic_templates_of_this_unit`: the two structural pieces stage B needs.

Tested on hand-built nodes rather than through a parse, because what is being checked is the shape of the
transform -- which positions may absorb more than one node, and where a replication unit stops -- and
neither depends on a schema being well formed. `apply` is exercised alongside, since `apply_expanding`
must agree with it wherever no list changes length.
"""

from __future__ import annotations

import pytest

from concept_hierarchy.data.jsonschema.parsed_schema import CHSchemaNode, CustomConceptDataConstraint
from concept_hierarchy.data.parsers.expression_parser import _collapse_empty_containers
from concept_hierarchy.data.types.concept_hierarchy_types import (
    ExpandedVariadicTemplateVariable,
    InstantiatedType,
    TypeValue,
)
from concept_hierarchy.definitions.concept_definition_domain_concept import ForPropertyOrFunction
from concept_hierarchy.errors import LocationId


def node(**kwargs) -> CHSchemaNode:
    """A schema node with the required fields filled in and nothing else assumed."""
    kwargs.setdefault("raw", {})
    kwargs.setdefault("canonical", {})
    return CHSchemaNode(schema_owner="V", location_id=LocationId([]), **kwargs)


def leaf(name: str) -> CHSchemaNode:
    return node(is_custom_type=True, canonical={"type": name})


def group(name: str) -> ExpandedVariadicTemplateVariable:
    """The parsed form of ``"<name>..."``: the variable an expanding node carries."""
    return ExpandedVariadicTemplateVariable(clean_name=name, unique_defining_location="V")


def variadic_leaf(name: str) -> CHSchemaNode:
    """A custom-type node written as ``"<name>..."``, i.e. what stage B replicates a unit for."""
    return node(is_custom_type=True, expands_variadic=True, custom_type=group(name), canonical={"type": f"{name}..."})


def selector(restriction: str | TypeValue | None, value: CHSchemaNode | None = None) -> CustomConceptDataConstraint:
    """
    One ``props(...)`` block. A ``str`` means a variadic group of that name, ``None`` the *unrestricted*
    form -- bare ``props``, whose ``concept_restriction`` is ``None`` rather than an empty list.
    """
    if restriction is None:
        concept_restriction = None
    elif isinstance(restriction, str):
        concept_restriction = [group(restriction)]
    else:
        concept_restriction = [restriction]
    return CustomConceptDataConstraint(
        ForPropertyOrFunction.PROPERTY, False, concept_restriction, value or leaf("Integer"), False
    )


def concept(name: str) -> InstantiatedType:
    return InstantiatedType(clean_name=name, template_arguments=())


class TestApplyExpandingAgreesWithApplyWhenNothingReplicates:
    """Returning one node for one child must leave the tree shaped exactly as `apply` leaves it."""

    def test_a_branch_list_keeps_its_length(self):
        original = node(any_of=[leaf("A"), leaf("B")])
        result = original.apply_expanding(lambda child, _is_element: [child])
        assert len(result.any_of) == 2

    def test_an_items_tuple_keeps_its_length(self):
        original = node(items=[leaf("A"), leaf("B")])
        result = original.apply_expanding(lambda child, _is_element: [child])
        assert isinstance(result.items, list) and len(result.items) == 2

    def test_the_original_is_not_mutated(self):
        original = node(any_of=[leaf("A")])
        original.apply_expanding(lambda child, _is_element: [leaf("Z")])
        assert original.any_of[0].canonical == {"type": "A"}


class TestOnlyListValuedContainersAbsorbSeveral:
    @pytest.mark.parametrize("keyword", ["all_of", "any_of", "one_of"])
    def test_a_branch_list_absorbs_several(self, keyword):
        original = node(**{keyword: [leaf("A")]})
        result = original.apply_expanding(lambda child, _is_element: [leaf("X"), leaf("Y"), leaf("Z")])
        assert len(getattr(result, keyword)) == 3

    def test_an_items_tuple_absorbs_several(self):
        original = node(items=[leaf("A")])
        result = original.apply_expanding(lambda child, _is_element: [leaf("X"), leaf("Y")])
        assert [c.canonical["type"] for c in result.items] == ["X", "Y"]

    def test_the_results_are_concatenated_in_order(self):
        """Two elements, each replicating, splice in place rather than being grouped."""
        original = node(items=[leaf("A"), leaf("B")])
        produced = {"A": [leaf("A1"), leaf("A2")], "B": [leaf("B1")]}
        result = original.apply_expanding(lambda child, _is_element: produced[child.canonical["type"]])
        assert [c.canonical["type"] for c in result.items] == ["A1", "A2", "B1"]

    @pytest.mark.parametrize("field_name", ["contains", "not_", "if_", "property_names"])
    def test_a_single_valued_position_refuses_several(self, field_name):
        original = node(**{field_name: leaf("A")})
        with pytest.raises(AssertionError, match="position that holds only one"):
            original.apply_expanding(lambda child, _is_element: [leaf("X"), leaf("Y")])

    def test_the_items_schema_form_refuses_several(self):
        """``"items": <schema>`` constrains every element with one schema; it has no positions to splice."""
        original = node(items=leaf("A"))
        with pytest.raises(AssertionError, match="position that holds only one"):
            original.apply_expanding(lambda child, _is_element: [leaf("X"), leaf("Y")])

    def test_a_properties_value_refuses_several(self):
        original = node(properties={"k": leaf("A")})
        with pytest.raises(AssertionError, match="position that holds only one"):
            original.apply_expanding(lambda child, _is_element: [leaf("X"), leaf("Y")])


class TestWhereAReplicationUnitStops:
    """
    A unit is one element of an expandable container and stops where the next container begins, so that
    ``"items": [{"anyOf": ["T1...", "T2..."]}]`` is two units rather than one
    (`documentation/TODO_VARIADIC_SCHEMA_EXPANSION.md` 2.3).
    """

    def test_the_node_itself_counts(self):
        element = node(expands_variadic=True, is_custom_type=True)
        assert list(element.variadic_templates_of_this_unit()) == [element]

    def test_a_variadic_nested_through_ordinary_keywords_counts(self):
        deep = node(expands_variadic=True, is_custom_type=True)
        element = node(properties={"k": node(properties={"j": deep})})
        assert list(element.variadic_templates_of_this_unit()) == [deep]

    @pytest.mark.parametrize("keyword", ["all_of", "any_of", "one_of"])
    def test_a_variadic_below_a_nested_container_belongs_to_that_container(self, keyword):
        deep = node(expands_variadic=True, is_custom_type=True)
        element = node(**{keyword: [deep]})
        assert list(element.variadic_templates_of_this_unit()) == []

    def test_a_variadic_below_a_nested_items_tuple_belongs_to_it(self):
        deep = node(expands_variadic=True, is_custom_type=True)
        element = node(items=[deep])
        assert list(element.variadic_templates_of_this_unit()) == []

    def test_the_items_schema_form_is_not_a_container_so_the_unit_continues(self):
        deep = node(expands_variadic=True, is_custom_type=True)
        element = node(items=deep)
        assert list(element.variadic_templates_of_this_unit()) == [deep]

    def test_a_defs_subtree_is_skipped(self):
        """Its subtree is shared by the referrers, so it is not part of anybody's unit."""
        deep = node(expands_variadic=True, is_custom_type=True)
        element = node(definitions={"e": deep})
        assert list(element.variadic_templates_of_this_unit()) == []

    def test_two_variadics_of_one_unit_are_both_found(self):
        first = node(expands_variadic=True, is_custom_type=True)
        second = node(expands_variadic=True, is_custom_type=True)
        element = node(properties={"a": first, "b": second})
        assert list(element.variadic_templates_of_this_unit()) == [first, second]


class TestWhichGroupsAUnitNames:
    """
    `variadic_template_parameters_of_this_unit` is what both the one-group check and stage B ask, so it has to see
    **every** way a group can be named in a unit: a subschema node and a `props`/`funcs` argument alike.
    A version that only walked the schema nodes let the two-group case through in a selector and then
    zipped one group while splicing the other.
    """

    def test_a_schema_node_names_its_group(self):
        element = node(properties={"k": variadic_leaf("T")})
        assert element.variadic_template_parameters_of_this_unit() == ["T"]

    def test_a_selector_argument_names_its_group(self):
        element = node(custom_concept_data_constraints=[selector("T")])
        assert element.variadic_template_parameters_of_this_unit() == ["T"]

    def test_a_selector_below_the_element_counts_too(self):
        element = node(properties={"k": node(custom_concept_data_constraints=[selector("T")])})
        assert element.variadic_template_parameters_of_this_unit() == ["T"]

    def test_a_repeated_group_is_named_once(self):
        element = node(properties={"a": variadic_leaf("T"), "b": variadic_leaf("T")})
        assert element.variadic_template_parameters_of_this_unit() == ["T"]

    def test_two_groups_are_both_named(self):
        element = node(properties={"k": variadic_leaf("T1")}, custom_concept_data_constraints=[selector("T2")])
        assert sorted(element.variadic_template_parameters_of_this_unit()) == ["T1", "T2"]

    def test_a_selector_below_a_nested_container_belongs_to_that_container(self):
        """The unit boundary is the same one the schema nodes obey; a selector is not an exception to it."""
        element = node(any_of=[node(custom_concept_data_constraints=[selector("T")])])
        assert element.variadic_template_parameters_of_this_unit() == []

    def test_an_unrestricted_selector_names_nothing(self):
        """A bare ``props`` has no argument list at all -- ``None``, which is not an empty one."""
        element = node(custom_concept_data_constraints=[selector(None)])
        assert element.variadic_template_parameters_of_this_unit() == []

    def test_a_plain_concept_in_a_selector_names_nothing(self):
        element = node(custom_concept_data_constraints=[selector(concept("Pet"))])
        assert element.variadic_template_parameters_of_this_unit() == []


class TestIsExpandableContainerChild:
    """The relative paths come from `iter_children`, but the predicate is also asked about a node's own
    (empty) path while walking, and must answer rather than raise."""

    def test_an_empty_path_is_not_a_container_element(self):
        assert node().is_expandable_container_child(()) is False

    @pytest.mark.parametrize("keyword", ["allOf", "anyOf", "oneOf"])
    def test_a_branch_list_is_a_container(self, keyword):
        assert node().is_expandable_container_child((keyword, 0)) is True

    def test_an_items_tuple_is_a_container(self):
        assert node(items=[leaf("A")]).is_expandable_container_child(("items", 0)) is True

    def test_the_items_schema_form_is_not(self):
        assert node(items=leaf("A")).is_expandable_container_child(("items",)) is False

    @pytest.mark.parametrize("keyword", ["properties", "contains", "not", "definitions"])
    def test_everything_else_is_not(self, keyword):
        assert node().is_expandable_container_child((keyword, "k")) is False


class TestAConstraintKnowsWhetherItIsTemplateDependent:
    """
    `CustomConceptDataConstraint.is_template_dependent` has to read the argument list as well as the value
    schema -- ``props(T...)`` is template-dependent even when the value under it is not. ``None`` there is
    the unrestricted form, which is not a list and must be told apart before it is iterated.
    """

    def test_an_unrestricted_selector_is_not_template_dependent(self):
        assert selector(None).is_template_dependent is False

    def test_a_plain_concept_is_not_template_dependent(self):
        assert selector(concept("Pet")).is_template_dependent is False

    def test_a_variadic_group_is_template_dependent(self):
        assert selector("T").is_template_dependent is True

    def test_a_template_dependent_value_is_enough_on_its_own(self):
        constraint = selector(concept("Pet"))
        constraint.value = variadic_leaf("T")
        assert constraint.is_template_dependent is True


class TestCollapsingAnEmptiedDisjunction:
    """
    An `anyOf`/`oneOf` the expansion emptied becomes the boolean `false`, because an empty list there is
    not merely wrong but *silent*: `_parse_any_of` is guarded by ``if node.any_of``, so no branches means
    no check and the node accepts everything.

    Which means the collapse has to tell "declared and then emptied" from "never written", and the field
    it reads for that must be the **canonical** form. ``raw`` is typed ``object`` -- a bool, string, list
    or dict -- so a membership test against it is a substring test as often as a key lookup.
    """

    def test_an_emptied_disjunction_collapses(self):
        collapsed = _collapse_empty_containers(node(canonical={"anyOf": []}, raw={"anyOf": ["T..."]}))
        assert collapsed.canonical is False and collapsed.is_boolean_schema

    def test_an_emptied_one_of_collapses_too(self):
        collapsed = _collapse_empty_containers(node(canonical={"oneOf": []}, raw={"oneOf": ["T..."]}))
        assert collapsed.canonical is False

    def test_a_node_that_never_declared_one_is_left_alone(self):
        original = node(canonical={"type": "object"}, raw={"type": "object"})
        assert _collapse_empty_containers(original) is original

    def test_a_non_empty_disjunction_is_left_alone(self):
        original = node(canonical={"anyOf": [True]}, raw={"anyOf": ["T..."]}, any_of=[leaf("A")])
        assert _collapse_empty_containers(original) is original

    def test_a_conjunction_is_not_collapsed(self):
        """``allOf``'s identity is **true**, and an empty `all_of` already imposes nothing."""
        original = node(canonical={"allOf": []}, raw={"allOf": ["T..."]})
        assert _collapse_empty_containers(original) is original

    def test_a_boolean_schema_is_left_alone(self):
        original = node(canonical=True)
        assert _collapse_empty_containers(original) is original

    def test_a_raw_that_is_not_a_dict_does_not_break_it(self):
        """Shorthand leaves a string or a list in ``raw``; neither may be probed for a keyword."""
        for raw in ("integer", ["Integer", "Addr"], True):
            assert _collapse_empty_containers(node(canonical={"type": "integer"}, raw=raw)) is not None

    def test_the_text_of_a_raw_value_is_not_a_declaration(self):
        """A string that merely *contains* the keyword declared nothing, and must not collapse the node."""
        original = node(canonical={"type": "string"}, raw="anyOfTheseWillDo")
        assert _collapse_empty_containers(original) is original
