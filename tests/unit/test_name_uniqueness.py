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
Tests for the name-uniqueness rules of ``checker.check_after_parsing_concepts``.

The rules, per the comment on that block, are: property names are unique across the hierarchy, function
names are unique across the hierarchy, property and function names are distinct from each other, and
neither may collide with a global variable's name.

**The duplicate-name tests in this module fail on purpose.** ``ConceptHierarchyModel`` declares
``all_domain_concept_properties`` and ``all_domain_concept_functions`` (``prop_name -> defining concept``),
and ``checker.py`` reads both in four "is defined in multiple places!" errors -- but nothing ever writes to
either map, so all four errors are unreachable and duplicates are silently accepted. They are left failing
rather than xfailed so the gap stays visible until the maps are populated.

The global-variable collision checks in the same block *do* work, because ``ch.instances`` is populated;
those tests pass and guard the half of the rule that already holds.
"""

from __future__ import annotations

import pytest

from concept_hierarchy.errors import CHSemanticError
from tests.ch_support import check_concepts


def with_property(concept: str, prop_name: str, prop_type: str = "Integer", parents=("Concept",)) -> dict:
    return {concept: {"directParents": list(parents), "data": {"properties": {prop_name: prop_type}}}}


def with_function(concept: str, func_name: str, parents=("Concept",)) -> dict:
    return {
        concept: {
            "directParents": list(parents),
            "data": {"functions": {func_name: {"valueDomain": "CustomFunction"}}},
        }
    }


class TestGlobalVariableCollisions:
    """These checks work today: a member may not share its name with a global variable."""

    def test_a_property_may_not_share_a_global_variables_name(self):
        with pytest.raises(CHSemanticError, match="is also the name of a defined global variable"):
            check_concepts(with_property("A", "g"), instances={"g": 1})

    def test_a_function_may_not_share_a_global_variables_name(self):
        with pytest.raises(CHSemanticError, match="is also the name of a defined global variable"):
            check_concepts(with_function("A", "g"), instances={"g": 1})

    def test_an_unrelated_global_variable_name_is_fine(self):
        check_concepts(with_property("A", "legs"), instances={"g": 1})


class TestDuplicateMemberNames:
    """
    Every test in this class is expected to FAIL until ``all_domain_concept_properties`` and
    ``all_domain_concept_functions`` are populated -- see the module docstring. Each names the error that
    ``checker.py`` already contains and would raise once those maps are filled in.
    """

    def test_a_property_name_may_not_be_reused_in_another_concept(self):
        with pytest.raises(CHSemanticError, match="The property dup is defined in multiple places"):
            check_concepts({**with_property("A", "dup"), **with_property("B", "dup", "String")})

    def test_a_function_name_may_not_be_reused_in_another_concept(self):
        with pytest.raises(CHSemanticError, match="The function dup is defined in multiple places"):
            check_concepts({**with_function("A", "dup"), **with_function("B", "dup")})

    def test_a_property_name_may_not_collide_with_a_function_name(self):
        with pytest.raises(CHSemanticError, match="is defined in multiple places"):
            check_concepts({**with_property("A", "dup"), **with_function("B", "dup")})

    def test_a_subconcept_may_not_redefine_an_inherited_property(self):
        """
        The specialization regime exists so that a subconcept refines an inherited property rather than
        redefining it, so this is the same duplicate-name violation as between unrelated concepts.
        """
        with pytest.raises(CHSemanticError, match="is defined in multiple places"):
            check_concepts({**with_property("A", "dup"), **with_property("B", "dup", "String", parents=["A"])})
