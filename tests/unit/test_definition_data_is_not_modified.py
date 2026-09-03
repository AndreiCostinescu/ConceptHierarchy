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
Checking a Concept Hierarchy must not modify the definition data it was given.

A concept definition normalises its data while parsing it -- shorthand definitions are expanded, a new
variable's type is rewritten as a ``(type, bool)`` pair, keywords that have been read are consumed. All of
that happens on the definition's own copy: ``ConceptDefinition._check_data_content`` deep-copies the
``data`` block, so the caller's JSON is only ever read.

That matters because the same JSON is routinely shared: an external data file is resolved once and handed
to every concept that references it, test fixtures are reused between checks, and a caller may reasonably
check the same definition twice.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from concept_hierarchy.definitions.concept_hierarchy import ConceptHierarchyDefinition
from concept_hierarchy.validator.checker import ConceptHierarchyChecker
from tests.ch_support import build_hierarchy

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"

# A hierarchy that exercises every in-place normalisation the definitions perform.
NORMALISING_CONCEPTS: dict[str, dict] = {
    "Base": {
        "directParents": ["Concept"],
        "data": {
            "properties": {"prop": {"valueDomain": "String", "default": "s:base"}},
            # a shorthand function definition: `valueDomain`, `static` and `default` are filled in
            "functions": {"f": {}},
        },
    },
    "Sub": {
        "directParents": ["Base"],
        "data": {
            "properties": {
                "_specializations": {
                    "prop": {"default": "inheritFrom:"},
                    "_forThis": {"prop": {"confidenceHalfDecayTime": [1, "d"]}},
                }
            }
        },
    },
    "Duration": {"directParents": ["ValueDomain"], "data": {}},
}


def check(model_data: dict, external_data: object = None) -> None:
    ConceptHierarchyChecker(
        ConceptHierarchyDefinition.create_from_data(model_data), lambda _c, _f: external_data
    ).check()


class TestTheInputIsNotModified:
    def test_a_hierarchy_exercising_every_normalisation(self):
        model_data = build_hierarchy(NORMALISING_CONCEPTS)
        before = copy.deepcopy(model_data)
        check(model_data)
        assert model_data == before

    def test_a_cancelled_keyword_stays_in_the_definition_data(self):
        """The cancellation is recorded in the available-data map, not by deleting the keyword."""
        model_data = build_hierarchy(NORMALISING_CONCEPTS)
        check(model_data)
        specializations = model_data["concepts"]["Sub"]["data"]["properties"]["_specializations"]
        assert specializations["prop"] == {"default": "inheritFrom:"}
        assert "_forThis" in specializations

    def test_a_shorthand_definition_is_not_expanded_in_place(self):
        model_data = build_hierarchy(NORMALISING_CONCEPTS)
        check(model_data)
        assert model_data["concepts"]["Base"]["data"]["functions"]["f"] == {}

    def test_the_same_definition_can_be_checked_repeatedly(self):
        model_data = build_hierarchy(NORMALISING_CONCEPTS)
        for _ in range(3):
            check(model_data)

    def test_the_same_definition_can_be_checked_repeatedly_and_does_not_modify_data(self):
        model_data = build_hierarchy(NORMALISING_CONCEPTS)
        orig_model_data = copy.deepcopy(model_data)
        for _ in range(3):
            check(model_data)
            assert model_data == orig_model_data

    @pytest.mark.parametrize("example", ["animal_kingdom.json", "ownership.json"])
    def test_the_shipped_examples_are_not_modified(self, example: str):
        model_data = json.loads((EXAMPLES_DIR / example).read_text())
        external_data = json.loads((EXAMPLES_DIR / "external_animal_data.json").read_text())
        before, external_before = copy.deepcopy(model_data), copy.deepcopy(external_data)
        check(model_data, external_data)
        assert model_data == before
        assert external_data == external_before, "the resolved external data must not be modified either"

    def test_resolved_external_data_is_not_modified(self):
        """
        A concept's data may live in an external file, which the resolver returns as a plain object. It gets
        parsed exactly like inline data, so the same normalisations apply -- and the same file may be handed
        out again, for this hierarchy or another.
        """
        external_data = {
            "properties": {
                "p": {"valueDomain": "String", "default": "s:base"},
                "_specializations": {"_forThis": {"p": {"confidenceHalfDecayTime": [1, "d"]}}},
            },
            "functions": {"f": {}},
        }
        model_data = build_hierarchy(
            {
                "External": {"directParents": ["Concept"], "data": "some_file.json"},
                "Duration": {"directParents": ["ValueDomain"], "data": {}},
            }
        )
        before = copy.deepcopy(external_data)
        check(model_data, external_data)
        assert external_data == before
        assert "_forThis" in external_data["properties"]["_specializations"]
        assert external_data["functions"]["f"] == {}
