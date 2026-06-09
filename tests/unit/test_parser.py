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

"""Unit tests for concept_hierarchy.validator.checker.check_model"""

import pytest

from concept_hierarchy.errors import CHSemanticError, CHSyntaxError
from concept_hierarchy.models import ConceptHierarchyModel
from concept_hierarchy.validator.checker import check_model

MINIMAL = ConceptHierarchyModel.create_from_data(
    {
        "name": "TestHierarchy",
        "concepts": {"Concept": {}},
    }
)

MINIMAL_SHORT = ConceptHierarchyModel.create_from_data({"Concept": {}})

FULL = ConceptHierarchyModel.create_from_data(
    {
        "name": "Animals",
        "metadata": {"author": "Tester"},
        "concepts": {
            "Concept": {},
            "Animal": {"directParents": ["Concept"], "description": "Base", "data": {"properties": {"age": "Integer"}}},
            "Dog": {"directParents": ["Animal"], "data": {"properties": {"breed": "String"}}},
            "ValueDomain": {"directParents": ["Concept"], "data": {"abstract": True}},
            "Integer": {"directParents": ["ValueDomain"], "data": {"instantiation": "integer"}},
            "String": {"directParents": ["ValueDomain"], "data": {"instantiation": "string"}},
        },
    }
)


class TestParseMinimal:
    model = MINIMAL

    def test_returns_model(self):
        check_model(self.model)
        assert self.model.name == "TestHierarchy"
        assert len(self.model.concepts) == 1
        assert "Concept" in self.model.concepts

    def test_metadata_defaults_to_empty(self):
        check_model(self.model)
        assert dict(self.model.metadata) == {}


class TestParseMinimalShort(TestParseMinimal):
    model = MINIMAL_SHORT

    def test_returns_model(self):
        check_model(self.model)
        assert self.model.name == "ConceptHierarchy"
        assert len(self.model.concepts) == 1
        assert "Concept" in self.model.concepts


class TestParseFull:
    model = FULL

    def test_concept_count(self):
        check_model(self.model)
        assert len(self.model.concepts) == 6

    def test_concept_names(self):
        check_model(self.model)
        assert set(self.model.concept_names()) == {"Concept", "Animal", "Dog", "ValueDomain", "Integer", "String"}

    def test_parents_set(self):
        check_model(self.model)
        dog = self.model.concepts["Dog"]
        assert dog.parents == ["Animal"]

    def test_metadata_parsed(self):
        check_model(self.model)
        assert self.model.metadata["author"] == "Tester"


class TestParseErrors:
    def test_not_a_dict(self):
        with pytest.raises(CHSyntaxError):
            check_model(ConceptHierarchyModel.create_from_data(["not", "a", "dict"]))

    def test_missing_name(self):
        model = ConceptHierarchyModel.create_from_data({"concepts": {"Concept": {}}})
        check_model(model)
        assert model.name == "ConceptHierarchy"

    def test_missing_concepts(self):
        with pytest.raises(CHSyntaxError, match="concepts"):
            check_model(ConceptHierarchyModel.create_from_data({"name": "X"}))

    def test_concepts_empty(self):
        # with pytest.raises(CHSemanticError):  # <- an empty concept hierarchy is allowed
        check_model(ConceptHierarchyModel.create_from_data({"name": "X", "concepts": {}}))

    def test_concepts_without_normal_root_but_with_it_implied(self):
        with pytest.raises(CHSemanticError):
            check_model(
                ConceptHierarchyModel.create_from_data(
                    {"name": "X", "concepts": {"ValueDomain": {"directParents": ["Concept"]}}}
                )
            )

    def test_two_roots_one_non_concept(self):
        with pytest.raises(CHSemanticError):
            check_model(
                ConceptHierarchyModel.create_from_data({"name": "X", "concepts": {"Concept": {}, "Concept2": {}}})
            )

    def test_two_non_concept_roots(self):
        check_model(
            ConceptHierarchyModel.create_from_data(
                {
                    "name": "X",
                    "concepts": {
                        "DomainConcept": {"data": {"properties": {}}},
                        "Concept2": {"data": {"properties": {}}},
                    },
                }
            )
        )

    def test_non_concept_root(self):
        with pytest.raises(CHSemanticError):
            check_model(ConceptHierarchyModel.create_from_data({"name": "X", "concepts": {"Base": {}}}))
        model = ConceptHierarchyModel.create_from_data(
            {
                "name": "X",
                "concepts": {
                    "Base": {"data": {"properties": {}}},
                },
            }
        )
        check_model(model)
        assert model.concept_names() == ["Concept", "Base"]
        assert model.concepts["Base"].parents == ["Concept"]
