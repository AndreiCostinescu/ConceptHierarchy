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

"""Unit tests for concept_hierarchy.parser.parser."""

import pytest

from concept_hierarchy.errors import SyntaxError as CHSyntaxError
from concept_hierarchy.parser.parser import parse


MINIMAL = {
    "name": "TestHierarchy",
    "concepts": [],
}

FULL = {
    "name": "Animals",
    "metadata": {"author": "Tester"},
    "concepts": [
        {
            "name": "Animal",
            "parent": None,
            "description": "Base",
            "attributes": {"age": "int"},
        },
        {
            "name": "Dog",
            "parent": "Animal",
            "attributes": {"breed": "string"},
        },
    ],
}


class TestParseMinimal:
    def test_returns_model(self):
        model = parse(MINIMAL)
        assert model.name == "TestHierarchy"
        assert model.concepts == ()

    def test_metadata_defaults_to_empty(self):
        model = parse(MINIMAL)
        assert dict(model.metadata) == {}


class TestParseFull:
    def test_concept_count(self):
        model = parse(FULL)
        assert len(model.concepts) == 2

    def test_concept_names(self):
        model = parse(FULL)
        assert model.concept_names() == ["Animal", "Dog"]

    def test_parent_set(self):
        model = parse(FULL)
        dog = model.concepts[1]
        assert dog.parent == "Animal"

    def test_attributes_immutable(self):
        from frozendict import frozendict
        model = parse(FULL)
        assert isinstance(model.concepts[0].attributes, frozendict)

    def test_metadata_parsed(self):
        model = parse(FULL)
        assert model.metadata["author"] == "Tester"


class TestParseErrors:
    def test_not_a_dict(self):
        with pytest.raises(CHSyntaxError):
            parse(["not", "a", "dict"])

    def test_missing_name(self):
        with pytest.raises(CHSyntaxError, match="name"):
            parse({"concepts": []})

    def test_missing_concepts(self):
        with pytest.raises(CHSyntaxError, match="concepts"):
            parse({"name": "X"})

    def test_concepts_not_list(self):
        with pytest.raises(CHSyntaxError):
            parse({"name": "X", "concepts": {}})

    def test_concept_missing_name(self):
        with pytest.raises(CHSyntaxError):
            parse({"name": "X", "concepts": [{"attributes": {}}]})
