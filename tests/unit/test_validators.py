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

"""Unit tests for concept_hierarchy validators."""

import pytest

from concept_hierarchy.errors import SemanticError, SyntaxError as CHSyntaxError
from concept_hierarchy.parser.parser import parse
from concept_hierarchy.validator.semantics import check_semantics
from concept_hierarchy.validator.syntax import check_syntax


def _model(concepts, name="MyHierarchy"):
    return parse({"name": name, "concepts": concepts})


class TestSyntaxValidator:
    def test_valid_passes(self):
        model = _model([{"name": "Foo", "attributes": {"x": "int"}}])
        check_syntax(model)  # should not raise

    def test_invalid_concept_name(self):
        model = _model([{"name": "123invalid"}])
        with pytest.raises(CHSyntaxError):
            check_syntax(model)

    def test_invalid_attribute_name(self):
        model = _model([{"name": "Foo", "attributes": {"1bad": "int"}}])
        with pytest.raises(CHSyntaxError):
            check_syntax(model)

    def test_invalid_hierarchy_name(self):
        model = _model([], name="bad-name!")
        with pytest.raises(CHSyntaxError):
            check_syntax(model)


class TestSemanticValidator:
    def test_valid_hierarchy_passes(self):
        model = _model(
            [
                {"name": "Base"},
                {"name": "Child", "parent": "Base"},
            ]
        )
        check_semantics(model)  # should not raise

    def test_undefined_parent(self):
        model = _model([{"name": "Child", "parent": "Ghost"}])
        with pytest.raises(SemanticError, match="Ghost"):
            check_semantics(model)

    def test_duplicate_names(self):
        model = _model([{"name": "Foo"}, {"name": "Foo"}])
        with pytest.raises(SemanticError, match="Duplicate"):
            check_semantics(model)

    def test_cycle_detected(self):
        # Manually build a cyclic model (parser won't produce this, but
        # the validator must catch it).
        from frozendict import frozendict
        from concept_hierarchy.models import Concept, ConceptHierarchyModel

        a = Concept("A", parent="B", attributes=frozendict())
        b = Concept("B", parent="A", attributes=frozendict())
        model = ConceptHierarchyModel("Cyclic", (a, b), frozendict())
        with pytest.raises(SemanticError, match="[Cc]ircular"):
            check_semantics(model)
