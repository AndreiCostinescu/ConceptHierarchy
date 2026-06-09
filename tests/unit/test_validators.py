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

"""Unit tests for concept_hierarchy.validator.checker.check_model."""

import pytest

from concept_hierarchy.errors import CHSemanticError, CHSyntaxError
from concept_hierarchy.models import ConceptHierarchyModel
from concept_hierarchy.validator.checker import check_model


def _model(concepts, name="MyHierarchy"):
    return ConceptHierarchyModel.create_from_data({"name": name, "concepts": concepts})


class TestSyntaxValidator:
    def test_valid_passes(self):
        model = _model({"Foo": {"data": {"properties": {"x": "Integer"}}}})
        check_model(model)  # should not raise

    def test_invalid_concept_name(self):
        model = _model({"123invalid": {"data": {"properties": {"x": "Integer"}}}})
        with pytest.raises(CHSyntaxError):
            check_model(model)

    def test_invalid_property_name(self):
        model = _model({"Foo": {"data": {"properties": {"1bad": "Integer"}}}})
        with pytest.raises(CHSyntaxError):
            check_model(model)

    def test_invalid_hierarchy_name(self):
        model = _model({}, name="bad-name!")
        with pytest.raises(CHSyntaxError):
            check_model(model)


class TestSemanticValidator:
    def test_valid_hierarchy_passes(self):
        with pytest.raises(CHSemanticError):
            check_model(_model({"Base": {}, "Child": {"directParents": ["Base"]}}))
        with pytest.raises(CHSemanticError):
            check_model(_model({"Base": {"data": {}}, "Child": {"directParents": ["Base"]}}))
        with pytest.raises(CHSemanticError):
            check_model(_model({"Base": {"data": {"properties": {}}}, "Child": {"directParents": ["Base"]}}))
        with pytest.raises(CHSemanticError):
            check_model(
                _model({"Base": {"data": {"properties": {}}}, "Child": {"directParents": ["Base"], "data": {}}})
            )
        model = _model(
            {"Base": {"data": {"properties": {}}}, "Child": {"directParents": ["Base"], "data": {"properties": {}}}}
        )
        check_model(model)  # should not raise

    def test_undefined_parent(self):
        model = _model({"Child": {"directParents": ["Ghost"]}})
        with pytest.raises(CHSemanticError, match="Ghost"):
            check_model(model)

    def test_cycle_detected(self):
        model = _model({"A": {"directParents": ["B"]}, "B": {"directParents": ["A"]}})
        with pytest.raises(CHSemanticError, match="[Cc]ycle(s?)"):
            check_model(model)

    def test_reference_cycle_detected(self):
        model = _model({"Concept": {}, "A": "A"})
        with pytest.raises(CHSemanticError, match="There is a cycle in .* references"):
            check_model(model)
