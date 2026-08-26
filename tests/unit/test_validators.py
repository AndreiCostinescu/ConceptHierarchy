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

from concept_hierarchy.errors import CHSemanticError, CHSyntaxError, CHWarning
from concept_hierarchy.models import ConceptHierarchyModel
from concept_hierarchy.validator.checker import check_model


def _model(concepts, name="MyHierarchy"):
    return ConceptHierarchyModel.create_from_data({"name": name, "concepts": concepts})


class TestSyntaxValidator:
    def test_valid_passes(self):
        model = _model(
            {
                "Foo": {"data": {"properties": {"x": "Integer"}}},
                "Integer": {"directParents": ["ValueDomain"], "data": {"defaultSerialization": "integer"}},
                "ValueDomain": {"data": {}},
            }
        )
        check_model(model)  # should not raise

    def test_property_definition_without_value_domain(self):
        with pytest.raises(
            CHSemanticError,
            match=r"ParsedType 'ValueDomain' is not a template variable \(in this context\) nor a concept!",
        ):
            model = _model({"Foo": {"data": {"properties": {"x": "Integer"}}}, "Integer": {"data": {"properties": {}}}})
            check_model(model)  # should raise because ValueDomain is not defined

    def test_property_definition_with_no_value_domain_type(self):
        with pytest.raises(
            CHSemanticError, match="The defined ValueDomain of property x is not a subtype of ValueDomain!"
        ):
            model = _model(
                {
                    "Foo": {"data": {"properties": {"x": "Integer"}}},
                    "Integer": {"data": {"properties": {}}},
                    "ValueDomain": {"data": {}},
                }
            )
            check_model(model)  # should raise because Integer is not a ValueDomain

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
        with pytest.raises(CHSyntaxError):
            check_model(_model({"Base": {}, "Child": {"directParents": ["Base"]}}))
        with pytest.raises(CHSyntaxError):
            check_model(_model({"Base": {"data": {}}, "Child": {"directParents": ["Base"]}}))
        with pytest.raises(CHSyntaxError):
            check_model(_model({"Base": {"data": {"properties": {}}}, "Child": {"directParents": ["Base"]}}))
        # Concepts without any defined data are allowed; the below should not raise
        with pytest.warns(CHWarning, match="Found a domain concept with no data defined: 'Child'"):
            check_model(
                _model({"Base": {"data": {"properties": {}}}, "Child": {"directParents": ["Base"], "data": {}}})
            )
        model = _model(
            {"Base": {"data": {"properties": {}}}, "Child": {"directParents": ["Base"], "data": {"properties": {}}}}
        )
        check_model(model)  # should not raise

    def test_undefined_parent(self):
        model = _model({"Child": {"directParents": ["Ghost"], "data": {"properties": {}}}})
        with pytest.raises(
            CHSemanticError, match="The parent 'Ghost' of concept 'Child' is not defined in the hierarchy"
        ):
            check_model(model)

    def test_cycle_detected(self):
        model = _model({"A": {"directParents": ["B"], "data": {}}, "B": {"directParents": ["A"], "data": {}}})
        with pytest.raises(CHSemanticError, match="[Cc]ycle(s?)"):
            check_model(model)

    def test_undefined_reference(self):
        model = _model({"Concept": {}, "A": "B"})
        with pytest.raises(
            CHSemanticError, match="The referenced concept 'B' of A does not exist in the Concept Hierarchy!"
        ):
            check_model(model)

    def test_single_parent_string(self):
        model = _model({"Concept": {}, "ValueDomain": {"directParents": "Concept"}})
        with pytest.raises(
            CHSyntaxError,
            match="Direct parents of the concept ValueDomain must be a JSON array of strings, not 'Concept'!",
        ):
            check_model(model)

    def test_value_domain_reference(self):
        model = _model(
            {"Concept": {}, "ValueDomain": {"directParents": ["Concept"], "data": {}}, "Type": "ValueDomain"}
        )
        with pytest.warns(CHWarning, match="Found a domain concept with no data defined: 'Type'"):
            check_model(model)
        assert "Type" in model.domain_concepts
        assert "Type" not in model.value_domains

    def test_reference_chain(self):
        model = _model({"Concept": {}, "A": {"directParents": ["Concept"], "data": {"properties": {}}}, "B": "A"})
        check_model(model)

    def test_root_reference_chain(self):
        model = _model({"Concept": {}, "A": "Concept"})
        with pytest.raises(CHSemanticError, match=r"Concept Hierarchy has multiple roots: \['Concept', 'A'\]"):
            check_model(model)

    def test_long_root_reference_chain(self):
        model = _model({"Concept": {}, "A": "Concept", "B": "A"})
        with pytest.raises(CHSemanticError, match=r"Concept Hierarchy has multiple roots: \['Concept', 'A', 'B'\]"):
            check_model(model)

    def test_self_reference(self):
        model = _model({"Concept": {}, "A": "A"})
        with pytest.raises(CHSemanticError, match="There is a cycle in .* references"):
            check_model(model)

    def test_reference_cycle_detected(self):
        model = _model({"Concept": {}, "A": "B", "B": "A"})
        with pytest.raises(CHSemanticError, match="There is a cycle in .* references"):
            check_model(model)
