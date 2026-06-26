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

"""Unit tests for the location_of function inside ConceptHierarchyDefinition."""

import pytest

from concept_hierarchy.definitions.concept_definition import ConceptDefinition
from concept_hierarchy.definitions.concept_definition_hidden_implementation import HiddenImplementationDefinition
from concept_hierarchy.models import ConceptHierarchyModel
from concept_hierarchy.validator.checker import ConceptHierarchyChecker

model_data = {
    "name": "AnimalKingdom",
    "metadata": {"version": "v1.0", "author": "Your Name"},
    "concepts": {
        "Concept": {"data": None},
        "Animal": {"directParents": ["Concept"], "data": "external_animal_data.json"},
        "Dog": {
            "directParents": ["Animal"],
            "data": {
                "properties": {
                    "breed": {"valueDomain": "String", "confidenceHalfDecayTime": [1, "y"]},
                    "_specializations": {
                        "age": {"confidenceHalfDecayTime": [1, "d"], "hooks": "inheritFrom:"},
                        "_forThis": {"breed": {"confidenceHalfDecayTime": "inheritFrom:"}},
                    },
                },
                "functions": {
                    "f1": {},
                    "f2": {"Add<Number>": {"arg1": 2, "arg2": 2}},
                    "f3": {
                        "interface": {"arg": "Number", "res": "Number"},
                        "procedure": {"Return<Number>": {"what": {"Add<Number>": {"arg1": "arg", "arg2": 2}}}},
                    },
                    "f4": {
                        "valueDomain": "CustomFunction",
                        "static": True,
                        "default": {
                            "interface": {"arg": "Number", "res": "Number"},
                            "procedure": {"Return<Number>": {"what": {"Add<Number>": {"arg1": "arg", "arg2": 2}}}},
                        },
                    },
                    "_specializations": {
                        "_forThis": {"f1": {"Return<Number>": {"Add<Number>": {"arg1": 2, "arg2": 2}}}}
                    },
                },
            },
        },
        "DogClone": "Dog",
        "ValueDomain": {"directParents": ["Concept"], "data": {"abstract": True}},
        "Integer": {"directParents": ["ValueDomain"], "data": {}},
        "Duration": {"directParents": ["ValueDomain"], "data": {}},
        "String": {
            "directParents": ["ValueDomain"],
            "data": {"defaultSerialization": "string"},
        },
        "TypeName": {"directParents": ["String"], "data": {}},
        "ClosedInterval": {
            "directParents": ["ValueDomain"],
            "data": {"templateArguments": {"order": ["T"], "T": "Integer"}},
        },
        "CustomFunction": {
            "directParents": ["ValueDomain"],
            "data": {},
        },
    },
    "instances": {
        "MyAnimal": {"InstanceBase": {"instanceName": "MyAnimal", "concepts": ["Animal"], "properties": {"age": 2}}}
    },
}

external_data: dict[str, object] = {
    "Animal": {
        "properties": {
            "age": "Integer",
            "name": "String",
            "type": {"valueDomain": "String"},
            "description": {
                "valueDomain": "String",
                "description": "Test property",
                "hooks": {},
                "confidenceHalfDecayTime": [2, "d"],
            },
            "_specializations": {
                "_forThis": {"age": {"constraint": {"ClosedInterval<Integer>": {"min": 0, "max": 14}}}}
            },
        }
    }
}


def get_external_data(concept_name: str, external_data_path: str):
    if concept_name == "Animal" and external_data_path == "external_animal_data.json":
        return external_data[concept_name]
    raise RuntimeError(
        f"Can not resolve external data {external_data_path!r} for concept {concept_name!r} during test!!"
    )


def _model() -> ConceptHierarchyModel:
    model = ConceptHierarchyModel.create_from_data(model_data)
    ConceptHierarchyChecker(model, get_external_data).check()
    return model


class TestLocationOf:
    def test_location_of_concepts(self):
        concepts = _model().concepts
        for c_name, c in concepts.items():
            assert c.location_of("concepts") == ["concepts"]

    def test_not_existing_location(self):
        concepts = _model().concepts
        for c_name, c in concepts.items():
            with pytest.raises(
                RuntimeError, match=rf"Keyword\(s\) \('instance',\) not found in the .* definition of {c_name}"
            ):
                c.location_of("instance")

    def test_not_existing_location_partial_match(self):
        concepts = _model().concepts
        for c_name, c in concepts.items():
            with pytest.raises(
                RuntimeError,
                match=rf"Keyword\(s\) \('{c_name}', 'concepts'\) not found in the .* definition of {c_name}",
            ):
                c.location_of(c_name, "concepts")

    def test_not_existing_location_not_continuous_path(self):
        concepts = _model().concepts
        for c_name, c in concepts.items():
            with pytest.raises(
                RuntimeError,
                match=rf"Keyword\(s\) \('concepts', '{ConceptDefinition.concept_definition_data}'\) not found in the .*"
                rf"definition of {c_name}",
            ):
                c.location_of("concepts", ConceptDefinition.concept_definition_data)

    def test_location_of_concept_names(self):
        concepts = _model().concepts
        for c_name, c in concepts.items():
            expected_value = ["concepts", c_name]
            if c_name == "DogClone":
                expected_value += ["ref:Dog"]
            assert c.location_of(c_name) == expected_value

    def test_location_of_concept_names_composite(self):
        concepts = _model().concepts
        for c_name, c in concepts.items():
            expected_value = ["concepts", c_name]
            if c_name == "DogClone":
                expected_value += ["ref:Dog"]
            assert c.location_of("concepts", c_name) == expected_value

    def test_location_of_concept_direct_parents(self):
        concepts = _model().concepts
        for c_name, c in concepts.items():
            if c_name == "Concept":
                with pytest.raises(
                    RuntimeError,
                    match=rf"Keyword\(s\) \('{ConceptDefinition.concept_direct_parents}',\) not found in the Domain"
                    rf" Concept definition of {c_name}",
                ):
                    c.location_of(ConceptDefinition.concept_direct_parents)
            else:
                expected_value = ["concepts", c_name]
                if c_name == "DogClone":
                    expected_value += ["ref:Dog"]
                expected_value += [ConceptDefinition.concept_direct_parents]
                assert c.location_of(ConceptDefinition.concept_direct_parents) == expected_value

    def test_location_of_concept_description(self):
        concepts = _model().concepts
        for c_name, c in concepts.items():
            with pytest.raises(
                RuntimeError,
                match=rf"Keyword\(s\) \('{c_name}', '{ConceptDefinition.concept_description}'\) not found in the .* "
                rf"definition of {c_name}",
            ):
                # Use the concept_name before the description to not match
                #  any properties, for example, that are also named "description"!
                c_name, c.location_of(c_name, ConceptDefinition.concept_description).print()
        assert concepts["Animal"].location_of(ConceptDefinition.concept_description) == [
            "concepts",
            "Animal",
            "data",
            "ext:external_animal_data.json",
            "properties",
            "description",
        ]

    def test_location_of_concept_data(self):
        concepts = _model().concepts
        for c_name, c in concepts.items():
            expected_value = ["concepts", c_name]
            if c_name == "DogClone":
                expected_value += ["ref:Dog"]
            expected_value += ["data"]
            assert c.location_of(ConceptDefinition.concept_definition_data) == expected_value

    def test_location_of_concept_data_composite(self):
        concepts = _model().concepts
        for c_name, c in concepts.items():
            expected_value = ["concepts", c_name]
            if c_name == "DogClone":
                expected_value += ["ref:Dog"]
            expected_value += ["data"]
            assert c.location_of(c_name, ConceptDefinition.concept_definition_data) == expected_value
            assert c.location_of("concepts", c_name, ConceptDefinition.concept_definition_data) == expected_value

    def test_location_of_template_arguments(self):
        concepts = _model().concepts
        for c_name, c in concepts.items():
            if not isinstance(c, HiddenImplementationDefinition):
                continue
            expected_value = ["concepts", c_name]
            if c_name == "DogClone":
                expected_value += ["ref:Dog"]
            expected_value += ["data"]
            if c_name == "Animal":
                expected_value += ["ext:external_animal_data.json"]
            expected_value += ["templateArguments"]
            if c.is_templatable() or any(
                concepts[p].is_templatable()
                for p in c.parents
                if isinstance(concepts[p], HiddenImplementationDefinition)
            ):
                assert c.location_of(HiddenImplementationDefinition.hidden_template_arguments) == expected_value
            else:
                with pytest.raises(
                    RuntimeError,
                    match=rf"Keyword\(s\) \('{HiddenImplementationDefinition.hidden_template_arguments}',\) not "
                    rf"found in the .* definition of {c_name}",
                ):
                    c.location_of(HiddenImplementationDefinition.hidden_template_arguments)
