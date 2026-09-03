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
        # an alias: a second name for Dog, not a second definition of it -- so it never appears in
        # ``concepts`` and has no definition location of its own
        "DogClone": "Dog",
        "ValueDomain": {"directParents": ["Concept"], "data": {}, "abstract": True},
        "Integer": {"directParents": ["ValueDomain"], "data": {}},
        "Duration": {"directParents": ["ValueDomain"], "data": {}},
        "String": {
            "directParents": ["ValueDomain"],
            "data": {"defaultSerialization": "string"},
        },
        "TypeName": {"directParents": ["String"], "data": {}},
        "ClosedInterval": {
            "directParents": ["ValueDomain"],
            "data": {"templateContext": {"order": ["T"], "T": "Integer"}},
        },
        "CustomFunction": {
            "directParents": ["ValueDomain"],
            "data": {},
        },
        "InstanceBase": {
            "directParents": ["ValueDomain"],
            "data": {
                "instantiation": {
                    "type": "object",
                    "properties": {
                        "concepts": {"type": "List<ConceptValue>", "default": []},
                        "properties": "ConceptParameters",
                        "instanceName": "string",
                    },
                }
            },
        },
        "List": {
            "directParents": ["ValueDomain"],
            "data": {"templateContext": ["T"], "instantiation": {"type": "array", "items": "T"}},
        },
        "ConceptValue": {
            "directParents": ["String"],
            "data": {"instantiation": {"type": "string", "pattern": "^s:", "format": "Concept"}},
        },
        "ConceptParameters": {
            "directParents": ["ValueDomain"],
            "data": {"instantiation": {"type": "object", "properties": [["props", "x", True], ["funcs", "x", True]]}},
        },
        "FunctionComposition": {
            "directParents": ["ValueDomain"],
            "data": {
                "instantiation": {
                    "type": "object",
                    "minProperties": 1,
                    "maxProperties": 1,
                    "propertyNames": {"type": "string", "format": "Type", "constraint": "Function"},
                    "additionalProperties": {"type": "object", "properties": "args", "additionalProperties": False},
                }
            },
        },
        "Function": {"directParents": ["ValueDomain"], "abstract": True, "data": {}},
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
            assert c.location_of(c_name) == expected_value

    def test_location_of_concept_names_composite(self):
        concepts = _model().concepts
        for c_name, c in concepts.items():
            expected_value = ["concepts", c_name]
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
            expected_value += ["data"]
            assert c.location_of(ConceptDefinition.concept_definition_data) == expected_value

    def test_location_of_concept_data_composite(self):
        concepts = _model().concepts
        for c_name, c in concepts.items():
            expected_value = ["concepts", c_name]
            expected_value += ["data"]
            assert c.location_of(c_name, ConceptDefinition.concept_definition_data) == expected_value
            assert c.location_of("concepts", c_name, ConceptDefinition.concept_definition_data) == expected_value

    def test_location_of_template_arguments(self):
        concepts = _model().concepts
        for c_name, c in concepts.items():
            if not isinstance(c, HiddenImplementationDefinition):
                continue
            expected_value = ["concepts", c_name]
            expected_value += ["data"]
            if c_name == "Animal":
                expected_value += ["ext:external_animal_data.json"]
            expected_value += ["templateContext"]
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


class TestAliasLocations:
    """
    ``DogClone`` is an alias of ``Dog``: a name, not an entity. It has no definition of its own and so no
    definition location -- the ``ref:<alias>`` annotation it used to carry as a cloned concept moves to the
    use site (§6 of ``documentation/TODO_ALIASES_IMPLEMENTATION.md``, exercised in
    ``tests/unit/test_aliases.py``).
    """

    def test_an_alias_has_no_concept_entry(self):
        model = _model()
        assert "DogClone" not in model.concepts
        assert model.concept_aliases["DogClone"] == "Dog"

    def test_the_aliased_concepts_locations_are_unannotated(self):
        dog = _model().concepts["Dog"]
        assert dog.location_of("concepts", "Dog") == ["concepts", "Dog"]
        assert dog.location_of(ConceptDefinition.concept_definition_data) == ["concepts", "Dog", "data"]

    def test_the_alias_annotates_the_site_that_uses_it(self):
        """The alias shows up as ``ref:DogClone`` where it is *written*, not where ``Dog`` is defined."""
        model = _model()
        definition, location = model.get_concept_definition(
            "DogClone", ["concepts", "Puppy", ConceptDefinition.concept_direct_parents, 0]
        )
        assert definition is model.concepts["Dog"]
        assert location == ["concepts", "Puppy", ConceptDefinition.concept_direct_parents, 0, "ref:DogClone"]
