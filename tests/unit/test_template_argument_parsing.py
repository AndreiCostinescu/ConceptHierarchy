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
Unit tests for parsing template arguments in a ConceptHierarchyDefinition
including verifying template argument constraint formulae.
"""

from copy import deepcopy

import pytest

from concept_hierarchy.data.concept_hierarchy import ConceptHierarchy
from concept_hierarchy.definitions.concept_definition_hidden_implementation import HiddenImplementationDefinition
from concept_hierarchy.definitions.concept_hierarchy import ConceptHierarchyDefinition
from concept_hierarchy.errors import CHSemanticError, CHSyntaxError
from concept_hierarchy.validator.checker import ConceptHierarchyChecker


class TestTemplateArgumentParsing:
    _model_data = {
        "name": "AnimalKingdom",
        "metadata": {"version": "v1.0", "author": "Your Name"},
        "concepts": {
            "Concept": {"data": None},
            "Animal": {
                "directParents": ["Concept"],
                "data": {
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
                },
            },
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
            # an alias: a second name for Dog, not a second definition of it
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
            "Function": {"directParents": ["ValueDomain"], "data": {}, "abstract": True},
            "Instance": {
                "directParents": ["InstanceBase"],
                "data": {
                    "templateContext": {
                        "order": ["AcceptConcepts...", "RejectConcepts..."],
                        "AcceptConcepts": "And(Concept, Not(ValueDomain))",
                        "RejectConcepts": "And(Concept, Not(ValueDomain))",
                        "variadicGroupIdentifiers": {"AcceptConcepts": "", "RejectConcepts": "!"},
                    }
                },
            },
            "SubInstance": {
                "directParents": ["Instance"],
                "data": {
                    "templateContext": {
                        "order": ["SubAcceptConcepts...", "SubRejectConcepts..."],
                        "SubAcceptConcepts": "And(Concept, Not(ValueDomain))",
                        "SubRejectConcepts": "And(Concept, Not(ValueDomain))",
                        "variadicGroupIdentifiers": {"SubAcceptConcepts": "", "SubRejectConcepts": "!"},
                        "substitution": {
                            "AcceptConcepts": "SubAcceptConcepts",
                            "RejectConcepts": ["SubRejectConcepts..."],
                        },
                    }
                },
            },
            "FunctionReturning": {"directParents": ["Function"], "data": {"templateContext": ["T"]}, "abstract": True},
            "Add": {
                "directParents": ["FunctionReturning"],
                "data": {
                    "templateContext": {"order": ["T"], "substitution": {"T": "T"}},
                    "interface": {"arg1": "T", "arg2": "T", "res": "T"},
                },
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
                "data": {
                    "instantiation": {"type": "object", "properties": [["props", "x", True], ["funcs", "x", True]]}
                },
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
        },
        "instances": {
            "MyAnimal": {"InstanceBase": {"instanceName": "MyAnimal", "concepts": ["Animal"], "properties": {"age": 2}}}
        },
    }

    def clone_model(self) -> dict:
        return deepcopy(self._model_data)

    def get_external_data(self, x, y):
        pass

    def get_model(self, model_data) -> ConceptHierarchy:
        model = ConceptHierarchyDefinition.create_from_data(model_data)
        checker = ConceptHierarchyChecker(model, self.get_external_data)
        res = checker.model
        checker.check()
        return res

    def test_parsing_succeeds(self):
        model = self.get_model(self._model_data)
        for c_name, c in model.ch.concepts.items():
            assert isinstance(c, HiddenImplementationDefinition) == model.ch.is_value_domain(c_name)
            if isinstance(c, HiddenImplementationDefinition):
                if c.is_templatable():
                    assert c_name in ["ClosedInterval", "List", "Instance", "SubInstance", "FunctionReturning", "Add"]

    def test_parsing_fails(self):
        model_data = self.clone_model()
        model_data["concepts"]["Add"]["directParents"] = ["Function"]
        with pytest.raises(
            CHSemanticError, match=r"Extra key\(s\) \"T\" in template substitution definition of Add must be removed!"
        ):
            self.get_model(model_data)

    def test_parsing_succeeds_after(self):
        self.get_model(self._model_data)

    def test_parsing_variadic_argument_substitution(self):
        model_data = self.clone_model()
        substitution_data = model_data["concepts"]["SubInstance"]["data"]["templateContext"]["substitution"]
        substitution_data["AcceptConcepts"] = "SubAcceptConcepts"
        substitution_data["RejectConcepts"] = "SubRejectConcepts"
        # test should succeed
        self.get_model(model_data)

        model_data = self.clone_model()
        substitution_data = model_data["concepts"]["SubInstance"]["data"]["templateContext"]["substitution"]
        substitution_data["AcceptConcepts"] = "[SubAcceptConcepts...]"
        substitution_data["RejectConcepts"] = "[SubRejectConcepts...]"
        # test should succeed
        self.get_model(model_data)

        model_data = self.clone_model()
        substitution_data = model_data["concepts"]["SubInstance"]["data"]["templateContext"]["substitution"]
        substitution_data["AcceptConcepts"] = ["SubAcceptConcepts..."]
        substitution_data["RejectConcepts"] = ["SubRejectConcepts..."]
        # test should succeed
        self.get_model(model_data)

        model_data = self.clone_model()
        substitution_data = model_data["concepts"]["SubInstance"]["data"]["templateContext"]["substitution"]
        substitution_data["AcceptConcepts"] = ["[SubAcceptConcepts...]"]
        substitution_data["RejectConcepts"] = ["[SubRejectConcepts...]"]
        # test should fail syntactically because of nested list
        with pytest.raises(
            CHSemanticError,
            match=r"Parsing \['\[SubAcceptConcepts...]'\] into a template argument value for Instance:AcceptConcepts "
            r"substitution in SubInstance failed:[\s\S]*Variadic groups are only allowed in template arguments!",
        ):
            self.get_model(model_data)

        model_data = self.clone_model()
        substitution_data = model_data["concepts"]["SubInstance"]["data"]["templateContext"]["substitution"]
        substitution_data["AcceptConcepts"] = ["SubAcceptConcepts"]
        substitution_data["RejectConcepts"] = ["SubRejectConcepts"]
        # test should fail semantically because variadic template variables are used in variadic group without expansion
        with pytest.raises(
            CHSemanticError,
            match=r"Used the variadic template variable 'SubAcceptConcepts' in a variadic group without the expansion "
            r"operator '...'! Add the operator to make this usage valid!",
        ):
            self.get_model(model_data)

        model_data = self.clone_model()
        substitution_data = model_data["concepts"]["SubInstance"]["data"]["templateContext"]["substitution"]
        substitution_data["AcceptConcepts"] = ["SubAcceptConcepts...", "SubRejectConcepts..."]
        substitution_data["RejectConcepts"] = "[SubRejectConcepts..., SubAcceptConcepts...]"
        # test should succeed
        self.get_model(model_data)

        model_data = self.clone_model()
        substitution_data = model_data["concepts"]["SubInstance"]["data"]["templateContext"]["substitution"]
        substitution_data["AcceptConcepts"] = "[SubRejectConcepts..., SubAcceptConcepts...]"
        substitution_data["RejectConcepts"] = "[SubRejectConcepts..., SubAcceptConcepts...]"
        # test should succeed
        self.get_model(model_data)

        model_data = self.clone_model()
        substitution_data = model_data["concepts"]["SubInstance"]["data"]["templateContext"]["substitution"]
        substitution_data["AcceptConcepts"] = "[SubInstance<SubAcceptConcepts..., SubRejectConcepts...>]"
        substitution_data["RejectConcepts"] = "[SubInstance<!SubRejectConcepts..., !SubAcceptConcepts...>]"
        # test should fail because SubInstance is not a DomainConcept: it violates the constraint "Not(ValueDomain)"
        with pytest.raises(
            CHSemanticError,
            match=r"Substitution values for parent Instance defined in SubInstance does not satisfy its constraints\n"
            r"[\s\S]*\[\"concepts\": \"SubInstance\": \"data\": \"templateContext\": \"substitution\": "
            r"\"Not\(ValueDomain\) <-> \[SubInstance<\[SubInstance:SubAcceptConcepts..., "
            r"SubInstance:SubRejectConcepts...\], \[\]>\]\": "
            r"\"Not\(ValueDomain\) <-> SubInstance<\[SubInstance:SubAcceptConcepts..., "
            r"SubInstance:SubRejectConcepts...\], \[\]>\"\] \n"
            r"[\s\S]*Sub formula ValueDomain passed without constraints on template arguments "
            r"TemplateContext\(vars: \('SubAcceptConcepts', 'SubRejectConcepts'\), "
            r"variadic: \['SubAcceptConcepts', 'SubRejectConcepts'\], "
            r"constraint: <Not\(ValueDomain\), Not\(ValueDomain\)>\) => negation fails\n"
            r"[\s\S]*\[\"concepts\": \"SubInstance\": \"data\": \"templateContext\": \"substitution\": "
            r"\"Not\(ValueDomain\) <-> \[SubInstance<\[\], \[SubInstance:SubRejectConcepts..., "
            r"SubInstance:SubAcceptConcepts...\]>\]\": "
            r"\"Not\(ValueDomain\) <-> SubInstance<\[\], \[SubInstance:SubRejectConcepts..., "
            r"SubInstance:SubAcceptConcepts...\]>\"\] \n"
            r"[\s\S]*Sub formula ValueDomain passed without constraints on template arguments "
            r"TemplateContext\(vars: \('SubAcceptConcepts', 'SubRejectConcepts'\), "
            r"variadic: \['SubAcceptConcepts', 'SubRejectConcepts'\], "
            r"constraint: <Not\(ValueDomain\), Not\(ValueDomain\)>\) => negation fails",
        ):
            self.get_model(model_data)

    def test_specify_variadic_argument_in_constraints(self):
        model_data = self.clone_model()
        instance_data = model_data["concepts"]["Instance"]["data"]["templateContext"]
        res = instance_data.pop("AcceptConcepts")
        instance_data["AcceptConcepts..."] = res
        with pytest.raises(
            CHSemanticError,
            match=r"Can't define a constraint for 'AcceptConcepts\.\.\.' which is not a template argument of Instance\."
            r" \n\s+It only has these template arguments: \('AcceptConcepts', 'RejectConcepts'\)",
        ):
            self.get_model(model_data)

    def test_specify_variadic_argument_in_variadic_group_identifiers(self):
        model_data = self.clone_model()
        instance_data = model_data["concepts"]["Instance"]["data"]["templateContext"]["variadicGroupIdentifiers"]
        res = instance_data.pop("AcceptConcepts")
        instance_data["AcceptConcepts..."] = res
        with pytest.raises(
            CHSyntaxError,
            match=r"The key entry in the variadic group identifiers JSON must be a variadic template argument without "
            r"its '...' variadic identifier!\n\s+Got 'AcceptConcepts...'",
        ):
            self.get_model(model_data)
