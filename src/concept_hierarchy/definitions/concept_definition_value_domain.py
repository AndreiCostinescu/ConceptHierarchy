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

from __future__ import annotations

from types import NoneType
from typing import TypeAlias

from concept_hierarchy.definitions.concept_definition import ConceptDefinition
from concept_hierarchy.definitions.concept_definition_hidden_implementation import HiddenImplementationDefinition
from concept_hierarchy.errors import CHSemanticError, CHSyntaxError, PathPart

# the instantiation value is either a string value or a JSON object representing a json-schema-definition
InstantiationDefinition: TypeAlias = str | dict
# the instantiation definition can differ depending on the template arguments
# the string value in the template-order-tuple is a template-constraint formula!
TemplateDependentInstantiationDefinition: TypeAlias = dict[tuple[str | None, ...], InstantiationDefinition]


class ValueDomainDefinition(HiddenImplementationDefinition):
    value_domain_name: str = "ValueDomain"
    value_domain_instantiation: str = "instantiation"
    value_domain_default_serialization: str = "defaultSerialization"
    implementation_related_keys: set[str] = HiddenImplementationDefinition.implementation_related_keys | {
        value_domain_instantiation,
        value_domain_default_serialization,
    }
    allowed_default_serializations: set[str] = {"null", "boolean", "integer", "number", "string"}
    argument_reference_types = {"NoRef", "Reference"}

    def __init__(self, name: str, definition_data: object):
        super().__init__(name, definition_data)

        self.default_serialization: str | None = None
        # if the ValueDomain has no template arguments, the tuple dict entry will be empty: ()
        self.instantiation: TemplateDependentInstantiationDefinition | None = None

    @classmethod
    def from_node(cls, concept_definition: ConceptDefinition):
        domain_concept = super().from_node(concept_definition)
        domain_concept.default_serialization = None
        domain_concept.instantiation = None
        return domain_concept

    @property
    def definition_type(self) -> str:
        return ValueDomainDefinition.value_domain_name

    def check_default_serialization(self):
        # check "defaultSerialization"
        self.default_serialization = self.data.get(ValueDomainDefinition.value_domain_default_serialization, None)
        if self.default_serialization is not None:
            if not isinstance(self.default_serialization, str):
                raise CHSyntaxError(
                    f"The definition of a {self.definition_type}'s defaultSerialization must be a JSON string, not "
                    f"{self.default_serialization!r} for the {self.definition_type} {self.name}",
                    self.location_id(ValueDomainDefinition.value_domain_default_serialization),
                    part=PathPart.VALUE,
                )
            elif self.default_serialization not in self.allowed_default_serializations:
                raise CHSyntaxError(
                    f'Invalid "{ValueDomainDefinition.value_domain_default_serialization}" string value for '
                    f"{self.definition_type} {self.name}: got: {self.default_serialization!r}, allowed: "
                    f"{self.allowed_default_serializations}",
                    self.location_id(self.default_serialization),
                    part=PathPart.VALUE,
                )

    def check_instantiation(self):
        # check "instantiation"
        # can only be completely parsed after the concept data is initialized for all concepts
        #   because of template-instantiations of other concepts (whose template constraints/size must be validated)
        assert isinstance(self.abstract, bool)
        if self.abstract and ValueDomainDefinition.value_domain_instantiation in self.data:
            raise CHSemanticError(
                f"An abstract {self.definition_type} can not be instantiated!\n\t"
                f'Do not define the "{ValueDomainDefinition.value_domain_instantiation}" structure for {self.name}',
                self.location_id(ValueDomainDefinition.value_domain_instantiation),
                part=PathPart.KEY,
            )
        self.instantiation = self.data.get(ValueDomainDefinition.value_domain_instantiation, None)
        if self.instantiation is not None:
            if not isinstance(self.instantiation, (bool, str, dict, list)):
                raise CHSyntaxError(
                    f"The definition of a {self.definition_type}'s instantiation deserialization structure must be:\n"
                    f"\ta JSON boolean value\n\ta JSON string,\n"
                    f"\ta JSON object (representing the JSON schema of the to-be-deserialized value), or\n"
                    f"\ta 2-element JSON array mapping template argument constraint definitions to "
                    f"JSON string or object deserialization structures,\n"
                    f"not {self.instantiation!r} for the {self.definition_type} {self.name}",
                    self.location_id(ValueDomainDefinition.value_domain_instantiation),
                    part=PathPart.VALUE,
                )
            elif not isinstance(self.instantiation, list):
                # if there are no template arguments, template_argument_order is an empty tuple
                self.instantiation = {tuple(None for _ in self.template_argument_order): self.instantiation}
            else:
                instantiation_dict: dict[tuple[str | None, ...], str | dict] = {}
                for entry_index, instantiation_entry in enumerate(self.instantiation):
                    if len(instantiation_entry) != 2:
                        raise CHSyntaxError(
                            f"Invalid entry in template-specific instantiation definition:\n\tmust be a "
                            f"2-element array mapping template argument constraints to JSON (string or object) "
                            f"deserialization structures, not {instantiation_entry!r} for the {self.definition_type} "
                            f"{self.name}",
                            self.location_id(ValueDomainDefinition.value_domain_instantiation, entry_index),
                        )
                    # check template constraints
                    if not isinstance(instantiation_entry[0], list):
                        raise CHSyntaxError(
                            f"Invalid entry in template-specific instantiation definition:\n\tthe first entry of the "
                            f"2-element array must be a JSON array of string (template argument constraint formulae) or"
                            f" null (unconstrained) values for each template argument.\n\t"
                            f"Got {instantiation_entry[0]!r}",
                            self.location_id(ValueDomainDefinition.value_domain_instantiation, entry_index, 0),
                        )
                    elif len(instantiation_entry[0]) != len(self.template_argument_order):
                        raise CHSyntaxError(
                            f"Invalid entry in template-specific instantiation definition:\n\tthe first entry of the "
                            f"2-element array must be a JSON array of string (template argument constraint formulae) or"
                            f" null (unconstrained) values of length {len(self.template_argument_order)}.\n\t\t"
                            f"An entry for each template argument!\n\tGot {instantiation_entry[0]!r} of length "
                            f"{len(instantiation_entry[0])}",
                            self.location_id(ValueDomainDefinition.value_domain_instantiation, entry_index, 0),
                        )
                    else:
                        for constraint_index, template_arg_constraint in instantiation_entry[0]:
                            if not isinstance(template_arg_constraint, (str, NoneType)):
                                raise CHSyntaxError(
                                    f"Invalid entry in template-specific instantiation definition:\n\tthe first entry"
                                    f" of the 2-element array must be a JSON array of string (template argument "
                                    f"constraint formulae) or null (unconstrained) values for each template argument."
                                    f"\n\tGot {template_arg_constraint!r}",
                                    self.location_id(
                                        ValueDomainDefinition.value_domain_instantiation,
                                        entry_index,
                                        0,
                                        constraint_index,
                                    ),
                                )
                    # check deserialization structure specification (just its outer structure, not its content)
                    if not isinstance(instantiation_entry[1], (str, dict)):
                        raise CHSyntaxError(
                            f"Invalid entry in template-specific instantiation definition:\n\tthe second entry of the"
                            f"2-element array must be a JSON (string or object) deserialization structures, not "
                            f"{instantiation_entry[1]!r}.",
                            self.location_id(ValueDomainDefinition.value_domain_instantiation, entry_index, 1),
                        )
                    # add data to dictionary
                    instantiation_dict[tuple(instantiation_entry[0])] = instantiation_entry[1]
                self.instantiation = instantiation_dict
            assert isinstance(self.instantiation, dict) and all(isinstance(x, tuple) for x in self.instantiation)

    def concept_data_check(self):
        super().concept_data_check()

        self.check_instantiation()
        # missing checks:
        #  - template constraints of "instantiation" specializations
        #    TYPE CHECK
        #       (check valid constraint def. + warn if definition intersected with t-arg-constraint is empty)
        #  - json-schema specification formulae (including correctly template-instantiated types in its specification)
        #    REQUIRES: type parsing
        #    EXPRESSION CHECK
        self.check_default_serialization()
        # missing checks:
        #  - defaultSerialization value specifications are unique across all concepts!
        #    REQUIRES: all concept data initialized
        #    STRUCTURE CHECK
        #       - done in checker.py - check_after_parsing_concepts
