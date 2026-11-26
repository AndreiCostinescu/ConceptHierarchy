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

from typing import TypeAlias

from concept_hierarchy.definitions.concept_definition import ConceptDefinition
from concept_hierarchy.definitions.concept_definition_hidden_implementation import HiddenImplementationDefinition
from concept_hierarchy.definitions.definition import LocationOfCheckData
from concept_hierarchy.errors import CHSemanticError, CHSyntaxError, LocationId, PathPart

# the instantiation value is either a string value or a JSON object representing a json-schema-definition
InstantiationDefinition: TypeAlias = bool | str | list | dict
# the instantiation definition can differ depending on the template arguments
# the string value in the template-order-tuple is a template-constraint formula!
TemplateDependentInstantiationDefinition: TypeAlias = list[tuple[tuple[str, ...], InstantiationDefinition]]


class ValueDomainDefinition(HiddenImplementationDefinition):
    value_domain_name: str = "ValueDomain"
    value_domain_instantiation: str = "instantiation"
    value_domain_default_serialization: str = "defaultSerialization"
    implementation_related_keys: set[str] = HiddenImplementationDefinition.implementation_related_keys | {
        value_domain_instantiation,
        value_domain_default_serialization,
    }
    allowed_default_serializations: set[str] = {"null", "boolean", "integer", "number", "string"}
    argument_provenance_types = {"Any", "Addr"}

    def __init__(self, name: str, definition_data: object, definition_location_id: LocationId):
        super().__init__(name, definition_data, definition_location_id)

        self.default_serialization: str | None = None
        # if the ValueDomain has no template arguments, the list entry's tuple's first element will be empty: ()
        self.instantiation: TemplateDependentInstantiationDefinition | None = None
        self.was_template_dependent_instantiation_defined: bool = False

    @classmethod
    def from_node(cls, concept_definition: ConceptDefinition):
        domain_concept = super().from_node(concept_definition)
        domain_concept.default_serialization = None
        domain_concept.instantiation = None
        domain_concept.was_template_dependent_instantiation_defined = False
        return domain_concept

    def definition_type(self) -> str:
        return ValueDomainDefinition.value_domain_name

    def location_of_impl(self, *keywords: str) -> LocationOfCheckData:
        check_res = super().location_of_impl(*keywords)
        # check other top-level data keywords specific to ValueDomains
        # no subclasses of ValueDomain => the statement below is equivalent to ``raise StopLocationOfCheck(check...)``
        # and defaultSerialization and instantiation are leaf-definition-nodes => there is no more sub-data
        #   However, there is expression data (especially for instantiation which is the json schema)...
        #   -> delegate the remaining locationOf keywords to the json schema?
        return self.check_location_id(
            check_res,
            ValueDomainDefinition.definition_location(self) + [check_res.first_remaining],
            location_check=self.data,
            previous_location=ConceptDefinition.concept_definition_data,
            allow_start_at_this_location=True,
        )

    def check_default_serialization(self):
        # check "defaultSerialization"
        self.default_serialization = self.data.get(ValueDomainDefinition.value_domain_default_serialization, None)
        if self.default_serialization is not None:
            if len(self.template_argument_order) > 0:
                raise CHSemanticError(
                    f"Template-dependent {self.definition_type()} can not define a default serialization! "
                    f'Remove the keyword in "{self.name}". One can\'t infer the template arguments of "{self.name}" '
                    f"from the serialization value alone.\n\tIf you really need this feature, contact the developers "
                    f"and explain your use case.",
                    location_id=self.location_id(ValueDomainDefinition.value_domain_default_serialization),
                    part=PathPart.KEY,
                )
            if not isinstance(self.default_serialization, str):
                raise CHSyntaxError(
                    f"The definition of a {self.definition_type()}'s defaultSerialization must be a JSON string, not "
                    f"{self.default_serialization!r} for the {self.definition_type()} {self.name}",
                    location_id=self.location_id(ValueDomainDefinition.value_domain_default_serialization),
                    part=PathPart.VALUE,
                )
            elif self.default_serialization not in self.allowed_default_serializations:
                raise CHSyntaxError(
                    f'Invalid "{ValueDomainDefinition.value_domain_default_serialization}" string value for '
                    f"{self.definition_type()} {self.name}: got: {self.default_serialization!r}, allowed: "
                    f"{self.allowed_default_serializations}",
                    location_id=self.location_id(
                        ValueDomainDefinition.value_domain_default_serialization, self.default_serialization
                    ),
                    part=PathPart.VALUE,
                )
            elif self.abstract:
                raise CHSemanticError(
                    f'Can not define "{ValueDomainDefinition.value_domain_default_serialization}" for an abstract '
                    f"{self.definition_type()}: {self.name}",
                    location_id=self.location_id(
                        ValueDomainDefinition.value_domain_default_serialization, self.default_serialization
                    ),
                    part=PathPart.KEY,
                )

    def check_instantiation(self):
        # check "instantiation"
        # can only be completely parsed after the concept data is initialized for all concepts
        #   because of template-instantiations of other concepts (whose template constraints/size must be validated)
        assert isinstance(self.abstract, bool)
        if self.abstract and ValueDomainDefinition.value_domain_instantiation in self.data:
            raise CHSemanticError(
                f"An abstract {self.definition_type()} can not be instantiated!\n\t"
                f'Do not define the "{ValueDomainDefinition.value_domain_instantiation}" structure for {self.name}',
                location_id=self.location_id(ValueDomainDefinition.value_domain_instantiation),
                part=PathPart.KEY,
            )
        instantiation_data = self.data.get(ValueDomainDefinition.value_domain_instantiation, None)
        if instantiation_data is not None:
            fallback_instantiation_constraint = self.create_fallback_instantiation_constraint()
            if not isinstance(instantiation_data, (bool, dict, list, str)):
                raise CHSyntaxError(
                    f"The definition of a {self.definition_type()}'s instantiation deserialization structure must be:\n"
                    f"\ta JSON boolean value\n\ta JSON string,\n"
                    f"\ta JSON object (representing the JSON schema of the to-be-deserialized value), or\n"
                    f"\ta 2-element JSON array mapping template argument constraint definitions to "
                    f"JSON string or object deserialization structures,\n"
                    f"not {self.instantiation!r} for the {self.definition_type()} {self.name}",
                    location_id=self.location_id(ValueDomainDefinition.value_domain_instantiation),
                    part=PathPart.VALUE,
                )
            elif not isinstance(instantiation_data, list):
                # if there are no template arguments, template_argument_order is an empty tuple
                self.instantiation = [(fallback_instantiation_constraint, instantiation_data)]
            elif isinstance(instantiation_data, list) and len(instantiation_data) == 0:
                raise CHSyntaxError(
                    f"Can not specify an empty list of {self.definition_type()} instantiations for {self.name!r}!",
                    location_id=self.location_id(ValueDomainDefinition.value_domain_instantiation),
                    part=PathPart.VALUE,
                )
            else:
                assert isinstance(instantiation_data, list)
                if (
                    len(instantiation_data) == 2
                    and all(isinstance(x, str) for x in instantiation_data)
                    and instantiation_data[1] in ValueDomainDefinition.argument_provenance_types
                ):
                    self.instantiation = [(fallback_instantiation_constraint, instantiation_data)]
                else:
                    already_defined_specializations: set[tuple[str, ...]] = set()
                    self.instantiation = []
                    self.was_template_dependent_instantiation_defined = True
                    found_fallback_instantiation = False
                    for entry_index, instantiation_entry in enumerate(instantiation_data):
                        if len(instantiation_entry) != 2:
                            raise CHSyntaxError(
                                f"Invalid entry in template-specific instantiation definition:\n\tmust be a "
                                f"2-element array mapping template argument constraints to JSON (string or object) "
                                f"deserialization structures, not {instantiation_entry!r} for the "
                                f"{self.definition_type()} {self.name!r}",
                                location_id=self.location_id(
                                    ValueDomainDefinition.value_domain_instantiation, entry_index
                                ),
                            )
                        # check template constraints
                        if not isinstance(instantiation_entry[0], list):
                            raise CHSyntaxError(
                                f"Invalid entry in template-specific instantiation definition:\n\tthe first entry of "
                                f"the 2-element array must be a JSON array of string (template argument constraint "
                                f"formulae) values for each template argument.\n\t"
                                f"Got {instantiation_entry[0]!r}",
                                location_id=self.location_id(
                                    ValueDomainDefinition.value_domain_instantiation, entry_index, 0
                                ),
                            )
                        elif len(instantiation_entry[0]) != len(self.template_argument_order):
                            raise CHSemanticError(
                                f"Invalid entry in template-specific instantiation definition:\n\tthe first entry of "
                                f"the 2-element array must be a JSON array of string (template argument constraint "
                                f"formulae) values of length {len(self.template_argument_order)}.\n\t\tAn entry for "
                                f"each template argument!\n\tGot {instantiation_entry[0]!r} of length "
                                f"{len(instantiation_entry[0])}",
                                location_id=self.location_id(
                                    ValueDomainDefinition.value_domain_instantiation, entry_index, 0
                                ),
                            )
                        else:
                            for constraint_index, template_arg_constraint in enumerate(instantiation_entry[0]):
                                if not isinstance(template_arg_constraint, str):
                                    raise CHSyntaxError(
                                        f"Invalid entry in template-specific instantiation definition:\n\tthe first "
                                        f"entry of the 2-element array must be a JSON array of string (template "
                                        f"argument constraint formulae) values for each template argument.\n\tGot "
                                        f"{template_arg_constraint!r}",
                                        location_id=self.location_id(
                                            ValueDomainDefinition.value_domain_instantiation,
                                            entry_index,
                                            0,
                                            constraint_index,
                                        ),
                                    )
                        instantiation_specialization_key = tuple(instantiation_entry[0])
                        if instantiation_specialization_key in already_defined_specializations:
                            raise CHSemanticError(
                                f"Doubly-defined template-specific {self.definition_type()} instantiation key "
                                f"{instantiation_specialization_key!r}",
                                location_id=self.location_id(
                                    ValueDomainDefinition.value_domain_instantiation,
                                    entry_index,
                                ),
                            )
                        # check deserialization structure specification (just its outer structure, not its content)
                        if not isinstance(instantiation_entry[1], (bool, dict, list, str)):
                            raise CHSyntaxError(
                                f"Invalid entry in template-specific instantiation definition:\n\tthe second entry of "
                                f"the 2-element array must be a JSON (boolean, array, string or object) deserialization"
                                f" structures, not {instantiation_entry[1]!r}.",
                                location_id=self.location_id(
                                    ValueDomainDefinition.value_domain_instantiation, entry_index, 1
                                ),
                            )
                        # add data to ordered instantiation list
                        already_defined_specializations.add(instantiation_specialization_key)
                        found_fallback_instantiation |= (
                            instantiation_specialization_key == fallback_instantiation_constraint
                        )
                        self.instantiation.append((instantiation_specialization_key, instantiation_entry[1]))
                    if not found_fallback_instantiation:
                        raise CHSemanticError(
                            f"Template-dependent instantiation schemas must define a fallback instantiation schema!\n"
                            f'For {self.name}, this fallback schema must be "{fallback_instantiation_constraint}"."',
                            location_id=self.location_id(ValueDomainDefinition.value_domain_instantiation),
                            part=PathPart.VALUE,
                        )
            assert (
                isinstance(self.instantiation, list)
                and all(isinstance(x, tuple) for x in self.instantiation)
                and all(isinstance(y, str) for x in self.instantiation for y in x[0])
            )

    def concept_data_check(self):
        super().concept_data_check()

        self.check_instantiation()
        # missing checks:
        #  - template constraints of "instantiation" specializations ("instantiation" is template-dependent!)
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

    def create_fallback_instantiation_constraint(self) -> tuple[str, ...]:
        return tuple("" for _ in self.template_argument_order)
