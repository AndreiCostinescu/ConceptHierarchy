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
from typing import Callable

from concept_hierarchy.definitions.definition import (
    ConceptHierarchyDefinition,
    LocationOfCheckData,
    StopLocationOfCheck,
)
from concept_hierarchy.definitions.utils import check_ch_name
from concept_hierarchy.errors import CHSemanticError, CHSyntaxError, LocationId, PathPart


class ConceptDefinition(ConceptHierarchyDefinition):
    concept_name: str = "Concept"
    concept_definition_data: str = "data"
    concept_description: str = "description"
    concept_direct_parents: str = "directParents"
    concept_direct_children: str = "directChildren"
    concept_distinct_from: str = "distinctFrom"
    concept_distinct_group: str = "distinctGroup"
    concept_min_instances: str = "minInstances"
    concept_max_instances: str = "maxInstances"
    concept_abstract: str = "abstract"
    concept_data_keys: set[str] = {
        concept_direct_parents,
        concept_description,
        concept_definition_data,
        concept_direct_children,
        concept_distinct_from,
        concept_distinct_group,
        concept_min_instances,
        concept_max_instances,
        concept_abstract,
    }

    def __init__(
        self,
        name: str,
        definition_data: object,
        definition_location_id: LocationId,
        *,
        external_data_resolver: Callable | None = None,
    ):
        self.parents: tuple[str, ...] = ()
        self.description: str | None = None
        self.fixed_children: tuple[str, ...] = ()
        self.min_instances: int = 0
        self.max_instances: int | None = None
        self.distinct_from: tuple[str, ...] = ()
        self.distinct_group: tuple[str, ...] = ()
        self.abstract: bool | None = None
        """
        An abstract concept (DomainConcept, ValueDomain, Function) can not be instantiated. 
        An abstract Function can not be instantiated and does not have to define its interface (but it can)!
        """

        self.data: dict[str, object] = {}
        self._data_def: object = None
        self.external_data_resolver = external_data_resolver

        self._was_abstract_defined: bool = False

        # initialize this member before calling super, which calls the check function
        self._data_location_id: LocationId = LocationId([ConceptDefinition.concept_definition_data])

        super().__init__(name, definition_data, definition_location_id)
        # when the concept is initialized (just as a concept at the beginning) the function below does nothing
        self.concept_data_check()  # sets the members of subclasses of ConceptDefinition

    def update_parents(self, new_parents: tuple[str, ...]) -> None:
        self.parents = new_parents
        self.non_root_data_specified_check()

    @property
    def is_root_concept(self) -> bool:
        return self.parents == ()

    @property
    def data_location_id(self) -> LocationId:
        return ConceptDefinition.definition_location(self) + self._data_location_id

    def non_root_data_specified_check(self):
        if ConceptDefinition.concept_definition_data not in self.definition_data and self.parents != ():
            raise CHSyntaxError(
                f'Every non-root concept must define its data in the "{ConceptDefinition.concept_definition_data}" '
                f"keyword! Concept {self.name!r} does not, please add its data!",
                location_id=self.location_id(),
                part=PathPart.VALUE,
            )

    def check_abstract(self):
        # check "abstract"
        self.abstract = self.definition_data.get(ConceptDefinition.concept_abstract, None)
        self._was_abstract_defined = self.abstract is not None
        if self._was_abstract_defined:
            if not isinstance(self.abstract, bool):
                raise CHSyntaxError(
                    f"The definition of a {self.definition_type()}'s abstract marker must be a JSON boolean, not "
                    f"{self.abstract!r} for the {self.definition_type()} {self.name}",
                    location_id=self.location_id(ConceptDefinition.concept_abstract),
                    part=PathPart.VALUE,
                )
        else:
            self.abstract = False

    def check(self):
        super().check()

        if not check_ch_name(self.name, must_start_uppercase=True):
            raise CHSyntaxError(
                f"Name of concept definition {self.name!r} must be an uppercase string!",
                location_id=self.location_id(),
                part=PathPart.KEY,
            )
        if self.is_reference():
            return

        defined_keys = set(self.definition_data.keys())
        if not (defined_keys <= ConceptDefinition.concept_data_keys):
            extra_keys = defined_keys - ConceptDefinition.concept_data_keys
            raise CHSyntaxError(
                f"Found extra keys {extra_keys!r} in the concept definition of {self.name}",
                location_id=self.location_id(),
                part=PathPart.VALUE,
            )

        # do not check direct parents yet, because they may be updated after the topological sort!
        # they can be left empty and the parser will make them subconcepts of Concept
        parents_def = self.definition_data.get(ConceptDefinition.concept_direct_parents, [])
        if not isinstance(parents_def, list):
            raise CHSyntaxError(
                f"Direct parents of the concept {self.name} must be a JSON array of strings, not {parents_def!r}!",
                location_id=self.location_id(ConceptDefinition.concept_direct_parents),
                part=PathPart.VALUE,
            )
        for index, parent in enumerate(parents_def):
            if not isinstance(parent, str):
                raise CHSyntaxError(
                    f"Direct parents of the concept {self.name} must be a list of concepts. "
                    f"Encountered {parent!r} at parent {index}!",
                    location_id=self.location_id(ConceptDefinition.concept_direct_parents, index),
                )
            # Don't check here the concept-name string formatting requirements for the parent nodes,
            #  because the parent's concept name will be checked when it will be processed.
            #  And if the string is not a valid parent name, then the parent-in-ch semantic rule will determine an
            #  invalid parent specification when computing the topological sort of the hierarchy graph!
        self.parents = tuple(parents_def)
        # missing checks:
        #  - check that parents of Functions are Functions (except ValueDomain)
        #    STRUCTURE CHECK
        #       - done in checker.py - check_after_parsing_concepts
        #  - check that parents of ValueDomains are ValueDomains (except Concept)
        #    STRUCTURE CHECK
        #       - done in checker.py - check_after_parsing_concepts
        #  - check that parents of Domain Concepts are Domain Concepts
        #    STRUCTURE CHECK
        #       - done in checker.py - check_after_parsing_concepts

        self.description = self.definition_data.get(ConceptDefinition.concept_description, None)
        if not isinstance(self.description, (str, NoneType)):
            raise CHSyntaxError(
                f"The description of the concept {self.name} must be a JSON string or null, not {self.description!r}",
                location_id=self.location_id("definition"),
                part=PathPart.VALUE,
            )

        self.non_root_data_specified_check()
        self._data_def = self.definition_data.get(ConceptDefinition.concept_definition_data, None)
        while self._check_data_content(self._data_location_id):
            if self.external_data_resolver is None:
                raise RuntimeError(
                    f"Need to read external data for concept {self.name}, but the external data reader is None!"
                )
            try:
                self._data_location_id.append("ext:" + self._data_def)
                self._data_def = self.external_data_resolver(self.name, self._data_def)
            except RuntimeError as e:
                if str(e).startswith("Could not find external data file"):
                    raise CHSemanticError(
                        f"Incorrect external data file specified for concept {self.name}: {self._data_def!r}!",
                        location_id=self.location_id(*self._data_location_id),
                        part=PathPart.VALUE,
                    ) from e
                raise e

        self.check_abstract()

        for min_max_instances in [ConceptDefinition.concept_min_instances, ConceptDefinition.concept_max_instances]:
            if min_max_instances in self.definition_data:
                min_max_instances_value = self.definition_data[min_max_instances]
                if not isinstance(min_max_instances_value, int) or min_max_instances_value < 0:
                    raise CHSyntaxError(
                        f'The definition of {self.definition_type()} "{min_max_instances}" must be a non-negative '
                        f"integer, not {min_max_instances_value}",
                        location_id=self.location_id(min_max_instances),
                        part=PathPart.VALUE,
                    )
                if min_max_instances == ConceptDefinition.concept_min_instances:
                    self.min_instances = min_max_instances_value
                else:
                    assert min_max_instances == ConceptDefinition.concept_max_instances
                    self.max_instances = min_max_instances_value

        if (
            self.min_instances is not None
            and self.max_instances is not None
            and self.min_instances > self.max_instances
        ):
            raise CHSemanticError(
                f'Defined a {self.definition_type()} with "{ConceptDefinition.concept_min_instances}" greater than '
                f'"{ConceptDefinition.concept_max_instances}".\n\tThis effectively makes this concept not-instantiable.'
                f'\n\tThe clearer way to do this is to remove "{ConceptDefinition.concept_min_instances}" and to set '
                f'"{ConceptDefinition.concept_max_instances}" to 0.',
                location_id=self.location_id(),
                part=PathPart.VALUE,
            )

        if self.max_instances == 0 and self._was_abstract_defined and self.abstract is False:
            raise CHSemanticError(
                f'Setting "{ConceptDefinition.concept_max_instances}": 0 makes this concept abstract, but defined '
                f'"{ConceptDefinition.concept_abstract}": false.\n\tThis is a contradiction: resolve by removing the '
                f'"{ConceptDefinition.concept_abstract}" keyword, setting it to "true" or making '
                f'"{ConceptDefinition.concept_max_instances}" > 0.',
                location_id=self.location_id(ConceptDefinition.concept_abstract),
                part=PathPart.VALUE,
            )

        if ConceptDefinition.concept_distinct_from in self.data:
            self.distinct_from = self.data[ConceptDefinition.concept_distinct_from]
            if not isinstance(self.distinct_from, list) or not all(isinstance(x, str) for x in self.distinct_from):
                raise CHSyntaxError(
                    f'The definition of domain concept "{ConceptDefinition.concept_distinct_from}" distinct group must '
                    f"be a JSON array of strings, not {self.distinct_from!r}",
                    location_id=self.location_id(ConceptDefinition.concept_distinct_from),
                    part=PathPart.VALUE,
                )
            if len(set(self.distinct_from)) != len(self.distinct_from):
                raise CHSemanticError(
                    f"The definition of distinct group {self.distinct_from!r} contains duplicates! Please remove them",
                    location_id=self.location_id(ConceptDefinition.concept_distinct_from),
                    part=PathPart.VALUE,
                )
        # missing checks:
        #  - check that all concept names in distinct_from are:
        #   1) concepts,
        #   2) different from this concept, and
        #   3) not parents of this concept
        #   STRUCTURE CHECK
        #       - done in checker.py - check_after_parsing_concepts

    def definition_type(self) -> str:
        return ConceptDefinition.concept_name

    def definition_location(self) -> LocationId:
        location_res = super().definition_location() + [self.name]
        if self.from_reference is not None:
            location_res.append("ref:" + self.from_reference)
        return location_res

    def location_of_impl(self, *keywords: str) -> LocationOfCheckData:
        # processes name-of-concept keyword (after processing parent keywords: "concepts"/"instances")
        check_res = self.check_location_id(
            super().location_of_impl(*keywords),
            ConceptDefinition.definition_location(self),
            location_check=self.name,
            previous_location=self.definition_location_id[-1],
            allow_start_at_this_location=True,
        )
        # processes top-level concept keys:
        #  data, description, directParents, directChildren, distinctFrom, distinctGroup,
        #  minInstances, maxInstances, abstract
        assert isinstance(self.definition_data, dict)
        self.check_location_id(
            check_res,
            ConceptDefinition.definition_location(self) + [check_res.remaining_keywords[0]],
            location_check=self.definition_data,
            previous_location=self.name,
            allow_start_at_this_location=True,
        )
        # stop traversal if the keyword is not "data" (because other keywords are leaf-nodes: there is no more sub-data)
        if (
            check_res.check_successful is True
            and check_res.consumed_keywords[-1] != ConceptDefinition.concept_definition_data
        ):
            raise StopLocationOfCheck(check_res)
        return check_res

    # returns whether the data is NOT an external file
    def _check_data_content(self, data_location_id: LocationId) -> bool:
        if not isinstance(self._data_def, (str, dict, NoneType)):
            raise CHSyntaxError(
                f"The data of the concept {self.name} must be a JSON object, null, or a string file path "
                f"(absolute or relative to the root directory), not {self._data_def!r}",
                location_id=self.location_id(*data_location_id),
                part=PathPart.VALUE,
            )
        elif self._data_def is None:
            self.data = {}
        elif isinstance(self._data_def, dict):
            self.data = self._data_def
        return isinstance(self._data_def, str)

    def concept_data_check(self):
        pass
