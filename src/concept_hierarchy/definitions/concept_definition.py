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

from concept_hierarchy.definitions.definition import ConceptHierarchyDefinition
from concept_hierarchy.definitions.utils import check_ch_name
from concept_hierarchy.errors import CHSemanticError, CHSyntaxError, LocationId


class ConceptDefinition(ConceptHierarchyDefinition):
    concept_name: str = "Concept"
    concept_definition_data: str = "data"
    concept_description: str = "description"
    concept_direct_parents: str = "directParents"
    concept_data_keys: set[str] = {concept_direct_parents, concept_description, concept_definition_data}

    def __init__(
        self,
        name: str,
        definition_data: object,
        definition_location_str: str,
        *,
        external_data_resolver: Callable | None = None,
    ):
        self.parents: list[str] = []
        self.description: str | None = None
        self.data: dict[str, object] = {}
        self._data_def: object = None
        self.external_data_resolver = external_data_resolver
        super().__init__(name, definition_data, definition_location_str)
        self.concept_data_check()  # sets the members of subclasses of ConceptDefinition

    def check(self):
        super().check()

        if not check_ch_name(self.name, must_start_uppercase=True):
            raise CHSyntaxError(
                f"Name of concept definition {self.name!r} must be an uppercase string!", self.location_id()
            )
        if self.is_reference():
            return

        defined_keys = set(self.definition_data.keys())
        if not (defined_keys <= ConceptDefinition.concept_data_keys):
            extra_keys = defined_keys - ConceptDefinition.concept_data_keys
            raise CHSyntaxError(
                f"Found extra keys {extra_keys!r} in the concept definition of {self.name}", self.location_id()
            )

        self.parents = self.definition_data.get(ConceptDefinition.concept_direct_parents, [])
        if not isinstance(self.parents, list):
            raise CHSyntaxError(
                f"Direct parents of the concept {self.name} must be a JSON array of strings, not {self.parents!r}!",
                self.location_id(ConceptDefinition.concept_direct_parents),
            )
        else:
            for index, parent in enumerate(self.parents):
                if not isinstance(parent, str):
                    raise CHSyntaxError(
                        f"Direct parents of the concept {self.name} must be a list of concepts. "
                        f"Encountered {parent!r} at parent {index}!",
                        self.location_id(ConceptDefinition.concept_direct_parents, index),
                    )
                # Don't check here the concept-name string formatting requirements for the parent nodes,
                #  because the parent's concept name will be checked when it will be processed.
                #  And if the string is not a valid parent name, then the parent-in-ch semantic rule will determine an
                #  invalid parent specification when computing the topological sort of the hierarchy graph!

        self.description = self.definition_data.get(ConceptDefinition.concept_description, None)
        if not isinstance(self.description, (str, NoneType)):
            raise CHSyntaxError(
                f"The description of the concept {self.name} must be a JSON string or null, not {self.description!r}",
                self.location_id("definition"),
            )

        self._data_def = self.definition_data.get(ConceptDefinition.concept_definition_data, None)
        data_location_id = [ConceptDefinition.concept_definition_data]
        while self._check_data_content(data_location_id):
            if self.external_data_resolver is None:
                raise RuntimeError(
                    f"Need to read external data for concept {self.name}, but the external data reader is None!"
                )
            try:
                data_location_id.append(self._data_def)
                self._data_def = self.external_data_resolver(self.name, self._data_def)
            except RuntimeError as e:
                if str(e).startswith("Could not find external data file"):
                    raise CHSemanticError(
                        f"Incorrect external data file specified for concept {self.name}: {self._data_def!r}!",
                        self.location_id(*data_location_id),
                    ) from e
                raise e

    @property
    def definition_type(self) -> str:
        return ConceptDefinition.concept_name

    @property
    def definition_location(self) -> list[str]:
        return super().definition_location + [self.name]

    # returns whether the data is NOT an external file
    def _check_data_content(self, data_location_id: LocationId) -> bool:
        if not isinstance(self._data_def, (str, dict, NoneType)):
            raise CHSyntaxError(
                f"The data of the concept {self.name} must be a JSON object, null, or a string file path "
                f"(absolute or relative to the root directory), not {self._data_def!r}",
                location_id=self.location_id(*data_location_id),
            )
        elif self._data_def is None:
            self.data = {}
        elif isinstance(self._data_def, dict):
            self.data = self._data_def
        return isinstance(self._data_def, str)

    def concept_data_check(self):
        pass

    @classmethod
    def _from_node(cls, node: ConceptDefinition):
        obj = cls.__new__(cls)
        obj.__dict__.update(node.__dict__)
        return obj
