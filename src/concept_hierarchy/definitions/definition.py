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

from abc import ABC, abstractmethod

from concept_hierarchy.definitions.utils import check_ch_name
from concept_hierarchy.errors import CHSyntaxError, LocationId


class ConceptHierarchyDefinition(ABC):
    def __init__(self, name: str, definition_data: object, definition_location_str: str):
        self.name: str = name
        # noinspection PyTypeChecker
        self.definition_data: dict | str = definition_data
        self.definition_location_str: str = definition_location_str

        self.is_root: bool = False
        self.is_reference_to: str | None = None

        self.check()

    @abstractmethod
    def check(self):
        if not isinstance(self.definition_type, str):
            raise RuntimeError(f"Definition type must be a string, not {self.definition_type!r}")
        if not isinstance(self.name, str) or not check_ch_name(self.name):
            raise CHSyntaxError(
                f"Name of {self.definition_type!r} definition must be a valid string identifier, got {self.name!r}!",
                self.location_id(),
            )
        if not isinstance(self.definition_data, (dict, str)):
            raise CHSyntaxError(
                f"The concept definition of {self.name} is not a JSON object or a JSON string "
                f"concept-name reference, but {self.definition_data!r}!",
                self.location_id(),
            )
        if isinstance(self.definition_data, str):
            self.is_reference_to = self.definition_data

    @property
    @abstractmethod
    def definition_type(self) -> str:
        pass

    @property
    @abstractmethod
    def definition_location(self) -> list[str]:
        return [self.definition_location_str]

    def location_id(self, *location_ids: str | int) -> LocationId:
        return self.definition_location + [*location_ids]

    def is_reference(self) -> bool:
        return self.is_reference_to is not None
