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
from dataclasses import astuple, dataclass
from typing import Callable, TypeVar

from concept_hierarchy.definitions.utils import check_ch_name
from concept_hierarchy.errors import CHSyntaxError, LocationId, PathPart, PathSegment

T = TypeVar("T", bound="ConceptHierarchyDefinition")


@dataclass
class LocationOfCheckData:
    current_location_id: LocationId | None
    remaining_keywords: tuple[str, ...]
    consumed_keywords: tuple[str, ...]
    check_successful: bool

    @property
    def last_consumed(self):
        return self.consumed_keywords[-1]

    @property
    def first_remaining(self):
        return self.remaining_keywords[0]


class StopLocationOfCheck(Exception):
    def __init__(self, check_data: LocationOfCheckData):
        self.data = check_data


class FoundLocationId(StopLocationOfCheck):
    def __init__(self, check_data: LocationOfCheckData):
        super().__init__(check_data)


class ConceptHierarchyDefinition(ABC):
    def __init__(self, name: str, definition_data: object, definition_location_id: LocationId):
        self.name: str = name
        # noinspection PyTypeChecker
        self.definition_data: dict | str = definition_data
        self.definition_location_id: LocationId = definition_location_id

        self.is_root: bool = False
        self.is_reference_to: str | None = None
        self.from_reference: str | None = None

        self._definition_location_cache: dict[tuple[str, ...], LocationId] = {}

        self.check()

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"{self.definition_type()}({self.name})"

    @classmethod
    def _from_node(cls, node: T) -> T:
        if not issubclass(cls, type(node)):
            raise TypeError(f"Expected parent class of {cls.__name__}, got {type(node).__name__}")
        obj = cls.__new__(cls)
        obj.__dict__.update(node.__dict__)
        # don't copy the dictionary cache data (otherwise referencing-concepts will share the cache with the reference!)
        obj._definition_location_cache = {}
        obj._definition_location_cache.update(node._definition_location_cache)
        # Don't process definition_location_id specially, even if it is a mutable object (LocationId) because this is
        #  treated as frozen/fixed in all (subclass) Definitions!
        return obj

    def create_from_reference(self, referenced_definition: T) -> T:
        if not self.is_reference():
            raise RuntimeError(
                f"Can not call create_from_reference on a non-reference {self.definition_type()} {self.name}"
            )
        assert isinstance(self.is_reference_to, str)
        # This ``self.is_reference_to != referenced_definition.name`` can happen in long reference chains!
        # because is_reference_to is the direct reference; but this direct reference can reference other data itself...
        # So can't check correctness of the referenced_definition by the name alone...
        res = self.__class__._from_node(referenced_definition)
        res.is_reference_to = None
        res.from_reference = res.name
        res.name = self.name
        return res

    @abstractmethod
    def check(self):
        if not isinstance(self.definition_type(), str):
            raise RuntimeError(f"Definition type must be a string, not {self.definition_type()!r}")
        if not isinstance(self.name, str) or not check_ch_name(self.name):
            raise CHSyntaxError(
                f"Name of {self.definition_type()!r} definition must be a valid string identifier, got {self.name!r}!",
                location_id=ConceptHierarchyDefinition.definition_location(self) + [self.name],
                part=PathPart.KEY,
            )
        if not isinstance(self.definition_data, (dict, str)):
            raise CHSyntaxError(
                f"The concept definition of {self.name} is not a JSON object or a JSON string "
                f"concept-name reference, but {self.definition_data!r}!",
                location_id=ConceptHierarchyDefinition.definition_location(self) + [self.name],
                part=PathPart.VALUE,
            )
        if isinstance(self.definition_data, str):
            self.is_reference_to = self.definition_data

    @abstractmethod
    def definition_type(self) -> str:
        pass

    @abstractmethod
    def definition_location(self) -> LocationId:
        return self.definition_location_id

    def location_id(self, *location_ids: PathSegment) -> LocationId:
        return self.definition_location() + [*location_ids]

    def location_of(self, *keywords: str) -> LocationId:
        """
        THIS FUNCTION SHOULD ONLY BE CALLED AFTER THE STRUCTURAL CHECKS OF THE CONCEPTS HAVE PASSED!
        Do not use this function during the structural checks of the ConceptHierarchyDefinition subclasses!

        This function determines the concept-specific specification location of a piece of data.
        This is because data is not always at a fixed location in the Concept Hierarchy.
        For example, shorthand notations of Function argument types, shorthand template order definition, external data.

        This function returns the location of the requested data in the concept's definition (so where it is defined in
        this concept, not where it's canonical representation says it should be).
        """

        if keywords in self._definition_location_cache:
            return self._definition_location_cache[keywords]

        try:
            location_id, remaining_keywords, _, _ = self.location_of_impl(*keywords)
        except StopLocationOfCheck as e:
            location_id, remaining_keywords, _, _ = e.data
        if remaining_keywords != ():
            raise RuntimeError(
                f"Keyword(s) {keywords!r} not found in the {self.definition_type()} definition of {self.name}"
            )
        self._definition_location_cache[keywords] = location_id
        return location_id

    @abstractmethod
    def location_of_impl(self, *keywords: str) -> LocationOfCheckData:
        # processes "concepts"/"instances" location
        return self.check_location_id(
            LocationOfCheckData(None, keywords, (), False),
            ConceptHierarchyDefinition.definition_location(self),
            location_check=self.definition_location_id[-1],
            previous_location=None,
            allow_start_at_this_location=True,
        )

    def check_location_id(
        self,
        check_result: LocationOfCheckData,
        new_location_if_successful: LocationId,
        location_check: str | tuple[str, ...] | list[str] | set[str] | dict[str, object] | Callable[[str], bool],
        previous_location: str | None,
        allow_start_at_this_location: bool,
    ) -> LocationOfCheckData:
        """Modifies ``check_result`` in-place, but returns it as well (same object)."""
        if isinstance(location_check, str):
            original = location_check

            def location_check(x):
                return x == original
        elif isinstance(location_check, (set, tuple, list, dict)):
            original = location_check

            def location_check(x):
                return x in original

        current_location_id, remaining_keywords, consumed_keywords, _ = astuple(check_result)
        assert remaining_keywords != ()
        check_result.check_successful = location_check(remaining_keywords[0]) and (
            allow_start_at_this_location if consumed_keywords == () else consumed_keywords[-1] == previous_location
        )
        if check_result.check_successful:
            check_result.current_location_id = new_location_if_successful
            check_result.consumed_keywords += (remaining_keywords[0],)
            self._definition_location_cache[consumed_keywords] = current_location_id
            check_result.remaining_keywords = remaining_keywords[1:]
        if check_result.remaining_keywords == ():
            raise FoundLocationId(check_result)
        return check_result

    def is_reference(self) -> bool:
        return self.is_reference_to is not None
