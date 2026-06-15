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
errors.py — Custom exception hierarchy for the ConceptHierarchy compiler.
"""

from __future__ import annotations

import json
from collections import UserList
from enum import Enum
from typing import TypeAlias

from concept_hierarchy.utils import tab

PathSegment: TypeAlias = str | int


class LocationId(UserList[PathSegment]):
    def __str__(self):
        return repr(self)

    def __repr__(self):
        return self.print()

    def print(self):
        return ": ".join(json.dumps(x) for x in self)


class PathPart(Enum):
    """
    Whether a :class:`ConceptHierarchyError` refers to the *key* / property name at ``path``,
     or to the *value* found under that key.
    """

    KEY = "key"
    VALUE = "value"
    NONE = ""


class ConceptHierarchyError(Exception):
    """Base class for all ConceptHierarchy compiler errors."""

    def __init__(
        self, message: str, location_id: LocationId | None, part: PathPart, causes: list[ConceptHierarchyError] | None
    ):
        super().__init__(message)
        if location_id is None:
            self.prefix = ""
        elif not location_id:  # location_id == []
            self.prefix = "ROOT"
        else:
            self.prefix = location_id.print()
        if part is PathPart.KEY:
            self.prefix += f" ({part.value})"

        self.location_id = location_id
        self.part = part
        self.causes = causes or []

    def __repr__(self):
        return self.print()

    def print(self, indent: int = 0) -> str:
        """Prints the Concept Hierarchy error message with indents and the location causing the error."""
        message_lines = str(self).split("\n")
        prefix_str = f"[{self.prefix}] " if self.prefix else ""
        content_indent_str = tab * (indent + 1)
        indent_str = tab * indent + prefix_str + "\n" + content_indent_str
        text = indent_str + ("\n" + content_indent_str).join(message_lines)
        for cause in self.causes:
            text += "\n" + cause.print(indent + 1)
        return text


class CHSyntaxError(ConceptHierarchyError):
    """Raised when the JSON definition violates ConceptHierarchy syntax rules."""

    def __init__(
        self,
        message: str,
        location_id: LocationId | None = None,
        part: PathPart = PathPart.NONE,
        causes: list[ConceptHierarchyError] | None = None,
    ) -> None:
        super().__init__(message, location_id, part, causes)


class CHSemanticError(ConceptHierarchyError):
    """Raised when the hierarchy is syntactically valid but semantically incorrect."""

    def __init__(
        self,
        message: str,
        location_id: LocationId | None = None,
        part: PathPart = PathPart.NONE,
        causes: list[ConceptHierarchyError] | None = None,
    ) -> None:
        super().__init__(message, location_id, part, causes)


class CodegenError(ConceptHierarchyError):
    """Raised when code generation fails for a valid hierarchy."""
