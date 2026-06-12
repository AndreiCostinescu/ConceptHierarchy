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

import json
from typing import TypeAlias

from concept_hierarchy.utils import tab

PathSegment: TypeAlias = str | int
LocationId: TypeAlias = list[PathSegment]


def print_location_id(location_id: LocationId):
    return ":".join(json.dumps(x) for x in location_id)


class ConceptHierarchyError(Exception):
    """Base class for all ConceptHierarchy compiler errors."""

    def __init__(self, message: str, location_id: LocationId | None = None):
        super().__init__(message)
        if location_id is None:
            self.prefix = ""
        elif not location_id:  # location_id == []
            self.prefix = "ROOT"
        else:
            self.prefix = print_location_id(location_id)

    def __repr__(self):
        return self.print()

    def print(self, indent: int = 0):
        """Prints the Concept Hierarchy error message with indents and the location causing the error."""
        message_lines = str(self).split("\n")
        prefix_str = f"[{self.prefix}] " if self.prefix else ""
        indent_str = tab * indent + prefix_str
        return indent_str + ("\n" + indent_str).join(message_lines)


class CHSyntaxError(ConceptHierarchyError):
    """Raised when the JSON definition violates ConceptHierarchy syntax rules."""

    def __init__(self, message: str, location_id: LocationId | None = None) -> None:
        super().__init__(message, location_id)
        self.location_id = location_id


class CHSemanticError(ConceptHierarchyError):
    """Raised when the hierarchy is syntactically valid but semantically incorrect."""

    def __init__(self, message: str, location_id: LocationId | None = None) -> None:
        super().__init__(message, location_id)
        self.location_id = location_id


class CodegenError(ConceptHierarchyError):
    """Raised when code generation fails for a valid hierarchy."""
