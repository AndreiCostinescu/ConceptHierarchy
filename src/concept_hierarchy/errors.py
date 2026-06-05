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


class ConceptHierarchyError(Exception):
    """Base class for all ConceptHierarchy compiler errors."""


class SyntaxError(ConceptHierarchyError):
    """Raised when the JSON definition violates ConceptHierarchy syntax rules."""

    def __init__(self, message: str, location: str = "") -> None:
        self.location = location
        full = f"[{location}] {message}" if location else message
        super().__init__(full)


class SemanticError(ConceptHierarchyError):
    """Raised when the hierarchy is syntactically valid but semantically incorrect."""

    def __init__(self, message: str, node: str = "") -> None:
        self.node = node
        full = f"[{node}] {message}" if node else message
        super().__init__(full)


class CodegenError(ConceptHierarchyError):
    """Raised when code generation fails for a valid hierarchy."""
