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
models.py — Internal representation of a ConceptHierarchy.

All data structures are intentionally immutable (using frozendict where
mappings are needed) so that the validation and codegen stages can safely
share references without defensive copying.
"""

from typing import List, Optional, Tuple

from frozendict import frozendict


class Concept:
    """A single concept node in the hierarchy."""

    __slots__ = ("name", "parent", "attributes", "description")

    def __init__(
        self,
        name: str,
        parent: Optional[str],
        attributes: "frozendict[str, str]",
        description: str = "",
    ) -> None:
        self.name: str = name
        self.parent: Optional[str] = parent
        # mapping of attribute_name -> type_string
        self.attributes: "frozendict[str, str]" = attributes
        self.description: str = description

    def __repr__(self) -> str:
        return (
            f"Concept(name={self.name!r}, parent={self.parent!r}, "
            f"attributes={dict(self.attributes)!r})"
        )


class ConceptHierarchyModel:
    """Root model produced by the parser and consumed by validation / codegen."""

    __slots__ = ("name", "concepts", "metadata")

    def __init__(
        self,
        name: str,
        concepts: Tuple[Concept, ...],
        metadata: "frozendict[str, str]",
    ) -> None:
        self.name: str = name
        # ordered tuple — order matters for topological checks
        self.concepts: Tuple[Concept, ...] = concepts
        self.metadata: "frozendict[str, str]" = metadata

    def concept_names(self) -> List[str]:
        return [c.name for c in self.concepts]

    def __repr__(self) -> str:
        return (
            f"ConceptHierarchyModel(name={self.name!r}, "
            f"concepts={self.concept_names()!r})"
        )
