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
semantics.py — Semantic validation of a parsed ConceptHierarchyModel.

"Semantic" checks require cross-concept knowledge:
  - Every referenced parent must be declared as a concept.
  - No circular inheritance chains.
  - No duplicate concept names.
"""

from typing import Dict, List, Optional, Set

from concept_hierarchy.errors import SemanticError
from concept_hierarchy.models import ConceptHierarchyModel


def check_semantics(model: ConceptHierarchyModel) -> None:
    """Run all semantic checks on *model*.

    Raises
    ------
    concept_hierarchy.errors.SemanticError
    """
    _check_no_duplicate_names(model)
    _check_parents_exist(model)
    _check_no_cycles(model)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _check_no_duplicate_names(model: ConceptHierarchyModel) -> None:
    seen: Set[str] = set()
    for concept in model.concepts:
        if concept.name in seen:
            raise SemanticError(
                f"Duplicate concept name {concept.name!r}.",
                node=concept.name,
            )
        seen.add(concept.name)


def _check_parents_exist(model: ConceptHierarchyModel) -> None:
    known: Set[str] = {c.name for c in model.concepts}
    for concept in model.concepts:
        if concept.parent is not None and concept.parent not in known:
            raise SemanticError(
                f"Parent {concept.parent!r} is not defined in the hierarchy.",
                node=concept.name,
            )


def _check_no_cycles(model: ConceptHierarchyModel) -> None:
    """Detect cycles using DFS with three-colour marking."""
    parent_map: Dict[str, Optional[str]] = {
        c.name: c.parent for c in model.concepts
    }

    WHITE, GREY, BLACK = 0, 1, 2
    colour: Dict[str, int] = {name: WHITE for name in parent_map}

    def visit(name: str) -> None:
        if colour[name] == BLACK:
            return
        if colour[name] == GREY:
            raise SemanticError(
                f"Circular inheritance detected involving concept {name!r}.",
                node=name,
            )
        colour[name] = GREY
        parent = parent_map[name]
        if parent is not None:
            visit(parent)
        colour[name] = BLACK

    for name in parent_map:
        visit(name)
