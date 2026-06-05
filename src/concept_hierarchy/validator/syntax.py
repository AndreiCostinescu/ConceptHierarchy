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
syntax.py — Syntax-level validation of a parsed ConceptHierarchyModel.

"Syntax" here means rules that can be checked on a single concept in
isolation (naming conventions, required fields, etc.) — i.e. anything
that does *not* require cross-concept lookups.
"""

from __future__ import annotations

import re

from concept_hierarchy.errors import SyntaxError as CHSyntaxError
from concept_hierarchy.models import ConceptHierarchyModel

# Identifiers must start with a letter and contain only alphanumerics / underscores.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def check_syntax(model: ConceptHierarchyModel) -> None:
    """Validate syntax rules on *model*, raising on the first violation.

    Parameters
    ----------
    model:
        A :class:`~concept_hierarchy.models.ConceptHierarchyModel` produced
        by the parser.

    Raises
    ------
    concept_hierarchy.errors.SyntaxError
    """
    _check_hierarchy_name(model)
    for concept in model.concepts:
        _check_concept_name(concept.name)
        for attr_name in concept.attributes:
            _check_attribute_name(concept.name, attr_name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_hierarchy_name(model: ConceptHierarchyModel) -> None:
    if not _IDENTIFIER_RE.match(model.name):
        raise CHSyntaxError(
            f"Hierarchy name {model.name!r} is not a valid identifier "
            "(must match [A-Za-z][A-Za-z0-9_]*).",
            location="name",
        )


def _check_concept_name(name: str) -> None:
    if not _IDENTIFIER_RE.match(name):
        raise CHSyntaxError(
            f"Concept name {name!r} is not a valid identifier.",
            location=f"concept:{name}",
        )


def _check_attribute_name(concept_name: str, attr_name: str) -> None:
    if not _IDENTIFIER_RE.match(attr_name):
        raise CHSyntaxError(
            f"Attribute name {attr_name!r} in concept {concept_name!r} is not "
            "a valid identifier.",
            location=f"concept:{concept_name}.attributes.{attr_name}",
        )
