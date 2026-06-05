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
parser.py — Convert a raw (already JSON-decoded) dict into a
:class:`~concept_hierarchy.models.ConceptHierarchyModel`.

The parser does *not* perform semantic validation; it only converts the
JSON structure into typed Python objects and raises
:class:`~concept_hierarchy.errors.SyntaxError` for structural issues
(missing required keys, wrong value types, etc.).
"""

from __future__ import annotations

from typing import Any, Dict

from frozendict import frozendict

from concept_hierarchy.errors import SyntaxError as CHSyntaxError
from concept_hierarchy.models import Concept, ConceptHierarchyModel


def parse(raw: Dict[str, Any]) -> ConceptHierarchyModel:
    """Parse a JSON-decoded dict into a :class:`ConceptHierarchyModel`.

    Parameters
    ----------
    raw:
        A Python dict obtained by ``json.load`` / ``json.loads``.

    Returns
    -------
    ConceptHierarchyModel

    Raises
    ------
    concept_hierarchy.errors.SyntaxError
        On structural problems (missing keys, wrong types, …).
    """
    if not isinstance(raw, dict):
        raise CHSyntaxError("Top-level JSON value must be an object (dict).")

    # -- hierarchy name --------------------------------------------------
    if "name" not in raw:
        raise CHSyntaxError("Missing required top-level key: 'name'.")
    name = raw["name"]
    if not isinstance(name, str) or not name.strip():
        raise CHSyntaxError("'name' must be a non-empty string.")

    # -- metadata (optional) --------------------------------------------
    raw_meta = raw.get("metadata", {})
    if not isinstance(raw_meta, dict):
        raise CHSyntaxError("'metadata' must be an object (dict).")
    metadata: frozendict[str, str] = frozendict(
        {str(k): str(v) for k, v in raw_meta.items()}
    )

    # -- concepts --------------------------------------------------------
    if "concepts" not in raw:
        raise CHSyntaxError("Missing required top-level key: 'concepts'.")
    raw_concepts = raw["concepts"]
    if not isinstance(raw_concepts, list):
        raise CHSyntaxError("'concepts' must be a JSON array.")

    concepts = tuple(_parse_concept(item, idx) for idx, item in enumerate(raw_concepts))

    return ConceptHierarchyModel(name=name, concepts=concepts, metadata=metadata)


def _parse_concept(raw: Any, idx: int) -> Concept:
    loc = f"concepts[{idx}]"

    if not isinstance(raw, dict):
        raise CHSyntaxError("Each concept must be a JSON object.", location=loc)

    # name
    if "name" not in raw:
        raise CHSyntaxError("Missing required key 'name'.", location=loc)
    c_name = raw["name"]
    if not isinstance(c_name, str) or not c_name.strip():
        raise CHSyntaxError("'name' must be a non-empty string.", location=loc)

    # parent (optional)
    parent = raw.get("parent", None)
    if parent is not None and not isinstance(parent, str):
        raise CHSyntaxError("'parent' must be a string or null.", location=loc)

    # attributes (optional)
    raw_attrs = raw.get("attributes", {})
    if not isinstance(raw_attrs, dict):
        raise CHSyntaxError("'attributes' must be an object (dict).", location=loc)
    for attr_name, attr_type in raw_attrs.items():
        if not isinstance(attr_type, str):
            raise CHSyntaxError(
                f"Attribute '{attr_name}': type must be a string.", location=loc
            )
    attributes: frozendict[str, str] = frozendict(raw_attrs)

    # description (optional)
    description = raw.get("description", "")
    if not isinstance(description, str):
        raise CHSyntaxError("'description' must be a string.", location=loc)

    return Concept(
        name=c_name,
        parent=parent,
        attributes=attributes,
        description=description,
    )
