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
Shared support for tests that build and check a whole Concept Hierarchy.

A test writes a hierarchy the way a user does -- as JSON data -- and runs the real
:class:`ConceptHierarchyChecker` over it. :func:`check_concepts` is the usual entry point; it splices the
test's concepts into :data:`CH_PRELUDE` so a test only has to spell out what it is actually about.
"""

from __future__ import annotations

from copy import deepcopy

from concept_hierarchy.data.contexts.context import ConceptHierarchyContext
from concept_hierarchy.definitions.concept_hierarchy import ConceptHierarchyDefinition
from concept_hierarchy.validator.checker import ConceptHierarchyChecker

CH_PRELUDE: dict[str, dict] = {
    "Concept": {},
    "ValueDomain": {"directParents": ["Concept"], "data": {}, "abstract": True},
    "Numeric": {"directParents": ["ValueDomain"], "data": {}},
    "Number": {
        "directParents": ["Numeric"],
        "data": {"defaultSerialization": "number", "instantiation": [[[], "number"]]},
    },
    "Integer": {"directParents": ["Number"], "data": {"defaultSerialization": "integer", "instantiation": "integer"}},
    "String": {
        "directParents": ["ValueDomain"],
        "data": {"defaultSerialization": "string", "instantiation": {"type": "string", "pattern": "^s:"}},
    },
    "Function": {"directParents": ["ValueDomain"], "data": {}, "abstract": True},
    "FunctionReturning": {"directParents": ["Function"], "data": {"templateContext": ["T"]}, "abstract": True},
    # the default value domain of a DomainConcept's `functions` members
    "CustomFunction": {"directParents": ["ValueDomain"], "data": {}},
}
"""The built-in concepts essentially every Concept Hierarchy needs."""


def build_hierarchy(concepts: dict[str, dict], instances: dict | None = None, name: str = "TestHierarchy") -> dict:
    """
    Assemble a Concept Hierarchy definition from :data:`CH_PRELUDE` plus the given concepts.

    The result is a deep copy. The checker edits the definition data in place -- a cancelled specialization
    keyword is popped out of it, for one -- so tests that share a fixture dict between checks would
    otherwise corrupt each other.
    """
    model_data: dict = {"name": name, "concepts": {**CH_PRELUDE, **concepts}}
    if instances is not None:
        model_data["instances"] = instances
    return deepcopy(model_data)


def check_hierarchy(model_data: dict, external_data: object = None) -> ConceptHierarchyContext:
    """
    Run the full checker over a Concept Hierarchy definition and return the checked context.

    The context exposes the parsed model at ``.model`` (``.domain_concepts``, ``.value_domains``,
    ``.functions``, ``.instances``) and the raw definitions at ``.ch.concepts``.

    :raises ConceptHierarchyError: if the hierarchy does not check, as for any invalid definition.
    """
    model = ConceptHierarchyDefinition.create_from_data(model_data)
    checker = ConceptHierarchyChecker(model, lambda _concept, _instance: external_data)
    checker.check()
    return checker.context


def check_concepts(concepts: dict[str, str | dict], instances: dict | None = None, **kwargs) -> ConceptHierarchyContext:
    """Shorthand for ``check_hierarchy(build_hierarchy(concepts, instances), ...)``."""
    return check_hierarchy(build_hierarchy(concepts, instances=instances), **kwargs)


def domain_concept(properties: dict | None = None, functions: dict | None = None, parents=("Concept",)) -> dict:
    """A DomainConcept definition with the given ``properties``/``functions`` blocks."""
    data: dict = {}
    if properties is not None:
        data["properties"] = properties
    if functions is not None:
        data["functions"] = functions
    return {"directParents": list(parents), "data": data}
