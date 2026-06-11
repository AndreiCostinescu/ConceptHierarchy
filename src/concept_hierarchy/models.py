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
"""

from __future__ import annotations

import os

from concept_hierarchy.definitions.concept_definition import ConceptDefinition
from concept_hierarchy.definitions.global_variable_definition import GlobalVariableDefinition


class ConceptHierarchyModel:
    """Root model filled by the model checker and consumed by validation / codegen."""

    @staticmethod
    def create_by_parser(concept_hierarchy_file: str, parse_options: dict[str, object] = None) -> ConceptHierarchyModel:
        """Parse a JSON-decoded dict into a :class:`ConceptHierarchyModel`.

        Parameters
        ----------
        concept_hierarchy_file : str:
            The file to read and interpret.
        parse_options : dict[str, object]
            The options to pass to :func:`~concept_hierarchy.model.ConceptHierarchyModel`.

        Returns
        -------
        ConceptHierarchyModel
        """
        if not isinstance(concept_hierarchy_file, str):
            raise RuntimeError(
                f"The file path to the Concept Hierarchy definition must be a string, not {concept_hierarchy_file!r}."
            )
        if parse_options is None:
            parse_options = {}

        if "path_to_root_dir" in parse_options:
            path_to_root_dir = parse_options["path_to_root_dir"]
        else:
            path_to_root_dir = os.path.dirname(concept_hierarchy_file)

        return ConceptHierarchyModel(concept_hierarchy_file, path_to_root_dir)

    @staticmethod
    def create_from_data(data):
        ch = ConceptHierarchyModel("", "")
        ch.definition_data = data
        return ch

    def __init__(self, file: str, path_to_root_dir: str):
        self.file: str = file
        self.path_to_root_dir: str = path_to_root_dir
        self.definition_data = None

        self.root_concept_name: str = "Concept"

        self.checked_structure = False
        self.checked = False
        self.name: str = ""
        self.concepts: dict[str, ConceptDefinition] = {}
        self.instances: dict[str, GlobalVariableDefinition] = {}
        self.metadata: dict[str, str] = {}

        self.domain_concepts: set[str] = set()
        self.value_domains: set[str] = set()
        self.functions: set[str] = set()

        self.all_concept_parents: dict[str, set[str]] = {}  # does not include the concept itself
        self.topo_sort_concept_parents: dict[str, list[str]] = {}  # does not include the concept itself
        self.concept_topo_sort: list[str] = []

        self.all_domain_concept_properties: dict[str, str] = {}  # prop_name -> defining concept
        self.all_domain_concept_functions: dict[str, str] = {}  # func_name -> defining concept

        self.default_serializations: dict[str, str] = {}

        self.external_concept_data_resolver = None

    def assert_structure(self):
        if not self.checked_structure:
            raise RuntimeError("Can't verify Concept Hierarchy relations before it has been processed!")

    def is_concept(self, c: str) -> bool:
        self.assert_structure()
        return c in self.concepts

    def is_domain_concept(self, c: str) -> bool:
        return not self.is_value_domain(c)

    def is_value_domain(self, c: str) -> bool:
        return self.is_concept("ValueDomain") and self.is_a_subconcept_of_b(c, "ValueDomain", include_self=True)

    def is_pure_value_domain(self, c: str) -> bool:
        return self.is_value_domain(c) and not self.is_function(c)

    def is_function(self, c: str) -> bool:
        return self.is_concept("Function") and self.is_a_subconcept_of_b(c, "Function", include_self=True)

    def is_variable(self, v: str) -> bool:
        self.assert_structure()
        return v in self.instances

    def is_a_subconcept_of_b(self, a: str, b: str, *, include_self: bool) -> bool:
        self.assert_structure()
        if not self.is_concept(a):
            raise RuntimeError(f"{a!r} is not the name of a concept in the Concept Hierarchy!")
        if not self.is_concept(b):
            raise RuntimeError(f"{b!r} is not the name of a concept in the Concept Hierarchy!")
        if include_self and a == b:
            return True
        return b in self.all_concept_parents[a]

    def concept_names(self) -> list[str]:
        return self.concept_topo_sort

    def instance_names(self) -> list[str]:
        return [c for c in self.instances]

    def __repr__(self) -> str:
        return (
            f"ConceptHierarchyModel(name={self.name!r}, concepts={self.concept_names()!r}, "
            f"instances={self.instance_names()!r}, metadata={self.metadata!r})"
        )
