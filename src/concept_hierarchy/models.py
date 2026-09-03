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
from concept_hierarchy.errors import LocationId, LocationIdLike


class ConceptHierarchyModel:
    """Root model filled by the model checker and consumed by validation / codegen."""

    model_name: str = "name"
    model_metadata: str = "metadata"
    model_concepts: str = "concepts"
    model_instances: str = "instances"
    model_concepts_external: str = "external"
    model_keywords: set[str] = {model_name, model_metadata, model_concepts, model_instances}
    default_root_concept_name: str = "Concept"
    default_value_domain_name: str = "ValueDomain"
    default_function_name: str = "Function"

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

        self.root_concept_name: str = ConceptHierarchyModel.default_root_concept_name

        self.checked_structure = False
        self.checked = False
        self.name: str = ""
        self.concepts: dict[str, ConceptDefinition] = {}
        self.instances: dict[str, GlobalVariableDefinition] = {}
        self.concept_aliases: dict[str, str] = {}
        """
        Alias name -> the canonical concept it names.

        An alias is a *name*, not an entity: it is never an entry of :attr:`concepts`, never takes part in
        the subconcept relation, and is never a definition site. It is resolved at the lookup boundary --
        see :meth:`canonical_concept_name` and :meth:`concept` -- rather than by rewriting the definition
        data, so a use of the alias is still visible in the diagnostics of the use site. Chains are already
        followed here: the value is the concept that ultimately defines the data, not the next link.
        """
        self.variable_aliases: dict[str, str] = {}
        """
        Alias name -> the canonical global variable it names.

        As for :attr:`concept_aliases`: never an entry of :attr:`instances`, resolved at the lookup
        boundary by :meth:`canonical_variable_name`, and already resolved through any chain. Because there
        is one entry rather than a copy per name, an alias and its target are *the same*
        ``GlobalVariableData`` -- writing through either name writes the one object, with no propagation
        step.
        """
        self.metadata: dict[str, str] = {}

        self.domain_concepts: set[str] = set()
        self.value_domains: set[str] = set()
        self.functions: set[str] = set()

        self.all_concept_parents: dict[str, set[str]] = {}  # does not include the concept itself
        self.topo_sort_concept_parents: dict[str, list[str]] = {}  # does not include the concept itself
        self.concept_topo_sort: list[str] = []
        self.defined_direct_children: dict[str, set[str]] = {}  # this is the inverse "directParents" relation
        self.all_declared_distinct_pairs: set[tuple[str, str]] = set()
        """Entries are sorted (alphabetically) to eliminate the reflexivity of the "distinct" relation."""

        self.all_domain_concept_properties: dict[str, str] = {}  # prop_name -> defining concept
        self.all_domain_concept_functions: dict[str, str] = {}  # func_name -> defining concept

        self.default_serializations: dict[str, str] = {}

        self.external_concept_data_resolver = None

    def __repr__(self) -> str:
        return (
            f"ConceptHierarchyModel({ConceptHierarchyModel.model_name}={self.name!r}, "
            f"{ConceptHierarchyModel.model_concepts}={self.concept_names()!r}, "
            f"{ConceptHierarchyModel.model_instances}={self.instance_names()!r}, "
            f"{ConceptHierarchyModel.model_metadata}={self.metadata!r})"
        )

    def assert_structure(self):
        if not self.checked_structure:
            raise RuntimeError("Can't verify Concept Hierarchy relations before it has been processed!")

    def canonical_concept_name(self, c: str) -> str:
        """
        The name under which ``c`` is defined: ``c`` itself, or -- if ``c`` is an alias -- what it names.

        Every map keyed by a concept name is keyed by the canonical one, so a name that came from the
        user's JSON must pass through here before it indexes one. Names taken from an already-canonical
        source (:attr:`concept_topo_sort`, a concept's ``parents``, ...) need not.
        """
        return self.concept_aliases.get(c, c)

    def get_concept_definition(
        self, name: str, location_id: LocationId | LocationIdLike | None = None
    ) -> tuple[ConceptDefinition, LocationId]:
        """
        The definition ``name`` denotes, and the location to blame in an error about *this* use of it.

        Reaching a concept through an alias annotates the **use site** with ``ref:<alias>``: an alias has no
        definition of its own to annotate, and the use site is where a reader has to look to see that an
        alias was written at all. ``location_id`` is not modified; the annotated copy is returned.

        :raises RuntimeError: if ``name`` is neither a concept nor an alias of one.
        """
        self.assert_structure()
        location = LocationId(location_id) if location_id is not None else LocationId()
        canonical = self.concept_aliases.get(name)
        if canonical is None:
            if name not in self.concepts:
                raise RuntimeError(f"{name!r} is not the name of a concept in the Concept Hierarchy!")
            return self.concepts[name], location
        return self.concepts[canonical], location + ["ref:" + name]

    def is_concept(self, c: str) -> bool:
        self.assert_structure()
        return c in self.concepts or c in self.concept_aliases

    def is_domain_concept(self, c: str) -> bool:
        return not self.is_value_domain(c)

    def is_value_domain(self, c: str) -> bool:
        return self.is_concept(ConceptHierarchyModel.default_value_domain_name) and self.is_a_subconcept_of_b(
            c, ConceptHierarchyModel.default_value_domain_name, include_self=True
        )

    def is_pure_value_domain(self, c: str) -> bool:
        return self.is_value_domain(c) and not self.is_function(c)

    def is_function(self, c: str) -> bool:
        return self.is_concept(ConceptHierarchyModel.default_function_name) and self.is_a_subconcept_of_b(
            c, ConceptHierarchyModel.default_function_name, include_self=True
        )

    def canonical_variable_name(self, v: str) -> str:
        """
        The name under which the global variable ``v`` is defined: ``v`` itself, or what it aliases.

        The counterpart of :meth:`canonical_concept_name`, for :attr:`instances`.
        """
        return self.variable_aliases.get(v, v)

    def get_variable_definition(
        self, name: str, location_id: LocationIdLike | None = None
    ) -> tuple[GlobalVariableDefinition, LocationId]:
        """
        The definition ``name`` denotes, and the location to blame in an error about *this* use of it.

        Reaching a variable through an alias annotates the **use site** with ``ref:<alias>``: an alias has no
        definition of its own to annotate, and the use site is where a reader has to look to see that an
        alias was written at all. ``location_id`` is not modified; the annotated copy is returned.

        :raises RuntimeError: if ``name`` is neither a variable nor an alias of one.
        """
        self.assert_structure()
        location = LocationId(location_id) if location_id is not None else LocationId()
        canonical = self.variable_aliases.get(name)
        if canonical is None:
            if name not in self.instances:
                raise RuntimeError(f"{name!r} is not the name of a variable in the Concept Hierarchy!")
            return self.instances[name], location
        return self.instances[canonical], location + ["ref:" + name]

    def is_variable(self, v: str) -> bool:
        self.assert_structure()
        return v in self.instances or v in self.variable_aliases

    def is_a_subconcept_of_b(self, a: str, b: str, *, include_self: bool) -> bool:
        self.assert_structure()
        if not self.is_concept(a):
            raise RuntimeError(f"{a!r} is not the name of a concept in the Concept Hierarchy!")
        if not self.is_concept(b):
            raise RuntimeError(f"{b!r} is not the name of a concept in the Concept Hierarchy!")
        # either side may be written as an alias; an alias and its target are the same concept
        a, b = self.canonical_concept_name(a), self.canonical_concept_name(b)
        if include_self and a == b:
            return True
        return b in self.all_concept_parents[a]

    def concept_names(self) -> list[str]:
        return self.concept_topo_sort

    def instance_names(self) -> list[str]:
        return [c for c in self.instances]

    def add_distinct_pair(self, concept_name_1: str, concept_name_2: str):
        if concept_name_1 < concept_name_2:
            self.all_declared_distinct_pairs.add((concept_name_1, concept_name_2))
        else:
            self.all_declared_distinct_pairs.add((concept_name_2, concept_name_1))
