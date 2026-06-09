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
checker.py — Syntax- and semantic-level validation of a parsed ConceptHierarchyModel.
"""

import os

from concept_hierarchy.definitions.concept_definition import ConceptDefinition
from concept_hierarchy.definitions.concept_definition_domain_concept import DomainConceptDefinition
from concept_hierarchy.definitions.concept_definition_functions import FunctionDefinition
from concept_hierarchy.definitions.concept_definition_hidden_implementation import HiddenImplementationDefinition
from concept_hierarchy.definitions.concept_definition_value_domain import ValueDomainDefinition
from concept_hierarchy.definitions.definition import ConceptHierarchyDefinition
from concept_hierarchy.definitions.global_variable_definition import GlobalVariableDefinition
from concept_hierarchy.definitions.utils import check_ch_name
from concept_hierarchy.errors import CHSemanticError, CHSyntaxError
from concept_hierarchy.models import ConceptHierarchyModel
from concept_hierarchy.utils import (
    join_path,
    read_external_data_content,
    read_json_file,
    sanitize_relative_path,
    tab,
    topological_sort,
)


class ConceptHierarchyChecker:
    @staticmethod
    def read_concept_hierarchy(file: str, path_to_root_dir: str) -> object:
        def create_json(file_name: str) -> object:
            res = read_json_file(file_name)
            if "external" in res:
                assert isinstance(res["external"], list)
                for sub_file_name in res["external"]:
                    if os.path.isabs(sub_file_name):
                        res.update(create_json(sub_file_name))
                    else:
                        rel_sub_file_name = sanitize_relative_path(join_path(path_to_root_dir, sub_file_name))
                        res.update(create_json(rel_sub_file_name))
                res.pop("external")
            return res

        return create_json(file)

    @staticmethod
    def check_cycles_in_references_based_on_defined(
        references: dict[str, ConceptHierarchyDefinition], definitions: dict, data_type: str
    ) -> dict[str, str | None]:
        mapped_references: dict[str, str | None] = {x: None for x in definitions}
        total_length = len(definitions) + len(references)
        prev_length = len(mapped_references)
        while True:
            for concept_name, concept_def in references.items():
                if concept_name not in mapped_references:
                    if concept_def.is_reference_to in mapped_references:
                        mapped_references[concept_name] = mapped_references[concept_def.is_reference_to]
            current_length = len(mapped_references)
            assert current_length <= total_length
            if current_length == total_length:
                break
            elif current_length == prev_length:
                raise CHSemanticError(
                    f"There is a cycle in the {data_type[:-1]} references: {set(references.keys())!r}",
                    location_id=[data_type],
                )
            prev_length = current_length
        return mapped_references

    def __init__(self, concept_hierarchy_data: ConceptHierarchyModel):
        self.ch = concept_hierarchy_data
        self.ch.external_concept_data_resolver = lambda x, y: read_external_data_content(
            x, y, self.ch.file, self.ch.path_to_root_dir
        )

    def check_structure(self):
        if self.ch.checked:
            return

        if self.ch.definition_data is None:
            self.ch.definition_data = ConceptHierarchyChecker.read_concept_hierarchy(
                self.ch.file, self.ch.path_to_root_dir
            )
        if not isinstance(self.ch.definition_data, dict):
            raise CHSyntaxError("Concept Hierarchy definition must be a JSON object!", location_id=[])
        concept_hierarchy = self.ch.definition_data

        # interpret either as a meta-definition, or a direct definition of concepts
        ch_keys = set(concept_hierarchy.keys())
        if not (ch_keys <= {"name", "metadata", "concepts", "instances"}):
            # interpret this as a definition of concepts
            concept_hierarchy = {"concepts": concept_hierarchy}

        # -- hierarchy name --------------------------------------------------

        self.ch.name = concept_hierarchy.get("name", "ConceptHierarchy")
        if not check_ch_name(self.ch.name, allow_starting_with_underscore=True):
            raise CHSyntaxError(
                f"Concept Hierarchy name {self.ch.name!r} must be a non-empty, non-digit-starting string containing "
                f"only alphanumeric characters or '_'.",
                location_id=["name"],
            )

        # -- metadata (optional) --------------------------------------------
        raw_meta = concept_hierarchy.get("metadata", {})
        if not isinstance(raw_meta, dict):
            raise CHSyntaxError("Concept Hierarchy 'metadata' must be a JSON object.", location_id=["metadata"])
        self.ch.metadata = {str(k): str(v) for k, v in raw_meta.items()}

        # -- concepts --------------------------------------------------------
        if "concepts" not in concept_hierarchy:
            raise CHSyntaxError("Missing required top-level key: 'concepts'.", location_id=["concepts"])
        concept_definition = concept_hierarchy["concepts"]
        # type checks for concept_definition data
        if not isinstance(concept_definition, dict):
            raise CHSyntaxError(
                f"Concept Hierarchy 'concepts' data must be a JSON object of concept definitions, not "
                f"{concept_definition!r}.",
                location_id=["concepts"],
            )
        concepts_referencing_others: dict[str, ConceptDefinition] = {}
        defined_concepts: dict[str, ConceptDefinition] = {}
        for concept_name, concept_def in concept_definition.items():  # type: str, object
            concept_definition = ConceptDefinition(
                concept_name,
                concept_def,
                external_data_resolver=self.ch.external_concept_data_resolver,
            )
            if concept_definition.is_reference():
                concepts_referencing_others[concept_name] = concept_definition
            else:
                defined_concepts[concept_name] = concept_definition
        mapped_concepts = ConceptHierarchyChecker.check_cycles_in_references_based_on_defined(
            concepts_referencing_others, defined_concepts, "concepts"
        )
        assert all(mapped_concepts[x] is None for x in defined_concepts)
        for referencing_concept in concepts_referencing_others:
            referenced_instance_name = mapped_concepts[referencing_concept]
            assert referenced_instance_name is not None
            referenced_concept = defined_concepts[referenced_instance_name]
            defined_concepts[referencing_concept] = ConceptDefinition(
                referencing_concept,
                referenced_concept.definition_data,
                external_data_resolver=self.ch.external_concept_data_resolver,
            )

        # -- instances --------------------------------------------------------
        instance_definition = concept_hierarchy.get("instances", {})
        if not isinstance(instance_definition, dict):
            raise CHSyntaxError(
                f"Concept Hierarchy 'instances' data must be a JSON object of definitions of instances, "
                f"i.e. global variables, not {instance_definition!r}.",
                location_id=["instances"],
            )
        instances_referencing_others: dict[str, GlobalVariableDefinition] = {}
        defined_instances: dict[str, GlobalVariableDefinition] = {}
        for variable_name, variable_def in instance_definition.items():  # type: str, object
            variable_definition = GlobalVariableDefinition(variable_name, variable_def)
            if variable_definition.is_reference():
                instances_referencing_others[variable_name] = variable_definition
            else:
                defined_instances[variable_name] = variable_definition
        mapped_instances = ConceptHierarchyChecker.check_cycles_in_references_based_on_defined(
            instances_referencing_others, defined_instances, "instances"
        )
        assert all(mapped_instances[x] is None for x in defined_instances)
        for referencing_instance in instances_referencing_others:
            referenced_instance_name = mapped_concepts[referencing_instance]
            assert referenced_instance_name is not None
            referenced_instance = defined_instances[referenced_instance_name]
            defined_instances[referenced_instance_name] = GlobalVariableDefinition(
                referencing_instance, referenced_instance.definition_data
            )

        # perform topological sort of concepts and instances
        concept_parent_mapping = {c_name: c.parents for c_name, c in defined_concepts.items()}
        for c_name, parents in concept_parent_mapping.items():
            for index, p_name in enumerate(parents):
                if p_name not in defined_concepts:
                    raise CHSemanticError(
                        f"The parent {p_name!r} of concept {c_name!r} is not defined in the hierarchy.",
                        location_id=["concepts", c_name, "directParents", index],
                    )
        try:
            self.ch.concept_topo_sort, roots = topological_sort(concept_parent_mapping)
            if self.ch.root_concept_name in roots and len(roots) != 1:
                raise CHSemanticError(f"Concept hierarchy has multiple roots: {roots!r}", location_id=["concepts"])
            elif self.ch.root_concept_name not in roots:
                for root in roots:
                    defined_concepts[root].parents.append(self.ch.root_concept_name)
                self.ch.concept_topo_sort = [self.ch.root_concept_name] + self.ch.concept_topo_sort
                defined_concepts[self.ch.root_concept_name] = ConceptDefinition(self.ch.root_concept_name, {})
                roots = [self.ch.root_concept_name]
            assert len(roots) == 1
            root = roots[0]
            if root != self.ch.root_concept_name:
                raise CHSemanticError(
                    f'The root concept of the Concept Hierarchy must be called "{self.ch.root_concept_name}", not '
                    f"{root!r}!",
                    location_id=["concepts"],
                )
            defined_concepts[root].is_root = True
            self.ch.root_concept_name = root
        except RuntimeError as e:
            if str(e).startswith("Non-hierarchy structure detected! The following items form one or more cycles:"):
                raise CHSemanticError(
                    f"Cycles detected in Concept Hierarchy:\n{tab}{e!s}", location_id=["concepts"]
                ) from e
            raise e

        topo_index: dict[str, int] = {name: i for i, name in enumerate(self.ch.concept_topo_sort)}
        all_ancestors: dict[str, set[str]] = {}
        all_concept_topo_sort_parents: dict[str, list[str]] = {}
        for concept_name in self.ch.concept_topo_sort:
            direct_parents = defined_concepts[concept_name].parents
            ancestors: set[str] = set(direct_parents)
            for p in direct_parents:
                ancestors |= all_ancestors[p]
            all_ancestors[concept_name] = ancestors
            all_concept_topo_sort_parents[concept_name] = sorted(ancestors, key=topo_index.__getitem__)
        self.ch.all_concept_parents = all_ancestors
        self.ch.topo_sort_concept_parents = all_concept_topo_sort_parents

        self.ch.concepts = defined_concepts

        for instance_name in defined_instances:
            if instance_name in defined_concepts:
                raise CHSemanticError(
                    f"The global variable name {instance_name!r} is also a concept name!\n\tThis can create ambiguity! "
                    f"Please rename the global variable name!",
                    location_id=["instances", instance_name],
                )
        self.ch.instances = defined_instances
        self.ch.checked_structure = True

    def check_after_parsing_concepts(self):
        # promote to subconcepts the concept definitions
        for c_name, c in self.ch.concepts.items():
            assert c.is_root == (c.parents == [])
            if self.ch.is_function(c_name):
                self.ch.concepts[c_name] = FunctionDefinition.from_node(c)
            elif self.ch.is_value_domain(c_name):
                self.ch.concepts[c_name] = ValueDomainDefinition.from_node(c)
            elif self.ch.is_domain_concept(c_name):
                self.ch.concepts[c_name] = DomainConceptDefinition.from_node(c)
            else:
                raise CHSemanticError(
                    f"Found concept {c_name} with parents {c.parents!r} that is neither a Function, ValueDomain, "
                    f"not a domain concept!",
                    location_id=["concepts", c_name, "directParents"],
                )
        # check domain_concept, value_domain, and function data!
        for c_name, c in self.ch.concepts.items():
            c.concept_data_check()
            if isinstance(c, HiddenImplementationDefinition):
                for t_arg_name in c.template_argument_order:
                    if t_arg_name in self.ch.concepts:
                        raise CHSemanticError(
                            f"The template argument name {t_arg_name!r} of {c_name} is also the name of a defined "
                            f"concept in this Concept Hierarchy!"
                            f"\n\tThis can cause ambiguity in the template argument's constraint formulae definition, "
                            f"in template instantiations and in template substitutions."
                            f"\nPlease rename the template argument!",
                            location_id=["concepts", c_name, "data", "templateArguments"],
                        )
            if isinstance(c, FunctionDefinition):
                for eval_arg_name in c.evaluation_interface:
                    if eval_arg_name in self.ch.instances:
                        raise CHSemanticError(
                            f"The name of the evaluation argument {eval_arg_name!r} of {c_name} is also the name of a "
                            f"defined global variable (global instance) in this Concept Hierarchy."
                            f"\n\tThis can cause ambiguity in the context of the FunctionComposition of Function "
                            f"procedures, inversions, and variations!"
                            f"\nPlease rename the Function argument or the global variable!",
                            location_id=["concepts", c_name, "data", "interface"],
                        )
            if isinstance(c, DomainConceptDefinition):
                # check unique property names, unique function names, distinct function and property names,
                # and non-ambiguous definitions of properties or functions with the same name as a global variable
                for prop_name in c.properties:
                    if prop_name in self.ch.instances:
                        raise CHSemanticError(
                            f"The name of the concept property {prop_name!r} of {c_name} is also the name of a defined "
                            f"global variable (global instance) in this Concept Hierarchy."
                            f"\n\tThis can cause ambiguity in the context of property hooks, computations, concept "
                            f"functions, and management functions."
                            f"\nPlease rename the property or the global variable!",
                            location_id=["concepts", c_name, "data", "properties", prop_name],
                        )
                    if prop_name in self.ch.all_domain_concept_properties:
                        raise CHSemanticError(
                            f"The property {prop_name} is defined in multiple places!\nFound (non-inclusively) in "
                            f"{c_name!r} and in {self.ch.all_domain_concept_properties[prop_name]!r}."
                            f"\n\tPlease move the property to a common concept or rename one of them so that property "
                            f"names are unique!",
                            location_id=["concepts", c_name, "data", "properties", prop_name],
                        )
                    if prop_name in self.ch.all_domain_concept_functions:
                        raise CHSemanticError(
                            f"The property {prop_name} is defined in multiple places!\nFound (non-inclusively) in "
                            f"{c_name!r} as a property and in {self.ch.all_domain_concept_functions[prop_name]!r} as a "
                            f"function!\n\tPlease rename one of them so that property names are distinct from function "
                            f"names!",
                            location_id=["concepts", c_name, "data", "properties", prop_name],
                        )
                for func_name in c.functions:
                    if func_name in self.ch.instances:
                        raise CHSemanticError(
                            f"The name of the concept function {func_name!r} of {c_name} is also the name of a defined "
                            f"global variable (global instance) in this Concept Hierarchy."
                            f"\n\tThis can cause ambiguity in the context of property hooks, computations, concept "
                            f"functions, and management functions."
                            f"\nPlease rename the function or the global variable!",
                            location_id=["concepts", c_name, "data", "functions", func_name],
                        )
                    if func_name in self.ch.all_domain_concept_functions:
                        raise CHSemanticError(
                            f"The function {func_name} is defined in multiple places!\nFound (non-inclusively) in "
                            f"{c_name!r} and in {self.ch.all_domain_concept_functions[func_name]!r}."
                            f"\n\tPlease move the function to a common concept or rename one of them so that function "
                            f"names are unique!",
                            location_id=["concepts", c_name, "data", "functions", func_name],
                        )
                    if func_name in self.ch.all_domain_concept_properties:
                        raise CHSemanticError(
                            f"The function {func_name} is defined in multiple places!\nFound (non-inclusively) in "
                            f"{c_name!r} as a function and in {self.ch.all_domain_concept_properties[func_name]!r} as a"
                            f" property!\n\tPlease rename one of them so that function names are distinct from property"
                            f" names!",
                            location_id=["concepts", c_name, "data", "functions", func_name],
                        )


def check_model(model: ConceptHierarchyModel) -> None:
    """Validate syntax and semantic rules on *model*, raising on the first violation.

    Parameters
    ----------
    model:
        A :class:`~concept_hierarchy.models.ConceptHierarchyModel` produced by the parser.

    Raises
    ------
    concept_hierarchy.errors.SyntaxError
    concept_hierarchy.errors.SemanticError
    """
    checker = ConceptHierarchyChecker(model)
    checker.check_structure()
    checker.check_after_parsing_concepts()
