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
checker.py — Syntax- and semantic-level validation of a JSON-converted-to-python ConceptHierarchyDefinition.
"""

import os
from pathlib import Path
from typing import Callable

from frozendict import frozendict

from concept_hierarchy.data.concept_hierarchy import (
    ConceptData,
    ConceptHierarchy,
    DomainConceptData,
    FunctionData,
    ValueDomainData,
)
from concept_hierarchy.data.contexts.context import ConceptHierarchyContext, TemplateContext
from concept_hierarchy.data.parsers.type_parser import ParsedType, TypeParser
from concept_hierarchy.data.utils import UNINITIALIZED
from concept_hierarchy.data.validators.type_validator import parse_convert_type
from concept_hierarchy.definitions.concept_definition import ConceptDefinition
from concept_hierarchy.definitions.concept_definition_domain_concept import (
    DomainConceptDefinition,
    PropertyDefinitionKeywords,
)
from concept_hierarchy.definitions.concept_definition_functions import FunctionDefinition
from concept_hierarchy.definitions.concept_definition_hidden_implementation import HiddenImplementationDefinition
from concept_hierarchy.definitions.concept_definition_value_domain import ValueDomainDefinition
from concept_hierarchy.definitions.concept_hierarchy import ConceptHierarchyDefinition
from concept_hierarchy.definitions.global_variable_definition import GlobalVariableDefinition
from concept_hierarchy.definitions.utils import check_ch_name
from concept_hierarchy.errors import CHSemanticError, CHSyntaxError, ConceptHierarchyError, LocationId, PathPart
from concept_hierarchy.utils import (
    join_path,
    read_external_data_content,
    read_json_file,
    sanitize_relative_path,
    tab,
    topological_sort,
)
from concept_hierarchy.validator.concept_hierarchy_type_checks import (
    check_types_in_concept_hierarchy,
)
from concept_hierarchy.validator.domain_concept_specialization_checks import (
    process_specialization_for_domain_concepts,
)
from concept_hierarchy.validator.expression_checks import (
    check_expressions_in_concept_hierarchy,
)
from concept_hierarchy.validator.validators.constraint_formula_validator import ConstraintFormulaValidator
from concept_hierarchy.validator.validators.expression_validator import ExpressionValidator
from concept_hierarchy.validator.validators.type_application_constraints_validator import TypeApplicationValidator
from concept_hierarchy.validator.validators.type_validator import ConceptHierarchyTypeValidator
from concept_hierarchy.validator.validators.value_instantiation_schema_validator import SchemaValidator
from concept_hierarchy.validator.validators.value_instantiation_value_validator import ValueValidator
from concept_hierarchy.validator.value_domain_template_constraint_checks import (
    check_value_domain_template_constraint_formulae,
)


class ConceptHierarchyChecker:
    @staticmethod
    def read_concept_hierarchy(
        file: str, path_to_root_dir: str | None, visited_files: set[str], location_id: LocationId
    ) -> tuple[object, str, str]:
        if not os.path.isabs(file):
            if path_to_root_dir is None:
                path_to_root_dir = str(Path.cwd().absolute())
            file = sanitize_relative_path(join_path(path_to_root_dir, file))
        if file in visited_files:
            raise CHSemanticError(
                f"Found an infinite loop in file traversal when reading the Concept Hierarchy!\n"
                f'File "{file}" was read again while processing the data in itself!',
                location_id=location_id,
            )
        return read_json_file(file), file, str(Path(file).parent)

    @staticmethod
    def alias_target_names(target: str) -> tuple[str, ...]:
        """
        Every concept name the alias target ``target`` mentions: its head, plus each template argument, at
        any depth. Literal arguments (the ``3`` of ``Vector<3>``) name nothing and drop out.

        This is pure syntax -- :class:`TypeParser` needs no hierarchy -- which is what lets the alias graph
        be built in ``check_structure``, long before any type can be resolved.
        """
        parsed_targets = TypeParser(target, None).parse_types()
        names: list[str] = []

        def collect(parsed: ParsedType) -> None:
            names.append(parsed.clean_name)
            for template_argument in parsed.template_arguments or ():
                if isinstance(template_argument, ParsedType):
                    collect(template_argument)

        for parsed_target in parsed_targets:
            collect(parsed_target)
        return tuple(names)

    @staticmethod
    def order_aliases(
        alias_dependencies: dict[str, tuple[str, ...]],
        defined_names: dict,
        locations: dict[str, LocationId],
        base_location_id: LocationId,
        for_concepts: bool,
    ) -> list[str]:
        """
        Order the aliases so that each comes after every name it mentions, and reject cycles.

        ``alias_dependencies`` maps each alias to *all* the names its target mentions -- one, for a plain
        name; the head plus every template argument, for an applied type. Because every mention is an edge
        of one graph, a cycle is a cycle here whatever it runs through: a chain of names, a type's head, one
        of its template arguments, or any mixture of the three across the alias kinds.

        The order is what the resolution then walks, so an alias is only ever built out of parts that are
        already resolved -- and it is what tells a bare-name alias whether it ended up naming a concept or a
        type, which decides its kind.
        """
        entry_kind = "concept" if for_concepts else "instance"
        graph: dict[str, tuple[str, ...]] = {}
        for alias_name, mentioned_names in alias_dependencies.items():
            for mentioned_name in mentioned_names:
                if mentioned_name not in alias_dependencies and mentioned_name not in defined_names:
                    raise CHSemanticError(
                        f"The referenced {entry_kind} {mentioned_name!r} of {alias_name} does not exist in "
                        f"the Concept Hierarchy!",
                        location_id=locations[alias_name] + [alias_name],
                        part=PathPart.VALUE,
                    )
                # every mentioned name has to be a node of its own: `topological_sort` raises KeyError
                # for a name it has no entry for. A name that is not an alias is simply a leaf.
                graph.setdefault(mentioned_name, ())
            graph[alias_name] = mentioned_names
        try:
            ordered_names, _roots = topological_sort(graph)
        except RuntimeError as e:
            if str(e).startswith("Non-hierarchy structure detected!"):
                raise CHSemanticError(
                    f"There is a cycle in the {entry_kind} references:\n{tab}{e!s}",
                    location_id=base_location_id,
                    part=PathPart.VALUE,
                ) from e
            raise
        return [name for name in ordered_names if name in alias_dependencies]

    def __init__(
        self,
        concept_hierarchy_data: ConceptHierarchyDefinition,
        external_data_resolver: Callable[[str, str], object] | None = None,
    ):
        self.model = ConceptHierarchy(concept_hierarchy_data)
        if external_data_resolver is None:
            self.ch.external_concept_data_resolver = lambda x, y: read_external_data_content(
                x, y, self.ch.file, self.ch.path_to_root_dir
            )
        else:
            self.ch.external_concept_data_resolver = external_data_resolver
        self.context = ConceptHierarchyContext(self.model)
        self._type_alias_targets: dict[str, ParsedType] = {}
        """
        Alias name -> the applied type it names, parsed but not yet resolved, in resolution order.

        Filled in ``check_structure`` (which can only parse the target's *syntax*) and turned into
        ``ch.type_aliases`` in ``check_types``, once concepts and their template contexts exist. Python
        dicts keep insertion order, so iterating this is walking the alias graph's topological order.
        """

    @property
    def ch(self) -> ConceptHierarchyDefinition:
        return self.model.ch

    def merge_concept_hierarchy_structure_across_files(
        self,
        concept_hierarchy: dict[str, object],
        location_id: LocationId,
        file_dir: str | None,
        visited_files: set[str],
        external_from_concepts: bool,
        concept_hierarchy_concept_only_shortform: bool,
    ) -> tuple[dict, dict, dict[str, LocationId], dict[str, LocationId]]:
        """

        Parameters
        ----------
        concept_hierarchy
        file_dir
        visited_files
        location_id
        external_from_concepts: whether only concepts are allowed to be specified
        concept_hierarchy_concept_only_shortform: whether the shorthand form of the file was used for this data
        require_concepts_to_be_present: whether concepts must be present in this file

        Returns
        -------
        the concept definitions present in `concept_hierarchy`: concept_name -> concept_def_data
        the instance definitions present in `concept_hierarchy`: instance_name -> instance_def_data
        a map of the defined concepts to their definition location: concept_name -> concept_def_location_id
        a map of the defined instances to their definition location: instance_name -> instance_def_location_id
        """
        if any(
            x in concept_hierarchy
            for x in [ConceptHierarchyDefinition.model_name, ConceptHierarchyDefinition.model_name]
        ):
            if external_from_concepts:
                raise CHSyntaxError(
                    f"In external Concept Hierarchy concept files, the data must be either a JSON object of concepts "
                    f"with optionally other external concept files!\n"
                    f"Got top-level keys: {list(concept_hierarchy.keys())}",
                    location_id=location_id,
                    part=PathPart.VALUE,
                )
            else:
                raise CHSyntaxError(
                    f"In external Concept Hierarchy files, the data must be either a JSON object of concepts optionally"
                    f" containing other external concept files, or a JSON object only containing "
                    f'"{ConceptHierarchyDefinition.model_instances}", "{ConceptHierarchyDefinition.model_concepts}", or'
                    f'" {ConceptHierarchyDefinition.model_external}" keys!\n'
                    f"Got top-level keys: {list(concept_hierarchy.keys())}",
                    location_id=location_id,
                    part=PathPart.VALUE,
                )
        if ConceptHierarchyDefinition.model_instances in concept_hierarchy and external_from_concepts:
            raise CHSyntaxError(
                f"This external concept hierarchy was included as a concept-only file! "
                f'"{ConceptHierarchyDefinition.model_instances}" is not allowed here!'
            )

        # -- concepts --------------------------------------------------------
        concept_location_id: LocationId = location_id.copy()
        if not concept_hierarchy_concept_only_shortform:
            concept_location_id += [ConceptHierarchyDefinition.model_concepts]
        concept_definition = concept_hierarchy.get(ConceptHierarchyDefinition.model_concepts, {})
        # type checks for concept_definition data
        if not isinstance(concept_definition, dict):
            raise CHSyntaxError(
                f'Concept Hierarchy "{ConceptHierarchyDefinition.model_concepts}" data must be a JSON object of concept'
                f" definitions, not {concept_definition!r}.",
                location_id=concept_location_id,
                part=PathPart.VALUE,
            )
        concept_definition = {
            k: v for k, v in concept_definition.items() if k != ConceptHierarchyDefinition.model_concepts_external
        }  # exclude external from this dictionary
        concept_locations = {concept_name: concept_location_id for concept_name in concept_definition}

        # -- external concepts -----------------------------------------------
        external_concepts_content = concept_hierarchy[ConceptHierarchyDefinition.model_concepts].get(
            ConceptHierarchyDefinition.model_concepts_external, []
        )
        if not isinstance(external_concepts_content, list):
            raise CHSyntaxError(
                f"Concept Hierarchy {ConceptHierarchyDefinition.model_concepts_external} must be an array of string "
                f"values (paths to external Concept Hierarchy concept definition files relative to the directory of "
                f"this file)!\nGot {external_concepts_content}",
                location_id=concept_location_id + [ConceptHierarchyDefinition.model_concepts_external],
                part=PathPart.VALUE,
            )
        for external_concept_index, external_concepts_file in enumerate(external_concepts_content):
            if not isinstance(external_concepts_file, str):
                raise CHSyntaxError(
                    f"Concept Hierarchy {ConceptHierarchyDefinition.model_concepts_external} must be an array of string"
                    f" values (paths to external Concept Hierarchy concept definition files relative to the directory "
                    f"of this file)!\nGot non string file path: {external_concepts_file}",
                    location_id=concept_location_id
                    + [ConceptHierarchyDefinition.model_concepts_external, external_concept_index],
                    part=PathPart.VALUE,
                )
            external_location_id = concept_location_id + [
                ConceptHierarchyDefinition.model_concepts_external,
                external_concept_index,
                "ext:" + external_concepts_file,
            ]
            try:
                external_data_of_concepts, ext_file, ext_file_dir = ConceptHierarchyChecker.read_concept_hierarchy(
                    external_concepts_file, file_dir, visited_files, external_location_id
                )
            except FileNotFoundError:
                raise CHSemanticError(
                    f'External concepts file "{external_concepts_file}" not found relative to the path of the root '
                    f"concept hierarchy file!",
                    location_id=external_location_id,
                    part=PathPart.VALUE,
                )
            if not isinstance(external_data_of_concepts, dict):
                raise CHSyntaxError(
                    f"External Concept Hierarchy concept file at {external_concepts_file} is not a JSON object!\n"
                    f"Got {external_data_of_concepts}",
                    location_id=external_location_id,
                    part=PathPart.VALUE,
                )
            visited_files.add(ext_file)
            is_shortform = ConceptHierarchyDefinition.model_concepts not in external_data_of_concepts
            if is_shortform:
                external_data_of_concepts = {ConceptHierarchyDefinition.model_concepts: external_data_of_concepts}
            ext_concepts, ext_instances, ext_concept_locations, ext_instance_locations = (
                self.merge_concept_hierarchy_structure_across_files(
                    external_data_of_concepts, external_location_id, ext_file_dir, visited_files, True, is_shortform
                )
            )
            visited_files.remove(ext_file)
            assert not ext_instances
            assert not ext_instance_locations
            for ext_concept_name, ext_concept_data in ext_concepts.items():
                if ext_concept_name in concept_definition:
                    raise CHSemanticError(
                        f"Found duplicate concept {ext_concept_name} defined at locations:\n"
                        f"\t[{(concept_locations[ext_concept_name] + [ext_concept_name])!r}] and\n"
                        f"\t[{(ext_concept_locations[ext_concept_name] + [ext_concept_name])!r}]!\n"
                        f"Merging external files failed: duplicate concepts are not allowed!",
                        location_id=external_location_id,
                        part=PathPart.VALUE,
                    )
                concept_definition[ext_concept_name] = ext_concept_data
            concept_locations.update(ext_concept_locations)

        # -- instances (optional: default {}) --------------------------------
        assert (
            ConceptHierarchyDefinition.model_instances not in concept_hierarchy
            or not concept_hierarchy_concept_only_shortform
        )  # instances can only appear if this is not a shorthand form
        instances_location_id: LocationId = location_id + [ConceptHierarchyDefinition.model_instances]
        instance_definition = concept_hierarchy.get(ConceptHierarchyDefinition.model_instances, {})
        if not isinstance(instance_definition, dict):
            raise CHSyntaxError(
                f'Concept Hierarchy "{ConceptHierarchyDefinition.model_instances}" data must be a JSON object of'
                f" definitions of instances, i.e. global variables, not {instance_definition!r}.",
                location_id=instances_location_id,
                part=PathPart.VALUE,
            )
        instance_locations = {instance_name: instances_location_id for instance_name in instance_definition}

        # -- external (optional) --------------------------------------------
        external_content = concept_hierarchy.get(ConceptHierarchyDefinition.model_external, [])
        if not isinstance(external_content, list):
            raise CHSyntaxError(
                f"Concept Hierarchy {ConceptHierarchyDefinition.model_external} must be an array of string values "
                f"(paths to external Concept Hierarchy definition files relative to the directory of this file)!\n"
                f"Got {external_content!r}",
                location_id=location_id + [ConceptHierarchyDefinition.model_external],
                part=PathPart.VALUE,
            )
        for external_index, external_ch_file in enumerate(external_content):
            if not isinstance(external_ch_file, str):
                raise CHSyntaxError(
                    f"Concept Hierarchy {ConceptHierarchyDefinition.model_external} must be an array of string values "
                    f"(paths to external Concept Hierarchy definition files relative to the directory of this file)!\n"
                    f"Got non string file path: {external_ch_file}",
                    location_id=location_id + [ConceptHierarchyDefinition.model_external, external_index],
                    part=PathPart.VALUE,
                )
            external_location_id = location_id + [
                ConceptHierarchyDefinition.model_external,
                external_index,
                "ext:" + external_ch_file,
            ]
            try:
                external_data_of_ch, ext_file, ext_file_dir = ConceptHierarchyChecker.read_concept_hierarchy(
                    external_ch_file, self.ch.path_to_root_dir, visited_files, external_location_id
                )
            except FileNotFoundError:
                raise CHSyntaxError(
                    f'External concepts file "{external_ch_file}" not found relative to the path of the root '
                    f"concept hierarchy file!",
                    location_id=external_location_id,
                    part=PathPart.VALUE,
                )
            if not isinstance(external_data_of_ch, dict):
                raise CHSyntaxError(
                    f"External Concept Hierarchy file at {external_ch_file} is not a JSON object!\n"
                    f"Got {external_data_of_ch}",
                    location_id=external_location_id,
                    part=PathPart.VALUE,
                )
            visited_files.add(ext_file)
            is_shortform = ConceptHierarchyDefinition.model_concepts not in external_data_of_ch
            if is_shortform:
                external_data_of_ch = {ConceptHierarchyDefinition.model_concepts: external_data_of_ch}
            ext_concepts, ext_instances, ext_concept_locations, ext_instance_locations = (
                self.merge_concept_hierarchy_structure_across_files(
                    external_data_of_ch, external_location_id, ext_file_dir, visited_files, False, is_shortform
                )
            )
            visited_files.remove(ext_file)
            for ext_concept_name, ext_concept_data in ext_concepts.items():
                if ext_concept_name in concept_definition:
                    raise CHSemanticError(
                        f"Found duplicate concept {ext_concept_name} defined at locations:\n"
                        f"\t[{(concept_locations[ext_concept_name] + [ext_concept_name])!r}] and\n"
                        f"\t[{(ext_concept_locations[ext_concept_name] + [ext_concept_name])!r}]!\n"
                        f"Merging external files failed: duplicate concepts are not allowed!",
                        location_id=external_location_id,
                        part=PathPart.VALUE,
                    )
                concept_definition[ext_concept_name] = ext_concept_data
            concept_locations.update(ext_concept_locations)
            for ext_instance_name, ext_instance_data in ext_instances.items():
                if ext_instance_name in concept_definition:
                    raise CHSemanticError(
                        f"Found duplicate instance {ext_instance_name} defined at locations:\n"
                        f"\t[{(concept_locations[ext_instance_name] + [ext_instance_name])!r}] and\n"
                        f"\t[{(ext_concept_locations[ext_instance_name] + [ext_instance_name])!r}]!\n"
                        f"Merging external files failed: duplicate instances are not allowed!",
                        location_id=external_location_id,
                        part=PathPart.VALUE,
                    )
                instance_definition[ext_instance_name] = ext_instance_data
            instance_locations.update(ext_instance_locations)

        return concept_definition, instance_definition, concept_locations, instance_locations

    def check_structure(self):
        if self.ch.checked:
            return

        base_location_id: LocationId = LocationId()
        if self.ch.file:
            base_location_id.append(self.ch.file)

        file_path: str | None = None
        file_dir: str | None = None
        """
        When this remains None after the statement below, external files will be read relative to the 
        directory in which the program was started.
        """
        visited_files: set[str] = set()
        """Set of visited files while resolving external files (to prevent infinite recursion reading the same file)."""
        if self.ch.definition_data is None:
            self.ch.definition_data, file_path, file_dir = ConceptHierarchyChecker.read_concept_hierarchy(
                self.ch.file, self.ch.path_to_root_dir, visited_files, base_location_id
            )  # does not process external data!
            assert file_path is not None
            visited_files.add(file_path)
        if not isinstance(self.ch.definition_data, dict):
            raise CHSyntaxError("Concept Hierarchy definition must be a JSON object!", location_id=base_location_id)
        concept_hierarchy = self.ch.definition_data

        # interpret either as a meta-definition, or a direct definition of concepts
        ch_keys = set(concept_hierarchy.keys())
        shorthand_concept_hierarchy_concept_only_definition = False
        if not (ch_keys <= ConceptHierarchyDefinition.model_keywords):
            # interpret this as a definition of concepts
            concept_hierarchy = {ConceptHierarchyDefinition.model_concepts: concept_hierarchy}
            shorthand_concept_hierarchy_concept_only_definition = True
        elif len(ch_keys) == 0:
            concept_hierarchy = {ConceptHierarchyDefinition.model_concepts: {}}
            shorthand_concept_hierarchy_concept_only_definition = True

        # -- hierarchy name (optional: default "ConceptHierarchy") ----------
        self.ch.name = concept_hierarchy.get(ConceptHierarchyDefinition.model_name, "ConceptHierarchy")
        if not check_ch_name(self.ch.name, allow_starting_with_underscore=True):
            raise CHSyntaxError(
                f'Concept Hierarchy "{ConceptHierarchyDefinition.model_name}" {self.ch.name!r} must be a non-empty, '
                f"non-digit-starting string containing only alphanumeric characters or '_'.",
                location_id=base_location_id + [ConceptHierarchyDefinition.model_name],
                part=PathPart.VALUE,
            )

        # -- metadata (optional) --------------------------------------------
        raw_meta = concept_hierarchy.get(ConceptHierarchyDefinition.model_metadata, {})
        if not isinstance(raw_meta, dict):
            raise CHSyntaxError(
                f'Concept Hierarchy "{ConceptHierarchyDefinition.model_metadata}" must be a JSON object.',
                location_id=base_location_id + [ConceptHierarchyDefinition.model_metadata],
                part=PathPart.VALUE,
            )
        self.ch.metadata = {str(k): str(v) for k, v in raw_meta.items()}

        concept_hierarchy = {
            k: v
            for k, v in concept_hierarchy.items()
            if k not in {ConceptHierarchyDefinition.model_metadata, ConceptHierarchyDefinition.model_name}
        }
        concept_definition, instance_definition, concept_locations, instance_locations = (
            self.merge_concept_hierarchy_structure_across_files(
                concept_hierarchy,
                base_location_id,
                file_dir,
                visited_files,
                False,
                shorthand_concept_hierarchy_concept_only_definition,
            )
        )
        if file_path is not None:
            visited_files.remove(file_path)
        assert not visited_files, visited_files

        # -- concepts --------------------------------------------------------
        if not concept_definition:
            raise CHSemanticError("The Concept Hierarchy has no concepts!", base_location_id, part=PathPart.VALUE)
        concept_aliases: dict[str, ConceptDefinition] = {}
        defined_concepts: dict[str, ConceptDefinition] = {}
        for concept_name, concept_def in concept_definition.items():  # type: str, object
            concept_definition = ConceptDefinition(
                concept_name,
                concept_def,
                concept_locations[concept_name],
                external_data_resolver=self.ch.external_concept_data_resolver,
            )
            if concept_definition.is_reference():
                concept_aliases[concept_name] = concept_definition
            else:
                defined_concepts[concept_name] = concept_definition
        # Classify and order every alias through one graph. A target that parses to a bare name *may* still
        # be a type alias -- if the name it gives is itself one -- so the kind is settled by walking the
        # order, not by the target's syntax alone.
        alias_targets = {name: TypeParser(a.is_reference_to, None).parse_types() for name, a in concept_aliases.items()}
        alias_dependencies = {name: self.alias_target_names(a.is_reference_to) for name, a in concept_aliases.items()}
        for alias_name in self.order_aliases(
            alias_dependencies, defined_concepts, concept_locations, base_location_id, True
        ):
            parsed_target = alias_targets[alias_name]
            if len(parsed_target) != 1:
                raise CHSemanticError(
                    f"The alias {alias_name} must name exactly one concept or type, not "
                    f"{concept_aliases[alias_name].is_reference_to!r}.",
                    location_id=concept_locations[alias_name] + [alias_name],
                    part=PathPart.VALUE,
                )
            (target,) = parsed_target
            if target.is_templated or target.clean_name in self._type_alias_targets:
                # an applied type, or a name that resolved to one: only `check_types` can build it (§4)
                self._type_alias_targets[alias_name] = target
            else:
                self.ch.concept_aliases[alias_name] = self.ch.canonical_concept_name(target.clean_name)
        # An alias is a name, so the name it stands for has to be the one every derived structure is keyed
        # by -- starting with `parents`, which the topological sort below reads.
        for concept_def in defined_concepts.values():  # type: ConceptDefinition
            concept_def.canonicalize_concept_references(self.ch.canonical_concept_name)

        # -- instances (optional: default {}) --------------------------------
        variable_aliases: dict[str, GlobalVariableDefinition] = {}
        defined_instances: dict[str, GlobalVariableDefinition] = {}
        for variable_name, variable_def in instance_definition.items():  # type: str, object
            variable_definition = GlobalVariableDefinition(
                variable_name, variable_def, instance_locations[variable_name]
            )
            # An alias is recognized by specifying a "string" value; but that value may just as well be a
            # "string" *expression* -> it is only an alias if it names another entry of "instances".
            if variable_definition.is_reference() and variable_definition.is_reference_to in instance_definition:
                variable_aliases[variable_name] = variable_definition
            else:
                variable_definition.is_reference_to = None
                defined_instances[variable_name] = variable_definition
        # the same graph, for the kind whose targets are always plain names
        variable_alias_dependencies: dict[str, tuple[str]] = {
            name: (a.is_reference_to,) for name, a in variable_aliases.items()
        }
        for alias_name in self.order_aliases(
            variable_alias_dependencies, defined_instances, instance_locations, base_location_id, False
        ):
            self.ch.variable_aliases[alias_name] = self.ch.canonical_variable_name(
                variable_aliases[alias_name].is_reference_to
            )
        # missing checks:
        #  - valid expressions for all global variables
        #    EXPRESSION CHECK

        # perform topological sort of concepts and instances
        inverse_direct_parents: dict[str, set[str]] = {}
        concept_parent_mapping = {c_name: c.parents for c_name, c in defined_concepts.items()}
        for c_name, parents in concept_parent_mapping.items():
            for index, p_name in enumerate(parents):
                if p_name not in defined_concepts:
                    raise CHSemanticError(
                        f"The parent {p_name!r} of concept {c_name!r} is not defined in the hierarchy.",
                        location_id=concept_locations[c_name]
                        + [c_name, ConceptDefinition.concept_direct_parents, index],
                    )
                if p_name not in inverse_direct_parents:
                    inverse_direct_parents[p_name] = set()
                inverse_direct_parents[p_name].add(c_name)
        concept_definition_location_id = base_location_id
        if not shorthand_concept_hierarchy_concept_only_definition:
            concept_definition_location_id.append(ConceptHierarchyDefinition.model_concepts)
        try:
            self.ch.concept_topo_sort, roots = topological_sort(concept_parent_mapping)
            if self.ch.root_concept_name in roots and len(roots) != 1:
                raise CHSemanticError(
                    f"Concept Hierarchy has multiple roots: {roots!r}",
                    location_id=concept_definition_location_id,
                    part=PathPart.VALUE,
                )
            elif self.ch.root_concept_name not in roots:
                assert self.ch.root_concept_name not in inverse_direct_parents
                inverse_direct_parents[self.ch.root_concept_name] = set()
                for root in roots:
                    defined_concepts[root].update_parents((self.ch.root_concept_name,))
                    inverse_direct_parents[self.ch.root_concept_name].add(root)
                self.ch.concept_topo_sort = [self.ch.root_concept_name] + self.ch.concept_topo_sort
                defined_concepts[self.ch.root_concept_name] = ConceptDefinition(
                    self.ch.root_concept_name, {}, concept_definition_location_id
                )
                roots = [self.ch.root_concept_name]
            assert len(roots) == 1
            root = roots[0]
            if root != self.ch.root_concept_name:
                raise CHSemanticError(
                    f'The root concept of the Concept Hierarchy must be called "{self.ch.root_concept_name}", not '
                    f"{root!r}!",
                    location_id=concept_definition_location_id,
                    part=PathPart.VALUE,
                )
            self.ch.root_concept_name = root
        except RuntimeError as e:
            if str(e).startswith("Non-hierarchy structure detected! The following items form one or more cycles:"):
                raise CHSemanticError(
                    f"Cycles detected in Concept Hierarchy:\n{tab}{e!s}",
                    location_id=concept_definition_location_id,
                    part=PathPart.VALUE,
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
        self.ch.defined_direct_children = inverse_direct_parents

        self.ch.concepts = defined_concepts

        # an alias is a name like any other, so it collides like one -- on both sides
        for instance_name in list(defined_instances) + list(self.ch.variable_aliases):
            if instance_name in defined_concepts or instance_name in self.ch.concept_aliases:
                raise CHSemanticError(
                    f"The global variable name {instance_name!r} is also a concept name!\n\tThis can create ambiguity! "
                    f"Please rename the global variable name!",
                    location_id=instance_locations[instance_name] + [instance_name],
                    part=PathPart.KEY,
                )
        self.ch.instances = defined_instances
        self.ch.checked_structure = True

    def check_after_parsing_concepts(self):
        # promote to subconcepts the concept definitions
        errors = []
        for c_name in self.ch.concept_topo_sort:
            c = self.ch.concepts[c_name]
            parents_concept_data: dict[str, ConceptData] = {}
            for parent in c.parents:
                assert parent in self.model.concepts
                parents_concept_data[parent] = self.model.concepts[parent]
            parents_concepts = frozendict(parents_concept_data)
            try:
                if self.ch.is_function(c_name):
                    self.ch.concepts[c_name] = FunctionDefinition.from_node(c)
                    concept_data = FunctionData(c_name, parents_concepts)
                    self.model.concepts[c_name] = concept_data
                    self.model.value_domains[c_name] = concept_data
                    self.model.functions[c_name] = concept_data
                elif self.ch.is_value_domain(c_name):
                    self.ch.concepts[c_name] = ValueDomainDefinition.from_node(c)
                    concept_data = ValueDomainData(c_name, parents_concepts)
                    self.model.concepts[c_name] = concept_data
                    self.model.value_domains[c_name] = concept_data
                elif self.ch.is_domain_concept(c_name):
                    self.ch.concepts[c_name] = DomainConceptDefinition.from_node(c)
                    concept_data = DomainConceptData(c_name, parents_concepts)
                    self.model.concepts[c_name] = concept_data
                    self.model.domain_concepts[c_name] = concept_data
                else:
                    raise CHSemanticError(
                        f"Found concept {c_name} with parents {c.parents!r} that is neither a "
                        f"{FunctionDefinition.function_name}, {ValueDomainDefinition.value_domain_name}, "
                        f"nor a {DomainConceptDefinition.domain_concept_name}!",
                        location_id=c.location_of(ConceptDefinition.concept_direct_parents),
                        part=PathPart.VALUE,
                    )
            except ConceptHierarchyError as e:
                errors.append(e)
        if errors:
            if len(errors) == 1:
                raise errors[0]
            raise CHSemanticError("Processing concept data failed because of the errors below!", causes=errors)
        # check domain_concept, value_domain, and function data!
        self.ch.all_domain_concept_properties = {}  # reset from previous checks
        self.ch.all_domain_concept_functions = {}  # reset from previous checks
        for c_name, c in self.ch.concepts.items():
            try:
                c.concept_data_check()  # from now on, one can call c.location_of()
                if isinstance(c, HiddenImplementationDefinition):
                    # the parent half of a substitution key is a concept-name position: it may be an alias
                    c.canonicalize_template_substitution_parents(self.ch.canonical_concept_name)
                    for t_index, t_arg_name in enumerate(c.template_argument_order):
                        if t_arg_name in self.ch.concepts:
                            t_order_location = c.location_of(
                                HiddenImplementationDefinition.hidden_template_arguments_order
                            )
                            raise CHSemanticError(
                                f"The template argument name {t_arg_name!r} of {c_name} is also the name of a defined "
                                f"concept in this Concept Hierarchy!"
                                f"\n\tThis can cause ambiguity in the template argument's constraint formulae "
                                f"definition, in template instantiations and in template substitutions."
                                f"\nPlease rename the template argument!",
                                location_id=t_order_location + [t_index],
                            )
                    # maps names of template arguments of parents to the set of the parent concepts that use those names
                    template_arguments_to_substitute: dict[str, set[str]] = {}
                    extra_t_subst_keys: set[tuple[str | None, str]] = set(c.substitution_of_template_arguments)
                    matched_parents_of_shorthand_syntax: dict[str, str] = {}
                    # ensure all parent template arguments are specified in the substitution definition of this concept
                    for parent in c.parents:
                        parent_c = self.ch.concepts[parent]
                        if not isinstance(parent_c, HiddenImplementationDefinition):
                            continue
                        for parent_t_arg in parent_c.template_argument_order:
                            if parent_t_arg not in template_arguments_to_substitute:
                                template_arguments_to_substitute[parent_t_arg] = set()
                            template_arguments_to_substitute[parent_t_arg].add(parent)
                            # check that all parents are substituted in the concept
                            # check later if the substitution is unambiguous
                            full_key = (parent, parent_t_arg)
                            if full_key in c.substitution_of_template_arguments:
                                extra_t_subst_keys.remove(full_key)
                                continue
                            short_key = (None, parent_t_arg)
                            if short_key in c.substitution_of_template_arguments:
                                # use .discard instead or .remove because if there is an ambiguous specialization,
                                # .remove will be called multiple times on the same element, which will raise an error.
                                extra_t_subst_keys.discard(short_key)
                                matched_parents_of_shorthand_syntax[parent_t_arg] = parent
                                continue
                            if not c.has_location_of(HiddenImplementationDefinition.hidden_template_arguments):
                                raise CHSemanticError(
                                    f'Missing "{HiddenImplementationDefinition.hidden_template_arguments}" definition'
                                    f' in {c_name}, because it must define a substitution for "{parent}:{parent_t_arg}"'
                                    f"!",
                                    location_id=c.location_of(ConceptDefinition.concept_definition_data),
                                    part=PathPart.VALUE,
                                )
                            elif not c.has_location_of(
                                HiddenImplementationDefinition.hidden_template_arguments_substitutions
                            ):
                                raise CHSemanticError(
                                    f"Missing "
                                    f'"{HiddenImplementationDefinition.hidden_template_arguments_substitutions}" '
                                    f"definition in {c_name}, because it must define a substitution for "
                                    f'"{parent}:{parent_t_arg}"!',
                                    location_id=c.location_of(ConceptDefinition.concept_definition_data),
                                    part=PathPart.VALUE,
                                )
                            raise CHSemanticError(
                                f"Parent template argument {parent_t_arg} of {parent} is not specialized in {c_name}! "
                                f'The specialization syntax is "<ParentConceptName>:<ParentTemplateArgumentName>".',
                                location_id=c.location_of(
                                    HiddenImplementationDefinition.hidden_template_arguments_substitutions
                                ),
                                part=PathPart.VALUE,
                            )
                    # ensure that shorthand-syntax specified substitution arguments are unambiguous
                    for t_arg_name, parents_defining_t_arg in template_arguments_to_substitute.items():
                        if (
                            len(parents_defining_t_arg) > 1
                            and (None, t_arg_name) in c.substitution_of_template_arguments
                        ):
                            raise CHSemanticError(
                                f"The substitution specification of template argument {t_arg_name} is ambiguous in "
                                f"{c_name} because the parent concepts {sorted(parents_defining_t_arg)} define the "
                                f"template argument with the same name. Use the "
                                f'"<ParentConceptName>:<ParentTemplateArgumentName>" syntax to define the unambiguous '
                                f"substitution value for all parent template arguments",
                                location_id=c.location_of(
                                    HiddenImplementationDefinition.hidden_template_arguments_substitutions
                                ),
                                part=PathPart.KEY,
                            )
                    # ensure there are no extra keys specified in the substitution definition
                    if len(extra_t_subst_keys) > 0:
                        extra_keys_str = ", ".join(
                            '"' + (f"{p}:" if p is not None else "") + p_t_arg + '"'
                            for p, p_t_arg in extra_t_subst_keys
                        )
                        raise CHSemanticError(
                            f"Extra key(s) {extra_keys_str} in template substitution definition of {c_name} must be "
                            f"removed!\nThey are not template arguments of any of the defined parent concepts: "
                            f"{', '.join(c.parents)}!",
                            location_id=c.location_of(
                                HiddenImplementationDefinition.hidden_template_arguments_substitutions
                            ),
                            part=PathPart.VALUE,
                        )
                    # replace shorthand-syntax template arguments in concept's substitution member
                    for parent_t_arg, parent in matched_parents_of_shorthand_syntax.items():
                        existing_key = (None, parent_t_arg)
                        assert existing_key in c.substitution_of_template_arguments
                        c.substitution_of_template_arguments[(parent, parent_t_arg)] = (
                            c.substitution_of_template_arguments.pop((None, parent_t_arg))
                        )
                if isinstance(c, ValueDomainDefinition):
                    assert self.ch.is_pure_value_domain(c_name)
                    self.ch.value_domains.add(c_name)
                    for parent_index, parent in enumerate(c.parents):
                        if not self.ch.is_pure_value_domain(parent) and parent != ConceptDefinition.concept_name:
                            raise CHSemanticError(
                                f"Parents of {ValueDomainDefinition.value_domain_name}s must be "
                                f"{ValueDomainDefinition.value_domain_name}s.\nEncountered non "
                                f"{ValueDomainDefinition.value_domain_name} parent {parent!r} of {c_name}",
                                location_id=c.reference_location(
                                    ConceptDefinition.concept_direct_parents, parent_index
                                ),
                            )
                    if (
                        c.default_serialization is not None
                        and c.default_serialization in self.ch.default_serializations
                    ):
                        raise CHSemanticError(
                            f"The default serialization value of ValueDomains must be unique across all concepts!"
                            f"\nFound (non-inclusive) duplicate "
                            f'"{ValueDomainDefinition.value_domain_default_serialization}" specifications in {c_name} '
                            f"and {self.ch.default_serializations[c.default_serialization]}!",
                            location_id=c.location_of(ValueDomainDefinition.value_domain_default_serialization),
                        )
                    self.ch.default_serializations[c.default_serialization] = c_name
                if isinstance(c, FunctionDefinition):
                    assert self.ch.is_function(c_name)
                    self.ch.functions.add(c_name)
                    # check parent concept and merge the interface!
                    assert len(c.parents) == 1
                    parent_index, parent = 0, c.parents[0]
                    parent_c = self.ch.concepts[parent]
                    if not self.ch.is_function(parent) and parent != ValueDomainDefinition.value_domain_name:
                        raise CHSemanticError(
                            f"Parents of {FunctionDefinition.function_name}s must be "
                            f"{FunctionDefinition.function_name}s.\nEncountered non "
                            f"{FunctionDefinition.function_name} parent {parent!r} of {c_name}",
                            location_id=c.reference_location(ConceptDefinition.concept_direct_parents, parent_index),
                        )
                    elif parent != ValueDomainDefinition.value_domain_name:
                        assert isinstance(parent_c, FunctionDefinition)
                        if parent_c.result_defined_in is UNINITIALIZED:
                            if not c.abstract and c.has_interface_defined:
                                c.result_defined_in = c.name if c.returns_something else None
                        else:
                            if parent_c.result_defined_in is not None and c.returns_something:
                                raise CHSemanticError(
                                    f"Defined result twice: once in {parent_c.result_defined_in} and once in {c.name}. "
                                    f"If you need a different result type, define a new Function concept that is not a "
                                    f"subconcept of {parent!r}.",
                                    location_id=c.location_of(FunctionDefinition.function_result),
                                    part=PathPart.KEY,
                                )
                            if parent_c.result_defined_in is None and c.returns_something:
                                raise CHSemanticError(
                                    f"Defined result type in {c.name!r} after a parent concept defines that the "
                                    f"evaluation interface does not result in anything! Defined this in "
                                    f"{parent_c.result_defined_in}.\nIf you need a different result type, define a new "
                                    f"Function concept that is not a subconcept of {parent!r}.",
                                    location_id=c.location_of(FunctionDefinition.function_result),
                                    part=PathPart.KEY,
                                )
                            c.result_defined_in = parent_c.result_defined_in
                        for eval_arg, eval_arg_type in c.evaluation_argument_types.items():
                            if eval_arg in parent_c.all_evaluation_arguments:
                                defining_parent_of_eval_arg = parent_c.all_evaluation_arguments[eval_arg]
                                defining_parent_c = self.ch.concepts[defining_parent_of_eval_arg]
                                assert isinstance(defining_parent_c, FunctionDefinition)
                                eval_arg_type_in_first_definition = defining_parent_c.evaluation_argument_types[
                                    eval_arg
                                ]
                                eval_arg_access_in_first_definition = (
                                    defining_parent_c.evaluation_argument_access_types[eval_arg]
                                )
                                eval_arg_provenance_in_first_definition = (
                                    defining_parent_c.evaluation_argument_provenance_types[eval_arg]
                                )
                                if (
                                    (c.evaluation_argument_types[eval_arg] != eval_arg_type_in_first_definition)
                                    or (
                                        c.evaluation_argument_provenance_types[eval_arg]
                                        != eval_arg_access_in_first_definition
                                    )
                                    or (
                                        c.evaluation_argument_provenance_types[eval_arg]
                                        != eval_arg_provenance_in_first_definition
                                    )
                                ):
                                    raise CHSemanticError(
                                        f"{c.definition_type()} evaluation argument {eval_arg!r} is defined twice "
                                        f"in {c.name} and {parent} with a different definition.\nEither remove the "
                                        f"definition from {c.name!r} or don't make {c.name!r} a subconcept of "
                                        f"{defining_parent_of_eval_arg!r}.",
                                        location_id=c.location_of(FunctionDefinition.function_interface, eval_arg),
                                        part=PathPart.KEY,
                                    )
                                raise CHSemanticError(
                                    f"{c.definition_type()} evaluation argument {eval_arg!r} is defined twice in "
                                    f"{c.name} and {parent} with the same definition.\nRemove the definition from "
                                    f"{c.name!r}.",
                                    location_id=c.location_of(FunctionDefinition.function_interface, eval_arg),
                                    part=PathPart.KEY,
                                )
                            c.all_evaluation_arguments[eval_arg] = c.name
                        # there isn't anything to overwrite below
                        c.all_evaluation_arguments.update(parent_c.all_evaluation_arguments)
                        # ``all_sub_scope_data`` is initially empty;
                        #  first add parent data, then overwrite it with this concept's data.
                        c.all_sub_scope_data.update(parent_c.all_sub_scope_data)
                        c.all_sub_scope_data.update(c.sub_scopes)
                    for eval_arg_name in c.evaluation_argument_types:
                        if self.ch.is_variable(eval_arg_name):
                            raise CHSemanticError(
                                f"The name of the evaluation argument {eval_arg_name!r} of {c_name} is also the name of"
                                f" a defined global variable (global instance) in this Concept Hierarchy."
                                f"\n\tThis can cause ambiguity in the context of the FunctionComposition of Function "
                                f"procedures, inversions, and variations!"
                                f"\nPlease rename the Function argument or the global variable!",
                                location_id=c.location_of(FunctionDefinition.function_interface) + [eval_arg_name],
                                part=PathPart.KEY,
                            )
                    for default_arg_name in c.evaluation_argument_default_values:
                        if default_arg_name not in c.all_evaluation_arguments:
                            raise CHSemanticError(
                                f"The default argument {default_arg_name!r} specified in {c.name} is not an evaluation "
                                f"argument for this concept not for any of its parent concepts."
                                f'\nRemove it from "{FunctionDefinition.function_default_argument_values}".',
                                location_id=c.location_of(
                                    FunctionDefinition.function_default_argument_values, default_arg_name
                                ),
                                part=PathPart.KEY,
                            )
                if isinstance(c, DomainConceptDefinition):
                    assert self.ch.is_domain_concept(c_name)
                    self.ch.domain_concepts.add(c_name)
                    for parent_index, parent in enumerate(c.parents):
                        if not self.ch.is_domain_concept(parent):
                            raise CHSemanticError(
                                f"Parents of {DomainConceptDefinition.domain_concept_name}s must be "
                                f"{DomainConceptDefinition.domain_concept_name}s.\nEncountered non "
                                f"{DomainConceptDefinition.domain_concept_name} parent {parent!r} of {c_name}",
                                location_id=c.reference_location(
                                    ConceptDefinition.concept_direct_parents, parent_index
                                ),
                            )
                    # check unique property names, unique function names, distinct function and property names,
                    # and non-ambiguous definitions of properties or functions with the same name as a global variable
                    for prop_name, prop_def_data in c.properties.items():
                        if self.ch.is_variable(prop_name):
                            raise CHSemanticError(
                                f"The name of the concept property {prop_name!r} of {c_name} is also the name of a "
                                f"defined global variable (global instance) in this Concept Hierarchy."
                                f"\n\tThis can cause ambiguity in the context of property hooks, computations, concept "
                                f"functions, and management functions."
                                f"\nPlease rename the property or the global variable!",
                                location_id=c.location_of(DomainConceptDefinition.domain_concept_properties, prop_name),
                                part=PathPart.KEY,
                            )
                        if prop_name in self.ch.all_domain_concept_properties:
                            raise CHSemanticError(
                                f"The property {prop_name} is defined in multiple places!\nFound (non-inclusively) in "
                                f"{c_name!r} and in {self.ch.all_domain_concept_properties[prop_name]!r}."
                                f"\n\tPlease move the property to a common concept or rename one of them so that "
                                f"property names are unique!",
                                location_id=c.location_of(DomainConceptDefinition.domain_concept_properties, prop_name),
                                part=PathPart.KEY,
                            )
                        if prop_name in self.ch.all_domain_concept_functions:
                            raise CHSemanticError(
                                f"The property {prop_name} is defined in multiple places!\nFound (non-inclusively) in "
                                f"{c_name!r} as a property and in {self.ch.all_domain_concept_functions[prop_name]!r} "
                                f"as a function!\n\tPlease rename one of them so that property names are distinct from "
                                f"function names!",
                                location_id=c.location_of(DomainConceptDefinition.domain_concept_properties, prop_name),
                                part=PathPart.KEY,
                            )
                        self.ch.all_domain_concept_properties[prop_name] = c_name
                        if PropertyDefinitionKeywords.CONFIDENCE in prop_def_data:
                            confidence_location_id = c.location_of(
                                DomainConceptDefinition.domain_concept_properties,
                                prop_name,
                                PropertyDefinitionKeywords.CONFIDENCE,
                            )
                            # Check whether the Duration type is defined:
                            #  this is the expected value of the CONFIDENCE keyword => if used, it must be defined
                            if not self.ch.is_concept("Duration"):
                                raise CHSemanticError(
                                    f"The Duration concept is not defined in the Concept Hierarchy => can not use "
                                    f'"{PropertyDefinitionKeywords.CONFIDENCE}".\nPlease define the "Duration" concept '
                                    f"as a subconcept of ValueDomain or remove the "
                                    f'"{PropertyDefinitionKeywords.CONFIDENCE}" keyword from all property definitions '
                                    f"and specializations!",
                                    location_id=confidence_location_id,
                                    part=PathPart.KEY,
                                )
                    for func_name in c.functions:
                        if self.ch.is_variable(func_name):
                            raise CHSemanticError(
                                f"The name of the concept function {func_name!r} of {c_name} is also the name of a "
                                f"defined global variable (global instance) in this Concept Hierarchy."
                                f"\n\tThis can cause ambiguity in the context of property hooks, computations, concept "
                                f"functions, and management functions."
                                f"\nPlease rename the function or the global variable!",
                                location_id=c.location_of(DomainConceptDefinition.domain_concept_functions, func_name),
                                part=PathPart.KEY,
                            )
                        if func_name in self.ch.all_domain_concept_functions:
                            raise CHSemanticError(
                                f"The function {func_name} is defined in multiple places!\nFound (non-inclusively) in "
                                f"{c_name!r} and in {self.ch.all_domain_concept_functions[func_name]!r}."
                                f"\n\tPlease move the function to a common concept or rename one of them so that "
                                f"function names are unique!",
                                location_id=c.location_of(DomainConceptDefinition.domain_concept_functions, func_name),
                                part=PathPart.KEY,
                            )
                        if func_name in self.ch.all_domain_concept_properties:
                            raise CHSemanticError(
                                f"The function {func_name} is defined in multiple places!\nFound (non-inclusively) in "
                                f"{c_name!r} as a function and in {self.ch.all_domain_concept_properties[func_name]!r} "
                                f"as a property!\n\tPlease rename one of them so that function names are distinct from "
                                f"property names!",
                                location_id=c.location_of(DomainConceptDefinition.domain_concept_functions, func_name),
                                part=PathPart.KEY,
                            )
                        self.ch.all_domain_concept_functions[func_name] = c_name

                #  - check that all concept names in distinct_from are:
                #   1) concepts,
                #   2) different from this concept, and
                #   3) not parents of this concept
                for index, distinct_from_concept in enumerate(c.distinct_from):
                    if not self.ch.is_concept(distinct_from_concept):
                        raise CHSemanticError(
                            f"{ConceptDefinition.concept_distinct_from} entry {index} of {c.name} "
                            f'("{distinct_from_concept}") is not a concept!',
                            location_id=c.reference_location(ConceptDefinition.concept_distinct_from, index),
                        )
                    if self.ch.is_a_subconcept_of_b(c.name, distinct_from_concept, include_self=True):
                        raise CHSemanticError(
                            f'Concept {c.name} must be distinct from itself? Because "{distinct_from_concept}" is'
                            f" either the same concept as {c.name} or a parent concept of {c.name}!\n\tIf you want to "
                            f'make this concept non-instantiable, set the "{ConceptDefinition.concept_abstract}" '
                            f'keyword in the concept definition to "true"!',
                            location_id=c.reference_location(ConceptDefinition.concept_distinct_from, index),
                        )
                    self.ch.add_distinct_pair(c_name, distinct_from_concept)
                for index, distinct_group_entry in enumerate(c.distinct_group):
                    if not self.ch.is_concept(distinct_group_entry):
                        raise CHSemanticError(
                            f"{ConceptDefinition.concept_distinct_group} entry {index} of {c.name} "
                            f'("{distinct_group_entry}") is not a concept!',
                            location_id=c.reference_location(ConceptDefinition.concept_distinct_group, index),
                        )
                    # check that the child is a *direct* child, not just some concept
                    entry_concept = self.ch.concepts[distinct_group_entry]
                    if c.name not in entry_concept.parents:
                        raise CHSemanticError(
                            f"The {ConceptDefinition.concept_distinct_group} entry {distinct_group_entry} of {c.name} "
                            f"is not a direct child of {c.name}!",
                            location_id=c.reference_location(ConceptDefinition.concept_distinct_group, index),
                        )
                    # append the data from the group to the distinctFrom of the concepts in the distinctGroup
                    for other_entry_in_distinct_group in c.distinct_group:
                        if (
                            other_entry_in_distinct_group == distinct_group_entry
                            or other_entry_in_distinct_group in entry_concept.all_concepts_distinct_from_this
                        ):
                            continue
                        entry_concept.all_concepts_distinct_from_this += (other_entry_in_distinct_group,)
                        self.ch.add_distinct_pair(distinct_group_entry, other_entry_in_distinct_group)
                if c.fixed_children is not None:
                    for index, fixed_child in enumerate(c.fixed_children or []):
                        if not self.ch.is_concept(fixed_child):
                            raise CHSemanticError(
                                f"{ConceptDefinition.concept_direct_children} entry {index} of {c.name} "
                                f'("{fixed_child}") is not a concept! Either remove it from the list or ',
                                location_id=c.reference_location(ConceptDefinition.concept_direct_children, index),
                            )
                        # check that the child is a *direct* child, not just some concept
                        child_concept = self.ch.concepts[fixed_child]
                        if c.name not in child_concept.parents:
                            raise CHSemanticError(
                                f"The {ConceptDefinition.concept_direct_children} entry {fixed_child} of {c.name} is "
                                f"not a direct child of {c.name}!",
                                location_id=c.reference_location(ConceptDefinition.concept_direct_children, index),
                            )
                    # Verify that there are no other children of this concept except the ones defined there
                    defined_children = set()
                    if c_name in self.ch.defined_direct_children:
                        defined_children = self.ch.defined_direct_children[c_name]
                    assert len(set(c.fixed_children) - defined_children) == 0
                    extra_children = defined_children - set(c.fixed_children)
                    if len(extra_children) != 0:
                        raise CHSemanticError(
                            f'The definition of "{ConceptDefinition.concept_direct_children}" for a domain concept '
                            f"should exhaustively enumerate ALL direct children concepts.\n{c_name} has the following "
                            f'children not listed in "{ConceptDefinition.concept_direct_children}": '
                            f"{sorted(extra_children)}",
                            location_id=c.location_of(ConceptDefinition.concept_direct_children),
                            part=PathPart.VALUE,
                        )
            except ConceptHierarchyError as e:
                errors.append(e)
        # check distinctness of hierarchy only after all concepts' distinctFrom and distinctGroup entries were processed
        for c_name, c in self.ch.concepts.items():
            if len(c.parents) < 2:
                continue
            all_parents_of_c = self.ch.all_concept_parents[c_name]
            for distinct_pair in self.ch.all_declared_distinct_pairs:
                if distinct_pair[0] in all_parents_of_c and distinct_pair[1] in all_parents_of_c:
                    errors.append(
                        CHSemanticError(
                            f"The concept {c_name} is a subconcept of two distinct concepts {distinct_pair[0]} and "
                            f"{distinct_pair[1]}! Fix the hierarchy!",
                            location_id=c.location_of(ConceptDefinition.concept_direct_parents),
                            part=PathPart.VALUE,
                        )
                    )
        if errors:
            if len(errors) == 1:
                raise errors[0]
            raise CHSemanticError("Processing concept data failed because of the errors below!", causes=errors)
        # check global variable data
        for var_name, var in self.ch.instances.items():
            var.check_syntax()

    def check_specializations(self):
        process_specialization_for_domain_concepts(self.context)

    def check_types(self):
        # 1) check all template constraints of templated ValueDomains
        # set template_constraint_formula_validator
        self.context.template_constraint_formula_validator = ConstraintFormulaValidator(self.context)
        check_value_domain_template_constraint_formulae(self.context)
        # 2) check all the types used in the ConceptHierarchy:
        #   - Function arguments,
        #   - ValueDomain literal formulae
        #   - Domain Concept property and function types
        #   - ValueDomain template substitution values
        # set instantiation_value_validator + expression_parser_validator + instantiation_schema_validator
        self.context.type_application_constraints_validator = TypeApplicationValidator(self.context)
        self.context.type_validator = ConceptHierarchyTypeValidator(self.context)
        self.context.instantiation_schema_validator = SchemaValidator(self.context)
        # 3) resolve the type aliases -- possible only now that concepts and their template contexts exist,
        #    and necessarily before the types that use them are checked
        self.resolve_type_aliases()
        check_types_in_concept_hierarchy(self.context)

    def resolve_type_aliases(self):
        """
        Turn each alias of an applied type into the :class:`InstantiatedType` it names.

        ``_type_alias_targets`` is already in the alias graph's topological order, so an alias that names
        another is built after it, and the substitution in the type validator finds a resolved type waiting
        for it. Every entry is saturated and ground by construction: only an *applied* type gets here, and
        anything it mentions is either a concept or an already-resolved alias.
        """
        concepts_location_id = LocationId([ConceptHierarchyDefinition.model_concepts])
        # An alias is declared at the hierarchy level, so no template variable is in scope for it -- the
        # type parser still needs *a* context to answer `is_template_variable`, so give it an empty one.
        self.context.set_template_context(TemplateContext("global"))
        try:
            for alias_name, parsed_target in self._type_alias_targets.items():
                location_id = concepts_location_id + [alias_name]
                try:
                    self.ch.type_aliases[alias_name] = parse_convert_type(
                        parsed_target.full_name, self.context.type_validator, location_id
                    )
                except ConceptHierarchyError as e:
                    raise CHSemanticError(
                        f"Parsing the alias {alias_name} into a type failed: got {parsed_target.full_name!r}",
                        location_id=location_id,
                        part=PathPart.VALUE,
                        causes=[e],
                    ) from e
        finally:
            self.context.reset_template_context()

    def check_expressions(self):
        self.context.instantiation_values_validator = ValueValidator(self.context)
        self.context.expression_parser_validator = ExpressionValidator(self.context)
        check_expressions_in_concept_hierarchy(self.context)

    def check(self):
        self.check_structure()
        self.check_after_parsing_concepts()
        self.check_specializations()
        self.check_types()
        self.check_expressions()


def check_model(model: ConceptHierarchyDefinition, checker: ConceptHierarchyChecker | None = None) -> ConceptHierarchy:
    """Validate syntax and semantic rules on *model*, raising on the first violation.

    Parameters
    ----------
    model:
        A :class:`~concept_hierarchy.models.ConceptHierarchyDefinition` produced by the parser.
    checker:
        A :class:`~concept_hierarchy.checker.ConceptHierarchyChecker` instance or None.
        If not specified, will use a default-created ConceptHierarchyChecker instance.
        Using a custom checker is advantageous:
         - during testing (because it can customize the performed checks) or
         - when extending the capabilities of the ConceptHierarchy with new features
            that are not included in the base class.

    Raises
    ------
    concept_hierarchy.errors.CHSyntaxError
    concept_hierarchy.errors.CHSemanticError
    """
    if checker is None:
        checker = ConceptHierarchyChecker(model)
    checker.check()
    return checker.model
