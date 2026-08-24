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
from typing import Callable

from frozendict import frozendict

from concept_hierarchy.data.concept_hierarchy import (
    ConceptData,
    ConceptHierarchy,
    DomainConceptData,
    FunctionData,
    ValueDomainData,
)
from concept_hierarchy.data.contexts.context import ConceptHierarchyContext
from concept_hierarchy.data.utils import UNINITIALIZED
from concept_hierarchy.definitions.concept_definition import ConceptDefinition
from concept_hierarchy.definitions.concept_definition_domain_concept import DomainConceptDefinition, PropertyDefinition
from concept_hierarchy.definitions.concept_definition_functions import FunctionDefinition
from concept_hierarchy.definitions.concept_definition_hidden_implementation import HiddenImplementationDefinition
from concept_hierarchy.definitions.concept_definition_value_domain import ValueDomainDefinition
from concept_hierarchy.definitions.definition import ConceptHierarchyDefinition
from concept_hierarchy.definitions.global_variable_definition import GlobalVariableDefinition
from concept_hierarchy.definitions.utils import check_ch_name
from concept_hierarchy.errors import CHSemanticError, CHSyntaxError, ConceptHierarchyError, LocationId, PathPart
from concept_hierarchy.models import ConceptHierarchyModel
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
from concept_hierarchy.validator.validators.type_application_constraints_validator import TypeApplicationValidator
from concept_hierarchy.validator.validators.type_validator import ConceptHierarchyTypeValidator
from concept_hierarchy.validator.validators.value_instantiation_schema_validator import SchemaValidator
from concept_hierarchy.validator.value_domain_template_constraint_checks import (
    check_value_domain_template_constraint_formulae,
)


class ConceptHierarchyChecker:
    @staticmethod
    def read_concept_hierarchy(file: str, path_to_root_dir: str) -> object:
        def create_json(file_name: str) -> object:
            res = read_json_file(file_name)
            if ConceptHierarchyModel.model_concepts_external in res:
                assert isinstance(res[ConceptHierarchyModel.model_concepts_external], list)
                for sub_file_name in res[ConceptHierarchyModel.model_concepts_external]:
                    if os.path.isabs(sub_file_name):
                        res.update(create_json(sub_file_name))
                    else:
                        rel_sub_file_name = sanitize_relative_path(join_path(path_to_root_dir, sub_file_name))
                        res.update(create_json(rel_sub_file_name))
                res.pop(ConceptHierarchyModel.model_concepts_external)
            return res

        return create_json(file)

    @staticmethod
    def check_cycles_in_references_based_on_defined(
        references: dict[str, ConceptHierarchyDefinition], definitions: dict, location_id: LocationId
    ) -> dict[str, str | None]:
        mapped_references: dict[str, str | None] = {x: None for x in definitions}
        for reference, ref_data in references.items():
            assert isinstance(reference, str)
            assert reference not in mapped_references
            assert ref_data.is_reference()
            referenced_concept = ref_data.is_reference_to
            if referenced_concept not in references and referenced_concept not in definitions:
                raise CHSemanticError(
                    f"The referenced concept {referenced_concept!r} of {reference} does not exist in the "
                    f"Concept Hierarchy!",
                    location_id=location_id + [reference],
                    part=PathPart.VALUE,
                )
        total_length = len(definitions) + len(references)
        prev_length = len(mapped_references)
        while True:
            for concept_name, concept_def in references.items():
                if concept_name not in mapped_references:
                    referenced_concept = concept_def.is_reference_to
                    if referenced_concept in mapped_references:
                        if referenced_concept in definitions:
                            mapped_references[concept_name] = referenced_concept
                        else:
                            mapped_references[concept_name] = mapped_references[referenced_concept]
            current_length = len(mapped_references)
            assert current_length <= total_length
            if current_length == total_length:
                break
            elif current_length == prev_length:
                raise CHSemanticError(
                    f"There is a cycle in the {location_id[-1][:-1]} references: {set(references.keys())!r}",
                    location_id=location_id,
                    part=PathPart.VALUE,
                )
            prev_length = current_length
        return mapped_references

    def __init__(
        self,
        concept_hierarchy_data: ConceptHierarchyModel,
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

    @property
    def ch(self) -> ConceptHierarchyModel:
        return self.model.ch

    @staticmethod
    def resolve_references(referencing_others, defined_data, location_id: LocationId):
        mapped_data = ConceptHierarchyChecker.check_cycles_in_references_based_on_defined(
            referencing_others, defined_data, location_id
        )
        assert all(mapped_data[x] is None for x in defined_data)
        for referencing_name, referencing_def in referencing_others.items():
            referenced_name = mapped_data[referencing_name]
            assert referenced_name is not None
            defined_data[referencing_name] = referencing_def.create_from_reference(defined_data[referenced_name])

    def check_structure(self):
        if self.ch.checked:
            return

        base_location_id: LocationId = LocationId()
        if self.ch.file:
            base_location_id.append(self.ch.file)

        if self.ch.definition_data is None:
            self.ch.definition_data = ConceptHierarchyChecker.read_concept_hierarchy(
                self.ch.file, self.ch.path_to_root_dir
            )
        if not isinstance(self.ch.definition_data, dict):
            raise CHSyntaxError("Concept Hierarchy definition must be a JSON object!", location_id=base_location_id)
        concept_hierarchy = self.ch.definition_data

        # interpret either as a meta-definition, or a direct definition of concepts
        ch_keys = set(concept_hierarchy.keys())
        if not (ch_keys <= ConceptHierarchyModel.model_keywords):
            # interpret this as a definition of concepts
            concept_hierarchy = {ConceptHierarchyModel.model_concepts: concept_hierarchy}
        elif len(ch_keys) == 0:
            concept_hierarchy = {ConceptHierarchyModel.model_concepts: {}}

        # -- hierarchy name (optional: default "ConceptHierarchy") ----------

        self.ch.name = concept_hierarchy.get(ConceptHierarchyModel.model_name, "ConceptHierarchy")
        if not check_ch_name(self.ch.name, allow_starting_with_underscore=True):
            raise CHSyntaxError(
                f'Concept Hierarchy "{ConceptHierarchyModel.model_name}" {self.ch.name!r} must be a non-empty, '
                f"non-digit-starting string containing only alphanumeric characters or '_'.",
                location_id=base_location_id + [ConceptHierarchyModel.model_name],
                part=PathPart.VALUE,
            )

        # -- metadata (optional) --------------------------------------------
        raw_meta = concept_hierarchy.get(ConceptHierarchyModel.model_metadata, {})
        if not isinstance(raw_meta, dict):
            raise CHSyntaxError(
                f'Concept Hierarchy "{ConceptHierarchyModel.model_metadata}" must be a JSON object.',
                location_id=base_location_id + [ConceptHierarchyModel.model_metadata],
                part=PathPart.VALUE,
            )
        self.ch.metadata = {str(k): str(v) for k, v in raw_meta.items()}

        # -- concepts --------------------------------------------------------
        concept_location_id: LocationId = base_location_id + [ConceptHierarchyModel.model_concepts]
        if ConceptHierarchyModel.model_concepts not in concept_hierarchy:
            raise CHSyntaxError(
                f'Missing required top-level key: "{ConceptHierarchyModel.model_concepts}".',
                location_id=concept_location_id,
                part=PathPart.KEY,
            )
        concept_definition = concept_hierarchy[ConceptHierarchyModel.model_concepts]
        # type checks for concept_definition data
        if not isinstance(concept_definition, dict):
            raise CHSyntaxError(
                f'Concept Hierarchy "{ConceptHierarchyModel.model_concepts}" data must be a JSON object of concept '
                f"definitions, not {concept_definition!r}.",
                location_id=concept_location_id,
                part=PathPart.VALUE,
            )
        concepts_referencing_others: dict[str, ConceptDefinition] = {}
        defined_concepts: dict[str, ConceptDefinition] = {}
        for concept_name, concept_def in concept_definition.items():  # type: str, object
            concept_definition = ConceptDefinition(
                concept_name,
                concept_def,
                concept_location_id,
                external_data_resolver=self.ch.external_concept_data_resolver,
            )
            if concept_definition.is_reference():
                concepts_referencing_others[concept_name] = concept_definition
            else:
                defined_concepts[concept_name] = concept_definition
        self.resolve_references(concepts_referencing_others, defined_concepts, concept_location_id)

        # -- instances (optional: default {}) --------------------------------
        instances_location_id: LocationId = base_location_id + [ConceptHierarchyModel.model_instances]
        instance_definition = concept_hierarchy.get(ConceptHierarchyModel.model_instances, {})
        if not isinstance(instance_definition, dict):
            raise CHSyntaxError(
                f'Concept Hierarchy "{ConceptHierarchyModel.model_instances}" data must be a JSON object of definitions'
                f" of instances, i.e. global variables, not {instance_definition!r}.",
                location_id=instances_location_id,
                part=PathPart.VALUE,
            )
        instances_referencing_others: dict[str, GlobalVariableDefinition] = {}
        defined_instances: dict[str, GlobalVariableDefinition] = {}
        for variable_name, variable_def in instance_definition.items():  # type: str, object
            variable_definition = GlobalVariableDefinition(variable_name, variable_def, instances_location_id)
            if variable_definition.is_reference():
                instances_referencing_others[variable_name] = variable_definition
            else:
                defined_instances[variable_name] = variable_definition
        self.resolve_references(instances_referencing_others, defined_instances, instances_location_id)
        # missing checks:
        #  - valid expressions for all global variables
        #    EXPRESSION CHECK

        # perform topological sort of concepts and instances
        concept_parent_mapping = {c_name: c.parents for c_name, c in defined_concepts.items()}
        for c_name, parents in concept_parent_mapping.items():
            for index, p_name in enumerate(parents):
                if p_name not in defined_concepts:
                    raise CHSemanticError(
                        f"The parent {p_name!r} of concept {c_name!r} is not defined in the hierarchy.",
                        location_id=concept_location_id + [c_name, ConceptDefinition.concept_direct_parents, index],
                    )
        try:
            self.ch.concept_topo_sort, roots = topological_sort(concept_parent_mapping)
            if self.ch.root_concept_name in roots and len(roots) != 1:
                raise CHSemanticError(
                    f"Concept Hierarchy has multiple roots: {roots!r}",
                    location_id=concept_location_id,
                    part=PathPart.VALUE,
                )
            elif self.ch.root_concept_name not in roots:
                for root in roots:
                    defined_concepts[root].update_parents((self.ch.root_concept_name,))
                self.ch.concept_topo_sort = [self.ch.root_concept_name] + self.ch.concept_topo_sort
                defined_concepts[self.ch.root_concept_name] = ConceptDefinition(
                    self.ch.root_concept_name, {}, concept_location_id
                )
                roots = [self.ch.root_concept_name]
            assert len(roots) == 1
            root = roots[0]
            if root != self.ch.root_concept_name:
                raise CHSemanticError(
                    f'The root concept of the Concept Hierarchy must be called "{self.ch.root_concept_name}", not '
                    f"{root!r}!",
                    location_id=concept_location_id,
                    part=PathPart.VALUE,
                )
            self.ch.root_concept_name = root
        except RuntimeError as e:
            if str(e).startswith("Non-hierarchy structure detected! The following items form one or more cycles:"):
                raise CHSemanticError(
                    f"Cycles detected in Concept Hierarchy:\n{tab}{e!s}",
                    location_id=concept_location_id,
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

        self.ch.concepts = defined_concepts

        for instance_name in defined_instances:
            if instance_name in defined_concepts:
                raise CHSemanticError(
                    f"The global variable name {instance_name!r} is also a concept name!\n\tThis can create ambiguity! "
                    f"Please rename the global variable name!",
                    location_id=instances_location_id + [instance_name],
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
        for c_name, c in self.ch.concepts.items():
            try:
                c.concept_data_check()  # from now on, one can call c.location_of()
                if isinstance(c, HiddenImplementationDefinition):
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
                                location_id=c.location_of(ConceptDefinition.concept_direct_parents) + [parent_index],
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
                            location_id=c.location_of(ConceptDefinition.concept_direct_parents) + [parent_index],
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
                    for eval_arg_name in c.evaluation_interface:
                        if eval_arg_name in self.ch.instances:
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
                                location_id=c.location_of(ConceptDefinition.concept_direct_parents) + [parent_index],
                            )
                    # check unique property names, unique function names, distinct function and property names,
                    # and non-ambiguous definitions of properties or functions with the same name as a global variable
                    for prop_name, prop_def_data in c.properties.items():
                        if prop_name in self.ch.instances:
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
                        if PropertyDefinition.CONFIDENCE in prop_def_data:
                            confidence_location_id = c.location_of(
                                DomainConceptDefinition.domain_concept_properties,
                                prop_name,
                                PropertyDefinition.CONFIDENCE,
                            )
                            # Check whether the Duration type is defined:
                            #  this is the expected value of the CONFIDENCE keyword => if used, it must be defined
                            if not self.ch.is_concept("Duration"):
                                raise CHSemanticError(
                                    f"The Duration concept is not defined in the Concept Hierarchy => can not use "
                                    f'"{PropertyDefinition.CONFIDENCE}".\nPlease define the "Duration" concept as a '
                                    f'subconcept of ValueDomain or remove the "{PropertyDefinition.CONFIDENCE}" keyword'
                                    f" from all property definitions and specializations!",
                                    location_id=confidence_location_id,
                                    part=PathPart.KEY,
                                )
                    for func_name in c.functions:
                        if func_name in self.ch.instances:
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

                #  - check that all concept names in distinct_from are:
                #   1) concepts,
                #   2) different from this concept, and
                #   3) not parents of this concept
                for index, distinct_from_concept in enumerate(c.distinct_from):
                    if not self.ch.is_concept(distinct_from_concept):
                        raise CHSemanticError(
                            f"{ConceptDefinition.concept_distinct_from} entry {index} of {c.name} "
                            f'("{distinct_from_concept}") is not a concept!',
                            location_id=c.location_of(ConceptDefinition.concept_distinct_from) + [index],
                        )
                    if self.ch.is_a_subconcept_of_b(c.name, distinct_from_concept, include_self=True):
                        raise CHSemanticError(
                            f'Concept {c.name} must be distinct from itself? Because "{distinct_from_concept}" is'
                            f" either the same concept as {c.name} or a parent concept of {c.name}!\n\tIf you want to "
                            f'make this concept non-instantiable, set the "{ConceptDefinition.concept_abstract}" '
                            f'keyword in the concept definition to "true"!',
                            location_id=c.location_of(ConceptDefinition.concept_distinct_from) + [index],
                        )
            except ConceptHierarchyError as e:
                errors.append(e)
        if errors:
            if len(errors) == 1:
                raise errors[0]
            raise CHSemanticError("Processing concept data failed because of the errors below!", causes=errors)

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
        check_types_in_concept_hierarchy(self.context)

    def check_expressions(self):
        check_expressions_in_concept_hierarchy(self.context)

    def check(self):
        self.check_structure()
        self.check_after_parsing_concepts()
        self.check_specializations()
        self.check_types()
        self.check_expressions()


def check_model(model: ConceptHierarchyModel, checker: ConceptHierarchyChecker | None = None) -> None:
    """Validate syntax and semantic rules on *model*, raising on the first violation.

    Parameters
    ----------
    model:
        A :class:`~concept_hierarchy.models.ConceptHierarchyModel` produced by the parser.
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
