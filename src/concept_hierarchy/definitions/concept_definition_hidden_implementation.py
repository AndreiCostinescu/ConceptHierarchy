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

from __future__ import annotations

from abc import ABC, abstractmethod

from concept_hierarchy.definitions.concept_definition import ConceptDefinition
from concept_hierarchy.definitions.utils import check_ch_name
from concept_hierarchy.errors import CHSemanticError, CHSyntaxError


class HiddenImplementationDefinition(ConceptDefinition, ABC):
    hidden_implementation: str = "implementation"
    hidden_abstract: str = "abstract"
    hidden_template_arguments: str = "templateArguments"
    hidden_template_arguments_order: str = "order"
    hidden_template_arguments_substitutions: str = "substitutions"
    hidden_template_arguments_variadic_ids: str = "variadicGroupIdentifiers"
    default_template_argument_constraint: str = "ValueDomain"
    variadic_group_identifier_characters: str = "!$"
    implementation_related_keys: set[str] = {hidden_implementation, hidden_abstract, hidden_template_arguments}
    template_arguments_data_keys: set[str] = {
        hidden_template_arguments_order,
        hidden_template_arguments_substitutions,
        hidden_template_arguments_variadic_ids,
    }
    variadic_group_identifier_character_enumeration: str = ", ".join(
        "'" + x + "'" for x in variadic_group_identifier_characters
    )

    def __init__(self, name: str, definition_data: object):
        super().__init__(name, definition_data)

        self.implementation: str | None = None
        self.abstract: bool | None = None
        self.template_arguments: tuple[str, ...] | dict | None = None

        # if template_argument_order is (), then there are no template arguments
        self.template_argument_order: tuple[str, ...] = ()
        # unparsed template constraint formulae, must be strings
        self.template_argument_constraints: dict[str, str] = {}
        # mapping from (parent VD, parent template arg name) -> string value or list of strings variadic value
        self.substitution_of_template_arguments: dict[tuple[str | None, str], str | list[str]] = {}
        self.variadic_template_arguments: set[str] = set()
        self.variadic_template_argument_group_identifiers: dict[str, str] = {}

    @classmethod
    def from_node(cls, concept_definition: ConceptDefinition):
        domain_concept = cls._from_node(concept_definition)
        domain_concept.implementation = None
        domain_concept.abstract = None
        domain_concept.template_arguments = None

        domain_concept.template_argument_order = ()
        domain_concept.template_argument_constraints = {}
        domain_concept.substitution_of_template_arguments = {}
        domain_concept.variadic_template_arguments = set()
        domain_concept.variadic_template_argument_group_identifiers = {}

        return domain_concept

    @property
    @abstractmethod
    def definition_type(self) -> str:
        pass

    @property
    def definition_location(self) -> list[str]:
        return super().definition_location + ["data"]

    def check_template_argument_definition_list(self, t_arg_list: list):
        for t_arg_index, t_arg in enumerate(t_arg_list):
            if not isinstance(t_arg, str):
                raise CHSyntaxError(
                    f"Template argument names must be JSON uppercase-starting strings (ending in '...' "
                    f"if variadic), not {t_arg!r} at {self.definition_type} {self.name}!",
                    self.location_id(HiddenImplementationDefinition.hidden_template_arguments, t_arg_index),
                )
            is_variadic = t_arg.endswith("...")
            t_arg_clean = t_arg if not is_variadic else t_arg[:-3]
            if not check_ch_name(t_arg_clean, must_start_uppercase=True):
                raise CHSyntaxError(
                    f"Template argument names must be JSON uppercase-starting strings (ending in '...' "
                    f"if variadic), not {t_arg!r} at {self.definition_type} {self.name}!",
                    self.location_id(HiddenImplementationDefinition.hidden_template_arguments, t_arg_index),
                )
            self.template_argument_order += (t_arg_clean,)
            if is_variadic:
                self.variadic_template_arguments.add(t_arg_clean)
            self.template_argument_constraints[t_arg_clean] = (
                HiddenImplementationDefinition.default_template_argument_constraint
            )

    def check_implementation(self):
        # check "implementation"
        self.implementation = self.data.get(HiddenImplementationDefinition.hidden_implementation, None)
        if self.implementation is not None:
            if not isinstance(self.implementation, str):
                raise CHSyntaxError(
                    f"The definition of a {self.definition_type}'s implementation file must be a JSON string, not "
                    f"{self.implementation!r} for the {self.definition_type} {self.name}",
                    self.location_id(HiddenImplementationDefinition.hidden_implementation),
                )
            elif "." in self.implementation:
                raise CHSemanticError(
                    f"Do not define the file extensions for the {self.definition_type} implementation file! "
                    f"Found {self.implementation!r} for the {self.definition_type} {self.name}",
                    self.location_id(HiddenImplementationDefinition.hidden_implementation),
                )

    def check_abstract(self):
        # check "abstract"
        self.abstract = self.data.get(HiddenImplementationDefinition.hidden_abstract, None)
        if self.abstract is not None:
            if not isinstance(self.abstract, bool):
                raise CHSyntaxError(
                    f"The definition of a {self.definition_type}'s abstract marker must be a JSON boolean, not "
                    f"{self.abstract!r} for the {self.definition_type} {self.name}",
                    self.location_id(HiddenImplementationDefinition.hidden_abstract),
                )
        else:
            self.abstract = False

    def check_template_arguments(self):
        # check "templateArguments"
        # can only be completely parsed after the concept data is initialized for all concepts
        #   because of constraints and substitutions requiring template-instantiations of other concepts
        self.template_arguments = self.data.get(HiddenImplementationDefinition.hidden_template_arguments, None)
        if self.template_arguments is not None:
            if not isinstance(self.template_arguments, (dict, list)):
                raise CHSyntaxError(
                    f"The definition of a {self.definition_type}'s template arguments must be a JSON array or object, "
                    f"not {self.template_arguments!r} for the {self.definition_type} {self.name}",
                    self.location_id(HiddenImplementationDefinition.hidden_template_arguments),
                )
            elif isinstance(self.template_arguments, list):
                self.check_template_argument_definition_list(self.template_arguments)
            else:
                template_structure_data = {
                    k: v
                    for k, v in self.template_arguments.items()
                    if k in HiddenImplementationDefinition.template_arguments_data_keys
                }
                if HiddenImplementationDefinition.hidden_template_arguments_order in template_structure_data:
                    if not isinstance(
                        template_structure_data[HiddenImplementationDefinition.hidden_template_arguments_order], list
                    ):
                        raise CHSyntaxError(
                            f"The definition of a {self.definition_type}'s template arguments must contain the order of"
                            f" arguments (specified at the "
                            f'"{HiddenImplementationDefinition.hidden_template_arguments_order}" keyword) as a JSON '
                            f"array of strings!\n\tGot "
                            f"{template_structure_data[HiddenImplementationDefinition.hidden_template_arguments_order]!r}",
                            self.location_id(
                                HiddenImplementationDefinition.hidden_template_arguments,
                                HiddenImplementationDefinition.hidden_template_arguments_order,
                            ),
                        )
                    else:
                        self.check_template_argument_definition_list(
                            template_structure_data[HiddenImplementationDefinition.hidden_template_arguments_order]
                        )
                # missing checks: "substitution_of_template_arguments" TO BE CHECKED AFTER ALL CONCEPTS ARE INITIALIZED
                substitution_data = template_structure_data.get(
                    HiddenImplementationDefinition.hidden_template_arguments_substitutions, {}
                )
                if not isinstance(substitution_data, dict):
                    raise CHSyntaxError(
                        f"The definition of the substitution of parent {self.definition_type} template arguments must "
                        f"be a JSON object mapping template argument identifiers (i.e. "
                        f'"DirectParentName:NameOrDirectParentTemplateArgument") as JSON strings to the substitution '
                        f"formula also specified as JSON strings.\n\tGot {self.substitution_of_template_arguments!r}",
                        self.location_id(
                            HiddenImplementationDefinition.hidden_template_arguments,
                            HiddenImplementationDefinition.hidden_template_arguments_substitutions,
                        ),
                    )
                else:
                    for subst_key, subst_data in substitution_data.items():
                        # assertion, not check because this is a key of a JSON object
                        assert isinstance(subst_key, str)
                        split_res = subst_key.split(":")
                        colon_count = len(split_res)
                        if colon_count > 1:
                            raise CHSyntaxError(
                                f"The substitution identifier of a parent {self.definition_type} template arguments "
                                f'must be "DirectParentName:NameOrDirectParentTemplateArgument".\n\tGot {subst_key!r}',
                                self.location_id(
                                    HiddenImplementationDefinition.hidden_template_arguments,
                                    HiddenImplementationDefinition.hidden_template_arguments_substitutions,
                                    subst_key,
                                ),
                            )
                        elif colon_count == 1:
                            self.substitution_of_template_arguments[split_res[0], split_res[1]] = subst_data
                        else:
                            self.substitution_of_template_arguments[None, split_res[0]] = subst_data
                if (
                    self.template_argument_order == ()
                    and HiddenImplementationDefinition.hidden_template_arguments_variadic_ids in template_structure_data
                ):
                    raise CHSyntaxError(
                        f"Can't specify variadicGroupIdentifiers for {self.definition_type} that does not define any "
                        f"template argument (based on missing "
                        f'"{HiddenImplementationDefinition.hidden_template_arguments_order}" keyword in the '
                        f"definition of {self.name})",
                        self.location_id(
                            HiddenImplementationDefinition.hidden_template_arguments,
                            HiddenImplementationDefinition.hidden_template_arguments_variadic_ids,
                        ),
                    )
                self.variadic_template_argument_group_identifiers = template_structure_data.get(
                    HiddenImplementationDefinition.hidden_template_arguments_variadic_ids, {}
                )
                if not isinstance(self.variadic_template_argument_group_identifiers, dict):
                    raise CHSyntaxError(
                        f"The definition of variadic group identifiers must be a JSON object mapping variadic template "
                        f"argument names to a JSON string identifier containing only the characters '!' and '$' "
                        f"(or empty).\n\tGot {self.variadic_template_argument_group_identifiers!r}",
                        self.location_id(
                            HiddenImplementationDefinition.hidden_template_arguments,
                            HiddenImplementationDefinition.hidden_template_arguments_variadic_ids,
                        ),
                    )
                else:
                    allow_empty_identifier = len(self.variadic_template_arguments) == len(self.template_argument_order)
                    for var_t_arg, var_t_g_id in self.variadic_template_argument_group_identifiers.items():
                        # assertion, not check because this is a key of a JSON object
                        assert isinstance(var_t_arg, str)
                        if var_t_arg.endswith("...") and var_t_arg[:-3] in self.template_argument_constraints:
                            raise CHSyntaxError(
                                f"The key entry in the variadic group identifiers JSON must be a variadic template "
                                f"argument without its '...' variadic identifier!\n\tGot {var_t_arg!r}",
                                self.location_id(
                                    HiddenImplementationDefinition.hidden_template_arguments,
                                    HiddenImplementationDefinition.hidden_template_arguments_variadic_ids,
                                    var_t_arg,
                                ),
                            )
                        elif var_t_arg not in self.template_argument_constraints:
                            raise CHSemanticError(
                                f"{var_t_arg!r} is not a template argument of {self.name}!",
                                self.location_id(
                                    HiddenImplementationDefinition.hidden_template_arguments,
                                    HiddenImplementationDefinition.hidden_template_arguments_variadic_ids,
                                    var_t_arg,
                                ),
                            )
                        elif var_t_arg not in self.variadic_template_arguments:
                            raise CHSemanticError(
                                f"{var_t_arg!r} is not a variadic template argument of {self.name}!",
                                self.location_id(
                                    HiddenImplementationDefinition.hidden_template_arguments,
                                    HiddenImplementationDefinition.hidden_template_arguments_variadic_ids,
                                    var_t_arg,
                                ),
                            )
                        if any(
                            char not in HiddenImplementationDefinition.variadic_group_identifier_characters
                            for char in var_t_g_id
                        ):
                            raise CHSyntaxError(
                                f"Valid variadic group identifiers contain only the characters "
                                f"{HiddenImplementationDefinition.variadic_group_identifier_character_enumeration}."
                                f"\n\tGot invalid {var_t_g_id!r}!",
                                self.location_id(
                                    HiddenImplementationDefinition.hidden_template_arguments,
                                    HiddenImplementationDefinition.hidden_template_arguments_variadic_ids,
                                    var_t_arg,
                                ),
                            )
                        if var_t_g_id == "" and not allow_empty_identifier:
                            raise CHSemanticError(
                                f"The empty variadic group identifier is only allowed when the {self.definition_type}"
                                f" defines only variadic template arguments. This is not the case for {self.name}",
                                self.location_id(
                                    HiddenImplementationDefinition.hidden_template_arguments,
                                    HiddenImplementationDefinition.hidden_template_arguments_variadic_ids,
                                    var_t_arg,
                                ),
                            )

                for t_arg, t_arg_constraint in self.template_arguments.items():
                    if t_arg in template_structure_data:
                        continue
                    if t_arg not in self.template_argument_constraints:
                        raise CHSemanticError(
                            f"Can't define a constraint for {t_arg!r} which is not a template argument of {self.name}. "
                            f"\n\tIt only has these template arguments: {self.template_argument_order!r}",
                            self.location_id(HiddenImplementationDefinition.hidden_template_arguments, t_arg),
                        )
                    if not isinstance(t_arg_constraint, str):
                        raise CHSyntaxError(
                            f"The definition of a {self.definition_type} template argument constraint formulae must "
                            f"be a JSON string, got {t_arg_constraint!r}",
                            self.location_id(
                                HiddenImplementationDefinition.hidden_template_arguments, t_arg_constraint
                            ),
                        )
                    # parse the formula later, after all concepts are initialized!
                    self.template_argument_constraints[t_arg] = t_arg_constraint

    def concept_data_check(self):
        data_keys: set[str] = set(self.data.keys())
        if not (data_keys <= self.implementation_related_keys):
            extra_keys = data_keys - self.implementation_related_keys
            raise CHSyntaxError(
                f"Found extra keys {extra_keys!r} in the {self.definition_type} data definition of {self.name}",
                self.location_id(),
            )

        self.check_implementation()
        self.check_abstract()
        self.check_template_arguments()

    def is_templatable(self):
        return self.template_argument_order != ()
