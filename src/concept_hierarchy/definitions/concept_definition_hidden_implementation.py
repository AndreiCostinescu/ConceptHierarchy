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
from concept_hierarchy.definitions.definition import LocationOfCheckData, StopLocationOfCheck
from concept_hierarchy.definitions.utils import check_ch_name
from concept_hierarchy.errors import CHSemanticError, CHSyntaxError, LocationId, PathPart


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

    def __init__(self, name: str, definition_data: object, definition_location_id: LocationId):
        super().__init__(name, definition_data, definition_location_id)

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

    @abstractmethod
    def definition_type(self) -> str:
        pass

    def definition_location(self) -> LocationId:
        return self.data_location_id

    def location_of_impl(self, *keywords: str) -> LocationOfCheckData:
        # process top-level implementation-related data keywords (abstract/implementation/templateArguments)
        # after processing parent keywords
        check_res = super().location_of_impl(*keywords)
        self.check_location_id(
            check_res,
            HiddenImplementationDefinition.definition_location(self) + [check_res.first_remaining],
            location_check=self.data,
            previous_location=ConceptDefinition.concept_definition_data,
            allow_start_at_this_location=True,
        )
        # Saving this value here is needed because there are subclasses
        # This is the guarantee that the top-level data keywords were found in this function,
        # so no more processing in subclasses is required!
        found_keyword_in_here = check_res.check_successful is True
        # stop the check if reached the leaf-nodes: implementation or abstract
        if (
            found_keyword_in_here
            and check_res.last_consumed != HiddenImplementationDefinition.hidden_template_arguments
        ):
            raise StopLocationOfCheck(check_res)
        # other sub-keywords/-locations that can be processed in this class live in "templateArguments"
        if HiddenImplementationDefinition.hidden_template_arguments not in self.data:
            # if there are no templateArguments defined in the data, there is nothing left to process
            return check_res
        # differentiate between the array definition and the object definition
        template_arguments_data = self.data[HiddenImplementationDefinition.hidden_template_arguments]
        assert isinstance(template_arguments_data, (list, dict))
        if isinstance(template_arguments_data, list):
            # process "order" keyword, which is actually the top-level templateArguments definition for the array-syntax
            self.check_location_id(
                check_res,
                HiddenImplementationDefinition.definition_location(self)
                + [HiddenImplementationDefinition.hidden_template_arguments],
                location_check=HiddenImplementationDefinition.hidden_template_arguments_order,
                previous_location=HiddenImplementationDefinition.hidden_template_arguments,
                allow_start_at_this_location=True,
            )
            # stop check if templateArguments was found (found_keyword_in_here) or if order was found
            if found_keyword_in_here or check_res.check_successful:
                assert check_res.last_consumed == HiddenImplementationDefinition.hidden_template_arguments_order
                raise StopLocationOfCheck(check_res)
            return check_res

        # check "order", "variadicGroupIdentifiers", "substitution" keywords as well as template argument constraints
        self.check_location_id(
            check_res,
            HiddenImplementationDefinition.definition_location(self)
            + [HiddenImplementationDefinition.hidden_template_arguments, check_res.first_remaining],
            location_check=template_arguments_data,
            previous_location=HiddenImplementationDefinition.hidden_template_arguments,
            allow_start_at_this_location=True,
        )
        if check_res.check_successful and (
            check_res.last_consumed in self.template_argument_constraints
            or check_res.last_consumed == HiddenImplementationDefinition.hidden_template_arguments_order
        ):
            # processed a template argument or the "order" keyword; there is no more data
            raise StopLocationOfCheck(check_res)

        # check variadicGroupIdentifiers
        template_arguments_variadic_data = template_arguments_data.get(
            HiddenImplementationDefinition.hidden_template_arguments_variadic_ids, {}
        )
        self.check_location_id(
            check_res,
            HiddenImplementationDefinition.definition_location(self)
            + [
                HiddenImplementationDefinition.hidden_template_arguments,
                HiddenImplementationDefinition.hidden_template_arguments_variadic_ids,
                check_res.first_remaining,
            ],
            location_check=template_arguments_variadic_data,
            previous_location=HiddenImplementationDefinition.hidden_template_arguments_variadic_ids,
            allow_start_at_this_location=False,
        )
        if check_res.check_successful:
            assert check_res.last_consumed in self.template_argument_constraints
            # processed a template argument defined in variadicGroupIdentifiers; there is no more data
            raise StopLocationOfCheck(check_res)

        # check substitutions
        template_arguments_substitution_data = template_arguments_data.get(
            HiddenImplementationDefinition.hidden_template_arguments_substitutions, {}
        )
        check_res = self.check_location_id(
            check_res,
            HiddenImplementationDefinition.definition_location(self)
            + [
                HiddenImplementationDefinition.hidden_template_arguments,
                HiddenImplementationDefinition.hidden_template_arguments_substitutions,
                check_res.first_remaining,
            ],
            location_check=template_arguments_substitution_data,
            previous_location=HiddenImplementationDefinition.hidden_template_arguments_substitutions,
            allow_start_at_this_location=False,
        )
        if found_keyword_in_here or check_res.check_successful:
            # found_keyword_in_here means processed templateArguments keyword => don't continue parsing other keywords
            raise StopLocationOfCheck(check_res)
        return check_res

    def check_template_argument_definition_list(self, t_arg_list: list):
        for t_arg_index, t_arg in enumerate(t_arg_list):
            if not isinstance(t_arg, str):
                raise CHSyntaxError(
                    f"Template argument names must be JSON uppercase-starting strings (ending in '...' "
                    f"if variadic), not {t_arg!r} at {self.definition_type()} {self.name}!",
                    location_id=self.location_id(HiddenImplementationDefinition.hidden_template_arguments, t_arg_index),
                )
            is_variadic = t_arg.endswith("...")
            t_arg_clean = t_arg if not is_variadic else t_arg[:-3]
            if not check_ch_name(t_arg_clean, must_start_uppercase=True):
                raise CHSyntaxError(
                    f"Template argument names must be JSON uppercase-starting strings (ending in '...' "
                    f"if variadic), not {t_arg!r} at {self.definition_type()} {self.name}!",
                    location_id=self.location_id(HiddenImplementationDefinition.hidden_template_arguments, t_arg_index),
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
                    f"The definition of a {self.definition_type()}'s implementation file must be a JSON string, not "
                    f"{self.implementation!r} for the {self.definition_type()} {self.name}",
                    location_id=self.location_id(HiddenImplementationDefinition.hidden_implementation),
                    part=PathPart.VALUE,
                )
            elif "." in self.implementation:
                raise CHSemanticError(
                    f"Do not define the file extensions for the {self.definition_type()} implementation file! "
                    f"Found {self.implementation!r} for the {self.definition_type()} {self.name}",
                    location_id=self.location_id(HiddenImplementationDefinition.hidden_implementation),
                    part=PathPart.VALUE,
                )

    def check_abstract(self):
        # check "abstract"
        self.abstract = self.data.get(HiddenImplementationDefinition.hidden_abstract, None)
        if self.abstract is not None:
            if not isinstance(self.abstract, bool):
                raise CHSyntaxError(
                    f"The definition of a {self.definition_type()}'s abstract marker must be a JSON boolean, not "
                    f"{self.abstract!r} for the {self.definition_type()} {self.name}",
                    location_id=self.location_id(HiddenImplementationDefinition.hidden_abstract),
                    part=PathPart.VALUE,
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
                    f"The definition of a {self.definition_type()}'s template arguments must be a JSON array or "
                    f"object, not {self.template_arguments!r} for the {self.definition_type()} {self.name}",
                    location_id=self.location_id(HiddenImplementationDefinition.hidden_template_arguments),
                    part=PathPart.VALUE,
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
                            f"The definition of a {self.definition_type()}'s template arguments must contain the order "
                            f"of arguments (specified at the "
                            f'"{HiddenImplementationDefinition.hidden_template_arguments_order}" keyword) as a JSON '
                            f"array of strings!\n\tGot "
                            f"{template_structure_data[HiddenImplementationDefinition.hidden_template_arguments_order]!r}",
                            location_id=self.location_id(
                                HiddenImplementationDefinition.hidden_template_arguments,
                                HiddenImplementationDefinition.hidden_template_arguments_order,
                            ),
                            part=PathPart.VALUE,
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
                        f"The definition of the substitution of parent {self.definition_type()} template arguments must"
                        f" be a JSON object mapping template argument identifiers (i.e. "
                        f'"DirectParentName:NameOrDirectParentTemplateArgument") as JSON strings to the substitution '
                        f"formula also specified as JSON strings.\n\tGot {self.substitution_of_template_arguments!r}",
                        location_id=self.location_id(
                            HiddenImplementationDefinition.hidden_template_arguments,
                            HiddenImplementationDefinition.hidden_template_arguments_substitutions,
                        ),
                        part=PathPart.VALUE,
                    )
                elif (
                    HiddenImplementationDefinition.hidden_template_arguments_substitutions in template_structure_data
                    and substitution_data == {}
                ):
                    raise CHSyntaxError(
                        f'Defined empty "{HiddenImplementationDefinition.hidden_template_arguments_substitutions}" in'
                        f" concept {self.name}.\n\tThis is not allowed: either specify substitutions or remove the "
                        f'definition of the keyword from "{HiddenImplementationDefinition.hidden_template_arguments}"'
                        f" altogether.",
                        location_id=self.location_id(
                            HiddenImplementationDefinition.hidden_template_arguments,
                            HiddenImplementationDefinition.hidden_template_arguments_substitutions,
                        ),
                        part=PathPart.VALUE,
                    )
                else:
                    for subst_key, subst_data in substitution_data.items():
                        # assertion, not check because this is a key of a JSON object
                        assert isinstance(subst_key, str)
                        split_res = subst_key.split(":")
                        colon_count = len(split_res) - 1
                        if colon_count > 1:
                            raise CHSyntaxError(
                                f"The substitution identifier of a parent {self.definition_type()} template arguments "
                                f'must be "DirectParentName:NameOrDirectParentTemplateArgument".\n\tGot {subst_key!r}',
                                location_id=self.location_id(
                                    HiddenImplementationDefinition.hidden_template_arguments,
                                    HiddenImplementationDefinition.hidden_template_arguments_substitutions,
                                    subst_key,
                                ),
                                part=PathPart.KEY,
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
                        f"Can't specify variadicGroupIdentifiers for {self.definition_type()} that does not define any "
                        f"template argument (based on missing "
                        f'"{HiddenImplementationDefinition.hidden_template_arguments_order}" keyword in the '
                        f"definition of {self.name})",
                        location_id=self.location_id(
                            HiddenImplementationDefinition.hidden_template_arguments,
                            HiddenImplementationDefinition.hidden_template_arguments_variadic_ids,
                        ),
                        part=PathPart.KEY,
                    )
                self.variadic_template_argument_group_identifiers = template_structure_data.get(
                    HiddenImplementationDefinition.hidden_template_arguments_variadic_ids, {}
                )
                if not isinstance(self.variadic_template_argument_group_identifiers, dict):
                    raise CHSyntaxError(
                        f"The definition of variadic group identifiers must be a JSON object mapping variadic template "
                        f"argument names to a JSON string identifier containing only the characters '!' and '$' "
                        f"(or empty).\n\tGot {self.variadic_template_argument_group_identifiers!r}",
                        location_id=self.location_id(
                            HiddenImplementationDefinition.hidden_template_arguments,
                            HiddenImplementationDefinition.hidden_template_arguments_variadic_ids,
                        ),
                        part=PathPart.VALUE,
                    )
                elif (
                    HiddenImplementationDefinition.hidden_template_arguments_variadic_ids in template_structure_data
                    and self.variadic_template_argument_group_identifiers == {}
                ):
                    raise CHSyntaxError(
                        f'Defined empty "{HiddenImplementationDefinition.hidden_template_arguments_variadic_ids}" in'
                        f" concept {self.name}.\n\tThis is not allowed: either specify variadic group ids or remove the"
                        f' definition of the keyword from "{HiddenImplementationDefinition.hidden_template_arguments}"'
                        f" altogether.",
                        location_id=self.location_id(
                            HiddenImplementationDefinition.hidden_template_arguments,
                            HiddenImplementationDefinition.hidden_template_arguments_variadic_ids,
                        ),
                        part=PathPart.VALUE,
                    )
                else:
                    allow_empty_identifier = len(self.variadic_template_arguments) == len(self.template_argument_order)
                    # Check if either no variadic template argument has a variadic group identifier or all of them have
                    is_variadic_id_defined: str | None = None
                    for var_t_arg, var_t_g_id in self.variadic_template_argument_group_identifiers.items():
                        # assertion, not check because this is a key of a JSON object
                        assert isinstance(var_t_arg, str)
                        if var_t_arg.endswith("...") and var_t_arg[:-3] in self.template_argument_constraints:
                            raise CHSyntaxError(
                                f"The key entry in the variadic group identifiers JSON must be a variadic template "
                                f"argument without its '...' variadic identifier!\n\tGot {var_t_arg!r}",
                                location_id=self.location_id(
                                    HiddenImplementationDefinition.hidden_template_arguments,
                                    HiddenImplementationDefinition.hidden_template_arguments_variadic_ids,
                                    var_t_arg,
                                ),
                                part=PathPart.KEY,
                            )
                        elif var_t_arg not in self.template_argument_constraints:
                            raise CHSemanticError(
                                f"{var_t_arg!r} is not a template argument of {self.name}!",
                                location_id=self.location_id(
                                    HiddenImplementationDefinition.hidden_template_arguments,
                                    HiddenImplementationDefinition.hidden_template_arguments_variadic_ids,
                                    var_t_arg,
                                ),
                                part=PathPart.KEY,
                            )
                        elif var_t_arg not in self.variadic_template_arguments:
                            raise CHSemanticError(
                                f"{var_t_arg!r} is not a variadic template argument of {self.name}!",
                                location_id=self.location_id(
                                    HiddenImplementationDefinition.hidden_template_arguments,
                                    HiddenImplementationDefinition.hidden_template_arguments_variadic_ids,
                                    var_t_arg,
                                ),
                                part=PathPart.KEY,
                            )
                        if any(
                            char not in HiddenImplementationDefinition.variadic_group_identifier_characters
                            for char in var_t_g_id
                        ):
                            raise CHSyntaxError(
                                f"Valid variadic group identifiers contain only the characters "
                                f"{HiddenImplementationDefinition.variadic_group_identifier_character_enumeration}."
                                f"\n\tGot invalid {var_t_g_id!r}!",
                                location_id=self.location_id(
                                    HiddenImplementationDefinition.hidden_template_arguments,
                                    HiddenImplementationDefinition.hidden_template_arguments_variadic_ids,
                                    var_t_arg,
                                ),
                                part=PathPart.VALUE,
                            )
                        if var_t_g_id == "" and not allow_empty_identifier:
                            raise CHSemanticError(
                                f"The empty variadic group identifier is only allowed when the {self.definition_type()}"
                                f" defines only variadic template arguments. This is not the case for {self.name}",
                                location_id=self.location_id(
                                    HiddenImplementationDefinition.hidden_template_arguments,
                                    HiddenImplementationDefinition.hidden_template_arguments_variadic_ids,
                                    var_t_arg,
                                ),
                                part=PathPart.VALUE,
                            )
                        is_variadic_id_defined = var_t_arg
                    if isinstance(is_variadic_id_defined, str):
                        for var_t_arg in self.variadic_template_arguments:
                            if var_t_arg not in self.variadic_template_argument_group_identifiers:
                                raise CHSemanticError(
                                    f"Either no variadic group identifier is defined for variadic template arguments of"
                                    f" a variadic group identifier is defined for all variadic template arguments!\n"
                                    f"{self.name} has defined a variadic group id for {is_variadic_id_defined} but does"
                                    f" not define one for {var_t_arg}!",
                                    location_id=self.location_id(
                                        HiddenImplementationDefinition.hidden_template_arguments,
                                        HiddenImplementationDefinition.hidden_template_arguments_variadic_ids,
                                    ),
                                    part=PathPart.VALUE,
                                )

                for t_arg, t_arg_constraint in self.template_arguments.items():
                    if t_arg in template_structure_data:
                        continue
                    if t_arg not in self.template_argument_constraints:
                        raise CHSemanticError(
                            f"Can't define a constraint for {t_arg!r} which is not a template argument of {self.name}. "
                            f"\n\tIt only has these template arguments: {self.template_argument_order!r}",
                            location_id=self.location_id(
                                HiddenImplementationDefinition.hidden_template_arguments, t_arg
                            ),
                            part=PathPart.KEY,
                        )
                    if not isinstance(t_arg_constraint, str):
                        raise CHSyntaxError(
                            f"The definition of a {self.definition_type()} template argument constraint formulae must "
                            f"be a JSON string, got {t_arg_constraint!r}",
                            location_id=self.location_id(
                                HiddenImplementationDefinition.hidden_template_arguments, t_arg_constraint
                            ),
                            part=PathPart.VALUE,
                        )
                    # missing checks: parse the formula later, after all concepts are initialized!
                    self.template_argument_constraints[t_arg] = t_arg_constraint

    def concept_data_check(self):
        data_keys: set[str] = set(self.data.keys())
        if not (data_keys <= self.implementation_related_keys):
            extra_keys = data_keys - self.implementation_related_keys
            raise CHSyntaxError(
                f"Found extra keys {extra_keys!r} in the {self.definition_type()} data definition of {self.name}",
                location_id=self.location_id(),
                part=PathPart.VALUE,
            )

        self.check_implementation()
        self.check_abstract()
        self.check_template_arguments()
        # missing checks:
        #  - template constraint formulae
        #    REQUIRES all concepts to be initialized
        #  - valid template substitution keys
        #    check that the shorthand notation is allowed:
        #       only if two parent templated ValueDomains do not define the same template argument name
        #    REQUIRES all concepts to be initialized (parents with their value domains)
        #    STRUCTURE CHECK
        #  - check that all parent template arguments are substituted in this concept
        #    REQUIRES all concepts to be initialized
        #    STRUCTURE CHECK
        #  - valid template substitution values
        #    valid instantiated types or literals or variadic groups
        #       that satisfy the constraints of the parent template type!
        #    TYPE CHECK

    def is_templatable(self):
        return self.template_argument_order != ()
