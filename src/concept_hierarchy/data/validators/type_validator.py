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
from dataclasses import dataclass

from concept_hierarchy.data.contexts.template_context import TemplateContext
from concept_hierarchy.data.parsers.type_parser import parse_type
from concept_hierarchy.data.types.concept_hierarchy_types import (
    ConceptHierarchyTemplateArgument,
    ExpandedVariadicTemplateVariable,
    Instantiated,
    InstantiatedType,
    InstantiatedVariadicGroup,
    LiteralValue,
    NonVariadicTemplateVariable,
    TemplateDependent,
    TemplateDependentType,
    TemplateDependentVariadicGroup,
    TemplateVariable,
    VariadicTemplateVariable,
)
from concept_hierarchy.data.types.parsed_type import (
    ParsedType,
    TemplateArgumentLiteral,
    TemplateArgumentValue,
    TemplateArgumentVariadicGroup,
    TemplateArgumentWithVariadicId,
)
from concept_hierarchy.data.value_domain_type import ConceptHierarchyType
from concept_hierarchy.errors import CHSemanticError, CHSyntaxError, LocationId


@dataclass(frozen=True)
class TypeTemplateData:
    context: TemplateContext
    variadic_group_identifiers: dict[str, str]
    defined_variadic_group_identifiers: dict[str, str]


class TypeValidator(ABC):
    @abstractmethod
    def full_type_name(self, concept_name: str) -> str:
        pass

    @abstractmethod
    def get_template_data_of(self, concept_name: str) -> TypeTemplateData:
        """
        Validates:
         - that the number of template arguments are correctly defined,
         - that the variadic identifiers are correctly used in template argument values,
         - transforms the representation from using variadic identifiers to using only variadic groups
         - verifies that the constraints on template values are satisfied
        """
        pass

    @abstractmethod
    def is_concept(self, concept_name: str) -> bool:
        pass

    @abstractmethod
    def is_template_variable(self, concept_name: str) -> bool:
        pass

    @abstractmethod
    def is_variadic_template_variable(self, concept_name: str) -> bool:
        pass

    @abstractmethod
    def get_available_template_variables(self) -> list[str]:
        pass


def validate_template_argument_values_of_type(
    ch_type: ParsedType, validator: TypeValidator, location_id: LocationId, context: TemplateContext
) -> ParsedType:
    if not ch_type.is_templated:
        return ch_type
    assert len(ch_type.template_arguments) == context.nr_variables
    new_template_arguments = []
    for t_arg, t_arg_name in zip(ch_type.template_arguments, context.variables):
        is_variadic_template_argument = t_arg_name in context.variadic_variables
        new_location_id = location_id + [f"{ch_type.clean_name} template argument {t_arg.full_name}"]
        t_arg_valid = validate_template_argument_value(t_arg, validator, new_location_id, is_variadic_template_argument)
        new_template_arguments.append(t_arg_valid)
    return ParsedType(
        variadic_group_identifier=ch_type.variadic_group_identifier,
        name=ch_type.name,
        has_variadic_template_expansion=ch_type.has_variadic_template_expansion,
        template_arguments=tuple(new_template_arguments),
        function_arguments=ch_type.function_arguments,
    )


def validate_type_and_parse_to_variadic_groups(
    ch_type: ParsedType,
    validator: TypeValidator,
    location_id: LocationId,
    full_type_name: str,
    template_data: TypeTemplateData,
) -> ParsedType:
    # Check whether the name is a template variable or a concept;
    # If it is a template argument, the checks have already been done in validate_type
    # If it is a concept, check
    # - whether the template arguments are correctly specified
    # - whether the shorthand syntax for variadic groups is correct, etc
    if validator.is_template_variable(ch_type.clean_name):
        return ch_type
    if not validator.is_concept(ch_type.clean_name):
        available_template_variables = validator.get_available_template_variables()
        raise CHSemanticError(
            f"Unknown type identifier {ch_type.clean_name}: it is neither a template variable, nor a concept.\n"
            f"Available variables are {available_template_variables!r}",
            location_id=location_id,
        )
    # this is a concept
    if ch_type.has_variadic_template_expansion:
        raise CHSemanticError(
            f"The variadic expansion operator may only be used on template arguments, not on concepts. Used on "
            f"{ch_type.clean_name}",
            location_id=location_id,
        )

    # check type's template arguments
    type_is_templated = template_data.context.variables != ()
    if type_is_templated != ch_type.is_templated:
        if type_is_templated:
            raise CHSemanticError(
                f"Did not define template arguments for the templated type {full_type_name}",
                location_id=location_id,
            )
        raise CHSemanticError(
            f"Defined template arguments {ch_type.full_name} on the type {ch_type.clean_name} that does not "
            f"have template arguments!",
            location_id=location_id,
        )
    if not type_is_templated:  # finish validation
        return ch_type

    parsed_template_arguments = ch_type.template_arguments
    nr_template_arguments = template_data.context.nr_variables
    nr_variadic_template_arguments = len(template_data.context.variadic_variables)
    nr_parsed_template_arguments = len(parsed_template_arguments)

    if (nr_template_arguments - nr_variadic_template_arguments) > nr_parsed_template_arguments:
        raise CHSemanticError(
            f"Too few arguments specified for template type {full_type_name}: {parsed_template_arguments}",
            location_id=location_id,
        )

    uses_variadic_ids = nr_template_arguments != nr_parsed_template_arguments
    if uses_variadic_ids and nr_variadic_template_arguments == 0:
        raise CHSemanticError(
            f"Mismatch between the expected number of arguments {nr_template_arguments} of type {full_type_name} "
            f"and the parsed ones {parsed_template_arguments!r}.",
            location_id=location_id,
        )

    # check whether the variadic group identifiers syntax is mixed with the explicit list-syntax
    uses_variadic_groups = False
    allow_empty_variadic_identifier = (
        "" in template_data.defined_variadic_group_identifiers or nr_variadic_template_arguments < nr_template_arguments
    )
    count_values_without_variadic_identifier = 0
    for t_arg_name, parsed_t_arg_val in zip(template_data.context.variables, parsed_template_arguments):
        is_variadic_group = isinstance(parsed_t_arg_val, TemplateArgumentVariadicGroup)
        uses_variadic_groups |= is_variadic_group
        if (t_arg_name in template_data.context.variadic_variables) != is_variadic_group:
            # if found a variadic group at a position where none was expected,
            # or didn't find a variadic group at a position where one was expected,
            #  then the ParsedType is using the syntax with variadic group ids
            uses_variadic_ids = True
        # check that the used variadic ids actually match a variadic group defined in the concept definition
        if isinstance(parsed_t_arg_val, TemplateArgumentWithVariadicId):
            # if the variadic identifier is not known for this type (no variadic template argument defines it)
            if (
                parsed_t_arg_val.has_variadic_identifier
                and parsed_t_arg_val.variadic_group_identifier not in template_data.defined_variadic_group_identifiers
            ):
                raise CHSemanticError(
                    f"Found a template argument value {parsed_t_arg_val.full_name} with an undefined variadic "
                    f"identifier {parsed_t_arg_val.variadic_group_identifier}.\nDefined variadic identifiers "
                    f"are {template_data.defined_variadic_group_identifiers!r}",
                    location_id=location_id,
                )
            # if empty variadic identifier is not allowed
            #   (i.e. when all arguments are variadic and none of them defines the empty variadic identifier)
            # but the empty variadic identifier was used (by not-specifying one)
            if not parsed_t_arg_val.has_variadic_identifier and not allow_empty_variadic_identifier:
                raise CHSemanticError(
                    f"Empty variadic identifier is not allowed for template argument value {parsed_t_arg_val} of "
                    f"{full_type_name}!",
                    location_id=location_id,
                )
            count_values_without_variadic_identifier += parsed_t_arg_val.has_variadic_identifier
    if uses_variadic_ids and uses_variadic_groups:
        # Can't catch the mixed-syntax in the type parsing
        # (because the empty variadic identifier does not look like a variadic identifier).
        # This is the check that checks the correct use of the empty variadic identifier with variadic groups
        # Example: Instance<T1..., T2...> (with T1: "", and T2: "!") instantiated as Instance<Concept, []>
        #   looks valid at parse time, but is actually invalid
        assert "" in template_data.defined_variadic_group_identifiers
        raise CHSemanticError(
            f"Can not define template argument values combining variadic groups and types with variadic "
            f"identifiers! Found at {ch_type.full_name!r}",
            location_id=location_id,
        )
    non_variadic_template_arguments = [
        x for x in template_data.context.variables if x not in template_data.context.variadic_variables
    ]
    if "" not in template_data.defined_variadic_group_identifiers and count_values_without_variadic_identifier < len(
        non_variadic_template_arguments
    ):
        raise CHSemanticError(
            f"The template-instantiation of {full_type_name} in {ch_type.full_name} does not specify a value for "
            f"the non-variadic template argument(s) "
            f"{non_variadic_template_arguments[count_values_without_variadic_identifier:]}",
            location_id=location_id,
        )
    # Only bring to canonic form (i.e. only using variadic groups) if variadic ids are used
    if uses_variadic_ids:
        ch_type = make_canonic(ch_type, template_data.context.variables, template_data.variadic_group_identifiers)
    return validate_template_argument_values_of_type(
        ch_type,
        validator,
        location_id,
        template_data.context,
    )


def make_canonic(
    ch_type: ParsedType, template_argument_order: tuple[str, ...], variadic_group_identifiers: dict[str, str]
) -> ParsedType:
    # verify variadicGroupIdentifiers and create the variadic groups
    variadic_groups: dict[str | None, list[TemplateArgumentWithVariadicId]] = {}
    # collect variadic group elements and check whether the variadic group identifiers are correctly used!
    for parsed_t_arg_val in ch_type.template_arguments:
        assert isinstance(parsed_t_arg_val, TemplateArgumentWithVariadicId)
        variadic_identifier = parsed_t_arg_val.variadic_group_identifier
        if variadic_identifier not in variadic_groups:
            variadic_groups[variadic_identifier] = []
        variadic_groups[variadic_identifier].append(parsed_t_arg_val)
    # create curated template argument
    curated_template_arguments: list[TemplateArgumentValue] = []
    passed_number_of_variadic_template_arguments = 0
    for index, t_arg in enumerate(template_argument_order):
        if t_arg in variadic_group_identifiers:
            t_arg_var_id = variadic_group_identifiers[t_arg]
            if t_arg_var_id in variadic_groups:
                new_variadic_group_elements: list[ParsedType | TemplateArgumentLiteral] = []
                for group_elem in variadic_groups[t_arg_var_id]:
                    # remove the variadic identifier from all template argument values inside a variadic group!
                    if isinstance(group_elem, TemplateArgumentLiteral):
                        new_variadic_group_elements.append(
                            TemplateArgumentLiteral(None, group_elem.literal_value, group_elem.literal_type)
                        )
                    else:
                        assert isinstance(group_elem, ParsedType)
                        new_variadic_group_elements.append(
                            ParsedType(
                                variadic_group_identifier=None,
                                name=group_elem.name,
                                has_variadic_template_expansion=group_elem.has_variadic_template_expansion,
                                template_arguments=group_elem.template_arguments,
                                function_arguments=group_elem.function_arguments,
                            )
                        )
                curated_template_arguments.append(TemplateArgumentVariadicGroup(tuple(new_variadic_group_elements)))
            else:
                curated_template_arguments.append(TemplateArgumentVariadicGroup(()))
            passed_number_of_variadic_template_arguments += 1
        else:
            # this is clearly a non-variadic template argument;
            # all non-variadic template argument values live inside the variadic_groups[None] list entry
            index_in_empty_variadic_group = index - passed_number_of_variadic_template_arguments
            assert len(variadic_groups[None]) > index_in_empty_variadic_group
            curated_template_arguments.append(variadic_groups[None][index_in_empty_variadic_group])
    return ParsedType(
        variadic_group_identifier=ch_type.variadic_group_identifier,
        name=ch_type.name,
        has_variadic_template_expansion=ch_type.has_variadic_template_expansion,
        template_arguments=tuple(curated_template_arguments),
        function_arguments=ch_type.function_arguments,
    )


def validate_type(
    t: ParsedType,
    validator: TypeValidator,
    location_id: LocationId | None,
    *,
    can_expand_variadic_template_arguments: bool = False,
) -> ParsedType:
    if location_id is None:
        location_id = []

    # If it is a template variable, check
    # - whether variadic values are allowed,
    # - whether a variadic template variable is expanded,
    # - whether a non-variadic template variable is expanded,
    # - whether it is allowed to expand a variadic template variable (can expand_variadic_template_arguments), etc.
    if validator.is_template_variable(t.clean_name):
        if validator.is_variadic_template_variable(t.clean_name):
            if t.has_variadic_template_expansion and not can_expand_variadic_template_arguments:
                raise CHSemanticError(
                    f"Use of the variadic expansion operator is not allowed at this location! Got {t.full_name}",
                    location_id=location_id,
                )
        elif t.has_variadic_template_expansion:
            raise CHSemanticError(
                f"Used the variadic expansion operator '...' on the non variadic template variable {t.clean_name}",
                location_id=location_id,
            )
        # defined template arguments on a template variable!
        if t.is_templated:
            raise CHSemanticError(
                f"Can not define template arguments on a template argument in the current version of the "
                f"Concept Hierarchy!\nFound {t.full_name}",
                location_id=location_id,
            )
        return t

    if not validator.is_concept(t.clean_name):
        raise CHSemanticError(
            f"ParsedType {t.full_name!r} is not a template variable (in this context) nor a concept!",
            location_id=location_id,
        )

    # Bring to canonic form:
    #   Use only variadic groups (no more ParsedTypes with variadic identifiers)
    #   Make sure that mixed syntax with the empty variadic identifier is NOT used
    #   Make sure that there are a correct amount of template arguments specified in the template-instantiation
    # And validate sub-template arguments
    #   this has to happen from inside the validator because only it knows
    #   whether the template argument type is a variadic argument or not
    type_template_data = validator.get_template_data_of(t.clean_name)
    full_type_name = validator.full_type_name(t.clean_name)
    return validate_type_and_parse_to_variadic_groups(t, validator, location_id, full_type_name, type_template_data)


def validate_template_argument_value(
    t: TemplateArgumentValue,
    validator: TypeValidator,
    location_id: LocationId | None,
    is_used_as_a_variadic_argument: bool,
) -> TemplateArgumentValue:
    if location_id is None:
        location_id = []

    if isinstance(t, TemplateArgumentWithVariadicId) and t.has_variadic_identifier:
        raise CHSyntaxError(
            f"Variadic identifiers should not have been used in this context! Found {t.variadic_group_identifier} at "
            f"{t.full_name}!",
            location_id=location_id,
        )
    if is_used_as_a_variadic_argument:
        if isinstance(t, TemplateArgumentLiteral):
            raise CHSemanticError(
                f"The literal value {t.full_name!r} can not be used as a variadic template argument value!",
                location_id=location_id,
            )
        elif isinstance(t, ParsedType):
            if not validator.is_variadic_template_variable(t.clean_name):
                if validator.is_template_variable(t.clean_name):
                    raise CHSemanticError(
                        f"The non-variadic template variable {t.full_name!r} can not be used as a variadic template "
                        f"argument value!",
                        location_id=location_id,
                    )
                raise CHSemanticError(
                    f"The type value {t.full_name!r} can not be used as a variadic template argument value!",
                    location_id=location_id,
                )
            elif t.has_variadic_template_expansion:
                raise
    elif isinstance(t, TemplateArgumentVariadicGroup):
        raise CHSemanticError(
            f"The variadic group {t.full_name!r} can not be used as a non-variadic template argument value!",
            location_id=location_id,
        )

    if isinstance(t, TemplateArgumentLiteral):
        return t
    if isinstance(t, ParsedType):
        return validate_type(t, validator, location_id)
    assert isinstance(t, TemplateArgumentVariadicGroup)
    validated_variadic_group = []
    for elem_index, group_elem in enumerate(t.variadic_group):
        new_location_id = location_id + [f"group element {elem_index}"]
        if isinstance(group_elem, TemplateArgumentLiteral):
            validated_variadic_group.append(group_elem)
        else:
            assert isinstance(group_elem, ParsedType)
            validated_variadic_group.append(
                validate_type(group_elem, validator, new_location_id, can_expand_variadic_template_arguments=True)
            )
    return TemplateArgumentVariadicGroup(tuple(validated_variadic_group))


def convert_items(
    concept_name: str, items: tuple[TemplateArgumentValue, ...], validator: TypeValidator
) -> tuple[tuple[ConceptHierarchyTemplateArgument, ...], bool]:
    """
    :param concept_name: the name of the concept that defines the template variables present in the items
    :param items: contains the items to respectively convert to ConceptHierarchyTemplateArgument values
    :param validator: the type validator
    :return: the converted values, whether any of the value depends on template variables
    """
    converted_items: list[ConceptHierarchyTemplateArgument] = []
    has_template_dependent_items = False
    for item in items:
        converted_item = convert_template_argument_to_concept_hierarchy_template_argument(concept_name, item, validator)
        if isinstance(converted_item, TemplateDependent):
            has_template_dependent_items = True
        else:
            assert isinstance(converted_item, Instantiated)
        converted_items.append(converted_item)
    return tuple(converted_items), has_template_dependent_items


def convert_template_argument_to_concept_hierarchy_template_argument(
    concept_name: str, t_arg: TemplateArgumentValue, validator: TypeValidator
) -> ConceptHierarchyTemplateArgument:
    """
    :param concept_name: is the concept that defines the template variables
    :param t_arg: is the value to be converted
    :param validator: is the type validator
    :return: the converted value
    """
    if isinstance(t_arg, TemplateArgumentLiteral):
        return LiteralValue(t_arg.clean_name, t_arg.literal_type)
    if isinstance(t_arg, ParsedType):
        if not validator.is_concept(t_arg.clean_name):
            assert validator.is_template_variable(t_arg.clean_name)
            if validator.is_variadic_template_variable(t_arg.clean_name):
                if t_arg.has_variadic_template_expansion:
                    return ExpandedVariadicTemplateVariable(t_arg.clean_name, concept_name)
                else:
                    return VariadicTemplateVariable(t_arg.clean_name, concept_name)
            else:
                assert not t_arg.has_variadic_template_expansion
                return NonVariadicTemplateVariable(t_arg.clean_name, concept_name)
        assert not t_arg.has_variadic_template_expansion
        # check if all the template arguments are instantiated or not
        if not t_arg.is_templated:
            return InstantiatedType(t_arg.clean_name, ())
        converted_template_arguments, has_template_dependent_template_arguments = convert_items(
            concept_name, t_arg.template_arguments, validator
        )
        if has_template_dependent_template_arguments:
            return TemplateDependentType(t_arg.clean_name, converted_template_arguments)
        return InstantiatedType(t_arg.clean_name, converted_template_arguments)
    assert isinstance(t_arg, TemplateArgumentVariadicGroup)
    converted_group_elements, has_template_dependent_group_elements = convert_items(
        concept_name, t_arg.variadic_group, validator
    )
    assert all(isinstance(x, (ConceptHierarchyType, LiteralValue, TemplateVariable)) for x in converted_group_elements)
    if has_template_dependent_group_elements:
        return TemplateDependentVariadicGroup(t_arg.clean_name, converted_group_elements)
    return InstantiatedVariadicGroup(t_arg.clean_name, converted_group_elements)


def parse_convert_type(
    concept_name: str, type_def: str, validator: TypeValidator, location_id: LocationId
) -> InstantiatedType:
    # check the syntax of the type
    validated_type = validate_type(
        parse_type(type_def, location_id, expected_number_of_values=1)[0], validator, location_id
    )
    # check the semantics of the type
    ch_type = convert_template_argument_to_concept_hierarchy_template_argument(concept_name, validated_type, validator)
    if not isinstance(ch_type, InstantiatedType):
        raise CHSemanticError(f"Expected an InstantiatedType,  got {ch_type!r}", location_id=location_id)
    return ch_type
