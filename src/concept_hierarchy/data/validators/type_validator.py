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
from typing import Callable

from concept_hierarchy.data.contexts.template_context import TemplateContext
from concept_hierarchy.data.parsers.type_parser import TemplateArgumentParser
from concept_hierarchy.data.type_template_variables.constraint_formula import NonStructureConstraintFormula
from concept_hierarchy.data.types.concept_hierarchy_types import (
    TYPE_VALUE_IS_INSTANCE_CHECK,
    ConceptHierarchyTemplateArgument,
    ConceptHierarchyType,
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
    TypeValue,
    VariadicTemplateVariable,
)
from concept_hierarchy.data.types.parsed_type import (
    ParsedType,
    TemplateArgumentLiteral,
    TemplateArgumentValue,
    TemplateArgumentVariadicGroup,
    TemplateArgumentWithVariadicId,
)
from concept_hierarchy.errors import CHSemanticError, CHSyntaxError, LocationId


@dataclass(frozen=True)
class TypeTemplateData:
    get_context: Callable[[], TemplateContext]
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

    def canonical_concept_name(self, concept_name: str) -> str:
        """
        The name under which ``concept_name`` is defined: itself, unless it is an alias naming a concept.

        A hierarchy without aliases -- and any validator that does not know about them -- answers with the
        name it was given, which is why this is not abstract.
        """
        return concept_name

    def resolved_type_alias(self, concept_name: str) -> InstantiatedType | None:
        """
        The type ``concept_name`` names if it is a *type* alias, else ``None``.

        A type alias is not a concept -- ``is_concept`` is false for it -- so it is substituted here, where
        a name becomes a type, rather than resolved like a concept alias. Not abstract for the same reason
        as :meth:`canonical_concept_name`.
        """
        return None

    @abstractmethod
    def is_template_variable(self, concept_name: str) -> bool:
        pass

    @abstractmethod
    def is_variadic_template_variable(self, concept_name: str) -> bool:
        pass

    @abstractmethod
    def get_available_template_variables(self) -> list[str]:
        pass

    @abstractmethod
    def get_identifier_where_types_are_defined(self) -> str:
        pass

    @abstractmethod
    def add_template_variable(
        self,
        template_variable_name: str,
        template_variable_constraint: NonStructureConstraintFormula,
        location_id: LocationId,
    ):
        pass

    @abstractmethod
    def delete_template_variable(self, template_variable_name: str, location_id: LocationId):
        pass

    @abstractmethod
    def validate_fully_instantiated_types_in_converted_value(
        self, value: ConceptHierarchyTemplateArgument, location_id: LocationId
    ):
        """
        :param value: the value whose internal fully-instantiated types are to be checked
        :param location_id:
        :return:
        :raises CHSemanticError: if validation fails for (at least) a fully-instantiated type, an error is raised
        """


def _validate_type_and_parse_to_variadic_groups(
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
    template_context = template_data.get_context()
    type_is_templated = template_context.variables != ()
    if type_is_templated != ch_type.is_templated:
        if type_is_templated:
            raise CHSemanticError(
                f'Did not define template arguments for the templated type {full_type_name}. Got "{ch_type.full_name}"',
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
    nr_template_arguments = template_context.nr_variables
    nr_variadic_template_arguments = len(template_context.variadic_variables)
    nr_parsed_template_arguments = len(parsed_template_arguments)

    if (nr_template_arguments - nr_variadic_template_arguments) > nr_parsed_template_arguments:
        parsed_t_args_str = [x.full_name for x in parsed_template_arguments]
        raise CHSemanticError(
            f"Too few arguments specified for template type {full_type_name}: {parsed_t_args_str}",
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
    for t_arg_name, parsed_t_arg_val in zip(template_context.variables, parsed_template_arguments):
        is_variadic_group = isinstance(parsed_t_arg_val, TemplateArgumentVariadicGroup)
        uses_variadic_groups |= is_variadic_group
        if (t_arg_name in template_context.variadic_variables) != is_variadic_group:
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
            count_values_without_variadic_identifier += not parsed_t_arg_val.has_variadic_identifier
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
        x for x in template_context.variables if x not in template_context.variadic_variables
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
        ch_type = _make_canonic(ch_type, template_context.variables, template_data.variadic_group_identifiers)

    # `ch_type` is now definitely in canonic form (only contains variadic groups)
    if not ch_type.is_templated:
        return ch_type
    assert len(ch_type.template_arguments) == template_context.nr_variables
    new_template_arguments = []
    for t_arg, t_arg_name in zip(ch_type.template_arguments, template_context.variables):
        is_variadic_template_argument = t_arg_name in template_context.variadic_variables
        new_location_id = location_id + [f"{ch_type.clean_name} template argument {t_arg.full_name}"]
        t_arg_valid = _validate_template_argument_value(
            t_arg, validator, new_location_id, is_variadic_template_argument
        )
        new_template_arguments.append(t_arg_valid)
    return ParsedType(
        variadic_group_identifier=ch_type.variadic_group_identifier,
        name=ch_type.name,
        has_variadic_template_expansion=ch_type.has_variadic_template_expansion,
        template_arguments=tuple(new_template_arguments),
        function_arguments=ch_type.function_arguments,
    )


def _make_canonic(
    ch_type: ParsedType, template_argument_order: tuple[str, ...], variadic_group_identifiers: dict[str, str]
) -> ParsedType:
    # verify variadicGroupIdentifiers and create the variadic groups
    variadic_groups: dict[str | None, list[TemplateArgumentWithVariadicId]] = {}
    # collect variadic group elements
    for parsed_t_arg_val in ch_type.template_arguments:
        assert isinstance(parsed_t_arg_val, TemplateArgumentWithVariadicId)
        variadic_identifier = parsed_t_arg_val.variadic_group_identifier
        if variadic_identifier not in variadic_groups:
            variadic_groups[variadic_identifier] = []
        variadic_groups[variadic_identifier].append(parsed_t_arg_val)
    # create curated template argument
    curated_template_arguments: list[TemplateArgumentValue] = []
    passed_number_of_variadic_template_arguments = 0
    count_template_arguments_parsed = 0
    # check whether the variadic group identifiers are correctly used!
    for index, t_arg in enumerate(template_argument_order):
        if t_arg in variadic_group_identifiers:
            t_arg_var_id = variadic_group_identifiers[t_arg]
            if t_arg_var_id == "":
                t_arg_var_id = None
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
                count_template_arguments_parsed += len(new_variadic_group_elements)
                curated_template_arguments.append(TemplateArgumentVariadicGroup(tuple(new_variadic_group_elements)))
            else:
                curated_template_arguments.append(TemplateArgumentVariadicGroup(()))
            passed_number_of_variadic_template_arguments += 1
        else:
            # this is clearly a non-variadic template argument;
            # all non-variadic template argument values live inside the variadic_groups[None] list entry
            index_in_empty_variadic_group = index - passed_number_of_variadic_template_arguments
            assert len(variadic_groups[None]) > index_in_empty_variadic_group
            count_template_arguments_parsed += 1
            curated_template_arguments.append(variadic_groups[None][index_in_empty_variadic_group])
    assert count_template_arguments_parsed == len(ch_type.template_arguments), (
        f"Expected to parse {len(ch_type.template_arguments)} arguments, parsed only {count_template_arguments_parsed} "
        f"arguments!"
    )
    return ParsedType(
        variadic_group_identifier=ch_type.variadic_group_identifier,
        name=ch_type.name,
        has_variadic_template_expansion=ch_type.has_variadic_template_expansion,
        template_arguments=tuple(curated_template_arguments),
        function_arguments=ch_type.function_arguments,
    )


def _validate_type(
    t: ParsedType, validator: TypeValidator, location_id: LocationId | None, inside_variadic_group: bool
) -> ParsedType:
    """
    :param t:
    :param validator:
    :param location_id:
    :param inside_variadic_group: whether this argument is used inside a variadic group or not
    :return:
    """
    if location_id is None:
        location_id = []

    is_variadic_template_variable = validator.is_variadic_template_variable(t.clean_name)
    if not is_variadic_template_variable:
        # a not (variadic template variable) should not have an expansion operator
        if t.has_variadic_template_expansion:
            raise CHSemanticError(
                f"Used the variadic expansion operator on {t.full_name!r} which is not a variadic template variable!",
                location_id=location_id,
            )
    else:
        # variadic template variables outside the variadic group should not be expanded (in canonical form)
        # variadic template variables inside the variadic group should be expanded
        if inside_variadic_group and not t.has_variadic_template_expansion:
            raise CHSemanticError(
                f"Used the variadic template variable {t.full_name!r} in a variadic group without the "
                f"expansion operator '...'! Add the operator to make this usage valid!",
                location_id=location_id,
            )
        elif not inside_variadic_group and t.has_variadic_template_expansion:
            raise CHSemanticError(
                f"Used the variadic template variable {t.clean_name!r} with the expansion operator '...' outside a "
                f"variadic group! Remove the operator to make this usage valid!",
                location_id=location_id,
            )

    # defining template arguments on a template variable is invalid!
    if validator.is_template_variable(t.clean_name):
        if t.is_templated:
            raise CHSemanticError(
                f"Can not define template arguments on a template argument in the current version of the "
                f"Concept Hierarchy!\nFound {t.full_name}\nRemove template arguments to make this usage valid.",
                location_id=location_id,
            )
        return t

    # A type alias stands for a saturated type, so it is already complete: it takes no template arguments
    # and there is nothing further to validate about its name here -- the type it names was validated when
    # the alias was resolved. It is substituted in `_convert_template_argument_to_...`.
    if validator.resolved_type_alias(t.clean_name) is not None:
        if t.is_templated:
            raise CHSemanticError(
                f"Can not define template arguments on {t.clean_name!r}: it is an alias of a type that is "
                f"already fully applied.\nFound {t.full_name}\nRemove the template arguments to make this "
                f"usage valid.",
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
    return _validate_type_and_parse_to_variadic_groups(t, validator, location_id, full_type_name, type_template_data)


def _validate_template_argument_value(
    t: TemplateArgumentValue,
    validator: TypeValidator,
    location_id: LocationId | None,
    is_argument_for_variadic_template_parameter: bool,
) -> TemplateArgumentValue:
    """
    This is always called for parsing a template argument (from the var-groups canonical form) in a type application.
    Also converts variadic group id type-application-syntax to variadic group ids.
    """
    if location_id is None:
        location_id = []

    # check no variadic identifier in provided value: because this must be called from the var-groups canonical form
    if isinstance(t, TemplateArgumentWithVariadicId) and t.has_variadic_identifier:
        raise CHSyntaxError(
            f"Variadic identifiers should not have been used in this context! Found {t.variadic_group_identifier} at "
            f"{t.full_name}!",
            location_id=location_id,
        )
    # check correct supplied argument for variadic template parameter
    if is_argument_for_variadic_template_parameter:
        if isinstance(t, TemplateArgumentLiteral):
            raise CHSemanticError(
                f"The literal value {t.full_name!r} can not be used as a variadic template argument value!",
                location_id=location_id,
            )
        elif isinstance(t, ParsedType) and not validator.is_variadic_template_variable(t.clean_name):
            # this should be a variadic template argument or a variadic group!
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
    elif isinstance(t, TemplateArgumentVariadicGroup):
        raise CHSemanticError(
            f"The variadic group {t.full_name!r} can not be used as a non-variadic template argument value!",
            location_id=location_id,
        )

    if isinstance(t, TemplateArgumentLiteral):
        return t
    if isinstance(t, ParsedType):
        return _validate_type(t, validator, location_id, inside_variadic_group=False)
    assert isinstance(t, TemplateArgumentVariadicGroup)
    validated_variadic_group = []
    for elem_index, group_elem in enumerate(t.variadic_group):
        new_location_id = location_id + [f"group element {elem_index}"]
        if isinstance(group_elem, TemplateArgumentLiteral):
            validated_variadic_group.append(group_elem)
        else:
            assert isinstance(group_elem, ParsedType)
            # check that the value is not a non-expanded variadic template argument
            validated_elem = _validate_type(group_elem, validator, new_location_id, inside_variadic_group=True)
            validated_variadic_group.append(validated_elem)
    return TemplateArgumentVariadicGroup(tuple(validated_variadic_group))


def _convert_items(
    items: tuple[TemplateArgumentValue, ...], validator: TypeValidator
) -> tuple[tuple[ConceptHierarchyTemplateArgument, ...], bool]:
    """
    :param items: contains the items to respectively convert to ConceptHierarchyTemplateArgument values
    :param validator: the type validator
    :return: the converted values, whether any of the value depends on template variables
    """
    converted_items: list[ConceptHierarchyTemplateArgument] = []
    has_template_dependent_items = False
    for item in items:
        converted_item = _convert_template_argument_to_concept_hierarchy_template_argument(item, validator)
        if isinstance(converted_item, TemplateDependent):
            has_template_dependent_items = True
        else:
            assert isinstance(converted_item, Instantiated)
        converted_items.append(converted_item)
    return tuple(converted_items), has_template_dependent_items


def _convert_template_argument_to_concept_hierarchy_template_argument(
    t_arg: TemplateArgumentValue, validator: TypeValidator
) -> ConceptHierarchyTemplateArgument:
    """
    :param t_arg: is the value to be converted
    :param validator: is the type validator
    :return: the converted value
    """
    if isinstance(t_arg, TemplateArgumentLiteral):
        return LiteralValue(t_arg.clean_name, t_arg.literal_type)
    if isinstance(t_arg, ParsedType):
        # a type alias names a whole type, not a concept: substitute what it names
        alias_type = validator.resolved_type_alias(t_arg.clean_name)
        if alias_type is not None:
            return alias_type
        if not validator.is_concept(t_arg.clean_name):
            assert validator.is_template_variable(t_arg.clean_name)
            if validator.is_variadic_template_variable(t_arg.clean_name):
                if t_arg.has_variadic_template_expansion:
                    return ExpandedVariadicTemplateVariable(
                        t_arg.clean_name, validator.get_identifier_where_types_are_defined()
                    )
                else:
                    return VariadicTemplateVariable(
                        t_arg.clean_name, validator.get_identifier_where_types_are_defined()
                    )
            else:
                assert not t_arg.has_variadic_template_expansion
                return NonVariadicTemplateVariable(t_arg.clean_name, validator.get_identifier_where_types_are_defined())
        assert not t_arg.has_variadic_template_expansion
        # The user may have written an alias of the concept; the type is built from the canonical name, so
        # that everything downstream compares and indexes one name per concept.
        clean_name = validator.canonical_concept_name(t_arg.clean_name)
        # check if all the template arguments are instantiated or not
        if not t_arg.is_templated:
            return InstantiatedType(clean_name, ())
        converted_template_arguments, has_template_dependent_template_arguments = _convert_items(
            t_arg.template_arguments, validator
        )
        if has_template_dependent_template_arguments:
            return TemplateDependentType(clean_name, converted_template_arguments)
        return InstantiatedType(clean_name, converted_template_arguments)
    assert isinstance(t_arg, TemplateArgumentVariadicGroup)
    converted_group_elements, has_template_dependent_group_elements = _convert_items(t_arg.variadic_group, validator)
    assert all(isinstance(x, (ConceptHierarchyType, LiteralValue, TemplateVariable)) for x in converted_group_elements)
    if has_template_dependent_group_elements:
        return TemplateDependentVariadicGroup(t_arg.clean_name, converted_group_elements)
    return InstantiatedVariadicGroup(t_arg.clean_name, converted_group_elements)


def convert_template_argument_to_concept_hierarchy_template_argument(
    t_arg: str | list[str], validator: TypeValidator, location_id: LocationId, is_variadic_argument_value: bool
) -> ConceptHierarchyTemplateArgument:
    """
    This function is called with either a type (never a variadic argument) or a value from the `substitution` data.

    :param t_arg: is the value to be converted
    :param validator: is the type validator
    :param location_id: the location where the type was used
    :param is_variadic_argument_value: whether the value is for a variadic argument or not
    :return: the converted value
    """
    # Start validation of the syntax of the substitution value:
    #  1) Convert json object to TemplateArgumentValue
    parsed_t_arg_value = TemplateArgumentParser(t_arg, location_id).parse()
    #  2) Validate nr. template args, create variadic groups from var.ids., don't check template constraints
    validated_t_arg_value = _validate_template_argument_value(
        parsed_t_arg_value, validator, location_id, is_variadic_argument_value
    )
    # 3) Check the semantic of the value
    res = _convert_template_argument_to_concept_hierarchy_template_argument(validated_t_arg_value, validator)
    # 4) Check the instantiation constraints of every type in the value, not only of the outermost one:
    #    a nested application has to satisfy the constraints of *its* concept too. `Any<Box<String>>` is
    #    the case this catches -- `Box<String>` is a perfectly good ValueDomain as far as `Any` is
    #    concerned, so only checking `Box` against its own `T : Number` rejects it.
    #    Done here, over the finished value, rather than inside the (bottom-up, recursive) conversion:
    #    one flat walk visits each type once, where validating per construction would re-check every
    #    shared subtree once per level it sits under.
    already_validated: set[str] = set()
    for sub_value in res.iterate_subtypes(do_not_expand_instantiated_types=False):
        if sub_value.full_name in already_validated:
            continue
        already_validated.add(sub_value.full_name)
        validator.validate_fully_instantiated_types_in_converted_value(sub_value, location_id)
    return res


def _parse_convert_no_check(
    type_def: str, validator: TypeValidator, location_id: LocationId
) -> ConceptHierarchyTemplateArgument:
    if type_def.strip() == "":
        raise CHSyntaxError("The given type is empty!", location_id=location_id)
    # check the syntax and semantics of the type
    return convert_template_argument_to_concept_hierarchy_template_argument(type_def, validator, location_id, False)


def parse_convert_type(type_def: str, validator: TypeValidator, location_id: LocationId) -> InstantiatedType:
    """Parse and convert a type, checking the instantiation constraints of every type it contains."""
    ch_type = _parse_convert_no_check(type_def, validator, location_id)
    if not isinstance(ch_type, InstantiatedType):
        raise CHSemanticError(f"Expected an InstantiatedType, got {ch_type!r}", location_id=location_id)
    return ch_type


def parse_convert_type_in_template_context(
    type_def: str, validator: TypeValidator, location_id: LocationId
) -> TypeValue:
    """
    As :func:`parse_convert_type`, but the result may still depend on the template variables in scope.

    The instantiation constraints of every type it contains are checked here too; for a template-dependent
    one that means checking that *no* substitution could satisfy them, and that the constraints it implies
    are compatible with the ones already on those variables.
    """
    ch_type = _parse_convert_no_check(type_def, validator, location_id)
    if not isinstance(ch_type, TYPE_VALUE_IS_INSTANCE_CHECK):
        raise CHSemanticError(
            f"Expected an InstantiatedType, a TemplateDependentType, a NonVariadicTemplateVariable or a "
            f"VariadicTemplateVariable, but got {ch_type!r}",
            location_id=location_id,
        )
    return ch_type
