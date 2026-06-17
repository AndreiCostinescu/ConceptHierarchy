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

from concept_hierarchy.data.types.parsed_type import (
    ParsedType,
    TemplateArgumentLiteral,
    TemplateArgumentValue,
    TemplateArgumentVariadicGroup,
    TemplateArgumentWithVariadicId,
)
from concept_hierarchy.errors import CHSemanticError, CHSyntaxError, LocationId


class TypeValidator(ABC):
    @abstractmethod
    def validate_type_and_parse_to_variadic_groups(self, ch_type: ParsedType, location_id: LocationId) -> ParsedType:
        """
        Validates:
         - that the number of template arguments are correctly defined,
         - that the variadic identifiers are correctly used in template argument values,
         - transforms the representation from using variadic identifiers to using only variadic groups
         - verifies that the constraints on template values are satisfied
        """
        pass

    @abstractmethod
    def is_template_variable(self, ch_type: ParsedType) -> bool:
        pass

    @abstractmethod
    def is_variadic_template_variable(self, ch_type: ParsedType) -> bool:
        pass


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
    if validator.is_template_variable(t):
        if validator.is_variadic_template_variable(t):
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

    # Bring to canonic form:
    #   Use only variadic groups (no more ParsedTypes with variadic identifiers)
    #   Make sure that mixed syntax with the empty variadic identifier is NOT used
    #   Make sure that there are a correct amount of template arguments specified in the template-instantiation
    # And validate sub-template arguments
    #   this has to happen from inside the validator because only it knows
    #   whether the template argument type is a variadic argument or not
    return validator.validate_type_and_parse_to_variadic_groups(t, location_id)


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
            if not validator.is_variadic_template_variable(t):
                if validator.is_template_variable(t):
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
