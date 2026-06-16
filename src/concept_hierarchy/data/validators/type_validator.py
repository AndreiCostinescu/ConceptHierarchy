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
)
from concept_hierarchy.errors import LocationId


class TypeValidator(ABC):
    @abstractmethod
    def validate_type(self, ch_type: ParsedType, location_id: LocationId):
        pass

    @abstractmethod
    def make_canonic(self, ch_type: ParsedType) -> ParsedType:
        pass


def validate_type(t: ParsedType, validator: TypeValidator, location_id: LocationId | None) -> ParsedType:
    if location_id is None:
        location_id = []

    validated_template_arguments: list[TemplateArgumentValue] = []
    new_t_args: list[str | tuple[str, ...]] = []
    for t_index, t_arg in enumerate(t.template_argument_values):
        new_location_id = location_id + [f"{t.full_name} template argument {t_index}"]
        res = validate_template_argument_value(t_arg, validator, new_location_id)
        validated_template_arguments.append(res)
        if isinstance(t_arg, TemplateArgumentVariadicGroup):
            new_t_args.append(tuple(x.full_name for x in t_arg.variadic_group))
        else:
            new_t_args.append(t_arg.full_name)

    new_t = ParsedType(
        t.name,
        t.variadic_group_identifier,
        t.has_variadic_template_expansion,
        tuple(new_t_args),
        t.func_args,
        tuple(validated_template_arguments),
        t.sub_func_types,
    )
    validator.validate_type(new_t, location_id)
    return validator.make_canonic(new_t)


def validate_template_argument_value(
    t: TemplateArgumentValue, validator: TypeValidator, location_id: LocationId | None
) -> TemplateArgumentValue:
    if location_id is None:
        location_id = []
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
            validated_variadic_group.append(validate_type(group_elem, validator, new_location_id))
    return TemplateArgumentVariadicGroup(tuple(validated_variadic_group))
