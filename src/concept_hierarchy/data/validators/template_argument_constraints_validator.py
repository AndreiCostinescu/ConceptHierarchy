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
from enum import Enum
from typing import Callable

from concept_hierarchy.data.template_argument_constraints.constraint_formula import (
    LiteralValueConstraintFormula,
    NonTypeTemplateConstraintFormula,
    TemplateConstraintAbstractAscendants,
    TemplateConstraintAbstractDescendants,
    TemplateConstraintAnd,
    TemplateConstraintAscendants,
    TemplateConstraintDescendants,
    TemplateConstraintFormula,
    TemplateConstraintHierarchyOperator,
    TemplateConstraintNot,
    TemplateConstraintOr,
    TemplateConstraintSelf,
)
from concept_hierarchy.data.types.parsed_type import (
    ParsedType,
    TemplateArgumentLiteral,
    TemplateArgumentValue,
    TemplateArgumentVariadicGroup,
)
from concept_hierarchy.data.utils import record
from concept_hierarchy.errors import CHSemanticError, ConceptHierarchyError, LocationId
from concept_hierarchy.utils import Reference, is_integer, is_number


class HierarchyCheckType(Enum):
    DESCENDANTS_OF = (0,)
    ABSTRACT_DESCENDANTS_OF = (1,)
    ASCENDANTS_OF = (2,)
    ABSTRACT_ASCENDANTS_OF = (3,)
    SELF = 4


class TemplateConstraintArgumentValidator(ABC):
    @abstractmethod
    def type_check(self, a: ParsedType, b: ParsedType, check_type: HierarchyCheckType) -> bool:
        pass

    @abstractmethod
    def create_substitution_for(self, parent_type: ParsedType, sub_type: ParsedType) -> list[TemplateArgumentValue]:
        """
        (_, t_arg_value_clean, _, t_args_of_t_arg) = process_value_domain(template_argument_value)[0]
        t_arg_vd = ValueDomain.all_value_domains[t_arg_value_clean]
        literal_vd = ValueDomain.all_value_domains[self.literal]

        t_arg_subst_dict = t_arg_vd.create_template_substitution(t_args_of_t_arg)
        t_arg_vd_all_substs = t_arg_vd.template_type_substitution
        substituted_t_args_of_literal = []
        for literal_t_arg in literal_vd.template_order:
            if (self.literal, literal_t_arg) not in t_arg_vd_all_substs:
                assert t_arg_value_clean == self.literal, "t_arg_value_clean = {}, self.literal = {}".format(
                    t_arg_value_clean, self.literal
                )
                substitution_type = literal_t_arg
            else:
                substitution_type = t_arg_vd_all_substs[self.literal, literal_t_arg]
            substituted_t_args_of_literal.append(ValueDomain.replace_template(substitution_type, t_arg_subst_dict))
        """
        pass


def validate_template_argument(
    formula: TemplateConstraintFormula,
    template_argument_value: TemplateArgumentValue,
    context: TemplateConstraintArgumentValidator,
    location_id: LocationId = None,
    *,
    collect_all: bool = False,
) -> list[ConceptHierarchyError]:
    if location_id is None:
        location_id = []
    errors: list[ConceptHierarchyError] = []
    try:
        check_location_id = location_id + [f"{formula!r} <-> {template_argument_value.full_name}"]
        _delegate_constraint_check(formula, template_argument_value, context, check_location_id, errors, collect_all)
    except StopIteration:
        pass
    return errors


def _delegate_constraint_check(
    formula: TemplateConstraintFormula,
    template_argument_value: TemplateArgumentValue,
    context: TemplateConstraintArgumentValidator,
    location_id: LocationId,
    errors: list[ConceptHierarchyError],
    collect_all: bool = False,
):
    match formula:
        case TemplateConstraintAnd():
            _validate_and(formula, template_argument_value, context, location_id, errors, collect_all)
        case TemplateConstraintOr():
            _validate_or(formula, template_argument_value, context, location_id, errors, collect_all)
        case TemplateConstraintNot():
            _validate_not(formula, template_argument_value, context, location_id, errors, collect_all)
        case TemplateConstraintHierarchyOperator():
            _validate_type(formula, template_argument_value, context, location_id, errors, collect_all)
        case LiteralValueConstraintFormula() | NonTypeTemplateConstraintFormula():
            _validate_literal(formula, template_argument_value, location_id, errors, collect_all)
        case _:
            raise ValueError(f"Unknown formula type: {type(formula)!r}")


def _validate_and(
    formula: TemplateConstraintAnd,
    template_argument_value: TemplateArgumentValue,
    context: TemplateConstraintArgumentValidator,
    location_id: LocationId,
    errors: list[ConceptHierarchyError],
    collect_all: bool,
):
    success = True
    and_errors: list[list[ConceptHierarchyError]] = []
    for sub_f in formula.sub_formulae:
        sub_errors = []
        try:
            new_location_id = location_id + [f"{sub_f!r} <-> {template_argument_value.full_name}"]
            _delegate_constraint_check(
                sub_f, template_argument_value, context, new_location_id, sub_errors, collect_all
            )
        except StopIteration:
            pass
        if sub_errors:
            success = False
            and_errors.append(sub_errors)
            if not collect_all:
                break
    if not success:
        err = CHSemanticError(
            f"{formula!r} not satisfied!",
            location_id=location_id,
        )
        for and_error in and_errors:
            err.causes.extend(and_error)
        record(errors, collect_all, err)


def _validate_or(
    formula: TemplateConstraintOr,
    template_argument_value: TemplateArgumentValue,
    context: TemplateConstraintArgumentValidator,
    location_id: LocationId,
    errors: list[ConceptHierarchyError],
    collect_all: bool,
):
    success = False
    or_errors: list[list[ConceptHierarchyError]] = []
    for sub_f in formula.sub_formulae:
        sub_errors = []
        try:
            new_location_id = location_id + [f"{sub_f!r} <-> {template_argument_value.full_name}"]
            _delegate_constraint_check(
                sub_f, template_argument_value, context, new_location_id, sub_errors, collect_all
            )
        except StopIteration:
            pass
        if sub_errors:
            or_errors.append(sub_errors)
        else:
            success = True
            if not collect_all:
                break
    if not success:
        err = CHSemanticError(
            f"{formula!r} not satisfied!",
            location_id=location_id,
        )
        for or_error in or_errors:
            err.causes.extend(or_error)
        record(errors, collect_all, err)


def _validate_not(
    formula: TemplateConstraintNot,
    template_argument_value: TemplateArgumentValue,
    context: TemplateConstraintArgumentValidator,
    location_id: LocationId,
    errors: list[ConceptHierarchyError],
    collect_all: bool,
):
    sub_errors = []
    try:
        new_location_id = location_id + [f"{formula.sub_formula} <-> {template_argument_value.full_name}"]
        _delegate_constraint_check(
            formula.sub_formula, template_argument_value, context, new_location_id, sub_errors, collect_all
        )
    except StopIteration:
        pass
    if not sub_errors:
        err = CHSemanticError(f"Not formula {formula!r} was satisfied!", location_id=location_id)
        record(errors, collect_all, err)


def _create_iteration_data(
    template_argument_value: TemplateArgumentValue,
) -> tuple[tuple[ParsedType | TemplateArgumentLiteral, ...], bool]:
    if not isinstance(template_argument_value, TemplateArgumentVariadicGroup):
        is_variadic = False
        assert isinstance(template_argument_value, (ParsedType, TemplateArgumentLiteral))
        to_check: tuple[ParsedType | TemplateArgumentLiteral, ...] = (template_argument_value,)
    else:
        is_variadic = True
        to_check = template_argument_value.variadic_group
    return to_check, is_variadic


def _validate_type(
    formula: TemplateConstraintHierarchyOperator,
    template_argument_value: TemplateArgumentValue,
    context: TemplateConstraintArgumentValidator,
    location_id: LocationId,
    errors: list[ConceptHierarchyError],
    collect_all: bool,
):
    match formula:
        case TemplateConstraintSelf():
            hierarchy_check_type = HierarchyCheckType.SELF
        case TemplateConstraintDescendants():
            hierarchy_check_type = HierarchyCheckType.DESCENDANTS_OF
        case TemplateConstraintAbstractDescendants():
            hierarchy_check_type = HierarchyCheckType.ABSTRACT_DESCENDANTS_OF
        case TemplateConstraintAscendants():
            hierarchy_check_type = HierarchyCheckType.ASCENDANTS_OF
        case TemplateConstraintAbstractAscendants():
            hierarchy_check_type = HierarchyCheckType.ABSTRACT_ASCENDANTS_OF
        case _:
            raise ValueError(f"Unknown formula type: {type(formula)!r}")
    to_check, is_variadic = _create_iteration_data(template_argument_value)
    for index, t_arg in enumerate(to_check):
        if is_variadic:
            arg_str = f"Argument {index + 1} of {template_argument_value.full_name!r}"
        else:
            arg_str = f"{t_arg.full_name!r}"
        if isinstance(t_arg, TemplateArgumentLiteral):
            err = CHSemanticError(
                f"{arg_str} is a literal value, which can not match the type constraint {formula!r}",
                location_id=location_id + [f"{formula} <-> {t_arg.full_name}"],
            )
            record(errors, collect_all, err)
        assert isinstance(t_arg, ParsedType)
        if not context.type_check(t_arg, formula.literal_type, hierarchy_check_type):
            err = CHSemanticError(
                f"{arg_str} does not satisfy the constraint {formula!r}",
                location_id=location_id + [f"{formula} <-> {t_arg.full_name}"],
            )
            record(errors, collect_all, err)
            continue
        if not formula.has_specification_of_template_constraints:
            continue
        # Check the template argument constraints of the literal_type
        # It is not the template arguments of this value (template_argument_value) that must be checked,
        #  but the substitution value of the template arguments of literal_type that must match the constraints!
        literal_type_substituted_template_args: list[TemplateArgumentValue] = context.create_substitution_for(
            formula.literal_type, t_arg
        )
        if len(literal_type_substituted_template_args) != len(formula.literal_template_formulae):
            raise RuntimeError(
                f"Mismatch between the number of substituted template argument values "
                f"{literal_type_substituted_template_args!r} and the template argument's constraints "
                f"{len(formula.literal_template_formulae)}"
            )
        all_t_arg_errors: list[list[ConceptHierarchyError]] = []
        for t_arg_constraint, t_arg_value in zip(
            formula.literal_template_formulae, literal_type_substituted_template_args
        ):
            t_arg_errors: list[ConceptHierarchyError] = []
            try:
                new_location_id = location_id + [f"{t_arg_constraint!r} <-> {t_arg_value.full_name}"]
                _delegate_constraint_check(
                    t_arg_constraint, t_arg_value, context, new_location_id, t_arg_errors, collect_all
                )
            except StopIteration:
                pass
            if t_arg_errors:
                all_t_arg_errors.append(t_arg_errors)
                if not collect_all:
                    break
        if all_t_arg_errors:
            err = CHSemanticError("", location_id=location_id)
            for t_arg_error in all_t_arg_errors:
                err.causes.extend(t_arg_error)
            record(errors, collect_all, err)


def _check_literal_type(formula: NonTypeTemplateConstraintFormula, t_arg: TemplateArgumentLiteral) -> bool:
    if formula.constraint_type == "int":
        return is_integer(t_arg.literal_value)
    elif formula.constraint_type == "float":
        return is_number(t_arg.literal_value)
    elif formula.constraint_type == "bool":
        return t_arg.literal_value in ["true", "bool"]
    elif formula.constraint_type == "string":
        return t_arg.literal_value.startswith('"') and t_arg.literal_value.endswith('"')
    else:
        raise RuntimeError("Unknown constraint type: {}".format(formula.constraint_type))


def _check_literal_value(formula: LiteralValueConstraintFormula, t_arg: TemplateArgumentLiteral) -> bool:
    if formula.constraint_type == "int":
        ref = Reference()
        return is_integer(t_arg.literal_value, ref) and ref.ref == formula.value
    elif formula.constraint_type == "float":
        ref = Reference()
        return is_number(t_arg.literal_value, ref) and ref.ref == formula.value
    elif formula.constraint_type == "bool":
        # t_arg must be exactly "true" or "false"
        return t_arg.literal_value == formula.raw_value
    elif formula.constraint_type == "string":
        return t_arg.literal_value == formula.raw_value
    else:
        raise RuntimeError("Unknown constraint type: {}".format(formula.constraint_type))


def _validate_literal(
    formula: NonTypeTemplateConstraintFormula | LiteralValueConstraintFormula,
    template_argument_value: TemplateArgumentValue,
    location_id: LocationId,
    errors: list[ConceptHierarchyError],
    collect_all: bool,
):
    f_check: Callable[
        [NonTypeTemplateConstraintFormula | LiteralValueConstraintFormula, TemplateArgumentLiteral], bool
    ] = _check_literal_type if isinstance(formula, NonTypeTemplateConstraintFormula) else _check_literal_value
    to_check, is_variadic = _create_iteration_data(template_argument_value)
    for index, t_arg in enumerate(to_check):
        if is_variadic:
            arg_str = f"Argument {index + 1} of {template_argument_value.full_name!r}"
        else:
            arg_str = f"{t_arg.full_name!r}"
        if isinstance(t_arg, ParsedType):
            err = CHSemanticError(
                f"{arg_str} is a type value, which can not match the non-type constraint {formula!r}",
                location_id=location_id + [f"{formula} <-> {t_arg.full_name}"],
            )
            record(errors, collect_all, err)
        assert isinstance(t_arg, TemplateArgumentLiteral)
        if not f_check(formula, t_arg):
            err = CHSemanticError(
                f"{arg_str} does not satisfy the constraint {formula!r}",
                location_id=location_id + [f"{formula} <-> {t_arg.full_name}"],
            )
            record(errors, collect_all, err)
