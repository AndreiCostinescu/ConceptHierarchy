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
from typing import Callable

from concept_hierarchy.data.contexts.template_context import TemplateContext
from concept_hierarchy.data.template_argument_constraints.constraint_formula import (
    ConstraintGroup,
    Empty,
    HierarchyCheckType,
    LiteralValueConstraintFormula,
    NonStructureConstraintFormula,
    NonTypeTemplateConstraintFormula,
    StructureConjunction,
    StructureConstraintFormula,
    StructureDisjunction,
    StructureNegation,
    TemplateConstraintAnd,
    TemplateConstraintFormula,
    TemplateConstraintFormulaValidator,
    TemplateConstraintHierarchyOperator,
    TemplateConstraintNot,
    TemplateConstraintOr,
    TemplateConstraintSelf,
    TypeTemplateConstraintFormula,
    Unconstrained,
)
from concept_hierarchy.data.types.concept_hierarchy_types import (
    ConceptHierarchyTemplateArgument,
    ConceptHierarchyType,
    ConceptHierarchyVariadicGroup,
    ExpandedVariadicTemplateVariable,
    LiteralValue,
    TemplateVariable,
    VariadicArgument,
)
from concept_hierarchy.data.types.parsed_type import TemplateArgumentLiteral
from concept_hierarchy.errors import CHSemanticError, ConceptHierarchyError, LocationId
from concept_hierarchy.utils import Reference, is_integer, is_number


class TemplateConstraintArgumentValidator(TemplateConstraintFormulaValidator, ABC):
    @abstractmethod
    def concept_check(self, a_type: ConceptHierarchyType, b_name: str, check_type: HierarchyCheckType) -> bool:
        pass

    @abstractmethod
    def create_substitution_for(
        self, parent_type_name: str, sub_type: ConceptHierarchyType, location_id: LocationId
    ) -> tuple[ConceptHierarchyTemplateArgument, ...] | None:
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

    @abstractmethod
    def get_template_argument_names_of(self, concept_name: str) -> tuple[str, ...]:
        pass

    @abstractmethod
    def get_constraint_formula_of(self, name: str) -> StructureConstraintFormula | None:
        pass


def create_formula_from_substituted_value_for(
    value: ConceptHierarchyTemplateArgument, validator: TemplateConstraintFormulaValidator, location_id: LocationId
) -> NonStructureConstraintFormula:
    if isinstance(value, LiteralValue):
        return LiteralValueConstraintFormula(value.literal_type, value.full_name, location_id)
    elif isinstance(value, ConceptHierarchyType):
        template_constraints = []
        for t_arg in value.template_arguments:
            template_constraints.append(create_formula_from_substituted_value_for(t_arg, validator, location_id))
        return TemplateConstraintSelf(value.clean_name, tuple(template_constraints), validator, location_id)
    elif isinstance(value, VariadicArgument):
        raise RuntimeError("[Feature-Request] Did not implement support for using variadic constraints!")
    else:
        assert isinstance(value, TemplateVariable)
        return TemplateConstraintSelf(value.clean_name, (), validator, location_id)


def substitute_template_variables_in_formula(
    formula: TemplateConstraintFormula,
    substitution: dict[str, ConceptHierarchyTemplateArgument | None],
    validator: TemplateConstraintFormulaValidator,
    location_id: LocationId,
) -> TemplateConstraintFormula:
    match formula:
        case NonTypeTemplateConstraintFormula():
            return formula
        case TemplateConstraintHierarchyOperator():
            formula_is_template_variable = formula.literal in substitution
            subst_literal: str  # will be set in the complex if-statement below
            subst_t_args: list[NonStructureConstraintFormula] = []
            if not formula_is_template_variable:
                subst_literal = formula.literal
                for item in formula.literal_template_formulae:
                    res = substitute_template_variables_in_formula(item, substitution, validator, location_id)
                    if not isinstance(res, NonStructureConstraintFormula):
                        raise RuntimeError(f"Expected a NonStructureConstraintFormula, got {res!r}!")
                    subst_t_args.append(res)
            elif substitution[formula.literal] is None:
                return formula
            else:
                # has substitution: a template variable, a partially instantiated value, a fully instantiated value
                subst_value = substitution[formula.literal]
                if isinstance(subst_value, TemplateVariable):
                    subst_literal = subst_value.clean_name
                    # don't change subst_t_args, because a template variable does not have template arguments
                elif isinstance(subst_value, LiteralValue):
                    return LiteralValueConstraintFormula(subst_value.literal_type, subst_value.full_name, location_id)
                elif isinstance(subst_value, VariadicArgument):
                    raise RuntimeError("[Feature-Request] Did not implement support for using variadic constraints!")
                else:
                    assert isinstance(subst_value, ConceptHierarchyType)
                    subst_formula = create_formula_from_substituted_value_for(subst_value, validator, location_id)
                    assert isinstance(subst_formula, TemplateConstraintHierarchyOperator)
                    subst_formula = subst_formula.change_hierarchy_operator(
                        formula.hierarchy_op, validator, location_id
                    )
                    return substitute_template_variables_in_formula(subst_formula, substitution, validator, location_id)
            return formula.create_new_same_op(subst_literal, tuple(subst_t_args), validator, location_id)
        case TemplateConstraintAnd():
            sub_formulae: list[TypeTemplateConstraintFormula] = []
            for sub_f in formula.sub_formulae:
                res = substitute_template_variables_in_formula(sub_f, substitution, validator, location_id)
                if not isinstance(res, TypeTemplateConstraintFormula):
                    raise RuntimeError(f"Expected a TypeTemplateConstraintFormula, got {res!r}")
                sub_formulae.append(res)
            return TemplateConstraintAnd(location_id, tuple(sub_formulae))
        case TemplateConstraintOr():
            sub_formulae: list[TypeTemplateConstraintFormula] = []
            for sub_f in formula.sub_formulae:
                res = substitute_template_variables_in_formula(sub_f, substitution, validator, location_id)
                if not isinstance(res, TypeTemplateConstraintFormula):
                    raise RuntimeError(f"Expected a TypeTemplateConstraintFormula, got {res!r}")
                sub_formulae.append(res)
            return TemplateConstraintOr(location_id, tuple(sub_formulae))
        case TemplateConstraintNot():
            res = substitute_template_variables_in_formula(formula.sub_formula, substitution, validator, location_id)
            if not isinstance(res, TypeTemplateConstraintFormula):
                raise RuntimeError(f"Expected a TypeTemplateConstraintFormula, got {res!r}")
            return TemplateConstraintNot(location_id, res)
        case NonStructureConstraintFormula():
            # parse ``Unconstrained`` and ``Empty``
            return formula
        case ConstraintGroup():
            substituted_group: list[NonStructureConstraintFormula] = []
            for item in formula.group_constraints:
                res = substitute_template_variables_in_formula(item, substitution, validator, location_id)
                if not isinstance(res, NonStructureConstraintFormula):
                    raise RuntimeError(f"Expected a NonStructureConstraintFormula, got {res!r}")
                substituted_group.append(res)
            return ConstraintGroup(location_id, tuple(substituted_group))
        case StructureConjunction():
            sub_structures: list[StructureConstraintFormula] = []
            for sub_s in formula.structure_constraints:
                res = substitute_template_variables_in_formula(sub_s, substitution, validator, location_id)
                if not isinstance(res, StructureConstraintFormula):
                    raise RuntimeError(f"Expected a StructureConstraintFormula, got {res!r}")
                sub_structures.append(res)
            return StructureConjunction(location_id, tuple(sub_structures))
        case StructureDisjunction():
            sub_structures: list[StructureConstraintFormula] = []
            for sub_s in formula.structure_constraints:
                res = substitute_template_variables_in_formula(sub_s, substitution, validator, location_id)
                if not isinstance(res, StructureConstraintFormula):
                    raise RuntimeError(f"Expected a StructureConstraintFormula, got {res!r}")
                sub_structures.append(res)
            return StructureDisjunction(location_id, tuple(sub_structures))
        case StructureNegation():
            res = substitute_template_variables_in_formula(
                formula.structure_constraint, substitution, validator, location_id
            )
            if not isinstance(res, StructureConstraintFormula):
                raise RuntimeError(f"Expected a StructureConstraintFormula, got {res!r}")
            return StructureNegation(location_id, res)
        case _:
            raise RuntimeError(f"Unknown constraint type: {formula!r}")


def validate_template_argument_value_against_constraint(
    formula: NonStructureConstraintFormula,
    template_argument_value: ConceptHierarchyTemplateArgument,
    template_context: TemplateContext,
    validator: TemplateConstraintArgumentValidator,
    concept_template_argument_instantiation: dict[str, ConceptHierarchyTemplateArgument],
    location_id: LocationId = None,
    *,
    collect_all_errors: bool = False,
) -> list[ConceptHierarchyError]:
    if location_id is None:
        location_id = []
    assert template_context.is_unconstrained
    errors: list[ConceptHierarchyError] = []
    check_location_id = location_id + [f"{formula!r} <-> {template_argument_value.full_name}"]
    _delegate_constraint_check(
        formula,
        template_argument_value,
        template_context,
        validator,
        concept_template_argument_instantiation,
        check_location_id,
        errors,
        collect_all_errors,
    )
    return errors


def validate_complete_instantiation_of_concept(
    concept_name: str,
    complete_instantiation: tuple[ConceptHierarchyTemplateArgument, ...],
    template_context: TemplateContext,
    validator: TemplateConstraintArgumentValidator,
    location_id: LocationId = None,
) -> list[ConceptHierarchyError]:
    formula = validator.get_constraint_formula_of(concept_name)
    if (formula is None) != (complete_instantiation == ()):
        if formula is None:
            raise RuntimeError(
                f"No template variables but provided a instantiation template arguments {complete_instantiation}"
            )
        raise RuntimeError(f"No instantiation arguments provided for formula {formula!r}")
    if formula is None:
        return []
    return _validate_complete_instantiation_of_type(
        concept_name, formula, complete_instantiation, template_context, validator, location_id
    )


def _validate_complete_instantiation_of_type(
    concept_name: str,
    formula: StructureConstraintFormula,
    complete_instantiation: tuple[ConceptHierarchyTemplateArgument, ...],
    template_context: TemplateContext,
    validator: TemplateConstraintArgumentValidator,
    location_id: LocationId = None,
) -> list[ConceptHierarchyError]:
    assert template_context.is_unconstrained
    if isinstance(formula, ConstraintGroup):
        if len(formula.group_constraints) != len(complete_instantiation):
            raise RuntimeError(
                f"Incomplete/wrong substitution for {formula}: expected {len(formula.group_constraints)} variable "
                f"values, got {complete_instantiation!r}"
            )
        assert len(formula.group_constraints) > 0
        # assert that all values share the same template context object (modifying one will modify all of them)!
        sub_template_contexts: list[TemplateContext] = []
        total_errors = []
        concept_template_argument_names = validator.get_template_argument_names_of(concept_name)
        concept_template_argument_instantiation: dict[str, ConceptHierarchyTemplateArgument] = {
            key: value for key, value in zip(concept_template_argument_names, complete_instantiation)
        }
        for t_arg_index, (constraint, value) in enumerate(zip(formula.group_constraints, complete_instantiation)):
            new_location_id = location_id + [f"{constraint!r} <-> {value}"]
            sub_template_context = template_context.create_unconstrained_context(new_location_id)
            errors = validate_template_argument_value_against_constraint(
                constraint,
                value,
                sub_template_context,
                validator,
                concept_template_argument_instantiation,
                new_location_id,
                collect_all_errors=True,
            )
            if errors:
                # if the first substitution already fails, the next can't possibly validate them
                total_errors.extend(errors)
            else:
                sub_template_contexts.append(sub_template_context)
        # if there are errors, do not merge the sub-template context with the main one!
        if not total_errors:
            # merge the template contexts, only if there are no errors!
            template_context.merge_constraints_and(sub_template_contexts, location_id)
        return total_errors

    if isinstance(formula, StructureConjunction):
        total_errors, sub_template_contexts = _iterate_sub_structure_formulae(
            concept_name, formula, complete_instantiation, template_context, validator, location_id
        )
        if not total_errors:
            # merge the template contexts, only if there are no errors, i.e. if all conjunction branches have no errors!
            template_context.merge_constraints_and(sub_template_contexts, location_id)
        return total_errors

    if isinstance(formula, StructureDisjunction):
        total_errors, sub_template_contexts = _iterate_sub_structure_formulae(
            concept_name, formula, complete_instantiation, template_context, validator, location_id
        )
        if sub_template_contexts:
            # merge the template contexts, only if there is at least a disjunction branch with no errors!
            template_context.merge_constraints_or(sub_template_contexts, location_id)
        return total_errors

    if isinstance(formula, StructureNegation):
        sub_template_context = template_context.create_unconstrained_context(location_id)
        errors = _validate_complete_instantiation_of_type(
            concept_name,
            formula.structure_constraint,
            complete_instantiation,
            sub_template_context,
            validator,
            location_id,
        )
        if errors:
            # there was a non-template-variable-related error in substitution => this means unconstrained success!
            return []
        elif sub_template_context.is_unconstrained:
            # there was no non-template-variable-related error and no template constraints => this means failure!
            err = CHSemanticError(
                f"Sub structure-formula {formula.structure_constraint} passed without constraints on template arguments"
                f" {template_context} => negation fails",
                location_id=location_id,
            )
            return [err]
        else:
            # there was no non-template-variable-related error and there are template constraints => negate constraint
            template_context.constraint = sub_template_context.make_constraint_neg()
            return []

    raise RuntimeError(f"Unknown structure formula type of {formula}!")


def _iterate_sub_structure_formulae(
    concept_name, formula, complete_substitution, template_context: TemplateContext, validator, location_id
):
    sub_template_contexts: list[TemplateContext] = []
    total_errors = []
    for structure_formula in formula.structure_constraints:
        new_location_id = location_id + [f"{structure_formula!r} <-> {complete_substitution!r}"]
        sub_template_context = template_context.create_unconstrained_context(new_location_id)
        errors = _validate_complete_instantiation_of_type(
            concept_name, structure_formula, complete_substitution, sub_template_context, validator, new_location_id
        )
        if errors:
            total_errors.extend(errors)
        else:
            sub_template_contexts.append(sub_template_context)
    return total_errors, sub_template_contexts


def _delegate_constraint_check(
    formula: TemplateConstraintFormula,
    template_argument_value: ConceptHierarchyTemplateArgument,
    template_context: TemplateContext,
    validator: TemplateConstraintArgumentValidator,
    concept_template_argument_instantiation: dict[str, ConceptHierarchyTemplateArgument],
    location_id: LocationId,
    errors: list[ConceptHierarchyError],
    collect_all_errors: bool = False,
):
    match formula:
        case TemplateConstraintAnd():
            _validate_and(
                formula,
                template_argument_value,
                template_context,
                validator,
                concept_template_argument_instantiation,
                location_id,
                errors,
                collect_all_errors,
            )
        case TemplateConstraintOr():
            _validate_or(
                formula,
                template_argument_value,
                template_context,
                validator,
                concept_template_argument_instantiation,
                location_id,
                errors,
                collect_all_errors,
            )
        case TemplateConstraintNot():
            _validate_not(
                formula,
                template_argument_value,
                template_context,
                validator,
                concept_template_argument_instantiation,
                location_id,
                errors,
                collect_all_errors,
            )
        case TemplateConstraintHierarchyOperator():
            _validate_type(
                formula,
                template_argument_value,
                template_context,
                validator,
                concept_template_argument_instantiation,
                location_id,
                errors,
                collect_all_errors,
            )
        case NonTypeTemplateConstraintFormula():
            _validate_literal(
                formula, template_argument_value, template_context, location_id, errors, collect_all_errors
            )
        case Unconstrained():
            return  # this is always successful
        case Empty():
            errors.append(
                CHSemanticError(f"Nothing matches the Empty constraint, not even {formula!r}", location_id=location_id)
            )
            return  # this is always unsuccessful
        case StructureConstraintFormula():
            raise RuntimeError(
                f"There shouldn't be a structure constraint formula here! Found {formula!r} at {location_id.print()}"
            )
        case _:
            raise ValueError(f"Unknown formula type: {type(formula)!r}")


def _validate_and(
    formula: TemplateConstraintAnd,
    template_argument_value: ConceptHierarchyTemplateArgument,
    template_context: TemplateContext,
    validator: TemplateConstraintArgumentValidator,
    concept_template_argument_instantiation: dict[str, ConceptHierarchyTemplateArgument],
    location_id: LocationId,
    errors: list[ConceptHierarchyError],
    collect_all_errors: bool,
):
    success = True
    and_errors: list[list[ConceptHierarchyError]] = []
    sub_template_contexts: list[TemplateContext] = []
    for sub_f in formula.sub_formulae:
        sub_errors = []
        new_location_id = location_id + [f"{sub_f!r} <-> {template_argument_value.full_name}"]
        sub_template_context = template_context.create_unconstrained_context(new_location_id)
        _delegate_constraint_check(
            sub_f,
            template_argument_value,
            sub_template_context,
            validator,
            concept_template_argument_instantiation,
            new_location_id,
            sub_errors,
            collect_all_errors,
        )
        if sub_errors:
            success = False
            # do not stop at first error in And
            if not and_errors or collect_all_errors:
                and_errors.append(sub_errors)
        else:
            sub_template_contexts.append(sub_template_context)
    if not success:
        err = CHSemanticError(f"{formula!r} not satisfied!", location_id=location_id)
        for and_error in and_errors:
            err.causes.extend(and_error)
        errors.append(err)
    else:
        template_context.merge_constraints_and(sub_template_contexts, location_id)


def _validate_or(
    formula: TemplateConstraintOr,
    template_argument_value: ConceptHierarchyTemplateArgument,
    template_context: TemplateContext,
    validator: TemplateConstraintArgumentValidator,
    concept_template_argument_instantiation,
    location_id: LocationId,
    errors: list[ConceptHierarchyError],
    collect_all_errors: bool,
):
    success = False
    or_errors: list[list[ConceptHierarchyError]] = []
    sub_template_contexts: list[TemplateContext] = []
    for sub_f in formula.sub_formulae:
        sub_errors = []
        new_location_id = location_id + [f"{sub_f!r} <-> {template_argument_value.full_name}"]
        sub_template_context = template_context.create_unconstrained_context(new_location_id)
        _delegate_constraint_check(
            sub_f,
            template_argument_value,
            sub_template_context,
            validator,
            concept_template_argument_instantiation,
            new_location_id,
            sub_errors,
            collect_all_errors,
        )
        if sub_errors:
            if collect_all_errors or not success:
                or_errors.append(sub_errors)
        else:
            success = True
            sub_template_contexts.append(sub_template_context)
    if not success:
        err = CHSemanticError(f"{formula!r} not satisfied!", location_id=location_id)
        for or_error in or_errors:
            err.causes.extend(or_error)
        errors.append(err)
    else:
        template_context.merge_constraints_or(sub_template_contexts, location_id)


def _validate_not(
    formula: TemplateConstraintNot,
    template_argument_value: ConceptHierarchyTemplateArgument,
    template_context: TemplateContext,
    validator: TemplateConstraintArgumentValidator,
    concept_template_argument_instantiation: dict[str, ConceptHierarchyTemplateArgument],
    location_id: LocationId,
    errors: list[ConceptHierarchyError],
    collect_all_errors: bool,
):
    sub_errors = []
    new_location_id = location_id + [f"{formula.sub_formula} <-> {template_argument_value.full_name}"]
    sub_template_context = template_context.create_unconstrained_context(new_location_id)
    _delegate_constraint_check(
        formula.sub_formula,
        template_argument_value,
        sub_template_context,
        validator,
        concept_template_argument_instantiation,
        new_location_id,
        sub_errors,
        collect_all_errors,
    )

    if sub_errors:
        # there was a non-template-variable-related error in substitution => this means unconstrained success!
        return
    elif sub_template_context.is_unconstrained:
        # there was no non-template-variable-related error and no template constraints => this means failure!
        err = CHSemanticError(
            f"Sub structure-formula {formula} passed without constraints on template arguments"
            f" {template_context} => negation fails",
            location_id=location_id,
        )
        errors.append(err)
    else:
        # there was no non-template-variable-related error and there are template constraints => negate constraint
        template_context.constraint = sub_template_context.make_constraint_neg()


def _create_iteration_data(
    template_argument_value: ConceptHierarchyTemplateArgument,
) -> tuple[tuple[ConceptHierarchyType | LiteralValue | TemplateVariable, ...], bool]:
    if not isinstance(template_argument_value, ConceptHierarchyVariadicGroup):
        is_variadic = False
        assert isinstance(template_argument_value, (ConceptHierarchyType, LiteralValue, TemplateVariable))
        to_check: tuple[ConceptHierarchyType | LiteralValue | TemplateVariable, ...] = (template_argument_value,)
    else:
        is_variadic = True
        to_check = template_argument_value.variadic_group
    return to_check, is_variadic


def get_arg_str_formula_str_and_sub_location_id(
    is_variadic: bool,
    index: int,
    formula: NonStructureConstraintFormula,
    check_formula: NonStructureConstraintFormula,
    t_arg_val: ConceptHierarchyTemplateArgument,
    location_id: LocationId,
) -> tuple[str, str, LocationId]:
    if is_variadic:
        arg_str = f"Argument {index + 1} of {t_arg_val.full_name!r}"
        sub_location_id = location_id + [f"{formula!r} <-> {arg_str}"]
    else:
        arg_str = f"{t_arg_val.full_name!r}"
        sub_location_id = location_id
    formula_str = f"{formula!r}"
    if check_formula is not formula:
        sub_location_id += [f"{check_formula!r} <-> {arg_str}"]
        formula_str += f" (substituted as {formula_str})"
    return arg_str, formula_str, sub_location_id


def _validate_type(
    formula: TemplateConstraintHierarchyOperator,
    template_argument_value: ConceptHierarchyTemplateArgument,
    template_context: TemplateContext,
    validator: TemplateConstraintArgumentValidator,
    concept_template_argument_instantiation: dict[str, ConceptHierarchyTemplateArgument],
    location_id: LocationId,
    errors: list[ConceptHierarchyError],
    collect_all_errors: bool,
):
    # process the case where the formula is a reference to a previous template argument's value!
    formula_is_template_argument = formula.literal in concept_template_argument_instantiation
    if formula_is_template_argument:
        # Create new formula from the substitution value!
        # The formula will have all template arguments (and all template arguments thereof and so on) marked with a '.'
        #  to match exactly the substituted value.
        check_formula = create_formula_from_substituted_value_for(
            concept_template_argument_instantiation[formula.literal], validator, location_id
        )
        assert isinstance(check_formula, TemplateConstraintHierarchyOperator)
        check_formula = check_formula.change_hierarchy_operator(formula.hierarchy_op, validator, location_id)
    elif not validator.is_concept(formula.literal):
        raise RuntimeError(
            f"Literal value {formula.literal!r} from {formula!r} is not a template-variable and not a concept! "
            f"How was this formula validated?!"
        )
    else:
        check_formula = formula

    to_check, is_variadic = _create_iteration_data(template_argument_value)
    has_failed = False
    for index, t_arg in enumerate(to_check):
        arg_str, formula_str, sub_location_id = get_arg_str_formula_str_and_sub_location_id(
            is_variadic, index, formula, check_formula, template_argument_value, location_id
        )
        if isinstance(t_arg, LiteralValue):
            if collect_all_errors or not has_failed:
                err = CHSemanticError(
                    f"{arg_str} is a literal value {t_arg}, which can not match the type constraint {formula!r}",
                    location_id=sub_location_id,
                )
                errors.append(err)
                has_failed = True
            continue
        assert isinstance(t_arg, (ConceptHierarchyType, TemplateVariable))
        if isinstance(t_arg, TemplateVariable):
            if not is_variadic and isinstance(t_arg, ExpandedVariadicTemplateVariable):
                raise RuntimeError(
                    f"Shouldn't use the expanded operator in non variadic context: {template_argument_value}!"
                )
            assert template_context.has_template_variable(t_arg.clean_name)
            template_context.constraint = template_context.add_and_constraint_to(
                t_arg.clean_name, check_formula, sub_location_id
            )
            continue
        assert isinstance(t_arg, ConceptHierarchyType)
        if not validator.concept_check(t_arg, check_formula.literal, check_formula.hierarchy_op):
            if collect_all_errors or not has_failed:
                err = CHSemanticError(
                    f"{arg_str} does not satisfy the constraint {formula_str}: type checking failed!",
                    location_id=sub_location_id,
                )
                errors.append(err)
                has_failed = True
            # The `continue` below is important!
            # Do not process subconstraints, i.e. constraints on template arguments,
            #   (including constraints on template variables) if the concept-check fails.
            # Because subconstraints will modify the constraints on template variables, which actually do not matter
            # because the subconcept check has failed!
            continue
        if not formula.is_templated:
            continue
        # Check the template argument constraints of the literal_type
        # It is not the template arguments of this value (template_argument_value) that must be checked,
        #  but the substitution value of the template arguments of literal_type that must match the constraints!
        assert validator.is_concept(formula.literal)
        literal_type_substituted_template_args: tuple[ConceptHierarchyTemplateArgument, ...] = (
            validator.create_substitution_for(formula.literal, t_arg, location_id)
        )
        if len(literal_type_substituted_template_args) != len(formula.literal_template_formulae):
            raise RuntimeError(
                f"Mismatch between the number of substituted template argument values "
                f"{literal_type_substituted_template_args!r} and the template argument's constraints "
                f"{len(formula.literal_template_formulae)}"
            )
        structure_constraint = ConstraintGroup(formula.location_id, formula.literal_template_formulae)

        sub_template_context = template_context.create_unconstrained_context(sub_location_id)
        subst_errors = _validate_complete_instantiation_of_type(
            formula.literal,
            structure_constraint,
            literal_type_substituted_template_args,
            sub_template_context,
            validator,
            sub_location_id,
        )
        if subst_errors:
            if collect_all_errors or not has_failed:
                err = CHSemanticError(
                    f"Constraints of {formula_str} on template arguments of {t_arg.full_name} were not satisfied",
                    location_id=location_id,
                )
                err.causes.extend(subst_errors)
                errors.append(err)
                has_failed = True
        else:
            template_context.constraint = sub_template_context.constraint


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
    formula: NonTypeTemplateConstraintFormula,
    template_argument_value: ConceptHierarchyTemplateArgument,
    template_context: TemplateContext,
    location_id: LocationId,
    errors: list[ConceptHierarchyError],
    collect_all_errors: bool,
):
    f_check: Callable[
        [NonTypeTemplateConstraintFormula | LiteralValueConstraintFormula, TemplateArgumentLiteral], bool
    ] = _check_literal_value if isinstance(formula, LiteralValueConstraintFormula) else _check_literal_type
    to_check, is_variadic = _create_iteration_data(template_argument_value)
    has_failed = False
    for index, t_arg in enumerate(to_check):
        arg_str, formula_str, sub_location_id = get_arg_str_formula_str_and_sub_location_id(
            is_variadic, index, formula, formula, template_argument_value, location_id
        )
        if isinstance(t_arg, ConceptHierarchyType):
            if collect_all_errors or not has_failed:
                err = CHSemanticError(
                    f"{arg_str} is a type value {t_arg}, which can not match the non-type constraint {formula_str}",
                    location_id=sub_location_id,
                )
                errors.append(err)
                has_failed = True
            continue
        if isinstance(t_arg, TemplateVariable):
            assert template_context.has_template_variable(t_arg)
            template_context.constraint = template_context.add_and_constraint_to(
                t_arg.clean_name, formula, sub_location_id
            )
            continue
        assert isinstance(t_arg, TemplateArgumentLiteral)
        if not f_check(formula, t_arg) and (collect_all_errors or not has_failed):
            err = CHSemanticError(
                f"{arg_str} does not satisfy the constraint {formula_str}", location_id=sub_location_id
            )
            errors.append(err)
            has_failed = True
