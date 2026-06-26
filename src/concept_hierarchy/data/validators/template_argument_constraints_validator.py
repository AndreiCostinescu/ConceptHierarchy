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
from concept_hierarchy.data.type_template_variables.constraint_formula import (
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
    TemplateConstraintHierarchyOperator,
    TemplateConstraintNot,
    TemplateConstraintOr,
    Unconstrained,
)
from concept_hierarchy.data.type_template_variables.simplify_constraints import simplify_formula
from concept_hierarchy.data.types.concept_hierarchy_types import (
    ConceptHierarchyTemplateArgument,
    ConceptHierarchyType,
    ConceptHierarchyVariadicGroup,
    ExpandedVariadicTemplateVariable,
    Instantiated,
    InstantiatedType,
    InstantiatedVariadicGroup,
    LiteralValue,
    TemplateDependentType,
    TemplateVariable,
)
from concept_hierarchy.data.types.parsed_type import TemplateArgumentLiteral
from concept_hierarchy.errors import CHSemanticError, ConceptHierarchyError, LocationId
from concept_hierarchy.utils import Reference, is_integer, is_number


class TypeTemplateInstantiationValidator(ABC):
    @abstractmethod
    def is_concept(self, concept_name: str):
        pass

    @abstractmethod
    def is_a_subconcept_of_b(self, a: str, b: str, include_self: bool) -> bool:
        pass

    @abstractmethod
    def is_a_subtype_of_b(
        self, a: InstantiatedType, b: InstantiatedType, location_id: LocationId | None = None
    ) -> bool:
        pass

    @abstractmethod
    def concept_check(self, a_type: ConceptHierarchyType, b_name: str, check_type: HierarchyCheckType) -> bool:
        pass

    @abstractmethod
    def create_substitution_for(
        self, parent_type_name: str, sub_type: ConceptHierarchyType, location_id: LocationId
    ) -> tuple[tuple[str, ConceptHierarchyTemplateArgument], ...] | None:
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

    @abstractmethod
    def create_type_constraint_from_value(
        self,
        value: ConceptHierarchyTemplateArgument,
        location_id: LocationId,
        op: HierarchyCheckType = HierarchyCheckType.SELF,
    ) -> NonStructureConstraintFormula:
        pass

    @abstractmethod
    def update_existing_template_variables(self, new_template_variables: set[str]):
        pass

    @abstractmethod
    def get_template_context(self) -> TemplateContext:
        pass


class TemplateContextDeterminator:
    def __init__(self, template_context: TemplateContext | None = None):
        self.original: TemplateContext = template_context if template_context is not None else TemplateContext()
        self.determined: TemplateContext | None = None


def validate_template_argument_value_against_constraint(
    formula: NonStructureConstraintFormula,
    template_argument_value: ConceptHierarchyTemplateArgument,
    template_context: TemplateContextDeterminator,
    validator: TypeTemplateInstantiationValidator,
    concept_template_argument_instantiation: dict[str, ConceptHierarchyTemplateArgument],
    location_id: LocationId = None,
    *,
    collect_all_errors: bool = False,
) -> list[ConceptHierarchyError]:
    if location_id is None:
        location_id = []
    assert template_context.determined is None

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


def validate_instantiation_constraints_in_template_argument_value(
    ch_type: ConceptHierarchyTemplateArgument, validator: TypeTemplateInstantiationValidator, location_id: LocationId
) -> list[ConceptHierarchyError]:
    if isinstance(ch_type, (LiteralValue, TemplateVariable)):
        return []
    assert isinstance(ch_type, (Instantiated, TemplateDependentType))
    if isinstance(ch_type, TemplateDependentType):
        template_context: TemplateContext = validator.get_template_context()
        to_check_template_context = TemplateContextDeterminator(template_context)
        # check if there are no substitution errors => no matter which substitution, instantiation will fail
        print(f"At {ch_type}")
        if str(ch_type) == "Interval<Add:T1>":
            print("DEBUG")
        errors = validate_complete_instantiation_of_concept(
            ch_type.clean_name, ch_type.template_arguments, to_check_template_context, validator, location_id
        )
        if errors or to_check_template_context.determined is None:
            return errors
        # check if the resulting template context merged with the existing context is not empty
        #   => the existing constraints on the type are incompatible with the instantiation constraints!
        print(f"  existing formula: {template_context.constraint}")
        print(f"determined formula: {to_check_template_context.determined.constraint}")
        simplified_formula = simplify_formula(
            StructureConjunction(
                location_id, (template_context.constraint, to_check_template_context.determined.constraint)
            )
        )
        print(f"simplified formula: {simplified_formula}")
        # FIXME: should this new formula be added to the existing constraint?
        #  I think so, because the usage of the template arguments demands this constraint as well...
        #  So it must be remembered!
        if simplified_formula.is_empty:
            return [
                CHSemanticError(
                    f"The existing constraints on the template variables {template_context.variables} are incompatible "
                    f"with the instantiation constraints of {ch_type}!",
                    location_id=location_id,
                )
            ]
        return []

    assert isinstance(ch_type, Instantiated)
    if isinstance(ch_type, InstantiatedVariadicGroup):
        errors = []
        for group_elem in ch_type.variadic_group:
            new_location_id = location_id + [group_elem.full_name]
            sub_errors = validate_instantiation_constraints_in_template_argument_value(
                group_elem, validator, new_location_id
            )
            if sub_errors:
                errors.extend(sub_errors)
        return errors
    assert isinstance(ch_type, InstantiatedType)
    # Because this is applied only on instantiated types (i.e. not dependent on template variables),
    #  pass an empty TemplateContext
    return validate_complete_instantiation_of_concept(
        ch_type.clean_name, ch_type.template_arguments, TemplateContextDeterminator(), validator, location_id
    )


def validate_complete_instantiation_of_concept(
    concept_name: str,
    complete_instantiation: tuple[ConceptHierarchyTemplateArgument, ...],
    template_context: TemplateContextDeterminator,
    validator: TypeTemplateInstantiationValidator,
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
    if location_id is None:
        location_id = []
    t_arg_names = validator.get_template_argument_names_of(concept_name)
    assert len(t_arg_names) == len(complete_instantiation)
    complete_instantiation_with_names = tuple((name, value) for name, value in zip(t_arg_names, complete_instantiation))
    return validate_complete_instantiation_of_type(
        formula, complete_instantiation_with_names, template_context, validator, location_id
    )


def validate_complete_instantiation_of_type(
    formula: StructureConstraintFormula,
    complete_instantiation: tuple[tuple[str, ConceptHierarchyTemplateArgument], ...],
    template_context: TemplateContextDeterminator,
    validator: TypeTemplateInstantiationValidator,
    location_id: LocationId = None,
) -> list[ConceptHierarchyError]:
    if location_id is None:
        location_id = []
    assert template_context.determined is None
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
        concept_template_argument_instantiation: dict[str, ConceptHierarchyTemplateArgument] = {
            name: value for name, value in complete_instantiation
        }
        for t_arg_index, (constraint, (_, value)) in enumerate(zip(formula.group_constraints, complete_instantiation)):
            new_location_id = location_id
            sub_template_context = TemplateContextDeterminator(template_context.original)
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
            elif sub_template_context.determined is not None:
                sub_template_contexts.append(sub_template_context.determined)
        # if there are errors, do not merge the sub-template context with the main one!
        if not total_errors and sub_template_contexts:
            # merge the template contexts only if there are no errors and if there are contexts to merge!
            template_context.determined = template_context.original.create_unconstrained_context(location_id)
            template_context.determined.merge_constraints_and(sub_template_contexts, location_id)
        return total_errors

    if isinstance(formula, StructureConjunction):
        total_errors, sub_template_contexts = _iterate_sub_structure_formulae(
            formula, complete_instantiation, template_context, validator, location_id
        )
        if not total_errors and sub_template_contexts:
            # merge the template contexts, only if there are no errors, i.e. if all conjunction branches have no errors!
            template_context.determined = template_context.original.create_unconstrained_context(location_id)
            template_context.determined.merge_constraints_and(sub_template_contexts, location_id)
        return total_errors

    if isinstance(formula, StructureDisjunction):
        total_errors, sub_template_contexts = _iterate_sub_structure_formulae(
            formula, complete_instantiation, template_context, validator, location_id
        )
        if sub_template_contexts:
            # merge the template contexts, only if there is at least a disjunction branch with no errors!
            template_context.determined = template_context.original.create_empty_context(location_id)
            template_context.determined.merge_constraints_or(sub_template_contexts, location_id)
        return total_errors

    if isinstance(formula, StructureNegation):
        sub_template_context = TemplateContextDeterminator(template_context.original)
        errors = validate_complete_instantiation_of_type(
            formula.structure_constraint,
            complete_instantiation,
            sub_template_context,
            validator,
            location_id,
        )
        if errors:
            # there was a non-template-variable-related error in substitution => this means unconstrained success!
            return []
        elif sub_template_context.determined is None:
            # there was no non-template-variable-related error and no template constraints => this means failure!
            err = CHSemanticError(
                f"Sub structure-formula {formula.structure_constraint} passed without constraints on template arguments"
                f" {template_context} => negation fails",
                location_id=location_id,
            )
            return [err]
        else:
            # there was no non-template-variable-related error and there are template constraints => negate constraint
            template_context.determined = TemplateContext(
                template_context.original.variables,
                template_context.original.variadic_variables,
                sub_template_context.determined.make_constraint_neg(),
            )
            return []

    raise RuntimeError(f"Unknown structure formula type of {formula}!")


def _iterate_sub_structure_formulae(
    formula: StructureConjunction | StructureDisjunction,
    complete_instantiation: tuple[tuple[str, ConceptHierarchyTemplateArgument], ...],
    template_context: TemplateContextDeterminator,
    validator,
    location_id,
):
    sub_template_contexts: list[TemplateContext] = []
    total_errors = []
    for structure_formula in formula.structure_constraints:
        new_location_id = location_id + [f"{structure_formula!r} <-> {complete_instantiation!r}"]
        sub_template_context = TemplateContextDeterminator(template_context.original)
        errors = validate_complete_instantiation_of_type(
            structure_formula, complete_instantiation, sub_template_context, validator, new_location_id
        )
        if errors:
            total_errors.extend(errors)
        elif sub_template_context.determined is not None:
            sub_template_contexts.append(sub_template_context.determined)
    return total_errors, sub_template_contexts


def _delegate_constraint_check(
    formula: TemplateConstraintFormula,
    template_argument_value: ConceptHierarchyTemplateArgument,
    template_context: TemplateContextDeterminator,
    validator: TypeTemplateInstantiationValidator,
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
    template_context: TemplateContextDeterminator,
    validator: TypeTemplateInstantiationValidator,
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
        sub_template_context = TemplateContextDeterminator(template_context.original)
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
        elif sub_template_context.determined is not None:
            sub_template_contexts.append(sub_template_context.determined)
    if not success:
        err = CHSemanticError(f"{formula!r} not satisfied!", location_id=location_id)
        for and_error in and_errors:
            err.causes.extend(and_error)
        errors.append(err)
        return
    if sub_template_contexts:
        template_context.determined = template_context.original.create_unconstrained_context(location_id)
        template_context.determined.merge_constraints_and(sub_template_contexts, location_id)


def _validate_or(
    formula: TemplateConstraintOr,
    template_argument_value: ConceptHierarchyTemplateArgument,
    template_context: TemplateContextDeterminator,
    validator: TypeTemplateInstantiationValidator,
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
        sub_template_context = TemplateContextDeterminator(template_context.original)
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
            if sub_template_context.determined is not None:
                sub_template_contexts.append(sub_template_context.determined)
    if not success:
        err = CHSemanticError(f"{formula!r} not satisfied!", location_id=location_id)
        for or_error in or_errors:
            err.causes.extend(or_error)
        errors.append(err)
        return
    if sub_template_contexts:
        template_context.determined = template_context.original.create_empty_context(location_id)
        template_context.determined.merge_constraints_or(sub_template_contexts, location_id)


def _validate_not(
    formula: TemplateConstraintNot,
    template_argument_value: ConceptHierarchyTemplateArgument,
    template_context: TemplateContextDeterminator,
    validator: TypeTemplateInstantiationValidator,
    concept_template_argument_instantiation: dict[str, ConceptHierarchyTemplateArgument],
    location_id: LocationId,
    errors: list[ConceptHierarchyError],
    collect_all_errors: bool,
):
    sub_errors = []
    new_location_id = location_id + [f"{formula.sub_formula} <-> {template_argument_value.full_name}"]
    sub_template_context = TemplateContextDeterminator(template_context.original)
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
    elif sub_template_context.determined is None:
        # there was no non-template-variable-related error and no template constraints => this means failure!
        err = CHSemanticError(
            f"Sub formula {formula} passed without constraints on template arguments"
            f" {template_context} => negation fails",
            location_id=location_id,
        )
        errors.append(err)
    else:
        # there was no non-template-variable-related error and there are template constraints => negate constraint
        template_context.determined = TemplateContext(
            template_context.original.variables,
            template_context.original.variadic_variables,
            sub_template_context.determined.make_constraint_neg(),
        )


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
    template_context: TemplateContextDeterminator,
    validator: TypeTemplateInstantiationValidator,
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
        check_formula = validator.create_type_constraint_from_value(
            concept_template_argument_instantiation[formula.literal], location_id, formula.hierarchy_op
        )
    elif not validator.is_concept(formula.literal):
        raise RuntimeError(
            f"Literal value {formula.literal!r} from formula {formula!r} is not a template-variable and not a concept! "
            f"How was this formula validated?!\n{concept_template_argument_instantiation!r}"
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
            assert template_context.original.has_template_variable(t_arg.clean_name)
            if template_context.determined is None:
                template_context.determined = TemplateContext(
                    template_context.original.variables,
                    template_context.original.variadic_variables,
                    template_context.original.create_unconstrained_except_with_constraint_at_name(
                        sub_location_id, t_arg.clean_name, check_formula
                    ),
                )
            else:
                template_context.determined.set_constraint(
                    template_context.determined.add_and_constraint_to(t_arg.clean_name, check_formula, sub_location_id)
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
        if check_formula.is_template_variable:
            raise RuntimeError(
                "[Feature-Request] To process this, we should support constraints of the type "
                '"Sequence<T2> should be member of T1", where both T2 and T1 are unknown template-variables.'
            )
        # Check the template argument constraints of the literal_type
        # It is not the template arguments of this value (template_argument_value) that must be checked,
        #  but the substitution value of the template arguments of literal_type that must match the constraints!
        assert validator.is_concept(formula.literal)
        literal_type_substituted_template_args: tuple[tuple[str, ConceptHierarchyTemplateArgument], ...] = (
            validator.create_substitution_for(formula.literal, t_arg, sub_location_id)
        )
        if len(literal_type_substituted_template_args) != len(formula.literal_template_formulae):
            raise RuntimeError(
                f"Mismatch between the number of substituted template argument values "
                f"{literal_type_substituted_template_args!r} and the template argument's constraints "
                f"{len(formula.literal_template_formulae)}"
            )
        structure_constraint = ConstraintGroup(formula.location_id, formula.literal_template_formulae)

        sub_template_context = TemplateContextDeterminator(template_context.original)
        subst_errors = validate_complete_instantiation_of_type(
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
                    location_id=sub_location_id,
                )
                err.causes.extend(subst_errors)
                errors.append(err)
                has_failed = True
        elif sub_template_context.determined is not None:
            if template_context.determined is None:
                template_context.determined = sub_template_context.determined
            else:
                template_context.determined.merge_in_place(sub_template_context.determined, sub_location_id)


def _check_literal_type(formula: NonTypeTemplateConstraintFormula, t_arg: TemplateArgumentLiteral) -> bool:
    if formula.constraint_type == NonTypeTemplateConstraintFormula.INTEGER:
        return is_integer(t_arg.literal_value)
    elif formula.constraint_type == NonTypeTemplateConstraintFormula.NUMBER:
        return is_number(t_arg.literal_value)
    elif formula.constraint_type == NonTypeTemplateConstraintFormula.BOOLEAN:
        return t_arg.literal_value in ["true", "false"]
    elif formula.constraint_type == NonTypeTemplateConstraintFormula.STRING:
        return t_arg.literal_value.startswith('"') and t_arg.literal_value.endswith('"')
    else:
        raise RuntimeError("Unknown constraint type: {}".format(formula.constraint_type))


def _check_literal_value(formula: LiteralValueConstraintFormula, t_arg: TemplateArgumentLiteral) -> bool:
    if formula.constraint_type == NonTypeTemplateConstraintFormula.INTEGER:
        ref = Reference()
        return is_integer(t_arg.literal_value, ref) and ref.ref == formula.value
    elif formula.constraint_type == NonTypeTemplateConstraintFormula.NUMBER:
        ref = Reference()
        return is_number(t_arg.literal_value, ref) and ref.ref == formula.value
    elif formula.constraint_type == NonTypeTemplateConstraintFormula.BOOLEAN:
        # t_arg must be exactly "true" or "false"
        return t_arg.literal_value == formula.raw_value
    elif formula.constraint_type == NonTypeTemplateConstraintFormula.STRING:
        return t_arg.literal_value == formula.raw_value
    else:
        raise RuntimeError("Unknown constraint type: {}".format(formula.constraint_type))


def _validate_literal(
    formula: NonTypeTemplateConstraintFormula,
    template_argument_value: ConceptHierarchyTemplateArgument,
    template_context: TemplateContextDeterminator,
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
            assert template_context.original.has_template_variable(t_arg)
            if template_context.determined is None:
                template_context.determined = TemplateContext(
                    template_context.original.variables,
                    template_context.original.variadic_variables,
                    template_context.original.create_unconstrained_except_with_constraint_at_name(
                        sub_location_id, t_arg.clean_name, formula
                    ),
                )
            else:
                template_context.determined.set_constraint(
                    template_context.determined.add_and_constraint_to(t_arg.clean_name, formula, sub_location_id)
                )
            continue
        assert isinstance(t_arg, TemplateArgumentLiteral)
        if not f_check(formula, t_arg) and (collect_all_errors or not has_failed):
            err = CHSemanticError(
                f"{arg_str} does not satisfy the constraint {formula_str}", location_id=sub_location_id
            )
            errors.append(err)
            has_failed = True
