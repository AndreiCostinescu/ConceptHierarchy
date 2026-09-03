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

from concept_hierarchy.data.contexts.template_context import TemplateContext
from concept_hierarchy.data.type_template_variables.constraint_formula import (
    ConstraintGroup,
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
)
from concept_hierarchy.data.types.concept_hierarchy_types import (
    ConceptHierarchyTemplateArgument,
    ConceptHierarchyType,
    Instantiated,
    InstantiatedType,
    InstantiatedVariadicGroup,
    LiteralValue,
    TemplateDependent,
    TemplateDependentType,
    TemplateDependentVariadicGroup,
    TemplateVariable,
    VariadicArgument,
    VariadicTemplateVariable,
)
from concept_hierarchy.data.validators.template_argument_constraints_validator import (
    TemplateContextDeterminator,
    TypeTemplateInstantiationValidator,
    validate_complete_instantiation_of_concept,
)
from concept_hierarchy.errors import CHSemanticError, LocationId


def substitute_template_variables_in_formula(
    formula: TemplateConstraintFormula,
    substitution: dict[str, ConceptHierarchyTemplateArgument | None],
    validator: TypeTemplateInstantiationValidator,
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
                subst_value: ConceptHierarchyTemplateArgument = substitution[formula.literal]
                if isinstance(subst_value, TemplateVariable):
                    subst_literal = subst_value.clean_name
                    # don't change subst_t_args, because a template variable does not have template arguments
                elif isinstance(subst_value, LiteralValue):
                    return LiteralValueConstraintFormula(subst_value.literal_type, subst_value.full_name, location_id)
                elif isinstance(subst_value, VariadicArgument):
                    raise RuntimeError("[Feature-Request] Did not implement support for using variadic constraints!")
                else:
                    assert isinstance(subst_value, ConceptHierarchyType)
                    subst_formula = validator.create_type_constraint_from_value(
                        subst_value, location_id, formula.hierarchy_op
                    )
                    return substitute_template_variables_in_formula(subst_formula, substitution, validator, location_id)
            return formula.create_new_same_op(subst_literal, tuple(subst_t_args), validator, location_id)
        case TemplateConstraintAnd():
            sub_formulae: list[NonStructureConstraintFormula] = []
            for sub_f in formula.sub_formulae:
                res = substitute_template_variables_in_formula(sub_f, substitution, validator, location_id)
                if not isinstance(res, NonStructureConstraintFormula):
                    raise RuntimeError(f"Expected a NonStructureConstraintFormula, got {res!r}")
                sub_formulae.append(res)
            return TemplateConstraintAnd(location_id, tuple(sub_formulae))
        case TemplateConstraintOr():
            sub_formulae: list[NonStructureConstraintFormula] = []
            for sub_f in formula.sub_formulae:
                res = substitute_template_variables_in_formula(sub_f, substitution, validator, location_id)
                if not isinstance(res, NonStructureConstraintFormula):
                    raise RuntimeError(f"Expected a NonStructureConstraintFormula, got {res!r}")
                sub_formulae.append(res)
            return TemplateConstraintOr(location_id, tuple(sub_formulae))
        case TemplateConstraintNot():
            res = substitute_template_variables_in_formula(formula.sub_formula, substitution, validator, location_id)
            if not isinstance(res, NonStructureConstraintFormula):
                raise RuntimeError(f"Expected a NonStructureConstraintFormula, got {res!r}")
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


def substitute_non_template_variable(
    value: ConceptHierarchyTemplateArgument,
    template_context_of_value: TemplateContext,
    mapping: dict[str, ConceptHierarchyTemplateArgument],
    template_context_of_mapped_variables: TemplateContext,
    validator: TypeTemplateInstantiationValidator,
    location_id: LocationId,
):
    assert isinstance(value, TemplateDependent)
    assert not isinstance(value, TemplateVariable)
    if isinstance(value, TemplateDependentType):
        to_iterate = value.template_arguments
        is_type = True
        str_id = "sub-tArg"
    elif isinstance(value, TemplateDependentVariadicGroup):
        to_iterate = value.variadic_group
        is_type = False
        str_id = "sub-elem"
    else:
        raise RuntimeError(f"Unknown non_template_variable: {value!r}")
    new_items = []
    has_template_dependent_items = False
    sub_template_contexts: list[TemplateContext] = []
    for item in to_iterate:
        sub_location_id = location_id + [f"{str_id}-{item!r}"]
        sub_template_context = template_context_of_mapped_variables.create_unconstrained_context(sub_location_id)
        new_item = substitute_template_variables_in_value(
            item, mapping, template_context_of_value, sub_template_context, validator, sub_location_id
        )
        has_template_dependent_items |= isinstance(new_item, TemplateDependent)
        new_items.append(new_item)
    template_context_of_mapped_variables.merge_constraints_and(sub_template_contexts, location_id)
    if has_template_dependent_items:
        if is_type:
            return TemplateDependentType(value.clean_name, tuple(new_items))
        return TemplateDependentVariadicGroup(value.clean_name, tuple(new_items))
    tuple_new_items = tuple(new_items)
    if is_type:
        # check instantiation constraints only for completely instantiated types
        sub_template_context = TemplateContextDeterminator(
            template_context_of_mapped_variables.create_unconstrained_context(location_id)
        )
        errors = validate_complete_instantiation_of_concept(
            value.clean_name, tuple_new_items, sub_template_context, validator, location_id
        )
        if errors:
            raise CHSemanticError(
                f"Wrong substitution {tuple_new_items!r} for {value.clean_name!r}",
                location_id=location_id,
                causes=errors,
            )
        else:
            template_context_of_mapped_variables.merge_in_place(sub_template_context.determined, location_id)
        return InstantiatedType(value.clean_name, tuple_new_items)
    return InstantiatedVariadicGroup(value.clean_name, tuple_new_items)


# This only verifies that the constraints of (fully-instantiated) sub-values are satisfied with this substitution
# it does not verify the constraints of this value
def substitute_template_variables_in_value(
    value: ConceptHierarchyTemplateArgument,
    mapping: dict[str, ConceptHierarchyTemplateArgument],
    template_context_of_value: TemplateContext,
    template_context_of_mapped_variables: TemplateContext,
    validator: TypeTemplateInstantiationValidator,
    location_id: LocationId,
) -> ConceptHierarchyTemplateArgument:
    """
    Substitute value,
    update constraints on the substituting template variables, and
    verify that the instantiation of fully-instantiated types is correct with this substitution.
    """
    if isinstance(value, Instantiated):
        return value
    if all(x not in mapping for x in value.used_templates):
        return value
    assert isinstance(value, TemplateDependent)
    if not isinstance(value, TemplateVariable):
        return substitute_non_template_variable(
            value, template_context_of_value, mapping, template_context_of_mapped_variables, validator, location_id
        )
    assert isinstance(value, TemplateVariable)
    substituted_value = mapping[value.clean_name]
    if isinstance(value, VariadicTemplateVariable) and not isinstance(substituted_value, VariadicArgument):
        raise RuntimeError(
            f"Can not substitute a non-variadic value {substituted_value!r} for the variadic template variable {value}"
        )
    return substituted_value


def substitute_and_validate_constraints(
    template_context_to_substitute: TemplateContext,
    template_context_of_substitution: TemplateContext,
    mapping: dict[str, ConceptHierarchyTemplateArgument],
    validator: TypeTemplateInstantiationValidator,
    location_id: LocationId,
) -> TemplateContext:
    res_template_context = template_context_of_substitution.create_unconstrained_context(location_id)
    print(f"Substitution mapping: {mapping!r}")
    print(f" Pre substitution of constraint: {template_context_to_substitute.constraint}")
    # create the mapping between all template arguments and their substitution value
    #  put None if the template argument is not substituted
    complete_mapping = {x: mapping.get(x, None) for x in template_context_to_substitute.variables}
    substituted_old_constraint = substitute_template_variables_in_formula(
        template_context_to_substitute.constraint, complete_mapping, validator, location_id
    )
    assert isinstance(substituted_old_constraint, StructureConstraintFormula)
    print(f"Post substitution of constraint: {substituted_old_constraint}")
    # create_instantiation: only possible if all template variables of the parent are substituted!
    if len(mapping) == len(template_context_to_substitute.variables):
        template_instantiation = tuple((x, mapping[x]) for x in template_context_to_substitute.variables)
        instantiation_template_context = template_context_of_substitution.create_unconstrained_context(location_id)
        print(substituted_old_constraint)
        print(template_instantiation)
        print(instantiation_template_context)
        """
        errors = validate_complete_instantiation_of_type(
            substituted_old_constraint, template_instantiation, instantiation_template_context, validator, location_id
        )
        """
        # TODO: Need function that can validate an instantiation with
        #  subst-constraints: SubType<And(F, Not(F.)), ValueDomain>  where S, F are not template variables of SubType!
        #      instantiation: SubType<S,               F>            where S, F are not template variables of SubType!
        #            SubType: SubType<SubT,            T>            where SubT, T are the template variables of SubType
        errors = []
        if errors:
            raise CHSemanticError("Substitution does not satisfy the constraints?!", causes=errors)
        res_template_context.merge_in_place(instantiation_template_context, location_id)
    else:
        raise RuntimeError("[Feature-Request] Did not yet implement partial substitution!")
    return res_template_context


def substitute(
    value: ConceptHierarchyTemplateArgument,
    mapping: dict[str, ConceptHierarchyTemplateArgument],
    template_context_of_value: TemplateContext,
    template_context_of_mapped_template_variables: TemplateContext,
    validator: TypeTemplateInstantiationValidator,
    location_id: LocationId,
    validate_constraints: bool = False,
) -> tuple[ConceptHierarchyTemplateArgument, TemplateContext]:
    unsubstituted_variables = set(x for x in template_context_of_value.variables if x not in mapping)
    if unsubstituted_variables:
        raise RuntimeError("[Feature-Request] Partial substitution is not yet implemented!")
    substituted_template_context = template_context_of_mapped_template_variables.create_unconstrained_context(
        location_id
    )
    substituted_value = substitute_template_variables_in_value(
        value, mapping, template_context_of_value, substituted_template_context, validator, location_id
    )
    res_template_context = TemplateContext.create_from(template_context_of_mapped_template_variables)
    res_template_context.merge_in_place(substituted_template_context, location_id)

    if validate_constraints:
        res_template_context.merge_in_place(
            substitute_and_validate_constraints(
                template_context_of_value,
                template_context_of_mapped_template_variables,
                mapping,
                validator,
                location_id,
            ),
            location_id,
        )
    return substituted_value, substituted_template_context
