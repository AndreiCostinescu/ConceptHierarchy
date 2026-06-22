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
from frozendict import frozendict

from concept_hierarchy.data.concept_hierarchy import DomainConceptData, FunctionData, TypeData, ValueDomainData
from concept_hierarchy.data.contexts.context import ConceptHierarchyContext
from concept_hierarchy.data.contexts.template_context import TemplateContext
from concept_hierarchy.data.parsers.type_parser import TemplateArgumentParser
from concept_hierarchy.data.template_argument_constraints.constraint_formula import (
    StructureConstraintFormula,
)
from concept_hierarchy.data.types.concept_hierarchy_types import (
    ConceptHierarchyTemplateArgument,
    ConceptHierarchyType,
    Instantiated,
    InstantiatedType,
    InstantiatedVariadicGroup,
    LiteralValue,
    NonVariadicTemplateVariable,
    TemplateDependent,
    TemplateDependentType,
    TemplateDependentVariadicGroup,
    TemplateVariable,
    VariadicArgument,
    VariadicTemplateVariable,
)
from concept_hierarchy.data.validators.template_argument_constraints_validator import (
    HierarchyCheckType,
    TemplateConstraintArgumentValidator,
    substitute_template_variables_in_formula,
    validate_complete_instantiation_of_concept,
)
from concept_hierarchy.data.validators.type_validator import (
    TypeTemplateData,
    TypeValidator,
    convert_template_argument_to_concept_hierarchy_template_argument,
    parse_convert_type,
    validate_template_argument_value,
)
from concept_hierarchy.definitions.concept_definition_domain_concept import DomainConceptDefinition, PropertyDefinition
from concept_hierarchy.definitions.concept_definition_domain_concept import (
    FunctionDefinition as DomainConceptFunctionDefinition,
)
from concept_hierarchy.definitions.concept_definition_functions import FunctionDefinition
from concept_hierarchy.definitions.concept_definition_hidden_implementation import HiddenImplementationDefinition
from concept_hierarchy.definitions.concept_definition_value_domain import ValueDomainDefinition
from concept_hierarchy.errors import CHSemanticError, ConceptHierarchyError, LocationId, PathPart


def substitute_non_template_variable(
    value: ConceptHierarchyTemplateArgument,
    template_context_of_value: TemplateContext,
    mapping: dict[str, ConceptHierarchyTemplateArgument],
    template_context_of_mapped_variables: TemplateContext,
    validator: TemplateConstraintArgumentValidator,
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
        sub_template_context = template_context_of_mapped_variables.create_unconstrained_context(location_id)
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
            template_context_of_mapped_variables.merge_in_place(sub_template_context, location_id)
        return InstantiatedType(value.clean_name, tuple_new_items)
    return InstantiatedVariadicGroup(value.clean_name, tuple_new_items)


# This only verifies that the constraints of (fully-instantiated) sub-values are satisfied with this substitution
# it does not verify the constraints of this value
def substitute_template_variables_in_value(
    value: ConceptHierarchyTemplateArgument,
    mapping: dict[str, ConceptHierarchyTemplateArgument],
    template_context_of_value: TemplateContext,
    template_context_of_mapped_variables: TemplateContext,
    validator: TemplateConstraintArgumentValidator,
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
    validator: TemplateConstraintArgumentValidator,
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
    validator: TemplateConstraintArgumentValidator,
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


class ConstraintValidator(TemplateConstraintArgumentValidator):
    def __init__(self, context: ConceptHierarchyContext):
        self.context = context
        self.allowed_template_variables: set[str] = set()

    # --- Abstract methods of TemplateConstraintArgumentValidator ---

    def create_substitution_for(
        self, parent_type_name: str, sub_type: ConceptHierarchyType, location_id: LocationId
    ) -> tuple[tuple[str, ConceptHierarchyTemplateArgument], ...] | None:
        sub_type_def_data = self.context.ch.concepts[sub_type.clean_name]
        parent_def_data = self.context.ch.concepts[parent_type_name]
        if not isinstance(parent_def_data, HiddenImplementationDefinition) or not parent_def_data.is_templatable():
            return ()
        # now, parent is definitely templatable
        if not isinstance(sub_type_def_data, HiddenImplementationDefinition):
            # sub_type is not a ValueDomain or a Function...
            # so a substitution between the sub_type and parent type does not exist
            return None

        substitution: dict[str, ConceptHierarchyTemplateArgument] = {}
        for t_arg_name, t_arg_val in zip(sub_type_def_data.template_argument_order, sub_type.template_arguments):
            substitution[t_arg_name] = t_arg_val

        sub_type_model_data = self.context.model.concepts[sub_type.clean_name]
        assert isinstance(sub_type_model_data, TypeData)
        all_substitutions_of_sub_type = sub_type_model_data.parent_template_variable_substitution

        substituted_t_args_of_parent: list[ConceptHierarchyTemplateArgument] = []
        for literal_t_arg in parent_def_data.template_argument_order:
            if (parent_type_name, literal_t_arg) not in all_substitutions_of_sub_type:
                assert sub_type.clean_name == parent_type_name, (
                    f"sub_type.clean_name = {sub_type.clean_name}, parent_type_name = {parent_type_name}"
                )
                if literal_t_arg in parent_def_data.variadic_template_arguments:
                    substitution_type = VariadicTemplateVariable(literal_t_arg, parent_def_data.name)
                else:
                    substitution_type = NonVariadicTemplateVariable(literal_t_arg, parent_def_data.name)
            else:
                substitution_type = all_substitutions_of_sub_type[parent_type_name, literal_t_arg]
            subst_val, subst_context = substitute(
                substitution_type,
                substitution,
                sub_type_model_data.template_context,
                TemplateContext(),
                self,
                location_id,
            )
            if not subst_context.empty:
                raise RuntimeError(
                    f"Expected full substitution for value {substitution_type}, but produced {subst_context!r}"
                )
            substituted_t_args_of_parent.append(subst_val)
        subst_tuple = tuple(substituted_t_args_of_parent)
        errors = validate_complete_instantiation_of_concept(
            parent_type_name, subst_tuple, TemplateContext(), self, location_id
        )
        if errors:
            raise RuntimeError(
                f"Wrong substitution {substituted_t_args_of_parent!r} because it doesn't satisfy all constraints of "
                f"{parent_type_name}"
            )
        return tuple(zip(parent_def_data.template_argument_order, subst_tuple))

    def concept_check(self, a_type: ConceptHierarchyType, b_name: str, check_type: HierarchyCheckType) -> bool:
        # perform the subconcept check!
        a_name = a_type.clean_name
        a_def_data = self.context.ch.concepts[a_name]
        if not isinstance(a_def_data, HiddenImplementationDefinition):
            a_is_abstract = False
        else:
            a_is_abstract = a_def_data.abstract
        match check_type:
            case HierarchyCheckType.SELF:
                include_abstract = True
                return a_name == b_name and (not a_is_abstract and include_abstract)
            case HierarchyCheckType.DESCENDANTS_OF | HierarchyCheckType.ABSTRACT_DESCENDANTS_OF:
                include_abstract = check_type == HierarchyCheckType.ABSTRACT_DESCENDANTS_OF
                include_self = True
                return self.context.ch.is_a_subconcept_of_b(a_name, b_name, include_self=include_self) and (
                    not a_is_abstract and include_abstract
                )
            case HierarchyCheckType.ASCENDANTS_OF | HierarchyCheckType.ABSTRACT_ASCENDANTS_OF:
                include_abstract = check_type == HierarchyCheckType.ABSTRACT_DESCENDANTS_OF
                include_self = False
                return self.context.ch.is_a_subconcept_of_b(b_name, a_name, include_self=include_self) and (
                    not a_is_abstract and include_abstract
                )
            case _:
                raise RuntimeError(f"Unknown hierarchy check type: {check_type!r}")

    def get_template_argument_names_of(self, concept_name: str) -> tuple[str, ...]:
        if concept_name not in self.context.ch.concepts:
            raise RuntimeError(f"{concept_name} is not a concept!")
        c_data = self.context.ch.concepts[concept_name]
        if not isinstance(c_data, HiddenImplementationDefinition):
            return ()
        return c_data.template_argument_order

    def get_constraint_formula_of(self, name: str) -> StructureConstraintFormula | None:
        if name not in self.context.model.concepts:
            raise RuntimeError("fConcept {name} is not a concept!")
        model_data = self.context.model.concepts[name]
        if isinstance(model_data, TypeData):
            return model_data.template_context.constraint
        return None

    # --- Abstract methods of TemplateConstraintFormulaValidator ---

    def full_type_name(self, name: str) -> str:
        return name

    def get_nr_template_arguments(self, concept_name: str) -> int:
        model_data = self.context.model.concepts[concept_name]
        if isinstance(model_data, TypeData):
            return len(model_data.template_context.variables)
        return 0

    def is_concept(self, name: str) -> bool:
        return self.context.ch.is_concept(name)

    def is_template_variable(self, name: str) -> bool:
        return name in self.allowed_template_variables

    def get_existing_template_variables(self) -> set[str]:
        return self.allowed_template_variables

    def update_existing_template_variables(self, new_template_variables: set[str]):
        self.allowed_template_variables = new_template_variables


class ConceptHierarchyTypeValidator(TypeValidator):
    def __init__(self, context: ConceptHierarchyContext):
        self.context = context
        self.cached_template_data: dict[str, TypeTemplateData] = {}

    def full_type_name(self, concept_name: str) -> str:
        if self.is_concept(concept_name):
            type_def_data = self.context.ch.concepts[concept_name]
            if isinstance(type_def_data, HiddenImplementationDefinition):
                return type_def_data.name_with_template_variables()
        return concept_name

    def get_template_data_of(self, concept_name: str) -> TypeTemplateData:
        if concept_name not in self.context.ch.concepts:
            raise RuntimeError(f"Wrong concept name specified: {concept_name}")
        if concept_name not in self.cached_template_data:
            type_def_data = self.context.ch.concepts[concept_name]
            if not isinstance(type_def_data, HiddenImplementationDefinition):
                type_template_data = TypeTemplateData(
                    context=TemplateContext(),
                    variadic_group_identifiers={},
                    defined_variadic_group_identifiers={},
                )
            else:
                type_model_data = self.context.model.concepts[concept_name]
                assert isinstance(type_model_data, TypeData)
                type_template_data = TypeTemplateData(
                    context=type_model_data.template_context,
                    variadic_group_identifiers=type_def_data.variadic_template_argument_group_identifiers,
                    defined_variadic_group_identifiers=type_def_data.defined_variadic_group_identifiers,
                )
            self.cached_template_data[concept_name] = type_template_data
        return self.cached_template_data[concept_name]

    def is_concept(self, concept_name: str) -> bool:
        return self.context.ch.is_concept(concept_name)

    def is_template_variable(self, concept_name: str) -> bool:
        return self.context.template_context.has_template_variable(concept_name)

    def is_variadic_template_variable(self, concept_name: str) -> bool:
        return self.context.template_context.has_template_variable(
            concept_name
        ) and self.context.template_context.is_variadic(concept_name)

    def get_available_template_variables(self) -> list[str]:
        return list(self.context.template_context.variables)


def validate_template_argument_constraints_in_instantiated_types(
    ch_type: ConceptHierarchyTemplateArgument,
    template_context: TemplateContext,
    validator: TemplateConstraintArgumentValidator,
    location_id: LocationId,
) -> list[ConceptHierarchyError]:
    if isinstance(ch_type, (LiteralValue, TemplateDependent)):
        return []
    assert isinstance(ch_type, Instantiated)
    if isinstance(ch_type, InstantiatedVariadicGroup):
        errors = []
        for group_elem in ch_type.variadic_group:
            new_location_id = location_id + [group_elem.full_name]
            sub_errors = validate_template_argument_constraints_in_instantiated_types(
                group_elem, template_context, validator, new_location_id
            )
            if sub_errors:
                errors.extend(sub_errors)
        return errors
    assert isinstance(ch_type, InstantiatedType)
    return validate_complete_instantiation_of_concept(
        ch_type.clean_name, ch_type.template_arguments, template_context, validator, location_id
    )


def check_types_in_domain_concept_definition(
    c: DomainConceptDefinition, datum: DomainConceptData, context: ConceptHierarchyContext
):
    """
    The types to verify are:
    - domain concept properties
    - domain concept functions (if present)
    """
    type_validator = ConceptHierarchyTypeValidator(context)
    property_types: dict[str, InstantiatedType] = {}
    for prop_name, prop_def_data in c.properties.items():
        if PropertyDefinition.VALUE_DOMAIN in prop_def_data:
            # validate the type of the property!
            value_domain = prop_def_data[PropertyDefinition.VALUE_DOMAIN]
            assert isinstance(value_domain, str)
            location_id = c.location_of(
                DomainConceptDefinition.domain_concept_properties, prop_name, PropertyDefinition.VALUE_DOMAIN
            )
            try:
                # check syntax and semantics of types
                ch_type = parse_convert_type(c.name, value_domain, type_validator, location_id)
            except ConceptHierarchyError as e:
                raise CHSemanticError(
                    f'Parsing the "{PropertyDefinition.VALUE_DOMAIN}" definition of property {prop_name} into a type '
                    f"failed: got {value_domain!r}",
                    location_id=location_id,
                    part=PathPart.VALUE,
                    causes=[e],
                )
            property_types[prop_name] = ch_type
        else:
            # missing checks: infer the type from the expression that is the constraint!...
            #  but this can only be done later because we can't parse expressions yet...
            assert PropertyDefinition.CONSTRAINT in prop_def_data
    datum.property_types = frozendict(property_types)
    function_types: dict[str, InstantiatedType] = {}
    for func_name, func_def_data in c.functions.items():
        location_id = c.location_of(
            DomainConceptDefinition.domain_concept_functions, func_name, DomainConceptFunctionDefinition.VALUE_DOMAIN
        )
        assert DomainConceptFunctionDefinition.VALUE_DOMAIN in func_def_data
        # validate the type of the function!
        value_domain = func_def_data[DomainConceptFunctionDefinition.VALUE_DOMAIN]
        assert isinstance(value_domain, str)
        try:
            # check syntax and semantics of types
            ch_type = parse_convert_type(c.name, value_domain, type_validator, location_id)
        except ConceptHierarchyError as e:
            raise CHSemanticError(
                f'Parsing the "{DomainConceptFunctionDefinition.VALUE_DOMAIN}" definition of function '
                f"{func_name} into a type failed: got {value_domain!r}",
                location_id=location_id,
                part=PathPart.VALUE,
                causes=[e],
            )
        function_types[func_name] = ch_type

    datum.function_types = frozendict(function_types)
    pass


def check_types_in_hidden_implementation_definition(
    c: HiddenImplementationDefinition, datum: TypeData, context: ConceptHierarchyContext
):
    """
    Check template argument substitution values
    -> parse value to a ConceptHierarchyTemplateArgument and validate the value
      -> and check the template argument constraint values present in the substitution value!
    -> then validate that the substituted value satisfies the constraints of the parent type-instantiation!
    """
    substitution_values: dict[tuple[str, str], ConceptHierarchyTemplateArgument] = {}
    constraint_validator = ConstraintValidator(context)
    constraint_validator.update_existing_template_variables(set(datum.template_context.variables))
    local_context = context.set_template_context(datum.template_context)
    type_validator = ConceptHierarchyTypeValidator(local_context)
    subst_location_key = HiddenImplementationDefinition.hidden_template_arguments_substitutions
    for parent in c.parents:
        parent_def_data = context.ch.concepts[parent]
        if not isinstance(parent_def_data, HiddenImplementationDefinition) or not parent_def_data.is_templatable():
            continue
        parent_model_data = datum.parents[parent]
        assert isinstance(parent_model_data, TypeData)
        instantiation_values_of_parent_template_variables: list[ConceptHierarchyTemplateArgument] = []
        parent_template_substitution: dict[str, ConceptHierarchyTemplateArgument] = {}
        for parent_t_arg in parent_def_data.template_argument_order:
            subst_value = c.substitution_of_template_arguments[parent, parent_t_arg]
            if c.has_location_of(subst_location_key, f"{parent_t_arg}"):
                location_id = c.location_of(subst_location_key, f"{parent_t_arg}")
            else:
                location_id = c.location_of(subst_location_key, f"{parent}:{parent_t_arg}")
            try:
                # Start validation of the syntax of the substitution value:
                #  1) Convert json object to TemplateArgumentValue
                parsed_t_arg_value = TemplateArgumentParser(subst_value, location_id).parse()
                #  2) Validate nr. template args, create variadic groups from var.ids., don't check template constraints
                validated_t_arg_value = validate_template_argument_value(
                    parsed_t_arg_value,
                    type_validator,
                    location_id,
                    parent_t_arg in parent_def_data.variadic_template_arguments,
                )
                # syntax of the substitution value is validated from now on;
                # Start checking semantic of the value
                #  1) TemplateArgumentValue => (TemplateDependent / Instantiated) & (VariadicTemplateVar. / TemplateVar)
                ch_t_arg_value = convert_template_argument_to_concept_hierarchy_template_argument(
                    datum.name, validated_t_arg_value, type_validator
                )
                #  2) Validate the template constraints of subtypes
                errors = validate_template_argument_constraints_in_instantiated_types(
                    ch_t_arg_value, datum.template_context, constraint_validator, location_id
                )
                if errors:
                    raise CHSemanticError(
                        f"Type validation failed inside {ch_t_arg_value.full_name}! "
                        f"Template argument constraints of a fully-instantiated type not satisfied!",
                        causes=errors,
                    )
            except ConceptHierarchyError as e:
                raise CHSemanticError(
                    f"Parsing {subst_value!r} into a template argument value for {parent_t_arg} failed:",
                    location_id=location_id,
                    causes=[e],
                )
            instantiation_values_of_parent_template_variables.append(ch_t_arg_value)
            parent_template_substitution[parent_t_arg] = ch_t_arg_value
            substitution_values[(parent, parent_t_arg)] = ch_t_arg_value

        location_id = c.location_of(HiddenImplementationDefinition.hidden_template_arguments_substitutions)
        sub_template_context = datum.template_context.create_unconstrained_context(location_id)
        # Validate that the substitution is semantically valid:
        #  verify that the template constraints of the Parent concept are satisfied by the substitution
        #  + update the constraints on the template variables
        errors = validate_complete_instantiation_of_concept(
            parent,
            tuple(instantiation_values_of_parent_template_variables),
            sub_template_context,
            constraint_validator,
            location_id=location_id,
        )
        if errors:
            raise CHSemanticError(
                f"Substitution values for parent {parent} defined in {c.name} does not satisfy its constraints",
                causes=errors,
            )
        # Update the constraints of the template variables of this concept with the identified constraints
        # when checking that the substitution produces a valid instantiation
        datum.template_context.merge_in_place(sub_template_context, location_id)

        # Computed the substitution values for the direct parent.
        # Now iterate through all parents and substitute their ``parent_template_variable_substitution`` entries.
        # If there are two ways to get to the same template argument/parent: throw ERROR, process later!
        for parent_t_key, parent_sub_value in parent_model_data.parent_template_variable_substitution.items():
            if parent_t_key in substitution_values:
                raise RuntimeError(
                    "[Feature-Request] Did not handle diamond inheritance/substitution for template arguments!"
                )
            # Overwrite this template context with the data from the parents
            # Actually, new constraints can not be added:
            #  because the parent context, which we substitute in the previous for loop, already has the correct
            #  constraints of their parents, and we inherit them by substituting into the parent context.
            # Thus, the substitution result context is not considered here
            substitute_constraints = False
            substitution_values[parent_t_key], substituted_template_context = substitute(
                parent_sub_value,
                parent_template_substitution,
                parent_model_data.template_context,
                datum.template_context,
                constraint_validator,
                location_id,
                validate_constraints=substitute_constraints,
            )
            if substitute_constraints:
                print(
                    f"Just for fun:\n    Substituted: {substituted_template_context}\nContext of this: "
                    f"{datum.template_context}"
                )

    datum.parent_template_variable_substitution = frozendict(substitution_values)


def check_types_in_value_domain_definition(
    c: ValueDomainDefinition, datum: ValueDomainData, context: ConceptHierarchyContext
):
    pass


def check_types_in_function_definition(c: FunctionDefinition, datum: FunctionData, context: ConceptHierarchyContext):
    pass


def check_types_in_concept_hierarchy(context: ConceptHierarchyContext):
    for c_name, c in context.ch.concepts.items():
        if isinstance(c, DomainConceptDefinition):
            check_types_in_domain_concept_definition(c, context.model.domain_concepts[c_name], context)
        if isinstance(c, HiddenImplementationDefinition):
            check_types_in_hidden_implementation_definition(c, context.model.value_domains[c_name], context)
        if isinstance(c, ValueDomainDefinition):
            check_types_in_value_domain_definition(c, context.model.value_domains[c_name], context)
        if isinstance(c, FunctionDefinition):
            check_types_in_function_definition(c, context.model.functions[c_name], context)
