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

from concept_hierarchy.data.concept_hierarchy import (
    DomainConceptData,
    FunctionArgumentAccessor,
    FunctionArgumentProvenance,
    FunctionData,
    FunctionResultAccessor,
    TypeData,
    ValueDomainArgumentProvenance,
    ValueDomainData,
)
from concept_hierarchy.data.contexts.context import ConceptHierarchyContext
from concept_hierarchy.data.contexts.template_context import TemplateContext
from concept_hierarchy.data.jsonschema import CHSchemaNode
from concept_hierarchy.data.jsonschema.parsed_schema import CrossSchemaReference
from concept_hierarchy.data.parsers.jsonschema_parser import parse_schema, resolve_schema_pointer
from concept_hierarchy.data.parsers.template_argument_constraint_parser import parse_constraint_definition
from concept_hierarchy.data.type_template_variables.constraint_formula import (
    ConstraintGroup,
    NonStructureConstraintFormula,
)
from concept_hierarchy.data.type_template_variables.simplify_constraints import (
    create_unconstrained_structure_constraint,
)
from concept_hierarchy.data.type_template_variables.template_substitution import substitute
from concept_hierarchy.data.types.concept_hierarchy_types import (
    ConceptHierarchyTemplateArgument,
    InstantiatedType,
    TypeValue,
)
from concept_hierarchy.data.utils import UNINITIALIZED
from concept_hierarchy.data.validators.template_argument_constraints_validator import (
    TemplateContextDeterminator,
    validate_complete_instantiation_of_concept,
)
from concept_hierarchy.data.validators.type_validator import (
    convert_template_argument_to_concept_hierarchy_template_argument,
    parse_convert_type,
    parse_convert_type_in_template_context,
)
from concept_hierarchy.definitions.concept_definition_domain_concept import (
    DomainConceptDefinition,
    FunctionDefinitionKeywords,
    PropertyDefinitionKeywords,
)
from concept_hierarchy.definitions.concept_definition_functions import FunctionDefinition
from concept_hierarchy.definitions.concept_definition_hidden_implementation import HiddenImplementationDefinition
from concept_hierarchy.definitions.concept_definition_value_domain import ValueDomainDefinition
from concept_hierarchy.errors import CHSemanticError, CHSyntaxError, ConceptHierarchyError, LocationId, PathPart


def check_default_instance_naming(
    context: ConceptHierarchyContext, prop_type: InstantiatedType | None, location_id: LocationId
) -> None:
    """
    Verify one use of ``nameDefaultInstanceValuesWithThisInstanceName``.

    The keyword names the default *instance* values of a property after the instance holding it, so it is
    only meaningful when the property's type contains an instance type. It may be written at a property's
    definition site or in a specialization of it, and both are checked here.

    :param context: the Concept Hierarchy to check
    :param prop_type: the resolved type of the property, or None when it could not be resolved (in which
        case only the presence of the InstanceBase concept is checked).
    :param location_id: where the keyword was written, for the error.
    """
    if not context.ch.is_concept("InstanceBase"):
        raise CHSemanticError(
            f"The InstanceBase concept is not defined in the Concept Hierarchy => can not use "
            f'"{PropertyDefinitionKeywords.DEFAULT_INSTANCE_NAMING}".\nPlease define the "InstanceBase" concept as '
            f"a subconcept of ValueDomain (and as a parent concept of Instance, if defined) or remove the "
            f'"{PropertyDefinitionKeywords.DEFAULT_INSTANCE_NAMING}" keyword from all property definitions and '
            f"specializations!",
            location_id=location_id,
            part=PathPart.KEY,
        )
    if prop_type is None:
        return
    instance_base_type = parse_convert_type("InstanceBase", context.type_validator, LocationId())
    for subtype in prop_type.iterate_subtypes(do_not_expand_instantiated_types=False):
        if not isinstance(subtype, InstantiatedType):
            continue
        if context.type_application_constraints_validator.is_a_subtype_of_b(subtype, instance_base_type, location_id):
            return
    raise CHSemanticError(
        f'Can not set "{PropertyDefinitionKeywords.DEFAULT_INSTANCE_NAMING}" for a property whose type does '
        f"not contain any instance type: {prop_type.full_name!r}!",
        location_id=location_id,
        part=PathPart.VALUE,
    )


def resolve_property_type(
    context: ConceptHierarchyContext, concept_name: str, prop_name: str
) -> InstantiatedType | None:
    """
    The resolved type of ``prop_name`` as seen by ``concept_name``: from the concept itself if it defines
    the property, otherwise from the nearest ancestor that does.

    Relies on ``check_types_in_concept_hierarchy`` traversing concepts in topological order, so that every
    ancestor's ``property_types`` is already resolved by the time a concept is checked.

    Returns None when no ancestor defines the property with a type -- the specialization checks then have
    nothing to verify against, and the specialization regime has already reported an unavailable property.
    """
    datum = context.model.domain_concepts.get(concept_name)
    if datum is not None and prop_name in datum.property_types:
        return datum.property_types[prop_name]
    # `topo_sort_concept_parents` holds every ancestor, nearest last, so the closest definition wins
    for ancestor_name in reversed(context.ch.topo_sort_concept_parents.get(concept_name, [])):
        datum = context.model.domain_concepts.get(ancestor_name)
        if datum is not None and prop_name in datum.property_types:
            return datum.property_types[prop_name]
    return None


def check_default_instance_naming_in_specializations(
    c: DomainConceptDefinition, context: ConceptHierarchyContext
) -> None:
    """
    Check ``nameDefaultInstanceValuesWithThisInstanceName`` where it is switched on by a *specialization*
    rather than at a property's definition site.

    A specialization usually applies to an inherited property, whose type lives on an ancestor; the
    topological traversal guarantees that ancestor has already been checked, so its type is available here.
    """
    specialization_location = c.location_id() + [
        DomainConceptDefinition.domain_concept_properties,
        DomainConceptDefinition.domain_concept_specialization,
    ]
    for specializations, is_for_this in (
        (c.property_specializations_for_sub, False),
        (c.property_specializations_for_this, True),
    ):
        for prop_name, specialization_data in specializations.items():
            if (
                not isinstance(specialization_data, dict)
                or PropertyDefinitionKeywords.DEFAULT_INSTANCE_NAMING not in specialization_data
            ):
                continue
            value = specialization_data[PropertyDefinitionKeywords.DEFAULT_INSTANCE_NAMING]
            if is_for_this:
                # the "_forThis" slot stores (value, is_inherited_from_the_for-subconcepts slot)
                assert isinstance(value, tuple)
                value = value[0]
            if value is not True:
                continue
            location_id = specialization_location
            if is_for_this:
                location_id = location_id + [DomainConceptDefinition.domain_concept_specialization_for_this]
            check_default_instance_naming(
                context,
                resolve_property_type(context, c.name, prop_name),
                location_id + [prop_name, PropertyDefinitionKeywords.DEFAULT_INSTANCE_NAMING],
            )


def check_types_in_domain_concept_definition(
    c: DomainConceptDefinition, datum: DomainConceptData, context: ConceptHierarchyContext
):
    """
    The types to verify are:
    - domain concept properties
    - domain concept functions (if present)
    """
    context.set_template_context(TemplateContext())
    type_validator = context.type_validator
    type_validator.set_identifier_where_types_are_defined(c.name)
    constraint_validator = context.type_application_constraints_validator

    property_types: dict[str, InstantiatedType] = {}
    value_domain_type: InstantiatedType | None = None
    domain_concept_function_type: InstantiatedType | None = None
    for prop_name, prop_def_data in c.properties.items():
        if PropertyDefinitionKeywords.VALUE_DOMAIN in prop_def_data:
            # validate the type of the property!
            value_domain = prop_def_data[PropertyDefinitionKeywords.VALUE_DOMAIN]
            assert isinstance(value_domain, str)
            location_id = c.location_of(
                DomainConceptDefinition.domain_concept_properties, prop_name, PropertyDefinitionKeywords.VALUE_DOMAIN
            )
            try:
                # check syntax and semantics of types
                ch_type = parse_convert_type(value_domain, type_validator, location_id)
            except ConceptHierarchyError as e:
                raise CHSemanticError(
                    f'Parsing the "{PropertyDefinitionKeywords.VALUE_DOMAIN}" definition of property {prop_name} into a'
                    f" type failed: got {value_domain!r}",
                    location_id=location_id,
                    part=PathPart.VALUE,
                    causes=[e],
                )
            property_types[prop_name] = ch_type
            # check that the type is a subtype of ValueDomain!
            if value_domain_type is None:
                value_domain_type = parse_convert_type("ValueDomain", type_validator, LocationId())
            if not constraint_validator.is_a_subtype_of_b(ch_type, value_domain_type, None):
                raise CHSemanticError(
                    f"The defined ValueDomain of property {prop_name} is not a subtype of ValueDomain!",
                    location_id=location_id,
                )
        else:
            # missing checks: infer the type from the expression that is the constraint!...
            #  but this can only be done later because we can't parse expressions yet...
            assert PropertyDefinitionKeywords.CONSTRAINT in prop_def_data
        if (
            PropertyDefinitionKeywords.DEFAULT_INSTANCE_NAMING in prop_def_data
            and prop_def_data[PropertyDefinitionKeywords.DEFAULT_INSTANCE_NAMING] is True
        ):
            check_default_instance_naming(
                context,
                property_types.get(prop_name),
                c.location_of(
                    DomainConceptDefinition.domain_concept_properties,
                    prop_name,
                    PropertyDefinitionKeywords.DEFAULT_INSTANCE_NAMING,
                ),
            )
    datum.property_types = frozendict(property_types)
    check_default_instance_naming_in_specializations(c, context)
    function_types: dict[str, InstantiatedType] = {}
    for func_name, func_def_data in c.functions.items():
        location_id = c.location_of(
            DomainConceptDefinition.domain_concept_functions, func_name, FunctionDefinitionKeywords.VALUE_DOMAIN
        )
        assert FunctionDefinitionKeywords.VALUE_DOMAIN in func_def_data
        # validate the type of the function!
        value_domain = func_def_data[FunctionDefinitionKeywords.VALUE_DOMAIN]
        assert isinstance(value_domain, str)
        try:
            # check syntax and semantics of types
            ch_type = parse_convert_type(value_domain, type_validator, location_id)
        except ConceptHierarchyError as e:
            raise CHSemanticError(
                f'Parsing the "{FunctionDefinitionKeywords.VALUE_DOMAIN}" definition of function '
                f"{func_name} into a type failed: got {value_domain!r}",
                location_id=location_id,
                part=PathPart.VALUE,
                causes=[e],
            )
        function_types[func_name] = ch_type
        # check that the type is a subtype of CustomFunction!
        if domain_concept_function_type is None:
            domain_concept_function_type = parse_convert_type(
                DomainConceptDefinition.default_value_domain_type_of_domain_concept_functions,
                type_validator,
                LocationId(),
            )
        if not constraint_validator.is_a_subtype_of_b(ch_type, domain_concept_function_type, location_id):
            raise CHSemanticError(
                f"The defined ValueDomain of function {func_name} is not a subtype of "
                f"{DomainConceptDefinition.default_value_domain_type_of_domain_concept_functions}!",
                location_id=location_id,
            )
    datum.function_types = frozendict(function_types)

    type_validator.clear_identifier_where_types_are_defined()
    context.reset_template_context()


def check_types_in_hidden_implementation_definition(
    c: HiddenImplementationDefinition, datum: TypeData, context: ConceptHierarchyContext
):
    """
    Check template argument substitution values
    -> parse value to a ConceptHierarchyTemplateArgument and validate the value
      -> and check the template argument constraint values present in the substitution value!
    -> then validate that the substituted value satisfies the constraints of the parent type-instantiation!
    """
    context.set_template_context(datum.template_context)
    type_validator = context.type_validator
    type_validator.set_identifier_where_types_are_defined(datum.name)
    constraint_validator = context.type_application_constraints_validator

    substitution_values: dict[tuple[str, str], ConceptHierarchyTemplateArgument] = {}
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
                location_id = c.location_of(subst_location_key, c.substitution_key_as_written(parent, parent_t_arg))
            try:
                # Converts the subst_value to a ConceptHierarchyTemplateArgument
                # using TemplateArgumentValue as an intermediate representation
                # - validates syntax and semantic rules
                # - converts variadic identifier type application to canonic variadic group representation
                ch_t_arg_value = convert_template_argument_to_concept_hierarchy_template_argument(
                    subst_value,
                    type_validator,
                    location_id,
                    parent_t_arg in parent_def_data.variadic_template_arguments,
                )
            except ConceptHierarchyError as e:
                raise CHSemanticError(
                    f"Parsing {subst_value!r} into a template argument value for {parent}:{parent_t_arg} substitution "
                    f"in {c.name} failed:",
                    location_id=location_id,
                    causes=[e],
                )
            instantiation_values_of_parent_template_variables.append(ch_t_arg_value)
            parent_template_substitution[parent_t_arg] = ch_t_arg_value
            substitution_values[(parent, parent_t_arg)] = ch_t_arg_value

        location_id = c.location_of(HiddenImplementationDefinition.hidden_template_arguments_substitutions)
        sub_template_context = TemplateContextDeterminator(datum.template_context)
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
        if sub_template_context.determined is not None:
            datum.template_context.merge_in_place(sub_template_context.determined, location_id)
            if datum.template_context.is_empty_constraint:
                raise CHSemanticError(
                    f"Merging template context with determined constraints during substitution-instantiation of "
                    f"{parent} lead to no possible template-instantiation of {c.name}",
                    location_id=location_id,
                )

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

    type_validator.clear_identifier_where_types_are_defined()
    context.reset_template_context()


def check_types_in_value_domain_definition(
    c: ValueDomainDefinition, datum: ValueDomainData, context: ConceptHierarchyContext
):
    """
    Just parse the types of the json schema; don't validate the default value.
    """
    if c.abstract:
        assert c.instantiation is None
        datum.instantiation = ()
        return
    if c.instantiation is None:
        # No defined instantiation does not mean that the ValueDomain is abstract
        # It just means that there is no way of checking that the value is correct =>
        # => this is equivalent to an instantiation schema definition of `"instantiation": true`
        location_id = c.location_id()
        all_instantiations = [(location_id, (c.create_fallback_instantiation_constraint(), True))]
    else:
        location_id = c.location_id(ValueDomainDefinition.value_domain_instantiation)
        if c.was_template_dependent_instantiation_defined:
            all_instantiations = zip((location_id + [index] for index in range(len(c.instantiation))), c.instantiation)
        else:
            assert len(c.instantiation) == 1
            all_instantiations = [(location_id, c.instantiation[0])]

    constraint_validator = context.template_constraint_formula_validator
    context.set_template_context(datum.template_context)
    context.instantiation_schema_validator.set_identifier_where_types_are_defined(c.name)

    location_id = c.location_id(ValueDomainDefinition.value_domain_instantiation)
    parsed_instantiations: list[tuple[ConstraintGroup, CHSchemaNode]] = []
    for instantiation_location_id, (instantiation_constraints, instantiation_schema) in all_instantiations:
        parsed_instantiation_schema, errors = parse_schema(
            instantiation_schema, context.instantiation_schema_validator, instantiation_location_id
        )
        if errors:
            raise CHSyntaxError(f"Parsing {instantiation_schema!r} into a json schema failed!", causes=errors)
        group_constraints: list[NonStructureConstraintFormula] = []
        for constraint in instantiation_constraints:
            parsed_constraint = parse_constraint_definition(
                constraint, constraint_validator, location_id, allow_unconstrained=True
            )
            assert isinstance(parsed_constraint, NonStructureConstraintFormula)
            group_constraints.append(parsed_constraint)
        if not group_constraints:
            constraint = None
        else:
            constraint = ConstraintGroup(instantiation_location_id, tuple(group_constraints))
        parsed_instantiations.append((constraint, parsed_instantiation_schema))
    datum.instantiation = tuple(parsed_instantiations)

    context.instantiation_schema_validator.clear_identifier_where_types_are_defined()
    context.reset_template_context()


def check_types_in_function_definition(c: FunctionDefinition, datum: FunctionData, context: ConceptHierarchyContext):
    """
    Check Function evaluation arguments, result type if defined, addVariablesToScope, and scopeVariables

    :param c: Function concept for which to check types
    :param datum: the output data container
    :param context: the concept hierarchy in which the check is made
    """
    context.set_template_context(datum.template_context)
    type_validator = context.type_validator
    type_validator.set_identifier_where_types_are_defined(c.name)
    # Setting the allowed template variables for the type_validator's type_instantiation_validator is not needed
    #  Because here, the type_instantiation_validator only checks instantiated types, which do not have template vars.

    function_evaluation_argument_types: dict[str, InstantiatedType] = {}
    function_evaluation_argument_access: dict[str, FunctionArgumentAccessor] = {}
    function_evaluation_argument_provenance: dict[str, FunctionArgumentProvenance] = {}
    for f_eval_arg_name, f_eval_arg_type in c.evaluation_argument_types.items():
        assert f_eval_arg_name in c.evaluation_argument_access_types
        assert f_eval_arg_name in c.evaluation_argument_provenance_types
        # validate the type of the function evaluation argument!
        assert isinstance(f_eval_arg_type, str)
        location_id = c.location_of(FunctionDefinition.function_interface, f_eval_arg_name)
        try:
            # check syntax and semantics of types
            ch_type = parse_convert_type_in_template_context(f_eval_arg_type, type_validator, location_id)
        except ConceptHierarchyError as e:
            raise CHSemanticError(
                f"Parsing the definition of the evaluation argument {f_eval_arg_name} into a type "
                f"failed: got {f_eval_arg_type!r}",
                location_id=location_id,
                part=PathPart.VALUE,
                causes=[e],
            )
        function_evaluation_argument_types[f_eval_arg_name] = ch_type
        function_evaluation_argument_access[f_eval_arg_name] = FunctionArgumentAccessor(
            c.evaluation_argument_access_types[f_eval_arg_name]
        )
        function_evaluation_argument_provenance[f_eval_arg_name] = FunctionArgumentProvenance(
            c.evaluation_argument_provenance_types[f_eval_arg_name]
        )
    # merge with arguments of parent Function(s) and collect all default arguments (not their expressions)
    all_default_arguments: set[str] = set(c.evaluation_argument_default_values.keys())
    for p_name in c.parents:
        if p_name not in context.ch.functions:
            continue
        p = context.ch.concepts[p_name]
        assert isinstance(p, FunctionDefinition)
        p_model = context.model.functions[p_name]
        assert isinstance(p_model, FunctionData)
        all_default_arguments.update(p_model.evaluation_default_arguments)
        for f_eval_arg_name in c.evaluation_argument_types:
            assert f_eval_arg_name not in p_model.evaluation_argument_types, (
                f"There shouldn't be an argument with a duplicate name {f_eval_arg_name} defined in {c.name}"
            )
        function_evaluation_argument_types.update(p_model.evaluation_argument_types)
        function_evaluation_argument_provenance.update(p_model.evaluation_argument_provenance_type)
        function_evaluation_argument_access.update(p_model.evaluation_argument_access_type)

    datum.evaluation_default_arguments = frozenset(all_default_arguments)
    datum.evaluation_argument_types = frozendict(function_evaluation_argument_types)
    datum.evaluation_argument_access_type = frozendict(function_evaluation_argument_access)
    datum.evaluation_argument_provenance_type = frozendict(function_evaluation_argument_provenance)
    # process type of result
    if c.returns_something:
        assert isinstance(c.result_type, str)
        location_id = c.location_of(FunctionDefinition.function_interface, FunctionDefinition.function_result)
        try:
            # check syntax and semantics of types
            ch_type = parse_convert_type_in_template_context(c.result_type, type_validator, location_id)
        except ConceptHierarchyError as e:
            raise CHSemanticError(
                f"Parsing the definition of the {c.definition_type()} result into a type failed: got {c.result_type!r}",
                location_id=location_id,
                part=PathPart.VALUE,
                causes=[e],
            )
        datum.evaluation_result_type = ch_type
        datum.evaluation_result_access_type = FunctionResultAccessor(c.result_access_type)
        datum.evaluation_result_provenance_type = ValueDomainArgumentProvenance(c.result_provenance_type)
    elif isinstance(c.result_defined_in, str):
        datum.evaluation_result_type = context.model.functions[c.result_defined_in].evaluation_result_type
        datum.evaluation_result_access_type = context.model.functions[c.result_defined_in].evaluation_result_access_type
        datum.evaluation_result_provenance_type = context.model.functions[
            c.result_defined_in
        ].evaluation_result_provenance_type
    else:
        datum.evaluation_result_type = None if c.result_defined_in is None else UNINITIALIZED
        datum.evaluation_result_access_type = None
        datum.evaluation_result_provenance_type = None

    sub_scope_data: dict[str, dict[str, tuple[TypeValue, bool]]] = {}
    for f_arg, new_var_data in c.sub_scopes.items():
        for new_var_name, new_var_def_data in new_var_data.items():
            new_var_type = new_var_def_data[0]
            assert isinstance(new_var_type, str)
            location_id = c.location_of(FunctionDefinition.function_sub_scopes, f_arg, new_var_name)
            try:
                # check syntax and semantics of types
                ch_type = parse_convert_type_in_template_context(new_var_type, type_validator, location_id)
            except ConceptHierarchyError as e:
                raise CHSemanticError(
                    f"Parsing the definition of the new variable {new_var_name!r} to be added to the sub-scope of "
                    f"evaluation argument {f_arg!r} into a type failed: got {new_var_type!r}",
                    location_id=location_id,
                    part=PathPart.VALUE,
                    causes=[e],
                )
            if f_arg not in sub_scope_data:
                sub_scope_data[f_arg] = {}
            sub_scope_data[f_arg][new_var_name] = (ch_type, new_var_def_data[1])
    datum.sub_scope_vars = frozendict(sub_scope_data)

    add_to_existing_scope: dict[str, tuple[TypeValue, bool]] = {}
    for new_var_name, new_var_def_data in c.add_new_variables_in_existing_scope.items():
        new_var_type = new_var_def_data[0]
        assert isinstance(new_var_type, str)
        location_id = c.location_of(FunctionDefinition.function_add_new_variables_in_existing_scope, new_var_name)
        try:
            # check syntax and semantics of types
            ch_type = parse_convert_type_in_template_context(new_var_type, type_validator, location_id)
        except ConceptHierarchyError as e:
            raise CHSemanticError(
                f"Parsing the definition of the new variable {new_var_name!r} to be added to the existing scope into a "
                f"type failed: got {new_var_type!r}",
                location_id=location_id,
                part=PathPart.VALUE,
                causes=[e],
            )
        add_to_existing_scope[new_var_name] = (ch_type, new_var_def_data[1])
    datum.new_vars_in_scope = frozendict(add_to_existing_scope)

    # parse: default_function_instantiation_schema (if the Function is instantiable (not abstract))
    if not datum.instantiable:
        datum.instantiation = ()
    else:
        context.instantiation_schema_validator.set_identifier_where_types_are_defined(c.name)

        location_id = c.location_id(ValueDomainDefinition.value_domain_instantiation)

        instantiation_location_id = location_id + ["default Function instantiation"]
        parsed_instantiation_schema, errors = parse_schema(
            FunctionDefinition.default_function_instantiation_schema,
            context.instantiation_schema_validator,
            instantiation_location_id,
        )
        assert not errors

        datum.instantiation = (
            (
                create_unconstrained_structure_constraint(
                    datum.template_context.nr_variables, instantiation_location_id
                ),
                parsed_instantiation_schema,
            ),
        )

        context.instantiation_schema_validator.clear_identifier_where_types_are_defined()

    type_validator.clear_identifier_where_types_are_defined()
    context.reset_template_context()


def check_schema_references_in_concept_hierarchy(context: ConceptHierarchyContext) -> None:
    """
    Validate every ``#ch#`` reference, in a pass of its own after all instantiation schemas exist.

    A second pass because a cross-schema reference is the one thing about a schema that its own concept
    cannot decide. Schemas are parsed in `concept_topo_sort` order, which orders parents before children --
    a reference is neither, so the target may not have been reached yet, and may equally be a descendant.
    Waiting until every `ValueDomainData.instantiation` is filled removes the ordering question entirely.

    What is checked is everything that does not need an *application*: that the target names a type, that
    the type is a non-abstract ValueDomain with an instantiation, that the index obeys the rule below, and
    that the pointer addresses something in the target's declared schema. Binding the reference to a node
    is `bind_cross_schema_references`, and happens per application, because the node it must land on is one
    of the *target application's* resolved schema.

    **The index is present exactly when the target wrote a template-dependent instantiation.** Those have
    one entry per constraint group, and which one is meant is not something a reader should have to work
    out; the others have exactly one schema, and writing ``/0`` for it would suggest a choice that is not
    there -- so it is rejected rather than accepted as a synonym for the only possibility.
    """
    context.set_template_context(TemplateContext())
    for c_name, datum in context.model.value_domains.items():
        for schema_index, (_constraint, schema) in enumerate(datum.instantiation):
            for node in schema.walk():
                if node.cross_reference is not None:
                    _check_schema_reference(context, node.cross_reference, node.location_id)
    context.reset_template_context()


def _check_schema_reference(
    context: ConceptHierarchyContext, reference: CrossSchemaReference, location_id: LocationId
) -> None:
    """One ``#ch#`` reference; raises the first thing that is wrong with it."""
    reference_location = location_id + ["$ref"]

    def bad(message: str, causes: list | None = None) -> CHSemanticError:
        return CHSemanticError(
            f"Cannot resolve $ref {reference.written!r}: {message}",
            location_id=reference_location,
            part=PathPart.VALUE,
            causes=causes or [],
        )

    try:
        target_type = parse_convert_type(reference.type_name, context.type_validator, reference_location)
    except ConceptHierarchyError as e:
        # Deliberately parsed in the *empty* template context: the reference must name a ground
        # application, so a template variable of the referring concept is not in scope here and naming one
        # is exactly the mistake this reports.
        raise bad(f"{reference.type_name!r} is not a ground type of this Concept Hierarchy", [e])

    target_name = target_type.clean_name
    if target_name not in context.model.value_domains:
        raise bad(f"{target_type} is not a ValueDomain, so it has no instantiation schema to point into")
    if not context.model.concepts[target_name].instantiable:
        raise bad(f"{target_type} is abstract, so it has no instantiation schema to point into")

    target_definition = context.ch.concepts[target_name]
    assert isinstance(target_definition, ValueDomainDefinition)
    if target_definition.instantiation is None:
        raise bad(
            f"{target_type} declares no instantiation, so there is nothing in it to reference "
            f"(its schema is the unconstrained one, which has no definitions)"
        )

    schemas = context.model.value_domains[target_name].instantiation
    if target_definition.was_template_dependent_instantiation_defined:
        if reference.schema_index is None:
            raise bad(
                f"{target_type} defines {len(schemas)} template-dependent instantiation schemas, so the "
                f'reference must say which one: "#ch#/{reference.type_name}/<index>/#/...".'
            )
        if reference.schema_index >= len(schemas):
            raise bad(
                f"{target_type} defines {len(schemas)} instantiation schema(s), so there is no schema at "
                f"index {reference.schema_index}"
            )
    elif reference.schema_index is not None:
        raise bad(
            f"{target_type} does not define template-dependent instantiation schemas, so it has exactly "
            f"one and an index must not be written"
        )

    declared_schema = schemas[reference.schema_index or 0][1]
    resolved, failure = resolve_schema_pointer(declared_schema, list(reference.pointer))
    if resolved is None:
        raise bad(f"in the instantiation schema of {target_type}, {failure}")


def check_types_in_concept_hierarchy(context: ConceptHierarchyContext):
    # First process template types
    for c_name, c in context.ch.concepts.items():
        context.model.concepts[c_name].instantiable = not c.abstract
        if isinstance(c, HiddenImplementationDefinition):
            check_types_in_hidden_implementation_definition(c, context.model.value_domains[c_name], context)
    # Then check types in the Concept Hierarchy (property types, Function argument types, etc.).
    # This traverses in topological order so that a concept's parents are always checked before it, which
    # lets a check consult data that was resolved on an ancestor -- inherited property types, for instance.
    for c_name in context.ch.concept_topo_sort:
        c = context.ch.concepts[c_name]
        if isinstance(c, DomainConceptDefinition):
            check_types_in_domain_concept_definition(c, context.model.domain_concepts[c_name], context)
        if isinstance(c, ValueDomainDefinition):
            check_types_in_value_domain_definition(c, context.model.value_domains[c_name], context)
        if isinstance(c, FunctionDefinition):
            check_types_in_function_definition(c, context.model.functions[c_name], context)
    # Last, because it is the one check that reads *other* concepts' finished schemas.
    check_schema_references_in_concept_hierarchy(context)
