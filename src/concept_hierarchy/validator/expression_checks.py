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

from concept_hierarchy.data.concept_hierarchy import (
    DomainConceptData,
    FunctionData,
    GlobalVariableData,
    TypeData,
    ValueDomainData,
)
from concept_hierarchy.data.contexts.context import ConceptHierarchyContext, TemplateContext
from concept_hierarchy.data.contexts.variable_context import VariableContext, VariableStackFrame
from concept_hierarchy.data.expressions.expression import Expression
from concept_hierarchy.data.expressions.expression_utils import FunctionArgumentAccessor, FunctionArgumentProvenance
from concept_hierarchy.data.expressions.subexpressions import IllFormedExpression
from concept_hierarchy.data.parsers.expression_parser import get_expression_type, parse_expression
from concept_hierarchy.data.types.concept_hierarchy_types import TypeValue, frozendict
from concept_hierarchy.definitions.concept_definition_domain_concept import DomainConceptDefinition
from concept_hierarchy.definitions.concept_definition_functions import FunctionDefinition
from concept_hierarchy.definitions.concept_definition_hidden_implementation import HiddenImplementationDefinition
from concept_hierarchy.definitions.concept_definition_value_domain import ValueDomainDefinition
from concept_hierarchy.errors import CHSemanticError, PathPart


def check_expressions_in_domain_concept_definition(
    c: DomainConceptDefinition, datum: DomainConceptData, context: ConceptHierarchyContext
):
    """
    The expressions to verify are:
    - domain concept properties
        - constraints + set valueDomain if not set
        - confidence duration value
        - hook procedures
            - checks that functions exist,
            - function arguments exist,
            - function argument type is a parent type of the property type
        - computation procedures
        - assumption Variation values
        - default values
        - specialization values:
            - check all of the above
    - domain concept function procedures:
        - CustomFunction instantiations
        - Function procedures
        - specialization values:
            - default Function procedures / CustomFunction instantiations
    - domain concept management:
        - initialization function procedure
        - consolidation function procedure
    """
    pass


def check_expressions_in_hidden_implementation_definition(
    c: HiddenImplementationDefinition, datum: TypeData, context: ConceptHierarchyContext
):
    """No expressions in HiddenImplementationDefinitions."""
    pass


def check_expressions_in_value_domain_definition(
    c: ValueDomainDefinition, datum: ValueDomainData, context: ConceptHierarchyContext
):
    """
    Check default values of value domain instantiations (plus type-check value domain instantiations themselves)
    """
    pass


def check_expressions_in_function_definition(
    c: FunctionDefinition, datum: FunctionData, context: ConceptHierarchyContext
):
    """
    Check Function evaluation default argument values, variations, procedure, and inversions

    :param c: Function concept for which to check expressions
    :param datum: the output data container
    :param context: the concept hierarchy in which the check is made
    """
    pass


def check_expressions_in_concept_hierarchy(context: ConceptHierarchyContext):
    context.set_template_context(TemplateContext())
    context.set_variable_context(VariableContext([]))

    # First, process the type of global variables; this doesn't process the expression value!
    # It processes just the type so the variable can be used/registered!
    value_domain_type = None
    global_variable_context: dict[str, TypeValue] = {}
    remaining_variables_to_check: set[str] = set(context.ch.instances)
    while remaining_variables_to_check:
        processed_variables: int = 0
        for global_var_name in list(remaining_variables_to_check):
            global_var_data = context.ch.instances[global_var_name]
            # initialization
            if value_domain_type is None:
                value_domain_type = context.expression_parser_validator.create_instantiated_type(
                    "ValueDomain", global_var_data.definition_location_id
                )
            # skip unprocessed aliases
            if global_var_data.is_reference() and global_var_data.is_reference_to in remaining_variables_to_check:
                continue
            global_var_model = GlobalVariableData(global_var_name, global_var_data.is_reference())
            if global_var_model.is_alias:
                global_var_model.value_type = context.model.instances[global_var_data.is_reference_to].value_type
            else:
                expr_type_res = get_expression_type(
                    global_var_data.value,
                    value_domain_type,
                    TemplateContext(),
                    context.expression_parser_validator,
                    global_var_data.definition_location(),
                )
                if expr_type_res is None:
                    raise CHSemanticError(
                        f"Could not determine the type of expression {global_var_data.value}",
                        location_id=global_var_data.definition_location(),
                    )
                global_var_model.value_type = expr_type_res
            print(f"Type of global variable {global_var_name} is: {global_var_model.value_type}")
            global_variable_context[global_var_name] = global_var_model.value_type
            context.model.instances[global_var_name] = global_var_model
            processed_variables += 1
            remaining_variables_to_check.remove(global_var_name)
        if processed_variables == 0:
            raise RuntimeError(
                f"Apparently there is a cycle in global variables which was not detected before? Remaining variables to"
                f" check: {remaining_variables_to_check}"
            )

    context.push_new_variable_stack_frame(VariableStackFrame(global_variable_context))

    # First, process the expressions for Function default arguments; this allows creating the dependencies between the
    #   default Function argument values; which allows determining whether a Function evaluation/composition is
    #   well-formed (if all needed arguments are supplied and if there's no circular dependency in Function arguments)
    # - first process the expressions
    # - then check on which variables do the default values depend on
    # - create the dependency graph between default values; this must be checked at every Function evaluation in all
    #   future expressions!
    for c_name, c in context.model.functions.items():
        context.set_template_context(c.template_context)
        constraint_validator = context.template_constraint_formula_validator
        constraint_validator.update_existing_template_variables(set(context.template_context.variables))
        context.instantiation_schema_validator.set_identifier_where_types_are_defined(c.name)
        context.push_new_variable_stack_frame(VariableStackFrame())

        c_def = context.ch.concepts.get(c_name)
        assert isinstance(c_def, FunctionDefinition)

        all_default_argument_values: dict[str, Expression] = {}

        # add the other Function arguments as variables to the variable context!
        for arg_name, arg_type in c.evaluation_argument_types.items():
            context.add_new_variable(arg_name, arg_type)

        # process the default argument expressions of this concept
        if c_def.has_location_of(FunctionDefinition.function_default_argument_values):
            default_args_location_id = c_def.location_id(FunctionDefinition.function_default_argument_values)
            for default_arg_name, default_arg_expr_value in c_def.evaluation_argument_default_values.items():
                print(
                    f"Parsing Function default argument expression (at {c_name} {default_arg_name}):",
                    default_args_location_id,
                    default_arg_expr_value,
                    sep="\n",
                )
                parsed_default_value_expr = parse_expression(
                    default_arg_expr_value,
                    c.evaluation_argument_types[default_arg_name],
                    FunctionArgumentProvenance.ANY,
                    FunctionArgumentAccessor.GET,
                    c.template_context,
                    context.expression_parser_validator,
                    default_args_location_id,
                )
                if not parsed_default_value_expr.is_valid:
                    assert isinstance(parsed_default_value_expr.value, IllFormedExpression)
                    raise CHSemanticError(
                        f"Invalid expression: expected {parsed_default_value_expr.required_expression_type}, "
                        f"{parsed_default_value_expr.required_provenance_type}, "
                        f"{parsed_default_value_expr.required_access_type};"
                        f"\n\tgot {parsed_default_value_expr.unparsed}!"
                        f"\n\t\tReason: {parsed_default_value_expr.value.reason}",
                        location_id=default_args_location_id,
                        part=PathPart.VALUE,
                    )
                all_default_argument_values[default_arg_name] = parsed_default_value_expr

        # collect default arguments of parents as well and set them in the model's Function data!
        for p_name, p_model in c.parents.items():
            if p_name not in context.ch.functions:
                continue
            assert isinstance(p_model, FunctionData)
            for default_arg_name, default_arg_expr in p_model.evaluation_argument_default_value.items():
                if default_arg_name not in all_default_argument_values:
                    all_default_argument_values[default_arg_name] = default_arg_expr
        c.evaluation_argument_default_value = frozendict(all_default_argument_values)

        context.instantiation_schema_validator.clear_identifier_where_types_are_defined()
        constraint_validator.update_existing_template_variables(set())
        context.reset_template_context()
        context.pop_last_variable_stack_frame()

    # First process all default_expressions in the instantiation
    for c_name, c in context.model.value_domains.items():
        context.set_template_context(c.template_context)
        constraint_validator = context.template_constraint_formula_validator
        constraint_validator.update_existing_template_variables(set(context.template_context.variables))
        context.instantiation_schema_validator.set_identifier_where_types_are_defined(c.name)

        # ``c.instantiation`` is a tuple, so if something is modified here, the whole tuple should be modified...
        for instantiation_constraint, instantiation_schema in c.instantiation:
            for schema_node in instantiation_schema.walk():
                if not schema_node.has_default:
                    continue
                location_of_default = schema_node.location_id + ["default"]
                print(
                    f"Parsing default instantiation expression (at {c_name} and {instantiation_constraint}):",
                    location_of_default,
                    schema_node.default_expr,
                    sep="\n",
                )
                parsed_default_value_expr = parse_expression(
                    schema_node.default_expr,
                    schema_node.custom_type,
                    FunctionArgumentProvenance.ANY,
                    FunctionArgumentAccessor.GET,
                    c.template_context,
                    context.expression_parser_validator,
                    location_of_default,
                )
                if not parsed_default_value_expr.is_valid:
                    assert isinstance(parsed_default_value_expr.value, IllFormedExpression)
                    raise CHSemanticError(
                        f"Invalid expression: expected {parsed_default_value_expr.required_expression_type}, "
                        f"{parsed_default_value_expr.required_provenance_type}, "
                        f"{parsed_default_value_expr.required_access_type};"
                        f"\n\tgot {parsed_default_value_expr.unparsed}!"
                        f"\n\t\tReason: {parsed_default_value_expr.value.reason}",
                        location_id=location_of_default,
                        part=PathPart.VALUE,
                    )
                schema_node.default_expr = parsed_default_value_expr

        context.instantiation_schema_validator.clear_identifier_where_types_are_defined()
        constraint_validator.update_existing_template_variables(set())
        context.reset_template_context()

    # Second, process template types
    for c_name, c in context.ch.concepts.items():
        if isinstance(c, HiddenImplementationDefinition):
            check_expressions_in_hidden_implementation_definition(c, context.model.value_domains[c_name], context)
    # Then, check types in the Concept Hierarchy (property default values, Function argument default values, etc.)
    for c_name, c in context.ch.concepts.items():
        if isinstance(c, DomainConceptDefinition):
            check_expressions_in_domain_concept_definition(c, context.model.domain_concepts[c_name], context)
        if isinstance(c, ValueDomainDefinition):
            check_expressions_in_value_domain_definition(c, context.model.value_domains[c_name], context)
        if isinstance(c, FunctionDefinition):
            check_expressions_in_function_definition(c, context.model.functions[c_name], context)
