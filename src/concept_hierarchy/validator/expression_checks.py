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
from concept_hierarchy.data.expressions.subexpressions import IllFormedExpression, Variable
from concept_hierarchy.data.parsers.expression_parser import get_expression_type, parse_expression
from concept_hierarchy.data.types.concept_hierarchy_types import TypeValue, frozendict
from concept_hierarchy.definitions.concept_definition_domain_concept import DomainConceptDefinition
from concept_hierarchy.definitions.concept_definition_functions import FunctionDefinition
from concept_hierarchy.definitions.concept_definition_hidden_implementation import HiddenImplementationDefinition
from concept_hierarchy.definitions.concept_definition_value_domain import ValueDomainDefinition
from concept_hierarchy.errors import CHSemanticError, PathPart


def invalid_expression_error(expr: Expression, location_id) -> CHSemanticError:
    """
    The error for an expression that failed to parse, carrying its explanation trace as ``causes``.

    Without the trace this says only that the value did not match the expected type; the trace is what
    names the alternatives that were tried, the instantiation constraint groups that were tested, and the
    schema errors of the one that matched.
    """
    assert isinstance(expr.value, IllFormedExpression)
    error = CHSemanticError(
        f"Invalid expression: expected {expr.required_expression_type}, "
        f"{expr.required_provenance_type}, "
        f"{expr.required_access_type};"
        f"\n\tgot {expr.unparsed}!"
        f"\n\t\tReason: {expr.value.reason}",
        location_id=location_id,
        part=PathPart.VALUE,
    )
    error.causes.extend(expr.value.explanation_causes(location_id))
    return error


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


def init_expressions(context: ConceptHierarchyContext):
    """
    This initializes the type of all variables, but does not parse its definition expressions.
    This also initializes the dependencies between Function default argument expressions,
    but their expressions also must be rechecked (like the variable-expressions)
    to ensure that there is no remaining unprocessed default-circular Function evaluation expression.

    After this function, the VariableContext will have the bottom-most stack frame populated with the global variables.

    :param context: the Concept Hierarchy that is analyzed, i.e. for which the expressions are to be initialized.
    :return:
    """

    # First, process the type of global variables; this doesn't process the expression value!
    # It processes just the type so the variable can be used/registered!
    # ``ch.instances`` holds canonical entries only -- an alias is a second *name* for one of these, not a
    # second variable -- so every entry here is independent and no resolution order is needed.
    value_domain_type = None
    global_variable_context: dict[str, TypeValue] = {}
    for global_var_name, global_var_data in context.ch.instances.items():
        # initialization
        if value_domain_type is None:
            value_domain_type = context.expression_parser_validator.create_instantiated_type(
                "ValueDomain", global_var_data.definition_location_id
            )
        global_var_model = GlobalVariableData(global_var_name)
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

    context.push_new_variable_stack_frame(VariableStackFrame(global_variable_context))

    # Then, process the expressions for Function default arguments; this allows creating the dependencies between the
    #   default Function argument values; which allows determining whether a Function evaluation/composition is
    #   well-formed (if all needed arguments are supplied and if there's no circular dependency in Function arguments)
    # - first process the expressions
    # - then check on which variables do the default values depend on
    # - create the dependency graph between default values; this must be checked at every Function evaluation in all
    #   future expressions!
    for c_name, c in context.model.functions.items():
        context.set_template_context(c.template_context)
        context.instantiation_schema_validator.set_identifier_where_types_are_defined(c.name)
        context.push_new_variable_stack_frame(VariableStackFrame())

        c_def = context.ch.concepts.get(c_name)
        assert isinstance(c_def, FunctionDefinition)

        all_default_argument_values: dict[str, Expression] = {}

        # add the other Function arguments as variables to the variable context!
        for arg_name, arg_type in c.evaluation_argument_types.items():
            context.add_new_variable(arg_name, arg_type)

        # process the default argument expressions of this concept
        default_argument_dependencies: dict[str, set[str]] = {}
        if c_def.has_location_of(FunctionDefinition.function_default_argument_values):
            default_args_location_id = c_def.location_id(FunctionDefinition.function_default_argument_values)
            for default_arg_name, default_arg_expr_value in c_def.evaluation_argument_default_values.items():
                print(
                    f"Parsing Function default argument expression (at {c_name} {default_arg_name} "
                    f"of type {c.evaluation_argument_types[default_arg_name]}):",
                    default_args_location_id + [default_arg_name],
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
                    default_args_location_id + [default_arg_name],
                    parse_template_expressions_without_type_checks=True,
                )
                if not parsed_default_value_expr.is_valid:
                    assert isinstance(parsed_default_value_expr.value, IllFormedExpression)
                    raise invalid_expression_error(
                        parsed_default_value_expr, default_args_location_id + [default_arg_name]
                    )
                assert default_arg_name not in default_argument_dependencies
                default_argument_dependencies[default_arg_name] = set()
                for expr in parsed_default_value_expr.all_subexpressions(Variable):
                    assert isinstance(expr.value, Variable)
                    if expr.value.variable_name in c.evaluation_argument_types:
                        default_argument_dependencies[default_arg_name].add(expr.value.variable_name)
                all_default_argument_values[default_arg_name] = parsed_default_value_expr

        # collect default arguments of parents as well and set them in the model's Function data!
        for p_name, p_model in c.parents.items():
            if p_name not in context.ch.functions:
                continue
            assert isinstance(p_model, FunctionData)
            for default_arg_name, default_arg_expr in p_model.evaluation_argument_default_value_expressions.items():
                if default_arg_name not in all_default_argument_values:
                    all_default_argument_values[default_arg_name] = default_arg_expr
                    assert default_arg_name not in default_argument_dependencies
                    default_argument_dependencies[default_arg_name] = p_model.default_argument_dependencies.get(
                        default_arg_name, set()
                    )
        c.evaluation_argument_default_value_expressions = frozendict(all_default_argument_values)
        c.default_argument_dependencies = frozendict(
            {x: frozenset(y) for x, y in default_argument_dependencies.items()}
        )

        context.instantiation_schema_validator.clear_identifier_where_types_are_defined()
        context.reset_template_context()
        context.pop_last_variable_stack_frame()


def check_expressions_in_concept_hierarchy(context: ConceptHierarchyContext):
    context.set_template_context(TemplateContext())
    context.set_variable_context(VariableContext([]))

    init_expressions(context)

    # 1. process global variable expressions (aliases are already processed; process expressions of canonical variables)
    global_template_context = TemplateContext()
    context.set_template_context(global_template_context)
    for global_variable_name, global_variable_definition in context.ch.instances.items():
        global_variable = context.model.instances[global_variable_name]
        definition_value = global_variable_definition.value
        expression_location = global_variable_definition.location_of(global_variable_name)
        print(f"Parsing expression of global variable {global_variable_name}: {definition_value}")
        parsed_expr = parse_expression(
            definition_value,
            global_variable.value_type,
            FunctionArgumentProvenance.ANY,
            FunctionArgumentAccessor.GET,
            global_template_context,
            context.expression_parser_validator,
            expression_location,
        )
        if not parsed_expr.is_valid:
            assert isinstance(parsed_expr.value, IllFormedExpression)
            raise invalid_expression_error(parsed_expr, expression_location)
        global_variable.value = parsed_expr
    context.reset_template_context()

    # First process all default_expressions in the instantiation
    for c_name, c in context.model.value_domains.items():
        context.set_template_context(c.template_context)
        context.instantiation_schema_validator.set_identifier_where_types_are_defined(c.name)
        context.push_new_variable_stack_frame(VariableStackFrame())

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
                parsed_expr = parse_expression(
                    schema_node.default_expr,
                    schema_node.custom_type,
                    schema_node.provenance,
                    FunctionArgumentAccessor.GET,
                    c.template_context,
                    context.expression_parser_validator,
                    location_of_default,
                )
                if not parsed_expr.is_valid:
                    assert isinstance(parsed_expr.value, IllFormedExpression)
                    raise invalid_expression_error(parsed_expr, location_of_default)
                # This stores the parsed/processed default_expr in custom nodes.
                schema_node.parsed_default_expr = parsed_expr

        context.pop_last_variable_stack_frame()
        context.instantiation_schema_validator.clear_identifier_where_types_are_defined()
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
