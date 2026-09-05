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

from typing import Iterator

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
from concept_hierarchy.data.expressions.subexpressions import (
    FunctionEvaluation,
    IllFormedExpression,
    Variable,
    VerifiedTemplateDependentExpression,
)
from concept_hierarchy.data.parsers.expression_parser import get_expression_type, parse_expression
from concept_hierarchy.data.types.concept_hierarchy_types import TypeValue, frozendict
from concept_hierarchy.definitions.concept_definition_domain_concept import DomainConceptDefinition
from concept_hierarchy.definitions.concept_definition_functions import FunctionDefinition
from concept_hierarchy.definitions.concept_definition_hidden_implementation import HiddenImplementationDefinition
from concept_hierarchy.definitions.concept_definition_value_domain import ValueDomainDefinition
from concept_hierarchy.errors import CHSemanticError, LocationId, PathPart


def ill_formed_parts(expr: Expression) -> tuple[IllFormedExpression, ...]:
    """
    Every :class:`IllFormedExpression` that makes ``expr`` invalid, whatever shape it arrived in.

    There are two shapes, and a caller that knows only the first crashes on the second. At a site of
    **ground** type the parser settles on one alternative, so an invalid expression *is* the
    `IllFormedExpression`. At a site whose type is still **template dependent** it cannot settle --
    ``ensure_expression_invariant`` answers ``None``, so every applicable alternative is tried and they are
    collected into a `VerifiedTemplateDependentExpression`, which is invalid when *any* of them failed.

    Returns ``()`` for an expression that is invalid with nothing to point at: a template-dependent
    expression with no possible alternative at all, which `VerifiedTemplateDependentExpression.is_valid`
    also rejects.
    """
    value = expr.value
    if isinstance(value, IllFormedExpression):
        return (value,)
    if isinstance(value, VerifiedTemplateDependentExpression):
        return tuple(possible for possible in value.possible_expressions if isinstance(possible, IllFormedExpression))
    return ()


def invalid_expression_error(expr: Expression, location_id) -> CHSemanticError:
    """
    The error for an expression that failed to parse, carrying its explanation trace as ``causes``.

    Without the trace this says only that the value did not match the expected type; the trace is what
    names the alternatives that were tried, the instantiation constraint groups that were tested, and the
    schema errors of the one that matched.
    """
    assert not expr.is_valid, expr
    parts = ill_formed_parts(expr)
    reason = "; ".join(part.reason for part in parts) if parts else "no alternative could be parsed for this expression"
    error = CHSemanticError(
        f"Invalid expression: expected {expr.required_expression_type}, "
        f"{expr.required_provenance_type}, "
        f"{expr.required_access_type};"
        f"\n\tgot {expr.unparsed}!"
        f"\n\t\tReason: {reason}",
        location_id=location_id,
        part=PathPart.VALUE,
    )
    for part in parts:
        error.causes.extend(part.explanation_causes(location_id))
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
    # Every Function's parsed defaults are readable from the outset, before any of them are parsed.
    # Parsing one Function's default can reach an evaluation of *any* Function -- of itself, as when
    # `MakeT1`'s default instantiates a `T1` whose schema evaluates `MakeT1` again, or of one this loop has
    # not come to yet, as when a parent's default evaluates its own child.
    # **The grounding step asks for the defaults of whatever Function it meets,
    # so leaving the field unset makes that a crash.**
    # Empty is the honest answer during that window, and `get_function_argument_default_source` reads the declarations
    # rather than this, so the edge itself is not lost.
    for c in context.model.functions.values():
        c.evaluation_argument_default_value_expressions = frozendict()

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
                # A default this loop has not reached yet can still have been parsed already: an earlier
                # Function's default may evaluate this one, and grounding that call site parses it (see
                # `_offer_grounding_as_the_declarations_parse`). It is handed back here only when the two
                # parses cannot differ, so reparsing the source would rebuild the very same tree.
                parsed_default_value_expr = context.expression_parser_validator.get_parsed_function_argument_default(
                    c_name, default_arg_name
                )
                print(
                    f"{'Reusing the grounding of the' if parsed_default_value_expr is not None else 'Parsing the'} "
                    f"Function default argument expression (at {c_name} {default_arg_name} "
                    f"of type {c.evaluation_argument_types[default_arg_name]}):",
                    default_args_location_id + [default_arg_name],
                    default_arg_expr_value,
                    sep="\n",
                )
                if parsed_default_value_expr is None:
                    parsed_default_value_expr = parse_expression(
                        default_arg_expr_value,
                        c.evaluation_argument_types[default_arg_name],
                        FunctionArgumentProvenance.ANY,
                        FunctionArgumentAccessor.GET,
                        context.expression_parser_validator,
                        default_args_location_id + [default_arg_name],
                        parse_template_expressions_without_type_checks=True,
                    )
                else:
                    assert parsed_default_value_expr.is_valid
                if not parsed_default_value_expr.is_valid:
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

        context.pop_last_variable_stack_frame()
        context.instantiation_schema_validator.clear_identifier_where_types_are_defined()
        context.reset_template_context()

    # Then, after the Function default argument dependencies are registered,
    # process the default expressions in ValueDomain instantiations.
    # Process here in the initialization, so that the expressions for global variables, and Function default arguments
    # have the default instantiations completely stored once those global-variable-expressions are checked again.
    # The issue here is that default instantiations inside default instantiations are possible;
    # and I think the only way to check if all are processed is to iterate over all created ParsedCustomValues
    # and check if their expression is None instead of an expression (if the value is valid)!
    # There is a dependency between instantiation schema expressions:
    #  the default expressions may depend on other default expressions which may form a cycle of dependencies!
    #  That cycle (on other CHSchemaNodes that must use their default expressions to instantiate the value
    #  given by this schema's default expression) must be detected and reported.
    #  However, detection may only be possible once a ground type application is given
    #  because the schemas can depend on their template types...
    #  But this anyway means, that default expressions in instantiations are not always resolvable at this point;
    #  only at runtime, when an ground type application (instantiation) is given...
    for c_name, c in context.model.value_domains.items():
        context.set_template_context(c.template_context)
        context.instantiation_schema_validator.set_identifier_where_types_are_defined(c.name)
        context.push_new_variable_stack_frame(VariableStackFrame())

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
                    context.expression_parser_validator,
                    location_of_default,
                    parse_template_expressions_without_type_checks=True,
                )
                if not parsed_expr.is_valid:
                    raise invalid_expression_error(parsed_expr, location_of_default)
                # This stores the parsed/processed default_expr in custom nodes.
                schema_node.parsed_default_expr = parsed_expr

        context.pop_last_variable_stack_frame()
        context.instantiation_schema_validator.clear_identifier_where_types_are_defined()
        context.reset_template_context()


def expression_dependencies(expression: Expression) -> Iterator[Expression]:
    """
    Everything ``expression`` really depends on: its sub-expressions, **plus** the defaults a Function
    evaluation applied for the arguments its call site left out.

    `FunctionEvaluation.applied_defaults` is deliberately not yielded from `get_subexpressions` -- putting
    it there fabricates edges in `FunctionData.default_argument_dependencies`, whose scan filters the names
    it finds against one Function's arguments and would import a nested evaluation's (see the field's own
    documentation). A traversal that wants them therefore has to add them itself, and this is that
    traversal.
    """
    yield from expression.value.get_subexpressions()
    if isinstance(expression.value, FunctionEvaluation):
        yield from expression.value.applied_defaults.values()


def is_a_custom_function_global_variable(context: ConceptHierarchyContext, global_variable_name: str) -> bool:
    """
    Whether a global variable holds a `CustomFunction`, which is the one exception to §1 below.

    Such a variable's value is a *procedure*, and a procedure that names the variable it is stored in is
    ordinary recursion -- `animal_kingdom.json`'s ``factorial`` is exactly that -- rather than a value whose
    initialisation needs itself. These are never put on the resolution path and never descended into, so the
    check can only fail to fire on them, never fire wrongly.
    """
    validator = context.expression_parser_validator
    custom_function = DomainConceptDefinition.default_value_domain_type_of_domain_concept_functions
    if not validator.is_concept(custom_function):
        # `CustomFunction` is a concept of the standard prelude rather than of the language, so a hierarchy
        # is free not to define it -- and then no global can hold one.
        return False
    custom_function_type = validator.create_instantiated_type(custom_function, LocationId())
    value_type = context.model.instances[global_variable_name].value_type
    return validator.is_a_subtype_of_b(value_type, custom_function_type, LocationId())


def find_recursive_global_variable_initialisation(
    context: ConceptHierarchyContext, expression: Expression, chain: list[str]
) -> list[str] | None:
    """
    Walk a global variable's value looking for a reference back to one whose value ``chain`` is determining.

    ``chain`` is both the answer being built and the resolution path: a global is on the path exactly while
    it is an entry of it, so there is no second structure to keep in step with it and nothing to clear when
    a global's turn ends. It starts as the one global `check_global_variable_expressions` is resolving.

    The reference is found by `expression_dependencies`, which is what makes this see the cases neither
    existing graph does: the alias graph has an edge only where a variable's value *is* a bare name, and the
    sibling-argument graph has one only between arguments of one Function -- so ``v = {"Loopy": {}}`` with
    ``Loopy.x`` defaulting to ``v`` is an edge of neither, and is a genuine infinite regress.

    A reference closes the cycle when it resolved **at the global scope** and its **canonical** name is on
    the path. Both halves are needed: a Function argument or a nested call can introduce the same name in a
    frame above the globals, and an alias is a second name for one variable. Neither is spelled by the name
    alone, which is why `Variable.scope_index` is recorded at every usage and why the name is canonicalized here.

    A reference to some *other* global is not itself a cycle, but the cycle may run through it, so its value
    is walked in turn with its name appended -- which is what catches a cycle spanning two globals whatever
    order they are declared in. A global this pass has not parsed yet has no value to walk; it walks the
    other way round when its own turn comes, and by then both are parsed. The walk terminates because a
    global already on the chain is returned rather than descended into, so no name is ever appended twice.

    :return: the chain of global variable names closing the cycle, its last entry being the repeated one,
        or ``None`` if the value is well-founded.
    """
    to_visit = [expression]
    existing_chain_set: set[str] = set(chain)
    """optimization variable: process once to speed up the `in existing_chain_set` check"""
    while to_visit:
        current = to_visit.pop()
        value = current.value
        if isinstance(value, Variable) and value.is_global_variable:
            referenced_name = context.ch.canonical_variable_name(value.variable_name)
            if referenced_name in existing_chain_set:
                return chain + [referenced_name]
            referenced = context.model.instances[referenced_name]
            if referenced.is_value_initialized() and not is_a_custom_function_global_variable(context, referenced_name):
                found = find_recursive_global_variable_initialisation(
                    context, referenced.value, chain + [referenced_name]
                )
                if found is not None:
                    return found
        to_visit.extend(expression_dependencies(current))
    return None


def check_global_variable_expressions(context: ConceptHierarchyContext) -> None:
    """
    Parse the value expression of every global variable, and reject one that can not be determined.

    Aliases need nothing here: an alias is a second *name* for a variable rather than a second variable, so
    `ch.instances` holds only canonical entries and each is parsed once.

    Two things are checked beyond the parse succeeding, and neither subsumes the other:

    * **the value is decided** -- `is_value_template_dependent` is ``False`` and `is_fully_parsed` is
      ``True``. There is no application still to come that could settle it, since the parse happens in the
      empty template context, and no "materialise it, or supply the key instead" alternative of the kind
      that makes an undecided instantiation *default* legitimate. Vacuous on every hierarchy today, and
      cheap: what it buys is that a later change can not silently store a global nothing can read.
    * **the value does not need itself** -- `find_recursive_global_variable_initialisation`. The invariant
      above does not see this, because the recursion runs through an argument the call site never *wrote*
      and `FunctionEvaluation.is_fully_parsed` quantifies over the ones it did: ``all([])`` is ``True``.

    The resolution path lives in the ``chain`` handed to that walk and nowhere else -- there is no state
    here that outlives one global, and nothing is written to the validator, which is an interface another
    implementation is free to satisfy differently.
    """
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
            context.expression_parser_validator,
            expression_location,
        )
        if not parsed_expr.is_valid:
            raise invalid_expression_error(parsed_expr, expression_location)
        if parsed_expr.is_value_template_dependent or not parsed_expr.is_fully_parsed:
            raise CHSemanticError(
                f"The value of the global variable {global_variable_name!r} is not decided: it is "
                f"{'template dependent' if parsed_expr.is_value_template_dependent else 'only partially parsed'}"
                f", and nothing later can decide it."
                f"\n\tgot {definition_value!r}",
                location_id=expression_location,
                part=PathPart.VALUE,
            )
        # A `CustomFunction` global is the one kind whose value may name itself, so it is the one kind that
        # never goes on the path -- see `is_a_custom_function_global_variable`.
        if not is_a_custom_function_global_variable(context, global_variable_name):
            cycle = find_recursive_global_variable_initialisation(context, parsed_expr, [global_variable_name])
            if cycle is not None:
                raise CHSemanticError(
                    f"The global variable {cycle[-1]!r} can never be initialised: determining its value "
                    f"requires determining it again."
                    f"\n\tThe dependency runs {' -> '.join(cycle)}.",
                    location_id=expression_location,
                    part=PathPart.VALUE,
                )
        global_variable.value = parsed_expr
    context.reset_template_context()


def check_expressions_in_concept_hierarchy(context: ConceptHierarchyContext):
    context.set_template_context(TemplateContext())
    context.set_variable_context(VariableContext([]))

    init_expressions(context)  # this mutates context to contain in its variable context all the global variables

    # 1. process global variable expressions
    check_global_variable_expressions(context)

    # 2. reprocess Function default argument expressions
    for c_name, c in context.model.functions.items():
        context.set_template_context(c.template_context)
        context.instantiation_schema_validator.set_identifier_where_types_are_defined(c.name)
        context.push_new_variable_stack_frame(VariableStackFrame())

        c_def = context.ch.concepts.get(c_name)
        assert isinstance(c_def, FunctionDefinition)

        # add the other Function arguments as variables to the variable context!
        for arg_name, arg_type in c.evaluation_argument_types.items():
            context.add_new_variable(arg_name, arg_type)

        # Reparse the default argument expressions of this concept, now that *every* Function has its
        # `default_argument_dependencies`.
        #
        # This is not redundant with the identical parse in `init_expressions`, and the difference is ordering.
        # That loop fills the dependencies one Function at a time, so a default parsed early can
        # contain an evaluation of a Function the loop has not reached yet --
        # `get_default_argument_dependencies` answers `None` for it, and the acyclicity check is *skipped*
        # rather than failed. On this pass every answer is there, so the check actually runs.
        # Measured: removing this pass lets a mutually cyclic pair of default arguments through, when the Function
        # declaring them is written after the Function whose default evaluates it.
        if c_def.has_location_of(FunctionDefinition.function_default_argument_values):
            default_args_location_id = c_def.location_id(FunctionDefinition.function_default_argument_values)
            for default_arg_name, default_arg_expr_value in c_def.evaluation_argument_default_values.items():
                default_arg_location_id = default_args_location_id + [default_arg_name]
                parsed_default_value_expr = parse_expression(
                    default_arg_expr_value,
                    c.evaluation_argument_types[default_arg_name],
                    FunctionArgumentProvenance.ANY,
                    FunctionArgumentAccessor.GET,
                    context.expression_parser_validator,
                    default_arg_location_id,
                    parse_template_expressions_without_type_checks=True,
                )
                if not parsed_default_value_expr.is_valid:
                    raise invalid_expression_error(parsed_default_value_expr, default_arg_location_id)

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
