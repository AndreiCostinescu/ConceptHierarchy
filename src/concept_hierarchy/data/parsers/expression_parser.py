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

"""Parser of Expressions"""

from abc import ABC, abstractmethod
from collections import deque
from typing import Callable

from frozendict import frozendict

from concept_hierarchy.data.contexts.template_context import TemplateContext
from concept_hierarchy.data.expressions.expression import Expression, ExpressionValue
from concept_hierarchy.data.expressions.expression_utils import (
    FunctionArgumentAccessor,
    FunctionArgumentProvenance,
    FunctionResultAccessor,
    ValueDomainArgumentProvenance,
)
from concept_hierarchy.data.expressions.instantiated_value import ParsedValue
from concept_hierarchy.data.expressions.subexpressions import (
    FunctionEvaluation,
    IllFormedExpression,
    InstancePropertyChain,
    InstExpression,
    LiteralTemplateVariableValue,
    NarrowExpression,
    TemplateDependentExpression,
    Variable,
    VariableWithTemplateType,
)
from concept_hierarchy.data.jsonschema import CHSchemaNode
from concept_hierarchy.data.type_template_variables.constraint_formula import (
    ConstraintGroup,
    NonTypeTemplateConstraintFormula,
)
from concept_hierarchy.data.type_template_variables.template_substitution import substitute
from concept_hierarchy.data.types.concept_hierarchy_types import (
    TYPE_VALUE_IS_INSTANCE_CHECK,
    ConceptHierarchyTemplateArgument,
    ConceptHierarchyVariadicGroup,
    ExpandedVariadicTemplateVariable,
    InstantiatedType,
    LiteralValue,
    TemplateDependent,
    TemplateDependentType,
    TypeValue,
)
from concept_hierarchy.data.validators.template_argument_constraints_validator import TypeTemplateInstantiationValidator
from concept_hierarchy.definitions.concept_definition_value_domain import ValueDomainDefinition
from concept_hierarchy.errors import CHSemanticError, ConceptHierarchyError, LocationId, PathPart
from concept_hierarchy.utils import get_items_of_single_entry_dict


class ExpressionParserValidator(ABC):
    @abstractmethod
    def is_concept(self, candidate_concept_name: str) -> bool:
        pass

    @abstractmethod
    def is_template_variable(self, candidate_template_variable_name: str) -> bool:
        pass

    @abstractmethod
    def is_literal_template_variable(self, candidate_literal_template_variable_name: str) -> bool:
        pass

    @abstractmethod
    def get_literal_template_var_constraint(self, literal_template_variable_name: str) -> str:
        """Raises an error if the literal template variable name is not valid."""

    @abstractmethod
    def is_variable(self, candidate_variable_name: str) -> bool:
        pass

    @abstractmethod
    def get_variable_type(self, variable_name: str) -> TypeValue:
        pass

    @abstractmethod
    def is_type_abstract(self, candidate_type: InstantiatedType | TemplateDependentType) -> bool:
        pass

    @abstractmethod
    def is_a_subtype_of_b(self, a: InstantiatedType, b: InstantiatedType, location_id: LocationId) -> bool:
        pass

    @abstractmethod
    def create_instantiated_type(self, instantiated_type_name: str, location_id: LocationId) -> InstantiatedType:
        pass

    @abstractmethod
    def create_possibly_template_dependent_type(self, type_name: str) -> TypeValue:
        pass

    @abstractmethod
    def get_substituted_value_domain_instantiation_schema(
        self, type_name: InstantiatedType | TemplateDependentType
    ) -> TypeValue:
        pass

    @abstractmethod
    def get_substituted_function_interface(
        self, ch_type: InstantiatedType
    ) -> tuple[tuple[str, ...], set[str], InstantiatedType | None, ValueDomainArgumentProvenance | None]:
        pass

    @abstractmethod
    def get_properties_of_concepts(self, concepts: list[str]) -> dict[str, InstantiatedType]:
        pass

    @abstractmethod
    def get_type_of_instance_property(self, instance_type: InstantiatedType, prop_name: str) -> InstantiatedType:
        """
        This must raise a CHSemanticError if:
            - instance_type is not a subtype of InstanceBase
            - prop_name is not a property of the type represented by instance_type

        :param instance_type: the type which is to-be-checked that it is an instance type that has the property
        :param prop_name: the name of the property whose type is to be determined by the function
        :return: the InstantiatedType type of the property `prop_name` of `instance_type`
        """

    @abstractmethod
    def get_if_has_instantiation_schema(
        self, type_data: TypeValue
    ) -> tuple[tuple[ConstraintGroup, CHSchemaNode], ...] | None:
        pass

    @abstractmethod
    def validate_value_against_schema(
        self, schema: CHSchemaNode, value: object, location_id: LocationId
    ) -> ParsedValue:
        pass

    @abstractmethod
    def get_default_serialization_concept_name_for(self, json_value_type: str) -> str | None:
        pass

    @abstractmethod
    def get_function_return_interface(
        self, f_name
    ) -> tuple[TypeValue, FunctionResultAccessor, ValueDomainArgumentProvenance] | None:
        pass

    @abstractmethod
    def is_function_argument(self, f_name, f_arg_name) -> bool:
        pass

    @abstractmethod
    def get_function_arguments(self, f_name) -> set[str]:
        pass

    @abstractmethod
    def get_required_function_arguments(self, f_name) -> set[str]:
        pass

    @abstractmethod
    def get_function_argument_interface(
        self, f_name, f_arg_name
    ) -> tuple[TypeValue, FunctionArgumentAccessor, FunctionArgumentProvenance]:
        pass

    @abstractmethod
    def get_default_argument_dependencies(self, f_name) -> frozendict[str, frozenset[str]] | None:
        pass

    @abstractmethod
    def get_template_context(self, type_name_clean) -> TemplateContext:
        pass

    @abstractmethod
    def get_type_template_instantiation_validator(self) -> TypeTemplateInstantiationValidator:
        pass


def parse_expression(
    json_value: object,
    expr_type: TypeValue,
    expr_provenance: FunctionArgumentProvenance,
    expr_access: FunctionArgumentAccessor,
    expr_template_context: TemplateContext,
    validator: ExpressionParserValidator,
    location_id: LocationId,
) -> Expression:
    expr_candidate_value = _parse_syntax_of_expression(
        json_value, expr_type, expr_template_context, validator, location_id
    )
    is_strict_subtype = expr_candidate_value.is_strict_subtype
    is_addressable = isinstance(expr_candidate_value, Variable)
    if not is_addressable and isinstance(expr_candidate_value, FunctionEvaluation):
        is_addressable = expr_candidate_value.is_result_addressable
    if expr_provenance != FunctionArgumentProvenance.ANY and not is_addressable:
        expr_candidate_value = IllFormedExpression(
            f"Provenance violation: required {expr_provenance}, got {is_addressable}"
        )
    elif expr_access != FunctionArgumentAccessor.GET and is_strict_subtype:
        expr_candidate_value = IllFormedExpression(
            f"Access violation: required access {expr_access}, required type {expr_type}, value type "
            f"{expr_candidate_value}, with is_strict_subtype {is_strict_subtype}!"
        )

    expression = Expression(expr_type, expr_provenance, expr_access, json_value, expr_candidate_value)
    return expression

    # TODO: check the types // semantic of the expression:
    #  - check expr_ref, expr_mod, is_strict_subtype!
    #  - check that "isFunctionEvaluation" is used correctly
    #  - check that Function result types match the expected expression type
    #  -- subexpressions (i.e. the evaluation arguments) should already be sub-checked in the syntax-above!)
    #  - check that ValueDomain instantiations are subtypes of the expected expression type
    #  -- subexpressions thereof were already checked when they were parsed
    #  - check that the variables (literal template variables, code variables, and instance prop chains)
    #    match the expected expression type
    #  -- for literal template variables check that the literal constraint type is a registered defaultSerialization
    #     somewhere; if it is not registered, then it can't be used!
    #     if it is registered, interpret the template variable value as the type that registers
    raise NotImplementedError


def get_expression_type(
    json_value: object,
    expr_type: TypeValue,
    expr_template_context: TemplateContext,
    validator: ExpressionParserValidator,
    location_id: LocationId,
) -> TypeValue | None:
    return _parse_syntax_of_expression(
        json_value, expr_type, expr_template_context, validator, location_id, recursively_parse=False
    ).value_type


def _parse_syntax_of_expression(
    json_value: object,
    expr_type: TypeValue,
    expr_template_context: TemplateContext,
    validator: ExpressionParserValidator,
    location_id: LocationId,
    recursively_parse: bool = True,
) -> ExpressionValue:
    assert isinstance(expr_type, TYPE_VALUE_IS_INSTANCE_CHECK)
    if isinstance(expr_type, (ConceptHierarchyVariadicGroup, LiteralValue, ExpandedVariadicTemplateVariable)):
        raise RuntimeError(f"Can't parse an expression of type {expr_type}")

    if not isinstance(expr_type, InstantiatedType):
        return TemplateDependentExpression()
        # TODO: how do I select the correct instantiation formula from the list of template-constraints?
        #   - if the expression type is fully instantiated => verify constraints => check first matching instantiation
        #   - if the expression type depends on templates:
        #       - if there is a single instantiation (i.e. no template-dependent instantiation) => validate against it
        #       - if there are multiple instantiations
        #           - if at least one instantiation matches the constraints => do not verify the constraints yet
        #             (add the list of possible instantiation schemas, the
        #           - if no instantiation could ever match the constraints => proceed as if there is no instantiation
        #             defined, but signal that there are instantiation schemas that do not match the expected type
        # TODO: validate literal formula; if formula is not validated -> raise CHSemanticError

    is_function_evaluation, is_function_evaluation_present, len_content_keys = True, False, None
    # compute the amount of **content keys** in the JSON object
    if isinstance(json_value, dict):
        len_content_keys = len(json_value)
        is_function_evaluation_present = "isFunctionEvaluation" in json_value
        if is_function_evaluation_present:
            len_content_keys -= 1
            # remove the `"isFunctionEvaluation"` key from `json_value` ONLY in a Narrow/FEval expression type!
            is_function_evaluation = json_value["isFunctionEvaluation"]

    def ensure_unmodified_json_value(
        _value: object, _is_function_evaluation_present: bool, _is_function_evaluation_value: bool
    ) -> None:
        if isinstance(_value, dict) and _is_function_evaluation_present and "isFunctionEvaluation" not in _value:
            _value["isFunctionEvaluation"] = _is_function_evaluation_value

    try:
        expr_value_res = _parse_syntax_of_expression_with_instantiated_type(
            json_value,
            expr_type,
            expr_template_context,
            validator,
            location_id,
            recursively_parse,
            is_function_evaluation,
            is_function_evaluation_present,
            len_content_keys,
            ensure_unmodified_json_value,
        )
        ensure_unmodified_json_value(json_value, is_function_evaluation_present, is_function_evaluation)
        return expr_value_res
    except ConceptHierarchyError as e:
        ensure_unmodified_json_value(json_value, is_function_evaluation_present, is_function_evaluation)
        raise e


def _parse_syntax_of_expression_with_instantiated_type(
    json_value: object,
    expr_type: TypeValue,
    expr_template_context: TemplateContext,
    validator: ExpressionParserValidator,
    location_id: LocationId,
    recursively_parse: bool,
    is_function_evaluation: bool,
    is_function_evaluation_present: bool,
    len_content_keys: int,
    ensure_unmodified_json_value: Callable[[object, bool, bool], None],
) -> ExpressionValue:
    function_composition_type = validator.create_instantiated_type("FunctionComposition", location_id)
    function_type = validator.create_instantiated_type("Function", location_id)

    # check Narrow and FEval expressions
    if len_content_keys == 1:
        assert isinstance(json_value, dict)
        if is_function_evaluation_present:
            json_value.pop("isFunctionEvaluation")
        function_evaluation = (
            not validator.is_a_subtype_of_b(expr_type, function_composition_type, location_id)
            and is_function_evaluation
        )
        key, value = get_items_of_single_entry_dict(json_value)
        key_type = validator.create_instantiated_type(key, location_id)
        if validator.is_type_abstract(key_type):
            raise CHSemanticError(
                f"{key_type} is an abstract type! Thus, it can not be used in expression values "
                f"(neither as FEval nor as Narrow expressions)!",
                location_id=location_id + [key],
                part=PathPart.KEY,
            )
        is_function_subtype = validator.is_a_subtype_of_b(key_type, function_type, location_id)
        if function_evaluation and is_function_subtype:
            function_return = validator.get_function_return_interface(key_type.clean_name)
            if function_return is None:
                raise CHSemanticError(
                    f"Function {key} does not return anything; expected a return type of {expr_type}!",
                    location_id=location_id + [key],
                    part=PathPart.KEY,
                )
            function_return_type, function_return_access, function_return_provenance = function_return
            is_result_addressable = function_return_provenance == ValueDomainArgumentProvenance.ADDR

            # create substitution mapping
            f_substitution_mapping: dict[str, ConceptHierarchyTemplateArgument] = {}
            f_template_context: TemplateContext = validator.get_template_context(key_type.clean_name)
            for t_arg_name, t_arg_val in zip(f_template_context.variables, key_type.template_arguments):
                f_substitution_mapping[t_arg_name] = t_arg_val
            # substitute `function_return_type` with template instantiation of Function
            function_return_type, _ = substitute(
                function_return_type,
                f_substitution_mapping,
                expr_template_context,
                f_template_context,
                validator.get_type_template_instantiation_validator(),
                location_id,
            )
            assert isinstance(function_return_type, TYPE_VALUE_IS_INSTANCE_CHECK)
            if not validator.is_a_subtype_of_b(function_return_type, expr_type, location_id):
                return IllFormedExpression(
                    f"Function result type {function_return_type} is not a subtype of {expr_type}"
                )
            f_args: dict[str, Expression] = {}
            if recursively_parse:
                # verify sub-expressions + make sure that the Function arguments are actually correct ones
                all_arguments = validator.get_function_arguments(key_type.clean_name)
                for f_arg_name, f_arg_expr_val in value.items():
                    if not validator.is_function_argument(key_type.clean_name, f_arg_name):
                        raise CHSemanticError(
                            f'Function {key} does not have the argument "{f_arg_name}"; only {all_arguments}',
                            location_id=location_id,
                            part=PathPart.KEY,
                        )
                    f_arg_type, f_arg_access, f_arg_prov = validator.get_function_argument_interface(
                        key_type.clean_name, f_arg_name
                    )
                    # substitute `f_arg_type` with template instantiation of Function
                    f_arg_type, _ = substitute(
                        f_arg_type,
                        f_substitution_mapping,
                        expr_template_context,
                        f_template_context,
                        validator.get_type_template_instantiation_validator(),
                        location_id,
                    )
                    assert isinstance(f_arg_type, TYPE_VALUE_IS_INSTANCE_CHECK)
                    arg_expr = parse_expression(
                        f_arg_expr_val,
                        f_arg_type,
                        f_arg_prov,
                        f_arg_access,
                        expr_template_context,
                        validator,
                        location_id + [key, f_arg_name],
                    )
                    if not arg_expr.is_valid:
                        assert isinstance(arg_expr.value, IllFormedExpression)
                        return IllFormedExpression(
                            f"{key} argument {f_arg_name}'s value {f_arg_expr_val} is invalid: {arg_expr.value.reason}"
                        )
                    f_args[f_arg_name] = arg_expr
                # verify required arguments are present
                missing_arguments: set[str] = set()
                required_arguments: set[str] = validator.get_required_function_arguments(key_type.clean_name)
                for required_arg in required_arguments:
                    if required_arg not in f_args:
                        missing_arguments.add(required_arg)
                if missing_arguments:
                    raise CHSemanticError(
                        f"Argument(s) {missing_arguments} are missing from the Function evaluation interface of {key}!",
                        location_id=location_id + [key],
                        part=PathPart.VALUE,
                    )
                # verify that the dependencies between the remaining default arguments are not cyclic
                supplied_arguments = set(f_args)
                unsupplied_arguments: set[str] = all_arguments - supplied_arguments
                default_argument_dependencies = validator.get_default_argument_dependencies(key_type.clean_name)
                if default_argument_dependencies is not None and not _validate_acyclic_default_argument_dependencies(
                    default_argument_dependencies, supplied_arguments
                ):
                    raise CHSemanticError(
                        f"The dependency graph between the remaining default arguments {unsupplied_arguments} of "
                        f"the Function evaluation of {key} is not acyclic!",
                        location_id=location_id + [key],
                        part=PathPart.VALUE,
                    )
            return FunctionEvaluation(
                key_type,
                function_return_type,
                f_args,
                is_result_addressable,
                function_return_type != expr_type,
            )
        elif validator.is_a_subtype_of_b(key_type, expr_type, location_id):
            if is_function_evaluation_present and not is_function_subtype:
                raise CHSemanticError(
                    f'Invalid use of the "isFunctionEvaluation" keyword at single-content-key object "{key_type}"!',
                    location_id=location_id + ["isFunctionEvaluation"],
                    part=PathPart.KEY,
                )
            if not recursively_parse:
                narrow_res = None
            else:
                # abstract Types do not have instantiation schemas
                narrow_res = _check_instantiation_schema(value, key_type, validator, location_id)
            if not recursively_parse or (narrow_res is not None and narrow_res.is_valid()):
                return NarrowExpression(narrow_res, key_type, key_type != expr_type)
        ensure_unmodified_json_value(json_value, is_function_evaluation_present, is_function_evaluation)
    # check Var expression
    if isinstance(json_value, str):
        # Prioritize variables over template variables if there is a name clash!
        if validator.is_variable(json_value):
            if validator.is_template_variable(json_value):
                print(
                    f'[CH Warning] Prioritize variable "{json_value}" over template variable "{json_value}" in '
                    f"expression {json_value!r}"
                )
            var_type = validator.get_variable_type(json_value)
            if isinstance(var_type, TemplateDependent):
                return VariableWithTemplateType(json_value, var_type)
            assert isinstance(var_type, InstantiatedType), f"{var_type} of type {str(type(var_type))}"
            if validator.is_a_subtype_of_b(var_type, expr_type, location_id):
                return Variable(json_value, var_type, var_type != expr_type)
            else:
                return IllFormedExpression(f"Type {var_type} of variable {json_value} is not a subtype of {expr_type}!")
        else:
            possible_instance_property_chain = json_value.split(".")
            if len(possible_instance_property_chain) > 1 and validator.is_variable(possible_instance_property_chain[0]):
                # Validate that the instance property chain is actually a property chain.
                types_in_property_chain = [validator.get_variable_type(possible_instance_property_chain[0])]
                for prop in possible_instance_property_chain[1:]:
                    # The below raises a CHSemanticError if:
                    #  - instance_type is not a subtype of InstanceBase
                    #  - prop_name is not a property of the type represented by instance_typ
                    prop_type = validator.get_type_of_instance_property(types_in_property_chain[-1], prop)
                    types_in_property_chain.append(prop_type)
                var_type = types_in_property_chain[-1]
                if validator.is_a_subtype_of_b(var_type, expr_type, location_id):
                    return InstancePropertyChain(
                        possible_instance_property_chain, types_in_property_chain, var_type != expr_type
                    )
                else:
                    return IllFormedExpression(
                        f"Type {var_type} of instance property chain {json_value} is not a subtype of {expr_type}!"
                    )
            elif validator.is_template_variable(json_value):
                if not validator.is_literal_template_variable(json_value):
                    raise CHSemanticError(
                        f"Can not use a type template variable in an expression as a variable. Found {json_value}",
                        location_id=location_id,
                    )
                # check to see whether the template parameter's literal type is a registered defaultSerialization!
                literal_constraint_type = validator.get_literal_template_var_constraint(json_value)
                assert literal_constraint_type in NonTypeTemplateConstraintFormula.ALL_CONSTRAINT_TYPES
                match literal_constraint_type:
                    case NonTypeTemplateConstraintFormula.BOOLEAN:
                        value_type_str = "boolean"
                    case NonTypeTemplateConstraintFormula.INTEGER:
                        value_type_str = "integer"
                    case NonTypeTemplateConstraintFormula.NUMBER:
                        value_type_str = "number"
                    case NonTypeTemplateConstraintFormula.STRING:
                        value_type_str = "string"
                    case _:
                        raise RuntimeError(
                            f'Unknown constraint type "{literal_constraint_type}" of template variable "{json_value}"'
                        )
                type_name_str = validator.get_default_serialization_concept_name_for(value_type_str)
                if type_name_str is not None:
                    ch_value_type = validator.create_instantiated_type(type_name_str, location_id)
                    assert ch_value_type is not None
                    if validator.is_a_subtype_of_b(ch_value_type, expr_type, location_id):
                        return LiteralTemplateVariableValue(json_value, ch_value_type, ch_value_type != expr_type)
                    else:
                        return IllFormedExpression("")
                else:
                    return IllFormedExpression(
                        f'The literal constraint "{literal_constraint_type}" of "{json_value}" does not have a matching'
                        f' registered "{ValueDomainDefinition.value_domain_default_serialization}" ({value_type_str})!'
                    )

    # check Inst expression (abstract Types do not have instantiation schemas)
    inst_res = _check_instantiation_schema(json_value, expr_type, validator, location_id)
    if inst_res is not None and inst_res.is_valid():
        return InstExpression(inst_res, expr_type, True)
    # check DS (default serialization) expression (abstract Types do not have a defaultSerialization)
    if json_value is None:
        value_type_str = "null"
    elif isinstance(json_value, bool):
        value_type_str = "boolean"
    elif isinstance(json_value, int):
        value_type_str = "integer"
    elif isinstance(json_value, float):
        value_type_str = "number"
    elif isinstance(json_value, str):
        value_type_str = "string"
    elif isinstance(json_value, dict):
        value_type_str = "object"
    elif isinstance(json_value, list):
        value_type_str = "array"
    else:
        raise RuntimeError(
            "Impossible case that the json deserialization of a value produced a non-standard Python type ("
            f"{str(type(json_value))}); got {json_value} at {location_id}!"
        )
    type_name_str = validator.get_default_serialization_concept_name_for(value_type_str)
    if type_name_str is not None:
        ch_value_type = validator.create_instantiated_type(type_name_str, location_id)
        assert ch_value_type is not None
        if validator.is_a_subtype_of_b(ch_value_type, expr_type, location_id):
            return InstExpression(None, ch_value_type, ch_value_type != expr_type)

    return IllFormedExpression(f"Could not match a valid {expr_type} expression to value {json_value}")


def _check_instantiation_schema(
    expr_value: object, expr_type: InstantiatedType, validator: ExpressionParserValidator, location_id: LocationId
) -> ParsedValue | None:
    instantiation_schema = validator.get_if_has_instantiation_schema(expr_type)
    if instantiation_schema is None or len(instantiation_schema) == 0:
        raise RuntimeError(
            f"It can't be that there is no instantiation schema defined for a non-abstract ValueDomain {expr_type}!"
        )
    if len(instantiation_schema) > 1:
        raise NotImplementedError
    type_application_constraint, schema_to_match = instantiation_schema[0]
    # TODO: check type_application_constraint (check that the type application satisfies the instantiation constraints)
    return validator.validate_value_against_schema(schema_to_match, expr_value, location_id)


def _validate_acyclic_default_argument_dependencies(
    default_argument_dependencies: frozendict[str, frozenset[str]], supplied_arguments: set[str]
) -> bool:
    # Default arguments whose value must actually be evaluated at this call site:
    # supplied defaults are terminal (their expressions are never evaluated) and
    # are therefore excluded from the dependency graph entirely.
    unsupplied_arguments = default_argument_dependencies.keys() - supplied_arguments
    if not unsupplied_arguments:
        return True

    # in_degree[node]: number of not-yet-resolved unresolved dependencies of `node`.
    # successors[node]: unresolved default arguments that depend on `node`
    # (i.e., the reverse adjacency list, needed to propagate resolution in Kahn's algorithm).
    in_degree: dict[str, int] = {}
    successors: dict[str, list[str]] = {node: [] for node in unsupplied_arguments}

    for node in unsupplied_arguments:
        # Restrict this node's declared dependencies to the relevant subgraph:
        # non-default arguments and already-supplied defaults are always resolved,
        # so they contribute no edge and are dropped here.
        deps = default_argument_dependencies[node] & unsupplied_arguments
        in_degree[node] = len(deps)
        for dep in deps:
            successors[dep].append(node)

    # Nodes with no unresolved dependencies can be evaluated immediately.
    queue = deque(node for node, degree in in_degree.items() if degree == 0)
    resolved_count = 0

    # Standard Kahn's algorithm: repeatedly resolve nodes with in-degree 0 and
    # decrement the in-degree of their dependents. A node stuck with in-degree > 0
    # forever (never enqueued) is part of, or depends on, a cycle.
    while queue:
        node = queue.popleft()
        resolved_count += 1
        for successor in successors[node]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue.append(successor)

    # Acyclic iff every unresolved node was eventually resolved.
    # A self-dependency (node depends on itself) leaves in_degree >= 1 permanently,
    # so it is correctly caught here without special-casing.
    return resolved_count == len(unsupplied_arguments)
