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
from copy import copy
from dataclasses import dataclass
from enum import Enum
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
    ConstraintGroupAttempt,
    ExpressionAttempt,
    ExpressionKind,
    FunctionEvaluation,
    IllFormedExpression,
    InstancePropertyChain,
    InstExpression,
    LiteralTemplateVariableValue,
    NarrowExpression,
    PossibleInstExpression,
    PossibleVariableExpression,
    TemplateDependentExpression,
    Variable,
    VariableWithTemplateType,
    VerifiedTemplateDependentExpression,
)
from concept_hierarchy.data.jsonschema import CHSchemaNode
from concept_hierarchy.data.type_template_variables.constraint_formula import (
    ConstraintGroup,
    HierarchyCheckType,
    NonTypeTemplateConstraintFormula,
    TemplateConstraintOr,
)
from concept_hierarchy.data.type_template_variables.template_substitution import (
    substitute,
    substitute_template_variables_in_value,
)
from concept_hierarchy.data.types.concept_hierarchy_types import (
    TYPE_VALUE_IS_INSTANCE_CHECK,
    ConceptHierarchyTemplateArgument,
    ConceptHierarchyVariadicGroup,
    ExpandedVariadicTemplateVariable,
    InstantiatedType,
    LiteralValue,
    NonVariadicTemplateVariable,
    TemplateDependent,
    TemplateDependentType,
    TemplateVariable,
    TypeValue,
)
from concept_hierarchy.data.validators.template_argument_constraints_validator import (
    TemplateContextDeterminator,
    TypeTemplateInstantiationValidator,
    validate_template_argument_value_against_constraint,
    validate_type_against_constraint_formula,
)
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
    def create_possibly_template_dependent_type(self, type_name: str, location_id: LocationId) -> TypeValue:
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
    ) -> tuple[ParsedValue, list[ConceptHierarchyError]]:
        """
        Parse ``value`` against ``schema``, returning the result tree **and** the authoritative error list.

        Both are needed: the errors are what an :class:`IllFormedExpression` reports when this value turns
        out not to be a valid instantiation, and they cannot be recovered by walking the tree -- trial
        branches and failed ``allOf`` branches deliberately do not attach their errors to retained nodes.
        """

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


def substitute_schema(
    instantiation_schema: CHSchemaNode,
    template_context_of_concept: TemplateContext,
    expr_type: TypeValue,
    constraint_validator: TypeTemplateInstantiationValidator,
    location_id: LocationId,
) -> CHSchemaNode:
    template_substitution = {
        t_arg_name: t_arg_value
        for t_arg_name, t_arg_value in zip(template_context_of_concept.variables, expr_type.template_arguments)
    }
    if not template_substitution:
        return instantiation_schema

    def _parse_and_substitute(node: CHSchemaNode) -> CHSchemaNode:
        if node.is_boolean_schema:
            return node
        if not node.is_custom_type:
            return node.apply(_parse_and_substitute)
        assert node.custom_type is not None
        res = copy(node)
        # substitute
        subst_res = substitute_template_variables_in_value(
            node.custom_type,
            template_substitution,
            template_context_of_concept,
            TemplateContext(),
            constraint_validator,
            location_id + node.location_id,
        )
        assert isinstance(subst_res, TYPE_VALUE_IS_INSTANCE_CHECK)

        res.custom_type = subst_res
        assert res.custom_type is not None
        return res

    return instantiation_schema.apply(_parse_and_substitute)


def parse_expression(
    json_value: object,
    expr_type: TypeValue,
    expr_provenance: FunctionArgumentProvenance | ValueDomainArgumentProvenance,
    expr_access: FunctionArgumentAccessor | FunctionResultAccessor,
    expr_template_context: TemplateContext,
    validator: ExpressionParserValidator,
    location_id: LocationId,
    parse_template_expressions_without_type_checks: bool = False,
) -> Expression:
    if isinstance(expr_provenance, ValueDomainArgumentProvenance):
        expr_provenance = (
            FunctionArgumentProvenance.ADDR
            if expr_provenance == ValueDomainArgumentProvenance.ADDR
            else FunctionArgumentProvenance.ANY
        )
    if isinstance(expr_access, FunctionResultAccessor):
        expr_access = (
            FunctionArgumentAccessor.GET if expr_access == FunctionResultAccessor.GET else FunctionArgumentAccessor.MOD
        )
    expr_candidate_value = _parse_syntax_of_expression(
        json_value,
        expr_type,
        expr_template_context,
        validator,
        location_id,
        True,
        parse_template_expressions_without_type_checks,
    )
    is_strict_subtype = expr_candidate_value.is_strict_subtype
    is_addressable = isinstance(expr_candidate_value, Variable)
    if not is_addressable and isinstance(expr_candidate_value, FunctionEvaluation):
        is_addressable = expr_candidate_value.is_result_addressable
    if expr_provenance != FunctionArgumentProvenance.ANY and not is_addressable:
        # The expression parsed, but not into something that can be addressed. Say what it *is*: a bare
        # "got False" leaves the reader to work out which of the alternatives matched.
        expr_candidate_value = IllFormedExpression(
            f"Provenance violation: {expr_provenance.value} provenance requires an addressable expression "
            f"(a variable, an instance property chain, or a Function evaluation whose result is "
            f"{ValueDomainArgumentProvenance.ADDR.value}); got a "
            f"{_describe_expression_kind(expr_candidate_value)} of type {expr_candidate_value.value_type}"
        )
    elif expr_access != FunctionArgumentAccessor.GET and is_strict_subtype:
        expr_candidate_value = IllFormedExpression(
            f"Access violation: {expr_access.value} access requires the exact type {expr_type}, but this "
            f"{_describe_expression_kind(expr_candidate_value)} has type {expr_candidate_value.value_type}, "
            f"which is a strict subtype"
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


_EXPRESSION_KIND_NAMES: tuple[tuple[type, str], ...] = (
    # Most specific first: NarrowExpression subclasses InstExpression, InstancePropertyChain subclasses
    # Variable, so a plain isinstance sweep in the wrong order reports the base class.
    (NarrowExpression, "narrowed value domain instantiation"),
    (InstExpression, "value domain instantiation"),
    (FunctionEvaluation, "Function evaluation"),
    (InstancePropertyChain, "instance property chain"),
    (LiteralTemplateVariableValue, "literal template variable"),
    (Variable, "variable"),
)


def _describe_expression_kind(expr_value: ExpressionValue) -> str:
    """A reader-facing name for what an expression turned out to be, for provenance/access messages."""
    for kind, name in _EXPRESSION_KIND_NAMES:
        if isinstance(expr_value, kind):
            return name
    return type(expr_value).__name__


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
    parse_template_expressions_without_type_checks: bool = False,
) -> ExpressionValue:
    assert isinstance(expr_type, TYPE_VALUE_IS_INSTANCE_CHECK)
    if isinstance(expr_type, (ConceptHierarchyVariadicGroup, LiteralValue, ExpandedVariadicTemplateVariable)):
        raise RuntimeError(f"Can't parse an expression of type {expr_type}")

    is_expr_type_template_variable = isinstance(expr_type, TemplateVariable)
    if is_expr_type_template_variable:
        assert expr_template_context.has_template_variable(expr_type.clean_name), expr_type.full_name
    is_expr_type_template_containing = isinstance(expr_type, TemplateDependentType)
    is_expr_type_ground = isinstance(expr_type, InstantiatedType)
    assert is_expr_type_template_variable + is_expr_type_template_containing + is_expr_type_ground == 1

    if not parse_template_expressions_without_type_checks and not is_expr_type_ground:
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
            parse_template_expressions_without_type_checks,
            is_function_evaluation,
            is_function_evaluation_present,
            len_content_keys,
            ensure_unmodified_json_value,
        )
        ensure_unmodified_json_value(json_value, is_function_evaluation_present, is_function_evaluation)
        assert isinstance(expr_type, TemplateVariable) or len(expr_value_res) == 1
        if isinstance(expr_type, TemplateVariable):
            if len(expr_value_res) == 1:
                expr_res = expr_value_res[0]
            else:
                expr_res = VerifiedTemplateDependentExpression(expr_value_res)
        else:
            expr_res = expr_value_res[0]
        return expr_res
    except ConceptHierarchyError as e:
        ensure_unmodified_json_value(json_value, is_function_evaluation_present, is_function_evaluation)
        raise e


def _can_be_subtype_of_instantiated(
    type_to_be_checked: TypeValue,
    instantiated_type: InstantiatedType,
    validator: ExpressionParserValidator,
    template_context: TemplateContext,
    location_id: LocationId,
) -> bool:
    """
    Whether ``type_to_be_checked`` -- possibly a template variable or a template-dependent type such as
    ``Add<T>`` -- can be a subtype of the instantiated ``instantiated_type``.

    Examples: ``Add<T>`` is a subtype of ``Function``, and ``Increment<T>`` is a subtype of ``Add<Number>``
    if and only if ``T`` is ``Number``.
    """
    return _check_if_subtype(validator, type_to_be_checked, instantiated_type, template_context, location_id)


@dataclass(frozen=True)
class InstantiationSearch:
    """
    The outcome of searching a type's ``instantiation`` for a group that accepts a value.

    ``parsed`` is ``None`` when the type declares no instantiation schema at all (an abstract type).
    ``groups`` records every group that was tried, matched or not, so that a failure can say *why* --
    which constraints the type application did not satisfy, and how the one it did satisfy rejected the
    value.
    """

    parsed: ParsedValue | None
    errors: tuple[ConceptHierarchyError, ...] = ()
    groups: tuple[ConstraintGroupAttempt, ...] = ()


def _parse_syntax_of_expression_with_instantiated_type(
    json_value: object,
    expr_type: TypeValue,
    expr_template_context: TemplateContext,
    validator: ExpressionParserValidator,
    location_id: LocationId,
    recursively_parse: bool,
    parse_template_expressions_without_type_checks: bool,
    is_function_evaluation: bool,
    is_function_evaluation_present: bool,
    len_content_keys: int,
    ensure_unmodified_json_value: Callable[[object, bool, bool], None],
) -> list[ExpressionValue]:
    expressions_res: list[ExpressionValue] = []
    attempts: list[ExpressionAttempt] = []
    """Every alternative that was applicable to this value and was rejected; see IllFormedExpression."""

    def ensure_expression_invariant(_expressions_res: list[ExpressionValue], _expr_type: TypeValue) -> bool | None:
        assert isinstance(_expr_type, TYPE_VALUE_IS_INSTANCE_CHECK)
        # FIXME: Can TemplateDependentTypes contain multiple expression results?
        if isinstance(_expr_type, InstantiatedType):
            assert len(_expressions_res) <= 1
            return len(_expressions_res) == 1
        return None

    # check Narrow and FEval expressions
    if len_content_keys == 1:
        assert isinstance(json_value, dict)
        if is_function_evaluation_present:
            json_value.pop("isFunctionEvaluation")
        key, value = get_items_of_single_entry_dict(json_value)
        # TODO: verify if key is a Concept Hierarchy-specific type:
        #  - a (non-literal) Template Variable and
        #  - a Function or ValueDomain type application
        is_concept_hierarchy_expression = not key.startswith("s:")
        if validator.is_template_variable(key) and validator.is_literal_template_variable(key):
            is_concept_hierarchy_expression = False

        if is_concept_hierarchy_expression:
            expressions_res.extend(
                parse_expression_of_json_object(
                    key,
                    value,
                    expr_type,
                    expr_template_context,
                    validator,
                    location_id,
                    recursively_parse,
                    parse_template_expressions_without_type_checks,
                    is_function_evaluation,
                    is_function_evaluation_present,
                    ensure_expression_invariant,
                    attempts,
                )
            )
            if ensure_expression_invariant(expressions_res, expr_type):
                return expressions_res

        ensure_unmodified_json_value(json_value, is_function_evaluation_present, is_function_evaluation)
    # check Var expression
    if isinstance(json_value, str):
        assert expressions_res == []
        # Prioritize variables over template variables if there is a name clash!
        if validator.is_variable(json_value):
            if validator.is_template_variable(json_value):
                print(
                    f'[CH Warning] Prioritize variable "{json_value}" over template variable "{json_value}" in '
                    f"expression {json_value!r}"
                )
            var_type = validator.get_variable_type(json_value)
            if not isinstance(expr_type, InstantiatedType):
                return [PossibleVariableExpression(json_value, var_type)]
            elif isinstance(var_type, TemplateDependent):
                return [VariableWithTemplateType(json_value, var_type)]
            assert isinstance(var_type, InstantiatedType), f"{var_type} of type {str(type(var_type))}"
            if _check_if_subtype(validator, var_type, expr_type, expr_template_context, location_id):
                return [Variable(json_value, var_type, var_type != expr_type)]
            else:
                reason = f"Type {var_type} of variable {json_value} is not a subtype of {expr_type}!"
                attempts.append(ExpressionAttempt(ExpressionKind.VARIABLE, reason, tried_type=var_type))
                return [IllFormedExpression(reason, tuple(attempts))]
        else:
            possible_instance_property_chain = json_value.split(".")
            if len(possible_instance_property_chain) <= 1 or not validator.is_variable(
                possible_instance_property_chain[0]
            ):
                # Not a variable, and not a chain rooted at one. Record both, so that a string matching
                # nothing says which kinds of name were looked for rather than only that it matched none.
                attempts.append(
                    ExpressionAttempt(
                        ExpressionKind.VARIABLE, f'"{json_value}" is not a variable of this Concept Hierarchy'
                    )
                )
                if len(possible_instance_property_chain) > 1:
                    attempts.append(
                        ExpressionAttempt(
                            ExpressionKind.INSTANCE_PROPERTY_CHAIN,
                            f'"{possible_instance_property_chain[0]}" is not a variable, so "{json_value}" is not '
                            f"an instance property chain",
                        )
                    )
                if not validator.is_template_variable(json_value):
                    attempts.append(
                        ExpressionAttempt(
                            ExpressionKind.LITERAL_TEMPLATE_VARIABLE,
                            f'"{json_value}" is not a template variable in this context',
                        )
                    )
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
                assert isinstance(var_type, InstantiatedType)
                if isinstance(expr_type, TemplateDependent):
                    return [PossibleVariableExpression(json_value, var_type)]
                if _check_if_subtype(validator, var_type, expr_type, expr_template_context, location_id):
                    return [
                        InstancePropertyChain(
                            possible_instance_property_chain,
                            types_in_property_chain,
                            var_type != expr_type,
                        )
                    ]
                else:
                    reason = f"Type {var_type} of instance property chain {json_value} is not a subtype of {expr_type}!"
                    attempts.append(
                        ExpressionAttempt(ExpressionKind.INSTANCE_PROPERTY_CHAIN, reason, tried_type=var_type)
                    )
                    return [IllFormedExpression(reason, tuple(attempts))]
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
                    if _check_if_subtype(validator, ch_value_type, expr_type, expr_template_context, location_id):
                        return [LiteralTemplateVariableValue(json_value, ch_value_type, ch_value_type != expr_type)]
                    else:
                        reason = (
                            f"The type of the literal template variable {json_value} (matched via default "
                            f"serialization to {ch_value_type}) is not a subtype of {expr_type}!"
                        )
                        attempts.append(
                            ExpressionAttempt(
                                ExpressionKind.LITERAL_TEMPLATE_VARIABLE, reason, tried_type=ch_value_type
                            )
                        )
                        return [IllFormedExpression(reason, tuple(attempts))]
                else:
                    reason = (
                        f'The literal constraint "{literal_constraint_type}" of "{json_value}" does not have a '
                        f'matching registered "{ValueDomainDefinition.value_domain_default_serialization}" '
                        f"({value_type_str})!"
                    )
                    attempts.append(ExpressionAttempt(ExpressionKind.LITERAL_TEMPLATE_VARIABLE, reason))
                    return [IllFormedExpression(reason, tuple(attempts))]

    # check Inst expression (abstract Types do not have instantiation schemas)
    if isinstance(expr_type, TemplateVariable):
        expressions_res.append(PossibleInstExpression())
    else:
        inst_res = _check_instantiation_schema(json_value, expr_type, expr_template_context, validator, location_id)
        if inst_res.parsed is not None and inst_res.parsed.is_valid():
            expressions_res.append(InstExpression(inst_res.parsed, expr_type, True))
            if ensure_expression_invariant(expressions_res, expr_type):
                return expressions_res
        attempts.append(_instantiation_attempt(ExpressionKind.INSTANTIATION, expr_type, inst_res))

    # check DS (default serialization) expression (abstract Types do not have a defaultSerialization)
    value_type_str = get_json_type_as_string(json_value, location_id)
    type_name_str = validator.get_default_serialization_concept_name_for(value_type_str)
    if type_name_str is None:
        attempts.append(
            ExpressionAttempt(
                ExpressionKind.DEFAULT_SERIALIZATION,
                f'no concept of this Concept Hierarchy registers a "'
                f'{ValueDomainDefinition.value_domain_default_serialization}" for the JSON type '
                f'"{value_type_str}"',
            )
        )
    else:
        ch_value_type = validator.create_instantiated_type(type_name_str, location_id)
        assert ch_value_type is not None
        if isinstance(expr_type, TemplateVariable):
            expressions_res.append(PossibleInstExpression(ch_value_type))
        elif isinstance(expr_type, InstantiatedType) and _check_if_subtype(
            validator, ch_value_type, expr_type, expr_template_context, location_id
        ):
            expressions_res.append(InstExpression(None, ch_value_type, ch_value_type != expr_type))
            if ensure_expression_invariant(expressions_res, expr_type):
                return expressions_res
        else:
            attempts.append(
                ExpressionAttempt(
                    ExpressionKind.DEFAULT_SERIALIZATION,
                    f'the JSON type "{value_type_str}" serializes to {ch_value_type}, which is not a subtype '
                    f"of {expr_type}",
                    tried_type=ch_value_type,
                )
            )

    if isinstance(expr_type, InstantiatedType) and expressions_res == []:
        expressions_res.append(
            IllFormedExpression(
                f"Could not match a valid {expr_type} expression to value {json_value}", tuple(attempts)
            )
        )
    return expressions_res


def _instantiation_attempt(
    kind: ExpressionKind, tried_type: TypeValue, search: InstantiationSearch
) -> ExpressionAttempt:
    """Turn a failed :func:`_check_instantiation_schema` search into one attempt of the explanation trace."""
    if search.parsed is None:
        return ExpressionAttempt(kind, f"{tried_type} is abstract: it declares no instantiation schema", tried_type)
    # `search.errors` is the matched group's error list, which its own ConstraintGroupAttempt already
    # carries -- passing it as `schema_errors` too would render every schema error twice.
    return ExpressionAttempt(
        kind,
        f"the value does not satisfy the instantiation schema of {tried_type}",
        tried_type=tried_type,
        constraint_groups=search.groups,
    )


def _check_instantiation_schema(
    expr_value: object,
    expr_type: InstantiatedType,
    expr_template_context: TemplateContext,
    validator: ExpressionParserValidator,
    location_id: LocationId,
) -> InstantiationSearch:
    instantiation_schema = validator.get_if_has_instantiation_schema(expr_type)
    if instantiation_schema is None or len(instantiation_schema) == 0:
        assert validator.is_type_abstract(expr_type), (
            f'It can\'t be that there is no instantiation schema defined for a non-abstract ValueDomain "{expr_type}"!'
        )
        return InstantiationSearch(None)
    groups: list[ConstraintGroupAttempt] = []
    for i, (type_application_constraint, schema_to_match) in enumerate(instantiation_schema):
        type_template_instantiation_validator = validator.get_type_template_instantiation_validator()
        found_matching_schema = type_application_constraint is None
        constraint_errors: list[ConceptHierarchyError] = []
        if not found_matching_schema:
            constraint_errors = validate_type_against_constraint_formula(
                type_application_constraint,
                expr_type,
                TemplateContextDeterminator(expr_template_context),
                type_template_instantiation_validator,
                location_id,
            )
            found_matching_schema = len(constraint_errors) == 0
        if not found_matching_schema:
            groups.append(
                ConstraintGroupAttempt(type_application_constraint, matched=False, errors=tuple(constraint_errors))
            )
            continue
        # substitute schema's template arguments
        expr_type_template_context = validator.get_template_context(expr_type.clean_name)
        substituted_schema_to_match = substitute_schema(
            schema_to_match,
            expr_type_template_context,
            expr_type,
            type_template_instantiation_validator,
            location_id,
        )
        parsed, errors = validator.validate_value_against_schema(substituted_schema_to_match, expr_value, location_id)
        groups.append(ConstraintGroupAttempt(type_application_constraint, matched=True, errors=tuple(errors)))
        return InstantiationSearch(parsed, tuple(errors), tuple(groups))
    raise RuntimeError(f"There should always be a fallback matching schema... This was not reached at {expr_type}!")


def get_json_type_as_string(json_value: object, location_id: LocationId) -> str:
    if json_value is None:
        return "null"
    if isinstance(json_value, bool):
        return "boolean"
    if isinstance(json_value, int):
        return "integer"
    if isinstance(json_value, float):
        return "number"
    if isinstance(json_value, str):
        return "string"
    if isinstance(json_value, dict):
        return "object"
    if isinstance(json_value, list):
        return "array"
    raise RuntimeError(
        "Impossible case that the json deserialization of a value produced a non-standard Python type ("
        f"{str(type(json_value))}); got {json_value} at {location_id}!"
    )


def parse_expression_of_json_object(
    key: str,
    value: object,
    expr_type: TypeValue,
    expr_template_context: TemplateContext,
    validator: ExpressionParserValidator,
    location_id: LocationId,
    recursively_parse: bool,
    parse_template_expressions_without_type_checks: bool,
    is_function_evaluation: bool,
    is_function_evaluation_present: bool,
    ensure_expression_invariant: Callable[[list[ExpressionValue], TypeValue], bool | None],
    attempts: list[ExpressionAttempt],
) -> list[ExpressionValue]:
    expressions_res = []

    function_composition_type = validator.create_instantiated_type("FunctionComposition", location_id)
    function_type = validator.create_instantiated_type("Function", location_id)

    try:
        key_type = validator.create_possibly_template_dependent_type(key, location_id)
    except CHSemanticError as e:
        if e.args[0] == f"ParsedType '{key}' is not a template variable (in this context) nor a concept!":
            # The single key is not a type at all, so neither an FEval nor a Narrow was ever possible.
            reason = f'"{key}" is not a concept or a template variable of this Concept Hierarchy'
            attempts.append(ExpressionAttempt(ExpressionKind.FUNCTION_EVALUATION, reason))
            attempts.append(ExpressionAttempt(ExpressionKind.NARROW, reason))
            return expressions_res
        raise e
    if not isinstance(key_type, TemplateVariable) and validator.is_type_abstract(key_type):
        raise CHSemanticError(
            f"{key_type} is an abstract type! Thus, it can not be used in expression values "
            f"(neither as FEval nor as Narrow expressions)!",
            location_id=location_id + [key],
            part=PathPart.KEY,
        )
    function_evaluation = isinstance(expr_type, TemplateVariable) or (
        not _check_if_subtype(validator, expr_type, function_composition_type, expr_template_context, location_id)
        and is_function_evaluation
    )
    is_function_subtype = _check_if_subtype(validator, key_type, function_type, expr_template_context, location_id)
    if function_evaluation and is_function_subtype:
        function_return = validator.get_function_return_interface(key_type.clean_name)
        if function_return is None:
            raise CHSemanticError(
                f"Function {key} does not return anything; expected a return type of {expr_type}!",
                location_id=location_id + [key],
                part=PathPart.KEY,
            )
        if not isinstance(value, dict):
            reason = (
                f"Function evaluation expression should have the value of the json object an other json object, "
                f"not {value}!"
            )
            attempts.append(ExpressionAttempt(ExpressionKind.FUNCTION_EVALUATION, reason, tried_type=key_type))
            expressions_res.append(IllFormedExpression(reason, tuple(attempts)))
            if ensure_expression_invariant(expressions_res, expr_type):
                return expressions_res
        else:
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
            if not isinstance(expr_type, TemplateVariable) and not _check_if_subtype(
                validator, function_return_type, expr_type, expr_template_context, location_id
            ):
                reason = f"Function result type {function_return_type} is not a subtype of {expr_type}"
                attempts.append(ExpressionAttempt(ExpressionKind.FUNCTION_EVALUATION, reason, tried_type=key_type))
                expressions_res.append(IllFormedExpression(reason, tuple(attempts)))
                if ensure_expression_invariant(expressions_res, expr_type):
                    return expressions_res
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
                        parse_template_expressions_without_type_checks,
                    )
                    if not arg_expr.is_valid:
                        assert isinstance(arg_expr.value, IllFormedExpression)
                        reason = (
                            f"{key} argument {f_arg_name}'s value {f_arg_expr_val} is invalid: {arg_expr.value.reason}"
                        )
                        attempts.append(
                            ExpressionAttempt(
                                ExpressionKind.FUNCTION_EVALUATION,
                                f'argument "{f_arg_name}" is not a valid {f_arg_type} expression',
                                tried_type=key_type,
                                cause=arg_expr.value,
                            )
                        )
                        expressions_res.append(IllFormedExpression(reason, tuple(attempts)))
                        if ensure_expression_invariant(expressions_res, expr_type):
                            return expressions_res
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
            expressions_res.append(
                FunctionEvaluation(
                    key_type,
                    function_return_type,
                    f_args,
                    is_result_addressable,
                    function_return_type != expr_type,
                )
            )
            if ensure_expression_invariant(expressions_res, expr_type):
                return expressions_res

    if _check_if_subtype(validator, key_type, expr_type, expr_template_context, location_id):
        if is_function_evaluation_present and not is_function_subtype:
            raise CHSemanticError(
                f'Invalid use of the "isFunctionEvaluation" keyword at single-content-key object "{key_type}"!',
                location_id=location_id + ["isFunctionEvaluation"],
                part=PathPart.KEY,
            )
        if not recursively_parse:
            expressions_res.append(NarrowExpression(None, key_type, key_type != expr_type))
            if ensure_expression_invariant(expressions_res, expr_type):
                return expressions_res
        else:
            # abstract Types do not have instantiation schemas
            narrow_res = _check_instantiation_schema(value, key_type, expr_template_context, validator, location_id)
            if narrow_res.parsed is not None and narrow_res.parsed.is_valid():
                expressions_res.append(NarrowExpression(narrow_res.parsed, key_type, key_type != expr_type))
                if ensure_expression_invariant(expressions_res, expr_type):
                    return expressions_res
            attempts.append(_instantiation_attempt(ExpressionKind.NARROW, key_type, narrow_res))
    else:
        attempts.append(
            ExpressionAttempt(
                ExpressionKind.NARROW,
                f"{key_type} is not a subtype of {expr_type}",
                tried_type=key_type,
            )
        )

    # if expr_type is InstantiatedTypes, this expression is neither a `FEval` nor a `Narrow`
    return expressions_res


class SubtypeVerdict(Enum):
    """
    The outcome of a subtype check whose operands may be template-dependent.

    NO
        There is no instantiation of the template variables under which the check holds.
    MAYBE
        The check holds under some, but not necessarily all, instantiations. The accompanying
        ``TemplateContext`` records the constraint on the template variables under which it holds.
    YES
        The check holds under every instantiation permitted by the template context.
    """

    NO = 0
    MAYBE = 1
    YES = 2


@dataclass(frozen=True)
class SubtypeCheckResult:
    """
    The result of :func:`check_if_subtype`.

    ``template_context`` is only set for a ``MAYBE`` verdict; it holds the constraint on the template
    variables under which ``a`` is a subtype of ``b``, and is what a later instantiation-time check must
    re-verify. ``errors`` is only set for a ``NO`` verdict and explains why the check failed.
    """

    verdict: SubtypeVerdict
    template_context: TemplateContext | None = None
    errors: tuple[ConceptHierarchyError, ...] = ()

    def __bool__(self) -> bool:
        """Existential reading: only a definite ``NO`` is falsy; a ``MAYBE`` is a "not yet ruled out"."""
        return self.verdict is not SubtypeVerdict.NO


def _general_subtype_check(
    validator: ExpressionParserValidator,
    a: TypeValue,
    b: TypeValue,
    template_context: TemplateContext,
    location_id: LocationId,
) -> SubtypeCheckResult:
    """
    Check whether ``a`` is a subtype of ``b``, where either may be a template variable or a
    template-dependent type.

    All template variables occurring in ``a`` or ``b`` must be template parameters of the ValueDomain or
    Function that the expression lies in, i.e. they must be variables of ``template_context``; a reference
    to any other variable is an invalid type application and is reported as such.

    Template arguments are matched **invariantly**: ``Box<Integer>`` is not a subtype of ``Box<Number>``.

    :return: a :class:`SubtypeCheckResult`; a ``MAYBE`` carries the template-variable constraint under which
        the subtype relation holds, which the (still to be written) instantiation-time check must verify.
    """
    _verify_subtype_check_operands(validator, a, b)

    if isinstance(a, InstantiatedType) and isinstance(b, InstantiatedType):
        # Neither side depends on template variables, so the answer is definite.
        if validator.is_a_subtype_of_b(a, b, location_id):
            return SubtypeCheckResult(SubtypeVerdict.YES)
        return SubtypeCheckResult(SubtypeVerdict.NO)

    type_template_instantiation_validator = validator.get_type_template_instantiation_validator()
    if isinstance(b, TemplateVariable):
        return _check_if_subtype_of_template_variable(
            type_template_instantiation_validator, a, b, template_context, location_id
        )
    return _check_if_subtype_of_type_application(
        type_template_instantiation_validator, a, b, template_context, location_id
    )


def _verify_subtype_check_operands(validator: ExpressionParserValidator, a: TypeValue, b: TypeValue) -> None:
    """Neither operand of a subtype check may be a variadic or a literal template variable."""
    for type_value, role in ((a, "subtype"), (b, "supertype")):
        if not isinstance(type_value, TemplateVariable):
            continue
        if not isinstance(type_value, NonVariadicTemplateVariable):
            raise RuntimeError(
                f'It can not be that the type "{type_value}" to be checked as {role} is a variadic template variable!'
            )
        if validator.is_literal_template_variable(type_value.clean_name):
            raise RuntimeError(f'It can not be that a type "{type_value}" is a literal template variable!')


def _check_if_subtype_of_template_variable(
    validator: TypeTemplateInstantiationValidator,
    a: TypeValue,
    b: NonVariadicTemplateVariable,
    template_context: TemplateContext,
    location_id: LocationId,
) -> SubtypeCheckResult:
    """
    ``b`` is an uninstantiated template variable, so ``a`` is a subtype of it exactly when ``b`` is
    instantiated to ``a`` itself or to one of ``a``'s supertypes. That can not be decided while ``b`` is
    uninstantiated, so it is recorded as an additional constraint on ``b`` instead.
    """
    if a.full_name == b.full_name:
        # The very same template variable; whatever it is instantiated to, it is a subtype of itself.
        return SubtypeCheckResult(SubtypeVerdict.YES)
    # `ASCENDANTS_OF` always excludes the literal itself, so the reflexive case is added explicitly.
    supertype_or_equal = TemplateConstraintOr(
        location_id,
        (
            validator.create_type_constraint_from_value(a, location_id, HierarchyCheckType.SELF, template_context),
            validator.create_type_constraint_from_value(
                a, location_id, HierarchyCheckType.ASCENDANTS_OF, template_context
            ),
        ),
    )
    narrowed = TemplateContext(
        template_context.variables,
        template_context.variadic_variables,
        template_context.add_and_constraint_to(b.clean_name, supertype_or_equal, location_id),
    )
    if narrowed.is_empty_constraint:
        return SubtypeCheckResult(SubtypeVerdict.NO)
    return SubtypeCheckResult(SubtypeVerdict.MAYBE, narrowed)


def _check_if_subtype_of_type_application(
    validator: TypeTemplateInstantiationValidator,
    a: TypeValue,
    b: InstantiatedType | TemplateDependentType,
    template_context: TemplateContext,
    location_id: LocationId,
) -> SubtypeCheckResult:
    """
    ``b`` is a type application, so "being a subtype of ``b``" is expressible as the constraint formula
    "a descendant of ``b``, whose template arguments match ``b``'s exactly", and ``a`` can be validated
    against it. Where that validation meets a template variable -- on either side -- it does not decide the
    check but accumulates the constraint under which it would hold.
    """
    b_constraint = validator.create_type_constraint_from_value(
        b, location_id, HierarchyCheckType.DESCENDANTS_OF, template_context
    )
    determinator = TemplateContextDeterminator(template_context)
    errors = validate_template_argument_value_against_constraint(
        b_constraint, a, determinator, validator, {}, location_id, collect_all_errors=True
    )
    if errors:
        return SubtypeCheckResult(SubtypeVerdict.NO, errors=tuple(errors))
    if determinator.determined is None:
        # Nothing had to be constrained, so the check holds for every instantiation.
        return SubtypeCheckResult(SubtypeVerdict.YES)
    if determinator.determined.is_empty_constraint:
        return SubtypeCheckResult(SubtypeVerdict.NO)
    return SubtypeCheckResult(SubtypeVerdict.MAYBE, determinator.determined)


def _check_if_subtype(
    validator: ExpressionParserValidator,
    a: TypeValue,
    b: TypeValue,
    template_context: TemplateContext,
    location_id: LocationId,
) -> bool:
    """
    Whether ``a`` can be a subtype of ``b``, read existentially: ``False`` means that no instantiation of the
    template variables can make ``a`` a subtype of ``b``, and the expression is therefore ill-formed.

    Use :func:`_general_subtype_check` directly to also obtain the constraint that a ``MAYBE`` result depends on.
    """
    return bool(_general_subtype_check(validator, a, b, template_context, location_id))


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
