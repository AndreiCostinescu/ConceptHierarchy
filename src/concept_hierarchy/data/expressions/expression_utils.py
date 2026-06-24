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


from enum import Enum


class ValueDomainArgumentReference(Enum):
    NO_REF = "NoRef"
    REF = "Reference"


class FunctionResultModifier(Enum):
    GET = "Get"
    MOD = "Modify"


ExpressionRef = ValueDomainArgumentReference
ExpressionMod = FunctionResultModifier


class FunctionArgumentReference(Enum):
    NO_REF = "NoRef"
    REF = "Reference"
    EMPTY_REF = "EmptyReference"


class FunctionArgumentModifier(Enum):
    GET = "Get"
    MOD = "Modify"
    GET_MOD = "GetModify"


class ExpressionDefinition(Enum):
    """
    Don't allow empty instantiation: VALUE_DOMAIN_EMPTY_INSTANTIATION = { Type: None }
        If an empty instantiation is wanted, one can specify an empty creation interface.
    """

    WRONG_TYPE_EXPRESSION = 0
    """Expression can't be interpreted as the requested ValueDomain."""
    VALUE_DOMAIN_INSTANTIATION = 1
    """Literal formula: try to interpret data as formula specified in "instantiation"."""
    VALUE_DOMAIN_CAST_INSTANTIATION = 2
    """Subtype creation arguments: { SubType: <instantiation of SubType from "instantiation" data> }"""
    FUNCTION_RETURN_VALUE_REF = 3
    """Function evaluation with reference result."""
    FUNCTION_RETURN_VALUE_NO_REF = 4
    """Function evaluation with non-reference result."""
    VARIABLE = 5
    """var in variable_context"""
    INSTANCE_PROPERTY = 6
    """var.property (where var is a variable with a subtype of InstanceBase)"""


class ExpressionType(Enum):
    VALUE_DOMAIN_LITERAL = 1
    """
    Condenses ExpressionDefinition.VALUE_DOMAIN_INSTANTIATION and ExpressionDefinition.VALUE_DOMAIN_CAST_INSTANTIATION.
    """
    FUNCTION_EVALUATION_REF = 2
    """Function evaluation with reference result type"""
    FUNCTION_EVALUATION_NO_REF = 3
    """Function evaluation with non-reference result type"""
    VARIABLE = 4  #
    """Condenses ExpressionDefinition.VARIABLE and ExpressionDefinition.INSTANCE_PROPERTY."""


def get_permitted_expression_types_based_on_reference_and_modifier_type(
    reference_type: ExpressionRef, modifier_type: ExpressionMod, is_strict_subtype: bool
):
    # Build the set of permitted ExpressionTypes for this combination
    if reference_type == ExpressionRef.REF:
        if not modifier_type == ExpressionMod.MOD:
            # Reference / EmptyReference + Get:
            # Both exact-type and strict-subtype allow REF functions and variables.
            permitted = {
                ExpressionType.FUNCTION_EVALUATION_REF,
                ExpressionType.VARIABLE,
            }
        else:
            # Reference / EmptyReference + Modify|GetModify:
            # Strict subtypes are *not* permitted (cannot write back through a narrowed ref).
            if is_strict_subtype:
                permitted = set()
            else:
                permitted = {
                    ExpressionType.FUNCTION_EVALUATION_REF,
                    ExpressionType.VARIABLE,
                }
    else:
        if not modifier_type == ExpressionMod.MOD:
            # NoRef + Get: everything is permitted regardless of subtype relationship.
            permitted = {
                ExpressionType.VALUE_DOMAIN_LITERAL,
                ExpressionType.FUNCTION_EVALUATION_REF,
                ExpressionType.FUNCTION_EVALUATION_NO_REF,
                ExpressionType.VARIABLE,
            }
        else:
            # NoRef + Modify|GetModify:
            if is_strict_subtype:
                # Only value-producing expressions are safe (no aliasing via ref).
                permitted = {
                    ExpressionType.VALUE_DOMAIN_LITERAL,
                    ExpressionType.FUNCTION_EVALUATION_NO_REF,
                }
            else:
                # Exact type â€“ all expression types permitted.
                permitted = {
                    ExpressionType.VALUE_DOMAIN_LITERAL,
                    ExpressionType.FUNCTION_EVALUATION_REF,
                    ExpressionType.FUNCTION_EVALUATION_NO_REF,
                    ExpressionType.VARIABLE,
                }
    return permitted
