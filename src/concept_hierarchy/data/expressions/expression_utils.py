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


class ValueDomainArgumentProvenance(Enum):
    ANY = "Any"
    ADDR = "Addr"


class FunctionResultAccessor(Enum):
    GET = "Get"
    MOD = "Modify"


ExpressionProvenance = ValueDomainArgumentProvenance
ExpressionAccess = FunctionResultAccessor


class FunctionArgumentProvenance(Enum):
    ANY = "Any"
    ADDR = "Addr"
    RESET_ADDR = "ResetAddr"


class FunctionArgumentAccessor(Enum):
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
    FUNCTION_RETURN_VALUE_ADDR = 3
    """Function evaluation with Addr result."""
    FUNCTION_RETURN_VALUE_NO_ADDR = 4
    """Function evaluation with non-Addr result."""
    VARIABLE = 5
    """var in variable_context"""
    INSTANCE_PROPERTY = 6
    """var.property (where var is a variable with a subtype of InstanceBase)"""


class ExpressionType(Enum):
    VALUE_DOMAIN_LITERAL = 1
    """
    Condenses ExpressionDefinition.VALUE_DOMAIN_INSTANTIATION and ExpressionDefinition.VALUE_DOMAIN_CAST_INSTANTIATION.
    """
    FUNCTION_EVALUATION_ADDR = 2
    """Function evaluation with Addr result type"""
    FUNCTION_EVALUATION_NO_ADDR = 3
    """Function evaluation with not Addr result type"""
    VARIABLE = 4  #
    """Condenses ExpressionDefinition.VARIABLE and ExpressionDefinition.INSTANCE_PROPERTY."""


def get_permitted_expression_types_based_on_provenance_and_access_type(
    provenance_type: ExpressionProvenance, access_type: ExpressionAccess, is_strict_subtype: bool
):
    # Build the set of permitted ExpressionTypes for this combination
    if provenance_type == ExpressionProvenance.ADDR:
        if not access_type == ExpressionAccess.MOD:
            # Addr / ResetAddr + Get:
            # Both exact-type and strict-subtype allow Addr functions and variables.
            permitted = {
                ExpressionType.FUNCTION_EVALUATION_ADDR,
                ExpressionType.VARIABLE,
            }
        else:
            # Addr / ResetAddr + Modify|GetModify:
            # Strict subtypes are *not* permitted (cannot write back through a narrowed ref).
            if is_strict_subtype:
                permitted = set()
            else:
                permitted = {
                    ExpressionType.FUNCTION_EVALUATION_ADDR,
                    ExpressionType.VARIABLE,
                }
    else:
        if not access_type == ExpressionAccess.MOD:
            # Any + Get: everything is permitted regardless of subtype relationship.
            permitted = {
                ExpressionType.VALUE_DOMAIN_LITERAL,
                ExpressionType.FUNCTION_EVALUATION_ADDR,
                ExpressionType.FUNCTION_EVALUATION_NO_ADDR,
                ExpressionType.VARIABLE,
            }
        else:
            # Any + Modify|GetModify:
            if is_strict_subtype:
                # Only value-producing expressions are safe (no aliasing via addr).
                permitted = {
                    ExpressionType.VALUE_DOMAIN_LITERAL,
                    ExpressionType.FUNCTION_EVALUATION_NO_ADDR,
                }
            else:
                # Exact type -> all expression types permitted.
                permitted = {
                    ExpressionType.VALUE_DOMAIN_LITERAL,
                    ExpressionType.FUNCTION_EVALUATION_ADDR,
                    ExpressionType.FUNCTION_EVALUATION_NO_ADDR,
                    ExpressionType.VARIABLE,
                }
    return permitted
