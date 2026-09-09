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


class FunctionInterpretation(Enum):
    """
    Which of its readings a single-content-key JSON object keyed by a _Function_ has been decided to have.

    A `{F: args}` object has three: the **evaluation** of `F`, a `FunctionComposition` **composing** `F`,
    and a `Narrow` -- the **instantiation** of `F`, its value. Which one is meant is decided at the
    expression site, and the decision then has to *travel*, because a schema branch re-enters the
    expression parser with the *branch's* type: inside ``FunctionCompositionRes<T>``'s
    ``oneOf: ["T", "FunctionComposition"]`` the ``"T"`` branch is asked about a `Boolean`, where nothing is
    left of the fact that the site was a `FunctionComposition`.

    Carried from the expression parser into an instantiation schema and back out at each custom-type leaf;
    see `documentation/TODO_FUNCTION_EVALUATION_VS_COMPOSITION.md`. Every member but `UNSPECIFIED` is a
    decision, not a hint, and a leaf that cannot honor one must **fail** rather than fall back -- that is
    what makes the ``oneOf`` resolve to exactly one branch.
    """

    UNSPECIFIED = "unspecified"
    """No decision has been made; the site's own type and keyword decide, as usual."""

    RULED_OUT = "ruled out"
    """
    This object is *not* a Function evaluation -- but *which* of the other two it is, is not said.

    Written as ``"isFunctionEvaluation": false``, or implied by a site whose type is a
    `FunctionComposition` and that carries no keyword. A leaf must not read the object as a `FEval`.

    **This member is the conflation, and it is the next thing to split.** "Not an evaluation" cannot choose
    between a composition and an instantiation, which is why an argument-less Function at a
    ``FunctionCompositionRes<T>`` site whose ``T`` admits a Function type matches *both* branches and no
    keyword can separate them -- see `TODO_FUNCTION_INTERPRETATION_MARKER.md` §1 and
    `TestNarrowVersusCompositionIsStillAmbiguous`. Splitting it into ``COMPOSITION`` and ``INSTANTIATION``
    needs the `fComp:` / `fInst:` markers to exist first, because only they can say which was meant; §5 of
    that plan says how each present-day producer of this value has to be re-decided.
    """

    COMPOSITION = "composition"
    """
    This object is a `FunctionComposition` **composing** `K` -- not a call of it, and not its value.

    Written as ``fComp:``. Only an `Inst` whose type is a `FunctionComposition` may produce it; `Narrow`
    is off, which is what separates it from `INSTANTIATION` where both would otherwise match.
    """

    INSTANTIATION = "instantiation"
    """
    This object is the _Function_ **value** -- a `Narrow` to `K`.

    Written as ``fInst:``. Only a `Narrow` may produce it. This is the reading that never has a default:
    at every site one of the other two is what an unmarked key means, which is why
    ``isFunctionEvaluation: false`` could never select it on its own.
    """

    EVALUATION = "evaluation"
    """
    This object *is* a Function evaluation, and could not be one at the site itself.

    Written as ``"isFunctionEvaluation": true`` where ``res(K)`` is not a subtype of the site's type --
    always so at a `FunctionComposition` site, because no Function returns a `FunctionComposition`. The
    reading has to be consumed by a custom-type leaf of the site's instantiation schema whose own type
    ``res(K)`` does satisfy, and every other reading of the object is off the table: a leaf that cannot
    evaluate it fails, and no `Narrow`, `Var`, `Inst` or default serialization may stand in for it.
    """


FUNCTION_INTERPRETATION_MARKERS: dict[str, "FunctionInterpretation"] = {
    "fEval": FunctionInterpretation.EVALUATION,
    "fComp": FunctionInterpretation.COMPOSITION,
    "fInst": FunctionInterpretation.INSTANTIATION,
}
"""
The marker a key may carry to say which of its three readings it has.

One entry per reading, and the only place the three words are written down. `RULED_OUT` is deliberately
not among them: it is what ``isFunctionEvaluation: false`` still means -- "not an evaluation", without
saying which of the other two -- and a marker never needs to be that vague.
"""


def split_function_interpretation_marker(key: str) -> tuple["FunctionInterpretation | None", str]:
    """
    Split a leading ``fEval:`` / ``fComp:`` / ``fInst:`` marker off an expression key.

    Returns ``(interpretation, key_without_the_marker)``, and ``(None, key)`` when there is no marker.

    **The first colon only.** Everything after it is the key, however many colons it holds: a type
    application may carry a *string literal* template argument, which is arbitrary text and may contain
    colons -- including these very words. ``Tagged<"fEval:x">`` is a valid application today, and
    splitting anywhere but the first colon, or refusing keys with more than one, would break it.

    There is no competing reading of a leading ``word:``. A type application's name is alphanumeric plus
    ``_``, so it cannot contain a colon; ``s:`` marks a `String` *value* and never reaches a type position;
    and the qualified-variable form ``Add:T`` is not a valid type application. These three are therefore
    **markers, not reserved concept names** -- nothing ever resolves ``fEval`` as a type, so a concept of
    that name would not collide.
    """
    prefix, separator, remainder = key.partition(":")
    if not separator:
        return None, key
    interpretation = FUNCTION_INTERPRETATION_MARKERS.get(prefix)
    if interpretation is None:
        return None, key
    return interpretation, remainder


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
