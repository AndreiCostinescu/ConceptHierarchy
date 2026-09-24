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
    see `documentation/TODO_FUNCTION_INTERPRETATION_MARKER.md`. Every member but `UNSPECIFIED` is a
    decision, not a hint, and a leaf that cannot honor one must **fail** rather than fall back -- that is
    what makes the ``oneOf`` resolve to exactly one branch.

    **A member is a constraint, not a default.** This is the distinction the enum has been misread on
    twice, and both times the same way, so it is worth stating plainly:

    * the **default** is the reading the classification cascade lands on when nothing rules it out. It is
      not represented here at all -- it is implicit in the order of the cascade (`[CH].md` 10.2);
    * a **constraint** says which readings may be *attempted*. `COMPOSITION` does not mean "prefer the
      composition", it means "it must be the composition; do not try anything else".

    The two come apart wherever a site has a default but no exclusive claim on it, and a
    `FunctionComposition` site is exactly that: the composition is its default, and yet a `Narrow` to a
    `FunctionComposition` subtype is equally legal there, so the only thing the site can rule out is the evaluation.
    That is `NOT_AN_EVALUATION`, and it is why the member cannot be replaced by `COMPOSITION`
    even though "a composition is what you get here" is a true description of the site.
    """

    UNSPECIFIED = "unspecified"
    """No decision has been made; the site's own type decides, as usual."""

    NOT_AN_EVALUATION = "not an evaluation"
    """
    This object is *not* a Function evaluation. Nothing is said about which of the other readings it is.

    Every other member says what the value **is**; this one says only what it is **not**, and that is not
    a shortcoming to be split away -- it is a weaker statement than any marker can make, and three
    readings survive it: a `Narrow`, an `Inst` reading the object as a composition, and an `Inst` reading
    it as ordinary data for the site's schema. `COMPOSITION` admits only the second, `INSTANTIATION` only
    the first, so this is *not* their disjunction and cannot be rewritten as either.

    At a site whose type is a `FunctionComposition` and whose key carries no marker the composition is the **default**.
    (see the class docstring on why that is not the same as a
    constraint -- and a `Narrow` to a `FunctionComposition` subtype is equally legal, so the evaluation is
    all this site can exclude.

    *(measured)* Its whole effect is that exclusion, and it is load-bearing. Against a green tree:

    ===================================================== ==========
    change                                                failures
    ===================================================== ==========
    stop producing it (this branch yields `UNSPECIFIED`)   124
    produce it, but let carrying it constrain nothing      124
    yield `COMPOSITION` here instead                       135
    ===================================================== ==========

    The first two fail the *same* tests, which is what says producing and honoring it are one mechanism:
    suppress the `FEval` reading, so that `{F: args}` is read as the composition instead of hard-failing
    on ``res(F) <= tau`` before `Inst` is ever tried. The third is those 124 **plus 11**, and the 11 are
    the `Narrow` that `COMPOSITION` would forbid -- ten in `test_function_composition_args_schema.py` and
    `TestTheTypeDrivenDefaultIsNotTheCompositionMarker::test_a_narrow_is_still_admissible_at_a_composition_site`.
    Strengthening the default would also resolve the ``FunctionCompositionRes<T>`` collision by fiat,
    making silence *mean* "composition".

    It was also written as ``"isFunctionEvaluation": false`` until that keyword was removed; §9 of
    `TODO_FUNCTION_INTERPRETATION_MARKER.md` says what each of its uses became.
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
    at every site one of the other two is what an unmarked key means, which is why a directive saying
    only "not an evaluation" could never select it on its own.
    """

    EVALUATION = "evaluation"
    """
    This object *is* a Function evaluation, and could not be one at the site itself.

    Written as ``fEval:`` where ``res(K)`` is not a subtype of the site's type --
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

One entry per reading, and the only place the three words are written down. `NOT_AN_EVALUATION` is
deliberately not among them: it says only "not an evaluation", without saying which of the other two or 
even none of the two, and a marker never needs to be that vague.
"""


def split_function_interpretation_marker(key: str) -> tuple["FunctionInterpretation | None", str]:
    """
    Split a leading ``fEval:`` / ``fComp:`` / ``fInst:`` marker off an expression key.

    Returns ``(interpretation, key_without_the_marker)``, and ``(None, key)`` when there is no marker.

    **The first colon only.** Everything after it is the key, however many colons it holds: a type
    application may carry a *string literal* template argument, which is arbitrary text and may contain
    colons -- including these very words. ``Tagged<"fEval:x">`` is a valid application today, and
    splitting anywhere but the first colon, or refusing keys with more than one, would break it.
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
