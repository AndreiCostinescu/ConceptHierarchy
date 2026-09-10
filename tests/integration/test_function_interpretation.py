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

"""
How a _Function_-keyed object is read: every marker, at every kind of site.

A single-key object whose key names a _Function_ has three possible readings -- a call of it, a
composition of it, or the _Function_ value itself -- and one of them is the default, chosen by the site's
type. The prefixes `fEval:` / `fComp:` / `fInst:` on that key name a reading explicitly. This suite is the
specification of the whole decision (`[CH].md` 10.2 is the normative statement,
`documentation/TODO_FUNCTION_INTERPRETATION_MARKER.md` the plan it was built from).

Three things are asserted for every case, because any one of them alone can pass for the wrong reason:

* **the `ExpressionValue` subclass** -- `FunctionEvaluation` versus `InstExpression`/`NarrowExpression` is
  the *only* record of which reading was taken. Nothing is rewritten at parse time (`fEval:` at a
  `FunctionComposition` site is not turned into a ``Return`` composition here; that lowering happens
  elsewhere), so the class is the answer;
* **the message**, so that a rejection rejects for the stated reason and not by accident;
* **the location**, so that the error is attached to the JSON that is actually wrong.

The table being pinned (`[CH].md` 10.2), where K is the single content key's type application:

| tau                            | no marker | fEval: | fComp:                 | fInst:            |
|--------------------------------|-----------|--------|------------------------|-------------------|
| tau not <= FunctionComposition | FEval     | FEval  | only a composing Inst  | only the Narrow   |
| tau <= FunctionComposition     | Inst      | FEval  | Inst (the default)     | Inst, via a leaf  |

A marker **requires** its reading rather than permitting it: the value is refused unless that reading is
what stands at its own location, which a schema matching without reading it would otherwise discard.

A fourth reading has no marker and cannot have one -- the object is *ordinary data* for the site's
instantiation schema, whose property name it must keep, and a marker would change that name. It is written
as an explicit `Narrow`, ``{tau: w}``; see `TestTheTypeDrivenDefaultIsNotTheCompositionMarker`.

**On ``"isFunctionEvaluation"``.** A sibling keyword of that name used to carry this decision, two-valued
and unable to separate the composition from the instantiation, and it was removed once the markers landed
(`TODO_FUNCTION_INTERPRETATION_MARKER.md` 9). The name is now an identifier like any other, and the two
classes that still write it -- `TestTheKeywordIsOrdinaryData` and `TestTheKeywordReachesTheSchemaUntouched`
-- exist to keep it that way: an argument or a property may be called this, and reading it as anything
else would silently delete a value the author wrote. `TestTheKeywordIsNoLongerADirective` pins the removal
itself.
"""

from __future__ import annotations

import copy

import pytest

from concept_hierarchy.data.contexts.context import ConceptHierarchyContext
from concept_hierarchy.data.expressions.expression import Expression
from concept_hierarchy.data.expressions.instantiated_value import ParsedCustomValue, ParsedStructural, ParsedValue
from concept_hierarchy.data.expressions.subexpressions import (
    FunctionEvaluation,
    InstExpression,
    NarrowExpression,
    PossibleFunctionEvaluationExpression,
    PossibleNarrowExpression,
    VerifiedTemplateDependentExpression,
)
from concept_hierarchy.errors import ConceptHierarchyError
from tests.integration.test_expression_parsing import (
    build_hierarchy,
    check_hierarchy,
    function_default_expressions,
    refuses_commitment,
)
from tests.integration.test_function_default_arguments import GROUND, function
from tests.integration.test_schema_substitution import obj, vd

# --------------------------------------------------------------------------------------------------
# Hierarchy fragments
#
# `test_expression_parsing.CH_PRELUDE` already supplies a clean `FunctionComposition` -- a one-key object
# keyed by a Function, with ``"properties": "args"`` inside and *no* mention of ``isFunctionEvaluation``.
# Keeping it that way is half of what this suite is for.
# --------------------------------------------------------------------------------------------------

FUNCTION_COMPOSITION_RES = {
    "FunctionCompositionRes": {
        "directParents": ["FunctionComposition"],
        "data": {"templateContext": ["T"], "instantiation": {"oneOf": ["T", "FunctionComposition"]}},
    }
}
"""The one genuinely ambiguous site: a Function-keyed object satisfies *both* branches when res(K) <= T."""

ADD = function("Add", {"arg1": ["Integer"], "arg2": ["Integer"], "res": "Integer"}, template=GROUND)
"""Two arguments and an `Integer` result."""

NULLARY = function("Nullary", {"res": "Integer"}, template=GROUND)
"""No arguments: ``{"Nullary": {}}`` is *both* its evaluation and its instantiation."""

VOID = function("Void", {"arg1": ["Integer"]}, template=GROUND)
"""Declares no ``res``. Composable, never evaluable at a typed site."""

STRINGY = function("Stringy", {"res": "String"}, template=GROUND)
"""A nullary Function whose result is a `String`, for the res-not-a-subtype cells."""

WEIRD = function("Weird", {"isFunctionEvaluation": ["Integer"], "res": "Integer"}, template=GROUND)
"""A Function with an argument *named* ``isFunctionEvaluation``. Perfectly legal, and must stay legal."""

LEAF = vd("Leaf", {"type": "object", "additionalProperties": False})
"""A key that names a perfectly good type which is not a Function."""

BOXY = vd("Boxy", {"oneOf": ["FunctionComposition", obj({"lhs": {"type": "Integer"}})]})
"""
`CustomFunction`'s shape, reduced.

This is the fragment behind ``Dog.f2`` in ``examples/animal_kingdom.json``: a Function-keyed object at a
site whose type is *not* a `FunctionComposition`, which is nevertheless valid -- but only through the
enclosing ValueDomain's instantiation schema, which the FEval reading never lets it reach.
"""

FLAGGED = vd("Flagged", obj({"isFunctionEvaluation": {"type": "Integer"}}))
"""A ValueDomain whose instantiation declares a property *named* ``isFunctionEvaluation``."""

PAIRED = vd(
    "Paired",
    {
        "type": "object",
        "additionalProperties": False,
        "required": ["lhs", "isFunctionEvaluation"],
        "properties": {"lhs": {"type": "Integer"}, "isFunctionEvaluation": {"type": "boolean"}},
    },
)
"""
A ValueDomain that requires **two** keys, one of them a *boolean* named ``isFunctionEvaluation``.

The trap: counting content keys makes this a one-key object, so the classifier pops the keyword and looks
at ``lhs`` -- which names no type. The keyword was data all along, and `Inst` has to get it back, or the
schema's ``required`` rejects a value that is perfectly well formed.
"""

STRUCTURAL = vd(
    "Structural",
    {
        "type": "object",
        "additionalProperties": False,
        "required": ["Add"],
        "properties": {"Add": {"type": "object"}},
    },
)
"""
The same hole without a boolean schema: an ordinary object schema that happens to match ``{"Add": {...}}``
structurally, with no custom-type leaf at the value's own location.
"""

SWALLOW = vd(
    "Swallow",
    {
        "type": "object",
        "additionalProperties": False,
        "required": ["Leaf"],
        "properties": {"Leaf": {"type": "object"}},
    },
)
"""
A ValueDomain whose instantiation happens to accept an object keyed ``Leaf``.

The trap for a misused marker: ``{"fEval:Leaf": {}}`` is a misuse -- `Leaf` names a type that is not a
Function -- but set the misuse aside and `Inst` matches this schema and accepts the value. Every other
site in this module rejects that object for an unrelated reason, so this is the only fixture that can tell
a real check from an accident.
"""

OPAQUE_COMPOSITION = vd("OpaqueComposition", True, parents=("FunctionComposition",))
"""
A `FunctionComposition` whose instantiation accepts everything.

The only way, once the keyword is gone, to reach an accept-everything schema under `NOT_AN_EVALUATION`:
the interpretation comes from the *type* -- a site whose type is a `FunctionComposition` and whose key
carries no marker -- rather than from anything written at the site. It is what keeps the "no reading is
insisted on" arm of the consumption check under test.
"""

ANYTHING = vd("Anything", True)
"""
A ValueDomain whose instantiation accepts everything -- which is also what a ValueDomain that declares no
``instantiation`` at all gets (`[CH].md` 8.6). A boolean schema has no property names to resolve and no
leaf to reach, so it matches *any* JSON without ever reading it as anything; it is the one site where a
commitment can be silently dropped no matter which reading was asked for.
"""

PERMISSIVE = vd(
    "Permissive",
    {
        "oneOf": [
            "ValueDomain",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["lhs"],
                "properties": {"lhs": {"type": "Integer"}},
            },
        ]
    },
)
"""
A ValueDomain with a ``"ValueDomain"`` leaf, which every Function result satisfies.

A ``true`` whose result does not fit the site is therefore *accepted* here -- which is what makes it the
fixture that can show a malformed evaluation being rejected on its own merits rather than for want of a
leaf that would have taken it.
"""

TAGGED = vd(
    "Tagged",
    {"type": "object", "additionalProperties": False, "required": ["v"], "properties": {"v": {"type": "Integer"}}},
    template={"order": ["N"], "N": "Literal:string"},
)
"""
A ValueDomain with a *string literal* template parameter, so its argument may contain anything -- colons
included. ``Tagged<"fEval:x">`` is what makes first-colon splitting necessary for a key-prefix marker.
"""

CUSTOM_FUNCTION_INSTANTIATION = {
    "oneOf": [
        "FunctionComposition",
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["procedure"],
            "properties": {"interface": {"type": "object"}, "procedure": {"type": "FunctionComposition"}},
        },
    ]
}
"""`CustomFunction`'s instantiation as declared: a composition, or an interface-and-procedure object."""

CUSTOM_FUNCTION = {
    "CustomFunction": {"directParents": ["ValueDomain"], "data": {"instantiation": CUSTOM_FUNCTION_INSTANTIATION}}
}

LABELLED = vd(
    "Labelled",
    {
        "type": "object",
        "additionalProperties": False,
        "required": ["lhs", "isFunctionEvaluation"],
        "properties": {"lhs": {"type": "Integer"}, "isFunctionEvaluation": {"type": "string"}},
    },
)
"""The same, with a *non-boolean* ``isFunctionEvaluation`` -- which is never a directive at all."""

HOLDER = vd(
    "Holder",
    {
        "type": "object",
        "additionalProperties": False,
        "required": [],
        "properties": {
            "comp": {"type": "FunctionComposition"},
            "res": {"type": "FunctionCompositionRes<Integer>"},
            "resStr": {"type": "FunctionCompositionRes<String>"},
            "fn": {"type": "Function"},
            "num": {"type": "Integer"},
            "boxy": {"type": "Boxy"},
            "flagged": {"type": "Flagged"},
            "paired": {"type": "Paired"},
            "labelled": {"type": "Labelled"},
            "resLeaf": {"type": "FunctionCompositionRes<Leaf>"},
            "resAnyValue": {"type": "FunctionCompositionRes<ValueDomain>"},
            "resAnyFunction": {"type": "FunctionCompositionRes<Function>"},
            "tagged": {"type": 'Tagged<"fEval:x">'},
            "cf": {"type": "CustomFunction"},
            "swallow": {"type": "Swallow"},
            "permissive": {"type": "Permissive"},
            "anything": {"type": "Anything"},
            "opaqueComp": {"type": "OpaqueComposition"},
            "structural": {"type": "Structural"},
            "nested": {"type": "Holder"},
        },
    },
)
"""
One site of each kind, so a test picks a site by naming a key. ``nested`` is what makes depth cheap.

``additionalProperties: false`` is load-bearing *for the tests themselves*: without it, ``{"Holder": ...}``
also matches `Holder`'s own instantiation as an `Inst` -- one unrecognized key, every property optional --
so a value that fails at its site is silently rescued one level up and the test passes for the wrong reason.
"""

CONCEPTS = {
    **FUNCTION_COMPOSITION_RES,
    **ADD,
    **NULLARY,
    **VOID,
    **STRINGY,
    **WEIRD,
    **LEAF,
    **BOXY,
    **FLAGGED,
    **CUSTOM_FUNCTION,
    **TAGGED,
    **STRUCTURAL,
    **SWALLOW,
    **ANYTHING,
    **OPAQUE_COMPOSITION,
    **PERMISSIVE,
    **PAIRED,
    **LABELLED,
    **HOLDER,
}


# --------------------------------------------------------------------------------------------------
# Accessors
# --------------------------------------------------------------------------------------------------


def check(instances: dict) -> ConceptHierarchyContext:
    return check_hierarchy(build_hierarchy(CONCEPTS, instances=instances))


def at(**fields: object) -> dict:
    """A global variable ``p`` that is a `Holder`, so that each key is a site of a known type."""
    return {"p": {"Holder": dict(fields)}}


def field(context: ConceptHierarchyContext, key: str, name: str = "p") -> Expression:
    """The parsed expression sitting at ``Holder.<key>`` of global variable ``name``."""
    narrow = context.model.instances[name].value.value
    assert isinstance(narrow, NarrowExpression), f"expected the global to be a Narrow, got {type(narrow).__name__}"
    parsed = narrow.value
    assert isinstance(parsed, ParsedStructural)
    node = parsed.properties_parsed[key]
    assert isinstance(node, ParsedCustomValue), f"expected a custom leaf at {key!r}, got {type(node).__name__}"
    assert node.expression is not None, f"nothing was parsed at {key!r}"
    return node.expression


def oneof_branch(expression: Expression) -> str | None:
    """
    The type of the ``oneOf`` branch the value actually matched, or ``None`` if the node has no ``oneOf``.

    At a ``FunctionCompositionRes<T>`` site both readings are an `InstExpression` at the top -- the whole
    point is that a composition is recognized by its schema -- so the class no longer separates them. Which
    branch of ``oneOf: ["T", "FunctionComposition"]`` was retained does, and it is the exact question the
    keyword answers.
    """
    parsed = getattr(expression.value, "value", None)
    while isinstance(parsed, ParsedStructural) and parsed.one_of_parsed is not None:
        branch = parsed.one_of_parsed
        custom_type = getattr(branch, "custom_type", None)
        if custom_type is not None:
            return custom_type.full_name
        parsed = branch  # a branch that is itself a ``oneOf``; the answer is one level further down
    return None


def field_kind(context: ConceptHierarchyContext, key: str) -> type:
    return type(field(context, key).value)


def sub_field(expression: Expression, key: str) -> Expression:
    """The expression at ``<key>`` of a `Holder` that is itself the value of an expression."""
    parsed = expression.value.value
    assert isinstance(parsed, ParsedStructural), f"expected a parsed object, got {type(parsed).__name__}"
    node = parsed.properties_parsed[key]
    assert isinstance(node, ParsedCustomValue), f"expected a custom leaf at {key!r}, got {type(node).__name__}"
    assert node.expression is not None, f"nothing was parsed at {key!r}"
    return node.expression


def expression_kinds(expression: Expression) -> list[type]:
    """
    Every `ExpressionValue` class reachable from ``expression``, itself included.

    `Expression.all_subexpressions` stops at an `InstExpression`, because what hangs off it is a
    `ParsedValue` and not an expression -- but a composition's arguments live exactly there, so the
    descent has to cross the boundary in both directions to say anything about nesting.
    """
    found: list[type] = []
    seen: set[int] = set()

    def walk_expression(expr: Expression | None) -> None:
        if expr is None or id(expr) in seen:
            return
        seen.add(id(expr))
        value = expr.value
        found.append(type(value))
        for argument in getattr(value, "arguments", {}).values():
            walk_expression(argument)
        for argument in getattr(value, "applied_defaults", {}).values():
            walk_expression(argument)
        parsed = getattr(value, "value", None)
        if isinstance(parsed, ParsedValue):
            walk_parsed(parsed)

    def walk_parsed(parsed: ParsedValue) -> None:
        if id(parsed) in seen:
            return
        seen.add(id(parsed))
        for node in parsed.walk():
            if isinstance(node, ParsedCustomValue) and node.expression is not None:
                walk_expression(node.expression)
            if isinstance(node, ParsedStructural):
                for entries in node.custom_expressions.values():
                    for _discriminator, sub in entries:
                        if isinstance(sub, ParsedCustomValue) and sub.expression is not None:
                            walk_expression(sub.expression)

    walk_expression(expression)
    return found


def rejection(instances: dict) -> ConceptHierarchyError:
    with pytest.raises(ConceptHierarchyError) as excinfo:
        check(instances)
    return excinfo.value


def error_sites(error: ConceptHierarchyError) -> list[tuple[str, str]]:
    """Every ``(location, message)`` in the error's cause tree, so a test can name the exact node."""
    found: list[tuple[str, str]] = []

    def walk(err: ConceptHierarchyError) -> None:
        location = "" if err.location_id is None else err.location_id.print()
        if err.part.value == "key":
            location += " (key)"
        found.append((location, err.args[0] if err.args else ""))
        for cause in err.causes:
            walk(cause)

    walk(error)
    return found


def assert_reported(error: ConceptHierarchyError, *, message: str, location: str) -> None:
    """A message fragment must appear on an error whose location *ends with* ``location``."""
    sites = error_sites(error)
    matching = [(loc, msg) for loc, msg in sites if message in msg]
    assert matching, f"no error said {message!r}; got:\n" + "\n".join(f"  [{loc}] {msg}" for loc, msg in sites)
    assert any(loc.endswith(location) for loc, _ in matching), (
        f"{message!r} was reported, but not at a location ending {location!r}; got:\n"
        + "\n".join(f"  [{loc}] {msg}" for loc, msg in matching)
    )


# ==================================================================================================
# 1. Where the keyword is a directive, and where it is ordinary data
# ==================================================================================================


class TestTheKeywordIsOrdinaryData:
    """
    ``isFunctionEvaluation`` is an identifier like any other, and these are the placements that say so.

    They were the *exceptions* while the key was also a classifier directive -- everything that was not a
    two-key object beside a type application -- and they are what is left once it is not. An argument may
    be named this, and so may a property of a ValueDomain's instantiation; reading either as a directive
    would silently delete a value the author wrote. ``examples/animal_kingdom.json`` contained exactly
    such a placement, one level too deep inside ``FunctionSequence``'s argument object.
    """

    def test_beside_a_key_that_is_not_a_type_it_is_an_argument_name(self):
        """``{"arg1": 1, "isFunctionEvaluation": true}`` is an argument list with a key `Add` lacks."""
        error = rejection(at(num={"Add": {"arg1": 1, "isFunctionEvaluation": True}}))
        assert_reported(error, message='does not have the argument "isFunctionEvaluation"', location='"num"')

    def test_as_the_sole_key_it_is_a_property_name(self):
        """`Flagged` declares a property with that name, and is entitled to."""
        context = check(at(flagged={"isFunctionEvaluation": 3}))
        assert field_kind(context, "flagged") is InstExpression

    def test_a_value_domain_may_declare_a_property_of_that_name(self):
        """The same point from the schema's side, and the reason the identifier is not reserved."""
        assert field(check(at(flagged={"isFunctionEvaluation": 3})), "flagged").is_valid

    def test_among_three_keys_it_is_an_argument_name(self):
        context = check(at(num={"Weird": {"isFunctionEvaluation": 3}}))
        assert field_kind(context, "num") is FunctionEvaluation

    def test_a_function_may_declare_an_argument_of_that_name(self):
        context = check(at(num={"Weird": {"isFunctionEvaluation": 3}}))
        evaluation = field(context, "num").value
        assert isinstance(evaluation, FunctionEvaluation)
        assert sorted(evaluation.arguments) == ["isFunctionEvaluation"]

    def test_a_marker_and_an_argument_of_that_name_coexist(self):
        """
        The marker is on the key and the argument is inside the value, so the two cannot collide -- which
        is the structural reason a marker needs no recognition rule, where the keyword needed one.
        """
        context = check(at(num={"fEval:Weird": {"isFunctionEvaluation": 3}}))
        evaluation = field(context, "num").value
        assert isinstance(evaluation, FunctionEvaluation)
        assert sorted(evaluation.arguments) == ["isFunctionEvaluation"]


# ==================================================================================================
# 2. Ordinary sites: tau is not a FunctionComposition
# ==================================================================================================


class TestOrdinarySiteWithoutTheKeyword:
    def test_a_function_keyed_object_is_an_evaluation(self):
        context = check(at(num={"Add": {"arg1": 1, "arg2": 2}}))
        assert field_kind(context, "num") is FunctionEvaluation

    def test_an_argument_less_function_is_an_evaluation_by_default(self):
        context = check(at(num={"Nullary": {}}))
        assert field_kind(context, "num") is FunctionEvaluation

    def test_a_result_that_is_not_a_subtype_is_a_hard_failure(self):
        """`FEval` does not fall through to `Narrow`: once committed, ``res(K) <= tau`` is fatal."""
        error = rejection(at(num={"Stringy": {}}))
        assert_reported(error, message="not a subtype", location='"num"')

    def test_a_function_value_cannot_be_written_without_the_keyword(self):
        """``{"Nullary": {}}`` at a `Function` site is the evaluation, whose `Integer` result is not one."""
        error = rejection(at(fn={"Nullary": {}}))
        assert_reported(error, message="not a subtype", location='"fn"')


class TestOrdinarySiteWithTheEvaluationMarker:
    """`fEval:` restates the default here. It is accepted, and it changes nothing."""

    def test_the_marker_is_accepted_and_redundant(self):
        context = check(at(num={"fEval:Add": {"arg1": 1, "arg2": 2}}))
        assert field_kind(context, "num") is FunctionEvaluation

    def test_the_marker_produces_the_same_expression_as_omitting_it(self):
        marked = field(check(at(num={"fEval:Nullary": {}})), "num")
        without = field(check(at(num={"Nullary": {}})), "num")
        assert type(marked.value) is type(without.value)
        assert marked.value.value_type.full_name == without.value.value_type.full_name

    def test_the_marker_does_not_rescue_a_result_that_is_not_a_subtype(self):
        """It names the reading; it does not waive the reading's own condition."""
        error = rejection(at(num={"fEval:Stringy": {}}))
        assert_reported(error, message="not a subtype", location='"num"')


class TestOrdinarySiteWithFalse:
    """
    The first load-bearing cell, and the one that split in two: `fInst:` asks for the `Narrow`, `fComp:`
    for the enclosing `Inst`. `isFunctionEvaluation: false` said only "not the evaluation" and let the
    cascade try both in turn, which is why `Dog.f2` and an argument-less Function's value were the same
    spelling; now they are not.
    """

    def test_the_instantiation_marker_narrows_an_argument_less_function(self):
        context = check(at(fn={"fInst:Nullary": {}}))
        expression = field(context, "fn")
        assert isinstance(expression.value, NarrowExpression)
        assert expression.value.value_type.full_name == "Nullary"

    def test_the_composition_marker_reaches_the_enclosing_instantiation_schema(self):
        """`Add` is not a subtype of `Boxy`, so `Narrow` fails and only `Inst` can accept this."""
        context = check(at(boxy={"fComp:Add": {"arg1": 1, "arg2": 2}}))
        assert field_kind(context, "boxy") is InstExpression

    def test_the_same_value_unmarked_is_rejected_at_that_site(self):
        """The contrast that shows the cell is load-bearing rather than decorative."""
        error = rejection(at(boxy={"Add": {"arg1": 1, "arg2": 2}}))
        assert_reported(error, message="not a subtype", location='"boxy"')

    def test_the_instantiation_marker_with_arguments_is_rejected(self):
        """A Function's instantiation is the empty object; arguments have nowhere to go."""
        error = rejection(at(fn={"fInst:Add": {"arg1": 1, "arg2": 2}}))
        assert_reported(error, message="Add", location='"fn"')

    def test_the_marker_reports_the_abandoned_alternatives_when_nothing_matches(self):
        error = rejection(at(num={"fInst:Nullary": {}}))
        messages = " ".join(message for _, message in error_sites(error))
        assert "Narrow" in messages or "not a subtype" in messages


# ==================================================================================================
# 3. A plain FunctionComposition site
# ==================================================================================================


class TestFunctionCompositionSite:
    def test_absent_is_the_composition(self):
        context = check(at(comp={"Add": {"arg1": 1, "arg2": 2}}))
        assert field_kind(context, "comp") is InstExpression

    def test_false_is_the_composition(self):
        context = check(at(comp={"fComp:Add": {"arg1": 1, "arg2": 2}}))
        assert field_kind(context, "comp") is InstExpression

    def test_absent_and_false_agree(self):
        absent = field(check(at(comp={"Add": {"arg1": 1, "arg2": 2}})), "comp")
        false = field(check(at(comp={"fComp:Add": {"arg1": 1, "arg2": 2}})), "comp")
        assert type(absent.value) is type(false.value)

    def test_a_function_that_returns_nothing_may_be_composed(self):
        """Composition lifts the ``res`` restriction that evaluation imposes."""
        context = check(at(comp={"Void": {"arg1": 1}}))
        assert field_kind(context, "comp") is InstExpression

    def test_true_has_no_branch_to_select_here_so_it_is_rejected(self):
        """
        `FunctionComposition`'s declared instantiation is the one-key object form and nothing else -- no
        custom-type leaf that could read the value as a Function evaluation. Its schema *would* match the
        object structurally, as the composition it looks like, but that would accept the value while
        ignoring what the keyword said, so it is rejected instead.

        Still the schema author's call: adding a branch that accepts an evaluation is what makes the
        keyword usable here, exactly as it does for `CustomFunction`.
        """
        error = rejection(at(comp={"fEval:Add": {"arg1": 1, "arg2": 2}}))
        assert_reported(error, message="without reading it as one", location='"comp"')

    def test_true_on_a_function_that_returns_nothing_is_rejected_here_too(self):
        """The reason is the absent branch, not the absent result type."""
        error = rejection(at(comp={"fEval:Void": {"arg1": 1}}))
        assert_reported(error, message="without reading it as one", location='"comp"')

    def test_a_bad_argument_of_a_composition_is_reported_at_that_argument(self):
        error = rejection(at(comp={"Add": {"arg1": "s:x", "arg2": 2}}))
        assert_reported(error, message="arg1", location='"comp": "Add"')

    def test_an_unknown_argument_of_a_composition_is_reported(self):
        error = rejection(at(comp={"Add": {"arg1": 1, "nope": 2}}))
        assert_reported(error, message='does not have the argument "nope"', location='"comp": "Add"')

    def test_a_key_that_is_not_a_function_is_rejected(self):
        error = rejection(at(comp={"Leaf": {}}))
        assert_reported(error, message="Leaf", location='"comp"')


# ==================================================================================================
# 4. A FunctionCompositionRes<T> site -- the one genuine ambiguity
# ==================================================================================================


class TestFunctionCompositionResSite:
    """
    ``oneOf: ["T", "FunctionComposition"]`` accepts a Function-keyed object under **both** branches
    whenever ``res(K) <= T``. Without a decision carried in from the site, this is
    "Value matches 2 schemas in 'oneOf'".
    """

    def test_absent_takes_the_function_composition_branch(self):
        context = check(at(res={"Add": {"arg1": 1, "arg2": 2}}))
        assert oneof_branch(field(context, "res")) == "FunctionComposition"

    def test_false_takes_the_function_composition_branch(self):
        context = check(at(res={"fComp:Add": {"arg1": 1, "arg2": 2}}))
        assert oneof_branch(field(context, "res")) == "FunctionComposition"

    def test_true_takes_the_t_branch(self):
        """The whole design in one assertion: the same object, the other branch, because of the keyword."""
        context = check(at(res={"fEval:Add": {"arg1": 1, "arg2": 2}}))
        assert oneof_branch(field(context, "res")) == "Integer"

    def test_true_checks_the_result_against_the_template_argument(self):
        """``res(Stringy) = String`` is not the `Integer` that ``FunctionCompositionRes<Integer>`` promises."""
        error = rejection(at(res={"fEval:Stringy": {}}))
        assert_reported(error, message="not a subtype", location='"res"')

    def test_true_accepts_a_result_of_the_promised_type(self):
        context = check(at(resStr={"fEval:Stringy": {}}))
        assert oneof_branch(field(context, "resStr")) == "String"

    def test_true_on_a_function_that_returns_nothing_is_rejected(self):
        """Unlike a plain `FunctionComposition` site, this one promises a value."""
        error = rejection(at(res={"fEval:Void": {"arg1": 1}}))
        assert_reported(error, message="Void", location='"res"')

    def test_a_non_object_value_still_reaches_the_t_branch(self):
        """The ``"T"`` branch is not only for Function-keyed objects; a bare literal is its plain case."""
        context = check(at(res=3))
        assert field(context, "res").is_valid

    def test_a_variable_does_not_fall_through_to_the_t_branch(self):
        """
        `Var` is checked before `Inst` and does **not** fall through: once a string resolves to a variable
        in scope, that is the reading, and a type mismatch is the end of the classification rather than a
        reason to try the instantiation schema.

        So the ``"T"`` branch's coercion is unreachable for a variable, even though the literal ``3`` at
        the same site reaches it through `Inst`. That asymmetry is the cascade working as specified
        (`[CH].md` 10.2), not a defect: the alternative -- retrying an in-scope name as an instantiation --
        would make a variable's meaning depend on whether its type happened to fit.
        """
        error = rejection({"n": 3, "p": {"Holder": {"res": "n"}}})
        assert_reported(error, message="Type Integer of variable n is not a subtype of", location='"res"')

    def test_a_narrow_of_a_non_function_type_is_unambiguous(self):
        """``{"Integer": 1}`` matches the ``"T"`` branch only: the key is not a Function."""
        context = check(at(res={"Integer": 1}))
        assert field(context, "res").is_valid

    def test_the_ambiguity_is_never_reported_as_a_oneof_failure(self):
        """
        The regression this whole design exists to prevent. If the decision is not carried into the
        schema, both branches match and the value parser says so.
        """
        for value in ({"Add": {"arg1": 1, "arg2": 2}}, {"fEval:Add": {"arg1": 1, "arg2": 2}}):
            try:
                context = check(at(res=value))
            except ConceptHierarchyError as error:
                messages = " ".join(message for _, message in error_sites(error))
                assert "matches 2 schemas" not in messages, f"ambiguity leaked for {value}"
                raise
            assert field(context, "res").is_valid


# ==================================================================================================
# 5. Scope: the decision applies to one value, never to what is inside it
# ==================================================================================================


class TestTheDecisionDoesNotLeakDownwards:
    """
    A composition suppresses the evaluation reading of *the object it is*, and of nothing below it.

    An argument of a composed Function is an ordinary site again, and a Function-keyed object there is an
    ordinary evaluation. Getting this wrong is the difference between a scoped decision and a mode.
    """

    def test_an_argument_of_a_composition_is_still_an_evaluation(self):
        context = check(at(comp={"Add": {"arg1": {"Nullary": {}}, "arg2": 2}}))
        kinds = expression_kinds(field(context, "comp"))
        assert FunctionEvaluation in kinds, f"the inner evaluation was suppressed; got {kinds}"

    def test_a_deeply_nested_argument_is_still_an_evaluation(self):
        context = check(at(comp={"Add": {"arg1": {"Add": {"arg1": {"Nullary": {}}, "arg2": 1}}, "arg2": 2}}))
        kinds = expression_kinds(field(context, "comp"))
        assert kinds.count(FunctionEvaluation) >= 2, f"expected nested evaluations; got {kinds}"

    def test_an_argument_of_a_composition_may_itself_be_a_composition(self):
        context = check(at(comp={"Void": {"arg1": {"Nullary": {}}}}))
        assert field(context, "comp").is_valid

    def test_a_composition_site_nested_under_an_ordinary_site(self):
        context = check(at(nested={"Holder": {"comp": {"Add": {"arg1": 1, "arg2": 2}}}}))
        nested = field(context, "nested")
        assert nested.is_valid
        assert type(sub_field(nested, "comp").value) is InstExpression

    def test_a_composition_holds_its_evaluation_inside_rather_than_being_one(self):
        """
        The distinction is at the top: ``"properties": "args"`` stores a `FunctionEvaluation` *inside* the
        composition's parsed value, so finding one below says nothing. Only the class at the site does.
        """
        context = check(at(comp={"Add": {"arg1": 1, "arg2": 2}}))
        composition = field(context, "comp")
        assert type(composition.value) is InstExpression
        assert FunctionEvaluation in expression_kinds(composition), "the args node stores the evaluation"

    def test_an_ordinary_site_nested_under_a_composition_site(self):
        context = check(at(nested={"Holder": {"num": {"Add": {"arg1": 1, "arg2": 2}}}}))
        kinds = expression_kinds(field(context, "nested"))
        assert FunctionEvaluation in kinds

    def test_the_same_shape_at_two_sites_of_different_kinds_reads_two_ways(self):
        """The single sharpest statement of the design: identical JSON, different site, different class."""
        shape = {"Add": {"arg1": 1, "arg2": 2}}
        context = check(at(num=dict(shape), comp=dict(shape)))
        assert field_kind(context, "num") is FunctionEvaluation
        assert field_kind(context, "comp") is InstExpression

    def test_three_levels_of_alternating_sites(self):
        context = check(at(nested={"Holder": {"comp": {"Add": {"arg1": {"Add": {"arg1": 1, "arg2": 2}}, "arg2": 2}}}}))
        assert field(context, "nested").is_valid


class TestAMarkerAtDepth:
    """A marker works the same however deep the site is; several tests pin the location of failures."""

    def test_the_instantiation_marker_at_depth_two(self):
        context = check(at(nested={"Holder": {"fn": {"fInst:Nullary": {}}}}))
        assert field(context, "nested").is_valid

    def test_the_evaluation_marker_at_depth_two(self):
        """`Permissive` is the site that can take it; the point here is that depth changes nothing."""
        context = check(at(nested={"Holder": {"permissive": {"fEval:Add": {"arg1": 1, "arg2": 2}}}}))
        assert field(context, "nested").is_valid

    def test_a_marker_inside_a_composition_argument(self):
        context = check(at(comp={"Add": {"arg1": {"fEval:Nullary": {}}, "arg2": 2}}))
        assert field(context, "comp").is_valid

    def test_a_misused_marker_at_depth_is_located_at_the_site(self):
        """
        The location differs from the keyword's, and has to: the keyword was a key of its own and the
        failure could be pinned there, whereas a marker is part of the content key, so the object as a
        whole is what is wrong.
        """
        error = rejection(at(nested={"Holder": {"num": {"fEval:Leaf": {}}}}))
        assert_reported(error, message="only qualifies a key that names a Function", location='"num"')

    def test_a_stray_keyword_inside_an_argument_object_at_depth(self):
        """
        ``animal_kingdom.json``'s old placement, exactly: one level too deep, inside the argument list.
        Nothing about this depends on the keyword being a directive -- it is an argument name that `Add`
        does not have -- so it stays as written once the keyword is gone.
        """
        error = rejection(at(comp={"Add": {"arg1": 1, "arg2": 2, "isFunctionEvaluation": True}}))
        assert_reported(error, message='does not have the argument "isFunctionEvaluation"', location='"comp": "Add"')


# ==================================================================================================
# 6. What the ExpressionValue records, since nothing is rewritten
# ==================================================================================================


class TestTheExpressionValueIsTheAnswer:
    """
    No ``Return`` wrapper is synthesised at parse time. The class of the `ExpressionValue` is the only
    place the reading is recorded, so it has to be exactly right in both directions.
    """

    def test_a_composition_is_never_a_function_evaluation(self):
        context = check(at(comp={"Add": {"arg1": 1, "arg2": 2}}))
        assert not isinstance(field(context, "comp").value, FunctionEvaluation)

    def test_a_value_at_a_composition_site_is_always_an_inst(self):
        """
        Every reading that a `FunctionComposition` site accepts is an `InstExpression` at the top, because
        a composition is recognized by its instantiation schema and by nothing else -- the top-level
        `FEval` alternative is never even tried there, which is what stopped the parser asking whether
        `Boolean` is a subtype of ``FunctionCompositionRes<Boolean>``.
        """
        for value in (
            {"Add": {"arg1": 1, "arg2": 2}},
            {"fComp:Add": {"arg1": 1, "arg2": 2}},
        ):
            context = check(at(comp=value))
            assert isinstance(field(context, "comp").value, InstExpression), value

    def test_the_evaluation_selected_by_true_is_reachable_inside(self):
        """It is one level down, at the ``"T"`` leaf the keyword selected."""
        context = check(at(res={"fEval:Add": {"arg1": 1, "arg2": 2}}))
        expression = field(context, "res")
        assert oneof_branch(expression) == "Integer"
        evaluations = [k for k in expression_kinds(expression) if k is FunctionEvaluation]
        assert evaluations, f"no evaluation under the T branch; got {expression_kinds(expression)}"

    def test_the_expression_type_is_the_site_type_not_the_result_type(self):
        """Even for the ``true`` cell: the expression sits at the site, whatever the evaluation returns."""
        context = check(at(res={"fEval:Add": {"arg1": 1, "arg2": 2}}))
        assert field(context, "res").required_expression_type.full_name == "FunctionCompositionRes<Integer>"

    def test_the_two_readings_of_one_shape_differ_only_by_the_keyword(self):
        composition = field(check(at(res={"Add": {"arg1": 1, "arg2": 2}})), "res")
        evaluation = field(check(at(res={"fEval:Add": {"arg1": 1, "arg2": 2}})), "res")
        assert type(composition.value) is type(evaluation.value) is InstExpression, "both are instantiations"
        assert oneof_branch(composition) == "FunctionComposition"
        assert oneof_branch(evaluation) == "Integer"


# ==================================================================================================
# 7. The schemas stay clean
# ==================================================================================================


# ==================================================================================================
# 8. The carried verdict is not the written keyword
# ==================================================================================================


class TestTheCarriedVerdictIsNotTheWrittenKeyword:
    """
    A `FunctionComposition`-family site rules the evaluation reading out **from tau**, with no keyword in
    the value at all, and that verdict is carried into the instantiation schema so the branches cannot
    re-decide it. It is therefore tempting to implement the verdict by pretending the keyword was written
    -- setting the same ``is_function_evaluation_present`` flag the JSON sets.

    That conflation has two consequences, and both are tested here rather than reasoned about, because
    neither shows up on the cells of the classification table:

    * ``ensure_unmodified_json_value`` restores a *popped* keyword by writing it back. Told the keyword was
      present when it never was, it **adds** a key to the caller's JSON;
    * ``_parse_expression_of_json_object`` raises a misuse error
      when the flag is present and the key is not a Function. Told the same lie, it reports a keyword
      misuse against a value containing no keyword -- and it *raises*, inside a ``oneOf`` trial branch,
      where `child_silent` catches nothing but `StopValidation`.

    The two must stay separate: what the JSON says governs popping, restoring and the misuse error; the
    carried verdict governs only whether the evaluation reading is available.
    """

    @staticmethod
    def unchanged(instances: dict) -> dict:
        """Check ``instances`` (accepted or not) and return it, so the caller can compare it to a snapshot."""
        try:
            check(instances)
        except ConceptHierarchyError:
            pass  # the value may be rejected; the input must be intact either way
        return instances

    def test_no_keyword_is_added_at_a_composition_site(self):
        instances = at(comp={"Add": {"arg1": 1, "arg2": 2}})
        before = copy.deepcopy(instances)
        assert self.unchanged(instances) == before

    def test_no_keyword_is_added_at_a_composition_res_site(self):
        """The site whose verdict comes from tau alone, so nothing was written to restore."""
        instances = at(res={"Add": {"arg1": 1, "arg2": 2}})
        before = copy.deepcopy(instances)
        assert self.unchanged(instances) == before

    def test_no_keyword_is_added_at_an_ordinary_site(self):
        instances = at(num={"Add": {"arg1": 1, "arg2": 2}})
        before = copy.deepcopy(instances)
        assert self.unchanged(instances) == before

    def test_no_keyword_is_added_at_depth(self):
        instances = at(nested={"Holder": {"comp": {"Add": {"arg1": {"Nullary": {}}, "arg2": 2}}}})
        before = copy.deepcopy(instances)
        assert self.unchanged(instances) == before

    def test_no_keyword_is_added_when_the_value_is_rejected(self):
        """The restore runs on the error path too, so a rejected value is the sharper case."""
        instances = at(comp={"Leaf": {}})
        before = copy.deepcopy(instances)
        assert self.unchanged(instances) == before

    def test_a_written_keyword_is_restored_at_a_composition_site(self):
        """The other direction: what *was* written must come back, popped or not."""
        instances = at(comp={"fEval:Add": {"arg1": 1, "arg2": 2}})
        before = copy.deepcopy(instances)
        assert self.unchanged(instances) == before

    def test_a_written_keyword_is_restored_when_the_value_is_rejected(self):
        instances = at(num={"fEval:Leaf": {}})
        before = copy.deepcopy(instances)
        assert self.unchanged(instances) == before

    def test_a_carried_verdict_is_not_reported_as_a_keyword_misuse(self):
        """
        ``{"Leaf": {}}`` at a ``FunctionCompositionRes<Leaf>`` site is an ordinary `Narrow` through the
        ``"T"`` branch: `Leaf` is a subtype of `Leaf`, and ``{}`` is its instantiation. The
        `FunctionComposition` branch fails because `Leaf` is not a Function, so exactly one matches.

        The verdict *is* carried here -- the site is a `FunctionComposition` -- and the key is not a
        Function, which is precisely the pair that triggers the misuse error if the two are conflated.
        """
        context = check(at(resLeaf={"Leaf": {}}))
        assert field(context, "resLeaf").is_valid

    def test_a_carried_verdict_does_not_abort_the_parse(self):
        """
        The misuse error is a ``raise``, not a recorded error, and a ``oneOf``'s branches are *trial*
        branches whose `child_silent` catches nothing but `StopValidation`. So a carried verdict that can
        trigger it does not merely fail one branch -- it tears down the whole parse.

        `Leaf` at a ``FunctionCompositionRes<Leaf>`` site is the pair that triggers it if the two are
        conflated: the verdict is carried, and the key names a type that is a subtype of the site's ``T``
        but is not a Function. Checking that a *later* global still parsed is what tells an aborted parse
        apart from a branch that lost quietly -- the value's own validity cannot, since an abort would
        surface as a rejection of this very value.
        """
        context = check({"p": {"Holder": {"resLeaf": {"Leaf": {}}}}, "q": {"Holder": {"num": 7}}})
        assert field(context, "resLeaf").is_valid
        assert field(context, "num", name="q").is_valid, "the parse continued past the trial branch"

    def test_a_failing_trial_branch_stays_quiet(self):
        """
        The other half: the ``"T"`` branch must be *able* to fail without the composition branch losing.
        ``Nullary`` is not a subtype of `Leaf` and its `Integer` result is not one either, so only the
        `FunctionComposition` branch can accept it.
        """
        context = check(at(resLeaf={"Nullary": {}}))
        assert type(field(context, "resLeaf").value) is InstExpression


# ==================================================================================================
# 9. The pop is provisional: a keyword that turns out to be data must be given back
# ==================================================================================================


class TestTheKeywordReachesTheSchemaUntouched:
    """
    A value carrying the key arrives at the instantiation schema exactly as written.

    While the key was also a directive this was delicate: it had to be popped *before* the key beside it
    could be resolved -- content keys have to be counted to know there is a single one -- so the pop was
    a guess, wrong whenever the key turned out to name no type, and it had to be undone. Nothing in the
    shape told the cases apart: ``{"Add": {...}, "isFunctionEvaluation": true}`` and
    ``{"lhs": 1, "isFunctionEvaluation": true}`` were both "one content key plus the keyword".

    Getting it wrong was silent -- the value reached `Inst` one key short and the schema rejected it for
    a missing property the author did write -- so these stay as a guard, now against a much shorter path.
    """

    def test_a_boolean_keyword_beside_a_non_type_key_reaches_the_schema(self):
        context = check(at(paired={"lhs": 1, "isFunctionEvaluation": True}))
        assert field(context, "paired").is_valid

    def test_both_boolean_values_reach_the_schema(self):
        """``false`` is the value that used to look most like a directive."""
        context = check(at(paired={"lhs": 1, "isFunctionEvaluation": False}))
        assert field(context, "paired").is_valid

    def test_the_schema_really_does_require_the_keyword(self):
        """Without this, the two above would pass even if the key were dropped."""
        error = rejection(at(paired={"lhs": 1}))
        assert_reported(error, message="isFunctionEvaluation", location='"paired"')

    def test_a_non_boolean_value_reaches_the_schema_too(self):
        context = check(at(labelled={"lhs": 1, "isFunctionEvaluation": "a string value"}))
        assert field(context, "labelled").is_valid

    def test_beside_a_function_key_a_non_boolean_value_is_data(self):
        """
        ``{"Add": {...}, "isFunctionEvaluation": "yes"}`` is two content keys, so no `FEval` and no
        `Narrow` is attempted -- the value goes straight to `Inst`, and at an `Integer` site it fails as
        the object it is, not as a misused keyword.
        """
        error = rejection(at(num={"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": "yes"}))
        messages = " ".join(message for _, message in error_sites(error))
        assert "Invalid use of the" not in messages, f"read as a directive: {messages[:300]}"
        assert "not of type 'integer'" in messages, messages[:300]

    def test_the_input_is_not_modified(self):
        instances = at(paired={"lhs": 1, "isFunctionEvaluation": True})
        before = copy.deepcopy(instances)
        check(instances)
        assert instances == before

    def test_the_input_is_not_modified_for_a_non_boolean_value(self):
        instances = at(labelled={"lhs": 1, "isFunctionEvaluation": "a string value"})
        before = copy.deepcopy(instances)
        check(instances)
        assert instances == before


# ==================================================================================================
# 10. Whether `true` is admissible is the schema author's decision
# ==================================================================================================


class TestTheSchemaDecidesWhetherTrueIsAdmissible:
    """
    ``"isFunctionEvaluation": true`` commits the object to being a Function evaluation. Where the site's
    own type cannot take one -- ``res(K)`` is not a subtype of it -- the reading is carried into the site's
    instantiation schema, and a **custom-type leaf** there may accept it if ``res(K)`` fits *that* type.

    Whether any leaf can is a property of the schema, so the Concept Hierarchy's author decides whether the
    keyword is usable at a given site, without the parser holding an opinion. The two hierarchies below
    differ in exactly one branch and give opposite answers to the same value.
    """

    VALUE = {"fEval:Add": {"arg1": 1, "arg2": 2}}
    """``res(Add) = Integer``, which is not a subtype of `CustomFunction`."""

    @staticmethod
    def with_custom_function(instantiation: object, ch_verbose: bool = False) -> dict:
        concepts = {
            **CONCEPTS,
            "CustomFunction": {"directParents": ["ValueDomain"], "data": {"instantiation": instantiation}},
        }
        ch = build_hierarchy(concepts, instances=at(cf=dict(TestTheSchemaDecidesWhetherTrueIsAdmissible.VALUE)))
        if ch_verbose:
            print(ch)
        return ch

    def test_the_declared_instantiation_rejects_it(self):
        """
        `CustomFunction` as declared: ``oneOf: ["FunctionComposition", {interface, procedure}]``.

        The `FunctionComposition` branch *is* a custom-type leaf, but no Function returns a
        `FunctionComposition`, so it cannot accept an evaluation; the interface/procedure branch is
        structural and does not match a Function-keyed object at all. Nothing accepts it, and the value is
        rejected -- rather than quietly re-read as one of the readings the keyword ruled out.
        """
        with pytest.raises(ConceptHierarchyError) as excinfo:
            check_hierarchy(self.with_custom_function(CUSTOM_FUNCTION_INSTANTIATION, True))
        messages = " ".join(message for _, message in error_sites(excinfo.value))
        assert refuses_commitment(messages, "Function evaluation"), messages[:400]

    def test_adding_a_value_domain_branch_admits_it(self):
        """
        The same value, against ``oneOf: ["ValueDomain", <the declared instantiation>]``.

        ``"ValueDomain"`` is a custom-type leaf that every result type satisfies, so it takes the
        evaluation; the declared branches still refuse it, so ``oneOf`` gets exactly one match.
        """
        context = check_hierarchy(self.with_custom_function({"oneOf": ["ValueDomain", CUSTOM_FUNCTION_INSTANTIATION]}))
        expression = field(context, "cf")
        assert expression.is_valid
        assert oneof_branch(expression) == "ValueDomain"
        assert FunctionEvaluation in expression_kinds(expression)

    def test_the_permissive_schema_still_takes_the_composition_branch_for_false(self):
        """
        The added branch must not swallow the values the declared ones were there for.

        ``false`` is what a composition needs at this site -- ``Dog.f2``'s case: `Add` is not a subtype of
        `CustomFunction`, so without it the default evaluation reading commits and hard-fails before `Inst`
        is ever tried. With it, the `FunctionComposition` branch takes the value and ``"ValueDomain"``
        does not, because the evaluation reading is off the table there too.
        """
        concepts = {
            **CONCEPTS,
            "CustomFunction": {
                "directParents": ["ValueDomain"],
                "data": {"instantiation": {"oneOf": ["ValueDomain", CUSTOM_FUNCTION_INSTANTIATION]}},
            },
        }
        instances = at(cf={"fComp:Add": {"arg1": 1, "arg2": 2}})
        context = check_hierarchy(build_hierarchy(concepts, instances=instances))
        expression = field(context, "cf")
        assert expression.is_valid
        assert oneof_branch(expression) == "FunctionComposition"


# ==================================================================================================
# 11. Which failures may be set aside for the schema, and which may not
# ==================================================================================================


class TestOnlyASiteTypeFailureIsSetAsideForTheSchema:
    """
    A written ``true`` that does not fit the site is carried into the site's instantiation schema instead
    of rejecting the value, which means the `FEval` failure produced along the way is *dropped*. Dropping
    the wrong one is silent, so the rule has to be exact:

    > a failure is set aside **iff** it is about the site's expected type; a failure about the evaluation
    > itself never is.

    Read end to end, exactly two of `parse_function_evaluation_expression`'s rejections mention the
    expected type -- ``res(K)`` is not a subtype of it, and the Function returns nothing where one was
    expected -- and those two are the ones that may be set aside. Every other rejection (an abstract key,
    an argument object that is not an object, an unknown argument, a missing one, a bad argument value, a
    cyclic default) holds whatever the site expects, so no leaf could have accepted it either.

    Both directions are pinned, because pinning only the first would let the rule drift open.
    """

    # -- set aside: the evaluation is well formed, it just does not fit *here* ----------------------

    def test_a_result_that_does_not_fit_the_site_is_set_aside(self):
        """`Add` returns an `Integer`, which is not a `Permissive` -- but it is a `ValueDomain`."""
        context = check(at(permissive={"fEval:Add": {"arg1": 1, "arg2": 2}}))
        assert oneof_branch(field(context, "permissive")) == "ValueDomain"

    def test_a_result_of_another_type_is_also_set_aside(self):
        context = check(at(permissive={"fEval:Stringy": {}}))
        assert oneof_branch(field(context, "permissive")) == "ValueDomain"

    # -- never set aside: the evaluation is malformed, so no leaf could take it ---------------------

    def test_an_argument_object_that_is_not_an_object_is_not_set_aside(self):
        error = rejection(at(permissive={"fEval:Add": 5}))
        assert_reported(error, message="expected a JSON object", location='"permissive"')

    def test_an_unknown_argument_is_not_set_aside(self):
        error = rejection(at(permissive={"fEval:Add": {"nope": 1}}))
        assert_reported(error, message='does not have the argument "nope"', location='"permissive"')

    def test_a_missing_required_argument_is_not_set_aside(self):
        error = rejection(at(permissive={"fEval:Add": {"arg1": 1}}))
        assert_reported(error, message="missing from the Function evaluation interface", location='"permissive"')

    def test_an_abstract_key_is_not_set_aside(self):
        """`FunctionReturning` is abstract, so it cannot be evaluated whatever the site would accept."""
        error = rejection(at(permissive={"fEval:FunctionReturning<Integer>": {}}))
        assert_reported(error, message="abstract type", location='"permissive"')

    def test_a_bad_argument_value_is_not_set_aside(self):
        error = rejection(at(permissive={"fEval:Add": {"arg1": "s:x", "arg2": 2}}))
        assert_reported(error, message="arg1", location='"permissive"')

    def test_the_arguments_of_a_committed_evaluation_are_ordinary_sites(self):
        """
        The commitment is pinned to the value's own location, so it reaches a custom-type leaf sitting at
        that location and nothing below it. `Add`'s arguments are plain literals here: if the commitment
        leaked into them, only a Function evaluation would be admissible there and ``1`` would be rejected.
        """
        context = check(at(permissive={"fEval:Add": {"arg1": 1, "arg2": 2}}))
        assert oneof_branch(field(context, "permissive")) == "ValueDomain"

    def test_a_nested_evaluation_inside_a_committed_one_is_still_ordinary(self):
        """The same, one level deeper and with an argument that *is* an evaluation."""
        context = check(at(permissive={"fEval:Add": {"arg1": {"Nullary": {}}, "arg2": 2}}))
        expression = field(context, "permissive")
        assert oneof_branch(expression) == "ValueDomain"
        assert expression_kinds(expression).count(FunctionEvaluation) >= 2, expression_kinds(expression)

    def test_a_property_leaf_is_unaffected_by_the_commitment(self):
        """
        Only a leaf at the value's own location can honor it. `Permissive`'s other branch reaches an
        `Integer` leaf through the ``lhs`` *property*, one location further down, and parses as it always
        would.
        """
        context = check(at(permissive={"lhs": 7}))
        assert field(context, "permissive").is_valid


class TestAMisusedMarkerIsNeverSetAside:
    """
    The regression this class exists for. A misuse is about the *marker*, not about the site's type, so
    it must survive into the rejection. The failure it guards against is a real one that was fixed here:
    the misuse was set aside with the rest, and `Inst` then accepted the object at a `Swallow` site --
    schema matched, misuse gone.

    `Swallow` is the only site in this module whose schema accepts ``{"Leaf": {}}``, which is what makes
    these the tests that would have caught it; everywhere else the value is rejected either way.
    """

    def test_a_misuse_is_rejected_even_where_the_schema_would_accept_the_object(self):
        error = rejection(at(swallow={"fEval:Leaf": {}}))
        assert_reported(error, message="only qualifies a key that names a Function", location='"swallow"')

    def test_the_same_holds_for_every_marker(self):
        """The misuse is a marker on a non-Function key, whichever reading the marker names."""
        for marker in ("fComp", "fInst"):
            error = rejection(at(swallow={f"{marker}:Leaf": {}}))
            assert_reported(error, message="only qualifies a key that names a Function", location='"swallow"')

    def test_without_the_marker_that_very_value_is_accepted(self):
        """
        The control that proves the two above test the marker and not the fixture: the same object
        unmarked is a perfectly good `Swallow`.
        """
        context = check(at(swallow={"Leaf": {}}))
        assert field(context, "swallow").is_valid

    def test_a_misuse_at_a_permissive_site_is_rejected_too(self):
        """`Permissive`'s ``"ValueDomain"`` leaf accepts almost any result, but not a misused marker."""
        error = rejection(at(permissive={"fEval:Leaf": {}}))
        assert_reported(error, message="only qualifies a key that names a Function", location='"permissive"')


# ==================================================================================================
# 12. A schema may not accept a committed evaluation without reading it as one
# ==================================================================================================


class TestTheCommitmentMustBeHonoredNotMerelyMatched:
    """
    `fEval:` is honored at a **custom-type leaf**, and a schema can match the value without ever reaching
        one -- a structural schema that happens to fit, or a boolean one. Such a match would accept the value
        with the marker silently ignored.

        Two readings of "was it honored" are wrong, and both are pinned below:

        * *does a Function evaluation appear anywhere below?* -- a **composition** holds one too
          (``"properties": "args"`` stores it), so this answers yes for the reading the keyword ruled out;
        * *does one appear at any depth?* -- an **argument** of the value may be an evaluation, so this
          answers yes for a value that was not read as one at all.

        What must hold is that the node standing at the value's **own location** is the evaluation. Depth is
        not a factor: ``$ref`` and the composite keywords keep the location, only a property or item descent
        changes it, so the consumer is always exactly there however many schema nodes were crossed.
    """

    def test_a_structural_match_does_not_honor_it(self):
        """
        `Structural` matches an object; no custom-type leaf is reached. The marker has to be written on
        the *nested* value, because `Structural` requires the key spelled exactly ``"Add"`` and a marker
        on that key would no longer be that property -- which is the same fact that gives the explicit
        `Narrow` its job (`[CH].md` 10.2).
        """
        error = rejection(at(structural={"Add": {"fEval:Nullary": {}}}))
        assert_reported(error, message="not a subtype", location='"structural"')

    def test_the_same_value_is_accepted_when_it_claims_not_to_be_an_evaluation(self):
        """
        The control: nothing about this value is wrong except the keyword nobody honored. ``false`` is
        what it takes to reach `Structural`'s schema at all -- with the keyword absent the default
        evaluation reading commits and hard-fails on ``res(Add)`` before `Inst` is tried.
        """
        context = check(at(structural={"Structural": {"Add": {"arg1": 1, "arg2": 2}}}))
        assert field(context, "structural").is_valid

    def test_a_leaf_at_the_value_s_own_location_does_honor_it(self):
        """`Permissive`'s root-level ``"ValueDomain"`` branch is such a leaf, and the value is accepted."""
        context = check(at(permissive={"fEval:Add": {"arg1": 1, "arg2": 2}}))
        assert oneof_branch(field(context, "permissive")) == "ValueDomain"

    def test_a_composition_holding_an_evaluation_does_not_count_as_honoring_it(self):
        """
        The first wrong reading. A composition of `Add` contains a `FunctionEvaluation` in
        ``custom_expressions``, so "contains one" would accept a `FunctionComposition` site's ``true`` --
        the reading the keyword explicitly ruled out.
        """
        composition = field(check(at(comp={"Add": {"arg1": 1, "arg2": 2}})), "comp")
        assert FunctionEvaluation in expression_kinds(composition), "the composition does hold one"
        error = rejection(at(comp={"fEval:Add": {"arg1": 1, "arg2": 2}}))
        assert_reported(error, message="without reading it as one", location='"comp"')

    def test_an_evaluation_among_the_arguments_does_not_count_either(self):
        """
        The second wrong reading. ``arg1`` is itself an evaluation, so a depth-unbounded search would find
        one -- but the outer value is not read as an evaluation by `Structural`, and must be rejected.
        """
        error = rejection(at(anything={"fEval:Add": {"arg1": {"Nullary": {}}, "arg2": 2}}))
        assert_reported(error, message="without reading it as one", location='"anything"')

    def test_a_trial_branch_that_lost_does_not_count(self):
        """
        Only the *retained* parse is inspected. `CustomFunction`'s declared ``oneOf`` has a
        `FunctionComposition` branch that is tried and fails; nothing it built may satisfy the check.
        """
        with pytest.raises(ConceptHierarchyError) as excinfo:
            check_hierarchy(
                build_hierarchy(
                    {
                        **CONCEPTS,
                        "CustomFunction": {
                            "directParents": ["ValueDomain"],
                            "data": {"instantiation": CUSTOM_FUNCTION_INSTANTIATION},
                        },
                    },
                    instances=at(cf={"fEval:Add": {"arg1": 1, "arg2": 2}}),
                )
            )
        messages = " ".join(message for _, message in error_sites(excinfo.value))
        assert refuses_commitment(messages, "Function evaluation"), messages[:400]


# ==================================================================================================
# 13. The ambiguity the keyword cannot express: Narrow versus composition
# ==================================================================================================


class TestNarrowVersusCompositionIsStillAmbiguous:
    """
    A directive that settles only *evaluation versus not* has nothing to say about the other two
    readings, and at a ``FunctionCompositionRes<T>`` whose ``T`` admits a Function type they collide:

    * the ``"T"`` branch matches as a **Narrow** -- ``Nullary`` is a subtype of `ValueDomain` / `Function`,
      and ``{}`` is exactly `Nullary`'s instantiation (a Function's default, ``maxProperties: 0``);
    * the `FunctionComposition` branch matches as a **composition** of the same Function.

    It takes an argument-less Function to collide, because the two readings need the same JSON: a Function
    with arguments has a non-empty object, which its own instantiation refuses.

    These are the tests behind the "a new keyword is needed" conclusion, and the ``false`` row is the one
    worth keeping: it is the reading the keyword *can* select, and it selects neither of the two that
    collided. The hint the parser prints at exactly this point now names the three markers instead, which
    is advice that can be acted on -- `TestTheMarkersResolveTheNarrowVersusCompositionCollision` acts on it.
    """

    ARGUMENTLESS = {"Nullary": {}}

    def test_a_function_typed_t_collides(self):
        error = rejection(at(resAnyFunction=dict(self.ARGUMENTLESS)))
        assert "matches 2 schemas" in " ".join(m for _, m in error_sites(error))

    def test_a_value_domain_typed_t_collides(self):
        error = rejection(at(resAnyValue=dict(self.ARGUMENTLESS)))
        assert "matches 2 schemas" in " ".join(m for _, m in error_sites(error))

    def test_the_parser_prints_its_hint_here(self):
        """The hint in `_parse_one_of` is live, not dead code -- this is the value that reaches it."""
        error = rejection(at(resAnyFunction=dict(self.ARGUMENTLESS)))
        messages = " ".join(m for _, m in error_sites(error))
        assert "you may want to say which reading is meant" in messages, messages[:300]

    def test_the_hint_names_all_three_markers(self):
        """Which one is *acceptable* depends on the branch types, so the hint offers all three."""
        error = rejection(at(resAnyFunction=dict(self.ARGUMENTLESS)))
        messages = " ".join(m for _, m in error_sites(error))
        for marker in ('"fEval:Nullary"', '"fComp:Nullary"', '"fInst:Nullary"'):
            assert marker in messages, f"{marker} missing from {messages[:400]}"

    def test_the_hint_no_longer_recommends_the_keyword(self):
        """It used to name ``isFunctionEvaluation``, which provably cannot separate these two branches."""
        error = rejection(at(resAnyFunction=dict(self.ARGUMENTLESS)))
        hints = [m for _, m in error_sites(error) if "you may want to say" in m]
        assert hints and not any("isFunctionEvaluation" in m for m in hints), hints[:1]

    def test_a_key_that_already_carries_a_marker_is_not_offered_one(self):
        """
        A marked key that still matches two branches is ambiguous for some *other* reason, and repeating
        the advice would misdirect. ``Void`` declares no ``res``, so ``fEval:`` cannot be honored here.
        """
        error = rejection(at(resAnyFunction={"fEval:Void": {"arg1": 1}}))
        messages = " ".join(m for _, m in error_sites(error))
        assert "you may want to say which reading is meant" not in messages, messages[:300]

    def test_neither_branch_is_an_evaluation_so_ruling_it_out_settles_nothing(self):
        """
        The heart of it, and the reason the markers exist. Both colliding readings are *not* evaluations,
        so excluding the evaluation excludes neither -- which is exactly what the site's own type already
        does here, with nothing written. Whatever says "not an evaluation" and no more cannot separate
        them; only naming one of the two can.
        """
        for key in ("resAnyFunction", "resAnyValue"):
            error = rejection(at(**{key: dict(self.ARGUMENTLESS)}))
            messages = " ".join(m for _, m in error_sites(error))
            assert "matches 2 schemas" in messages, f"{key}: {messages[:300]}"

    def test_the_evaluation_marker_selects_neither_of_them_but_a_third_reading(self):
        """
        `fEval:` resolves the ``oneOf``, but by switching to the *evaluation* -- a reading neither branch
        offered. It therefore succeeds or fails on ``res(K)``, not on the collision: `Nullary` returns an
        `Integer`, which is a `ValueDomain` but not a `Function`. Only `fComp:` and `fInst:` name one of
        the two that actually collided.
        """
        context = check(at(resAnyValue={"fEval:Nullary": {}}))
        assert oneof_branch(field(context, "resAnyValue")) == "ValueDomain"

        error = rejection(at(resAnyFunction={"fEval:Nullary": {}}))
        messages = " ".join(m for _, m in error_sites(error))
        assert refuses_commitment(messages, "Function evaluation"), messages[:300]

    def test_a_function_with_arguments_does_not_collide(self):
        """
        The control that identifies what makes the collision: `Add`'s value is a non-empty object, which
        `Add`'s own instantiation refuses, so only the composition branch can take it.
        """
        context = check(at(resAnyFunction={"Add": {"arg1": 1, "arg2": 2}}))
        assert oneof_branch(field(context, "resAnyFunction")) == "FunctionComposition"

    def test_a_narrow_of_a_non_function_type_still_does_not_collide(self):
        """`Leaf` is not a Function, so the `FunctionComposition` branch cannot take it at all."""
        context = check(at(resAnyValue={"Leaf": {}}))
        assert field(context, "resAnyValue").is_valid


# ==================================================================================================
# 14. The grammar a key-prefix marker would rely on
# ==================================================================================================


class TestTheGrammarAKeyPrefixMarkerWouldRelyOn:
    """
    Two facts about ``:`` in a key position, pinned because a proposed marker --
    ``{"fEval:Nullary": {}}`` / ``fComp:`` / ``fInst:``, in place of a sibling keyword
    (`documentation/TODO_FUNCTION_INTERPRETATION_KEYWORD.md`) -- depends on both.

    They point in opposite directions, which is the interesting part: the qualified-variable form is *not*
    in the way, but a string literal is.
    """

    def test_a_qualified_template_variable_is_not_a_valid_expression_key(self):
        """
        ``Add:T`` is the ``templateContext.substitution`` syntax for disambiguating same-named variables
        of several parameterised parents, and it is compiler-facing besides. It is not a type application,
        so it cannot appear as the key of an expression -- which is why a reserved ``fEval:`` prefix would
        not collide with it.
        """
        error = rejection(at(num={"Add:T": {"arg1": 1, "arg2": 2}}))
        text = " ".join(m for _, m in error_sites(error))
        assert '"Add:T" is not a concept or a template variable' in text, text[:300]

    def test_the_same_holds_for_an_argument_less_function(self):
        error = rejection(at(num={"Nullary:T": {}}))
        text = " ".join(m for _, m in error_sites(error))
        assert '"Nullary:T" is not a concept or a template variable' in text, text[:300]

    def test_a_type_application_may_carry_a_colon_inside_a_string_literal_argument(self):
        """
        The other direction, and the reason a prefix must split on the **first** colon only: a literal
        template argument is arbitrary text. This one contains a colon *and* the marker word.
        """
        context = check(at(tagged={"v": 1}))
        assert field(context, "tagged").is_valid


# ==================================================================================================
# 15. What kind of type the key may name
# ==================================================================================================


TEMPLATED_ADD = function("AddT", {"arg1": ["T"], "arg2": ["T"], "res": "T"})
"""``AddT<T: Numeric>``, so a key can be written *template-dependently* as ``AddT<T>``."""


def with_defaulted_argument(default: object) -> dict:
    """A templated Function whose argument default is ``default``, parsed with ``T`` in scope."""
    return {**TEMPLATED_ADD, **function("Wrapper", {"a": ["T"], "res": "T"}, defaults={"a": default})}


def check_default(default: object) -> ConceptHierarchyContext:
    return check_hierarchy(build_hierarchy({**CONCEPTS, **with_defaulted_argument(default)}))


class TestTheKindOfTypeTheKeyMayName:
    """
    A key can name a type in three ways, and the parser can decide a different amount about each.

    * a **ground** application (`Add<Integer>`) -- everything is decidable;
    * a **template-dependent** application (`AddT<T>`) -- the concept is known to be a Function even though
      its argument is not settled, so the evaluation is parsed as usual;
    * a **bare template variable** (`T`) -- nothing is known. Not which Function this is, not its
      interface, not whether its ``res`` fits the site, and not even whether a Function may stand here at
      all. Exactly one thing is decidable, and it is decided: a Function evaluation's value is an *object*
      of arguments, whatever the Function turns out to be.

    The third case used to crash: `_check_if_subtype(T, Function)` answers **MAYBE**, and
    `SubtypeCheckResult.__bool__` treats only a definite NO as falsy, so control reached
    ``get_template_context("T")`` -- which asserts, because ``T`` names no ValueDomain. It is now recorded
    as *possible* and left to the grounded reparse, where ``T`` is substituted and the key names a real
    type.
    """

    def test_a_template_dependent_key_is_accepted(self):
        assert check_default({"AddT<T>": {"arg1": 1, "arg2": 2}}) is not None

    def test_a_template_dependent_key_is_accepted_with_the_keyword(self):
        assert check_default({"fEval:AddT<T>": {"arg1": 1, "arg2": 2}}) is not None

    def test_a_bare_template_variable_key_is_accepted(self):
        """No concept to look up, so nothing is decided -- but nothing crashes either."""
        assert check_default({"T": {"arg1": 1}}) is not None

    def test_it_is_left_undecided_rather_than_resolved(self):
        """
        Every reading the site still admits is retained. That is what `VerifiedTemplateDependentExpression`
        is for, and what the grounded reparse consumes.
        """
        default = _wrapper_default(check_default({"T": {"arg1": 1}}))
        assert isinstance(default.value, VerifiedTemplateDependentExpression), type(default.value).__name__
        kinds = [type(x) for x in default.value.possible_expressions]
        assert PossibleFunctionEvaluationExpression in kinds, [k.__name__ for k in kinds]

    def test_every_reading_the_value_could_still_have_is_kept(self):
        """
        Three, and exactly three. `Narrow` is among them: ``T`` names no concept, so there is no schema to
        check the value against -- but "unverifiable" is not "impossible", and dropping it would lose a
        reading the grounded reparse may well choose.

        The two that are *absent* are as informative: `Var` cannot apply because the value is an object and
        not a string, and no _ValueDomain_ may register ``"object"`` as a `defaultSerialization` (§8.7).
        """
        default = _wrapper_default(check_default({"T": {"arg1": 1}}))
        kinds = [type(x).__name__ for x in default.value.possible_expressions]
        assert kinds == [
            "PossibleFunctionEvaluationExpression",
            "PossibleNarrowExpression",
            "PossibleInstExpression",
        ], kinds

    def test_the_possible_narrow_carries_the_template_variable_as_its_type(self):
        default = _wrapper_default(check_default({"T": {"arg1": 1}}))
        narrow = next(x for x in default.value.possible_expressions if isinstance(x, PossibleNarrowExpression))
        assert narrow.value_type is not None and narrow.value_type.full_name.endswith("T")

    def test_it_is_template_dependent_and_not_fully_parsed(self):
        """The two properties that make the grounding pass come back to it."""
        default = _wrapper_default(check_default({"T": {"arg1": 1}}))
        assert default.is_value_template_dependent
        assert not default.is_fully_parsed

    def test_an_empty_argument_object_is_still_an_object(self):
        assert check_default({"T": {}}) is not None

    def test_the_keyword_does_not_change_that(self):
        assert check_default({"fEval:T": {"arg1": 1}}) is not None

    def test_a_non_object_value_is_not_accepted_as_an_evaluation(self):
        """
        The one check that survives: an evaluation's value is an object of arguments. ``5`` is not, and no
        substitution of ``T`` could make it one, so this is decidable now rather than later.
        """
        with pytest.raises(ConceptHierarchyError):
            check_default({"T": 5})

    def test_a_non_object_value_says_so(self):
        try:
            check_default({"T": 5})
        except ConceptHierarchyError as error:
            assert "expected a JSON object" in " ".join(m for _, m in error_sites(error))
        else:
            pytest.fail("expected a rejection")


def _wrapper_default(context: ConceptHierarchyContext) -> Expression:
    return function_default_expressions(context, "Wrapper")["a"]


# ==================================================================================================
# 16. An interpretation written on a template-variable key survives substitution
# ==================================================================================================


def caller_with_default(default: object, argument_type: str = "Function") -> dict:
    """
    ``Caller<F: Function>``, whose ``a`` argument defaults to ``default``.

    ``F`` is a template variable constrained to `Function`, so ``{"F": {...}}`` is a key naming a Function
    that is not known until the application is ground.
    """
    return {
        "Caller": {
            "directParents": ["FunctionReturning"],
            "data": {
                "templateContext": {
                    "order": ["F"],
                    "F": "Function",
                    "substitution": {"FunctionReturning:T": "Integer"},
                },
                "interface": {
                    "a": [argument_type],
                    "res": "Integer",
                    "_defaultArgumentValues": {"a": default},
                },
            },
        }
    }


def ground_caller(default: object, argument_type: str = "Function") -> ConceptHierarchyContext:
    """Apply ``Caller<Nullary>`` while omitting ``a``, which is what grounds the default."""
    concepts = {**CONCEPTS, **caller_with_default(default, argument_type)}
    return check_hierarchy(build_hierarchy(concepts, instances={"p": {"Caller<Nullary>": {}}}))


class TestAnInterpretationOnATemplateVariableKeySurvivesSubstitution:
    """
    Why a directive must be *allowed* on a bare template-variable key rather than rejected there.

    Grounding substitutes the **type** and nothing else: ``{"F": {}}`` with ``F := Nullary`` becomes
    ``{"Nullary": {}}``, and an argument-less Function object is the one shape whose three readings are
    all still open. If the directive could not be written on the template-variable key, there would be
    nowhere to write it -- the author never sees the substituted form.

    And the directive is exactly what does *not* depend on which Function arrives: "this is the Function
    value, not a call" is a statement about the object, decided where the object is written.
    """

    def test_the_key_type_is_substituted_on_grounding(self):
        """``{"F": {}}`` becomes an evaluation of `Nullary`, whose `Integer` result fits an `Integer`."""
        context = ground_caller({"F": {}}, argument_type="Integer")
        defaults = context.model.instances["p"].value.value.applied_defaults
        assert type(defaults["a"].value) is FunctionEvaluation, type(defaults["a"].value).__name__

    def test_without_a_directive_the_grounded_key_is_read_as_an_evaluation(self):
        """At a `Function`-typed argument that is wrong: `Nullary` returns an `Integer`, not a Function."""
        with pytest.raises(ConceptHierarchyError) as excinfo:
            ground_caller({"F": {}})
        assert "is not a subtype of Function" in str(excinfo.value)

    def test_a_directive_on_the_template_variable_key_decides_the_grounded_reading(self):
        """The same default, plus the directive: it grounds to the Function *value* and type-checks."""
        context = ground_caller({"fInst:F": {}})
        defaults = context.model.instances["p"].value.value.applied_defaults
        assert type(defaults["a"].value) is NarrowExpression, type(defaults["a"].value).__name__

    def test_the_directive_is_what_makes_the_difference(self):
        """One value, one substitution, opposite readings -- the directive is the only difference."""
        with pytest.raises(ConceptHierarchyError):
            ground_caller({"F": {}})
        assert ground_caller({"fInst:F": {}}) is not None


# ==================================================================================================
# 17. The markers, and the collision they were introduced to resolve
# ==================================================================================================


class TestTheMarkersResolveTheNarrowVersusCompositionCollision:
    """
    ``{"Nullary": {}}`` at a ``FunctionCompositionRes<T>`` whose ``T`` admits a Function type matches both
    branches, and nothing that says only "not an evaluation" can separate them. Each marker names one
    reading, so each selects a different branch of the very same ``oneOf``.
    """

    def test_the_composition_marker_takes_the_composition_branch(self):
        context = check(at(resAnyFunction={"fComp:Nullary": {}}))
        assert oneof_branch(field(context, "resAnyFunction")) == "FunctionComposition"

    def test_the_instantiation_marker_takes_the_t_branch(self):
        """The reading `isFunctionEvaluation` could never select: the Function *value*, via a `Narrow`."""
        context = check(at(resAnyFunction={"fInst:Nullary": {}}))
        assert oneof_branch(field(context, "resAnyFunction")) == "Function"

    def test_the_two_markers_disagree_on_the_same_json(self):
        """One value, one site, two markers, two branches -- which is the whole point."""
        composition = check(at(resAnyFunction={"fComp:Nullary": {}}))
        instantiation = check(at(resAnyFunction={"fInst:Nullary": {}}))
        assert oneof_branch(field(composition, "resAnyFunction")) != oneof_branch(
            field(instantiation, "resAnyFunction")
        )

    def test_the_evaluation_marker_takes_the_t_branch_where_the_result_fits(self):
        context = check(at(resAnyValue={"fEval:Nullary": {}}))
        assert oneof_branch(field(context, "resAnyValue")) == "ValueDomain"

    def test_the_evaluation_marker_is_refused_where_the_result_does_not_fit(self):
        """`Nullary` returns an `Integer`, which is not a `Function` -- so there is no leaf to take it."""
        error = rejection(at(resAnyFunction={"fEval:Nullary": {}}))
        assert refuses_commitment(" ".join(m for _, m in error_sites(error)), "Function evaluation")

    def test_an_unmarked_key_still_collides(self):
        """The markers add a way to say which; they do not change what an unmarked key means."""
        error = rejection(at(resAnyFunction={"Nullary": {}}))
        assert "matches 2 schemas" in " ".join(m for _, m in error_sites(error))


class TestAMarkerAtAnOrdinarySite:
    def test_the_instantiation_marker_narrows_an_argument_less_function(self):
        """What ``isFunctionEvaluation: false`` says at a `Function` site, said positively."""
        context = check(at(fn={"fInst:Nullary": {}}))
        assert type(field(context, "fn").value) is NarrowExpression

    def test_the_composition_marker_reaches_an_enclosing_composition_branch(self):
        """``Dog.f2``'s case: `Add` is not a subtype of `Boxy`, so only `Inst` can accept this."""
        context = check(at(boxy={"fComp:Add": {"arg1": 1, "arg2": 2}}))
        assert type(field(context, "boxy").value) is InstExpression

    def test_the_evaluation_marker_is_the_default_and_changes_nothing(self):
        context = check(at(num={"fEval:Add": {"arg1": 1, "arg2": 2}}))
        assert type(field(context, "num").value) is FunctionEvaluation

    def test_the_instantiation_marker_does_not_evaluate(self):
        """At an `Integer` site the Function *value* is not an `Integer`, so this must be refused."""
        error = rejection(at(num={"fInst:Nullary": {}}))
        assert error_sites(error)


class TestAMarkerOnlyQualifiesAFunction:
    def test_a_marker_on_a_non_function_type_is_rejected(self):
        error = rejection(at(num={"fEval:Leaf": {}}))
        assert "only qualifies a key that names a Function" in " ".join(m for _, m in error_sites(error))

    def test_the_message_names_the_marker_that_was_written(self):
        error = rejection(at(num={"fInst:Leaf": {}}))
        assert '"fInst:" marker' in " ".join(m for _, m in error_sites(error))

    def test_a_marker_on_a_key_that_names_nothing_is_rejected(self):
        error = rejection(at(num={"fComp:NoSuchThing": {}}))
        assert error_sites(error)

    def test_an_unknown_prefix_is_not_a_marker_and_names_nothing(self):
        """``fOther:`` is not in the vocabulary, so the whole string is the key -- and it is not a type."""
        error = rejection(at(num={"fOther:Add": {"arg1": 1, "arg2": 2}}))
        assert "is not a concept or a template variable" in " ".join(m for _, m in error_sites(error))


class TestEveryCommitmentMustBeHonoredNotMerelyMatched:
    """
        `TestTheCommitmentMustBeHonoredNotMerelyMatched` pins the check for an *evaluation*; a marker can
        commit a value to either of the other two readings, and each is dropped just as silently.

        `Anything` is the site that makes all three visible. A boolean instantiation schema -- what a
        ValueDomain that declares no ``instantiation`` gets -- resolves no property names and reaches no leaf,
        so it matches the raw JSON without reading it as anything at all. A structural schema cannot stand in
        here: the marker is part of the *content key*, so ``{"fInst:Add": ...}`` no longer has the property
        such a schema requires and is rejected on property names long before honoring could be at issue.

    `NOT_AN_EVALUATION` is the deliberate exception. It says only "not an evaluation", which both
        remaining readings satisfy, so there is no single kind to insist on and nothing to check.
    """

    def test_an_evaluation_commitment_is_not_honored_by_a_boolean_schema(self):
        error = rejection(at(anything={"fEval:Nullary": {}}))
        assert_reported(error, message="committed to being a Function evaluation", location='"anything"')

    def test_an_instantiation_commitment_is_not_honored_by_a_boolean_schema(self):
        """No `Narrow` is built, so the Function *value* the marker asked for is nowhere in the parse."""
        error = rejection(at(anything={"fInst:Nullary": {}}))
        assert_reported(error, message="committed to being a Function instantiation", location='"anything"')

    def test_a_composition_commitment_is_not_honored_by_a_boolean_schema(self):
        error = rejection(at(anything={"fComp:Nullary": {}}))
        assert_reported(error, message="committed to being a FunctionComposition", location='"anything"')

    def test_the_refusal_says_the_schema_accepted_without_reading(self):
        """The distinguishing half of the message: the schema *matched*, and that is the complaint."""
        error = rejection(at(anything={"fInst:Add": {"arg1": 1, "arg2": 2}}))
        assert_reported(error, message="without reading it as one", location='"anything"')

    def test_the_keyword_is_checked_the_same_way_as_a_marker(self):
        """``true`` reaches this site's boolean schema too, and is dropped by it just as quietly."""
        error = rejection(at(anything={"fEval:Nullary": {}}))
        assert_reported(error, message="committed to being a Function evaluation", location='"anything"')

    def test_ruling_an_evaluation_out_commits_to_no_single_reading_and_is_accepted(self):
        """
        The control for the exception. `OpaqueComposition` is a `FunctionComposition`, so the site's own
        type rules the evaluation out -- and that is all it does: it names no single reading, so there is
        nothing for the consumption check to insist on, and the accept-everything schema may take the
        value unread. Every marked value at such a site is refused by the three tests above.
        """
        context = check(at(opaqueComp={"Nullary": {}}))
        assert field(context, "opaqueComp").is_valid

    def test_a_marker_at_that_same_site_is_still_insisted_on(self):
        """The contrast that shows the acceptance above is about `NOT_AN_EVALUATION`, not about the site."""
        error = rejection(at(opaqueComp={"fInst:Nullary": {}}))
        assert_reported(error, message="without reading it as one", location='"opaqueComp"')

    def test_an_uncommitted_value_is_accepted(self):
        """The control for the check itself: an accept-everything schema does accept everything."""
        context = check(at(anything={"whatever": 1}))
        assert field(context, "anything").is_valid

    def test_the_check_runs_at_depth(self):
        """Nothing about it is particular to a top-level site."""
        error = rejection(at(nested={"Holder": {"anything": {"fInst:Nullary": {}}}}))
        assert_reported(error, message="committed to being a Function instantiation", location='"anything"')

    def test_a_composition_at_a_composition_site_is_honored_by_the_match_itself(self):
        """
        The one reading a leaf is not needed for: at a `FunctionComposition` site the composition *is* the
        `Inst` being built, not a node inside it, so there is nothing deeper to look for.
        """
        context = check(at(resAnyFunction={"fComp:Nullary": {}}))
        assert oneof_branch(field(context, "resAnyFunction")) == "FunctionComposition"


class TestTheTypeDrivenDefaultIsNotTheCompositionMarker:
    """
    ``NOT_AN_EVALUATION`` is produced by a site whose type is a `FunctionComposition` with nothing
    written on the key. It looks like it could simply be `COMPOSITION` -- the composition *is* the default
    reading there -- and it cannot be. These are the measurements that say why, each of which flips if the
    default is changed to `COMPOSITION`.

    The distinction is that `NOT_AN_EVALUATION` constrains what the value is **not**, while a marker constrains
    what it **is**. "Not an evaluation" leaves three readings open, not two: a `Narrow`, an `Inst` that
    reads the object as a composition, and an `Inst` that reads it as ordinary data. `COMPOSITION` admits
    only the second. So the honest reading of the member is *no evaluation*, and it is strictly weaker
    than `COMPOSITION` and `INSTANTIATION` together.
    """

    def test_a_narrow_is_still_admissible_at_a_composition_site(self):
        """
        `COMPOSITION` forbids the `Narrow` reading outright (``narrow_is_admissible``), but narrowing to
        `FunctionComposition` at a `FunctionComposition` site is ordinary and legal. This is what breaks
        first, and most widely, if the default is strengthened.
        """
        context = check(at(comp={"FunctionComposition": {"Add": {"arg1": 1, "arg2": 2}}}))
        assert type(field(context, "comp").value) is NarrowExpression

    def test_an_unmarked_key_at_the_ambiguous_site_stays_ambiguous(self):
        """
        The site the markers exist for. Were the default `COMPOSITION`, this would quietly resolve to the
        `FunctionComposition` branch -- silence would start *meaning* "composition", `fComp:` would become
        redundant here, and the collision the markers were introduced to expose would stop being reported.
        """
        error = rejection(at(resAnyFunction=dict(TestNarrowVersusCompositionIsStillAmbiguous.ARGUMENTLESS)))
        assert "matches 2 schemas" in " ".join(m for _, m in error_sites(error))

    def test_ruling_the_evaluation_out_admits_a_reading_no_marker_admits(self):
        """
        `Structural`'s schema requires the key spelled exactly ``"Add"``, and reads the object as plain
        data -- neither a composition nor a Function value. Both markers refuse it, and not for want of a
        reading: a marker becomes part of the key, so the property the schema requires is no longer there.
        A reading that no marker can name is why `NOT_AN_EVALUATION` is not `COMPOSITION_OR_INSTANTIATION`,
        and the explicit `Narrow` is how it is written instead (`[CH].md` 10.2).
        """
        payload = {"Add": {"arg1": 1, "arg2": 2}}
        assert field(check(at(structural={"Structural": payload})), "structural").is_valid
        for marker in ("fComp", "fInst"):
            error = rejection(at(structural={f"{marker}:Add": payload["Add"]}))
            assert error_sites(error), marker

    def test_the_same_value_written_as_an_explicit_narrow_needs_no_directive(self):
        """
        And the reason the keyword is nonetheless replaceable: naming the site's own type puts the object
        in the schema's payload, where no classification happens and the key keeps its spelling.
        """
        context = check(at(structural={"Structural": {"Add": {"arg1": 1, "arg2": 2}}}))
        assert type(field(context, "structural").value) is NarrowExpression


class TestTheKeywordIsNoLongerADirective:
    """
    The removal itself. Every placement below **was** a classifier directive and is now ordinary data,
    which at these sites means an unexpected property rather than a silent change of reading.

    The distinction worth keeping in view: `TestTheKeywordIsOrdinaryData` covers placements that were
    *always* data and had to survive the removal untouched; this class covers the ones that changed. If
    the recognition ever comes back, these fail and those do not.
    """

    def _unexpected_property(self, instances: dict) -> str:
        error = rejection(instances)
        return " ".join(message for _, message in error_sites(error))

    def test_beside_a_function_key_it_is_an_unexpected_property(self):
        """``false`` at a `Function` site used to select the `Narrow`; the key is now simply not allowed."""
        messages = self._unexpected_property(at(fn={"Nullary": {}, "isFunctionEvaluation": False}))
        assert "Additional property is not allowed" in messages, messages[:300]

    def test_the_instantiation_marker_is_how_that_is_written_now(self):
        """The replacement for the row above, so the two are read together."""
        assert type(field(check(at(fn={"fInst:Nullary": {}})), "fn").value) is NarrowExpression

    def test_true_no_longer_commits_the_value_to_an_evaluation(self):
        messages = self._unexpected_property(
            at(permissive={"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": True})
        )
        assert "Additional property is not allowed" in messages, messages[:300]

    def test_the_evaluation_marker_is_how_that_is_written_now(self):
        assert (
            oneof_branch(field(check(at(permissive={"fEval:Add": {"arg1": 1, "arg2": 2}})), "permissive"))
            == "ValueDomain"
        )

    def test_a_former_misuse_is_not_diagnosed_as_a_misuse(self):
        """
        ``{"Leaf": {}, "isFunctionEvaluation": true}`` was "the keyword only qualifies a Function". There
        is no such rule left to break: the object is two keys, so no `FEval` and no `Narrow` is attempted
        at all, and `Swallow`'s schema rejects it for the key it does not declare.
        """
        messages = self._unexpected_property(at(swallow={"Leaf": {}, "isFunctionEvaluation": True}))
        assert "only qualifies" not in messages, messages[:300]
        assert "Additional property is not allowed" in messages, messages[:300]

    def test_the_marker_carries_that_diagnosis_instead(self):
        error = rejection(at(swallow={"fEval:Leaf": {}}))
        assert_reported(error, message="only qualifies a key that names a Function", location='"swallow"')

    def test_no_value_of_the_keyword_reaches_the_classifier(self):
        """
        Neither boolean is treated differently from a string, which is the whole of the removal: the key
        is not read, so its value cannot matter.
        """
        outcomes = set()
        for written in (True, False, "yes"):
            outcomes.add(
                "Additional property is not allowed"
                in self._unexpected_property(at(fn={"Nullary": {}, "isFunctionEvaluation": written}))
            )
        assert outcomes == {True}, outcomes
