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
The specification of ``"isFunctionEvaluation"``: every placement, at every kind of site.

This is a *specification* suite, written before the implementation it describes
(``documentation/TODO_FUNCTION_EVALUATION_VS_COMPOSITION.md``). Tests that fail today name the cells of
the classification table the parser does not implement yet; nothing here is aspirational beyond that
document.

Three things are asserted for every case, because any one of them alone can pass for the wrong reason:

* **the `ExpressionValue` subclass** -- `FunctionEvaluation` versus `InstExpression`/`NarrowExpression` is
  the *only* record of which reading was taken. Nothing is rewritten at parse time (a ``true`` at a
  `FunctionComposition` site is not turned into a ``Return`` composition here; that lowering happens
  elsewhere), so the class is the answer;
* **the message**, so that a rejection rejects for the stated reason and not by accident;
* **the location**, so that the error is attached to the JSON that is actually wrong -- the
  ``isFunctionEvaluation`` key itself for a misuse, the argument object for a stray key inside one.

The table being pinned (``[CH].md`` 10.2), where K is the single content key's type application:

| tau                            | absent | true  | false                        |
|--------------------------------|--------|-------|------------------------------|
| tau not <= FunctionComposition | FEval  | FEval | not FEval; Narrow, then Inst |
| tau <= FunctionComposition     | Inst   | FEval | Inst                         |

and the recognition rule: the key is a directive **only** on an object of exactly two keys whose other key
names a type application; that type must be a Function, or the use is an error. Anywhere else it is
ordinary data -- an argument name, or a key of some ValueDomain's instantiation.
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
)
from concept_hierarchy.errors import ConceptHierarchyError
from tests.integration.test_expression_parsing import build_hierarchy, check_hierarchy
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

The trap for a misused keyword: ``{"Leaf": {}, "isFunctionEvaluation": true}`` is a misuse -- `Leaf` names
a type that is not a Function -- but set the misuse aside and `Inst` matches this schema and accepts the
value, keyword and all. Every other site in this module rejects that object for an unrelated reason, so
this is the only fixture that can tell a real check from an accident.
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
            "cf": {"type": "CustomFunction"},
            "swallow": {"type": "Swallow"},
            "permissive": {"type": "Permissive"},
            "structural": {"type": "Structural"},
            "nested": {"type": "Holder"},
        },
    },
)
"""
One site of each kind, so a test picks a site by naming a key. ``nested`` is what makes depth cheap.

``additionalProperties: false`` is load-bearing *for the tests themselves*: without it, ``{"Holder": ...}``
also matches `Holder`'s own instantiation as an `Inst` -- one unrecognised key, every property optional --
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
    **STRUCTURAL,
    **SWALLOW,
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
    point is that a composition is recognised by its schema -- so the class no longer separates them. Which
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


class TestRecognition:
    """
    The key is a directive **only** on a two-key object whose other key names a type application.

    Everywhere else it is data. This is not a nicety: an argument may be named ``isFunctionEvaluation``,
    and so may a property in a ValueDomain's instantiation, and stripping it there would silently delete a
    value. ``examples/animal_kingdom.json`` contains exactly such a placement, one level too deep inside
    ``FunctionSequence``'s argument object.
    """

    def test_two_keys_with_a_function_key_is_a_directive(self):
        context = check(at(fn={"Nullary": {}, "isFunctionEvaluation": False}))
        assert field_kind(context, "fn") is NarrowExpression

    def test_two_keys_with_a_non_function_type_key_is_an_error(self):
        error = rejection(at(num={"Leaf": {}, "isFunctionEvaluation": True}))
        assert_reported(error, message="isFunctionEvaluation", location='"isFunctionEvaluation" (key)')

    def test_the_error_names_the_offending_type(self):
        error = rejection(at(num={"Leaf": {}, "isFunctionEvaluation": True}))
        assert_reported(error, message="Leaf", location='"isFunctionEvaluation" (key)')

    def test_two_keys_whose_other_key_is_not_a_type_is_ordinary_data(self):
        """``{"arg1": 1, "isFunctionEvaluation": true}`` is an argument list with a bad key, not a directive."""
        error = rejection(at(num={"Add": {"arg1": 1, "isFunctionEvaluation": True}}))
        assert_reported(error, message='does not have the argument "isFunctionEvaluation"', location='"num"')

    def test_the_sole_key_is_ordinary_data(self):
        """A ValueDomain may declare a property with that name; one key is never a directive."""
        context = check(at(flagged={"isFunctionEvaluation": 3}))
        assert field_kind(context, "flagged") is InstExpression

    def test_three_keys_are_ordinary_data(self):
        context = check(at(num={"Weird": {"isFunctionEvaluation": 3}}))
        assert field_kind(context, "num") is FunctionEvaluation

    def test_an_argument_may_be_named_is_function_evaluation(self):
        context = check(at(num={"Weird": {"isFunctionEvaluation": 3}}))
        evaluation = field(context, "num").value
        assert isinstance(evaluation, FunctionEvaluation)
        assert sorted(evaluation.arguments) == ["isFunctionEvaluation"]

    def test_a_directive_and_an_argument_of_that_name_coexist(self):
        """
        The outer key is the directive, the inner key is the argument. Both readings are correct at once,
        and neither may consume the other's key.
        """
        context = check(at(num={"Weird": {"isFunctionEvaluation": 3}, "isFunctionEvaluation": True}))
        evaluation = field(context, "num").value
        assert isinstance(evaluation, FunctionEvaluation)
        assert sorted(evaluation.arguments) == ["isFunctionEvaluation"]

    def test_a_directive_is_stripped_before_the_instantiation_schema_sees_it(self):
        """
        No ValueDomain declares the key, so an `Inst` that still carried it would be rejected by
        `Boxy`'s ``additionalProperties``. Reaching `Boxy` at all is the ``false`` cell's whole job.
        """
        context = check(at(boxy={"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": False}))
        assert field_kind(context, "boxy") is InstExpression

    def test_the_input_json_is_not_modified(self):
        """The key is popped while the value is classified; it must be put back."""
        value = {"Nullary": {}, "isFunctionEvaluation": False}
        check(at(fn=value))
        assert value == {"Nullary": {}, "isFunctionEvaluation": False}


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


class TestOrdinarySiteWithTrue:
    """``true`` restates the default here. It is accepted, and it changes nothing."""

    def test_true_is_accepted_and_redundant(self):
        context = check(at(num={"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": True}))
        assert field_kind(context, "num") is FunctionEvaluation

    def test_true_produces_the_same_expression_as_omitting_it(self):
        with_flag = field(check(at(num={"Nullary": {}, "isFunctionEvaluation": True})), "num")
        without = field(check(at(num={"Nullary": {}})), "num")
        assert type(with_flag.value) is type(without.value)
        assert with_flag.value.value_type.full_name == without.value.value_type.full_name

    def test_true_does_not_rescue_a_result_that_is_not_a_subtype(self):
        error = rejection(at(num={"Stringy": {}, "isFunctionEvaluation": True}))
        assert_reported(error, message="not a subtype", location='"num"')


class TestOrdinarySiteWithFalse:
    """
    The first load-bearing cell. ``false`` takes the evaluation reading off the table -- and the cascade
    then continues to `Narrow` **and then `Inst`**, which is what makes ``Dog.f2`` legal.
    """

    def test_false_narrows_an_argument_less_function(self):
        context = check(at(fn={"Nullary": {}, "isFunctionEvaluation": False}))
        expression = field(context, "fn")
        assert isinstance(expression.value, NarrowExpression)
        assert expression.value.value_type.full_name == "Nullary"

    def test_false_reaches_the_enclosing_instantiation_schema(self):
        """`Add` is not a subtype of `Boxy`, so `Narrow` fails and only `Inst` can accept this."""
        context = check(at(boxy={"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": False}))
        assert field_kind(context, "boxy") is InstExpression

    def test_the_same_value_without_false_is_rejected_at_that_site(self):
        """The contrast that shows the cell is load-bearing rather than decorative."""
        error = rejection(at(boxy={"Add": {"arg1": 1, "arg2": 2}}))
        assert_reported(error, message="not a subtype", location='"boxy"')

    def test_false_with_arguments_at_a_function_site_is_rejected(self):
        """A Function's instantiation is the empty object; arguments have nowhere to go."""
        error = rejection(at(fn={"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": False}))
        assert_reported(error, message="Add", location='"fn"')

    def test_false_reports_the_abandoned_alternatives_when_nothing_matches(self):
        error = rejection(at(num={"Nullary": {}, "isFunctionEvaluation": False}))
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
        context = check(at(comp={"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": False}))
        assert field_kind(context, "comp") is InstExpression

    def test_absent_and_false_agree(self):
        absent = field(check(at(comp={"Add": {"arg1": 1, "arg2": 2}})), "comp")
        false = field(check(at(comp={"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": False})), "comp")
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
        error = rejection(at(comp={"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": True}))
        assert_reported(error, message="without reading it as one", location='"comp"')

    def test_true_on_a_function_that_returns_nothing_is_rejected_here_too(self):
        """The reason is the absent branch, not the absent result type."""
        error = rejection(at(comp={"Void": {"arg1": 1}, "isFunctionEvaluation": True}))
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
        context = check(at(res={"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": False}))
        assert oneof_branch(field(context, "res")) == "FunctionComposition"

    def test_true_takes_the_t_branch(self):
        """The whole design in one assertion: the same object, the other branch, because of the keyword."""
        context = check(at(res={"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": True}))
        assert oneof_branch(field(context, "res")) == "Integer"

    def test_true_checks_the_result_against_the_template_argument(self):
        """``res(Stringy) = String`` is not the `Integer` that ``FunctionCompositionRes<Integer>`` promises."""
        error = rejection(at(res={"Stringy": {}, "isFunctionEvaluation": True}))
        assert_reported(error, message="not a subtype", location='"res"')

    def test_true_accepts_a_result_of_the_promised_type(self):
        context = check(at(resStr={"Stringy": {}, "isFunctionEvaluation": True}))
        assert oneof_branch(field(context, "resStr")) == "String"

    def test_true_on_a_function_that_returns_nothing_is_rejected(self):
        """Unlike a plain `FunctionComposition` site, this one promises a value."""
        error = rejection(at(res={"Void": {"arg1": 1}, "isFunctionEvaluation": True}))
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
        for value in ({"Add": {"arg1": 1, "arg2": 2}}, {"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": True}):
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


class TestTheKeywordAtDepth:
    """The keyword works the same however deep the site is; several tests pin the location of failures."""

    def test_false_at_depth_two(self):
        context = check(at(nested={"Holder": {"fn": {"Nullary": {}, "isFunctionEvaluation": False}}}))
        assert field(context, "nested").is_valid

    def test_true_at_depth_two(self):
        """`Permissive` is the site that can take it; the point here is that depth changes nothing."""
        context = check(
            at(nested={"Holder": {"permissive": {"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": True}}})
        )
        assert field(context, "nested").is_valid

    def test_the_keyword_inside_a_composition_argument(self):
        context = check(at(comp={"Add": {"arg1": {"Nullary": {}, "isFunctionEvaluation": True}, "arg2": 2}}))
        assert field(context, "comp").is_valid

    def test_a_misused_keyword_at_depth_is_located_at_the_key(self):
        error = rejection(at(nested={"Holder": {"num": {"Leaf": {}, "isFunctionEvaluation": True}}}))
        assert_reported(error, message="isFunctionEvaluation", location='"isFunctionEvaluation" (key)')

    def test_a_stray_keyword_inside_an_argument_object_at_depth(self):
        """``animal_kingdom.json``'s placement, exactly: one level too deep, inside the argument list."""
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
        a composition is recognised by its instantiation schema and by nothing else -- the top-level
        `FEval` alternative is never even tried there, which is what stopped the parser asking whether
        `Boolean` is a subtype of ``FunctionCompositionRes<Boolean>``.
        """
        for value in (
            {"Add": {"arg1": 1, "arg2": 2}},
            {"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": False},
        ):
            context = check(at(comp=value))
            assert isinstance(field(context, "comp").value, InstExpression), value

    def test_the_evaluation_selected_by_true_is_reachable_inside(self):
        """It is one level down, at the ``"T"`` leaf the keyword selected."""
        context = check(at(res={"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": True}))
        expression = field(context, "res")
        assert oneof_branch(expression) == "Integer"
        evaluations = [k for k in expression_kinds(expression) if k is FunctionEvaluation]
        assert evaluations, f"no evaluation under the T branch; got {expression_kinds(expression)}"

    def test_the_expression_type_is_the_site_type_not_the_result_type(self):
        """Even for the ``true`` cell: the expression sits at the site, whatever the evaluation returns."""
        context = check(at(res={"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": True}))
        assert field(context, "res").required_expression_type.full_name == "FunctionCompositionRes<Integer>"

    def test_the_two_readings_of_one_shape_differ_only_by_the_keyword(self):
        composition = field(check(at(res={"Add": {"arg1": 1, "arg2": 2}})), "res")
        evaluation = field(check(at(res={"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": True})), "res")
        assert type(composition.value) is type(evaluation.value) is InstExpression, "both are instantiations"
        assert oneof_branch(composition) == "FunctionComposition"
        assert oneof_branch(evaluation) == "Integer"


# ==================================================================================================
# 7. The schemas stay clean
# ==================================================================================================


class TestNoSchemaMentionsTheKeyword:
    """
    Point six of the design: the keyword is a directive to the classifier and never part of a value, so no
    ValueDomain declares it and no schema branches on it. These tests fail the moment somebody reaches for
    a ``"const": false`` to break the ``oneOf`` tie.
    """

    def test_function_composition_declares_no_such_property(self):
        context = check(at(comp={"Add": {"arg1": 1, "arg2": 2}}))
        for _constraint, schema in context.model.value_domains["FunctionComposition"].instantiation:
            for node in schema.walk():
                assert "isFunctionEvaluation" not in (node.properties or {})

    def test_function_composition_res_declares_no_such_property(self):
        context = check(at(res=3))
        for _constraint, schema in context.model.value_domains["FunctionCompositionRes"].instantiation:
            for node in schema.walk():
                assert "isFunctionEvaluation" not in (node.properties or {})

    def test_a_value_domain_may_still_declare_a_property_of_that_name(self):
        """The ban is on the *prelude* needing it, not on the identifier."""
        context = check(at(flagged={"isFunctionEvaluation": 3}))
        assert field(context, "flagged").is_valid


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
      present when it never was, it **adds** ``"isFunctionEvaluation": false`` to the caller's JSON;
    * ``_parse_expression_of_json_object`` raises ``Invalid use of the "isFunctionEvaluation" keyword``
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
        instances = at(comp={"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": True})
        before = copy.deepcopy(instances)
        assert self.unchanged(instances) == before

    def test_a_written_keyword_is_restored_when_the_value_is_rejected(self):
        instances = at(num={"Leaf": {}, "isFunctionEvaluation": True})
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


class TestTheKeywordIsGivenBackWhenItWasNeverADirective:
    """
    The keyword is popped *before* the key beside it can be resolved -- content keys have to be counted to
    know there is a single one at all -- so the pop is a guess, and it is wrong whenever the key turns out
    to name no type.

    Nothing about the object's shape distinguishes the two cases in advance:
    ``{"Add": {...}, "isFunctionEvaluation": true}`` is a directive on a Function evaluation, and
    ``{"lhs": 1, "isFunctionEvaluation": true}`` is a two-property value of some ValueDomain. Both are
    "one content key plus the keyword". Only resolving the key tells them apart, and by then the pop has
    happened -- so it has to be undone.

    Getting this wrong is silent: the value reaches `Inst` one key short, and the schema rejects it for a
    missing property the author did write.
    """

    def test_a_boolean_keyword_beside_a_non_type_key_reaches_the_schema(self):
        context = check(at(paired={"lhs": 1, "isFunctionEvaluation": True}))
        assert field(context, "paired").is_valid

    def test_both_boolean_values_reach_the_schema(self):
        """``false`` is the value that would otherwise look most like a directive."""
        context = check(at(paired={"lhs": 1, "isFunctionEvaluation": False}))
        assert field(context, "paired").is_valid

    def test_the_schema_really_does_require_the_keyword(self):
        """Without this, the test above would pass even if the key were dropped."""
        error = rejection(at(paired={"lhs": 1}))
        assert_reported(error, message="isFunctionEvaluation", location='"paired"')

    def test_a_non_boolean_keyword_is_never_a_directive(self):
        """A string value leaves the object with two content keys, so it is never even a candidate."""
        context = check(at(labelled={"lhs": 1, "isFunctionEvaluation": "a string value"}))
        assert field(context, "labelled").is_valid

    def test_a_non_boolean_keyword_beside_a_function_key_is_data_too(self):
        """
        ``{"Add": {...}, "isFunctionEvaluation": "yes"}`` has two content keys, so no `FEval` and no
        `Narrow` is attempted -- the value goes straight to `Inst`, and at an `Integer` site it fails as
        the object it is, not as a misused keyword.
        """
        error = rejection(at(num={"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": "yes"}))
        messages = " ".join(message for _, message in error_sites(error))
        assert "Invalid use of the" not in messages, f"read as a directive: {messages[:300]}"
        assert "not of type 'integer'" in messages, messages[:300]

    def test_the_input_is_not_modified_when_the_keyword_is_given_back(self):
        instances = at(paired={"lhs": 1, "isFunctionEvaluation": True})
        before = copy.deepcopy(instances)
        check(instances)
        assert instances == before

    def test_the_input_is_not_modified_for_a_non_boolean_keyword(self):
        instances = at(labelled={"lhs": 1, "isFunctionEvaluation": "a string value"})
        before = copy.deepcopy(instances)
        check(instances)
        assert instances == before

    def test_a_directive_beside_a_type_key_is_still_consumed(self):
        """The other side of the same decision: beside a type, the keyword must *not* come back."""
        context = check(at(boxy={"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": False}))
        assert type(field(context, "boxy").value) is InstExpression


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

    VALUE = {"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": True}
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
        assert "committed to being a Function evaluation" in messages, messages[:400]

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
        instances = at(cf={"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": False})
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
        context = check(at(permissive={"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": True}))
        assert oneof_branch(field(context, "permissive")) == "ValueDomain"

    def test_a_result_of_another_type_is_also_set_aside(self):
        context = check(at(permissive={"Stringy": {}, "isFunctionEvaluation": True}))
        assert oneof_branch(field(context, "permissive")) == "ValueDomain"

    # -- never set aside: the evaluation is malformed, so no leaf could take it ---------------------

    def test_an_argument_object_that_is_not_an_object_is_not_set_aside(self):
        error = rejection(at(permissive={"Add": 5, "isFunctionEvaluation": True}))
        assert_reported(error, message="expected a JSON object", location='"permissive"')

    def test_an_unknown_argument_is_not_set_aside(self):
        error = rejection(at(permissive={"Add": {"nope": 1}, "isFunctionEvaluation": True}))
        assert_reported(error, message='does not have the argument "nope"', location='"permissive"')

    def test_a_missing_required_argument_is_not_set_aside(self):
        error = rejection(at(permissive={"Add": {"arg1": 1}, "isFunctionEvaluation": True}))
        assert_reported(error, message="missing from the Function evaluation interface", location='"permissive"')

    def test_an_abstract_key_is_not_set_aside(self):
        """`FunctionReturning` is abstract, so it cannot be evaluated whatever the site would accept."""
        error = rejection(at(permissive={"FunctionReturning<Integer>": {}, "isFunctionEvaluation": True}))
        assert_reported(error, message="abstract type", location='"permissive"')

    def test_a_bad_argument_value_is_not_set_aside(self):
        error = rejection(at(permissive={"Add": {"arg1": "s:x", "arg2": 2}, "isFunctionEvaluation": True}))
        assert_reported(error, message="arg1", location='"permissive"')

    def test_the_arguments_of_a_committed_evaluation_are_ordinary_sites(self):
        """
        The commitment is pinned to the value's own location, so it reaches a custom-type leaf sitting at
        that location and nothing below it. `Add`'s arguments are plain literals here: if the commitment
        leaked into them, only a Function evaluation would be admissible there and ``1`` would be rejected.
        """
        context = check(at(permissive={"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": True}))
        assert oneof_branch(field(context, "permissive")) == "ValueDomain"

    def test_a_nested_evaluation_inside_a_committed_one_is_still_ordinary(self):
        """The same, one level deeper and with an argument that *is* an evaluation."""
        context = check(at(permissive={"Add": {"arg1": {"Nullary": {}}, "arg2": 2}, "isFunctionEvaluation": True}))
        expression = field(context, "permissive")
        assert oneof_branch(expression) == "ValueDomain"
        assert expression_kinds(expression).count(FunctionEvaluation) >= 2, expression_kinds(expression)

    def test_a_property_leaf_is_unaffected_by_the_commitment(self):
        """
        Only a leaf at the value's own location can honour it. `Permissive`'s other branch reaches an
        `Integer` leaf through the ``lhs`` *property*, one location further down, and parses as it always
        would.
        """
        context = check(at(permissive={"lhs": 7}))
        assert field(context, "permissive").is_valid


class TestAMisusedKeywordIsNeverSetAside:
    """
    The regression this class exists for. The misuse is about the *keyword*, not about the site's type, so
    it must survive into the rejection. It used to be set aside with the rest, and `Inst` then accepted
    ``{"Leaf": {}, "isFunctionEvaluation": true}`` at a `Swallow` site -- schema matched, misuse gone.

    `Swallow` is the only site here whose schema accepts that object, which is what makes these the tests
    that would have caught it; everywhere else the value is rejected either way.
    """

    def test_a_misuse_is_rejected_even_where_the_schema_would_accept_the_object(self):
        error = rejection(at(swallow={"Leaf": {}, "isFunctionEvaluation": True}))
        assert_reported(
            error,
            message="only qualifies an object keyed by a Function",
            location='"isFunctionEvaluation" (key)',
        )

    def test_the_same_holds_for_false(self):
        """The misuse is the keyword's presence beside a non-Function key, not the value it carries."""
        error = rejection(at(swallow={"Leaf": {}, "isFunctionEvaluation": False}))
        assert_reported(
            error,
            message="only qualifies an object keyed by a Function",
            location='"isFunctionEvaluation" (key)',
        )

    def test_without_the_keyword_that_very_value_is_accepted(self):
        """
        The control that proves the two above test the keyword and not the fixture: the same object
        without the keyword is a perfectly good `Swallow`.
        """
        context = check(at(swallow={"Leaf": {}}))
        assert field(context, "swallow").is_valid

    def test_a_misuse_at_a_permissive_site_is_rejected_too(self):
        """`Permissive`'s ``"ValueDomain"`` leaf accepts almost any result, but not a misused keyword."""
        error = rejection(at(permissive={"Leaf": {}, "isFunctionEvaluation": True}))
        assert_reported(
            error,
            message="only qualifies an object keyed by a Function",
            location='"isFunctionEvaluation" (key)',
        )


# ==================================================================================================
# 12. A schema may not accept a committed evaluation without reading it as one
# ==================================================================================================


class TestTheCommitmentMustBeHonouredNotMerelyMatched:
    """
    ``"isFunctionEvaluation": true`` is honoured at a **custom-type leaf**, and a schema can match the
    value without ever reaching one -- a structural schema that happens to fit, or a boolean one. Such a
    match would accept the value with the keyword silently ignored.

    Two readings of "was it honoured" are wrong, and both are pinned below:

    * *does a Function evaluation appear anywhere below?* -- a **composition** holds one too
      (``"properties": "args"`` stores it), so this answers yes for the reading the keyword ruled out;
    * *does one appear at any depth?* -- an **argument** of the value may be an evaluation, so this
      answers yes for a value that was not read as one at all.

    What must hold is that the node standing at the value's **own location** is the evaluation. Depth is
    not a factor: ``$ref`` and the composite keywords keep the location, only a property or item descent
    changes it, so the consumer is always exactly there however many schema nodes were crossed.
    """

    def test_a_structural_match_does_not_honour_it(self):
        """`Structural` matches ``{"Add": {...}}`` as an object; no custom-type leaf is reached."""
        error = rejection(at(structural={"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": True}))
        assert_reported(error, message="without reading it as one", location='"structural"')

    def test_the_same_value_is_accepted_when_it_claims_not_to_be_an_evaluation(self):
        """
        The control: nothing about this value is wrong except the keyword nobody honoured. ``false`` is
        what it takes to reach `Structural`'s schema at all -- with the keyword absent the default
        evaluation reading commits and hard-fails on ``res(Add)`` before `Inst` is tried.
        """
        context = check(at(structural={"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": False}))
        assert field(context, "structural").is_valid

    def test_a_leaf_at_the_value_s_own_location_does_honour_it(self):
        """`Permissive`'s root-level ``"ValueDomain"`` branch is such a leaf, and the value is accepted."""
        context = check(at(permissive={"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": True}))
        assert oneof_branch(field(context, "permissive")) == "ValueDomain"

    def test_a_composition_holding_an_evaluation_does_not_count_as_honouring_it(self):
        """
        The first wrong reading. A composition of `Add` contains a `FunctionEvaluation` in
        ``custom_expressions``, so "contains one" would accept a `FunctionComposition` site's ``true`` --
        the reading the keyword explicitly ruled out.
        """
        composition = field(check(at(comp={"Add": {"arg1": 1, "arg2": 2}})), "comp")
        assert FunctionEvaluation in expression_kinds(composition), "the composition does hold one"
        error = rejection(at(comp={"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": True}))
        assert_reported(error, message="without reading it as one", location='"comp"')

    def test_an_evaluation_among_the_arguments_does_not_count_either(self):
        """
        The second wrong reading. ``arg1`` is itself an evaluation, so a depth-unbounded search would find
        one -- but the outer value is not read as an evaluation by `Structural`, and must be rejected.
        """
        error = rejection(at(structural={"Add": {"arg1": {"Nullary": {}}, "arg2": 2}, "isFunctionEvaluation": True}))
        assert_reported(error, message="without reading it as one", location='"structural"')

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
                    instances=at(cf={"Add": {"arg1": 1, "arg2": 2}, "isFunctionEvaluation": True}),
                )
            )
        messages = " ".join(message for _, message in error_sites(excinfo.value))
        assert "committed to being a Function evaluation" in messages, messages[:400]
