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
Integration tests: the ``"properties": "args"`` schema node, and what it delegates to.

`FunctionComposition`'s instantiation schema says a value is a one-key object whose key names a `Function`
and whose value is *that Function's arguments*:

```json
{"type": "object", "minProperties": 1, "maxProperties": 1,
 "propertyNames": {"type": "string", "format": "Type", "constraint": "Function"},
 "additionalProperties": {"type": "object", "properties": "args", "additionalProperties": false}}
```

Only the Function's interface knows which keys of that inner object are legal, so the schema cannot spell
them and `value_instantiation_parser` cannot check them. ``"properties": "args"`` is the placeholder for
that, and `_parse_evaluation_arguments_of_function` is where it is decided -- by handing the whole thing to
`parse_function_evaluation_expression` through the one `ValueInstantiationContext.parse_function_evaluation`
seam, so that the value parser gains a Function evaluation parser without gaining a dependency on one.

Two things about that node are load-bearing and are pinned below:

* **it claims every key**, or the ``additionalProperties: false`` beside it rejects each argument in turn --
  which was `animal_kingdom.json`'s ``factorial`` failing with four "Additional property is not allowed";
* **it records errors, it does not raise them.** The node runs inside ``anyOf``/``oneOf`` trial branches --
  `CustomFunction`'s own ``oneOf`` tries the whole value as a `FunctionComposition` first -- and
  `child_silent` catches no exceptions, so anything raised there would tear down the entire parse rather
  than fail one branch.

See ``documentation/TODO_FUNCTION_COMPOSITION_ARGS_SCHEMA.md``, and
``tests/integration/test_function_evaluation_expression.py`` for the contract of the callee.
"""

from __future__ import annotations

import pytest

from concept_hierarchy.data.contexts.context import ConceptHierarchyContext
from concept_hierarchy.data.expressions.instantiated_value import ParsedCustomValue, ParsedStructural, ParsedValue
from concept_hierarchy.data.expressions.subexpressions import FunctionEvaluation
from concept_hierarchy.errors import ConceptHierarchyError
from tests.integration.test_expression_parsing import build_hierarchy, check_hierarchy
from tests.integration.test_function_default_arguments import GROUND, function
from tests.integration.test_schema_substitution import obj, vd

# --------------------------------------------------------------------------------------------------
# Hierarchy fragments
# --------------------------------------------------------------------------------------------------

ADD = function("Add", {"arg1": ["Integer"], "arg2": ["Integer"], "res": "Integer"}, template=GROUND)
DEFAULTED = function("Defaulted", {"arg1": ["Integer"], "arg2": ["Integer"], "res": "Integer"}, {"arg2": 3}, GROUND)
CYCLIC = function("Cyclic", {"a": ["Integer"], "b": ["Integer"], "res": "Integer"}, {"a": "b", "b": "a"}, GROUND)
LEAF = vd("Leaf", {"type": "object", "additionalProperties": False})
"""A ValueDomain: a key that names a perfectly good type which is not a Function."""

HOLDER = vd("Holder", obj({"proc": {"type": "FunctionComposition"}}))
"""An ordinary ValueDomain with a `FunctionComposition` property, i.e. the node one schema deeper."""

EITHER = vd("Either", {"oneOf": ["FunctionComposition", obj({"lhs": {"type": "Integer"}})]})
"""
`CustomFunction`'s shape, reduced: a ``oneOf`` whose first branch is a `FunctionComposition`.

Any value of the second branch is *tried* as a composition first, so this is the fragment that says whether
the args node can fail a trial branch without taking the parse down with it.
"""

CONCEPTS = {**ADD, **DEFAULTED, **CYCLIC, **LEAF, **HOLDER, **EITHER}


# --------------------------------------------------------------------------------------------------
# Accessors
# --------------------------------------------------------------------------------------------------


def check(instances: dict) -> ConceptHierarchyContext:
    return check_hierarchy(build_hierarchy(CONCEPTS, instances=instances))


def rejection(instances: dict) -> str:
    """
    The hierarchy must be rejected, and the *parse* must have run to completion to report it.

    `pytest.raises` alone would not tell the two apart: an exception escaping a trial branch also arrives
    as a `ConceptHierarchyError`. What distinguishes them is that a recorded error is *reported through the
    expression's explanation trace*, so the headline is the enclosing value's and the cause is nested under
    it -- which is what the callers below assert on.
    """
    with pytest.raises(ConceptHierarchyError) as excinfo:
        check(instances)
    text = str(excinfo.value)
    assert "Invalid expression" in text, f"expected a reported failure, not a raised one; got: {text[:400]}"
    return text


def stored_evaluations(context: ConceptHierarchyContext, global_variable: str) -> dict[str, ParsedValue]:
    """
    Every ``custom_expressions`` entry the ``args`` node left anywhere under a global's parsed value.

    `ParsedValue.walk` stops at a custom-type leaf, because what hangs off it is an `Expression` and not a
    parsed value -- but that expression's *own* value is another parsed tree, and a nested
    `FunctionComposition` lives in exactly that position. So the descent has to cross the boundary.
    """
    found: dict[str, ParsedValue] = {}

    def collect(parsed_value: ParsedValue) -> None:
        for node in parsed_value.walk():
            if isinstance(node, ParsedStructural):
                for key, entries in node.custom_expressions.items():
                    assert len(entries) == 1, f"an args entry must be the only one for {key}"
                    discriminator, parsed = entries[0]
                    assert discriminator == "args"
                    found[key] = parsed
            if isinstance(node, ParsedCustomValue) and node.expression is not None:
                nested = getattr(node.expression.value, "value", None)
                if isinstance(nested, ParsedValue):
                    collect(nested)

    collect(context.model.instances[global_variable].value.value.value)
    return found


def evaluation_of(context: ConceptHierarchyContext, global_variable: str, function_name: str) -> FunctionEvaluation:
    parsed = stored_evaluations(context, global_variable)[function_name]
    assert isinstance(parsed, ParsedCustomValue), f"expected a custom value, got {type(parsed).__name__}"
    assert parsed.expression is not None
    value = parsed.expression.value
    assert isinstance(value, FunctionEvaluation), f"expected a FunctionEvaluation, got {type(value).__name__}"
    return value


# ==================================================================================================
# 1. A FunctionComposition value is parsed as an evaluation
# ==================================================================================================


class TestTheValueIsParsedAsAnEvaluation:
    def test_a_composition_value_checks_and_its_evaluation_is_stored(self):
        context = check({"p": {"FunctionComposition": {"Add": {"arg1": 1, "arg2": 2}}}})
        evaluation = evaluation_of(context, "p", "Add")
        assert sorted(evaluation.arguments) == ["arg1", "arg2"]
        assert all(argument.is_valid for argument in evaluation.arguments.values())

    def test_the_argument_keys_are_claimed_so_additional_properties_does_not_fire(self):
        """
        The regression that was `animal_kingdom.json`'s ``factorial``: ``"properties": "args"`` empties the
        node's ``properties``, so unless the handler claims the keys, the ``additionalProperties: false``
        beside it rejects every argument.
        """
        context = check({"p": {"FunctionComposition": {"Add": {"arg1": 1, "arg2": 2}}}})
        assert evaluation_of(context, "p", "Add").arguments, "the arguments must survive the schema"

    def test_the_node_is_reached_one_schema_deeper(self):
        """A property of type `FunctionComposition`, rather than the whole value being one."""
        context = check({"p": {"Holder": {"proc": {"Add": {"arg1": 1, "arg2": 2}}}}})
        assert sorted(evaluation_of(context, "p", "Add").arguments) == ["arg1", "arg2"]

    def test_the_arguments_are_parsed_not_merely_counted(self):
        """An argument whose value does not fit its declared type is a failure of the whole value."""
        text = rejection({"p": {"FunctionComposition": {"Add": {"arg1": "s:x", "arg2": 2}}}})
        assert "argument arg1" in text

    def test_an_unsupplied_default_is_grounded_from_here_too(self):
        """Grounding runs inside the callee, so this caller gets it by calling it."""
        context = check({"p": {"FunctionComposition": {"Defaulted": {"arg1": 1}}}})
        evaluation = evaluation_of(context, "p", "Defaulted")
        assert sorted(evaluation.arguments) == ["arg1"], "still only what the value wrote"
        assert sorted(evaluation.applied_defaults) == ["arg2"]


# ==================================================================================================
# 2. The key must name a Function
# ==================================================================================================


class TestTheKeyMustNameAFunction:
    """
    The case the expression parser reports nothing for -- there it just means "try `Narrow` instead", and
    `is_function_subtype` is the only signal. Here an evaluation is the only thing allowed, so the args
    node has to notice it for itself.
    """

    def test_a_key_naming_a_type_that_is_not_a_function_is_rejected(self):
        text = rejection({"p": {"FunctionComposition": {"Leaf": {}}}})
        assert "'Leaf' names Leaf, which is not a Function" in text

    def test_a_key_naming_nothing_at_all_is_rejected(self):
        text = rejection({"p": {"FunctionComposition": {"NoSuchThing": {}}}})
        assert "'NoSuchThing' is not a valid Function name!" in text


# ==================================================================================================
# 3. Invalid evaluations are recorded, not raised
# ==================================================================================================


class TestInvalidEvaluationsAreRecordedNotRaised:
    """
    Each of these was a ``raise`` inside `parse_function_evaluation_expression` until they became
    `IllFormedExpression`\\ s. From this caller a raise is not merely untidy: the node runs inside trial
    branches, and `child_silent` catches nothing, so a raised error aborts the whole parse instead of
    failing the branch that provoked it.

    `rejection` is what keeps that honest -- it requires the failure to arrive through the expression's
    explanation trace rather than as a bare exception.
    """

    def test_an_argument_the_function_does_not_have(self):
        text = rejection({"p": {"FunctionComposition": {"Add": {"arg1": 1, "arg2": 2, "zzz": 3}}}})
        assert 'does not have the argument "zzz"' in text
        assert "Additional property is not allowed" not in text, (
            "the evaluation parser names the Function and the argument; the schema keyword names neither"
        )

    def test_a_missing_required_argument(self):
        text = rejection({"p": {"FunctionComposition": {"Add": {"arg1": 1}}}})
        assert "are missing from the Function evaluation interface" in text
        assert "arg2" in text

    def test_cyclic_dependencies_between_the_remaining_defaults(self):
        text = rejection({"p": {"FunctionComposition": {"Cyclic": {}}}})
        assert "is not acyclic" in text

    def test_an_abstract_key(self):
        text = rejection({"p": {"FunctionComposition": {"ValueDomain": {}}}})
        assert "is an abstract type" in text


# ==================================================================================================
# 4. Trial branches
# ==================================================================================================


class TestTrialBranchesStaySilent:
    """
    `Either`'s ``oneOf`` tries every value as a `FunctionComposition` first. That branch fails for anything
    that is not one -- and its failure must stay inside the branch.
    """

    def test_a_value_matching_the_other_branch_still_checks(self):
        """
        ``{"lhs": 1}`` is not a composition: the args node sees ``lhs`` as the "Function" key and fails.
        The value is nonetheless valid, because the second branch matches.
        """
        context = check({"p": {"Either": {"lhs": 1}}})
        assert context.model.instances["p"].value.is_valid
        assert stored_evaluations(context, "p") == {}, "the failed trial branch leaves nothing behind"

    def test_a_value_matching_the_composition_branch_still_checks(self):
        context = check({"p": {"Either": {"Add": {"arg1": 1, "arg2": 2}}}})
        assert sorted(evaluation_of(context, "p", "Add").arguments) == ["arg1", "arg2"]

    def test_a_raise_inside_a_trial_branch_would_take_the_whole_parse_down(self):
        """
        The one that would have failed before the ``raise``\\ s became `IllFormedExpression`\\ s: ``Cyclic``
        supplies no argument, so the *first* branch tried reaches the acyclicity check. The value is still
        a valid `Either` by way of... nothing -- so it must be rejected, but rejected *as a reported
        failure*, which `rejection` asserts.
        """
        text = rejection({"p": {"Either": {"Cyclic": {}}}})
        assert "is not acyclic" in text
