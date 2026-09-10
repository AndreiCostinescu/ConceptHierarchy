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
A _ValueDomain_ that declares no ``instantiation`` says nothing about its values, so nothing is checked.

`[CH].md` 8.6: a non-abstract _ValueDomain_ with no instantiation gets the accept-everything schema, and
its value is therefore **opaque**. A boolean schema is handled before the custom-type branch of `_parse`,
so such a value never reaches a custom-type leaf -- and a custom-type leaf is the only place an expression
is parsed. Nothing inside is read as one: not a variable, not a Function name, not an argument name, and
not the `fEval:` / `fComp:` / `fInst:` markers.

That is the intended reading of "no instantiation", not a gap. It is worth pinning because it is the
**default** shape for a ValueDomain -- ``"data": {}`` silently disables all checking of its values -- and
because it is the one place a marker's commitment
(`documentation/TODO_FUNCTION_EVALUATION_VS_COMPOSITION.md` 2.4) provably cannot be enforced.

`examples/animal_kingdom.json` is the base: it declares `Open` (no instantiation) and the global
``unconstrained``, which is exactly this shape.
"""

from __future__ import annotations

import json
import warnings
from copy import deepcopy

import pytest

from concept_hierarchy.data.contexts.context import ConceptHierarchyContext
from concept_hierarchy.errors import ConceptHierarchyError
from tests.integration.test_expression_parsing import EXAMPLES_DIR, check_hierarchy

BASE = json.loads((EXAMPLES_DIR / "animal_kingdom.json").read_text())
EXTERNAL = json.loads((EXAMPLES_DIR / "external_animal_data.json").read_text())

HOLDER = {
    "Holder": {
        "directParents": ["ValueDomain"],
        "data": {
            "instantiation": {
                "type": "object",
                "additionalProperties": False,
                "required": [],
                "properties": {"open": {"type": "Open"}, "num": {"type": "Integer"}},
            }
        },
    }
}
"""A *typed* site next to the opaque one, so the contrast is between schemas and not between harnesses."""


def check(instances: dict) -> ConceptHierarchyContext:
    data = deepcopy(BASE)
    data["concepts"].update(deepcopy(HOLDER))
    data["instances"].update(deepcopy(instances))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return check_hierarchy(data, deepcopy(EXTERNAL))


def rejection(instances: dict) -> ConceptHierarchyError:
    with pytest.raises(ConceptHierarchyError) as excinfo:
        check(instances)
    return excinfo.value


def opaque(payload: object) -> dict:
    """A global whose value is an `Open` -- the ValueDomain that declares no instantiation."""
    return {"probe": {"Open": payload}}


NONSENSE = {
    "a Function evaluation shape": {"Add<Number>": {"arg1": 1, "arg2": 2}},
    "a variable that does not exist": {"Add<Number>": {"arg1": "no_such_variable", "arg2": 2}},
    "an argument the Function does not have": {"Add<Number>": {"nope": 1}},
    "a Function that does not exist": {"NoSuchFunction<Q>": {"arg1": 1}},
    "structure that is not an expression at all": {"???": [1, {"x": None}]},
    "a committed marker": {"fEval:Add<Number>": {"arg1": 1, "arg2": 2}},
}


class TestAnOpaquePayloadIsNotChecked:
    """
    Every one of these is accepted. They are listed together because the point is the *range*: it is not
    that one particular mistake slips through, it is that the whole payload is unread.
    """

    @pytest.mark.parametrize("payload", list(NONSENSE.values()), ids=list(NONSENSE))
    def test_the_payload_is_accepted_whatever_it_contains(self, payload):
        assert check(opaque(payload)).model.instances["probe"].value.is_valid

    def test_the_shipped_example_relies_on_this(self):
        """``unconstrained`` in `animal_kingdom.json` is exactly this shape, marker and all."""
        declared = BASE["instances"]["unconstrained"]
        assert "Open" in declared, declared
        assert any(k.startswith("fEval:") for k in declared["Open"]), declared
        assert check({}).model.instances["unconstrained"].value.is_valid

    def test_a_marker_inside_an_opaque_payload_is_inert(self):
        """
        Not a contradiction of the consumption check, which refuses a commitment an accept-everything
        schema drops: that check fires where the marker is written on the value's *own* key. Here the
        site's key is ``Open`` -- unmarked -- and everything under it is `Open`'s unread payload, so the
        ``fEval:`` inside is text in that payload and commits nothing.
        """
        assert check(opaque({"fEval:Add<Number>": {"arg1": 1, "arg2": 2}})).model.instances["probe"].value.is_valid
        assert check(opaque({"fInst:NoSuchFunction": {"nonsense": True}})).model.instances["probe"].value.is_valid


class TestTheSameValuesAreCheckedAtATypedSite:
    """
    The contrast that makes the point above about the *schema* rather than about the values: put the same
    payloads where a schema does describe them, and each is diagnosed.
    """

    def test_a_variable_that_does_not_exist_is_diagnosed(self):
        error = rejection({"probe": {"Holder": {"num": {"Add<Integer>": {"arg1": "no_such_variable", "arg2": 2}}}}})
        assert "not a variable" in _messages(error), _messages(error)[:300]

    def test_an_argument_the_function_does_not_have_is_diagnosed(self):
        error = rejection({"probe": {"Holder": {"num": {"Add<Integer>": {"nope": 1}}}}})
        assert 'does not have the argument "nope"' in _messages(error), _messages(error)[:300]

    def test_a_function_that_does_not_exist_is_diagnosed(self):
        error = rejection({"probe": {"Holder": {"num": {"NoSuchFunction<Q>": {"arg1": 1}}}}})
        assert _messages(error)

    def test_an_open_typed_site_is_not_itself_opaque(self):
        """
        The distinction the whole module turns on. ``{"Open": payload}`` is a `Narrow`, and the *payload*
        goes straight to `Open`'s accept-everything schema without ever being classified. A value **at** an
        `Open`-typed site is a different thing: it is an expression like any other, so the cascade runs and
        `FEval` commits on ``Add<Number>`` -- whose `Number` result is not an `Open`.

        So the opacity belongs to the payload of a `Narrow`, not to the type.
        """
        error = rejection({"probe": {"Holder": {"open": {"Add<Number>": {"arg1": 1, "arg2": 2}}}}})
        assert "not a subtype" in _messages(error), _messages(error)[:300]

    def test_the_very_same_value_is_accepted_as_a_narrow_payload(self):
        """One value, two positions, opposite answers."""
        assert check(opaque({"Add<Number>": {"arg1": 1, "arg2": 2}})).model.instances["probe"].value.is_valid


class TestACommitmentCannotReachHere:
    """
    A marker is honored at a custom-type leaf; an accept-everything schema has none, so the commitment is
    never created and there is nothing to enforce. This is the boundary of the check in
    `_check_instantiation_schema`, and it is by design rather than by omission.
    """

    def test_a_committed_evaluation_is_accepted_unread(self):
        assert check(opaque({"fEval:Add<Number>": {"arg1": 1, "arg2": 2}})).model.instances["probe"].value.is_valid

    def test_it_is_accepted_for_the_same_reason_nonsense_is(self):
        """If the marker were being read, these two would not behave alike."""
        marked = check(opaque({"fEval:Add<Number>": {"arg1": 1, "arg2": 2}}))
        nonsense = check(opaque({"???": [1, {"x": None}]}))
        assert marked.model.instances["probe"].value.is_valid
        assert nonsense.model.instances["probe"].value.is_valid

    def test_a_misused_marker_is_not_diagnosed_either(self):
        """On a non-Function key this is an error anywhere a schema describes the value."""
        assert check(opaque({"fEval:Integer": 1})).model.instances["probe"].value.is_valid


def _messages(error: ConceptHierarchyError) -> str:
    collected: list[str] = []

    def walk(err: ConceptHierarchyError) -> None:
        collected.append(err.args[0] if err.args else "")
        for cause in err.causes:
            walk(cause)

    walk(error)
    return " | ".join(collected)
