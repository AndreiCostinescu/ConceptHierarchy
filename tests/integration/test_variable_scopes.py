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
What a name means, and where: instance property chains, and the three scoping constructs.

Four features share one subject -- which names an expression may use at a given place -- so they are
tested together, and the interactions between them get as much attention as the features themselves:

1. **`instance.property` / `instance.function`** in a variable position, chained across `Instance`-typed
   properties. On an `Instance<C...>` variable the members of `C...` are known; on an `InstanceBase`
   variable nothing is, so the access is *warned* about rather than rejected.
2. **A `CustomFunction` runs in its own frame.** Its procedure sees the globals, its own interface
   arguments, and its locals -- and nothing from the scope it was written in.
3. **`subScopes`** puts a variable in scope inside *one argument* of an evaluation, and nowhere else.
4. **`addNewVariablesInExistingScope`** puts a variable in the surrounding scope, *after* the evaluation.

`examples/animal_kingdom.json` is the base: it already supplies `Instance`, `InstanceBase`, `Animal`,
`Dog`, `CustomFunction`, `Return`, `FunctionSequence`, `CreateLocalVariable`, `List`, `Add` and `Integer`,
so a test writes only what it is about. `IterateList` is spliced in, copied from the `subScopes`
declaration of `IterateSet` in `examples/ownership.json`, because `animal_kingdom` declares none.

Two facts about the harness shape every test here, and both were learned the hard way:

* **a global's type is inferred from its value**, so a bare ``"a.b"`` is checked against `String` -- what a
  JSON string serializes to -- and says nothing about chains. Anything whose *site type* matters goes in
  a Function argument, via `integer_site` / `string_site`;
* **a Function that returns nothing** (`FunctionSequence`, `IterateList`) cannot be a `Return`'s ``what``
  or a global's value. It goes straight into a `CustomFunction`'s ``procedure``, via `in_procedure`.

And one trap that costs a test its meaning: **a bare JSON string that is not in scope is not an error, it
is a `String`**, by default serialization. So ``"breed"`` at a `String`-typed site parses whether or not
``breed`` is a variable, and a test written there proves nothing. Every test that asks *"is this name in
scope"* therefore puts it at a site a `String` cannot satisfy -- `integer_site` -- where the answer is
legible in the diagnosis: ``as variable (String): Type String of variable breed ...`` means it resolved,
``as default serialization (String)`` means it did not.
"""

from __future__ import annotations

import json
import warnings
from copy import deepcopy

import pytest

from concept_hierarchy.data.contexts.context import ConceptHierarchyContext
from concept_hierarchy.data.expressions.expression import Expression
from concept_hierarchy.data.expressions.subexpressions import InstancePropertyChain, Variable
from concept_hierarchy.errors import CHWarning, ConceptHierarchyError
from tests.integration.test_expression_parsing import EXAMPLES_DIR, check_hierarchy

BASE = json.loads((EXAMPLES_DIR / "animal_kingdom.json").read_text())
EXTERNAL = json.loads((EXAMPLES_DIR / "external_animal_data.json").read_text())


# --------------------------------------------------------------------------------------------------
# Spliced vocabulary
# --------------------------------------------------------------------------------------------------

ITERATE_LIST = {
    "IterateList": {
        "directParents": ["Function"],
        "data": {
            "templateContext": ["T"],
            "interface": {"list": "List<T>", "what": "List<FunctionComposition>"},
            "subScopes": {"what": {"value": "T"}},
        },
    }
}
"""``IterateSet``'s shape from ``examples/ownership.json``: one statically named sub-scope variable."""

ITERATE_NAMED = {
    "IterateNamed": {
        "directParents": ["Function"],
        "data": {
            "templateContext": ["T"],
            "interface": {"list": "List<T>", "varName": "String", "what": "List<FunctionComposition>"},
            "subScopes": {"what": {"varName": ["T", True]}},
        },
    }
}
"""The dynamic form: the sub-scope variable's *name* is whatever the ``varName`` argument evaluates to."""

TAKES_STRING = {
    "TakesString": {"directParents": ["Function"], "data": {"interface": {"s": "String", "res": "Boolean"}}}
}
"""A `String`-typed argument, so an expression can be put at a site whose type it does not satisfy."""

KENNEL = {
    "Kennel": {
        "directParents": ["Concept"],
        "data": {
            "properties": {
                "resident": {"valueDomain": "Instance<Dog>"},
                "capacity": {"valueDomain": "Integer"},
            }
        },
    }
}
"""A DomainConcept with an `Instance`-typed property, so a chain has something to chain *through*."""

MAKE_STRING = {"MakeString": {"directParents": ["Function"], "data": {"interface": {"res": "String"}}}}
"""A Function that actually returns a `String` -- `Return<T>` declares no ``res`` and cannot stand in."""

VOCABULARY = {**ITERATE_LIST, **ITERATE_NAMED, **TAKES_STRING, **KENNEL, **MAKE_STRING}

INSTANCES = {
    "animalInst": {
        "Instance<Animal>": {"instanceName": "animalInst", "concepts": ["s:Animal"], "properties": {"age": 3}}
    },
    "dogInst": {
        "Instance<Dog>": {
            "instanceName": "dogInst",
            "concepts": ["s:Dog"],
            "properties": {"age": 2, "breed": "s:lab"},
        }
    },
    "kennelInst": {
        "Instance<Kennel>": {"instanceName": "kennelInst", "concepts": ["s:Kennel"], "properties": {"capacity": 4}}
    },
}


# --------------------------------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------------------------------


def check(instances: dict | None = None, concepts: dict | None = None) -> ConceptHierarchyContext:
    """`animal_kingdom` plus this module's vocabulary, plus whatever the test adds."""
    data = deepcopy(BASE)
    data["concepts"].update(deepcopy(VOCABULARY))
    data["instances"].update(deepcopy(INSTANCES))
    if concepts:
        data["concepts"].update(deepcopy(concepts))
    if instances:
        data["instances"].update(deepcopy(instances))
    return check_hierarchy(data, deepcopy(EXTERNAL))


def rejection(instances: dict | None = None, concepts: dict | None = None) -> ConceptHierarchyError:
    """The hierarchy must be rejected. Warnings are not the subject here, so they are silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ConceptHierarchyError) as excinfo:
            check(instances, concepts)
    return excinfo.value


def accepted(instances: dict | None = None, concepts: dict | None = None) -> ConceptHierarchyContext:
    """Check, ignoring warnings -- for tests whose subject is acceptance, not the warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return check(instances, concepts)


def messages(error: ConceptHierarchyError) -> str:
    collected: list[str] = []

    def walk(err: ConceptHierarchyError) -> None:
        collected.append(err.args[0] if err.args else "")
        for cause in err.causes:
            walk(cause)

    walk(error)
    return " | ".join(collected)


def _locations(error: ConceptHierarchyError, fragment: str) -> list[str]:
    """Every location in the cause tree whose message contains ``fragment``."""
    found: list[str] = []

    def walk(err: ConceptHierarchyError) -> None:
        if fragment in (err.args[0] if err.args else ""):
            found.append("" if err.location_id is None else err.location_id.print())
        for cause in err.causes:
            walk(cause)

    walk(error)
    return found


def expression_of(context: ConceptHierarchyContext, name: str = "probe") -> Expression:
    return context.model.instances[name].value


def argument(context: ConceptHierarchyContext, argument_name: str, name: str = "probe") -> Expression:
    """One argument of a global whose value is a Function evaluation."""
    return expression_of(context, name).value.arguments[argument_name]


def probe(expression: object) -> dict:
    return {"probe": expression}


def assert_resolved_as_variable(error: ConceptHierarchyError, name: str, type_name: str) -> None:
    """``name`` was in scope: the diagnosis names it as a *variable*, with the type it resolved to."""
    text = messages(error)
    assert f"of variable {name}" in text, f"{name!r} did not resolve as a variable; got: {text[:400]}"
    assert f"Type {type_name} of variable {name}" in text, text[:400]


def assert_not_a_variable(error: ConceptHierarchyError, name: str) -> None:
    """
    ``name`` was *not* in scope, so the bare string fell through to default serialization as a `String`.

    Asserting only "the hierarchy was rejected" would pass either way; asserting the absence of the
    variable reading is what makes this mean something.
    """
    text = messages(error)
    assert f"of variable {name}" not in text, f"{name!r} did resolve as a variable; got: {text[:400]}"
    assert "as default serialization" in text, text[:400]


def in_procedure(procedure: object, interface: dict | None = None) -> dict:
    """A `CustomFunction` whose body *is* ``procedure`` -- where a returns-nothing Function belongs."""
    body: dict = {"procedure": procedure}
    if interface is not None:
        body["interface"] = interface
    return {"CustomFunction": body}


def custom_function(expression: object, interface: dict | None = None, result: str = "Integer") -> dict:
    """A `CustomFunction` whose procedure returns ``expression``."""
    return in_procedure({f"Return<{result}>": {"what": expression}}, interface)


def integer_site(expression: object) -> dict:
    return {"Add<Integer>": {"arg1": expression, "arg2": 1}}


def string_site(expression: object) -> dict:
    return {"TakesString": {"s": expression}}


def sequence(*compositions: object) -> dict:
    return {"FunctionSequence": {"fs": list(compositions)}}


def iterate(what: list, list_value: object = None, function: str = "IterateList<Integer>") -> dict:
    return {function: {"list": [1, 2] if list_value is None else list_value, "what": what}}


def create(name: object = "s:x", init: object = True, template: str = "Boolean") -> dict:
    return {f"CreateLocalVariable<{template}>": {"variableName": name, "init": init}}


def _dog_with_functions(functions: dict) -> dict:
    """`Dog` from the example, with ``functions`` merged into its declared ones."""
    dog = deepcopy(BASE["concepts"]["Dog"])
    dog["data"].setdefault("functions", {}).update(functions)
    return dog


# ==================================================================================================
# 1. Instance property chains
# ==================================================================================================


class TestAChainResolvesThroughAnInstanceType:
    def test_a_property_of_an_accepted_concept_resolves(self):
        context = check(probe(integer_site("animalInst.age")))
        assert expression_of(context).is_valid

    def test_the_chain_is_its_own_expression_kind(self):
        """`InstancePropertyChain` is a `Variable` subclass, so the check is on the exact class."""
        context = check(probe(integer_site("animalInst.age")))
        value = argument(context, "arg1").value
        assert type(value) is InstancePropertyChain, type(value).__name__

    def test_a_plain_variable_is_not_read_as_a_chain(self):
        context = check(probe(integer_site("one")))
        assert type(argument(context, "arg1").value) is Variable

    def test_the_chains_type_is_the_property_s_type(self):
        context = check(probe(integer_site("animalInst.age")))
        assert argument(context, "arg1").value.value_type.full_name == "Integer"

    def test_a_string_property_resolves_to_its_own_type(self):
        """At a `String` site a bare string would also parse, so the *class* is what settles it."""
        context = check(probe(string_site("dogInst.breed")))
        parsed = argument(context, "s").value
        assert type(parsed) is InstancePropertyChain, type(parsed).__name__
        assert parsed.value_type.full_name == "String"

    def test_an_inherited_property_resolves(self):
        """``age`` is declared on `Animal`; ``dogInst`` accepts `Dog`."""
        context = check(probe(integer_site("dogInst.age")))
        assert argument(context, "arg1").value.value_type.full_name == "Integer"

    def test_a_chain_at_a_site_whose_type_it_does_not_satisfy_is_rejected(self):
        error = rejection(probe(string_site("animalInst.age")))
        assert "not a subtype" in messages(error), messages(error)[:300]

    def test_two_chains_in_one_evaluation(self):
        context = check(probe({"Add<Integer>": {"arg1": "animalInst.age", "arg2": "dogInst.age"}}))
        assert expression_of(context).is_valid


class TestChainingThroughAnInstanceTypedProperty:
    def test_a_two_step_chain_resolves(self):
        context = check(probe(string_site("kennelInst.resident.breed")))
        parsed = argument(context, "s").value
        assert type(parsed) is InstancePropertyChain, type(parsed).__name__
        assert parsed.value_type.full_name == "String"

    def test_a_two_step_chain_is_still_one_chain_expression(self):
        context = check(probe(integer_site("kennelInst.resident.age")))
        assert isinstance(argument(context, "arg1").value, InstancePropertyChain)

    def test_a_step_through_a_non_instance_property_is_rejected(self):
        """``capacity`` is an `Integer`, so nothing can be read off it."""
        error = rejection(probe(integer_site("kennelInst.capacity.age")))
        assert messages(error)

    def test_a_missing_property_at_the_second_step_is_diagnosed(self):
        error = rejection(probe(string_site("kennelInst.resident.nosuchprop")))
        assert messages(error)


class TestWhatTheRootMustBe:
    def test_a_root_that_is_not_a_variable_is_rejected(self):
        error = rejection(probe(integer_site("nosuchinstance.age")))
        assert "not a variable" in messages(error), messages(error)[:300]

    def test_a_root_that_is_not_an_instance_is_rejected(self):
        """``one`` is a global `Integer`; there is nothing to read a property off."""
        error = rejection(probe(integer_site("one.age")))
        assert messages(error)

    def test_a_string_literal_is_not_a_chain(self):
        """The ``s:`` prefix makes it a `String`, dots and all."""
        context = check(probe(string_site("s:animalInst.age")))
        assert type(argument(context, "s").value) is not InstancePropertyChain

    def test_a_dotless_name_is_never_a_chain(self):
        context = check(probe(integer_site("one")))
        assert type(argument(context, "arg1").value) is Variable


class TestAnInstanceBaseRootIsWarnedAboutRatherThanRejected:
    """
    An `InstanceBase` variable carries no concept list, so nothing can be said about which properties it
    has -- and the same holds for a property that no concept of an ``Instance<C...>`` declares. Neither is
    an error, because it may well succeed at runtime; both are warnings.
    """

    def test_a_property_access_on_an_instance_base_warns(self):
        with pytest.warns(CHWarning):
            check(probe(integer_site("MyAnimal.age")))

    def test_the_warned_access_is_still_accepted(self):
        context = accepted(probe(integer_site("MyAnimal.age")))
        assert expression_of(context).is_valid

    def test_a_property_no_accepted_concept_declares_warns(self):
        """``breed`` is a `Dog` property; ``animalInst`` accepts only `Animal`."""
        with pytest.warns(CHWarning):
            check(probe(string_site("animalInst.breed")))

    def test_a_property_declared_by_an_accepted_concept_does_not_warn(self):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            check(probe(integer_site("animalInst.age")))
        about_the_chain = [w for w in record if isinstance(w.message, CHWarning) and "animalInst" in str(w.message)]
        assert not about_the_chain, [str(w.message) for w in about_the_chain]


class TestAFunctionMemberOfAnInstance:
    """``EvaluateFunctionRes<T>`` declares ``f: CustomFunction``, which is the site a member access fits."""

    def test_a_function_member_resolves(self):
        context = accepted(probe({"EvaluateFunctionRes<Integer>": {"f": "dogInst.f1", "args": {}}}))
        assert expression_of(context).is_valid

    def test_a_function_member_has_the_custom_function_type(self):
        context = accepted(probe({"EvaluateFunctionRes<Integer>": {"f": "dogInst.f1", "args": {}}}))
        assert argument(context, "f").value.value_type.full_name == "CustomFunction"

    def test_a_member_that_is_neither_property_nor_function_is_diagnosed(self):
        error = rejection(probe(integer_site("dogInst.nosuchmember")))
        assert messages(error)


# ==================================================================================================
# 2. A CustomFunction runs in its own frame
# ==================================================================================================


class TestWhatACustomFunctionProcedureCanSee:
    def test_it_sees_the_globals(self):
        context = check(probe(custom_function("one")))
        assert expression_of(context).is_valid

    def test_it_sees_its_own_interface_arguments(self):
        context = check(probe(custom_function("n", interface={"n": ["s:Integer", "Get"]})))
        assert expression_of(context).is_valid

    def test_a_name_that_is_neither_is_not_a_variable(self):
        error = rejection(probe(custom_function("nosuchname")))
        assert "not a variable" in messages(error), messages(error)[:300]

    def test_without_an_interface_there_are_no_argument_variables(self):
        error = rejection(probe(custom_function("n")))
        assert "not a variable" in messages(error), messages(error)[:300]

    def test_an_interface_argument_shadows_a_global_of_the_same_name(self):
        """``one`` is a global `Integer`; here it is a `String` argument, and the argument must win."""
        error = rejection(probe(custom_function(integer_site("one"), interface={"one": ["s:String", "Get"]})))
        assert "not a subtype" in messages(error), messages(error)[:300]

    def test_an_instance_chain_may_be_rooted_at_an_interface_argument(self):
        context = check(
            probe(custom_function(integer_site("inst.age"), interface={"inst": ["s:Instance<Animal>", "Get"]}))
        )
        assert expression_of(context).is_valid


class TestTheFrameDoesNotSeeTheEnclosingScope:
    def test_a_sub_scope_variable_of_an_enclosing_evaluation_is_not_visible(self):
        """``value`` is in scope inside ``what``; a `CustomFunction` written there does not inherit it."""
        error = rejection(
            probe(in_procedure(iterate([{"Return<CustomFunction>": {"what": custom_function("value")}}])))
        )
        assert "not a variable" in messages(error), messages(error)[:300]

    def test_a_variable_added_to_the_enclosing_scope_is_not_visible(self):
        error = rejection(
            probe(
                in_procedure(
                    sequence(
                        create(),
                        {"Return<CustomFunction>": {"what": custom_function("x", result="Boolean")}},
                    )
                )
            )
        )
        assert "not a variable" in messages(error), messages(error)[:300]

    def test_a_nested_custom_function_does_not_see_the_outer_interface(self):
        error = rejection(
            probe(
                custom_function(
                    custom_function("outer"),
                    interface={"outer": ["s:Integer", "Get"]},
                    result="CustomFunction",
                )
            )
        )
        assert "not a variable" in messages(error), messages(error)[:300]

    def test_the_nested_one_still_sees_the_globals(self):
        context = check(
            probe(
                custom_function(
                    custom_function("one"),
                    interface={"outer": ["s:Integer", "Get"]},
                    result="CustomFunction",
                )
            )
        )
        assert expression_of(context).is_valid


class TestEveryShorthandFormGetsTheSameFrame:
    """
    A `CustomFunction` may be written several ways -- including a plain `FunctionComposition`, which its
    instantiation accepts. The frame is a property of the concept, not of the spelling.
    """

    def test_the_plain_function_composition_form(self):
        context = check(probe({"CustomFunction": {"Return<Integer>": {"what": "one"}}}))
        assert expression_of(context).is_valid

    def test_the_plain_function_composition_form_has_no_argument_variables(self):
        error = rejection(probe({"CustomFunction": {"Return<Integer>": {"what": "n"}}}))
        assert "not a variable" in messages(error), messages(error)[:300]

    def test_the_narrowed_interface_and_procedure_form(self):
        context = check(probe(custom_function("n", interface={"n": ["s:Integer", "Get"]})))
        assert expression_of(context).is_valid

    def test_a_domain_concept_function_written_bare(self):
        """``Dog.f3``'s shape: the `{interface, procedure}` object as the function's value."""
        concepts = {
            "Dog": _dog_with_functions(
                {"probeFn": {"interface": {"arg": "s:Number"}, "procedure": {"Return<Number>": {"what": "arg"}}}}
            )
        }
        assert "Dog" in check(concepts=concepts).model.domain_concepts

    def test_a_domain_concept_function_written_with_value_domain_and_default(self):
        """``Dog.f4``'s shape."""
        concepts = {
            "Dog": _dog_with_functions(
                {
                    "probeFn": {
                        "valueDomain": "CustomFunction",
                        "static": True,
                        "default": {
                            "interface": {"arg": "s:Number"},
                            "procedure": {"Return<Number>": {"what": "arg"}},
                        },
                    }
                }
            )
        }
        assert "Dog" in check(concepts=concepts).model.domain_concepts

    def test_a_domain_concept_function_body_sees_its_own_interface(self):
        concepts = {
            "Dog": _dog_with_functions(
                {"probeFn": {"interface": {"arg": "s:Number"}, "procedure": {"Return<Number>": {"what": "arg"}}}}
            )
        }
        assert "Dog" in check(concepts=concepts).model.domain_concepts

    def test_a_domain_concept_function_body_does_not_see_an_unknown_name(self):
        concepts = {
            "Dog": _dog_with_functions(
                {"probeFn": {"interface": {"arg": "s:Number"}, "procedure": {"Return<Number>": {"what": "nosuch"}}}}
            )
        }
        error = rejection(concepts=concepts)
        assert "not a variable" in messages(error), messages(error)[:300]


# ==================================================================================================
# 3. subScopes
# ==================================================================================================


class TestASubScopeVariableIsVisibleInItsArgument:
    def test_the_variable_is_in_scope_inside_that_argument(self):
        context = check(probe(in_procedure(iterate([{"Return<Integer>": {"what": "value"}}]))))
        assert expression_of(context).is_valid

    def test_its_type_is_the_substituted_template_argument(self):
        error = rejection(probe(in_procedure(iterate([{"Return<Boolean>": {"what": string_site("value")}}]))))
        assert "not a subtype" in messages(error), messages(error)[:300]

    def test_it_is_visible_at_depth_inside_that_argument(self):
        context = check(probe(in_procedure(iterate([{"Return<Integer>": {"what": integer_site("value")}}]))))
        assert expression_of(context).is_valid

    def test_it_is_visible_in_every_element_of_the_argument(self):
        context = check(
            probe(
                in_procedure(
                    iterate(
                        [
                            {"Return<Integer>": {"what": "value"}},
                            {"Return<Integer>": {"what": integer_site("value")}},
                        ]
                    )
                )
            )
        )
        assert expression_of(context).is_valid


class TestASubScopeVariableDoesNotLeak:
    def test_it_is_not_visible_in_a_sibling_argument(self):
        error = rejection(probe(in_procedure(iterate([{"Return<Integer>": {"what": 1}}], list_value=["value"]))))
        assert "not a variable" in messages(error), messages(error)[:300]

    def test_it_is_not_visible_after_the_evaluation(self):
        error = rejection(
            probe(
                in_procedure(
                    sequence(
                        iterate([{"Return<Integer>": {"what": "value"}}]),
                        {"Return<Integer>": {"what": "value"}},
                    )
                )
            )
        )
        assert "not a variable" in messages(error), messages(error)[:300]

    def test_it_is_not_visible_in_an_unrelated_expression(self):
        error = rejection(probe(integer_site("value")))
        assert "not a variable" in messages(error), messages(error)[:300]

    def test_two_nested_iterations_each_have_their_own(self):
        """The inner shadows the outer; both are `Integer` here, so this must simply parse."""
        context = check(probe(in_procedure(iterate([iterate([{"Return<Integer>": {"what": "value"}}])]))))
        assert expression_of(context).is_valid


class TestADynamicallyNamedSubScopeVariable:
    @staticmethod
    def named(var_name: object, what: list) -> dict:
        return {"IterateNamed<Integer>": {"list": [1, 2], "varName": var_name, "what": what}}

    def test_a_default_serialized_name_is_accepted(self):
        context = check(probe(in_procedure(self.named("s:element", [{"Return<Integer>": {"what": "element"}}]))))
        assert expression_of(context).is_valid

    def test_a_variable_as_the_name_is_not_implemented(self):
        with pytest.raises(NotImplementedError):
            check(probe(in_procedure(self.named("oneStr", [{"Return<Integer>": {"what": 1}}]))))

    def test_a_function_evaluation_as_the_name_is_not_implemented(self):
        with pytest.raises(NotImplementedError):
            check(probe(in_procedure(self.named({"MakeString": {}}, [{"Return<Integer>": {"what": 1}}]))))

    def test_a_narrowed_name_is_not_implemented(self):
        """
        ``{"String": "s:element"}`` is a `Narrow`, which subclasses `InstExpression` -- so by the stated
        rule it ought to be read, or refused with `NotImplementedError`.
        """
        with pytest.raises(NotImplementedError):
            check(probe(in_procedure(self.named({"String": "s:element"}, [{"Return<Integer>": {"what": "element"}}]))))


# ==================================================================================================
# 4. addNewVariablesInExistingScope
# ==================================================================================================


class TestANewVariableAppearsAfterTheEvaluation:
    def test_a_later_element_of_the_sequence_sees_it(self):
        context = check(probe(in_procedure(sequence(create(), {"Return<Boolean>": {"what": "x"}}))))
        assert expression_of(context).is_valid

    def test_an_earlier_element_does_not(self):
        error = rejection(probe(in_procedure(sequence({"Return<Boolean>": {"what": "x"}}, create()))))
        assert "not a variable" in messages(error), messages(error)[:300]

    def test_the_evaluations_own_arguments_do_not(self):
        """``init`` is evaluated as part of the very call that creates ``x``."""
        error = rejection(probe(in_procedure(sequence(create(init="x")))))
        assert "not a variable" in messages(error), messages(error)[:300]

    def test_the_new_variable_has_the_substituted_template_type(self):
        error = rejection(probe(in_procedure(sequence(create(), {"Return<Integer>": {"what": integer_site("x")}}))))
        assert "not a subtype" in messages(error), messages(error)[:300]


class TestTheNewVariableIsScopedToItsFrame:
    def test_it_does_not_escape_a_custom_function(self):
        error = rejection(
            probe(
                in_procedure(
                    sequence(
                        {"Return<CustomFunction>": {"what": in_procedure(sequence(create()))}},
                        {"Return<Boolean>": {"what": "x"}},
                    )
                )
            )
        )
        assert "not a variable" in messages(error), messages(error)[:300]

    def test_it_is_not_visible_in_another_global(self):
        error = rejection({"maker": in_procedure(sequence(create())), "user": integer_site("x")})
        assert "not a variable" in messages(error), messages(error)[:300]


class TestADynamicallyNamedNewVariable:
    def test_a_default_serialized_name_is_accepted(self):
        context = check(probe(in_procedure(sequence(create("s:flag"), {"Return<Boolean>": {"what": "flag"}}))))
        assert expression_of(context).is_valid

    def test_a_variable_as_the_name_is_not_implemented(self):
        with pytest.raises(NotImplementedError):
            check(probe(in_procedure(sequence(create("oneStr")))))

    def test_a_function_evaluation_as_the_name_is_not_implemented(self):
        with pytest.raises(NotImplementedError):
            check(probe(in_procedure(sequence(create({"MakeString": {}})))))

    def test_a_narrowed_name_is_not_implemented(self):
        """
        ``{"String": "s:flag"}`` is a `Narrow`, which *subclasses* `InstExpression` -- so the rule "only
        an Inst or a default serialization is read" has to decide it explicitly rather than fall into an
        assertion. The `subScopes` path still does the latter; see its own test.
        """
        with pytest.raises(NotImplementedError):
            check(probe(in_procedure(sequence(create({"String": "s:flag"})))))


# ==================================================================================================
# 5. Redeclaration and shadowing
# ==================================================================================================


class TestReDeclaringAName:
    """
    A variable is created once in a scope. What a *second* creation of the same name does is the open
    question these pin -- and whichever way it goes, it should go the same way for both types.
    """

    def test_the_same_name_twice_in_one_scope_is_reported(self):
        error = rejection(
            probe(in_procedure(sequence(create(), create(init=False), {"Return<Boolean>": {"what": "x"}})))
        )
        assert "already contains these variables" in messages(error), messages(error)[:300]

    def test_it_is_reported_against_the_second_creation(self):
        """Index 1 of the sequence -- the one that could not be made, not the one that succeeded."""
        error = rejection(
            probe(in_procedure(sequence(create(), create(init=False), {"Return<Boolean>": {"what": "x"}})))
        )
        sites = _locations(error, "already contains these variables")
        assert any(site.endswith('"fs": 1: "CreateLocalVariable<Boolean>"') for site in sites), sites

    def test_the_same_name_twice_with_different_types_is_reported_the_same_way(self):
        error = rejection(
            probe(
                in_procedure(sequence(create(), create(init=1, template="Integer"), {"Return<Boolean>": {"what": "x"}}))
            )
        )
        assert "already contains these variables" in messages(error), messages(error)[:300]

    def test_a_local_may_shadow_a_global(self):
        """Shadowing across scopes is not redeclaration and must be allowed."""
        context = check(probe(in_procedure(sequence(create("s:one"), {"Return<Boolean>": {"what": "one"}}))))
        assert expression_of(context).is_valid

    def test_a_sub_scope_variable_may_shadow_a_global(self):
        context = check(
            probe(
                in_procedure(
                    {
                        "IterateNamed<Integer>": {
                            "list": [1, 2],
                            "varName": "s:one",
                            "what": [{"Return<Integer>": {"what": integer_site("one")}}],
                        }
                    }
                )
            )
        )
        assert expression_of(context).is_valid

    def test_re_declaring_a_variable_with_the_same_name_same_type_as_a_sub_scope_variable_raises(self):
        error = rejection(
            probe(
                in_procedure(
                    {
                        "IterateNamed<Integer>": {
                            "list": [1, 2],
                            "varName": "s:one",
                            "what": [
                                {"CreateLocalVariable<Integer>": {"variableName": "s:one", "init": 0}},
                                {"Return<Integer>": {"what": integer_site("one")}},
                            ],
                        }
                    }
                )
            )
        )
        sites = _locations(error, "already contains these variables: ['one']")
        assert any(
            site.endswith('"IterateNamed<Integer>": "what": 0: "CreateLocalVariable<Integer>"') for site in sites
        ), sites

    def test_re_declaring_a_variable_with_the_same_name_different_type_as_a_sub_scope_variable_raises(self):
        error = rejection(
            probe(
                in_procedure(
                    {
                        "IterateNamed<Integer>": {
                            "list": [1, 2],
                            "varName": "s:one",
                            "what": [
                                {
                                    "CreateLocalVariable<String>": {
                                        "variableName": "s:one",
                                        "init": "OneVariableToRuleThemAll",
                                    }
                                },
                                {"Return<Integer>": {"what": integer_site("one")}},
                            ],
                        }
                    }
                )
            )
        )
        sites = _locations(error, "already contains these variables: ['one']")
        assert any(
            site.endswith('"IterateNamed<Integer>": "what": 0: "CreateLocalVariable<String>"') for site in sites
        ), sites


# ==================================================================================================
# 6. DomainConcept functions: the instance frame
# ==================================================================================================


def dog_function(procedure: object, interface: dict | None = None, static: bool = False) -> dict:
    """`Dog` with one added function, written non-statically or as ``Dog.f4``'s static form."""
    body: dict = {"procedure": procedure}
    body["interface"] = {"arg": "s:Number"} if interface is None else interface
    fn = {"valueDomain": "CustomFunction", "static": True, "default": body} if static else body
    return {"Dog": _dog_with_functions({"probeFn": fn})}


class TestANonStaticDomainConceptFunctionSeesItsInstance:
    """
    A non-static function of a _DomainConcept_ runs against an instance, so its frame carries that
    concept's declared properties **and** a variable ``instance`` of type ``Instance<ThatConcept>``.
    """

    def test_it_sees_a_property_declared_by_its_concept(self):
        """
        ``breed`` is a `String`, so it is put at an `Integer` site: the rejection then says *why* it was
        rejected, and "as variable (String)" is the proof that it resolved at all.
        """
        concepts = dog_function({"Return<Integer>": {"what": integer_site("breed")}})
        assert_resolved_as_variable(rejection(concepts=concepts), "breed", "String")

    def test_it_can_use_that_property_at_a_site_of_its_own_type(self):
        concepts = dog_function({"Return<String>": {"what": "breed"}})
        assert "Dog" in check(concepts=concepts).model.domain_concepts

    def test_it_sees_a_property_inherited_from_a_parent_concept(self):
        concepts = dog_function({"Return<Integer>": {"what": "age"}})
        assert "Dog" in check(concepts=concepts).model.domain_concepts

    def test_it_has_an_instance_variable(self):
        concepts = dog_function({"Return<Instance<Dog>>": {"what": "instance"}})
        assert "Dog" in check(concepts=concepts).model.domain_concepts

    def test_the_instance_variable_resolves_as_a_variable(self):
        """At an `Integer` site the diagnosis names it, which a bare string could never produce."""
        concepts = dog_function({"Return<Integer>": {"what": integer_site("instance")}})
        assert_resolved_as_variable(rejection(concepts=concepts), "instance", "Instance<[Dog], []>")

    def test_the_instance_variable_is_typed_by_the_declaring_concept(self):
        """``Instance<Animal>`` is not ``Instance<Dog>``, so a wrong annotation must be refused."""
        concepts = dog_function({"Return<Instance<Animal>>": {"what": "instance"}})
        error = rejection(concepts=concepts)
        assert "not a subtype" in messages(error), messages(error)[:300]

    def test_a_chain_may_be_rooted_at_the_instance_variable(self):
        concepts = dog_function({"Return<Integer>": {"what": "instance.age"}})
        assert "Dog" in accepted(concepts=concepts).model.domain_concepts

    def test_an_unknown_name_is_still_unknown(self):
        concepts = dog_function({"Return<Integer>": {"what": "nosuchthing"}})
        error = rejection(concepts=concepts)
        assert "not a variable" in messages(error), messages(error)[:300]


class TestAStaticDomainConceptFunctionHasNoInstance:
    def test_it_has_no_instance_variable(self):
        concepts = dog_function({"Return<Instance<Dog>>": {"what": "instance"}}, static=True)
        error = rejection(concepts=concepts)
        assert messages(error)

    def test_it_does_not_see_its_concepts_properties(self):
        """
        Without an instance there is nothing to read ``breed`` off, so the name must not resolve -- and
        the `Integer` site is what makes that legible: at a `String` site the bare name parses either way.
        """
        concepts = dog_function({"Return<Integer>": {"what": integer_site("breed")}}, static=True)
        assert_not_a_variable(rejection(concepts=concepts), "breed")

    def test_it_does_not_see_inherited_properties_either(self):
        concepts = dog_function({"Return<Integer>": {"what": integer_site("age")}}, static=True)
        assert_not_a_variable(rejection(concepts=concepts), "age")

    def test_it_still_sees_the_globals(self):
        concepts = dog_function({"Return<Integer>": {"what": "one"}}, static=True)
        assert "Dog" in check(concepts=concepts).model.domain_concepts


# ==================================================================================================
# 7. A CustomFunction written at an argument still gets its own frame
# ==================================================================================================


def local_function(body: object, name: str = "s:fn") -> dict:
    """
    ``CreateLocalVariable<CustomFunction>`` with a `CustomFunction` value.

    The value is written as a `Narrow`, not as a bare `FunctionComposition`: at a `CustomFunction` site a
    bare ``{"Return<T>": ...}`` commits to a Function evaluation and hard-fails, because `Return` declares
    no ``res`` -- so `Inst` is never reached and the composition shorthand never gets a turn.
    """
    return {"CreateLocalVariable<CustomFunction>": {"variableName": name, "init": {"CustomFunction": body}}}


class TestACustomFunctionAtAnArgumentGetsItsOwnFrame:
    """
    A `CustomFunction` value sitting at a Function *argument* is still a `CustomFunction`, so the frame
    rule of section 2 applies to it unchanged. It reaches its site through ``Optional<CustomFunction>``
    rather than a bare `CustomFunction` leaf, which is the only difference from the cases above.
    """

    def test_a_trivial_body_is_accepted(self):
        context = check(probe(in_procedure(sequence(local_function({"Return<Integer>": {"what": 1}})))))
        assert expression_of(context).is_valid

    def test_the_body_sees_the_globals(self):
        context = check(probe(in_procedure(sequence(local_function({"Return<Integer>": {"what": "one"}})))))
        assert expression_of(context).is_valid

    def test_the_body_does_not_see_an_enclosing_local(self):
        error = rejection(
            probe(in_procedure(sequence(create("s:x"), local_function({"Return<Boolean>": {"what": "x"}}))))
        )
        assert '"x" is not a variable of this ' in messages(error), messages(error)[:300]

    def test_the_body_sees_the_global_even_where_a_local_shadows_it(self):
        """
        ``one`` is a global `Integer`; an enclosing local makes it a `Boolean`. Inside the frame only the
        global exists, so ``one`` must still be usable as an `Integer`.
        """
        context = check(
            probe(
                in_procedure(
                    sequence(
                        create("s:one"),
                        local_function({"Return<Integer>": {"what": integer_site("one")}}),
                    )
                )
            )
        )
        assert expression_of(context).is_valid

    def test_the_body_does_not_see_a_sub_scope_variable(self):
        error = rejection(
            probe(in_procedure(iterate([sequence(local_function({"Return<Integer>": {"what": "value"}}))])))
        )
        assert '"value" is not a variable of this ' in messages(error), messages(error)[:300]


# ==================================================================================================
# 8. A sub-scope variable and a local of the same name
# ==================================================================================================


class TestASubScopeVariableAndALocalOfTheSameName:
    """
    They do not conflict: a sub-scope is a scope, so its variable shadows an outer one of the same name
    rather than clashing with it. The types are chosen to differ, so which one is in scope is observable.
    """

    def test_a_local_of_the_same_name_does_not_clash(self):
        context = check(probe(in_procedure(sequence(create("s:value", init=0, template="Integer"), iterate([])))))
        assert expression_of(context).is_valid

    def test_the_sub_scope_variable_shadows_the_local_inside_the_argument(self):
        """The local is a `Boolean`; inside ``what`` the name must be the sub-scope `Integer`."""
        context = check(
            probe(
                in_procedure(
                    sequence(
                        create("s:value"),
                        iterate([{"Return<Integer>": {"what": integer_site("value")}}]),
                    )
                )
            )
        )
        assert expression_of(context).is_valid

    def test_the_local_is_the_one_in_scope_outside_the_argument(self):
        """After the iteration the `Boolean` local is back, so an `Integer` use must fail."""
        error = rejection(
            probe(
                in_procedure(
                    sequence(
                        create("s:value"),
                        iterate([]),
                        {"Return<Integer>": {"what": integer_site("value")}},
                    )
                )
            )
        )
        assert "not a subtype" in messages(error), messages(error)[:300]

    def test_a_local_of_that_name_may_be_created_inside_the_argument(self):
        context = check(probe(in_procedure(iterate([sequence(create("s:value", init=0, template="Integer"))]))))
        assert expression_of(context).is_valid


# ==================================================================================================
# 9. Shadowing a CustomFunction's own interface argument
# ==================================================================================================


ARGUMENT_N = {"n": ["s:Integer", "Get"]}
"""One `Integer` argument, so a `Boolean` local of the same name is distinguishable from it."""


class TestShadowingACustomFunctionArgument:
    """
    Whether a procedure may declare a local named after one of its own interface arguments.

    The language already refuses ``create n; create n`` -- two locals of one name in a frame. It does
    *not* refuse ``argument n; create n``, and the difference is not a decision anyone took: the interface
    frame ends up below the frame `CreateLocalVariable` writes into, so the duplicate check never sees it.

    These tests describe today's behavior rather than endorse it, so that changing it is visible. What
    they pin either way is the *shape* of the collision: the argument is in scope before the local exists
    and hidden after it, with no block boundary marking the switch.
    """

    def test_the_argument_is_in_scope_before_the_local_is_created(self):
        context = check(
            probe(
                in_procedure(
                    sequence({"Return<Integer>": {"what": integer_site("n")}}, create("s:n")),
                    interface=ARGUMENT_N,
                )
            )
        )
        assert expression_of(context).is_valid

    def test_a_local_named_after_an_argument_is_accepted_today(self):
        """The open question. `create n; create n` is refused; this is the same collision and is not."""
        context = check(
            probe(
                in_procedure(
                    sequence(create("s:n", init=1, template="Integer"), {"Return<Integer>": {"what": "n"}}),
                    interface=ARGUMENT_N,
                )
            )
        )
        assert expression_of(context).is_valid

    def test_the_local_hides_the_argument_for_the_rest_of_the_procedure(self):
        """
        After a `Boolean` local ``n``, the `Integer` argument is unreachable -- so ``n`` means one thing
        above the ``create`` and another below it, in a flat sequence with nothing marking the change.
        """
        error = rejection(
            probe(
                in_procedure(
                    sequence(create("s:n"), {"Return<Integer>": {"what": integer_site("n")}}),
                    interface=ARGUMENT_N,
                )
            )
        )
        assert_resolved_as_variable(error, "n", "Boolean")

    def test_a_sub_scope_variable_may_also_take_an_arguments_name(self):
        """
        This one has a real block boundary -- it is scoped to a single argument -- so it has a better
        claim to being allowed than the flat `CreateLocalVariable` above.
        """
        context = check(
            probe(
                in_procedure(
                    {
                        "IterateNamed<Integer>": {
                            "list": [1, 2],
                            "varName": "s:n",
                            "what": [{"Return<Integer>": {"what": integer_site("n")}}],
                        }
                    },
                    interface=ARGUMENT_N,
                )
            )
        )
        assert expression_of(context).is_valid

    def test_but_two_locals_of_one_name_are_refused(self):
        """The contrast that makes the case above look accidental rather than chosen."""
        error = rejection(probe(in_procedure(sequence(create("s:n"), create("s:n", init=False)))))
        assert "already contains these variables" in messages(error), messages(error)[:300]

    def test_inside_a_custom_function_procedure_defining_create_local_variable_with_a_variable_from_its_interface_fails(
        self,
    ):
        """
        Test that a CreateLocalVariable as the only Function inside a CustomFunction's procedure
         whose variableName is a name of the CustomFunction's interface raises an error
         and one inside a sub-scope argument does not.
        """
        error = rejection(probe(in_procedure(create("s:n"), interface={"n": "s:Integer"})))
        assert "already contains these variables: ['n']" in messages(error), messages(error)[:300]
        context = check(probe(in_procedure(sequence(create("s:n")), interface={"n": "s:Integer"})))
        assert expression_of(context).is_valid


class TestWhereADuplicateVariableIsReported:
    def test_it_points_at_the_argument_that_names_the_variable(self):
        """
        The per-variable cause should name a path that tells the author what is wrong.
        Currently, the duplicate variable inserts ``"addNewVariablesInExistingScope"``,
        a key of the *Function's declaration*, to know why this error occurred.
        """
        error = rejection(
            probe(in_procedure(sequence(create(), create(init=False), {"Return<Boolean>": {"what": "x"}})))
        )
        sites = _locations(error, "already contains the variable")
        assert any(
            site.endswith('"CreateLocalVariable<Boolean>": "addNewVariablesInExistingScope": "variableName"')
            for site in sites
        ), sites
