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
Integration tests: grounding a Function's default *argument* expressions at a ground call site (stage 2.5).

A Function's ``_defaultArgumentValues`` are written in the Function's own template context and parsed once,
with ``T`` bound to nothing -- so an argument of type ``T`` defaulting to a String was accepted and never
looked at again. Only an application decides whether that default can hold, and only a call site produces
an application. `_ground_unsupplied_argument_defaults` reparses each *unsupplied* argument's declared
default there, under that site's template arguments.

Three things about that reparse are load-bearing and are each pinned below:

* it is **transient** -- the result is dropped, never written into `FunctionEvaluation.arguments`, because
  the acyclicity check reads ``supplied_arguments`` off that dict and materialising defaults there would
  make every argument look supplied;
* it happens **per application**, so the same declared default may be fine under one and rejected under
  another, and a call site that *supplies* the argument never grounds it at all;
* it runs in the **Function's** scope, not the call site's -- the Function's template context, the
  Function's substitution (replacing the caller's, not merged with it), and the Function's arguments back
  in variable scope, since a default may name a sibling.

See ``documentation/TODO_DEFAULT_EXPANSION_CYCLES.md`` §6.
"""

from __future__ import annotations

import re

import pytest

from concept_hierarchy.data.expressions.subexpressions import FunctionEvaluation
from concept_hierarchy.definitions.concept_hierarchy import ConceptHierarchyDefinition
from concept_hierarchy.errors import CHSemanticError, ConceptHierarchyError
from tests.integration.test_expression_parsing import build_hierarchy, check_concepts, check_hierarchy
from tests.integration.test_schema_substitution import default_site_expression, obj, uses_feval, vd

DEPTH_KEY = ConceptHierarchyDefinition.metadata_expansion_depth_limit_for_default_instantiation_expressions


def grounding_failed(argument: str, application: str) -> str:
    """
    The message grounding produces for one rejected default.

    Every negative test below matches on it rather than on `ConceptHierarchyError` alone: these hierarchies
    have a Function whose default is wrong *and* a call site that needs it, and an error raised anywhere
    else -- a missing argument, an unresolvable name, a dependency cycle -- would satisfy a bare
    ``raises`` while proving nothing about this stage.
    """
    return re.escape(f'the default of the unsupplied argument "{argument}" does not hold for {application}')


# --------------------------------------------------------------------------------------------------
# Hierarchy fragments
# --------------------------------------------------------------------------------------------------


def function(
    name: str,
    interface: dict,
    defaults: dict | None = None,
    template: object = None,
    parents: tuple = ("FunctionReturning",),
) -> dict:
    """A Function concept; ``template`` defaults to the single ``Numeric`` variable ``T``."""
    if template is None:
        template = {"order": ["T"], "T": "Numeric", "substitution": {"FunctionReturning:T": "T"}}
    full_interface = dict(interface)
    if defaults is not None:
        full_interface["_defaultArgumentValues"] = defaults
    data: dict = {"interface": full_interface}
    if template is not False:
        data["templateContext"] = template
    return {name: {"directParents": list(parents), "data": data}}


def add_like(name: str, defaults: dict, arg1_type: str = "T", arg2_type: str = "T") -> dict:
    """``name<T: Numeric>(arg1: arg1_type, arg2: arg2_type) -> T`` with the given declared defaults."""
    return function(name, {"arg1": [arg1_type], "arg2": [arg2_type], "res": "T"}, defaults)


ANY_T = {"order": ["T"], "T": "ValueDomain", "substitution": {"FunctionReturning:T": "T"}}
"""An unconstrained template variable, for the Functions that must take a `String` or an `Integer`."""

BOX = {
    "Box": {
        "directParents": ["ValueDomain"],
        "data": {
            "templateContext": ["P"],
            "instantiation": {"type": "object", "properties": {"b": {"type": "P"}}, "required": ["b"]},
        },
    }
}


def templated_site(applied_function: str, arguments: dict) -> dict:
    """A *templated* ValueDomain ``H<T>`` whose only site defaults into a Function evaluation."""
    return vd(
        "H",
        obj({"h": {"type": "T", "default": {applied_function: arguments}}}),
        {"order": ["T"], "T": "Numeric"},
    )


def feval_site(applied_function: str, arguments: dict, site_type: str) -> dict:
    """
    `uses_feval` with the site's own type spelled out.

    Its `Integer` site is right for the `Numeric`-constrained Functions above, but an evaluation whose
    result does not fit the site is rejected on the *return type*, long before any argument is grounded --
    which would make a test about defaults pass without ever reaching one.
    """
    return vd("Site", obj({"p": {"type": site_type, "default": {applied_function: arguments}}}))


def uses(applied: str, name: str = "Uses", prop: str = "u") -> dict:
    """A ground ValueDomain that instantiates ``applied``, forcing its schema to be built."""
    return vd(name, obj({prop: {"type": applied, "default": {}}}))


# ==================================================================================================
# The defect this stage closes
# ==================================================================================================


class TestATemplateDependentDefaultIsDecidedByTheApplication:
    def test_a_default_that_holds_for_the_application_is_accepted(self):
        check_concepts({**add_like("Add2", {"arg2": 3}), **uses_feval("Add2<Integer>", {"arg1": 1})})

    def test_a_default_that_can_not_hold_for_the_application_is_rejected(self):
        """``arg2`` defaults to a String while ``T`` is ``Integer`` -- invisible until the application."""
        with pytest.raises(ConceptHierarchyError, match=grounding_failed("arg2", "BadAdd<Integer>")):
            check_concepts(
                {**add_like("BadAdd", {"arg2": "s:not-a-number"}), **uses_feval("BadAdd<Integer>", {"arg1": 1})}
            )

    def test_the_verdict_follows_the_application_and_not_the_declaration(self):
        """
        The same declared default, the same Function, two applications: `3` is an `Integer` and is not a
        `String`. Nothing about the *declaration* separates these two hierarchies -- only the application
        does, which is the whole reason this check cannot live where the default is written.
        """
        good = {**add_like("Pick", {"arg2": 3}), **uses_feval("Pick<Integer>", {"arg1": 1})}
        check_concepts(good)

        stringly = function(
            "Pick2",
            {"arg1": ["T"], "arg2": ["T"], "res": "T"},
            {"arg2": 3},
            template={"order": ["T"], "T": "ValueDomain", "substitution": {"FunctionReturning:T": "T"}},
        )
        with pytest.raises(ConceptHierarchyError, match=grounding_failed("arg2", "Pick2<String>")):
            check_concepts({**stringly, **feval_site("Pick2<String>", {"arg1": "s:x"}, "String")})

    def test_the_error_names_the_argument_and_the_application(self):
        with pytest.raises(ConceptHierarchyError) as caught:
            check_concepts(
                {**add_like("BadAdd", {"arg2": "s:not-a-number"}), **uses_feval("BadAdd<Integer>", {"arg1": 1})}
            )
        message = str(caught.value)
        assert "arg2" in message, message
        assert "BadAdd<Integer>" in message, message

    def test_a_ground_function_is_still_checked_where_it_is_declared(self):
        """
        The control that locates the gap: with no template variable in play the default is decided at the
        declaration, and the call site adds nothing. Only template-dependence defers it.
        """
        shout = function(
            "Shout",
            {"what": ["Integer"], "res": "Integer"},
            {"what": "s:not-a-number"},
            template={"order": [], "substitution": {"FunctionReturning:T": "Integer"}},
        )
        with pytest.raises(CHSemanticError, match="Invalid expression"):
            check_concepts(shout)


class TestOnlyUnsuppliedDefaultsAreGrounded:
    def test_supplying_the_argument_means_its_default_is_never_grounded(self):
        """
        A default that could never hold for this application is not an error while nothing uses it. The
        call site writes ``arg2`` itself, so the declared default is dead text here.
        """
        check_concepts(
            {
                **add_like("BadAdd", {"arg2": "s:not-a-number"}),
                **uses_feval("BadAdd<Integer>", {"arg1": 1, "arg2": 2}),
            }
        )

    def test_the_same_default_is_rejected_at_the_site_that_leaves_it_unsupplied(self):
        """The pair of the test above: identical hierarchy but for the one supplied argument."""
        with pytest.raises(ConceptHierarchyError, match=grounding_failed("arg2", "BadAdd<Integer>")):
            check_concepts(
                {**add_like("BadAdd", {"arg2": "s:not-a-number"}), **uses_feval("BadAdd<Integer>", {"arg1": 1})}
            )

    def test_an_argument_with_no_default_is_required_and_never_reaches_grounding(self):
        """
        Having a declared default is exactly what makes an argument optional, so "unsupplied and
        undefaulted" is already an error by the time grounding would look at it. Grounding therefore has
        nothing to say about a missing default, and must not invent something to say.
        """
        hierarchy = {
            **function("Maybe", {"arg1": ["T"], "arg2": ["T"], "res": "T"}, {"arg1": 1}),
            **uses_feval("Maybe<Integer>", {}),
        }
        with pytest.raises(CHSemanticError, match="missing from the Function evaluation interface"):
            check_concepts(hierarchy)

    def test_the_defaulted_argument_alone_may_be_left_unsupplied(self):
        context = check_concepts(
            {
                **function("Maybe", {"arg1": ["T"], "arg2": ["T"], "res": "T"}, {"arg1": 1}),
                **uses_feval("Maybe<Integer>", {"arg2": 2}),
            }
        )
        feval = default_site_expression(context, "Site", "p").value
        assert isinstance(feval, FunctionEvaluation)
        assert sorted(feval.arguments) == ["arg2"]


class TestTheGroundedDefaultIsNotKept:
    def test_grounding_leaves_the_arguments_untouched(self):
        context = check_concepts({**add_like("Add2", {"arg2": 3}), **uses_feval("Add2<Integer>", {"arg1": 1})})
        feval = default_site_expression(context, "Site", "p").value
        assert isinstance(feval, FunctionEvaluation)
        assert sorted(feval.arguments) == ["arg1"]

    def test_the_declared_default_is_not_rewritten_by_grounding(self):
        """
        `evaluation_argument_default_value_expressions` is the *declaration*, shared by every call site; a per-site
        grounding must not write back into it, or the second application would see the first one's result.
        """
        context = check_concepts({**add_like("Add2", {"arg2": 3}), **uses_feval("Add2<Integer>", {"arg1": 1})})
        declared = context.model.functions["Add2"].evaluation_argument_default_value_expressions["arg2"]
        assert declared.is_value_template_dependent, "the declaration stays undecided, as it must"


# ==================================================================================================
# What the default may name, and in whose scope
# ==================================================================================================


class TestADefaultMayNameTheFunctionsTemplateVariables:
    def test_a_template_variable_in_a_nested_function_evaluation_key(self):
        """``{"Add<T>": ...}`` inside the default: ``T`` is the *Function's*, ground only at the call site."""
        hierarchy = {
            **add_like("Add", {}),
            **add_like("Outer", {"arg2": {"Add<T>": {"arg1": 1, "arg2": 2}}}),
            **uses_feval("Outer<Integer>", {"arg1": 1}),
        }
        check_concepts(hierarchy)

    def test_a_template_variable_that_grounds_to_something_the_nested_key_forbids(self):
        """
        The same default under an application ``Add`` cannot take: ``Add<T: Numeric>`` and ``T := String``.
        Rejecting this is only possible once ``T`` is known.

        A violated *template constraint* is raised by the substitution rather than returned as a failed
        alternative, so this one does not carry the grounding message; what identifies it as grounding is
        the location, which is the default of ``Outer<String>``'s ``arg2``.
        """
        outer = function(
            "Outer",
            {"arg1": ["T"], "arg2": ["T"], "res": "T"},
            {"arg2": {"Add<T>": {"arg1": 1, "arg2": 2}}},
            template={"order": ["T"], "T": "ValueDomain", "substitution": {"FunctionReturning:T": "T"}},
        )
        with pytest.raises(CHSemanticError, match=re.escape('"Outer<String>": "arg2"')) as caught:
            check_concepts({**add_like("Add", {}), **outer, **feval_site("Outer<String>", {"arg1": "s:x"}, "String")})
        assert "does not satisfy the constraint Numeric" in str(caught.value), str(caught.value)

    def test_a_template_variable_nested_inside_a_value(self):
        """``T`` buried in a `Box<T>` the default instantiates, rather than being the argument's own type."""
        hierarchy = {
            **BOX,
            **function("Wrap", {"arg1": ["T"], "arg2": ["Box<T>"], "res": "T"}, {"arg2": {"b": 1}}),
            **uses_feval("Wrap<Integer>", {"arg1": 1}),
        }
        check_concepts(hierarchy)

    def test_the_nested_value_is_checked_against_the_substituted_type(self):
        """The same shape with a value no ``Box<Integer>`` admits."""
        hierarchy = {
            **BOX,
            **function("Wrap", {"arg1": ["T"], "arg2": ["Box<T>"], "res": "T"}, {"arg2": {"b": "s:x"}}),
            **uses_feval("Wrap<Integer>", {"arg1": 1}),
        }
        with pytest.raises(ConceptHierarchyError, match=grounding_failed("arg2", "Wrap<Integer>")):
            check_concepts(hierarchy)


class TestTheDefaultIsGroundedInTheFunctionsOwnScope:
    def test_a_default_may_name_a_sibling_argument(self):
        """
        ``"arg1"`` resolves where the default is *written*, not where it is used: the call site's
        variables are a different set entirely, so grounding has to put the Function's arguments back.
        """
        check_concepts(
            {
                **add_like("Copy", {"arg2": "arg1"}),
                **uses_feval("Copy<Integer>", {"arg1": 1}),
            }
        )

    def test_a_default_may_name_a_sibling_the_call_site_supplied(self):
        """
        Every argument goes into scope, not only the unsupplied ones -- and what ``arg1`` means there is its
        declared type, not the `Integer` literal this site happened to write.
        """
        hierarchy = {
            **function("Copy3", {"arg1": ["T"], "arg2": ["T"], "arg3": ["T"], "res": "T"}, {"arg3": "arg1"}),
            **uses_feval("Copy3<Integer>", {"arg1": 1, "arg2": 2}),
        }
        check_concepts(hierarchy)

    def test_a_sibling_of_an_incompatible_type_is_rejected_at_the_application(self):
        """``arg1: String`` cannot stand in for ``arg2: T`` once ``T`` is ``Integer``."""
        with pytest.raises(ConceptHierarchyError, match=grounding_failed("arg2", "Mismatch<Integer>")):
            check_concepts(
                {
                    **add_like("Mismatch", {"arg2": "arg1"}, arg1_type="String"),
                    **uses_feval("Mismatch<Integer>", {"arg1": "s:x"}),
                }
            )

    def test_a_default_naming_nothing_in_scope_survives_until_it_is_grounded(self):
        """
        ``"nowhere"`` is no variable, no concept and no template variable. With ``T`` unbound the parser
        cannot rule out that some application makes it a valid value, so it defers -- and a hierarchy that
        never applies ``Ghost`` is genuinely fine.
        """
        check_concepts({**add_like("Ghost", {"arg2": "nowhere"})})

    def test_and_is_then_rejected_at_the_application(self):
        with pytest.raises(ConceptHierarchyError, match=grounding_failed("arg2", "Ghost<Integer>")):
            check_concepts({**add_like("Ghost", {"arg2": "nowhere"}), **uses_feval("Ghost<Integer>", {"arg1": 1})})

    def test_an_enclosing_functions_argument_can_not_capture_a_string_default(self):
        """
        The scope is **replaced**, not extended, and this is what turns on it.

        `Inner`'s default for `x: T` is the bare string ``"leak"``, which under `Inner<Integer>` is not a
        valid value and must be rejected. Grounding it happens *inside* the grounding of `Outer`'s
        default -- and `Outer` has an argument called `leak`. The parser classifies a bare string as a
        variable before it considers anything else, so leaving `Outer`'s frame on the stack makes
        ``"leak"`` that argument, of exactly the right type, and the hierarchy checks. Only the standalone
        call below says what the answer should be.
        """
        inner = function("Inner", {"x": ["T"], "res": "T"}, {"x": "leak"}, template=ANY_T)
        outer = function(
            "Outer", {"leak": ["Integer"], "y": ["T"], "res": "T"}, {"y": {"Inner<T>": {}}}, template=ANY_T
        )
        with pytest.raises(ConceptHierarchyError, match=grounding_failed("y", "Outer<Integer>")):
            check_concepts({**inner, **outer, **feval_site("Outer<Integer>", {"leak": 1}, "Integer")})

    def test_the_same_default_alone_is_rejected_too(self):
        """The control: without the enclosing Function there is no name to capture it, and it fails."""
        inner = function("Inner", {"x": ["T"], "res": "T"}, {"x": "leak"}, template=ANY_T)
        with pytest.raises(ConceptHierarchyError, match=grounding_failed("x", "Inner<Integer>")):
            check_concepts({**inner, **feval_site("Inner<Integer>", {}, "Integer")})

    def test_global_variables_stay_in_scope(self):
        """
        Replacing the scope keeps frame 0. Globals are visible wherever an expression is written, including
        inside a Function's declared default, so dropping them would reject definitions that are fine.
        """
        hierarchy = build_hierarchy(
            {**add_like("UsesGlobal", {"arg2": "g"}), **uses_feval("UsesGlobal<Integer>", {"arg1": 1})},
            instances={"g": {"Integer": 7}},
        )
        check_hierarchy(hierarchy)

    def test_the_call_sites_variables_do_not_leak_into_the_default(self):
        """
        The grounding frame is pushed on top, so a call-site variable of the same name is shadowed by the
        Function's argument. ``arg1`` here means ``Mismatch``'s `String` argument, never the site's
        `Integer` instance -- which is why this is still rejected.
        """
        hierarchy = build_hierarchy(
            {
                **add_like("Mismatch", {"arg2": "arg1"}, arg1_type="String"),
                **uses_feval("Mismatch<Integer>", {"arg1": "s:x"}),
            },
            instances={"arg1": {"Integer": 7}},
        )
        with pytest.raises(ConceptHierarchyError, match=grounding_failed("arg2", "Mismatch<Integer>")):
            check_hierarchy(hierarchy)


SUB_ADD = {
    "SubAdd": {
        "directParents": ["BaseAdd"],
        "data": {
            "templateContext": {"order": ["T"], "T": "Numeric", "substitution": {"BaseAdd:T": "T"}},
            "interface": {},
        },
    }
}


class TestInheritedDefaults:
    """
    A Function's model folds in its parents' defaults, and an inherited default makes its argument optional
    exactly as a declared one does -- `get_required_function_arguments` subtracts the merged
    `FunctionData.evaluation_default_arguments`, not the definition's own `_defaultArgumentValues`.

    So a child's call site grounds text the child never declared, and the application that decides it is the
    child's.
    """

    def test_an_inherited_default_makes_the_argument_optional(self):
        parent = add_like("BaseAdd", {"arg2": 3})
        check_concepts({**parent, **SUB_ADD, **uses_feval("SubAdd<Integer>", {"arg1": 1})})

    def test_the_inherited_default_is_recorded_on_the_child(self):
        context = check_concepts({**add_like("BaseAdd", {"arg2": 3}), **SUB_ADD})
        inherited = context.model.functions["SubAdd"].evaluation_argument_default_value_expressions
        assert "arg2" in inherited

    def test_an_inherited_default_is_grounded_at_the_childs_call_site(self):
        """
        The parent declares it, the child inherits it, and `SubAdd<Integer>` is what decides it. Nothing
        about the declaration changed between this and the test above -- only the application.
        """
        parent = add_like("BaseAdd", {"arg2": "s:not-a-number"})
        with pytest.raises(ConceptHierarchyError, match=grounding_failed("arg2", "SubAdd<Integer>")):
            check_concepts({**parent, **SUB_ADD, **uses_feval("SubAdd<Integer>", {"arg1": 1})})

    def test_supplying_the_inherited_argument_leaves_it_ungrounded(self):
        """As for a declared default: what the site writes is never grounded."""
        parent = add_like("BaseAdd", {"arg2": "s:not-a-number"})
        check_concepts({**parent, **SUB_ADD, **uses_feval("SubAdd<Integer>", {"arg1": 1, "arg2": 2})})


# ==================================================================================================
# Termination
# ==================================================================================================


class TestADefaultThatGroundsItself:
    def test_a_self_referential_default_is_reported_rather_than_hung(self):
        """
        ``x`` defaults to an evaluation of the very Function it belongs to, leaving ``x`` unsupplied again:
        grounding ``Loop<Integer>.x`` requires grounding ``Loop<Integer>.x``. The application repeats
        exactly, so the on-path guard sees it -- no growth, no depth limit, just a cycle.
        """
        hierarchy = {
            **function("Loop", {"x": ["T"], "res": "T"}, {"x": {"Loop<T>": {}}}),
            **uses_feval("Loop<Integer>", {}),
        }
        with pytest.raises(ConceptHierarchyError) as caught:
            check_concepts(hierarchy)
        assert "grounding it again" in str(caught.value), str(caught.value)

    def test_a_mutually_recursive_pair_of_defaults_is_reported(self):
        """Two Functions whose defaults evaluate each other; the guard is on the path, not on one site."""
        hierarchy = {
            **function("Ping", {"x": ["T"], "res": "T"}, {"x": {"Pong<T>": {}}}),
            **function("Pong", {"x": ["T"], "res": "T"}, {"x": {"Ping<T>": {}}}),
            **uses_feval("Ping<Integer>", {}),
        }
        with pytest.raises(ConceptHierarchyError, match="grounding it again"):
            check_concepts(hierarchy)

    def test_a_self_evaluation_that_supplies_the_argument_terminates(self):
        """
        The same shape, but the inner evaluation writes ``x`` itself, so nothing is left to ground and the
        recursion stops after one step. The guard must not fire on this.
        """
        hierarchy = {
            **function("Once", {"x": ["T"], "res": "T"}, {"x": {"Once<T>": {"x": 1}}}),
            **uses_feval("Once<Integer>", {}),
        }
        check_concepts(hierarchy)


class TestADefaultThatKeepsGrowingItsApplication:
    """
    The runaway the on-path guard cannot catch: each level is a *different* application, so nothing ever
    repeats and only the depth limit ends it. This is the reason grounding counts against
    ``maxExpansionDepthForDefaultInstantiationExpressions`` at all -- every level of this recursion is, by
    construction, a newly generated application.
    """

    GROWING = {
        **BOX,
        **{
            "Grow": {
                "directParents": ["FunctionReturning"],
                "data": {
                    # The result is `Integer` at every application, not `T`: an argument of type `T`
                    # defaulting to a `Grow<Box<T>>` would be rejected at the *first* step for returning a
                    # `Box<T>`, and the recursion this class is about would never get going.
                    "templateContext": {"order": ["T"], "substitution": {"FunctionReturning:T": "Integer"}},
                    "interface": {
                        "x": ["ValueDomain"],
                        "res": "Integer",
                        "_defaultArgumentValues": {"x": {"Grow<Box<T>>": {}}},
                    },
                },
            }
        },
    }

    def _run(self, limit: int) -> str:
        hierarchy = build_hierarchy({**self.GROWING, **uses_feval("Grow<Integer>", {})})
        hierarchy["metadata"] = {DEPTH_KEY: limit}
        with pytest.raises(CHSemanticError) as caught:
            check_hierarchy(hierarchy)
        return str(caught.value)

    @pytest.mark.parametrize("limit", [2, 8])
    def test_the_configured_limit_is_what_stops_it(self, limit):
        """
        The message has to be the *limit's*, not the interpreter's. Both mention `DEPTH_KEY` -- the
        stack-exhaustion fallback names it to say the limit was never reached -- so matching on the key
        alone cannot tell "the limit worked" from "the limit was never counted against and Python gave
        out first", which is exactly what happens if grounding does not advance the depth.
        """
        message = self._run(limit)
        assert f"more than {limit} levels deep" in message, message
        assert "Ran out of stack" not in message, message

    def test_a_lower_limit_stops_it_sooner(self):
        """Nothing about the hierarchy changed; the limit alone decides how far it gets."""
        assert "more than 2 levels deep" in self._run(2)
        assert "more than 8 levels deep" in self._run(8)


class TestTheAcyclicityCheckStillRunsFirst:
    def test_two_defaults_naming_each_other_are_reported_as_a_dependency_cycle(self):
        """
        ``arg1`` defaults to ``arg2`` and ``arg2`` to ``arg1``, neither supplied. This is the *existing*
        check, and it runs before grounding -- grounding these would only recurse through variables that
        can never be given a value.
        """
        hierarchy = {
            **add_like("Swap", {"arg1": "arg2", "arg2": "arg1"}),
            **uses_feval("Swap<Integer>", {}),
        }
        with pytest.raises(CHSemanticError, match="not acyclic"):
            check_concepts(hierarchy)

    def test_supplying_one_of_them_breaks_the_cycle_and_the_rest_is_grounded(self):
        hierarchy = {
            **add_like("Swap", {"arg1": "arg2", "arg2": "arg1"}),
            **uses_feval("Swap<Integer>", {"arg1": 1}),
        }
        check_concepts(hierarchy)


# ==================================================================================================
# Where grounding does and does not happen
# ==================================================================================================


class TestOnlyGroundCallSitesGround:
    def test_a_call_site_inside_a_template_defers_until_the_template_is_applied(self):
        """
        ``H<T>``'s site writes ``BadAdd<T>``; that is not an application yet, so nothing is decided there.
        Building ``H<Integer>`` (stage 1) makes it one, and *that* is where the bad default is caught.
        """
        hierarchy = {
            **add_like("BadAdd", {"arg2": "s:not-a-number"}),
            **templated_site("BadAdd<T>", {"arg1": 1}),
            **uses("H<Integer>"),
        }
        with pytest.raises(ConceptHierarchyError, match=grounding_failed("arg2", "BadAdd<Integer>")):
            check_concepts(hierarchy)

    def test_the_same_template_is_fine_while_it_is_never_applied(self):
        """
        Without the ground application there is no verdict to reach -- deferring is right, not lenient.
        Compare with the test above: the only difference is the ValueDomain that applies ``H``.
        """
        check_concepts({**add_like("BadAdd", {"arg2": "s:not-a-number"}), **templated_site("BadAdd<T>", {"arg1": 1})})

    def test_an_application_that_makes_the_default_hold_is_accepted(self):
        """``H<Integer>`` again, but with a default that fits -- the substitution is what decides."""
        hierarchy = {
            **add_like("Add2", {"arg2": 3}),
            **templated_site("Add2<T>", {"arg1": 1}),
            **uses("H<Integer>"),
        }
        check_concepts(hierarchy)


class TestSeveralDefaultsAtOnce:
    def test_every_unsupplied_default_is_grounded_not_just_the_first(self):
        """``arg1`` is fine and ``arg2`` is not; stopping at the first would let this through."""
        hierarchy = {
            **add_like("Half", {"arg1": 1, "arg2": "s:not-a-number"}),
            **uses_feval("Half<Integer>", {}),
        }
        with pytest.raises(ConceptHierarchyError, match=grounding_failed("arg2", "Half<Integer>")):
            check_concepts(hierarchy)

    def test_all_valid_defaults_are_accepted_together(self):
        check_concepts({**add_like("Both", {"arg1": 1, "arg2": 2}), **uses_feval("Both<Integer>", {})})


# ==================================================================================================
# What grounding records, and what it therefore does not repeat
# ==================================================================================================


def grounded_defaults(context) -> dict:
    """
    The validator's per-application cache of grounded defaults, keyed ``(application, argument)``.

    Reached through the private attribute on purpose: the cache is not observable any other way -- the
    grounded expression is deliberately dropped rather than stored on the `FunctionEvaluation` -- and both
    what it holds and what it *declines* to hold are behaviour worth pinning.
    """
    return dict(context.expression_parser_validator._grounded_function_defaults)


class TestADefaultIsGroundedOncePerApplication:
    def test_two_call_sites_of_one_application_share_the_result(self):
        """
        The cache is keyed by the application, not by the call site, so the second evaluation of
        `Add2<Integer>` finds the first one's work. Keying it on the declared `Expression` instead could
        not do this: the declaration is one object shared by every application.
        """
        two_sites = vd(
            "Site",
            obj(
                {
                    "p": {"type": "Integer", "default": {"Add2<Integer>": {"arg1": 1}}},
                    "q": {"type": "Integer", "default": {"Add2<Integer>": {"arg1": 2}}},
                }
            ),
        )
        context = check_concepts({**add_like("Add2", {"arg2": 3}), **two_sites})
        assert sorted(grounded_defaults(context)) == [("Add2<Integer>", "arg2")]

    def test_two_applications_do_not_share_it(self):
        """The other half of the key: the same declared default, two applications, two verdicts."""
        two_sites = vd(
            "Site",
            obj(
                {
                    "p": {"type": "Integer", "default": {"Add2<Integer>": {"arg1": 1}}},
                    "q": {"type": "Number", "default": {"Add2<Number>": {"arg1": 1.5}}},
                }
            ),
        )
        context = check_concepts({**add_like("Add2", {"arg2": 3}), **two_sites})
        assert sorted(grounded_defaults(context)) == [("Add2<Integer>", "arg2"), ("Add2<Number>", "arg2")]

    def test_the_cached_entry_holds_the_grounded_expression(self):
        context = check_concepts({**add_like("Add2", {"arg2": 3}), **uses_feval("Add2<Integer>", {"arg1": 1})})
        entry = grounded_defaults(context)[("Add2<Integer>", "arg2")]
        assert not entry.in_progress
        assert entry.expression is not None and entry.expression.is_valid
        assert entry.expression.required_expression_type.full_name == "Integer"
        assert not entry.expression.is_value_template_dependent, "grounding is what decides it"


class TestAlreadyDecidedDefaultsAreNotReground:
    def test_a_ground_typed_default_is_not_grounded_again(self):
        """
        `arg1: Integer` defaulting to `1` was decided where it was declared: its type is ground, so
        substitution cannot change it, and it was parsed there under the same scope and the same type
        checks. Re-parsing it at every call site would only re-derive that verdict, so it is skipped --
        and the empty cache entry for it is how that is visible.
        """
        both = function("Mixed", {"arg1": ["Integer"], "arg2": ["T"], "res": "T"}, {"arg1": 1, "arg2": 2})
        context = check_concepts({**both, **uses_feval("Mixed<Integer>", {})})
        assert sorted(grounded_defaults(context)) == [("Mixed<Integer>", "arg2")]

    def test_a_ground_typed_default_naming_a_template_variable_is_grounded(self):
        """
        The type being ground is not enough on its own: `arg1: Integer` here defaults to an evaluation
        whose *key* names `T`, so the application still decides it.
        """
        both = function(
            "Mixed2",
            {"arg1": ["Integer"], "arg2": ["T"], "res": "T"},
            {"arg1": {"Add<T>": {"arg1": 1, "arg2": 2}}, "arg2": 2},
        )
        context = check_concepts({**add_like("Add", {}), **both, **uses_feval("Mixed2<Integer>", {})})
        assert sorted(grounded_defaults(context)) == [("Mixed2<Integer>", "arg1"), ("Mixed2<Integer>", "arg2")]

    def test_a_decided_default_at_a_template_dependent_typed_argument_is_still_ground(self):
        """
        The skip needs *both* halves of its condition, and this is the case that shows it.

        `a: Cell<T>` is a `TemplateDependentType`, and its default is a **ground** Function evaluation --
        `FunctionEvaluation.is_template_dependent` reads the evaluated Function's type and its arguments,
        never the type the expression is being checked *against*, so the default reports as decided while
        the one thing still open is exactly the check that matters: is `Cell<Integer>` a `Cell<T>`? Under
        `W<String>` it is not. Skipping on template-dependence alone would let `Cell<Integer>` be passed
        off as a `Cell<String>`.
        """
        cell = {
            "Cell": {
                "directParents": ["ValueDomain"],
                "data": {
                    "templateContext": ["P"],
                    "instantiation": {"type": "object", "properties": {"b": {"type": "P"}}, "required": ["b"]},
                },
            }
        }
        mk = function(
            "Mk",
            {"v": ["T"], "res": "Cell<T>"},
            template={"order": ["T"], "T": "ValueDomain", "substitution": {"FunctionReturning:T": "Cell<T>"}},
        )
        w = function(
            "W",
            {"a": ["Cell<T>"], "b": ["T"], "res": "T"},
            {"a": {"Mk<Integer>": {"v": 1}}},
            template=ANY_T,
        )
        with pytest.raises(ConceptHierarchyError, match=grounding_failed("a", "W<String>")):
            check_concepts({**cell, **mk, **w, **feval_site("W<String>", {"b": "s:x"}, "String")})

    def test_the_same_default_holds_for_the_application_it_fits(self):
        """The control: `Cell<Integer>` is a `Cell<T>` when `T` is `Integer`."""
        cell = {
            "Cell": {
                "directParents": ["ValueDomain"],
                "data": {
                    "templateContext": ["P"],
                    "instantiation": {"type": "object", "properties": {"b": {"type": "P"}}, "required": ["b"]},
                },
            }
        }
        mk = function(
            "Mk",
            {"v": ["T"], "res": "Cell<T>"},
            template={"order": ["T"], "T": "ValueDomain", "substitution": {"FunctionReturning:T": "Cell<T>"}},
        )
        w = function("W", {"a": ["Cell<T>"], "b": ["T"], "res": "T"}, {"a": {"Mk<Integer>": {"v": 1}}}, template=ANY_T)
        check_concepts({**cell, **mk, **w, **feval_site("W<Integer>", {"b": 1}, "Integer")})

    def test_a_ground_typed_default_is_still_checked_where_it_is_written(self):
        """Skipping it at the call site is only sound because the declaration already decided it."""
        bad = function("Mixed3", {"arg1": ["Integer"], "arg2": ["T"], "res": "T"}, {"arg1": "s:x", "arg2": 2})
        with pytest.raises(CHSemanticError, match="Invalid expression"):
            check_concepts({**bad, **uses_feval("Mixed3<Integer>", {})})


class TestTheSiblingDependenciesAreCompletedByGrounding:
    """
    `FunctionData.default_argument_dependencies` is collected at definition time from a parse with the
    template variables unbound, and that parse stops wherever the type stops being decidable. A sibling
    reference nested inside a *template-dependent* value therefore never becomes a `Variable` and never
    enters the set -- so the cycle it is part of is not seen. Grounding is where the rest of the tree
    finally exists, and the set it produces is unioned into the declared one before the check runs.
    """

    CELL = {
        "Cell": {
            "directParents": ["ValueDomain"],
            "data": {
                "templateContext": ["P"],
                "instantiation": {"type": "object", "properties": {"b": {"type": "P"}}, "required": ["b"]},
            },
        }
    }

    def _cycle_through(self, cell: str, wrapper: str) -> dict:
        """``arg1 -> arg2`` hidden inside a `Cell`, and ``arg2 -> arg1`` in plain sight."""
        return {
            **self.CELL,
            **function("Id2", {"v": [cell], "res": "T"}),
            **function(
                "F",
                {"arg1": [cell], "arg2": ["T"], "res": "T"},
                {"arg1": {"b": "arg2"}, "arg2": {wrapper: {"v": "arg1"}}},
            ),
            **uses_feval("F<Integer>", {}),
        }

    def test_the_definition_time_scan_alone_records_only_the_visible_edge(self):
        """What the incompleteness looks like, measured on the declaration rather than argued about."""
        context = check_concepts(
            {
                **self.CELL,
                **function("G", {"arg1": ["Cell<T>"], "arg2": ["T"], "res": "T"}, {"arg1": {"b": "arg2"}}),
            }
        )
        declared = context.model.functions["G"].default_argument_dependencies
        assert set(declared["arg1"]) == set(), "the nested `arg2` was never reached with `T` unbound"

    def test_the_ground_scan_finds_it(self):
        context = check_concepts(
            {
                **self.CELL,
                **function("G", {"arg1": ["Cell<T>"], "arg2": ["T"], "res": "T"}, {"arg1": {"b": "arg2"}}),
                **uses_feval("G<Integer>", {"arg2": 1}),
            }
        )
        entry = grounded_defaults(context)[("G<Integer>", "arg1")]
        assert set(entry.sibling_dependencies) == {"arg2"}

    def test_a_cycle_hidden_by_a_template_dependent_value_is_rejected(self):
        with pytest.raises(CHSemanticError, match="not acyclic"):
            check_concepts(self._cycle_through("Cell<T>", "Id2<T>"))

    def test_the_same_cycle_written_with_ground_types_is_rejected_as_before(self):
        """The control: with `Cell<Integer>` the definition-time scan already saw both edges."""
        with pytest.raises(CHSemanticError, match="not acyclic"):
            check_concepts(self._cycle_through("Cell<Integer>", "Id2<Integer>"))

    def test_a_non_cycle_through_the_same_shape_survives(self):
        """The completed graph must add the edges that exist, not edges in general."""
        hierarchy = {
            **self.CELL,
            **function("H2", {"arg1": ["Cell<T>"], "arg2": ["T"], "res": "T"}, {"arg1": {"b": "arg2"}, "arg2": 1}),
            **uses_feval("H2<Integer>", {}),
        }
        check_concepts(hierarchy)
