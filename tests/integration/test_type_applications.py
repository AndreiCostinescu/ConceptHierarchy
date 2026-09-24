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
Writing a type application: which argument forms a templated type accepts, and what it says when it does not.

The subject is the *type*, not the value: everything here is decided at declaration time, by writing a type
somewhere a type is expected. No instance is created, so nothing an instantiation schema does can mask a
wrong answer.

A template parameter is **plain** or **variadic**. A variadic one takes a *group*, and a group may be
written three ways:

* **bracketed** -- ``Target<[Integer, String]>``, the canonical form, always available;
* **forwarded by name** -- ``Target<G>`` where ``G`` is the enclosing type's own variadic parameter. A
  variadic parameter already denotes a group, so this hands the whole group over; it is not a member of a
  new one, which is why ``Target<[G]>`` (a group inside a group) means nothing and ``Target<[G...]>`` is
  how the members are spliced in;
* **the bare shorthand** -- ``Target<Integer, String>``, single types written with no brackets, collected
  into a group. This is the only form that can be ambiguous: with several groups, nothing in
  ``Integer`` says which one it joins, and that is what `variadicGroupIdentifiers` exists to settle.

``variadicGroupIdentifiers`` is therefore a **disambiguator for the bare shorthand and nothing else**.
Every case below is run both with and without it declared wherever the distinction could matter, because
a form that works only when an identifier happens to be declared is a bug, not a feature.

Each rejection asserts its *message*, not merely that something was raised. A type application has many
ways to be wrong and they are not interchangeable -- "you left a group out" and "which group does this
belong to" call for opposite fixes -- so a test that only checks for a raise cannot tell a good diagnosis
from a bad one, and every message here was at some point either unformatted, unfinished, or about the
wrong thing.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from concept_hierarchy.data.types.concept_hierarchy_types import InstantiatedType, InstantiatedVariadicGroup
from concept_hierarchy.errors import CHSemanticError, ConceptHierarchyError
from tests.integration.test_expression_parsing import build_hierarchy, check_hierarchy, error_messages

# --------------------------------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------------------------------


def template_context(
    order: list[str], identifiers: dict[str, str] | None = None, constraints: dict[str, str] | None = None
) -> dict:
    """
    A `templateContext` over ``order``; ``"A..."`` declares a variadic parameter.

    Every parameter is constrained to `ValueDomain` unless ``constraints`` says otherwise, which is what
    lets a case be about the *constraint* -- ``Not(ValueDomain)``, ``Literal:integer`` -- rather than about
    the shape of the argument list.
    """
    context: dict = {"order": order}
    for name in order:
        clean = name.removesuffix("...")
        context[clean] = (constraints or {}).get(clean, "ValueDomain")
    if identifiers is not None:
        context["variadicGroupIdentifiers"] = identifiers
    return context


VOCABULARY = {
    "Pet": {"directParents": ["Concept"], "data": {"properties": {"age": {"valueDomain": "Integer"}}}},
    "List": {"directParents": ["ValueDomain"], "data": {"templateContext": ["T"], "instantiation": True}},
}
"""A DomainConcept (something that is *not* a `ValueDomain`) and a templated type, for nested arguments."""


def written(target_order: list[str], application: str, outer_order: list[str] | None = None, **kwargs) -> dict:
    """
    ``Target`` declared over ``target_order``, and ``application`` written as a type inside ``Outer``.

    ``outer_order`` gives `Outer` template parameters of its own, which is what makes forwarding
    expressible at all: only a type that *has* a variadic parameter can hand one on.
    """
    target = {
        "Target": {
            "directParents": ["ValueDomain"],
            "data": {
                "templateContext": template_context(target_order, kwargs.get("identifiers"), kwargs.get("constraints")),
                "instantiation": True,
            },
        }
    }
    outer_data: dict = {"instantiation": {"type": "object", "properties": {"v": {"type": application}}}}
    if outer_order:
        outer_data["templateContext"] = template_context(
            outer_order, kwargs.get("outer_identifiers"), kwargs.get("outer_constraints")
        )
    return {**VOCABULARY, **target, "Outer": {"directParents": ["ValueDomain"], "data": outer_data}}


def accepts(*args, **kwargs) -> None:
    """The application is a well-formed type. Raises on a crash as loudly as on a rejection."""
    with contextlib.redirect_stdout(io.StringIO()):
        check_hierarchy(build_hierarchy(written(*args, **kwargs)))


def refuses(*args, **kwargs) -> str:
    """
    The application is refused *as a diagnosis*, and the diagnosis is returned for inspection.

    A non-`ConceptHierarchyError` propagates rather than being caught: an `AssertionError` or a `KeyError`
    out of the type machinery is a crash, not a rejection, and the difference is the whole point of
    several of the cases below.
    """
    with pytest.raises(ConceptHierarchyError) as excinfo:
        accepts(*args, **kwargs)
    return error_messages(excinfo.value)


ONE_GROUP = ["A..."]
TWO_GROUPS = ["A...", "B..."]
PLAIN_THEN_GROUP = ["P", "A..."]
GROUP_THEN_PLAIN = ["A...", "P"]
ONE_PLAIN = ["P"]

FORWARDER = ["G..."]
"""`Outer`'s own single variadic parameter, the one a forwarding test hands over."""

TWO_FORWARDERS = ["G...", "H..."]


# ==================================================================================================
# 1. The canonical form: a bracketed group per variadic parameter
# ==================================================================================================


class TestTheBracketedFormAlwaysWorks:
    """
    The form every other one is measured against. It says exactly which argument belongs to which group,
    so it needs no identifiers and cannot be ambiguous -- if this ever needs a declaration to work, the
    declaration has stopped being a disambiguator and become a prerequisite.
    """

    @pytest.mark.parametrize("identifiers", [None, {"A": ""}], ids=["undeclared", "declared"])
    @pytest.mark.parametrize("application", ["Target<[]>", "Target<[Integer]>", "Target<[Integer, String]>"])
    def test_one_group_takes_any_arity(self, identifiers, application):
        accepts(ONE_GROUP, application, identifiers=identifiers)

    @pytest.mark.parametrize("identifiers", [None, {"A": "", "B": "!"}], ids=["undeclared", "declared"])
    @pytest.mark.parametrize(
        "application", ["Target<[], []>", "Target<[Integer], []>", "Target<[Integer, String], [Number]>"]
    )
    def test_two_groups_take_any_arity(self, identifiers, application):
        accepts(TWO_GROUPS, application, identifiers=identifiers)

    def test_a_plain_parameter_before_a_group(self):
        accepts(PLAIN_THEN_GROUP, "Target<Integer, [String]>")

    def test_a_plain_parameter_after_a_group(self):
        accepts(GROUP_THEN_PLAIN, "Target<[String], Integer>")

    def test_a_type_with_no_variadic_parameter_is_unaffected(self):
        """The control: none of this machinery may disturb an ordinary templated type."""
        accepts(ONE_PLAIN, "Target<Integer>")


# ==================================================================================================
# 2. The bare shorthand: single types, no brackets
# ==================================================================================================


class TestTheBareShorthand:
    """
    Writing the members without brackets, which is what the identifiers exist for. With one group there is
    nothing to settle and it must work undeclared; with several, it is genuinely ambiguous unless an
    identifier says which group is which.
    """

    @pytest.mark.parametrize("identifiers", [None, {"A": ""}], ids=["undeclared", "declared"])
    @pytest.mark.parametrize("application", ["Target<Integer>", "Target<Integer, String>"])
    def test_one_group_takes_the_shorthand_declared_or_not(self, identifiers, application):
        accepts(ONE_GROUP, application, identifiers=identifiers)

    @pytest.mark.parametrize("order", [PLAIN_THEN_GROUP, GROUP_THEN_PLAIN])
    def test_a_plain_parameter_beside_a_group_takes_the_shorthand(self, order):
        """The plain parameter takes one bare value and the group takes the rest, by position."""
        accepts(order, "Target<Integer, String>")

    def test_the_plain_parameter_may_be_the_only_argument(self):
        accepts(PLAIN_THEN_GROUP, "Target<Integer>")

    def test_two_groups_take_the_shorthand_when_identifiers_say_which(self):
        accepts(TWO_GROUPS, "Target<Integer, !String>", identifiers={"A": "", "B": "!"})

    def test_two_groups_without_identifiers_refuse_the_shorthand(self):
        """The one genuinely ambiguous case, and the only thing a declaration is needed for."""
        assert refuses(TWO_GROUPS, "Target<Integer, String>")

    def test_the_ambiguity_is_reported_as_an_ambiguity(self):
        messages = refuses(TWO_GROUPS, "Target<Integer, String>")
        assert "which" in messages.lower(), messages
        assert "A" in messages and "B" in messages, messages

    def test_the_ambiguity_names_a_way_out(self):
        """Brackets need no declaration, so naming them first is the cheaper advice."""
        messages = refuses(TWO_GROUPS, "Target<Integer, String>")
        assert "[" in messages, messages


# ==================================================================================================
# 3. Forwarding a group by name
# ==================================================================================================


class TestForwardingAGroupByName:
    """
    ``Target<G>`` where ``G`` is the enclosing type's variadic parameter. A variadic parameter denotes a
    group, so there is exactly one thing this can mean -- hand the group over -- and it is the form an
    author reaches for when passing their own group on.

    It is not the shorthand: nothing is being collected into a group, so it is never ambiguous, and having
    several groups does not make it so. That is the case this suite exists for.
    """

    @pytest.mark.parametrize("identifiers", [None, {"A": ""}], ids=["undeclared", "declared"])
    def test_one_group_forwards(self, identifiers):
        accepts(ONE_GROUP, "Target<G>", FORWARDER, identifiers=identifiers)

    @pytest.mark.parametrize("identifiers", [None, {"A": "", "B": "!"}], ids=["undeclared", "declared"])
    def test_two_groups_forward_positionally(self, identifiers):
        """``Target<G, H>``: one group variable per group parameter, matched in order."""
        accepts(TWO_GROUPS, "Target<G, H>", TWO_FORWARDERS, identifiers=identifiers)

    def test_forwarding_and_bracketing_compose(self):
        accepts(TWO_GROUPS, "Target<G, [Integer]>", FORWARDER)

    def test_forwarding_beside_a_plain_parameter(self):
        accepts(PLAIN_THEN_GROUP, "Target<Integer, G>", FORWARDER)

    def test_the_spliced_form_is_equivalent(self):
        """``Target<[G...]>`` writes the same thing out: a group holding the members of ``G``."""
        accepts(ONE_GROUP, "Target<[G...]>", FORWARDER)

    def test_a_group_inside_a_group_is_refused(self):
        """``[G]`` puts a group where a member belongs. The operator is what says which was meant."""
        assert refuses(ONE_GROUP, "Target<[G]>", FORWARDER)

    def test_the_refusal_points_at_the_operator(self):
        messages = refuses(ONE_GROUP, "Target<[G]>", FORWARDER)
        assert "..." in messages, messages

    def test_a_plain_variable_is_not_a_group(self):
        """The control: only a *variadic* parameter denotes a group, so only it can be forwarded."""
        accepts(ONE_PLAIN, "Target<S>", ["S"])


# ==================================================================================================
# 4. Supplying the wrong number of groups
# ==================================================================================================


class TestTheGroupsMustAllBeSupplied:
    """
    A variadic parameter is not optional -- an empty group is written ``[]`` -- so every group has to be
    accounted for. Leaving one out used to reach an ``assert`` inside the type machinery, which is a crash
    rather than a diagnosis: no location, no message, nothing an author could act on.
    """

    def test_too_few_groups_is_refused(self):
        assert refuses(TWO_GROUPS, "Target<[Integer]>")

    def test_too_few_groups_does_not_crash(self):
        """Pinned separately: `refuses` lets a non-`ConceptHierarchyError` through, so a crash fails here."""
        with pytest.raises(ConceptHierarchyError):
            accepts(TWO_GROUPS, "Target<[Integer]>")

    def test_the_refusal_says_how_many_groups_there_are(self):
        messages = refuses(TWO_GROUPS, "Target<[Integer]>")
        assert "A" in messages and "B" in messages, messages

    def test_the_refusal_says_to_write_them_all(self):
        messages = refuses(TWO_GROUPS, "Target<[Integer]>")
        assert "[" in messages, messages

    def test_too_few_forwarded_groups_is_refused(self):
        """
        Forwarding is positional, so leaving a group out leaves it *unsaid* rather than empty. The bare
        shorthand below is the form that may be partial, and the difference is what the argument denotes:
        a group variable fills one group parameter, a plain value joins one group.
        """
        assert refuses(TWO_GROUPS, "Target<G>", FORWARDER)

    def test_the_shorthand_may_still_leave_a_group_unmentioned(self):
        """
        The control, and the behaviour that must not change: a *plain* value is routed by identifier, and
        a group no value names stays empty. `Instance<Kennel>` all over the shipped examples is this form.
        """
        accepts(TWO_GROUPS, "Target<Integer>", identifiers={"A": "", "B": "!"})

    def test_forwarding_is_not_the_shorthand_even_when_identifiers_exist(self):
        """The same shape as the previous case, with a group variable instead of a value: now incomplete."""
        assert refuses(TWO_GROUPS, "Target<G>", FORWARDER, identifiers={"A": "", "B": "!"})

    def test_too_many_groups_is_refused(self):
        assert refuses(ONE_GROUP, "Target<[Integer], [String]>")

    def test_a_missing_plain_argument_is_refused(self):
        assert refuses(PLAIN_THEN_GROUP, "Target<>")

    def test_no_arguments_at_all_where_some_are_expected(self):
        assert refuses(ONE_PLAIN, "Target<>")


# ==================================================================================================
# 5. The empty group
# ==================================================================================================


class TestAnEmptyGroupIsAValue:
    """
    ``[]`` is a group with no members, which is a different thing from omitting the group. Both spellings
    of "no arguments" have to reach the same place when there is one group and nothing else to supply.
    """

    @pytest.mark.parametrize("application", ["Target<[]>", "Target<>"])
    def test_one_group_may_be_empty(self, application):
        accepts(ONE_GROUP, application)

    def test_every_group_may_be_empty(self):
        accepts(TWO_GROUPS, "Target<[], []>")

    def test_an_empty_forwarded_group_is_still_a_group(self):
        accepts(ONE_GROUP, "Target<G>", FORWARDER)


# ==================================================================================================
# 6. What the identifiers actually buy
# ==================================================================================================


class TestVariadicGroupIdentifiers:
    """
    Declared identifiers change exactly one thing: which group a **bare** argument joins. They may not
    change whether a bracketed or forwarded application is legal, and they may not be required for one.
    """

    @pytest.mark.parametrize("application", ["Target<[Integer], [String]>", "Target<G, H>", "Target<[], []>"])
    def test_the_unambiguous_forms_do_not_depend_on_them(self, application):
        undeclared = written(TWO_GROUPS, application, TWO_FORWARDERS)
        declared = written(TWO_GROUPS, application, TWO_FORWARDERS, identifiers={"A": "", "B": "!"})
        for concepts in (undeclared, declared):
            with contextlib.redirect_stdout(io.StringIO()):
                check_hierarchy(build_hierarchy(concepts))

    def test_an_identifier_directs_a_bare_argument(self):
        accepts(TWO_GROUPS, "Target<!Integer>", identifiers={"A": "", "B": "!"})

    def test_an_undeclared_identifier_is_refused(self):
        messages = refuses(TWO_GROUPS, "Target<$Integer>", identifiers={"A": "", "B": "!"})
        assert messages

    def test_mixing_an_identifier_with_a_bracketed_group_is_refused(self):
        """The two syntaxes answer the same question, and answering it twice has no defined meaning."""
        assert refuses(TWO_GROUPS, "Target<[Integer], !String>", identifiers={"A": "", "B": "!"})

    def test_a_variadic_group_with_non_empty_variadic_identifier_can_not_use_the_empty_shorthand_form(self):
        applications = ["Target<>", "Target<!CustomFunction>", "Target<!CustomFunction, !CustomFunction>"]
        reject_applications = ["Target<CustomFunction>", "Target<!CustomFunction, CustomFunction>"]
        for application in applications:
            check_hierarchy(build_hierarchy(written(ONE_GROUP, application, identifiers={"A": "!"})))
        for reject_application in reject_applications:
            with pytest.raises(
                CHSemanticError,
                match=(
                    rf"CustomFunction in {reject_application} does not specify to which variadic group of Target<A...> "
                    rf"it belongs"
                ),
            ):
                check_hierarchy(build_hierarchy(written(ONE_GROUP, reject_application, identifiers={"A": "!"})))

    def test_more_arguments_than_template_parameters_are_processed_correctly(self):
        model = check_hierarchy(
            build_hierarchy(
                written(
                    ["Group...", "Plain"],
                    "Target<$CustomFunction, $CustomFunction, CustomFunction>",
                    identifiers={"Group": "$"},
                )
            )
        ).model
        # {"instantiation": {"type": "object", "properties": {"v": {"type": application}}}}
        parsed_type = model.value_domains["Outer"].instantiation[0][1].properties["v"].custom_type
        assert isinstance(parsed_type, InstantiatedType)
        assert len(parsed_type.template_arguments) == 2
        assert isinstance(parsed_type.template_arguments[0], InstantiatedVariadicGroup)
        assert all(x.full_name == "CustomFunction" for x in parsed_type.template_arguments[0].variadic_group)
        assert isinstance(parsed_type.template_arguments[1], InstantiatedType)
        assert parsed_type.template_arguments[1].full_name == "CustomFunction"

    def test_more_arguments_than_template_parameters_are_rejected_correctly(self):
        reject_applications = ["Target<CustomFunction, !CustomFunction>", "Target<!CustomFunction, CustomFunction>"]
        for reject_application in reject_applications:
            with pytest.raises(
                CHSemanticError,
                match=(
                    rf"CustomFunction in {reject_application} does not specify to which variadic group of Target<A...> "
                    rf"it belongs"
                ),
            ):
                check_hierarchy(build_hierarchy(written(ONE_GROUP, reject_application, identifiers={"A": "!"})))

    def test_argument_with_unknown_identifier_past_number_of_template_parameters(self):
        with pytest.raises(
            CHSemanticError,
            match=r"Found a template argument value !!CustomFunction with an undefined variadic identifier '!!'.",
        ):
            check_hierarchy(
                build_hierarchy(written(ONE_GROUP, "Target<!!CustomFunction, !CustomFunction>", identifiers={"A": "!"}))
            )


TWO_GROUPS_TWO_PLAIN = ["A...", "B...", "P", "Q"]
"""Groups *and* plain parameters together. Counting arguments without separating the two kinds makes the
groups look like they cover the plain parameters, which is how too few arguments used to go unnoticed."""

THREE_GROUPS = ["A...", "B...", "C..."]
GROUP_BETWEEN_PLAINS = ["P", "A...", "Q"]


# ==================================================================================================
# 7. Arity, once groups and plain parameters are in the same list
# ==================================================================================================


class TestPlainParametersBesideGroups:
    """
    A written group accounts for one *group* parameter and for nothing else, so the plain parameters still
    need one argument each. Counting the arguments as a single total hides that entirely: two groups and
    two plain parameters is four arguments, and ``Target<G, H>`` is two -- which passes a total-count check
    and then leaves the plain parameters with nothing.

    What that produced was an `AssertionError` from inside the type machinery, with no location and no
    message: the author is told nothing, and the checker stops rather than reporting.
    """

    def test_every_plain_parameter_still_needs_an_argument(self):
        assert refuses(TWO_GROUPS_TWO_PLAIN, "Target<G, H>", TWO_FORWARDERS)

    def test_it_is_reported_rather_than_asserted(self):
        """`refuses` lets anything that is not a `ConceptHierarchyError` through, so a crash fails here."""
        with pytest.raises(ConceptHierarchyError):
            accepts(TWO_GROUPS_TWO_PLAIN, "Target<G, H>", TWO_FORWARDERS)

    def test_the_refusal_names_the_parameters_left_unsupplied(self):
        messages = refuses(TWO_GROUPS_TWO_PLAIN, "Target<G, H>", TWO_FORWARDERS)
        assert "P" in messages and "Q" in messages, messages

    def test_the_same_hole_with_bracketed_groups(self):
        """Nothing about this is specific to forwarding; a bracket leaves the same gap."""
        assert refuses(TWO_GROUPS_TWO_PLAIN, "Target<[], []>")

    def test_one_plain_argument_short(self):
        assert refuses(TWO_GROUPS_TWO_PLAIN, "Target<[], [], Integer>")

    def test_one_plain_argument_too_many(self):
        assert refuses(TWO_GROUPS_TWO_PLAIN, "Target<[], [], Integer, String, Number>")

    @pytest.mark.parametrize(
        "application",
        ["Target<[], [], Integer, String>", "Target<G, H, Integer, String>", "Target<G, [Number], Integer, String>"],
    )
    def test_the_complete_forms_are_accepted(self, application):
        accepts(TWO_GROUPS_TWO_PLAIN, application, TWO_FORWARDERS)

    def test_a_group_between_two_plain_parameters(self):
        """Position matters for the plain ones: the group takes the middle and they take the ends."""
        accepts(GROUP_BETWEEN_PLAINS, "Target<Integer, [String], Number>")

    def test_three_groups_all_written(self):
        accepts(THREE_GROUPS, "Target<[], [], []>")

    def test_three_groups_one_missing(self):
        assert refuses(THREE_GROUPS, "Target<[], []>")


class TestAGroupWhereTheTypeHasNone:
    """
    A group written at a type with no variadic parameter at all. The count is right and only the *kind* is
    wrong, so the arity checks have nothing to say and the message has to be about the kind.
    """

    def test_a_bracketed_group_at_a_plain_parameter(self):
        assert refuses(ONE_PLAIN, "Target<[Integer]>")

    def test_a_forwarded_group_at_a_plain_parameter(self):
        assert refuses(ONE_PLAIN, "Target<G>", FORWARDER)

    def test_the_refusal_does_not_talk_about_groups_to_write(self):
        """
        There is no group to write, so advising the author to write them all is advice they cannot take.
        Since the match is positional, the honest thing to say is which parameter is at that position and
        that it is not variadic -- which is the same diagnosis as any other kind mismatch.
        """
        messages = refuses(ONE_PLAIN, "Target<G>", FORWARDER)
        assert "is declared, which is not variadic" in messages, messages
        assert "write every group" not in messages.lower(), messages


# ==================================================================================================
# 8. What may stand as an argument
# ==================================================================================================


class TestNestedAndLiteralArguments:
    """
    A template argument is a type, and a type may itself be an application -- so the group machinery has to
    leave a nested one alone. Literals are the other kind of argument and are constrained differently.
    """

    def test_an_application_inside_a_group(self):
        accepts(ONE_GROUP, "Target<[List<Integer>]>")

    def test_an_application_as_the_bare_shorthand(self):
        accepts(ONE_GROUP, "Target<List<Integer>>")

    def test_several_applications_in_one_group(self):
        accepts(ONE_GROUP, "Target<[List<Integer>, List<String>]>")

    def test_a_literal_parameter_takes_a_literal(self):
        accepts(["N"], "Target<1>", constraints={"N": "Literal:integer"})

    def test_a_literal_group_bracketed(self):
        accepts(["N..."], "Target<[1, 2]>", constraints={"N": "Literal:integer"})

    def test_a_literal_group_as_the_shorthand(self):
        accepts(["N..."], "Target<1, 2>", constraints={"N": "Literal:integer"})

    def test_a_literal_where_a_type_is_expected(self):
        assert refuses(ONE_GROUP, "Target<[1]>")

    def test_an_unknown_name(self):
        assert refuses(ONE_GROUP, "Target<[Nope]>")


class TestTheExpansionOperator:
    """
    ``...`` splices the *members* of a group. Where it may stand is decided by what encloses it, and each
    of the four combinations below means something different or nothing at all.
    """

    def test_spliced_inside_a_group(self):
        accepts(ONE_GROUP, "Target<[G...]>", FORWARDER)

    def test_spliced_as_the_bare_shorthand(self):
        """``Target<G...>`` collects the members into the group the shorthand is building."""
        accepts(ONE_GROUP, "Target<G...>", FORWARDER)

    def test_spliced_at_a_plain_parameter(self):
        """Nothing to splice into: a plain parameter takes one value, not a sequence."""
        assert refuses(ONE_PLAIN, "Target<G...>", FORWARDER)

    def test_on_a_concept(self):
        """Only a variadic parameter has members; a concept is one type."""
        assert refuses(ONE_GROUP, "Target<[Integer...]>")

    def test_on_a_plain_parameter_of_the_enclosing_type(self):
        assert refuses(ONE_GROUP, "Target<[S...]>", ["S"])

    def test_a_group_inside_a_group_without_it(self):
        assert refuses(ONE_GROUP, "Target<[G]>", FORWARDER)

    def test_two_groups_spliced_into_one(self):
        """Concatenation: both sets of members land in the one group being written."""
        accepts(ONE_GROUP, "Target<[G..., H...]>", TWO_FORWARDERS)


class TestConstraintsOnTheArguments:
    """
    The parameter's constraint applies to whatever fills it, however the argument was written -- the group
    forms must not become a way of getting an argument past a constraint.
    """

    def test_a_plain_parameter_enforces_its_constraint(self):
        accepts(["P"], "Target<Pet>", constraints={"P": "Not(ValueDomain)"})
        assert refuses(["P"], "Target<Integer>", constraints={"P": "Not(ValueDomain)"})

    def test_a_bracketed_group_enforces_it_per_member(self):
        accepts(ONE_GROUP, "Target<[Pet]>", constraints={"A": "Not(ValueDomain)"})
        assert refuses(ONE_GROUP, "Target<[Integer]>", constraints={"A": "Not(ValueDomain)"})

    def test_the_shorthand_enforces_it_too(self):
        accepts(ONE_GROUP, "Target<Pet>", constraints={"A": "Not(ValueDomain)"})
        assert refuses(ONE_GROUP, "Target<Integer>", constraints={"A": "Not(ValueDomain)"})

    def test_one_bad_member_among_good_ones(self):
        assert refuses(ONE_GROUP, "Target<[Pet, Integer]>", constraints={"A": "Not(ValueDomain)"})

    def test_an_empty_group_violates_nothing(self):
        accepts(ONE_GROUP, "Target<[]>", constraints={"A": "Not(ValueDomain)"})


# ==================================================================================================
# 9. Declaring the identifiers
# ==================================================================================================


class TestDeclaringTheIdentifiers:
    """
    The declaration side, which is checked when the *type* is defined rather than when it is applied. All
    of these are already diagnosed; they are pinned so that relaxing the application side never quietly
    relaxes the declaration side with it.
    """

    def test_an_identifier_on_a_plain_parameter(self):
        messages = refuses(PLAIN_THEN_GROUP, "Target<Integer, [String]>", identifiers={"P": "!"})
        assert "not a variadic template argument" in messages, messages

    def test_two_parameters_with_the_same_identifier(self):
        messages = refuses(TWO_GROUPS, "Target<[], []>", identifiers={"A": "!", "B": "!"})
        assert "Duplicate variadic group identifier" in messages, messages

    def test_declaring_one_identifier_of_two(self):
        """All or none: a half-declared set leaves exactly the ambiguity the declaration exists to remove."""
        messages = refuses(TWO_GROUPS, "Target<[], []>", identifiers={"A": "!"})
        assert "does not define one for B" in messages, messages

    def test_the_empty_identifier_beside_a_plain_parameter(self):
        """The empty identifier means "the unprefixed ones", which the plain parameters already claim."""
        messages = refuses(PLAIN_THEN_GROUP, "Target<Integer, [String]>", identifiers={"A": ""})
        assert "empty variadic group identifier" in messages, messages

    def test_an_unknown_identifier_on_an_argument(self):
        messages = refuses(TWO_GROUPS, "Target<$Integer, [String]>", identifiers={"A": "", "B": "!"})
        assert messages

    def test_a_declared_identifier_is_usable(self):
        accepts(TWO_GROUPS, "Target<Integer, !String>", identifiers={"A": "", "B": "!"})


# ==================================================================================================
# 10. Positional matching
# ==================================================================================================


class TestTheMatchIsPositional:
    """
    Argument *i* fills parameter *i* of ``order``. Nothing reorders, and nothing is matched by counting:
    two groups and two plain parameters is not "two of each somewhere", it is a group at 0, a group at 1,
    a value at 2 and a value at 3.

    The one exception is the identifier syntax, and it is the subject of the next class: there, and only
    there, the members of a group may be written intertwined with the non-variadic arguments, because the
    prefix says which group each belongs to. The moment anything is written *as* a group -- bracketed, or
    a variadic parameter named on its own -- that syntax is not in use and position is all there is.

    Counting instead of matching accepts every permutation of a correct application, which is the failure
    this class exists for: ``Target<G, Pet, H, Number>`` has the right number of groups and the right
    number of values, and is wrong at two of its four positions.
    """

    ORDER = ["A...", "B...", "P", "Q"]
    """Two groups then two plain parameters, so a swap between the two kinds is expressible."""

    def test_the_correct_positions_are_accepted(self):
        accepts(self.ORDER, "Target<G, H, Integer, Number>", TWO_FORWARDERS)

    def test_a_value_written_at_a_group_position(self):
        """``Pet`` sits where ``B...`` is declared. The counts are right; the position is not."""
        assert refuses(self.ORDER, "Target<G, Integer, H, Number>", TWO_FORWARDERS)

    def test_a_group_written_at_a_plain_position(self):
        """The mirror image, and the second error in the same application."""
        assert refuses(self.ORDER, "Target<G, H, ValueDomainTypes, Number>", TWO_FORWARDERS + ["ValueDomainTypes..."])

    def test_the_refusal_names_the_position(self):
        messages = refuses(self.ORDER, "Target<G, Integer, H, Number>", TWO_FORWARDERS)
        assert "Integer" in messages, messages
        assert "B" in messages, messages

    def test_the_refusal_says_the_match_is_positional(self):
        messages = refuses(self.ORDER, "Target<G, Integer, H, Number>", TWO_FORWARDERS)
        assert "position" in messages.lower(), messages

    @pytest.mark.parametrize(
        "application",
        [
            pytest.param("Target<Integer, G, H, Number>", id="value-at-group-0"),
            pytest.param("Target<G, Integer, H, Number>", id="value-at-group-1"),
            pytest.param("Target<G, H, [Integer], Number>", id="group-at-plain-2"),
            pytest.param("Target<G, H, Integer, [Number]>", id="group-at-plain-3"),
            pytest.param("Target<[Integer], Integer, H, Number>", id="bracketed-then-value"),
        ],
    )
    def test_every_single_position_swap_is_refused(self, application):
        assert refuses(self.ORDER, application, TWO_FORWARDERS)

    def test_a_plain_parameter_before_a_group_keeps_its_position(self):
        accepts(PLAIN_THEN_GROUP, "Target<Integer, [String]>")
        assert refuses(PLAIN_THEN_GROUP, "Target<[String], Integer>")

    def test_a_plain_parameter_after_a_group_keeps_its_position(self):
        accepts(GROUP_THEN_PLAIN, "Target<[String], Integer>")
        assert refuses(GROUP_THEN_PLAIN, "Target<Integer, [String]>")

    def test_a_group_between_two_plain_parameters_keeps_its_position(self):
        accepts(GROUP_BETWEEN_PLAINS, "Target<Integer, [String], Number>")
        assert refuses(GROUP_BETWEEN_PLAINS, "Target<[String], Integer, Number>")
        assert refuses(GROUP_BETWEEN_PLAINS, "Target<Integer, Number, [String]>")

    def test_forwarding_is_positional_too(self):
        """A forwarded group is a group, so it is bound by the same rule a bracket is."""
        accepts(GROUP_THEN_PLAIN, "Target<G, Integer>", FORWARDER)
        assert refuses(GROUP_THEN_PLAIN, "Target<Integer, G>", FORWARDER)

    def test_the_group_syntax_requires_an_argument_at_every_position(self):
        """No defaulting once groups are written: every parameter is matched, or none is."""
        assert refuses(self.ORDER, "Target<G, H, Integer>", TWO_FORWARDERS)
        assert refuses(self.ORDER, "Target<G, H, Integer, Number, String>", TWO_FORWARDERS)


class TestOnlyTheIdentifierSyntaxMayIntertwine:
    """
    The exception, and its boundary. With no argument written as a group, the members of a group are
    written bare and told apart by their prefix -- so they may appear anywhere among the non-variadic
    arguments, which keep their own relative order.

    What must *not* happen is this leniency leaking into the group syntax: the moment one argument is
    written as a group, the whole application is positional again.
    """

    ORDER = ["A...", "P"]
    IDENTIFIED = {"A": "!"}

    def test_a_group_member_may_precede_the_plain_argument(self):
        accepts(self.ORDER, "Target<!String, Integer>", identifiers=self.IDENTIFIED)

    def test_a_group_member_may_follow_it(self):
        accepts(self.ORDER, "Target<Integer, !String>", identifiers=self.IDENTIFIED)

    def test_members_may_surround_it(self):
        accepts(self.ORDER, "Target<!String, Integer, !Number>", identifiers=self.IDENTIFIED)

    def test_the_plain_arguments_keep_their_relative_order(self):
        """Two plain parameters of different constraints, so an order swap is observable."""
        order = ["A...", "P", "Q"]
        accepts(order, "Target<!String, Pet, Integer>", identifiers={"A": "!"}, constraints={"P": "Not(ValueDomain)"})
        assert refuses(
            order, "Target<!String, Integer, Pet>", identifiers={"A": "!"}, constraints={"P": "Not(ValueDomain)"}
        )

    def test_writing_the_group_switches_back_to_positional(self):
        """A bracket is the group syntax even where identifiers are declared, so position rules again."""
        accepts(self.ORDER, "Target<[String], Integer>", identifiers=self.IDENTIFIED)
        assert refuses(self.ORDER, "Target<Integer, [String]>", identifiers=self.IDENTIFIED)

    def test_an_identifier_may_not_be_mixed_with_a_written_group(self):
        """Both syntaxes answering the same question at once has no defined meaning."""
        assert refuses(self.ORDER, "Target<[String], !Number>", identifiers=self.IDENTIFIED)
