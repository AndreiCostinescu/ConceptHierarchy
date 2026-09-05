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
Integration tests: expansion of ValueDomain instantiation ``default`` expressions, and the dependency
cycles between them.

Background
----------
A ``default`` on a custom-type node of an instantiation schema is *constructive*: when the key is absent
from the supplied JSON, ``value_instantiation_parser._parse_absent`` materialises the default instead.
That default may itself be an ``Inst``/``Narrow`` expression whose value leaves further keys absent,
which materialises *their* defaults, and so on. If that expansion ever returns to a default site it has
already entered, the value being described is infinite and no value satisfies the schemas -- the
hierarchy is ill-formed and must be rejected.

Stage 2 implements that rejection. These tests pin three things:

1. **The forcing semantics** (:class:`TestForcingSemantics`) -- which absences actually materialise a
   default, and which do not. The edge relation of the cycle check is defined directly in terms of them.
2. **Resolution completeness** (:class:`TestAcyclicDefaultsResolve`) -- an *acyclic* default chain is
   fully resolved, whatever order its ValueDomains were declared in, and leaves no site that applied a
   default without working out what it is.
3. **Cycle rejection** (:class:`TestDefaultExpansionCyclesAreRejected`,
   :class:`TestTemplateDependentCyclesAreApplicationSpecific`) -- and, just as importantly, *passing*
   guards for the benign cases that must **not** be rejected. The obvious implementation (one graph keyed
   by schema-node location) over-rejects, because a template-dependent default site is reached under
   several ground applications and is only cyclic under some of them.

`assert_rejected_as_a_cycle` is what keeps (3) honest: it requires the hierarchy to be rejected *as a
diagnosed cycle*, not merely rejected. Without that distinction a runaway that hit the expansion depth
bound, or the interpreter's stack, would pass these tests while being a far worse outcome reported far
later.

The one remaining ``xfail`` is the cycle that runs through an unsupplied Function argument, which needs
the Function-side work; it is non-strict, so it reports XPASS once that lands.

See ``documentation/EXPRESSIONS_AND_INSTANTIATION_SCHEMAS.md`` for how the two parsers interlock, and
``documentation/TODO_DEFAULT_EXPANSION_CYCLES.md`` for the implementation plan these tests describe.
"""

from __future__ import annotations

import pytest

from concept_hierarchy.data.contexts.context import ConceptHierarchyContext
from concept_hierarchy.data.expressions.expression import Expression
from concept_hierarchy.data.expressions.instantiated_value import ParsedCustomValue
from concept_hierarchy.data.expressions.subexpressions import FunctionEvaluation, InstExpression
from concept_hierarchy.errors import ConceptHierarchyError
from tests.integration.test_expression_parsing import build_hierarchy, check_concepts, check_hierarchy

# --------------------------------------------------------------------------------------------------
# Hierarchy fragments
# --------------------------------------------------------------------------------------------------

LEAF = {
    "Leaf": {
        "directParents": ["ValueDomain"],
        "data": {"instantiation": {"type": "object", "additionalProperties": False}},
    }
}
"""A ValueDomain that instantiates from ``{}`` and forces nothing. The base case of every benign chain."""


def value_domain(name: str, prop: str, prop_type: str, default: object = ..., required: bool = True) -> dict:
    """A ValueDomain whose instantiation is a single-property object, optionally with a default."""
    prop_schema: dict = {"type": prop_type}
    if default is not ...:
        prop_schema["default"] = default
    instantiation: dict = {"type": "object", "properties": {prop: prop_schema}}
    if required:
        instantiation["required"] = [prop]
    return {name: {"directParents": ["ValueDomain"], "data": {"instantiation": instantiation}}}


# --------------------------------------------------------------------------------------------------
# Accessors
# --------------------------------------------------------------------------------------------------


def default_sites(context: ConceptHierarchyContext, value_domain_name: str) -> dict[str, Expression | None]:
    """Every default site of a ValueDomain's instantiation schema(s), keyed by its property name."""
    sites: dict[str, Expression | None] = {}
    for _constraint, schema in context.model.value_domains[value_domain_name].instantiation:
        for node in schema.walk():
            if node.has_default:
                sites[str(node.location_id[-2])] = node.parsed_default_expr
    return sites


def materialised_leaves(expression: Expression | None) -> list[ParsedCustomValue]:
    """
    The leaves whose defaults ``expression``'s value materialises, directly or through a value it
    supplies. Stops at each materialised leaf -- that leaf's *own* onward expansion belongs to it, not here.
    """
    found: list[ParsedCustomValue] = []
    stack = [expression] if expression is not None else []
    while stack:
        current = stack.pop()
        value = current.value
        if not isinstance(value, InstExpression) or value.value is None:
            continue
        for node in value.value.walk():
            if not isinstance(node, ParsedCustomValue):
                continue
            if node.used_default:
                found.append(node)
            elif node.expression is not None:
                stack.append(node.expression)
    return found


def materialised_defaults(expression: Expression | None) -> list[str]:
    """The property names of :func:`materialised_leaves`."""
    return [str(leaf.location_id[-1]) for leaf in materialised_leaves(expression)]


def unresolved_default_leaves(expression: Expression | None) -> list[str]:
    """
    Materialised-default leaves anywhere below ``expression`` that carry no expression at all.

    Such a leaf says "this default was applied" and, in the same breath, "and I never worked out what it
    is". Nothing in the codebase currently reports one: ``ParsedCustomValue.is_valid()`` inspects only
    ``errors``, and ``InstExpression.is_fully_parsed`` skips leaves whose ``expression`` is ``None``.
    """
    unresolved: list[str] = []
    stack = [(expression, "")] if expression is not None else []
    while stack:
        current, path = stack.pop()
        value = current.value
        if not isinstance(value, InstExpression) or value.value is None:
            continue
        for node in value.value.walk():
            if not isinstance(node, ParsedCustomValue) or not node.used_default:
                continue
            leaf_path = f"{path}.{node.location_id[-1]}"
            if node.expression is None:
                unresolved.append(leaf_path.lstrip("."))
            else:
                stack.append((node.expression, leaf_path))
    return unresolved


# --------------------------------------------------------------------------------------------------
# Navigating the parsed result, and asserting what is actually there
# --------------------------------------------------------------------------------------------------


def step_into(expression: Expression, key: str) -> Expression:
    """
    The expression one level down, at ``key``.

    ``key`` is a property name when the expression instantiates a value domain (an ``Inst`` or a
    ``Narrow``), and an argument name when it evaluates a Function. Descending is how the alternation
    between the two ASTs is crossed: an `InstExpression` holds a `ParsedValue`, whose custom-type leaves
    each hold an `Expression` again.
    """
    value = expression.value
    if isinstance(value, FunctionEvaluation):
        assert key in value.arguments, f"no argument {key!r}; have {sorted(value.arguments)}"
        return value.arguments[key]
    assert isinstance(value, InstExpression), f"cannot descend into {type(value).__name__} looking for {key!r}"
    assert value.value is not None, f"{key!r}: this Inst has no value tree (it is a default serialization)"
    leaves = {str(node.location_id[-1]): node for node in value.value.walk() if isinstance(node, ParsedCustomValue)}
    assert key in leaves, f"no custom-type leaf {key!r}; have {sorted(leaves)}"
    nested = leaves[key].expression
    assert nested is not None, f"leaf {key!r} carries no expression"
    return nested


def assert_expression_is(expression: Expression, kind: type, *, value_type: str | None = None) -> Expression:
    """
    Assert what an expression actually parsed *to*, not merely that parsing succeeded.

    Asserting only "the hierarchy checks" lets a value be quietly parsed as the wrong kind -- a `Narrow`
    read as a plain instantiation because a schema left ``additionalProperties`` open, say, or a template
    variable left ungrounded. Both have happened here.
    """
    assert isinstance(expression.value, kind), (
        f"expected {kind.__name__}, got {type(expression.value).__name__} from {expression.unparsed!r}"
    )
    if value_type is not None:
        actual = expression.value.value_type
        assert actual is not None and actual.full_name == value_type, (
            f"expected the expression's type to be {value_type}, got {actual}"
        )
    return expression


def leaf_at(expression: Expression, *path: str) -> ParsedCustomValue:
    """The custom-type leaf at ``path``, so a test can assert on `used_default` and the substituted type."""
    for key in path[:-1]:
        expression = step_into(expression, key)
    value = expression.value
    assert isinstance(value, InstExpression) and value.value is not None
    leaves = {str(node.location_id[-1]): node for node in value.value.walk() if isinstance(node, ParsedCustomValue)}
    assert path[-1] in leaves, f"no leaf {path[-1]!r}; have {sorted(leaves)}"
    return leaves[path[-1]]


def assert_rejected_as_a_cycle(concepts: dict, *named: str) -> str:
    """
    The hierarchy must be rejected, and rejected *as a cycle* -- not merely rejected.

    Without this distinction the tests cannot tell a diagnosed cycle from a runaway that happened to hit
    the expansion depth bound or the interpreter's stack, which is a much worse outcome reported much
    later. Each name in ``named`` must appear, so the report identifies the application at fault.
    """
    with pytest.raises(ConceptHierarchyError) as excinfo:
        check_concepts(concepts)
    text = str(excinfo.value)
    assert "can never be applied" in text, f"expected a cycle diagnosis, got: {text[:600]}"
    assert "levels deep" not in text, "the cycle must be diagnosed, not left to the depth bound"
    assert "Ran out of stack" not in text, "the cycle must be diagnosed, not left to the interpreter"
    for name in named:
        assert name in text, f"expected {name!r} to be named in: {text[:600]}"
    return text


# --------------------------------------------------------------------------------------------------
# 1. Forcing semantics -- what actually materialises a default
# --------------------------------------------------------------------------------------------------


class TestForcingSemantics:
    """
    Which absences materialise a default. The edge relation of any cycle check is defined in terms of
    exactly these cases, so they are pinned before the check that consumes them exists.
    """

    def test_absent_required_property_materialises_its_default(self):
        """``Outer``'s default ``{}`` instantiates an ``Inner``, whose required ``i`` is absent."""
        context = check_concepts(
            {
                **LEAF,
                **value_domain("Inner", "i", "Leaf", default={}),
                **value_domain("Outer", "o", "Inner", default={}),
            }
        )
        outer = default_sites(context, "Outer")["o"]
        assert isinstance(outer.value, InstExpression), f"expected an Inst expression, got {outer.value!r}"
        assert materialised_defaults(outer) == ["i"]

    def test_absent_optional_property_also_materialises_its_default(self):
        """
        ``required`` plays no part in it: ``_parse_object`` calls ``_parse_absent`` for *every* declared
        property that is absent, and ``_parse_absent`` materialises whatever default it finds.

        This is why an optional self-referential default is just as non-terminating as a required one --
        see :meth:`TestDefaultExpansionCyclesAreRejected.test_optional_self_cycle_is_rejected`.
        """
        context = check_concepts(
            {
                **LEAF,
                **value_domain("Inner", "i", "Leaf", default={}, required=False),
                **value_domain("Outer", "o", "Inner", default={}),
            }
        )
        assert materialised_defaults(default_sites(context, "Outer")["o"]) == ["i"], (
            "an absent *optional* property with a default is materialised just like a required one"
        )

    def test_explicitly_supplied_key_does_not_materialise_the_default(self):
        """A default that spells the key out does not force it; only the keys it leaves absent count."""
        context = check_concepts(
            {
                **LEAF,
                **value_domain("Inner", "i", "Leaf", default={}),
                **value_domain("Outer", "o", "Inner", default={"i": {}}),
            }
        )
        assert materialised_defaults(default_sites(context, "Outer")["o"]) == []

    def test_required_property_without_a_default_is_rejected(self):
        """
        The one unsatisfiable case that *is* caught today: no default means nothing can fill the gap, so
        the value fails to match and the enclosing expression is ill-formed.
        """
        with pytest.raises(ConceptHierarchyError, match="Could not match a valid Req expression"):
            check_concepts(
                {
                    **value_domain("Req", "p", "Integer"),
                    **value_domain("Uses", "u", "Req", default={}),
                }
            )


# --------------------------------------------------------------------------------------------------
# 2. Resolution completeness on acyclic chains
# --------------------------------------------------------------------------------------------------


class TestAcyclicDefaultsResolve:
    """An acyclic default chain is perfectly legitimate and must end up completely resolved."""

    def test_two_deep_chain_is_fully_resolved(self):
        context = check_concepts(
            {**LEAF, **value_domain("Y", "y", "Leaf", default={}), **value_domain("X", "x", "Y", default={})}
        )
        assert unresolved_default_leaves(default_sites(context, "X")["x"]) == []

    def test_three_deep_chain_resolves_when_declared_bottom_up(self):
        """
        ``X -> Y -> Z -> Leaf``, declared deepest-first. Each site's dependency is already resolved by the
        time the site itself is reached, so one pass suffices and everything comes out complete.

        Nothing arranges this -- it is luck. See the next test.
        """
        context = check_concepts(
            {
                **LEAF,
                **value_domain("Z", "z", "Leaf", default={}),
                **value_domain("Y", "y", "Z", default={}),
                **value_domain("X", "x", "Y", default={}),
            }
        )
        assert unresolved_default_leaves(default_sites(context, "X")["x"]) == []

    def test_three_deep_chain_is_fully_resolved_regardless_of_declaration_order(self):
        """
        The *same* hierarchy as above, declared shallowest-first, is left half-resolved -- the unresolved
        leaf sits at ``x -> y -> z``.

        Pass 1 parses ``X`` before ``Y`` exists, so ``X``'s ``y`` leaf is empty; pass 2 fills it from
        pass 1's ``Y``, whose own ``z`` leaf was empty for the same reason; there is no pass 3. The
        resolution depth is a property of the number of passes and the declaration order, not of the
        hierarchy -- which is the artifact a real resolution driver has to remove.
        """
        context = check_concepts(
            {
                **LEAF,
                **value_domain("X", "x", "Y", default={}),
                **value_domain("Y", "y", "Z", default={}),
                **value_domain("Z", "z", "Leaf", default={}),
            }
        )
        assert unresolved_default_leaves(default_sites(context, "X")["x"]) == []

    def test_nothing_is_left_unresolved_anywhere(self):
        """
        `ParsedValue.unresolved_default_sites()` is the standing check that resolution finished: a site
        that says both "this default was used" and "and I do not know its value" is reported by no other
        predicate -- `is_valid` inspects only ``errors``, and `is_fully_parsed` skips leaves whose
        ``expression`` is ``None``.
        """
        context = check_concepts(
            {
                **LEAF,
                **value_domain("Z", "z", "Leaf", default={}),
                **value_domain("Y", "y", "Z", default={}),
                **value_domain("X", "x", "Y", default={}),
            }
        )
        for name in ("X", "Y", "Z"):
            expression = default_sites(context, name)[name.lower()]
            assert isinstance(expression.value, InstExpression)
            unresolved = list(expression.value.value.unresolved_default_sites())
            assert unresolved == [], f"{name} left {len(unresolved)} default(s) unresolved"

    def test_a_hierarchy_without_defaults_still_checks(self):
        """Guard against a cycle check that trips over a hierarchy with nothing to check."""
        context = check_concepts({**LEAF, **value_domain("Plain", "p", "Leaf")})
        assert default_sites(context, "Plain") == {}


# --------------------------------------------------------------------------------------------------
# 3. Cycles that must be rejected
# --------------------------------------------------------------------------------------------------


class TestDefaultExpansionCyclesAreRejected:
    """
    Every hierarchy here describes an infinite value and must be rejected.

    The two-concept case is the one that used to live in ``examples/animal_kingdom.json`` as
    ``TestDefaultDependentInstantiation_1``/``_2``; it belongs here, not in a shipped example.
    """

    def test_two_concept_cycle_is_rejected(self):
        assert_rejected_as_a_cycle(
            {
                **value_domain("Test1", "p1", "Test2", default={}),
                **value_domain("Test2", "p2", "Test1", default={}),
            },
            "Test2",
        )

    def test_four_concept_cycle_is_rejected(self):
        assert_rejected_as_a_cycle(
            {
                **value_domain("A", "a", "B", default={}),
                **value_domain("B", "b", "C", default={}),
                **value_domain("C", "c", "D", default={}),
                **value_domain("D", "d", "A", default={}),
            }
        )

    def test_self_cycle_is_rejected(self):
        assert_rejected_as_a_cycle(value_domain("Selfy", "p", "Selfy", default={}), "Selfy")

    def test_optional_self_cycle_is_rejected(self):
        """
        ``p`` is not required, but an absent optional property with a default is still materialised
        (:meth:`TestForcingSemantics.test_absent_optional_property_also_materialises_its_default`), so
        this expands forever exactly like the required version.
        """
        assert_rejected_as_a_cycle(value_domain("Opt", "p", "Opt", default={}, required=False), "Opt")

    def test_cycle_through_an_unsupplied_function_argument_is_rejected(self):
        """
        The cycle runs ``T1.q -> T2.r -> MakeT1.a -> T1.q``, where ``MakeT1.a`` is an *unsupplied*
        Function argument falling back on its own default.

        Unsupplied arguments never appear in ``FunctionEvaluation.arguments``, so this edge is invisible to
        any graph built purely from the parsed value tree. What makes it visible is stage 2.5: grounding an
        unsupplied argument's default parses it, and parsing it materialises whatever it forces -- so the
        Function-argument default site joins the same resolution path the instantiation defaults are on,
        without a second graph.
        """
        assert_rejected_as_a_cycle(
            {
                **value_domain("T1", "q", "T2", default={}),
                **value_domain("T2", "r", "T1", default={"MakeT1": {}}),
                "MakeT1": {
                    "directParents": ["FunctionReturning"],
                    "data": {
                        "templateContext": {"order": [], "substitution": {"FunctionReturning:T": "T1"}},
                        "interface": {"a": ["T1"], "res": "T1", "_defaultArgumentValues": {"a": {}}},
                    },
                },
            }
        )


NUMERIC_FREE_T = {"order": ["T"], "T": "ValueDomain", "substitution": {"FunctionReturning:T": "T"}}
"""An unconstrained template variable, so the Function can be applied to any of the domains below."""


def maker(
    name: str,
    argument: str,
    argument_type: str,
    default: object,
    returns: str,
    template: object = None,
) -> dict:
    """A Function with one defaulted argument, used to route a cycle through an *unsupplied* argument."""
    if template is None:
        template = {"order": [], "substitution": {"FunctionReturning:T": returns}}
    return {
        name: {
            "directParents": ["FunctionReturning"],
            "data": {
                "templateContext": template,
                "interface": {
                    argument: [argument_type],
                    "res": returns,
                    "_defaultArgumentValues": {argument: default},
                },
            },
        }
    }


class TestCyclesThroughUnsuppliedFunctionArguments:
    """
    The second kind of default site: a Function argument the call site leaves out.

    Such an argument never appears in `FunctionEvaluation.arguments`, so the edge it contributes is
    invisible to anything reading the parsed value tree. It becomes visible because grounding an unsupplied
    default (stage 2.5) *parses* it, and parsing materialises whatever it forces -- which puts the
    Function-argument site on the same resolution path as the instantiation defaults, rather than needing a
    second graph beside it.

    **Where** it is caught depends on whether the Function is templated, and the two cases are separated
    below. A ground Function's default has a ground type, so it is parsed and expanded where it is
    *declared* -- a cycle there is unconditional and no call site is needed to find it. A templated
    Function's default is deferred at the declaration, so only an application closes the loop, and only for
    that application.
    """

    def test_a_templated_functions_default_closes_a_cycle_at_one_application(self):
        """
        The case that needed stage 3. Nothing about `MakeW`'s declaration is wrong -- its default is
        deferred there, with ``T`` bound to nothing. ``MakeW<C1>`` is what closes
        ``C1.q -> C2.r -> MakeW<C1>.a -> C1.q``.
        """
        assert_rejected_as_a_cycle(
            {
                **value_domain("C1", "q", "C2", default={}),
                **value_domain("C2", "r", "C1", default={"MakeW<C1>": {}}),
                **maker("MakeW", "a", "T", {}, "T", template=NUMERIC_FREE_T),
            }
        )

    def test_the_same_function_at_another_application_is_fine(self):
        """
        The pair of the test above, and the reason the edge is per-application: the identical declaration,
        applied to a `Leaf`, forces nothing. A check that condemned `MakeW` itself would reject this.
        """
        check_concepts(
            {
                **LEAF,
                **value_domain("Holder", "h", "Leaf", default={"MakeW<Leaf>": {}}),
                **maker("MakeW", "a", "T", {}, "T", template=NUMERIC_FREE_T),
            }
        )

    def test_both_applications_in_one_hierarchy(self):
        """The benign application must not be rescued by, nor rescue, the cyclic one."""
        assert_rejected_as_a_cycle(
            {
                **LEAF,
                **value_domain("C1", "q", "C2", default={}),
                **value_domain("C2", "r", "C1", default={"MakeW<C1>": {}}),
                **value_domain("Holder", "h", "Leaf", default={"MakeW<Leaf>": {}}),
                **maker("MakeW", "a", "T", {}, "T", template=NUMERIC_FREE_T),
            }
        )

    def test_a_ground_functions_cyclic_default_is_caught_at_its_declaration(self):
        """
        With a ground argument type the default is parsed and expanded where it is written, so the cycle is
        reported against the declaration rather than against any call site. Still the same edge:
        ``MakeT1.a`` forces ``T1.q``.
        """
        text = assert_rejected_as_a_cycle(
            {
                **value_domain("T1", "q", "T2", default={}),
                **value_domain("T2", "r", "T1", default={"MakeT1": {}}),
                **maker("MakeT1", "a", "T1", {}, "T1"),
            }
        )
        assert "_defaultArgumentValues" in text, f"expected the declaration to be named: {text[:400]}"

    def test_a_self_cycle_through_a_function_argument(self):
        """``Selfy.p`` evaluates a Function whose unsupplied argument instantiates a `Selfy` again."""
        assert_rejected_as_a_cycle(
            {
                **value_domain("Selfy", "p", "Selfy", default={"MakeSelfy": {}}),
                **maker("MakeSelfy", "a", "Selfy", {}, "Selfy"),
            }
        )

    def test_a_cycle_through_two_functions(self):
        assert_rejected_as_a_cycle(
            {
                **value_domain("U1", "q", "U2", default={"MakeU2<U2>": {}}),
                **value_domain("U2", "r", "U1", default={"MakeU1<U1>": {}}),
                **maker("MakeU1", "a", "T", {}, "T", template=NUMERIC_FREE_T),
                **maker("MakeU2", "b", "T", {}, "T", template=NUMERIC_FREE_T),
            }
        )

    def test_a_finite_chain_through_an_unsupplied_argument_is_accepted(self):
        """
        The edge is real but the chain terminates: the default instantiates a `Leaf`, which forces nothing.
        The check must find cycles, not Function arguments.
        """
        check_concepts(
            {
                **LEAF,
                **value_domain("Holder", "h", "Leaf", default={"MakeLeaf": {}}),
                **maker("MakeLeaf", "a", "Leaf", {}, "Leaf"),
            }
        )

    def test_an_argument_with_no_default_forces_nothing(self):
        """An argument that is merely *required* contributes no edge; the call site has to write it."""
        check_concepts(
            {
                **LEAF,
                **value_domain("Holder", "h", "Leaf", default={"Identity": {"a": {}}}),
                "Identity": {
                    "directParents": ["FunctionReturning"],
                    "data": {
                        "templateContext": {"order": [], "substitution": {"FunctionReturning:T": "Leaf"}},
                        "interface": {"a": ["Leaf"], "res": "Leaf"},
                    },
                },
            }
        )

    def test_an_inherited_default_argument_closes_the_cycle_too(self):
        """
        The child declares no defaults at all; the cycle runs through one it inherits. The declared JSON is
        looked up over the Function's parents for exactly this.
        """
        assert_rejected_as_a_cycle(
            {
                **value_domain("V1", "q", "V2", default={}),
                **value_domain("V2", "r", "V1", default={"SubMake": {}}),
                **maker("BaseMake", "a", "V1", {}, "V1"),
                "SubMake": {
                    "directParents": ["BaseMake"],
                    "data": {"templateContext": {"order": []}, "interface": {}},
                },
            }
        )

    def test_the_diagnosis_says_what_repeats(self):
        text = assert_rejected_as_a_cycle(
            {
                **value_domain("C1", "q", "C2", default={}),
                **value_domain("C2", "r", "C1", default={"MakeW<C1>": {}}),
                **maker("MakeW", "a", "T", {}, "T", template=NUMERIC_FREE_T),
            }
        )
        assert "materialising it requires materialising it again" in text, text[:400]


# --------------------------------------------------------------------------------------------------
# 4. Template-dependent defaults: cyclicity is a property of the ground application
# --------------------------------------------------------------------------------------------------


LEAF2 = {
    "Leaf2": {
        "directParents": ["ValueDomain"],
        "data": {"instantiation": {"type": "object", "additionalProperties": False}},
    }
}


def pick(fallback_type: str) -> dict:
    """
    Two instantiation schemas chosen by the ground argument: ``Pick<Leaf>`` takes the first, anything else
    takes the fallback. Which schema applies -- and so which default site exists at all -- is a property of
    the application, not of the concept.
    """
    return {
        "Pick": {
            "directParents": ["ValueDomain"],
            "data": {
                "templateContext": ["Q"],
                "instantiation": [
                    [
                        ["Leaf"],
                        {"type": "object", "properties": {"v": {"type": "Leaf", "default": {}}}, "required": ["v"]},
                    ],
                    [
                        [""],
                        {
                            "type": "object",
                            "properties": {"v": {"type": fallback_type, "default": {}}},
                            "required": ["v"],
                        },
                    ],
                ],
            },
        }
    }


"""Two instantiation schemas; which applies depends on the ground argument."""

WRAP_HIERARCHY = {
    **LEAF,
    # Wrap<Q>'s default site has type Q -- what it forces depends entirely on what Q is bound to.
    "Wrap": {
        "directParents": ["ValueDomain"],
        "data": {
            "templateContext": ["Q"],
            "instantiation": {"type": "object", "properties": {"v": {"type": "Q", "default": {}}}, "required": ["v"]},
        },
    },
    # Q = Leaf: Wrap<Leaf>'s default instantiates a Leaf and stops. Benign.
    **value_domain("UsesLeaf", "u", "Wrap<Leaf>", default={}),
    # Q = Loop: Wrap<Loop>'s default instantiates a Loop, which forces a Wrap<Loop>, ... Cyclic.
    **value_domain("Loop", "w", "Wrap<Loop>", default={}),
}
"""
One hierarchy, one template-dependent default site (``Wrap.v``), reached under two different ground
applications with two different answers.
"""


class TestTemplateDependentCyclesAreApplicationSpecific:
    """
    A template-dependent default site cannot be classified on its own. ``Wrap<Q>``'s default has type
    ``Q``; under ``Q = Leaf`` it forces nothing, under ``Q = Loop`` it closes a cycle. The site is the
    *same* schema node in both cases, so a graph whose nodes are schema-node locations conflates the two
    and must give a wrong answer for one of them.

    The graph's nodes therefore have to be ``(default site, ground type application)`` pairs, and the
    check has to run per ground application -- the same "definition-time edges, call-site-time nodes"
    split that ``_validate_acyclic_default_argument_dependencies`` already uses for Function evaluations.

    Since stage 2 the site is expanded *and* cycle-checked per application, so the benign application
    checks and the cyclic one is rejected -- naming the application, not the shared declaration.
    """

    def test_template_dependent_default_site_is_expanded_per_application(self):
        """
        The positive form of the whole section: under ``Wrap<Leaf>``, ``Wrap.v``'s ``Q``-typed default
        must actually be expanded, and expanded *as a Leaf* -- not left deferred, and not resolved against
        the unsubstituted ``Q``.

        ``UsesLeaf.u``'s default ``{}`` instantiates a ``Wrap<Leaf>``, whose required ``v`` is absent and
        so is materialised from ``Wrap.v``. That materialised leaf is where the per-application result has
        to land: on the substituted schema copy, keyed by this application. Asserting on
        ``default_sites(context, "Wrap")["v"]`` instead would be asserting on the *declared* node, which
        legitimately stays template-dependent -- it is one site with one entry and two applications.
        """
        context = check_concepts(
            {
                **LEAF,
                "Wrap": WRAP_HIERARCHY["Wrap"],
                **value_domain("UsesLeaf", "u", "Wrap<Leaf>", default={}),
            }
        )
        leaves = materialised_leaves(default_sites(context, "UsesLeaf")["u"])
        assert [str(leaf.location_id[-1]) for leaf in leaves] == ["v"], (
            f"expected Wrap<Leaf>'s `v` to be materialised from its default; got {leaves!r}"
        )
        leaf = leaves[0]
        assert leaf.expression is not None, "the materialised default was never expanded for this application"
        assert leaf.custom_type.full_name == "Leaf", (
            f"expected the site's type substituted to Leaf under this application, got {leaf.custom_type}"
        )

    def test_benign_ground_application_keeps_checking(self):
        """
        ``Wrap<Leaf>`` is well-formed and must stay accepted. This is the over-rejection guard: a check
        that keys its graph by schema-node location alone will reject this hierarchy because the *same*
        ``Wrap.v`` site is cyclic under ``Q = Loop``.
        """
        context = check_concepts(
            {
                **LEAF,
                "Wrap": WRAP_HIERARCHY["Wrap"],
                **value_domain("UsesLeaf", "u", "Wrap<Leaf>", default={}),
            }
        )
        assert "UsesLeaf" in context.model.value_domains

    def test_cyclic_ground_application_is_rejected(self):
        """``Wrap<Loop>`` closes the cycle ``Loop.w -> Wrap<Loop>.v -> Loop.w`` and must be rejected."""
        with pytest.raises(ConceptHierarchyError):
            check_concepts(
                {
                    **LEAF,
                    "Wrap": WRAP_HIERARCHY["Wrap"],
                    **value_domain("Loop", "w", "Wrap<Loop>", default={}),
                }
            )

    def test_cyclic_application_is_rejected_without_condemning_the_benign_one(self):
        """
        Both applications in one hierarchy. The cyclic one must be reported; the report must name
        ``Wrap<Loop>``, not ``Wrap`` -- naming the site alone would implicate ``Wrap<Leaf>`` too.
        """
        with pytest.raises(ConceptHierarchyError, match="Loop"):
            check_concepts(WRAP_HIERARCHY)

    def test_the_benign_constraint_group_alone_keeps_checking(self):
        """
        The split one level up: which instantiation schema applies is chosen per ground application, so
        two applications of ``Pick`` do not even share a default site. With both fallbacks benign, both
        applications must check -- and each must take its own group.
        """
        context = check_concepts(
            {
                **LEAF,
                **LEAF2,
                **pick("Leaf2"),
                **value_domain("UsesPickLeaf", "u", "Pick<Leaf>", default={}),
                **value_domain("UsesPickOther", "o", "Pick<Integer>", default={}),
            }
        )
        assert "UsesPickLeaf" in context.model.value_domains
        assert "UsesPickOther" in context.model.value_domains

    def test_the_cyclic_constraint_group_is_rejected_without_blaming_the_benign_one(self):
        """
        Adding ``Cyc`` makes ``Pick<Cyc>`` cyclic -- ``Cyc.c -> Pick<Cyc>.v -> Cyc.c`` -- so the hierarchy
        must be rejected. The report has to name the cyclic *application*: the two groups share the concept
        ``Pick``, and ``Pick<Leaf>`` is perfectly well-formed.
        """
        with pytest.raises(ConceptHierarchyError) as excinfo:
            check_concepts(
                {
                    **LEAF,
                    **pick("Cyc"),
                    **value_domain("Cyc", "c", "Pick<Cyc>", default={}),
                    **value_domain("UsesPickLeaf", "u", "Pick<Leaf>", default={}),
                }
            )
        text = str(excinfo.value)
        assert "Cyc" in text, "the cyclic application must be named"
        assert "Pick<Leaf>" not in text, "the benign application must not be implicated"


# --------------------------------------------------------------------------------------------------
# Builders for the adversarial cases below
# --------------------------------------------------------------------------------------------------


def schema_vd(name: str, instantiation: object, parents: tuple = ("ValueDomain",)) -> dict:
    """A ValueDomain with an arbitrary instantiation schema."""
    return {name: {"directParents": list(parents), "data": {"instantiation": instantiation}}}


def closed_obj(properties: dict, required: list | None = None) -> dict:
    """
    An object schema that forbids additional properties.

    ``additionalProperties`` matters more than it looks here: left open, ``{"Sub": {}}`` matches the
    *parent* type directly as an instantiation with an ignored extra key, so a `Narrow` that would have
    cycled is never attempted. Several of these cases only bite with the schema closed.
    """
    schema: dict = {"type": "object", "properties": properties, "additionalProperties": False}
    schema["required"] = list(properties) if required is None else required
    return schema


MAKE_INT = {
    "MakeInt": {
        "directParents": ["FunctionReturning"],
        "data": {
            "templateContext": {"order": [], "substitution": {"FunctionReturning:T": "Integer"}},
            "interface": {"seed": ["D"], "res": "Integer"},
        },
    }
}
"""A Function taking a ``D``, used to reach back into a schema from inside one of its own defaults."""


# --------------------------------------------------------------------------------------------------
# Cycles through each structural keyword
# --------------------------------------------------------------------------------------------------


class TestCyclesThroughEveryStructuralKeyword:
    """
    The expansion follows whatever the value parser follows, so a cycle can close through any keyword that
    descends into a value -- not only ``properties``.
    """

    def test_through_array_items(self):
        assert_rejected_as_a_cycle(
            {
                **schema_vd("Arr", {"type": "array", "items": {"type": "ArrHolder"}}),
                **schema_vd("ArrHolder", closed_obj({"h": {"type": "Arr", "default": [{}]}})),
            }
        )

    def test_through_additional_properties(self):
        assert_rejected_as_a_cycle(
            {
                **schema_vd("Map", {"type": "object", "additionalProperties": {"type": "MapHolder"}}),
                **schema_vd("MapHolder", closed_obj({"h": {"type": "Map", "default": {"k": {}}}})),
            }
        )

    def test_through_pattern_properties(self):
        assert_rejected_as_a_cycle(
            {
                **schema_vd("PP", {"type": "object", "patternProperties": {"^k": {"type": "PPHolder"}}}),
                **schema_vd("PPHolder", closed_obj({"h": {"type": "PP", "default": {"k1": {}}}})),
            }
        )

    def test_through_contains(self):
        assert_rejected_as_a_cycle(
            {
                **schema_vd(
                    "Con", {"type": "array", "items": {"type": "ConHolder"}, "contains": {"type": "ConHolder"}}
                ),
                **schema_vd("ConHolder", closed_obj({"h": {"type": "Con", "default": [{}]}})),
            }
        )

    def test_through_all_of(self):
        assert_rejected_as_a_cycle(
            {
                **schema_vd("All", {"allOf": [closed_obj({"c": {"type": "All", "default": {}}})]}),
                **schema_vd("UseAll", closed_obj({"u": {"type": "All", "default": {}}})),
            }
        )

    def test_through_the_taken_any_of_branch(self):
        assert_rejected_as_a_cycle(
            {
                **schema_vd("Any", {"anyOf": [closed_obj({"c": {"type": "Any", "default": {}}}), {"type": "integer"}]}),
                **schema_vd("UseAny", closed_obj({"u": {"type": "Any", "default": {}}})),
            }
        )

    def test_through_one_of(self):
        assert_rejected_as_a_cycle(
            {
                **schema_vd(
                    "One",
                    {
                        "oneOf": [
                            closed_obj({"c": {"type": "One", "default": {}}}),
                            {"type": "string", "pattern": "^s:"},
                        ]
                    },
                ),
                **schema_vd("UseOne", closed_obj({"u": {"type": "One", "default": {}}})),
            }
        )

    def test_through_a_then_branch(self):
        assert_rejected_as_a_cycle(
            {
                **schema_vd(
                    "Cond",
                    {
                        "type": "object",
                        "if": {"type": "object"},
                        "then": closed_obj({"c": {"type": "Cond", "default": {}}}),
                        "properties": {},
                        "additionalProperties": True,
                    },
                ),
                **schema_vd("UseCond", closed_obj({"u": {"type": "Cond", "default": {}}})),
            }
        )

    def test_through_a_narrow_key(self):
        assert_rejected_as_a_cycle(
            schema_vd("SelfNarrow", closed_obj({"n": {"type": "ValueDomain", "default": {"SelfNarrow": {}}}})),
            "SelfNarrow",
        )

    def test_through_a_narrow_to_a_subtype(self):
        """
        ``Sub.s`` is declared as the *parent* type and defaults to a `Narrow` back to ``Sub``. This only
        closes when ``Base`` forbids extra properties -- otherwise ``{"Sub": {}}`` is a valid ``Base``
        instantiation with an ignored key, and the `Narrow` is never reached. See :func:`closed_obj`.
        """
        assert_rejected_as_a_cycle(
            {
                **LEAF,
                **schema_vd("Base", closed_obj({"b": {"type": "Leaf", "default": {}}})),
                **schema_vd("Sub", closed_obj({"s": {"type": "Base", "default": {"Sub": {}}}}), parents=("Base",)),
            },
            "Sub",
        )


# --------------------------------------------------------------------------------------------------
# Reached through supplied values, not just defaults
# --------------------------------------------------------------------------------------------------


class TestCyclesReachedThroughSuppliedValues:
    """
    A default's *supplied* content is parsed like any other value, so the keys it omits materialise their
    own defaults. A cycle can therefore be entered from a value that is written out, not only from a
    default that stands alone.
    """

    def test_a_supplied_sub_value_enters_the_cycle(self):
        assert_rejected_as_a_cycle(
            {
                **value_domain("A", "a", "B", default={}),
                **value_domain("B", "b", "A", default={}),
                **value_domain("Holder", "h", "A", default={"a": {"b": {}}}),
            }
        )

    def test_supplying_a_key_cannot_break_a_cycle_further_down(self):
        """Unrolling the value one level does not help: the innermost omission still closes the loop."""
        assert_rejected_as_a_cycle(
            {
                **value_domain("Self", "p", "Self", default={}),
                **value_domain("Use", "u", "Self", default={"p": {"p": {}}}),
            },
            "Self",
        )

    def test_a_global_variable_can_enter_the_cycle(self):
        """Not only instantiation defaults reach the expansion -- an instance's value does too."""
        with pytest.raises(ConceptHierarchyError) as excinfo:
            check_hierarchy(
                build_hierarchy(
                    {
                        **value_domain("G1", "g", "G2", default={}),
                        **value_domain("G2", "g", "G1", default={}),
                    },
                    instances={"v": {"G1": {}}},
                )
            )
        assert "can never be applied" in str(excinfo.value)


# --------------------------------------------------------------------------------------------------
# Sibling resolution order
# --------------------------------------------------------------------------------------------------


class TestSiblingResolutionOrder:
    """
    Resolving one default site can materialise a *sibling* of the same schema, and the order the sites
    happen to be walked in must not decide the verdict.

    ``D.a``'s default is a Function evaluation whose argument is a ``D`` that supplies ``a`` and omits
    ``b`` -- so resolving ``a`` materialises ``b``, which has not been resolved yet. The value is finite
    (``seed`` supplies ``a``, ``b`` defaults to 0) and the hierarchy must be accepted. A first
    implementation stamped every site with a cycle marker up front and rejected this outright.
    """

    def _hierarchy(self, properties: dict) -> dict:
        return {
            **MAKE_INT,
            **schema_vd("D", closed_obj(properties)),
            **schema_vd("Use", closed_obj({"u": {"type": "D", "default": {}}})),
        }

    A_FIRST = {
        "a": {"type": "Integer", "default": {"MakeInt": {"seed": {"a": 1}}}},
        "b": {"type": "Integer", "default": 0},
    }
    B_FIRST = {
        "b": {"type": "Integer", "default": 0},
        "a": {"type": "Integer", "default": {"MakeInt": {"seed": {"a": 1}}}},
    }

    def test_the_sibling_is_reached_before_it_is_resolved(self):
        assert "Use" in check_concepts(self._hierarchy(self.A_FIRST)).model.value_domains

    def test_the_verdict_does_not_depend_on_declaration_order(self):
        assert "Use" in check_concepts(self._hierarchy(self.B_FIRST)).model.value_domains

    def test_a_chain_of_siblings_each_needing_the_next(self):
        """
        The worst shape for the retry: site ``p1`` needs ``p2``, ``p2`` needs ``p3``, and so on, so each
        pass repairs exactly one site and the whole chain takes as many passes as there are sites.

        Every one of them is finite, so the hierarchy must be accepted. This is also the shape that makes
        the retry expensive -- it costs O(sites^2) reparses -- which is why the guard wants replacing with
        a path-aware one rather than leaving the retry to sort it out.
        """
        sites = 8
        properties = {}
        for k in range(1, sites + 1):
            omitted = f"p{k + 1}"
            seed = {f"p{i}": 0 for i in range(1, sites + 1) if f"p{i}" != omitted}
            properties[f"p{k}"] = {"type": "Integer", "default": {"MakeD": {"seed": seed}}}
        make_d = {
            "MakeD": {
                "directParents": ["FunctionReturning"],
                "data": {
                    "templateContext": {"order": [], "substitution": {"FunctionReturning:T": "Integer"}},
                    "interface": {"seed": ["D"], "res": "Integer"},
                },
            }
        }
        context = check_concepts({**make_d, **schema_vd("D", closed_obj(properties))})
        assert "D" in context.model.value_domains

    def test_the_sibling_case_still_holds_with_no_consumer(self):
        """Without ``Use`` nothing materialises ``D.a``, so the failure was latent rather than absent."""
        context = check_concepts({**MAKE_INT, **schema_vd("D", closed_obj(self.A_FIRST))})
        assert "D" in context.model.value_domains


# --------------------------------------------------------------------------------------------------
# Benign hierarchies that must not be rejected
# --------------------------------------------------------------------------------------------------


class TestBenignHierarchiesAreNotRejected:
    """The over-rejection guards. Each of these expands, and each of them terminates."""

    def test_a_deep_acyclic_chain(self):
        context = check_concepts(
            {
                **LEAF,
                **value_domain("L5", "e", "Leaf", default={}),
                **value_domain("L4", "d", "L5", default={}),
                **value_domain("L3", "c", "L4", default={}),
                **value_domain("L2", "b", "L3", default={}),
                **value_domain("L1", "a", "L2", default={}),
                **value_domain("L0", "z", "L1", default={}),
            }
        )
        # Every level must actually be an instantiation of the right type, all the way to the Leaf --
        # "the hierarchy checks" would also be true of a chain that quietly stopped resolving half way.
        top = default_sites(context, "L0")["z"]
        expected = [("z", "L1"), ("a", "L2"), ("b", "L3"), ("c", "L4"), ("d", "L5"), ("e", "Leaf")]
        assert_expression_is(top, InstExpression, value_type="L1")
        current = top
        for key, value_type in expected[1:]:
            leaf = leaf_at(current, key)
            assert leaf.used_default, f"{key!r} must be materialised from its default"
            current = step_into(current, key)
            assert_expression_is(current, InstExpression, value_type=value_type)

    def test_a_diamond_reaching_one_site_by_two_paths(self):
        context = check_concepts(
            {
                **LEAF,
                **value_domain("Shared", "s", "Leaf", default={}),
                **value_domain("L", "l", "Shared", default={}),
                **value_domain("R", "r", "Shared", default={}),
                **schema_vd(
                    "Top",
                    closed_obj({"x": {"type": "L", "default": {}}, "y": {"type": "R", "default": {}}}),
                ),
            }
        )
        assert "Top" in context.model.value_domains

    def test_a_self_typed_property_without_a_default(self):
        """No default means nothing is materialised, so self-reference is perfectly fine."""
        context = check_concepts(
            {
                **LEAF,
                **schema_vd(
                    "Tree",
                    closed_obj({"kid": {"type": "Tree"}, "leaf": {"type": "Leaf", "default": {}}}, required=["leaf"]),
                ),
                **value_domain("UseTree", "u", "Tree", default={"leaf": {}}),
            }
        )
        assert "UseTree" in context.model.value_domains

    def test_the_same_application_used_at_two_sites(self):
        context = check_concepts(
            {
                **LEAF,
                **value_domain("Twice", "m", "Leaf", default={}),
                **schema_vd(
                    "UsesBoth",
                    closed_obj({"x": {"type": "Twice", "default": {}}, "y": {"type": "Twice", "default": {}}}),
                ),
            }
        )
        assert "UsesBoth" in context.model.value_domains

    def test_an_any_of_branch_that_is_never_taken_by_the_consumer(self):
        """
        The consumer takes the integer branch, but ``Any2`` still *declares* a default that can never be
        materialised, and a declared default that cannot be applied is reported wherever it is declared.
        """
        assert_rejected_as_a_cycle(
            {
                **schema_vd(
                    "Any2", {"anyOf": [{"type": "integer"}, closed_obj({"c": {"type": "Any2", "default": {}}})]}
                ),
                **value_domain("UseAny2", "u", "Any2", default=3),
            }
        )


# --------------------------------------------------------------------------------------------------
# What the diagnosis says
# --------------------------------------------------------------------------------------------------


class TestCycleDiagnosis:
    def test_the_site_and_the_application_are_both_named(self):
        text = assert_rejected_as_a_cycle(
            {
                **value_domain("Test1", "p1", "Test2", default={}),
                **value_domain("Test2", "p2", "Test1", default={}),
            }
        )
        assert '"p2"' in text, "the failing site must be named"
        assert "Test2" in text, "and the application it belongs to"

    def test_the_diagnosis_survives_several_levels_of_nesting(self):
        text = assert_rejected_as_a_cycle(
            {
                **value_domain("A", "a", "B", default={}),
                **value_domain("B", "b", "C", default={}),
                **value_domain("C", "c", "A", default={}),
            }
        )
        assert text.count("can never be applied") >= 1
        assert "levels deep" not in text

    def test_two_independent_cycles_are_still_reported(self):
        assert_rejected_as_a_cycle(
            {
                **value_domain("A1", "a", "A2", default={}),
                **value_domain("A2", "a", "A1", default={}),
                **value_domain("B1", "b", "B2", default={}),
                **value_domain("B2", "b", "B1", default={}),
            }
        )


class TestHardErrorsDuringResolution:
    """
    The resolved schema is published to the cache *before* its defaults are resolved, so a resolution that
    raises leaves a half-resolved entry behind. Nothing may then read that entry and blame a cycle for it.
    """

    def test_a_hard_error_is_reported_as_itself(self):
        add = {
            "Add": {
                "directParents": ["FunctionReturning"],
                "data": {
                    "templateContext": {"order": ["T"], "T": "Numeric", "substitution": {"FunctionReturning:T": "T"}},
                    "interface": {"arg1": ["T"], "arg2": ["T"], "res": "T"},
                },
            }
        }
        concepts = {
            **LEAF,
            **add,
            **schema_vd(
                "D",
                closed_obj(
                    {
                        # An unknown Function argument raises rather than returning an ill-formed value,
                        # and does so half way through resolving D's default sites.
                        "a": {"type": "Integer", "default": {"Add<Integer>": {"nope": 1}}},
                        "b": {"type": "Leaf", "default": {}},
                    }
                ),
            ),
            **value_domain("Use", "u", "D", default={}),
        }
        with pytest.raises(ConceptHierarchyError) as excinfo:
            check_concepts(concepts)
        text = str(excinfo.value)
        assert "does not have the argument" in text, "the real cause must survive"
        assert "can never be applied" not in text, "a half-resolved cache entry must not be read as a cycle"
