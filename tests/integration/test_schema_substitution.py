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
Integration tests: substituting a ground type application into an instantiation schema.

A ValueDomain's instantiation schema is written in terms of its own template variables, so nothing in it
is decided until a ground application picks values for them. ``expression_parser.substitute_schema``
substitutes those values into every custom-type node, and ``resolve_substituted_defaults`` then reparses
every ``default`` against the substituted type -- reparses rather than rewrites, because an expression's
shape is decided by its type.

**A template variable can be named in more places than the default's own type**, and that is what these
tests are organised around. It can be the site's type, nested inside it, the key of a Function evaluation
or a Narrow, or buried in a value the default *supplies* -- an object property, an array item, an
``additionalProperties`` entry, a Function argument -- at any depth. The mapping therefore has to be
carried through the expression parser *and* across the ``ValueInstantiationContext`` seam into
``parse_value``. An earlier version of this file tested only the first of those, and missed that the
seam-crossing was never wired up.

See ``documentation/TODO_DEFAULT_EXPANSION_CYCLES.md`` §12 and §15.
"""

from __future__ import annotations

import pytest

from concept_hierarchy.data.expressions.expression import Expression
from concept_hierarchy.data.expressions.instantiated_value import ParsedCustomValue
from concept_hierarchy.data.expressions.subexpressions import (
    FunctionEvaluation,
    InstExpression,
    TemplateDependentExpression,
)
from concept_hierarchy.definitions.concept_hierarchy import ConceptHierarchyDefinition
from concept_hierarchy.errors import CHSemanticError, CHSyntaxError, ConceptHierarchyError
from tests.integration.test_expression_parsing import build_hierarchy, check_concepts, check_hierarchy

DEPTH_KEY = ConceptHierarchyDefinition.metadata_expansion_depth_limit_for_default_instantiation_expressions

# --------------------------------------------------------------------------------------------------
# Hierarchy fragments
# --------------------------------------------------------------------------------------------------

LEAF = {
    "Leaf": {
        "directParents": ["ValueDomain"],
        "data": {"instantiation": {"type": "object", "additionalProperties": False}},
    }
}
LEAF2 = {
    "Leaf2": {
        "directParents": ["ValueDomain"],
        "data": {"instantiation": {"type": "object", "additionalProperties": False}},
    }
}
BOX = {
    "Box": {
        "directParents": ["ValueDomain"],
        "data": {
            "templateContext": ["P"],
            "instantiation": {"type": "object", "properties": {"b": {"type": "P"}}, "required": ["b"]},
        },
    }
}
ADD = {
    "Add": {
        "directParents": ["FunctionReturning"],
        "data": {
            "templateContext": {"order": ["T"], "T": "Numeric", "substitution": {"FunctionReturning:T": "T"}},
            "interface": {"arg1": ["T"], "arg2": ["T"], "res": "T"},
        },
    }
}
NUMERIC_T = {"order": ["T"], "T": "Numeric"}
ADD_T = {"Add<T>": {"arg1": 1, "arg2": 2}}
"""A Function evaluation naming the enclosing concept's `T` in its key."""


def vd(name: str, instantiation: object, template: object = None, parents: tuple = ("ValueDomain",)) -> dict:
    data: dict = {"instantiation": instantiation}
    if template is not None:
        data["templateContext"] = template
    return {name: {"directParents": list(parents), "data": data}}


def obj(properties: dict, required: list | None = None) -> dict:
    schema: dict = {"type": "object", "properties": properties}
    schema["required"] = list(properties) if required is None else required
    return schema


def uses(applied: str, name: str = "Uses", prop: str = "u", default: object = None) -> dict:
    """A non-templated ValueDomain whose only property defaults into ``applied``."""
    return vd(name, obj({prop: {"type": applied, "default": {} if default is None else default}}))


def uses_feval(applied_function: str, arguments: dict) -> dict:
    """A ValueDomain whose only property defaults to a Function evaluation of ``applied_function``."""
    return vd("Site", obj({"p": {"type": "Integer", "default": {applied_function: arguments}}}))


def holder(default: object, site_type: str = "T", template: object = None) -> dict:
    """A templated ValueDomain whose single default site carries ``default``."""
    return vd("H", obj({"h": {"type": site_type, "default": default}}), NUMERIC_T if template is None else template)


# --------------------------------------------------------------------------------------------------
# Accessors and invariants
# --------------------------------------------------------------------------------------------------


def default_site_expression(context, value_domain_name: str, prop: str):
    for _constraint, schema in context.model.value_domains[value_domain_name].instantiation:
        for node in schema.walk():
            if node.has_default and str(node.location_id[-2]) == prop:
                return node.parsed_default_expr
    raise AssertionError(f"no default site {prop!r} on {value_domain_name}")


def parsed_custom_values(expression) -> list[ParsedCustomValue]:
    """Every custom-type leaf reachable from ``expression``, across the Expression/ParsedValue alternation."""
    found: list[ParsedCustomValue] = []
    stack, seen = [expression], set()
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        value = current.value
        if isinstance(value, InstExpression) and value.value is not None:
            for node in value.value.walk():
                if isinstance(node, ParsedCustomValue):
                    found.append(node)
                    stack.append(node.expression)
        elif isinstance(value, FunctionEvaluation):
            stack.extend(value.arguments.values())
    return found


def materialised_leaf(expression) -> ParsedCustomValue:
    leaves = [n for n in parsed_custom_values(expression) if n.used_default]
    assert len(leaves) >= 1, f"expected a materialised default, got {leaves!r}"
    return leaves[0]


def assert_fully_grounded(context, value_domain_name: str, prop: str) -> None:
    """
    No custom-type leaf reachable from this default may still mention a template variable.

    ``ParsedCustomValue.schema_node`` is the *substituted* node, so this checks the actual result of
    substitution rather than the declaration -- and it holds however deeply the variable was buried.
    """
    expression = default_site_expression(context, value_domain_name, prop)
    for node in parsed_custom_values(expression):
        assert not node.custom_type.depends_on_templates, (
            f"{node.custom_type} at {node.location_id} was left template-dependent"
        )


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


# --------------------------------------------------------------------------------------------------
# A. Every position a template variable can occupy
# --------------------------------------------------------------------------------------------------


class TestWhereTheTemplateVariableCanAppear:
    """
    Each case names ``T``/``Q`` in a different position and checks that grounding reaches it. The last six
    are the ones that go through the ``ValueInstantiationContext`` seam.
    """

    def test_the_site_type_is_the_bare_variable(self):
        context = check_concepts(
            {**LEAF, **vd("W", obj({"v": {"type": "Q", "default": {}}}), ["Q"]), **uses("W<Leaf>")}
        )
        assert_fully_grounded(context, "Uses", "u")
        top = default_site_expression(context, "Uses", "u")
        assert_expression_is(top, InstExpression, value_type="W<Leaf>")
        leaf = leaf_at(top, "v")
        assert leaf.used_default, "`v` must come from its default, not from the supplied value"
        assert leaf.custom_type.full_name == "Leaf"
        assert_expression_is(step_into(top, "v"), InstExpression, value_type="Leaf")

    def test_the_variable_is_nested_in_the_site_type(self):
        context = check_concepts(
            {**LEAF, **BOX, **vd("W", obj({"v": {"type": "Box<Q>", "default": {"b": {}}}}), ["Q"]), **uses("W<Leaf>")}
        )
        assert_fully_grounded(context, "Uses", "u")

    def test_the_variable_is_a_function_evaluation_key(self):
        context = check_concepts({**ADD, **holder(ADD_T), **uses("H<Integer>")})
        top = default_site_expression(context, "Uses", "u")
        feval = step_into(top, "h")
        assert_expression_is(feval, FunctionEvaluation, value_type="Integer")
        assert feval.value.f_type.full_name == "Add<Integer>", "the `T` in the FEval key must be grounded"
        # ...and both of its arguments are grounded Integer instantiations, not deferred values.
        for argument in ("arg1", "arg2"):
            assert_expression_is(step_into(feval, argument), InstExpression, value_type="Integer")

    def test_the_variable_is_a_narrow_key(self):
        context = check_concepts(
            {
                **LEAF,
                **BOX,
                **vd("W", obj({"v": {"type": "ValueDomain", "default": {"Box<Q>": {"b": {}}}}}), ["Q"]),
                **uses("W<Leaf>"),
            }
        )
        assert_fully_grounded(context, "Uses", "u")

    def test_the_variable_is_inside_a_supplied_object_property(self):
        """The case the first implementation missed entirely: it only reaches the seam."""
        pair = vd("Pair", obj({"a": {"type": "Integer"}}))
        context = check_concepts({**ADD, **pair, **holder({"a": ADD_T}, site_type="Pair"), **uses("H<Integer>")})
        assert_fully_grounded(context, "Uses", "u")
        top = default_site_expression(context, "Uses", "u")
        # Uses.u -> Inst(H<Integer>) -> h -> Inst(Pair) -> a -> FEval(Add<Integer>), four levels down.
        assert_expression_is(top, InstExpression, value_type="H<Integer>")
        assert_expression_is(step_into(top, "h"), InstExpression, value_type="Pair")
        feval = step_into(step_into(top, "h"), "a")
        assert_expression_is(feval, FunctionEvaluation, value_type="Integer")
        assert feval.value.f_type.full_name == "Add<Integer>"
        assert not leaf_at(top, "h", "a").used_default, "`a` is supplied by the default, not materialised"

    def test_the_variable_is_inside_a_supplied_array_item(self):
        arr = vd("Arr", {"type": "array", "items": "Integer"})
        context = check_concepts({**ADD, **arr, **holder([ADD_T], site_type="Arr"), **uses("H<Integer>")})
        assert_fully_grounded(context, "Uses", "u")

    def test_the_variable_is_inside_an_additional_properties_value(self):
        mapping = vd("Map", {"type": "object", "additionalProperties": "Integer"})
        context = check_concepts({**ADD, **mapping, **holder({"k": ADD_T}, site_type="Map"), **uses("H<Integer>")})
        assert_fully_grounded(context, "Uses", "u")

    def test_the_variable_is_two_supplied_levels_down(self):
        inner = vd("In", obj({"i": {"type": "Integer"}}))
        outer = vd("Out", obj({"o": {"type": "In"}}))
        context = check_concepts(
            {**ADD, **inner, **outer, **holder({"o": {"i": ADD_T}}, site_type="Out"), **uses("H<Integer>")}
        )
        assert_fully_grounded(context, "Uses", "u")

    def test_the_variable_is_inside_a_function_argument_value(self):
        context = check_concepts({**ADD, **holder({"Add<T>": {"arg1": ADD_T, "arg2": 2}}), **uses("H<Integer>")})
        assert_fully_grounded(context, "Uses", "u")

    def test_the_variable_is_inside_a_narrows_supplied_value(self):
        pair = vd("Pair", obj({"a": {"type": "Integer"}}))
        context = check_concepts(
            {**ADD, **pair, **holder({"Pair": {"a": ADD_T}}, site_type="ValueDomain"), **uses("H<Integer>")}
        )
        assert_fully_grounded(context, "Uses", "u")


# --------------------------------------------------------------------------------------------------
# B. What the variable is substituted with
# --------------------------------------------------------------------------------------------------


class TestWhatTheVariableIsSubstitutedWith:
    def test_a_builtin_serialized_type(self):
        context = check_concepts({**vd("W", obj({"v": {"type": "Q", "default": 1}}), ["Q"]), **uses("W<Integer>")})
        assert materialised_leaf(default_site_expression(context, "Uses", "u")).custom_type.full_name == "Integer"

    def test_a_nested_application(self):
        context = check_concepts(
            {**LEAF, **BOX, **vd("W", obj({"v": {"type": "Q", "default": {"b": {}}}}), ["Q"]), **uses("W<Box<Leaf>>")}
        )
        leaf = materialised_leaf(default_site_expression(context, "Uses", "u"))
        assert leaf.custom_type.full_name == "Box<Leaf>"
        assert_fully_grounded(context, "Uses", "u")

    def test_two_distinct_template_variables(self):
        context = check_concepts(
            {
                **LEAF,
                **vd("P2", obj({"k": {"type": "K", "default": {}}, "v": {"type": "V", "default": 1}}), ["K", "V"]),
                **uses("P2<Leaf, Integer>"),
            }
        )
        grounded = {
            n.custom_type.full_name for n in parsed_custom_values(default_site_expression(context, "Uses", "u"))
        }
        assert {"Leaf", "Integer"} <= grounded

    def test_the_same_variable_used_at_two_sites(self):
        context = check_concepts(
            {
                **LEAF,
                **vd("Two", obj({"a": {"type": "Q", "default": {}}, "b": {"type": "Q", "default": {}}}), ["Q"]),
                **uses("Two<Leaf>"),
            }
        )
        assert_fully_grounded(context, "Uses", "u")

    def test_two_applications_of_one_site_are_grounded_independently(self):
        """One declared site, two applications: neither may leak into the other."""
        context = check_concepts(
            {
                **LEAF,
                **LEAF2,
                **vd("W", obj({"v": {"type": "Q", "default": {}}}), ["Q"]),
                **uses("W<Leaf>"),
                **uses("W<Leaf2>", name="UsesTwo", prop="t"),
            }
        )
        assert materialised_leaf(default_site_expression(context, "Uses", "u")).custom_type.full_name == "Leaf"
        assert materialised_leaf(default_site_expression(context, "UsesTwo", "t")).custom_type.full_name == "Leaf2"

    def test_the_declared_site_keeps_its_template_dependent_parse(self):
        """
        The per-application result belongs on the substituted copy. Overwriting the declaration would make
        the last application checked win, and the others silently wrong.
        """
        context = check_concepts(
            {**LEAF, **vd("W", obj({"v": {"type": "Q", "default": {}}}), ["Q"]), **uses("W<Leaf>")}
        )
        declared = default_site_expression(context, "W", "v")
        assert isinstance(declared.value, TemplateDependentExpression)


# --------------------------------------------------------------------------------------------------
# C. Substitution decides whether the default is valid at all
# --------------------------------------------------------------------------------------------------


class TestSubstitutionDecidesValidity:
    """
    A default only has to type-check under the applications that actually materialise it. Both directions
    matter: an unusable one must be reported, and a usable one must not be condemned for it.
    """

    def test_a_default_that_does_not_type_check_after_substitution_is_rejected(self):
        """``v: Q`` defaults to ``1``; under ``Q := String`` that is not a String expression."""
        with pytest.raises(ConceptHierarchyError) as excinfo:
            check_concepts({**vd("W", obj({"v": {"type": "Q", "default": 1}}), ["Q"]), **uses("W<String>")})
        text = str(excinfo.value)
        assert "String" in text
        assert "default" in text.lower()

    def test_supplying_the_key_instead_is_still_accepted(self):
        """
        The over-rejection guard. ``W<String>``'s default for ``v`` is unusable, but an instantiation that
        supplies ``v`` never materialises it, and must stay legal (D1: a supplied key is terminal).
        """
        context = check_concepts(
            {**vd("W", obj({"v": {"type": "Q", "default": 1}}), ["Q"]), **uses("W<String>", default={"v": "s:x"})}
        )
        assert "Uses" in context.model.value_domains

    def test_one_bad_application_does_not_condemn_a_good_one(self):
        with pytest.raises(ConceptHierarchyError):
            check_concepts(
                {
                    **vd("W", obj({"v": {"type": "Q", "default": 1}}), ["Q"]),
                    **uses("W<Integer>"),
                    **uses("W<String>", name="Bad", prop="b"),
                }
            )

    def test_the_good_application_alone_still_checks(self):
        context = check_concepts({**vd("W", obj({"v": {"type": "Q", "default": 1}}), ["Q"]), **uses("W<Integer>")})
        assert materialised_leaf(default_site_expression(context, "Uses", "u")).custom_type.full_name == "Integer"

    def test_the_constraint_group_is_chosen_per_application(self):
        pick = vd(
            "Pick",
            [
                [["Integer"], obj({"i": {"type": "Integer", "default": 0}})],
                [[""], obj({"o": {"type": "Leaf", "default": {}}})],
            ],
            ["Q"],
        )
        context = check_concepts({**LEAF, **pick, **uses("Pick<Integer>"), **uses("Pick<Leaf>", name="U2", prop="w")})
        assert materialised_leaf(default_site_expression(context, "Uses", "u")).custom_type.full_name == "Integer"
        assert materialised_leaf(default_site_expression(context, "U2", "w")).custom_type.full_name == "Leaf"


# --------------------------------------------------------------------------------------------------
# D. Robustness
# --------------------------------------------------------------------------------------------------


class TestRobustness:
    def test_a_templated_value_domain_that_is_never_applied_still_checks(self):
        """Its default stays deferred: there is no application to ground it against."""
        context = check_concepts({**LEAF, **vd("W", obj({"v": {"type": "Q", "default": {}}}), ["Q"])})
        assert isinstance(default_site_expression(context, "W", "v").value, TemplateDependentExpression)

    def test_a_templated_value_domain_with_no_defaults(self):
        context = check_concepts(
            {
                **LEAF,
                **vd("W", obj({"v": {"type": "Q"}}), ["Q"]),
                **vd("U", obj({"u": {"type": "W<Leaf>", "default": {"v": {}}}})),
            }
        )
        assert "U" in context.model.value_domains

    def test_a_non_templated_value_domain_is_untouched(self):
        context = check_concepts({**LEAF, **vd("N", obj({"n": {"type": "Leaf", "default": {}}})), **uses("N")})
        assert "Uses" in context.model.value_domains

    def test_the_declared_schema_is_not_mutated_by_substitution(self):
        """
        ``resolve_substituted_defaults`` mutates in place, and ``substitute_schema`` returns the *same*
        object when there is nothing to substitute. Grounding one application must therefore never write
        through to the declaration.
        """
        context = check_concepts(
            {**LEAF, **vd("W", obj({"v": {"type": "Q", "default": {}}}), ["Q"]), **uses("W<Leaf>")}
        )
        for _constraint, schema in context.model.value_domains["W"].instantiation:
            for node in schema.walk():
                if node.has_default:
                    assert node.custom_type.depends_on_templates, (
                        f"the declared site's type was overwritten with {node.custom_type}"
                    )


# --------------------------------------------------------------------------------------------------
# E. The depth limit
# --------------------------------------------------------------------------------------------------

GROWING = {
    # Nest<Q>'s only default site has type Nest<Box<Q>>, so every expansion step produces a *strictly
    # larger* application that no cache can serve: Nest<Leaf>, Nest<Box<Leaf>>, Nest<Box<Box<Leaf>>>, ...
    # No node ever repeats, so cycle detection never fires and only the depth bound stops it.
    "Box": {
        "directParents": ["ValueDomain"],
        "data": {
            "templateContext": ["P"],
            "instantiation": {"type": "object", "properties": {"b": {"type": "P", "default": {}}}, "required": ["b"]},
        },
    },
    "Nest": {
        "directParents": ["ValueDomain"],
        "data": {
            "templateContext": ["Q"],
            "instantiation": {
                "type": "object",
                "properties": {"n": {"type": "Nest<Box<Q>>", "default": {}}},
                "required": ["n"],
            },
        },
    },
    "Seed": {
        "directParents": ["ValueDomain"],
        "data": {
            "instantiation": {
                "type": "object",
                "properties": {"s": {"type": "Nest<Leaf>", "default": {}}},
                "required": ["s"],
            }
        },
    },
}

TERMINATING = {**LEAF, **BOX}
"""`Box<Leaf>` bottoms out immediately: one level of expansion, whatever the limit."""


def growing_hierarchy(metadata: dict | None = None) -> dict:
    data = build_hierarchy({**LEAF, **GROWING})
    if metadata is not None:
        data["metadata"] = metadata
    return data


def terminating_hierarchy(metadata: dict | None = None) -> dict:
    data = build_hierarchy({**TERMINATING, **uses("Box<Leaf>", default={"b": {}})})
    if metadata is not None:
        data["metadata"] = metadata
    return data


MAKE_D = {
    "MakeD": {
        "directParents": ["FunctionReturning"],
        "data": {
            "templateContext": {"order": [], "substitution": {"FunctionReturning:T": "Integer"}},
            "interface": {"seed": ["D"], "res": "Integer"},
        },
    }
}


def chained_hierarchy(sites: int, metadata: dict | None = None) -> dict:
    """
    One schema whose sites form a chain: resolving ``p1`` materialises ``p2``, which materialises ``p3``,
    and so on. Every value is finite, so the only thing that can stop it is the depth bound.

    This is the *other* way expansion nests, and the one that must **not** be bounded: `GROWING` generates
    a new application at each step, so its node set is unbounded, while this stays inside one schema whose
    sites are finite and each resolved once. Only the first can fail to terminate.
    """
    properties = {}
    for k in range(1, sites + 1):
        omitted = f"p{k + 1}"
        seed = {f"p{i}": 0 for i in range(1, sites + 1) if f"p{i}" != omitted}
        properties[f"p{k}"] = {"type": "Integer", "default": {"MakeD": {"seed": seed}}}
    data = build_hierarchy({**MAKE_D, **vd("D", obj(properties))})
    if metadata is not None:
        data["metadata"] = metadata
    return data


class TestDefaultExpansionDepthLimit:
    """
    The bound that cycle detection cannot replace. `GROWING` expands forever while never repeating a node,
    so nothing but the depth limit stops it.
    """

    @pytest.mark.parametrize("limit", ["1", "8", "1024"])
    def test_a_terminating_hierarchy_is_accepted_at_any_limit(self, limit):
        assert "Uses" in check_hierarchy(terminating_hierarchy({DEPTH_KEY: limit})).model.value_domains

    @pytest.mark.parametrize("limit", ["4", "12"])
    def test_an_unbounded_expansion_is_rejected_at_the_limit(self, limit):
        with pytest.raises(CHSemanticError) as excinfo:
            check_hierarchy(growing_hierarchy({DEPTH_KEY: limit}))
        text = str(excinfo.value)
        assert f"more than {limit} levels deep" in text
        assert DEPTH_KEY in text, "the message must name the key that raises the limit"
        assert "Nest<Box<" in text, "and the application it had grown to"

    def test_an_unbounded_expansion_is_stopped_even_at_the_default_limit(self):
        """
        At the shipped default of 1024 the interpreter's own stack runs out first — one expansion level
        costs a dozen-odd Python frames. That is converted into an ordinary error rather than a
        `RecursionError`, so the failure mode is the same either way.
        """
        with pytest.raises(CHSemanticError, match="Ran out of stack|levels deep"):
            check_hierarchy(growing_hierarchy())

    @pytest.mark.parametrize("bad", ["banana", "1.5", "", "-3", "0"])
    def test_a_non_positive_integer_limit_is_rejected(self, bad):
        with pytest.raises(CHSyntaxError, match="must be a positive integer"):
            check_hierarchy(terminating_hierarchy({DEPTH_KEY: bad}))

    def test_the_limit_error_is_located_at_the_metadata_entry(self):
        with pytest.raises(CHSyntaxError) as excinfo:
            check_hierarchy(terminating_hierarchy({DEPTH_KEY: "banana"}))
        assert f'"metadata": "{DEPTH_KEY}"' in str(excinfo.value)

    @pytest.mark.parametrize("limit", ["1", "3", "10"])
    def test_a_chain_within_one_schema_is_not_bounded_by_the_limit(self, limit):
        """
        A 20-deep chain of resolutions inside one schema is accepted even at a limit of 1, and must be.

        The limit exists to stop an expansion that cannot terminate, and this one **terminates by
        construction**: resolution visits each site at most once -- it is memoized the moment it is
        parsed, and re-entering one that is still on the path is reported as a cycle -- so the chain can
        be no longer than the number of default sites in the schemas already built. It is deep, not
        infinite, and a hierarchy is not invalid for being deep.
        """
        assert "D" in check_hierarchy(chained_hierarchy(20, {DEPTH_KEY: limit})).model.value_domains

    def test_the_same_limit_still_stops_a_growing_expansion(self):
        """
        The contrast that gives the limit its meaning. Both nest; only one can run away, because only one
        keeps *generating* applications, each bringing default sites of its own.
        """
        limit = {DEPTH_KEY: "4"}
        assert "D" in check_hierarchy(chained_hierarchy(20, limit)).model.value_domains
        with pytest.raises(CHSemanticError, match="more than 4 levels deep"):
            check_hierarchy(growing_hierarchy(limit))

    def test_a_chain_within_the_interpreter_stack_is_accepted_by_default(self):
        assert "D" in check_hierarchy(chained_hierarchy(20)).model.value_domains

    def test_a_chain_beyond_the_interpreter_stack_still_reports_cleanly(self):
        """
        The one thing that *does* stop a long chain, and it is not the limit: resolution recurses as deep
        as the chain, so past roughly 60 sites the interpreter's own stack runs out. This hierarchy is
        valid and is rejected anyway -- the open question in ``TODO_DEFAULT_EXPANSION_CYCLES.md`` §9. What
        is pinned here is only that it fails as an ordinary error rather than as a `RecursionError`.
        """
        with pytest.raises(CHSemanticError, match="Ran out of stack"):
            check_hierarchy(chained_hierarchy(80))

    def test_an_unrelated_metadata_entry_is_ignored(self):
        assert "Uses" in check_hierarchy(terminating_hierarchy({"version": "v1.0"})).model.value_domains


# --------------------------------------------------------------------------------------------------
# F. The mapping has to cross the ValueInstantiationContext seam
# --------------------------------------------------------------------------------------------------


class TestTheMappingCrossesTheSeam:
    """
    A value nested inside an instantiation schema is still text from the *enclosing* expression, so it can
    still name the enclosing concept's template variables however many schemas down it sits. The mapping
    therefore has to be carried across ``ValueInstantiationContext`` into ``parse_value`` and back out at
    each custom-type leaf -- not only applied to the default at the top.

    This is the regression guard for a substitution that was wired everywhere *except* the seam, so the
    plumbing existed and was never reached.
    """

    PAIR = {
        "Pair": {
            "directParents": ["ValueDomain"],
            "data": {"instantiation": {"type": "object", "properties": {"a": {"type": "Integer"}}, "required": ["a"]}},
        }
    }
    HOLDER = {
        "Holder": {
            "directParents": ["ValueDomain"],
            "data": {
                "templateContext": NUMERIC_T,
                "instantiation": {
                    "type": "object",
                    "properties": {"h": {"type": "Pair", "default": {"a": ADD_T}}},
                    "required": ["h"],
                },
            },
        }
    }

    def test_a_template_variable_inside_a_supplied_sub_value_is_grounded(self):
        """``Holder<T>``'s default supplies a ``Pair``, and ``T`` is named *inside* that supplied value."""
        context = check_concepts({**ADD, **self.PAIR, **self.HOLDER, **uses("Holder<Integer>")})
        assert "Uses" in context.model.value_domains

    def test_the_same_hierarchy_without_an_application_still_checks(self):
        """Negative control: with nothing grounding ``Holder``, its default stays deferred and unparsed."""
        context = check_concepts({**ADD, **self.PAIR, **self.HOLDER})
        assert "Holder" in context.model.value_domains


# --------------------------------------------------------------------------------------------------
# G. Function default *argument* expressions -- the same problem, not yet solved
# --------------------------------------------------------------------------------------------------


def function_with_default(name: str, defaults: dict, arg_type: str = "T") -> dict:
    return {
        name: {
            "directParents": ["FunctionReturning"],
            "data": {
                "templateContext": {"order": ["T"], "T": "Numeric", "substitution": {"FunctionReturning:T": "T"}},
                "interface": {"arg1": [arg_type], "arg2": [arg_type], "res": "T", "_defaultArgumentValues": defaults},
            },
        }
    }


class TestFunctionDefaultArgumentsAreGroundedAtTheCallSite:
    """
    A Function's ``_defaultArgumentValues`` had exactly the problem stage 1 solved for instantiation
    defaults: written in the Function's own template context, parsed once with ``T`` unbound, and only
    checkable when a call site supplies a ground application.

    Stage 2.5 closes that: a ground FEval reparses each *unsupplied* argument's declared default under its
    own template arguments and reports one that cannot hold. The result is checked and dropped -- the
    default is still not materialised into `FunctionEvaluation.arguments`, which continues to mean "what
    the call site wrote". See ``TODO_DEFAULT_EXPANSION_CYCLES.md`` §6.
    """

    def test_a_non_template_default_argument_is_checked_at_definition_time(self):
        """
        The control: with a ground argument type the default is checked where it is written, and the call
        site never gets a say. Only template-dependence defers it.
        """
        shout = {
            "Shout": {
                "directParents": ["FunctionReturning"],
                "data": {
                    "templateContext": {"order": [], "substitution": {"FunctionReturning:T": "Integer"}},
                    "interface": {
                        "what": ["Integer"],
                        "res": "Integer",
                        "_defaultArgumentValues": {"what": "s:not-a-number"},
                    },
                },
            }
        }
        with pytest.raises(CHSemanticError, match="Invalid expression"):
            check_concepts(shout)

    def test_a_template_dependent_default_argument_stays_deferred(self):
        context = check_concepts(
            {**function_with_default("Add2", {"arg2": 3}), **uses_feval("Add2<Integer>", {"arg1": 1})}
        )
        declared = context.model.functions["Add2"].evaluation_argument_default_value_expressions["arg2"]
        assert declared.is_value_template_dependent, "with T unbound the default can not be resolved yet"

    def test_an_unsupplied_argument_is_not_materialised_at_the_call_site(self):
        context = check_concepts(
            {**function_with_default("Add2", {"arg2": 3}), **uses_feval("Add2<Integer>", {"arg1": 1})}
        )
        feval = default_site_expression(context, "Site", "p").value
        assert isinstance(feval, FunctionEvaluation)
        assert sorted(feval.arguments) == ["arg1"]

    def test_grounding_a_default_argument_does_not_materialise_it(self):
        """
        The default of ``arg2`` is checked against ``Integer`` at this call site and then dropped.

        Materialising it would be actively wrong, not merely wasteful: the acyclicity check reads
        ``supplied_arguments`` off this very dict, so a materialised default would make every argument look
        supplied and switch that check off. `test_an_invalid_default_argument_is_rejected_at_a_ground_call_site`
        is what shows the check still ran despite nothing being kept.
        """
        context = check_concepts(
            {**function_with_default("Add2", {"arg2": 3}), **uses_feval("Add2<Integer>", {"arg1": 1})}
        )
        feval = default_site_expression(context, "Site", "p").value
        assert isinstance(feval, FunctionEvaluation)
        assert sorted(feval.arguments) == ["arg1"], "grounding a default must not add it to the arguments"

    def test_an_invalid_default_argument_is_rejected_at_a_ground_call_site(self):
        """
        ``arg2`` defaults to a String literal while ``T`` is ``Numeric``. Deferring at definition time is
        right -- ``T`` is unknown there -- but at ``BadAdd<Integer>`` it is definitely wrong.
        """
        with pytest.raises(ConceptHierarchyError):
            check_concepts(
                {
                    **function_with_default("BadAdd", {"arg2": "s:not-a-number"}),
                    **uses_feval("BadAdd<Integer>", {"arg1": 1}),
                }
            )


class TestPartiallySubstitutedSitesAreLeftUnresolved:
    """
    Only a site whose type is fully ground is registered for resolution, so one that is still
    template-dependent after substitution is never resolved -- it ends up with ``used_default=True`` and no
    expression, which is what `ParsedValue.unresolved_default_sites()` reports.

    That is the intended behaviour (there is no application to ground it against yet, D5), but it is the
    one route by which a materialised default legitimately has no value, so it is pinned rather than left
    to be rediscovered.
    """

    def test_a_site_that_stays_template_dependent_is_not_resolved(self):
        """
        ``Nest<Q>``'s site has type ``Nest<Box<Q>>``; substituting ``Q`` one level leaves
        ``Nest<Box<Box<Nest:Q>>>``, which still mentions a template variable.
        """
        with pytest.raises(CHSemanticError):
            check_hierarchy(growing_hierarchy({DEPTH_KEY: "4"}))

    def test_a_fully_ground_site_always_is_resolved(self):
        """The contrast: nothing else may leave a materialised default without an expression."""
        context = check_hierarchy(terminating_hierarchy())
        for value_domain in context.model.value_domains.values():
            for _constraint, schema in value_domain.instantiation:
                for node in schema.walk():
                    expression = node.parsed_default_expr
                    if expression is None or not isinstance(expression.value, InstExpression):
                        continue
                    if expression.value.value is None:
                        continue
                    unresolved = list(expression.value.value.unresolved_default_sites())
                    assert unresolved == [], f"{node.location_id} left {unresolved!r} unresolved"
