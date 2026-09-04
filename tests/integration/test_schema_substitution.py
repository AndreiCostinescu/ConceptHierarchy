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
substitutes those values into every custom-type node, and ``resolve_substituted_defaults`` then re-parses
every ``default`` against the substituted type -- re-parses rather than rewrites, because an expression's
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
        assert materialised_leaf(default_site_expression(context, "Uses", "u")).custom_type.full_name == "Leaf"

    def test_the_variable_is_nested_in_the_site_type(self):
        context = check_concepts(
            {**LEAF, **BOX, **vd("W", obj({"v": {"type": "Box<Q>", "default": {"b": {}}}}), ["Q"]), **uses("W<Leaf>")}
        )
        assert_fully_grounded(context, "Uses", "u")

    def test_the_variable_is_a_function_evaluation_key(self):
        context = check_concepts({**ADD, **holder(ADD_T), **uses("H<Integer>")})
        leaf = materialised_leaf(default_site_expression(context, "Uses", "u"))
        value = leaf.expression.value
        assert isinstance(value, FunctionEvaluation)
        assert value.f_type.full_name == "Add<Integer>"

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

MID = {
    "Mid": {
        "directParents": ["ValueDomain"],
        "data": {
            "templateContext": ["R"],
            "instantiation": {"type": "object", "properties": {"m": {"type": "R", "default": {}}}, "required": ["m"]},
        },
    }
}
TWO_LEVEL_WRAP = {
    "Wrap": {
        "directParents": ["ValueDomain"],
        "data": {
            "templateContext": ["Q"],
            "instantiation": {
                # `w` names a second templated application as a ground type, so expanding Wrap<Leaf>'s
                # defaults reaches a second level of substitution without nesting Q inside an application.
                "type": "object",
                "properties": {"v": {"type": "Q", "default": {}}, "w": {"type": "Mid<Leaf>", "default": {}}},
                "required": ["v", "w"],
            },
        },
    }
}


def two_level_hierarchy(metadata: dict | None = None) -> dict:
    data = build_hierarchy({**LEAF, **MID, **TWO_LEVEL_WRAP, **uses("Wrap<Leaf>")})
    if metadata is not None:
        data["metadata"] = metadata
    return data


class TestDefaultExpansionDepthLimit:
    def test_a_two_level_expansion_is_accepted_by_default(self):
        assert "Uses" in check_hierarchy(two_level_hierarchy()).model.value_domains

    @pytest.mark.parametrize("limit", ["2", "8", "1024"])
    def test_a_sufficient_limit_accepts_it(self, limit):
        assert "Uses" in check_hierarchy(two_level_hierarchy({DEPTH_KEY: limit})).model.value_domains

    def test_an_insufficient_limit_rejects_it(self):
        with pytest.raises(CHSemanticError) as excinfo:
            check_hierarchy(two_level_hierarchy({DEPTH_KEY: "1"}))
        text = str(excinfo.value)
        assert "more than 1 levels deep" in text
        assert DEPTH_KEY in text, "the message must name the key that raises the limit"

    @pytest.mark.parametrize("bad", ["banana", "1.5", "", "-3", "0"])
    def test_a_non_positive_integer_limit_is_rejected(self, bad):
        with pytest.raises(CHSyntaxError, match="must be a positive integer"):
            check_hierarchy(two_level_hierarchy({DEPTH_KEY: bad}))

    def test_the_limit_error_is_located_at_the_metadata_entry(self):
        with pytest.raises(CHSyntaxError) as excinfo:
            check_hierarchy(two_level_hierarchy({DEPTH_KEY: "banana"}))
        assert f'"metadata": "{DEPTH_KEY}"' in str(excinfo.value)

    def test_an_unrelated_metadata_entry_is_ignored(self):
        assert "Uses" in check_hierarchy(two_level_hierarchy({"version": "v1.0"})).model.value_domains


# --------------------------------------------------------------------------------------------------
# F. Function default *argument* expressions -- the same problem, not yet solved
# --------------------------------------------------------------------------------------------------

FUNCTION_TEMPLATE_CONTEXT = {"order": ["T"], "T": "Numeric", "substitution": {"FunctionReturning:T": "T"}}


def function_with_default(name: str, defaults: dict, arg_type: str = "T") -> dict:
    return {
        name: {
            "directParents": ["FunctionReturning"],
            "data": {
                "templateContext": FUNCTION_TEMPLATE_CONTEXT,
                "interface": {
                    "arg1": [arg_type],
                    "arg2": [arg_type],
                    "res": "T",
                    "_defaultArgumentValues": defaults,
                },
            },
        }
    }


class TestFunctionDefaultArgumentsAreNotSubstitutedYet:
    """
    A Function's ``_defaultArgumentValues`` have exactly the problem stage 1 solved for instantiation
    defaults: they are written in the Function's own template context, parsed once at definition time with
    ``T`` unbound, and only become checkable when a call site supplies a ground application.

    Stage 1 does **not** cover them. `parse_expression_of_json_object` substitutes the *types*
    (`function_return_type`, each `f_arg_type`) but never the unsupplied arguments' default *expressions* --
    those are not even materialised into `FunctionEvaluation.arguments`, which holds supplied arguments
    only. Nothing therefore re-checks them against the ground application.

    The first three tests pin what happens today; the last two are the gap. See
    ``TODO_DEFAULT_EXPANSION_CYCLES.md`` §16.
    """

    def test_a_non_template_default_argument_is_checked_at_definition_time(self):
        """
        The control that locates the gap: when the argument type is ground, the default *is* checked, and
        this bad one is rejected. Only template-dependence defers the check -- and the deferral never ends.
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
            {**function_with_default("Add", {"arg2": 3}), **uses_feval("Add<Integer>", {"arg1": 1})}
        )
        declared = context.model.functions["Add"].evaluation_argument_default_value["arg2"]
        assert declared.is_value_template_dependent, "with T unbound the default can not be resolved yet"

    def test_an_unsupplied_argument_is_not_materialised_at_the_call_site(self):
        """``FunctionEvaluation.arguments`` holds only what the call site wrote."""
        context = check_concepts(
            {**function_with_default("Add", {"arg2": 3}), **uses_feval("Add<Integer>", {"arg1": 1})}
        )
        feval = default_site_expression(context, "Site", "p").value
        assert isinstance(feval, FunctionEvaluation)
        assert sorted(feval.arguments) == ["arg1"]

    @pytest.mark.xfail(
        reason="Function default argument expressions are not substituted per ground application yet",
        strict=False,
    )
    def test_a_default_argument_is_grounded_at_a_ground_call_site(self):
        """At ``Add<Integer>``, ``arg2``'s default must be resolved as an ``Integer`` expression."""
        context = check_concepts(
            {**function_with_default("Add", {"arg2": 3}), **uses_feval("Add<Integer>", {"arg1": 1})}
        )
        feval = default_site_expression(context, "Site", "p").value
        assert "arg2" in feval.arguments, "the unsupplied argument must be materialised from its default"
        assert feval.arguments["arg2"].required_expression_type.full_name == "Integer"

    @pytest.mark.xfail(
        reason="nothing re-checks a template-dependent default argument once the application is known",
        strict=False,
    )
    def test_an_invalid_default_argument_is_rejected_at_a_ground_call_site(self):
        """
        ``arg2`` defaults to a String literal while ``T`` is constrained to ``Numeric``. Deferring at
        definition time is right -- ``T`` is unknown there -- but at ``BadAdd<Integer>`` it is definitely
        wrong, and is silently accepted.
        """
        with pytest.raises(ConceptHierarchyError):
            check_concepts(
                {
                    **function_with_default("BadAdd", {"arg2": "s:not-a-number"}),
                    **uses_feval("BadAdd<Integer>", {"arg1": 1}),
                }
            )
