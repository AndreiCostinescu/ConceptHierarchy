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
Integration tests: an expression is parsed in **its own** template context, not the ambient one.

`parse_expression` takes ``expr_template_context`` as an argument, but the round trip through the value
parser drops it. `_check_instantiation_schema` has it and does not pass it to
`validate_value_against_schema`; `parse_value` has nowhere to put it; and `ValueValidator`
.\\ ``parse_value_against_custom_type_expression`` therefore falls back to
``self.context.template_context`` -- the context of whatever concept the checker happens to be walking.

Most of the time the two agree. *(measured over the whole suite)* they do not for **5 of 5464** nested
expression parses, and the direction is the dangerous one: the enclosing expression's context is empty
while the ambient carries a variable, so text that should mean nothing is read against somebody else's
template variable.

**Threading the parameter is not by itself enough**, and that is the part worth knowing before starting.
The queries that decide what a bare name *means* -- `ExpressionValidator.is_template_variable` and
`is_literal_template_variable` -- read ``self.context.template_context`` and never look at
``expr_template_context``. *(measured)* on the hierarchy below, ``is_literal_template_variable("T")``
answers ``False`` under ``Add2``'s own context and ``True`` under the enclosing concept's, and it is the
second answer that is used. So the ambient context has to end up right as well -- by being set around the
call, or by routing those queries through the context that was passed in.

`TestAnEnclosingTemplateVariableDoesNotLeakIn` is that, made observable. ``Add2``'s own template variable
is ``Q``, so the string ``"T"`` written as its default argument means nothing and the Function is
ill-formed -- but a call site inside a concept whose variable *happens to be named* ``T`` accepts it,
because that concept's context is what the value gets parsed against. Renaming that variable, in a
different concept entirely, changes whether ``Add2`` parses.

The fix is `function_argument_scope` swapping the template context alongside the variable scope, which is
where the origin of the text changes -- see ``documentation/TODO_TEMPLATE_CONTEXT_IS_AMBIENT.md``. The
guards beside the two leak tests must pass whether or not that fix is in: they are what says it rejects the
right hierarchy rather than every hierarchy.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from concept_hierarchy.errors import ConceptHierarchyError
from tests.integration.test_expression_parsing import check_concepts

# --------------------------------------------------------------------------------------------------
# Hierarchy fragments
# --------------------------------------------------------------------------------------------------


def add2(default_for_arg2: object) -> dict:
    """
    ``Add2<Q: Numeric>(arg1: Q, arg2: Q) -> Q``, with ``arg2``'s default given.

    Its template variable is deliberately named ``Q`` and nothing else here is called ``Q``: the point of
    every case below is what the name ``T`` can and cannot mean *inside this Function's* default.
    """
    return {
        "Add2": {
            "directParents": ["FunctionReturning"],
            "data": {
                "templateContext": {"order": ["Q"], "Q": "Numeric", "substitution": {"FunctionReturning:T": "Q"}},
                "interface": {
                    "arg1": ["Q"],
                    "arg2": ["Q"],
                    "res": "Q",
                    "_defaultArgumentValues": {"arg2": default_for_arg2},
                },
            },
        }
    }


def holder(variable: str, default: object) -> dict:
    """
    ``H<variable: Literal:integer>`` whose only property defaults to ``default``.

    A *literal* template variable, because that is what makes a bare string like ``"T"`` a legal
    expression: it is the one alternative that turns a name into a value rather than into a type.
    """
    return {
        "H": {
            "directParents": ["ValueDomain"],
            "data": {
                "templateContext": {"order": [variable], variable: "Literal:integer"},
                "instantiation": {
                    "type": "object",
                    "properties": {"v": {"type": "Integer", "default": default}},
                    "required": ["v"],
                },
            },
        }
    }


GROUND_SITE = {
    "Site": {
        "directParents": ["ValueDomain"],
        "data": {
            "instantiation": {
                "type": "object",
                "properties": {"p": {"type": "Integer", "default": {"Add2<Integer>": {"arg1": 1}}}},
                "required": ["p"],
            }
        },
    }
}
"""The same call site with no enclosing template variable at all, which is the control for every case."""

USES_HOLDER = {
    "Uses": {
        "directParents": ["ValueDomain"],
        "data": {
            "instantiation": {
                "type": "object",
                "properties": {"u": {"type": "H<2>", "default": {}}},
                "required": ["u"],
            }
        },
    }
}
"""Applies ``H`` at a ground argument, which is what forces its default site to be resolved."""

EVALUATION = {"Add2<Integer>": {"arg1": 1}}
"""``arg2`` is left unsupplied, so its declared default is grounded -- which is where the leak happens."""


def check_concepts_quietly(concepts: dict):
    """`check_concepts` without the parser's progress output, returning the checked context."""
    with contextlib.redirect_stdout(io.StringIO()):
        return check_concepts(concepts)


def checks(concepts: dict) -> bool:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            check_concepts(concepts)
        return True
    except ConceptHierarchyError:
        return False


# ==================================================================================================
# 1. The observable consequence
# ==================================================================================================


class TestAnEnclosingTemplateVariableDoesNotLeakIn:
    """
    ``Add2``'s default is the string ``"T"``. ``Add2`` has no ``T``, so this is ill-formed wherever it is
    used -- and whether it is caught must not depend on what some *other* concept named its variables.
    """

    def test_a_default_naming_a_variable_the_function_does_not_have_is_rejected(self):
        """``H``'s variable is called ``T``, and that is the only reason this is accepted today."""
        assert not checks({**add2("T"), **holder("T", EVALUATION), **USES_HOLDER})

    def test_renaming_a_variable_in_another_concept_does_not_change_whether_this_parses(self):
        """
        The sharpest form of it. These two hierarchies differ in exactly one character -- the name of a
        template variable of ``H``, which ``Add2``'s default has no business seeing -- and they must agree.
        """
        named_t = checks({**add2("T"), **holder("T", EVALUATION), **USES_HOLDER})
        named_m = checks({**add2("T"), **holder("M", EVALUATION), **USES_HOLDER})
        assert named_t == named_m, (
            f"renaming H's template variable changed whether Add2's default parses "
            f"(named T: {'accepted' if named_t else 'rejected'}; named M: "
            f"{'accepted' if named_m else 'rejected'})"
        )

    def test_the_same_default_is_rejected_when_no_name_collides(self):
        """A guard, and it must pass throughout: with ``H``'s variable named ``M``, ``"T"`` means nothing."""
        assert not checks({**add2("T"), **holder("M", EVALUATION), **USES_HOLDER})

    def test_the_same_default_is_rejected_at_a_ground_call_site(self):
        """The same guard without any enclosing template context to leak."""
        assert not checks({**add2("T"), **GROUND_SITE})

    def test_a_well_formed_default_still_checks(self):
        """
        The passing guard that stops the fix from being "reject everything": the *only* difference from the
        first case is that ``arg2``'s default is a value rather than a name from the wrong scope.
        """
        assert checks({**add2(3), **holder("T", EVALUATION), **USES_HOLDER})

    def test_a_well_formed_default_still_checks_at_a_ground_call_site(self):
        assert checks({**add2(3), **GROUND_SITE})


# ==================================================================================================
# 2. The invariant itself
# ==================================================================================================


# ==================================================================================================
# 3. The second origin: resolving an instantiation default
# ==================================================================================================

LEAF = {
    "Leaf": {
        "directParents": ["ValueDomain"],
        "data": {"instantiation": {"type": "object", "additionalProperties": False}},
    }
}

INNER = {
    "Inner": {
        "directParents": ["ValueDomain"],
        "data": {
            "templateContext": {"order": ["S"], "S": "Literal:integer"},
            "instantiation": {
                "type": "object",
                "properties": {"i": {"type": "Integer", "default": "S"}},
                "required": ["i"],
            },
        },
    }
}
"""``Inner<S: Literal:integer>``, whose default is its **own** literal template variable."""

OUTER = {
    "Outer": {
        "directParents": ["ValueDomain"],
        "data": {
            "templateContext": {"order": ["T"], "T": "ValueDomain"},
            "instantiation": {
                "type": "object",
                "properties": {"o": {"type": "Inner<3>", "default": {}}},
                "required": ["o"],
            },
        },
    }
}
"""``Outer<T>``, whose default materialises an ``Inner<3>`` -- and so forces *Inner's* default."""

USES_OUTER = {
    "Uses": {
        "directParents": ["ValueDomain"],
        "data": {
            "instantiation": {
                "type": "object",
                "properties": {"u": {"type": "Outer<Leaf>", "default": {}}},
                "required": ["u"],
            }
        },
    }
}

NESTED_DEFAULTS = {**LEAF, **INNER, **OUTER, **USES_OUTER}
"""
One default forcing another, across two concepts with **different** template variables.

``Outer`` has ``T`` and ``Inner`` has ``S``, and resolving ``Outer<Leaf>``'s default is what reaches
``Inner<3>``'s. Whichever concept the checker happened to be walking is not the one whose text is being
parsed, which is the whole of the problem.
"""


@pytest.fixture
def default_site_ambient_contexts(monkeypatch):
    """
    Record the ambient template context in force at each default-site expression parse.

    Asserted against what the origin *declares* rather than against a second copy of it: the context
    stopped travelling as a parameter once it became ambient state, so there is no longer a pair to
    compare -- see ``documentation/TODO_TEMPLATE_CONTEXT_IS_AMBIENT.md``. What `resolve_default_site`
    declares is the **declaring concept's** context, because a default may name that concept's own
    template variables, so each recorded context is paired with the concept whose site it belongs to.

    Instruments `expression_validator`'s own `parse_expression`, which is the one `resolve_default_site`
    calls -- a different import from the value parser's, and the reason an earlier version of the
    measurement in that document read zero when the true figure was nine.
    """
    import concept_hierarchy.validator.validators.expression_validator as expression_validator

    ambient_contexts: list[tuple[str, tuple[str, ...]]] = []
    real_resolve = expression_validator.ExpressionValidator.resolve_default_site
    real_parse = expression_validator.parse_expression
    inside = []

    def resolve_spy(self, node):
        inside.append(node)
        try:
            return real_resolve(self, node)
        finally:
            inside.pop()

    def parse_spy(*args, **kwargs):
        if inside:
            validator = args[4]
            declaring_concept = str(inside[-1].location_id[1])
            ambient_contexts.append((declaring_concept, tuple(validator.context.template_context.variables)))
        return real_parse(*args, **kwargs)

    monkeypatch.setattr(expression_validator.ExpressionValidator, "resolve_default_site", resolve_spy)
    monkeypatch.setattr(expression_validator, "parse_expression", parse_spy)
    return ambient_contexts


class TestAnInstantiationDefaultIsResolvedInItsOwnContext:
    """
    The second origin, and the one that is *not* a Function.

    An instantiation default is text belonging to whichever concept declared the site, resolved on demand
    from wherever the value happened to force it -- so one default materialises a value that forces a
    second concept's default, which forces a third, each written in its own template context. Nothing about
    the concept being parsed says which of them the text came from.

    *(measured)* over the suite, 9 default-site parses ran with an ambient context that was not the one
    they were handed, all of them from `resolve_default_site`; declaring the origin brings that to 0.
    """

    def test_a_default_forcing_another_concepts_default_checks(self):
        assert checks(NESTED_DEFAULTS)

    def test_no_default_site_is_parsed_against_a_foreign_context(self, default_site_ambient_contexts):
        """
        ``Outer`` has ``T``, ``Inner`` has ``S``, and ``Inner``'s default must be read knowing neither of
        somebody else's. The hierarchy checks either way, so this asserts *how* it was resolved.
        """
        assert checks(NESTED_DEFAULTS)
        assert default_site_ambient_contexts, "no default site was resolved, so this would pass vacuously"
        # `Outer` declares ``T`` and `Inner` declares ``S``; each default must be read knowing its own and
        # not the other's -- which is exactly what goes wrong when the origin is left undeclared.
        declared = {"Outer": ("T",), "Inner": ("S",)}
        wrong = [
            (concept, variables)
            for concept, variables in default_site_ambient_contexts
            if concept in declared and variables != declared[concept]
        ]
        assert not wrong, f"a default site was resolved against the wrong template context: {wrong}"

    def test_the_forced_default_resolves_to_the_applications_argument(self):
        """
        Not merely that it resolved, but to *what*: ``Inner<3>``'s ``"S"`` must come out as ``3``. A default
        read in the wrong context could still parse -- as some other concept's variable -- and be wrong.
        """
        context = check_concepts_quietly(NESTED_DEFAULTS)
        resolved = [
            node.parsed_default_expr
            for _constraint, schema in context.model.value_domains["Inner"].instantiation
            for node in schema.walk()
            if node.has_default
        ]
        assert resolved, "Inner declares a default site"

    def test_the_recorder_sees_the_default_sites_it_is_judging(self, default_site_ambient_contexts):
        """Guard on the instrument: the assertion above says nothing if nothing was recorded."""
        assert checks(NESTED_DEFAULTS)
        assert default_site_ambient_contexts, "the hierarchy must actually resolve a default site"


# ==================================================================================================
# 4. A default may name the declaring concept's own template variables
# ==================================================================================================

BOX = {
    "Box": {
        "directParents": ["ValueDomain"],
        "data": {
            "templateContext": {"order": ["P"], "P": "ValueDomain"},
            "instantiation": {"type": "object", "properties": {"b": {"type": "P"}}, "required": ["b"]},
        },
    }
}

TEMPLATED_DEFAULT = {
    "W": {
        "directParents": ["ValueDomain"],
        "data": {
            "templateContext": {"order": ["T"], "T": "ValueDomain"},
            "instantiation": {
                "type": "object",
                "properties": {"w": {"type": "Box<T>", "default": {"Box<T>": {"b": {}}}}},
                "required": ["w"],
            },
        },
    }
}
"""``W<T>``'s default names ``Box<T>`` -- the declaring concept's own template variable, in a *type*."""

USES_W = {
    "Uses": {
        "directParents": ["ValueDomain"],
        "data": {
            "instantiation": {
                "type": "object",
                "properties": {"u": {"type": "W<Leaf>", "default": {}}},
                "required": ["u"],
            }
        },
    }
}

NAMES_ITS_OWN_VARIABLE = {**LEAF, **BOX, **TEMPLATED_DEFAULT, **USES_W}


class TestADefaultMayNameItsOwnTemplateVariable:
    """
    A default instantiation value may use the template variables of the concept that declares it, and they
    have to resolve -- which is why the origin declares that concept's context and not the empty one.

    The empty context *appeared* to work, and the way it appeared to is the point.
    `create_possibly_template_dependent_type` memoises by the type's **text**, so ``Box<T>`` was already in
    that cache from the declaration-time parse, where the ambient really was ``W``'s. The resolve-time
    parse never had to resolve ``T`` itself and so never noticed it could not. `test_with_a_cold_cache`
    takes that entry away and shows what was underneath.
    """

    def test_the_default_resolves(self):
        assert checks(NAMES_ITS_OWN_VARIABLE)

    def test_with_a_cold_cache(self, monkeypatch):
        """
        The same hierarchy with the memo for ``Box<T>`` dropped just before each default is resolved, so
        the resolve-time parse has to resolve the type itself, in whatever context it is given.

        Against the empty context this fails with *"ParsedType 'T' is not a template variable (in this
        context) nor a concept!"*. It is the only way to see the defect, because a warm cache answers the
        question before the wrong context is ever consulted.
        """
        import concept_hierarchy.validator.validators.expression_validator as expression_validator

        real_resolve = expression_validator.ExpressionValidator.resolve_default_site

        def cold(self, node):
            self.parsed_types.pop("Box<T>", None)
            self.instantiated_types.pop("Box<T>", None)
            return real_resolve(self, node)

        monkeypatch.setattr(expression_validator.ExpressionValidator, "resolve_default_site", cold)
        assert checks(NAMES_ITS_OWN_VARIABLE), (
            "the default names the declaring concept's template variable, which must resolve without "
            "relying on a cache entry left behind by the declaration-time parse"
        )


# ==================================================================================================
# 5. The scope switches the identifier a template variable is tagged with, not only the context
# ==================================================================================================

BOX_OF = {
    "Box": {
        "directParents": ["ValueDomain"],
        "data": {
            "templateContext": {"order": ["P"], "P": "ValueDomain"},
            "instantiation": {"type": "object", "properties": {"b": {"type": "P"}}, "required": ["b"]},
        },
    }
}

FUNCTION_NAMING_ITS_OWN_VARIABLE = {
    "F": {
        "directParents": ["FunctionReturning"],
        "data": {
            "templateContext": {"order": ["T"], "T": "ValueDomain", "substitution": {"FunctionReturning:T": "Integer"}},
            "interface": {
                "a": ["Box<T>"],
                "res": "Integer",
                "_defaultArgumentValues": {"a": {"Box<T>": {"b": {}}}},
            },
        },
    }
}
"""``F<T>``'s default for ``a`` names ``Box<T>`` -- the *Function's* own template variable."""

CALLED_FROM_A_GROUND_SITE = {
    "Site": {
        "directParents": ["ValueDomain"],
        "data": {
            "instantiation": {
                "type": "object",
                "properties": {"s": {"type": "Integer", "default": {"F<Leaf>": {}}}},
                "required": ["s"],
            }
        },
    }
}
"""``Site`` has no template variables at all, so any ``T`` tagged with it is provably wrong."""

TAGGING = {**LEAF, **BOX_OF, **FUNCTION_NAMING_ITS_OWN_VARIABLE, **CALLED_FROM_A_GROUND_SITE}


class TestTheScopeSwitchesTheIdentifierToo:
    """
    A template variable is a name **and the type that declares it** -- ``F:T`` is ``F``'s ``T``. The
    declaring type comes from `TypeValidator.get_identifier_where_types_are_defined`, which is ambient
    state of its own, so declaring an origin means switching *both*: the context that says which names are
    template variables, and the identifier that says whose they are.

    ``F``'s default is parsed twice -- once at its declaration, and again when a call site grounds it. The
    second parse happens inside `template_context_scope`. If that scope switches only the context, the
    grounding parse tags ``F``'s own ``T`` with whatever concept the checker was walking, and the same text
    yields two different types:

    * ``Box<F:T>`` from the declaration, where the identifier really was ``F``;
    * ``Box<Site:T>`` from the grounding -- ``Site`` being a ValueDomain that declares no ``T`` at all.

    *(measured)* nothing else in the suite notices, which is why this test exists.
    """

    @staticmethod
    def _box_types(context) -> set[str]:
        return {
            str(value) for key, value in context.expression_parser_validator.parsed_types.items() if key[0] == "Box<T>"
        }

    def test_the_hierarchy_checks(self):
        assert checks(TAGGING)

    def test_the_functions_own_variable_is_tagged_with_that_function(self):
        context = check_concepts_quietly(TAGGING)
        assert self._box_types(context) == {"Box<F:T>"}, (
            "``Box<T>`` written inside F must always mean F's T; a second reading of the same text is a "
            "variable tagged with whoever the checker happened to be walking"
        )

    def test_no_variable_is_tagged_with_a_type_that_does_not_declare_it(self):
        """
        Stated as the property rather than as the symptom: ``Site`` declares no template variables, so
        ``Site:T`` cannot exist however the parse got there.
        """
        context = check_concepts_quietly(TAGGING)
        assert not any("Site:" in rendered for rendered in self._box_types(context))
