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
Tests for the type checks over a whole Concept Hierarchy
(``checker.check_types`` step 2 -> ``check_types_in_concept_hierarchy``).

This step resolves and validates every type written anywhere in the hierarchy:

* DomainConcept property and function value domains;
* the template arguments of each type application, against the constraints the ValueDomain declared
  (the *instantiation constraints*);
* the substitution values a subconcept supplies for its parents' template arguments;
* Function evaluation argument and result types;
* ValueDomain instantiation schemas.

A type application is checked **both** for arity and for whether its arguments satisfy the declared
constraints; both surface as a failure to parse the containing definition into a type, with the specific
reason attached as a cause.
"""

from __future__ import annotations

import pytest

from concept_hierarchy.errors import CHSemanticError, CHSyntaxError
from tests.ch_support import check_concepts

# ``Box<T>`` requires its argument to be a Number.
BOX = {"Box": {"directParents": ["ValueDomain"], "data": {"templateContext": {"order": ["T"], "T": "Number"}}}}


def holder(property_type: str) -> dict:
    """A DomainConcept with a single property of the given type."""
    return {"Holder": {"directParents": ["Concept"], "data": {"properties": {"p": property_type}}}}


# --------------------------------------------------------------------------------------------------
# Instantiation constraints on a type application
# --------------------------------------------------------------------------------------------------


class TestInstantiationConstraints:
    def test_an_argument_satisfying_the_constraint_is_accepted(self):
        check_concepts({**BOX, **holder("Box<Integer>")})

    def test_the_constraint_accepts_the_bound_itself(self):
        check_concepts({**BOX, **holder("Box<Number>")})

    def test_an_argument_violating_the_constraint_is_rejected(self):
        with pytest.raises(CHSemanticError) as raised:
            check_concepts({**BOX, **holder("Box<String>")})
        assert "Box<String>" in str(raised.value)

    def test_too_few_template_arguments(self):
        with pytest.raises(CHSemanticError, match="into a type failed"):
            check_concepts({**BOX, **holder("Box")})

    def test_too_many_template_arguments(self):
        with pytest.raises(CHSemanticError, match="into a type failed"):
            check_concepts({**BOX, **holder("Box<Integer, Integer>")})

    def test_a_nested_type_application_is_checked_too(self):
        """``Box<Box<Integer>>`` fails because ``Box<Integer>`` is not a Number."""
        with pytest.raises(CHSemanticError, match="into a type failed"):
            check_concepts({**BOX, **holder("Box<Box<Integer>>")})

    @pytest.mark.parametrize("argument", ["3", "4"], ids=["matching", "mismatching"])
    def test_a_literal_template_argument_is_checked_against_its_literal_constraint(self, argument):
        concepts = {
            "Vector": {"directParents": ["ValueDomain"], "data": {"templateContext": {"order": ["N"], "N": 3}}},
        }
        if argument == "3":
            check_concepts({**concepts, **holder(f"Vector<{argument}>")})
        else:
            with pytest.raises(CHSemanticError):
                check_concepts({**concepts, **holder(f"Vector<{argument}>")})


# --------------------------------------------------------------------------------------------------
# Substitutions a subconcept supplies for its parents
# --------------------------------------------------------------------------------------------------


class TestParentTemplateSubstitutions:
    def test_a_substitution_satisfying_the_parent_constraint(self):
        check_concepts(
            {
                **BOX,
                "IntBox": {
                    "directParents": ["Box"],
                    "data": {"templateContext": {"substitution": {"Box:T": "Integer"}}},
                },
            }
        )

    def test_a_substitution_violating_the_parent_constraint(self):
        with pytest.raises(CHSemanticError, match="does not satisfy its constraints"):
            check_concepts(
                {
                    **BOX,
                    "StrBox": {
                        "directParents": ["Box"],
                        "data": {"templateContext": {"substitution": {"Box:T": "String"}}},
                    },
                }
            )

    def test_a_templated_subconcept_passing_its_own_variable_through(self):
        check_concepts(
            {
                **BOX,
                "TwinBox": {
                    "directParents": ["Box"],
                    "data": {"templateContext": {"order": ["E"], "E": "Integer", "substitution": {"Box:T": "E"}}},
                },
            }
        )

    def test_a_looser_variable_is_narrowed_by_the_parent_constraint(self):
        """
        ``E`` is declared over any ValueDomain but is substituted into ``Box:T``, which must be a Number.
        That is not an error: the parent's requirement is conjoined onto the child's own constraint.
        """
        context = check_concepts(
            {
                **BOX,
                "LooseBox": {
                    "directParents": ["Box"],
                    "data": {"templateContext": {"order": ["E"], "E": "ValueDomain", "substitution": {"Box:T": "E"}}},
                },
            }
        )
        constraint = repr(context.model.value_domains["LooseBox"].template_context.constraint)
        assert "Number" in constraint and "ValueDomain" in constraint, constraint

    def test_a_missing_substitution_is_rejected(self):
        with pytest.raises(CHSemanticError, match="substitution"):
            check_concepts({**BOX, "SubBox": {"directParents": ["Box"], "data": {}}})


# --------------------------------------------------------------------------------------------------
# DomainConcept property value domains
# --------------------------------------------------------------------------------------------------


class TestPropertyTypes:
    def test_a_property_typed_by_a_value_domain(self):
        check_concepts(holder("Integer"))

    def test_a_property_typed_by_a_non_value_domain(self):
        with pytest.raises(CHSemanticError, match="is not a subtype of ValueDomain"):
            check_concepts(holder("Concept"))

    def test_a_property_typed_by_an_unknown_concept(self):
        with pytest.raises(CHSemanticError, match="into a type failed"):
            check_concepts(holder("NoSuchType"))


def domain_concept_function(function_definition: dict, extra: dict | None = None) -> dict:
    """A DomainConcept with a single ``functions`` member ``f``."""
    return {
        **(extra or {}),
        "Animal": {"directParents": ["Concept"], "data": {"functions": {"f": function_definition}}},
    }


class TestDomainConceptFunctionTypes:
    """
    The ``functions`` members of a DomainConcept -- distinct from the Function *concepts* covered by
    :class:`TestFunctionInterfaceTypes`. Their value domain must be a subtype of ``CustomFunction``.
    """

    def test_the_default_function_value_domain_is_accepted(self):
        context = check_concepts(domain_concept_function({"valueDomain": "CustomFunction"}))
        assert str(context.model.domain_concepts["Animal"].function_types["f"]) == "CustomFunction"

    def test_a_subtype_of_the_default_function_value_domain_is_accepted(self):
        context = check_concepts(
            domain_concept_function(
                {"valueDomain": "MyFunc"}, {"MyFunc": {"directParents": ["CustomFunction"], "data": {}}}
            )
        )
        assert str(context.model.domain_concepts["Animal"].function_types["f"]) == "MyFunc"

    def test_a_templated_function_value_domain_is_resolved(self):
        context = check_concepts(
            domain_concept_function(
                {"valueDomain": "Box<Integer>"},
                {
                    "Box": {
                        "directParents": ["CustomFunction"],
                        "data": {"templateContext": {"order": ["T"], "T": "Number"}},
                    }
                },
            )
        )
        assert str(context.model.domain_concepts["Animal"].function_types["f"]) == "Box<Integer>"

    def test_a_templated_function_value_domain_is_constraint_checked(self):
        with pytest.raises(CHSemanticError, match="into a type failed"):
            check_concepts(
                domain_concept_function(
                    {"valueDomain": "Box<String>"},
                    {
                        "Box": {
                            "directParents": ["CustomFunction"],
                            "data": {"templateContext": {"order": ["T"], "T": "Number"}},
                        }
                    },
                )
            )

    def test_a_function_typed_by_an_unknown_concept(self):
        with pytest.raises(CHSemanticError, match="into a type failed"):
            check_concepts(domain_concept_function({"valueDomain": "NoSuchType"}))

    def test_a_function_typed_by_a_non_function_value_domain(self):
        with pytest.raises(CHSemanticError, match="is not a subtype of"):
            check_concepts(domain_concept_function({"valueDomain": "Integer"}))


# --------------------------------------------------------------------------------------------------
# nameDefaultInstanceValuesWithThisInstanceName
# --------------------------------------------------------------------------------------------------

INSTANCE_BASE = {
    "InstanceBase": {
        "directParents": ["ValueDomain"],
        "data": {"instantiation": {"type": "object", "properties": {"instanceName": "string"}}},
    },
}


def naming_property(property_type: str) -> dict:
    return {
        "Animal": {
            "directParents": ["Concept"],
            "data": {
                "properties": {
                    "p": {"valueDomain": property_type, "nameDefaultInstanceValuesWithThisInstanceName": True}
                }
            },
        }
    }


def naming_specialization(property_type: str, *, for_this: bool = False) -> dict:
    """``Animal`` defines the property; ``Dog`` switches the flag on for it in a specialization."""
    specialization = {"p": {"nameDefaultInstanceValuesWithThisInstanceName": True}}
    return {
        "Animal": {"directParents": ["Concept"], "data": {"properties": {"p": property_type}}},
        "Dog": {
            "directParents": ["Animal"],
            "data": {"properties": {"_specializations": {"_forThis": specialization} if for_this else specialization}},
        },
    }


class TestDefaultInstanceNaming:
    """
    ``nameDefaultInstanceValuesWithThisInstanceName`` is only meaningful for a property whose type contains
    an instance type -- there is otherwise no default instance value to name.
    """

    def test_requires_the_instance_base_concept_to_exist(self):
        with pytest.raises(CHSemanticError, match="InstanceBase concept is not defined"):
            check_concepts(naming_property("Integer"))

    def test_only_applies_when_the_flag_is_literally_true(self):
        """The guard is ``is True``, so ``false`` -- or a specialization directive -- does not trigger it."""
        check_concepts(
            {
                **INSTANCE_BASE,
                "Animal": {
                    "directParents": ["Concept"],
                    "data": {
                        "properties": {
                            "p": {"valueDomain": "Integer", "nameDefaultInstanceValuesWithThisInstanceName": False}
                        }
                    },
                },
            }
        )

    def test_rejects_a_property_with_no_instance_type(self):
        with pytest.raises(CHSemanticError, match="does not contain any instance type"):
            check_concepts({**INSTANCE_BASE, **naming_property("Integer")})

    def test_accepts_an_instance_typed_property(self):
        check_concepts({**INSTANCE_BASE, **naming_property("InstanceBase")})

    def test_accepts_a_subtype_of_instance_base(self):
        check_concepts(
            {
                **INSTANCE_BASE,
                "Instance": {"directParents": ["InstanceBase"], "data": {}},
                **naming_property("Instance"),
            }
        )


class TestDefaultInstanceNamingInSpecializations:
    """
    The flag may also be switched on in a specialization, on a property that is inherited rather than
    defined here -- the error text itself says the keyword must be removed "from all property definitions
    *and specializations*". Those spellings must be checked the same way as the definition site.
    """

    def test_a_specialization_on_an_instance_typed_property_is_accepted(self):
        check_concepts({**INSTANCE_BASE, **naming_specialization("InstanceBase")})

    def test_a_for_this_specialization_on_an_instance_typed_property_is_accepted(self):
        check_concepts({**INSTANCE_BASE, **naming_specialization("InstanceBase", for_this=True)})

    def test_a_specialization_on_a_property_with_no_instance_type_is_rejected(self):
        with pytest.raises(CHSemanticError, match="does not contain any instance type"):
            check_concepts({**INSTANCE_BASE, **naming_specialization("Integer")})

    def test_a_for_this_specialization_on_a_property_with_no_instance_type_is_rejected(self):
        with pytest.raises(CHSemanticError, match="does not contain any instance type"):
            check_concepts({**INSTANCE_BASE, **naming_specialization("Integer", for_this=True)})

    def test_a_for_this_specialization_on_a_locally_defined_property_is_rejected(self):
        """The definition-site loop reads the property body, so a ``_forThis`` override slips past it."""
        with pytest.raises(CHSemanticError, match="does not contain any instance type"):
            check_concepts(
                {
                    **INSTANCE_BASE,
                    "Animal": {
                        "directParents": ["Concept"],
                        "data": {
                            "properties": {
                                "p": "Integer",
                                "_specializations": {
                                    "_forThis": {"p": {"nameDefaultInstanceValuesWithThisInstanceName": True}}
                                },
                            }
                        },
                    },
                }
            )

    def test_a_specialization_still_needs_the_instance_base_concept(self):
        with pytest.raises(CHSemanticError, match="InstanceBase concept is not defined"):
            check_concepts(naming_specialization("Integer"))


# --------------------------------------------------------------------------------------------------
# Function concept interfaces
# --------------------------------------------------------------------------------------------------


def function_concept(interface: dict, name: str = "Add") -> dict:
    return {
        name: {
            "directParents": ["FunctionReturning"],
            "data": {
                "templateContext": {"order": ["T"], "T": "Numeric", "substitution": {"FunctionReturning:T": "T"}},
                "interface": interface,
            },
        }
    }


class TestFunctionInterfaceTypes:
    def test_a_well_typed_interface(self):
        check_concepts(function_concept({"arg1": ["T"], "res": "T"}))

    def test_an_argument_of_an_unknown_type(self):
        with pytest.raises(CHSemanticError, match="into a type failed"):
            check_concepts(function_concept({"arg1": "NoSuchType", "res": "T"}))

    def test_an_argument_using_an_out_of_scope_template_variable(self):
        with pytest.raises(CHSemanticError, match="into a type failed"):
            check_concepts(function_concept({"arg1": ["Z"], "res": "T"}))

    def test_an_argument_of_a_constrained_type_application(self):
        check_concepts({**BOX, **function_concept({"arg1": "Box<Integer>", "res": "T"})})

    def test_an_argument_violating_a_type_application_constraint(self):
        with pytest.raises(CHSemanticError, match="into a type failed"):
            check_concepts({**BOX, **function_concept({"arg1": "Box<String>", "res": "T"})})


# --------------------------------------------------------------------------------------------------
# ValueDomain instantiation schemas
# --------------------------------------------------------------------------------------------------


class TestInstantiationSchemas:
    def test_a_builtin_schema(self):
        check_concepts({"Flag": {"directParents": ["ValueDomain"], "data": {"instantiation": "boolean"}}})

    def test_an_object_schema_with_a_custom_typed_property(self):
        check_concepts(
            {
                "Point": {
                    "directParents": ["ValueDomain"],
                    "data": {"instantiation": {"type": "object", "properties": {"x": "Integer"}}},
                }
            }
        )

    def test_a_schema_naming_an_unknown_type(self):
        with pytest.raises(CHSyntaxError, match="into a json schema failed"):
            check_concepts(
                {
                    "Point": {
                        "directParents": ["ValueDomain"],
                        "data": {"instantiation": {"type": "object", "properties": {"x": "NoSuchType"}}},
                    }
                }
            )

    def test_a_schema_using_the_value_domains_own_template_argument(self):
        check_concepts(
            {
                "Box": {
                    "directParents": ["ValueDomain"],
                    "data": {
                        "templateContext": {"order": ["T"], "T": "Number"},
                        "instantiation": {"type": "object", "properties": {"content": "T"}},
                    },
                }
            }
        )


# --------------------------------------------------------------------------------------------------
# Instantiation constraints of *nested* type applications
# --------------------------------------------------------------------------------------------------

# ``Any<T>`` accepts any ValueDomain, so it never rejects an argument on its own account. That is what
# makes it able to hide a bad application: only checking the outermost type says nothing about what is
# inside it. ``Box<T : Number>`` from above is the strict one that the nested applications violate.
ANY = {"Any": {"directParents": ["ValueDomain"], "data": {"templateContext": {"order": ["T"], "T": "ValueDomain"}}}}


class TestNestedInstantiationConstraints:
    """
    Every type application in a value is checked against the constraints of *its own* concept, at any
    depth -- not just the outermost one.

    ``Box<Box<Integer>>`` above already failed, but only because the *outer* ``Box`` rejects a non-Number
    argument; the inner application was never looked at. Wrapping in a permissive ``Any`` removes that
    accident, so these are the cases that actually pin the recursion.
    """

    def test_a_violating_argument_one_level_down(self):
        with pytest.raises(CHSemanticError, match="into a type failed") as raised:
            check_concepts({**BOX, **ANY, **holder("Any<Box<String>>")})
        assert "Box<String>" in str(raised.value), "the message should name the application that is wrong"

    def test_a_violating_argument_two_levels_down(self):
        with pytest.raises(CHSemanticError, match="into a type failed"):
            check_concepts({**BOX, **ANY, **holder("Any<Any<Box<String>>>")})

    def test_a_satisfying_nested_argument_is_still_accepted(self):
        context = check_concepts({**BOX, **ANY, **holder("Any<Any<Box<Integer>>>")})
        assert str(context.model.domain_concepts["Holder"].property_types["p"]) == "Any<Any<Box<Integer>>>"

    def test_a_violating_argument_beside_a_satisfying_one(self):
        """The bad application sits in the second argument of a two-argument type."""
        concepts = {
            **BOX,
            "Pair": {
                "directParents": ["ValueDomain"],
                "data": {"templateContext": {"order": ["A", "B"], "A": "ValueDomain", "B": "ValueDomain"}},
            },
        }
        check_concepts({**concepts, **holder("Pair<Box<Integer>, Box<Number>>")})
        with pytest.raises(CHSemanticError, match="into a type failed"):
            check_concepts({**concepts, **holder("Pair<Box<Integer>, Box<String>>")})

    def test_a_violating_argument_in_a_function_argument_type(self):
        with pytest.raises(CHSemanticError):
            check_concepts(
                {
                    **BOX,
                    **ANY,
                    "Unwrap": {
                        "directParents": ["FunctionReturning"],
                        "data": {
                            "templateContext": {"substitution": {"FunctionReturning:T": "Integer"}},
                            "interface": {"arg": "Any<Box<String>>", "res": "Integer"},
                        },
                    },
                }
            )

    def test_a_violating_argument_in_an_instantiation_schema(self):
        """The schema parser wraps the violation, as it wraps any type failure inside a schema."""
        with pytest.raises(CHSyntaxError, match="into a json schema failed") as raised:
            check_concepts(
                {
                    **BOX,
                    **ANY,
                    "Wrap": {
                        "directParents": ["ValueDomain"],
                        "data": {"instantiation": {"type": "object", "properties": {"y": "Any<Box<String>>"}}},
                    },
                }
            )
        assert "Box<String>" in str(raised.value), "the wrapped cause should name the application that is wrong"

    def test_a_violating_argument_in_a_type_alias_target(self):
        """A type alias is resolved through the same conversion, so it is checked the same way."""
        with pytest.raises(CHSemanticError, match="into a type failed"):
            check_concepts({**BOX, **ANY, "BadBox": "Any<Box<String>>"})

    def test_a_violating_argument_reached_through_a_concept_alias(self):
        """The alias resolves to ``Box`` before the constraint is checked, so it is caught all the same."""
        with pytest.raises(CHSemanticError, match="into a type failed"):
            check_concepts({**BOX, **ANY, "MyBox": "Box", **holder("Any<MyBox<String>>")})
