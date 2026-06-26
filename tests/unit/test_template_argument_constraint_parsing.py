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
Comprehensive test suite for _ConstraintParser / parse_constraint_definition.

Grammar under test
------------------
    constraint                 ::= structureConstraint | nonStructureConstraint
    structureConstraint        ::= constraintGroup | structureOperator
    constraintGroup            ::= '<' nonStructureConstraint (', ' nonStructureConstraint)* '>'
    structureOperator          ::= conjExpr | disjExpr | negExpr
    conjExpr                   ::= 'Conj(' structureConstraintList ')'
    disjExpr                   ::= 'Disj(' structureConstraintList ')'
    negExpr                    ::= 'Neg('  structureConstraint     ')'
    structureConstraintList    ::= structureConstraint (', ' structureConstraint)*
    nonStructureConstraint     ::= unconstrained | nonTypeConstraint | typeConstraint
    unconstrained              ::= ''
    nonTypeConstraint          ::= literalConstraint | literalValue
    literalConstraint          ::= 'Literal:' ('boolean' | 'int' | 'number' | 'string')
    literalValue               ::= stringValue | boolValue | numberValue
    typeConstraint             ::= operatorCall | hierarchyLiteral
    operatorCall               ::= andExpr | orExpr | notExpr
    andExpr                    ::= 'And(' nonStructureConstraintList ')'
    orExpr                     ::=  'Or(' nonStructureConstraintList ')'
    notExpr                    ::= 'Not(' nonStructureConstraint     ')'
    nonStructureConstraintList ::= nonStructureConstraint (', ' nonStructureConstraint)*
    hierarchyLiteral           ::= '^'? letterStartingName ('*' | '.')?
    letterStartingName         ::= ALPHA (ALNUM | '_')* constraintGroup?

Validator convention
--------------------
    Uppercase-initial name   →  concept          (may be templatable)
    Lowercase-initial name   →  template variable (never has template args)
    Underscore-initial name  →  neither           (CHSyntaxError at parse level: not isalpha)

    Template-arg counts enforced by the validator:
        0-arg concepts: A, B, C, D, E, Animal, Plant, Fungus, Protist,
                        Container, Outer, Middle, Inner, Foo, Bar, Baz,
                        KeyType, ValueType, First, Second, Third, Fourth (and any unlisted uppercase)
        1-arg concepts: Vector, Collection, Precision, Flag, Tagged
        2-arg concepts: Map, Pair
        3-arg concepts: Triple

Known attribute names (from source)
------------------------------------
    TemplateConstraintAnd / TemplateConstraintOr   →  .sub_formulae : tuple
    TemplateConstraintNot                          →  .sub_formula  : TemplateConstraintFormula
    TemplateConstraintHierarchyOperator            →  .literal : str
                                                      .literal_template_formulae : tuple
    NonTypeTemplateConstraintFormula               →  .constraint_type : str
    LiteralValueConstraintFormula                  →  .raw_value : str
    ConstraintGroup                                →  .group_constraints : tuple
    StructureConjunction / StructureDisjunction    →  .structure_constraints : tuple
    StructureNegation                              →  .structure_constraint : StructureConstraintFormula

Critical behavioural changes vs. the previous test suite
---------------------------------------------------------
    * And / Or / Not now REQUIRE TypeTemplateConstraintFormula operands.
      Unconstrained, NonType, LiteralValue  all cause CHSyntaxError.
      Empty And() / Or() / Not()  therefore raise ChSyntaxError (Unconstrained
      produced for the implicit slot is not a TypeTemplateConstraintFormula).

    * Conj / Disj / Neg  require StructureConstraintFormula operands.
      Empty Conj() / Disj() / Neg()  raise CHSyntaxError because no
      StructureConstraintFormula matches ')'.

    * ConstraintGroup cannot have zero elements: the grammar always populates
      ≥ 1 slot.  '<>' produces ConstraintGroup((Unconstrained,)), never ConstraintGroup(()).

    * Lowercase names are now valid as template-variable literals.  'animal'
      parses correctly; 'animal<Animal>' raises CHSemanticError.

    * Template-argument count is validated against the validator:
      'Animal<>' (0-arg concept with 1 explicit arg)  →  CHSemanticError.
      'Vector<>' (1-arg concept, Unconstrained fills the slot)  →  valid.
      'Vector<A, B>' (1-arg concept with 2 args)  →  CHSemanticError.
"""

from __future__ import annotations

import pytest

from concept_hierarchy.data.parsers.template_argument_constraint_parser import parse_constraint_definition
from concept_hierarchy.data.type_template_variables.constraint_formula import (
    ConstraintGroup,
    LiteralValueConstraintFormula,
    NonStructureConstraintFormula,
    NonTypeTemplateConstraintFormula,
    StructureConjunction,
    StructureConstraintFormula,
    StructureDisjunction,
    StructureNegation,
    TemplateConstraintAbstractAscendants,
    TemplateConstraintAbstractDescendants,
    TemplateConstraintAnd,
    TemplateConstraintAscendants,
    TemplateConstraintDescendants,
    TemplateConstraintFormula,
    TemplateConstraintFormulaValidator,
    TemplateConstraintNot,
    TemplateConstraintOr,
    TemplateConstraintSelf,
    TypeTemplateConstraintFormula,
    Unconstrained,
)
from concept_hierarchy.errors import CHSemanticError, CHSyntaxError, LocationId

# ===========================================================================
# Validator implementation
# ===========================================================================


class _TestValidator(TemplateConstraintFormulaValidator):
    """
    Concrete validator used throughout the test suite.

    Naming convention:
        Uppercase-initial  →  concept (is_concept = True)
        Lowercase-initial  →  template variable (is_template_variable = True)
        Underscore / digit start  →  blocked at the parser level (.isalpha() guard),
                                     so the validator is never reached for those names.

    Template-argument counts:
        0  (default)      Animal, Plant, Fungus, Protist, A…Z single-letters,
                          KeyType, ValueType, Container, Foo, Bar, Baz, etc.
        1                 Vector, Collection, Precision, Flag, Tagged
        2                 Map, Pair
        3                 Triple
    """

    _TEMPLATE_ARG_COUNTS: dict[str, int] = {
        "Vector": 1,
        "Collection": 1,
        "Precision": 1,
        "Flag": 1,
        "Tagged": 1,
        "Map": 2,
        "Pair": 2,
        "Triple": 3,
    }

    # --- TemplateConstraintFormulaValidator abstract methods ---------------

    def is_concept(self, name: str) -> bool:
        return bool(name) and name[0].isupper()

    def is_template_variable(self, name: str) -> bool:
        return bool(name) and name[0].islower()

    def get_nr_template_arguments(self, concept_name: str) -> int:
        return self._TEMPLATE_ARG_COUNTS.get(concept_name, 0)

    def full_type_name(self, name: str) -> str:
        count = self.get_nr_template_arguments(name)
        if count == 0:
            return name
        args = ", ".join(f"T{i}" for i in range(1, count + 1))
        return f"{name}<{args}>"

    def get_existing_template_variables(self) -> set[str]:
        pass

    def update_existing_template_variables(self, new_template_variables: set[str]):
        pass


# ===========================================================================
# Fixtures and helpers
# ===========================================================================


@pytest.fixture(scope="module")
def V() -> _TestValidator:
    return _TestValidator()


@pytest.fixture(scope="module")
def loc() -> LocationId:
    return LocationId(["test_constraint_parser"])


def _parse(text: str, V: _TestValidator, loc: LocationId) -> TemplateConstraintFormula:
    return parse_constraint_definition(text, V, loc)


# ===========================================================================
# 1.  Unconstrained
# ===========================================================================


class TestUnconstrained:
    """Empty / whitespace-only input, and implicit empty slots, yield Unconstrained."""

    def test_empty_string(self, V, loc):
        result = _parse("", V, loc)
        assert isinstance(result, Unconstrained)
        assert isinstance(result, NonStructureConstraintFormula)

    def test_spaces_only(self, V, loc):
        assert isinstance(_parse("   ", V, loc), Unconstrained)

    def test_tab_only(self, V, loc):
        assert isinstance(_parse("\t", V, loc), Unconstrained)

    def test_mixed_whitespace(self, V, loc):
        assert isinstance(_parse("  \t\n  ", V, loc), Unconstrained)

    def test_newline_only(self, V, loc):
        assert isinstance(_parse("\n", V, loc), Unconstrained)

    def test_unconstrained_is_non_structure(self, V, loc):
        """Unconstrained belongs to NonStructureConstraintFormula, not Structure."""
        result = _parse("", V, loc)
        assert isinstance(result, NonStructureConstraintFormula)
        assert not isinstance(result, StructureConstraintFormula)


# ===========================================================================
# 2.  NonTypeTemplateConstraintFormula  (Literal:X)
# ===========================================================================


class TestNonTypeConstraint:
    @pytest.mark.parametrize(
        "text, expected_ctype",
        [
            ("Literal:boolean", NonTypeTemplateConstraintFormula.BOOLEAN),
            ("Literal:int", NonTypeTemplateConstraintFormula.INTEGER),
            ("Literal:number", NonTypeTemplateConstraintFormula.NUMBER),
            ("Literal:string", NonTypeTemplateConstraintFormula.STRING),
        ],
    )
    def test_produces_non_type_formula(self, V, loc, text, expected_ctype):
        result = _parse(text, V, loc)
        assert isinstance(result, NonTypeTemplateConstraintFormula)
        assert result.constraint_type == expected_ctype

    @pytest.mark.parametrize(
        "text",
        ["Literal:boolean", "Literal:int", "Literal:number", "Literal:string"],
    )
    def test_not_a_type_formula(self, V, loc, text):
        result = _parse(text, V, loc)
        assert not isinstance(result, TypeTemplateConstraintFormula)

    def test_repr(self, V, loc):
        assert repr(_parse("Literal:int", V, loc)) == "Literal:int"
        assert repr(_parse("Literal:boolean", V, loc)) == "Literal:bool"
        assert repr(_parse("Literal:number", V, loc)) == "Literal:float"
        assert repr(_parse("Literal:string", V, loc)) == "Literal:string"

    # --- error cases ---

    @pytest.mark.parametrize(
        "text",
        [
            "Literal:char",
            "Literal:float",  # correct keyword is 'number'
            "Literal:Boolean",  # wrong casing
            "Literal:INT",
            "Literal:",  # missing keyword
        ],
    )
    def test_unknown_literal_type_raises(self, V, loc, text):
        with pytest.raises(CHSyntaxError):
            _parse(text, V, loc)

    def test_trailing_content_raises(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse("Literal:boolean trailing", V, loc)


# ===========================================================================
# 3.  LiteralValueConstraintFormula  (concrete literal values)
# ===========================================================================


class TestLiteralValueConstraint:
    # --- booleans ---

    def test_true(self, V, loc):
        r = _parse("true", V, loc)
        assert isinstance(r, LiteralValueConstraintFormula)
        assert r.raw_value == "true"
        assert r.constraint_type == NonTypeTemplateConstraintFormula.BOOLEAN
        assert r.value is True

    def test_false(self, V, loc):
        r = _parse("false", V, loc)
        assert isinstance(r, LiteralValueConstraintFormula)
        assert r.raw_value == "false"
        assert r.value is False

    def test_bool_word_boundary_raises(self, V, loc):
        """'trueValue' must not be silently consumed as bool 'true'."""
        with pytest.raises(CHSyntaxError):
            _parse("trueValue", V, loc)

    def test_falsehood_raises(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse("falsehood", V, loc)

    # --- string literals ---

    def test_simple_string(self, V, loc):
        r = _parse('"hello"', V, loc)
        assert isinstance(r, LiteralValueConstraintFormula)
        assert r.constraint_type == NonTypeTemplateConstraintFormula.STRING
        assert r.raw_value == '"hello"'

    def test_empty_string_literal(self, V, loc):
        r = _parse('""', V, loc)
        assert r.raw_value == '""'

    def test_string_with_spaces(self, V, loc):
        assert isinstance(_parse('"hello world"', V, loc), LiteralValueConstraintFormula)

    def test_string_with_escaped_quote(self, V, loc):
        r = _parse(r'"say \"hi\""', V, loc)
        assert isinstance(r, LiteralValueConstraintFormula)

    # --- integers ---

    @pytest.mark.parametrize("text", ["0", "1", "42", "-1", "-42"])
    def test_integer_literals(self, V, loc, text):
        r = _parse(text, V, loc)
        assert isinstance(r, LiteralValueConstraintFormula)
        assert r.constraint_type == NonTypeTemplateConstraintFormula.INTEGER
        assert r.raw_value == text

    def test_integer_preferred_over_float(self, V, loc):
        """Bare '3' must be classified as int, not float."""
        r = _parse("3", V, loc)
        assert isinstance(r, LiteralValueConstraintFormula)
        assert r.constraint_type == NonTypeTemplateConstraintFormula.INTEGER
        assert r.raw_value == "3"

    # --- floats ---

    @pytest.mark.parametrize("text", ["3.14", "0.5", "-2.5", "3.0"])
    def test_float_literals(self, V, loc, text):
        r = _parse(text, V, loc)
        assert isinstance(r, LiteralValueConstraintFormula)
        assert r.constraint_type == NonTypeTemplateConstraintFormula.NUMBER

    def test_float_not_classified_as_int(self, V, loc):
        assert _parse("3.14", V, loc).constraint_type != NonTypeTemplateConstraintFormula.INTEGER

    # --- error cases ---

    def test_bare_minus_raises(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse("-", V, loc)

    def test_unterminated_string_raises(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse('"hello', V, loc)


# ===========================================================================
# 4.  Hierarchy literals  (T, T*, T., ^T, ^T*)
# ===========================================================================


class TestHierarchyLiterals:
    """
    Four kinds of hierarchy operator, applied to both concepts and template variables.
    The validator is called during construction: underscore-starting names are blocked
    by the parser-level .isalpha() guard before the validator is even consulted.
    """

    # --- Descendants ---

    def test_descendants_concept(self, V, loc):
        r = _parse("Animal", V, loc)
        assert isinstance(r, TemplateConstraintDescendants)
        assert isinstance(r, TypeTemplateConstraintFormula)
        assert r.literal == "Animal"
        assert r.literal_template_formulae == ()

    def test_descendants_template_variable(self, V, loc):
        """Lowercase names are valid template variables since parser now checks .isalpha()."""
        r = _parse("animal", V, loc)
        assert isinstance(r, TemplateConstraintDescendants)
        assert r.literal == "animal"
        assert r.literal_template_formulae == ()

    def test_descendants_repr(self, V, loc):
        assert repr(_parse("Animal", V, loc)) == "Animal"

    # --- AbstractDescendants ---

    def test_abstract_descendants_concept(self, V, loc):
        r = _parse("Animal*", V, loc)
        assert isinstance(r, TemplateConstraintAbstractDescendants)
        assert r.literal == "Animal"

    def test_abstract_descendants_template_variable(self, V, loc):
        r = _parse("t*", V, loc)
        assert isinstance(r, TemplateConstraintAbstractDescendants)
        assert r.literal == "t"

    def test_abstract_descendants_repr(self, V, loc):
        assert repr(_parse("Animal*", V, loc)) == "Animal*"

    # --- Self ---

    def test_self_concept(self, V, loc):
        r = _parse("Animal.", V, loc)
        assert isinstance(r, TemplateConstraintSelf)
        assert r.literal == "Animal"

    def test_self_template_variable(self, V, loc):
        r = _parse("t.", V, loc)
        assert isinstance(r, TemplateConstraintSelf)

    def test_self_repr(self, V, loc):
        assert repr(_parse("Animal.", V, loc)) == "Animal."

    # --- Ascendants ---

    def test_ascendants_concept(self, V, loc):
        r = _parse("^Animal", V, loc)
        assert isinstance(r, TemplateConstraintAscendants)
        assert r.literal == "Animal"

    def test_ascendants_template_variable(self, V, loc):
        r = _parse("^t", V, loc)
        assert isinstance(r, TemplateConstraintAscendants)

    def test_ascendants_repr(self, V, loc):
        assert repr(_parse("^Animal", V, loc)) == "^Animal"

    # --- AbstractAscendants ---

    def test_abstract_ascendants_concept(self, V, loc):
        r = _parse("^Animal*", V, loc)
        assert isinstance(r, TemplateConstraintAbstractAscendants)
        assert r.literal == "Animal"

    def test_abstract_ascendants_template_variable(self, V, loc):
        r = _parse("^t*", V, loc)
        assert isinstance(r, TemplateConstraintAbstractAscendants)

    def test_abstract_ascendants_repr(self, V, loc):
        assert repr(_parse("^Animal*", V, loc)) == "^Animal*"

    # --- all five are TypeTemplateConstraintFormula ---

    @pytest.mark.parametrize(
        "text",
        ["Animal", "Animal*", "Animal.", "^Animal", "^Animal*"],
    )
    def test_hierarchy_is_type_constraint(self, V, loc, text):
        assert isinstance(_parse(text, V, loc), TypeTemplateConstraintFormula)

    # --- invalid combinations ---

    @pytest.mark.parametrize(
        "text",
        ["^Animal.", "^Animal*.", "Animal*.", "Animal.*"],
    )
    def test_invalid_hierarchy_combo_raises(self, V, loc, text):
        with pytest.raises(CHSyntaxError):
            _parse(text, V, loc)

    def test_trailing_content_raises(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse("Animal extra", V, loc)

    # --- underscore-initial blocked by parser before validator ---

    def test_underscore_initial_raises_syntax_error(self, V, loc):
        """`_A` is not isalpha() → CHSyntaxError from _parse_literal_name."""
        with pytest.raises(CHSyntaxError):
            _parse("_Animal", V, loc)

    def test_digit_initial_does_not_reach_hierarchy(self, V, loc):
        """A leading digit is consumed as a number, leaving trailing text → CHSyntaxError."""
        with pytest.raises(CHSyntaxError):
            _parse("3D", V, loc)


# ===========================================================================
# 5.  Boolean operators  (And, Or, Not)
# ===========================================================================


class TestBooleanOperators:
    """
    And / Or / Not only accept TypeTemplateConstraintFormula operands.
    This means:
      - Unconstrained        →  CHSyntaxError
      - LiteralValue         →  CHSyntaxError
      - NonTypeConstraint    →  CHSyntaxError
      - ConstraintGroup      →  cannot even reach And/Or/Not
                                (structure constraints are routed before non-structure)

    Empty And() / Or() / Not() parse to (Unconstrained,) / Unconstrained,
    which then fails the TypeTemplateConstraintFormula check.
    """

    # --- valid uses ---

    def test_and_two_hierarchy_concepts(self, V, loc):
        r = _parse("And(Animal, Plant)", V, loc)
        assert isinstance(r, TemplateConstraintAnd)
        assert len(r.sub_formulae) == 2
        assert isinstance(r.sub_formulae[0], TemplateConstraintDescendants)
        assert isinstance(r.sub_formulae[1], TemplateConstraintDescendants)
        assert r.sub_formulae[0].literal == "Animal"
        assert r.sub_formulae[1].literal == "Plant"

    def test_and_three_args(self, V, loc):
        r = _parse("And(Animal, Plant, Fungus)", V, loc)
        assert len(r.sub_formulae) == 3

    def test_and_single_arg(self, V, loc):
        r = _parse("And(Animal)", V, loc)
        assert isinstance(r, TemplateConstraintAnd)
        assert len(r.sub_formulae) == 1

    def test_and_repr(self, V, loc):
        assert repr(_parse("And(Animal, Plant)", V, loc)) == "And(Animal, Plant)"

    def test_or_two_args(self, V, loc):
        r = _parse("Or(Animal, Plant)", V, loc)
        assert isinstance(r, TemplateConstraintOr)
        assert len(r.sub_formulae) == 2

    def test_or_repr(self, V, loc):
        assert repr(_parse("Or(Animal, Plant)", V, loc)) == "Or(Animal, Plant)"

    def test_not_single_arg(self, V, loc):
        r = _parse("Not(Animal)", V, loc)
        assert isinstance(r, TemplateConstraintNot)
        assert isinstance(r.sub_formula, TemplateConstraintDescendants)
        assert r.sub_formula.literal == "Animal"

    def test_not_repr(self, V, loc):
        assert repr(_parse("Not(Animal)", V, loc)) == "Not(Animal)"

    def test_and_with_template_variables(self, V, loc):
        r = _parse("And(animal, plant)", V, loc)
        assert isinstance(r, TemplateConstraintAnd)
        assert r.sub_formulae[0].literal == "animal"

    def test_and_with_abstract_descendant(self, V, loc):
        r = _parse("And(Animal*, Plant)", V, loc)
        assert isinstance(r.sub_formulae[0], TemplateConstraintAbstractDescendants)

    def test_and_with_ascendant(self, V, loc):
        r = _parse("And(Animal, ^Plant)", V, loc)
        assert isinstance(r.sub_formulae[1], TemplateConstraintAscendants)

    def test_and_with_nested_and(self, V, loc):
        r = _parse("And(And(A, B), C)", V, loc)
        assert isinstance(r.sub_formulae[0], TemplateConstraintAnd)

    def test_or_with_not(self, V, loc):
        r = _parse("Or(Animal, Not(Plant))", V, loc)
        assert isinstance(r.sub_formulae[1], TemplateConstraintNot)

    def test_double_negation(self, V, loc):
        r = _parse("Not(Not(Animal))", V, loc)
        assert isinstance(r, TemplateConstraintNot)
        assert isinstance(r.sub_formula, TemplateConstraintNot)
        assert isinstance(r.sub_formula.sub_formula, TemplateConstraintDescendants)

    def test_deeply_nested(self, V, loc):
        r = _parse("And(Or(A, B), Not(And(C, D)))", V, loc)
        assert isinstance(r, TemplateConstraintAnd)
        assert isinstance(r.sub_formulae[0], TemplateConstraintOr)
        assert isinstance(r.sub_formulae[1], TemplateConstraintNot)
        inner_and = r.sub_formulae[1].sub_formula
        assert isinstance(inner_and, TemplateConstraintAnd)
        assert len(inner_and.sub_formulae) == 2

    # --- operand type enforcement: non-TypeTemplateConstraintFormula raises CHSyntaxError ---

    def test_and_empty_raises(self, V, loc):
        """And() → Unconstrained slot → not TypeTemplateConstraintFormula → CHSyntaxError."""
        with pytest.raises(CHSyntaxError):
            _parse("And()", V, loc)

    def test_or_empty_raises(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse("Or()", V, loc)

    def test_not_empty_raises(self, V, loc):
        """Not() → Unconstrained slot → not TypeTemplateConstraintFormula → CHSyntaxError."""
        with pytest.raises(CHSyntaxError):
            _parse("Not()", V, loc)

    def test_and_whitespace_only_raises(self, V, loc):
        """And( ) still resolves operand to Unconstrained → CHSyntaxError."""
        with pytest.raises(CHSyntaxError):
            _parse("And( )", V, loc)

    def test_not_whitespace_only_raises(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse("Not( )", V, loc)

    def test_and_with_literal_value_raises(self, V, loc):
        """LiteralValueConstraintFormula is not TypeTemplateConstraintFormula."""
        with pytest.raises(CHSyntaxError):
            _parse("And(3, Animal)", V, loc)

    def test_not_with_literal_value_raises(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse('Not("hello")', V, loc)

    def test_or_with_bool_value_raises(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse("Or(true, Animal)", V, loc)

    def test_and_with_non_type_constraint_raises(self, V, loc):
        """NonTypeTemplateConstraintFormula (Literal:X) is not TypeTemplateConstraintFormula."""
        with pytest.raises(CHSyntaxError):
            _parse("And(Literal:int, Animal)", V, loc)

    def test_not_with_non_type_constraint_raises(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse("Not(Literal:string)", V, loc)

    # --- unclosed operators raise ---

    def test_unclosed_and_raises(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse("And(Animal", V, loc)

    def test_unclosed_not_raises(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse("Not(Animal", V, loc)

    def test_trailing_content_after_and(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse("And(Animal, Plant) trailing", V, loc)


# ===========================================================================
# 6.  Template arguments in hierarchy literals
# ===========================================================================


class TestTemplateArguments:
    """
    Template arguments are now parsed as a ConstraintGroup and the
    group_constraints tuple is extracted as literal_template_formulae.
    The validator enforces argument-count matching.
    """

    # --- valid uses ---

    def test_one_arg_one_slot_concept(self, V, loc):
        """Vector has 1 template arg; Vector<Animal> fills it."""
        r = _parse("Vector<Animal>", V, loc)
        assert isinstance(r, TemplateConstraintDescendants)
        assert r.literal == "Vector"
        assert len(r.literal_template_formulae) == 1
        assert isinstance(r.literal_template_formulae[0], TemplateConstraintDescendants)
        assert r.literal_template_formulae[0].literal == "Animal"

    def test_two_args_two_slot_concept(self, V, loc):
        """Map has 2 template args."""
        r = _parse("Map<Animal, Plant>", V, loc)
        assert r.literal == "Map"
        args = r.literal_template_formulae
        assert len(args) == 2
        assert args[0].literal == "Animal"
        assert args[1].literal == "Plant"

    def test_three_args(self, V, loc):
        r = _parse("Triple<A, B, C>", V, loc)
        assert len(r.literal_template_formulae) == 3

    def test_repr_with_args(self, V, loc):
        assert repr(_parse("Vector<Animal>", V, loc)) == "Vector<Animal>"
        assert repr(_parse("Map<Animal, Plant>", V, loc)) == "Map<Animal, Plant>"

    def test_arg_ascendant(self, V, loc):
        args = _parse("Vector<^Animal>", V, loc).literal_template_formulae
        assert isinstance(args[0], TemplateConstraintAscendants)

    def test_arg_abstract_descendant(self, V, loc):
        args = _parse("Vector<Animal*>", V, loc).literal_template_formulae
        assert isinstance(args[0], TemplateConstraintAbstractDescendants)

    def test_arg_self(self, V, loc):
        args = _parse("Vector<Animal.>", V, loc).literal_template_formulae
        assert isinstance(args[0], TemplateConstraintSelf)

    def test_arg_abstract_ascendant(self, V, loc):
        args = _parse("Vector<^Animal*>", V, loc).literal_template_formulae
        assert isinstance(args[0], TemplateConstraintAbstractAscendants)

    def test_arg_and_operator(self, V, loc):
        args = _parse("Vector<And(A, B)>", V, loc).literal_template_formulae
        assert isinstance(args[0], TemplateConstraintAnd)

    def test_arg_or_operator(self, V, loc):
        args = _parse("Vector<Or(A, B)>", V, loc).literal_template_formulae
        assert isinstance(args[0], TemplateConstraintOr)

    def test_arg_not_operator(self, V, loc):
        args = _parse("Vector<Not(Animal)>", V, loc).literal_template_formulae
        assert isinstance(args[0], TemplateConstraintNot)

    def test_arg_non_type_literal(self, V, loc):
        args = _parse("Vector<Literal:int>", V, loc).literal_template_formulae
        assert isinstance(args[0], NonTypeTemplateConstraintFormula)
        assert args[0].constraint_type == NonTypeTemplateConstraintFormula.INTEGER

    def test_arg_integer_literal(self, V, loc):
        args = _parse("Vector<3>", V, loc).literal_template_formulae
        assert isinstance(args[0], LiteralValueConstraintFormula)
        assert args[0].raw_value == "3"

    def test_arg_float_literal(self, V, loc):
        args = _parse("Precision<0.5>", V, loc).literal_template_formulae
        assert isinstance(args[0], LiteralValueConstraintFormula)

    def test_arg_string_literal(self, V, loc):
        args = _parse('Tagged<"label">', V, loc).literal_template_formulae
        assert isinstance(args[0], LiteralValueConstraintFormula)

    def test_arg_bool_literal(self, V, loc):
        args = _parse("Flag<true>", V, loc).literal_template_formulae
        assert isinstance(args[0], LiteralValueConstraintFormula)
        assert args[0].value is True

    def test_abstract_descendant_with_arg(self, V, loc):
        r = _parse("Vector<Animal>*", V, loc)
        assert isinstance(r, TemplateConstraintAbstractDescendants)
        assert len(r.literal_template_formulae) == 1

    def test_self_with_arg(self, V, loc):
        r = _parse("Vector<Animal>.", V, loc)
        assert isinstance(r, TemplateConstraintSelf)

    def test_ascendant_with_arg(self, V, loc):
        r = _parse("^Vector<Animal>", V, loc)
        assert isinstance(r, TemplateConstraintAscendants)

    def test_abstract_ascendant_with_arg(self, V, loc):
        r = _parse("^Vector<Animal>*", V, loc)
        assert isinstance(r, TemplateConstraintAbstractAscendants)

    # --- validator-enforced count errors ---

    def test_zero_arg_concept_with_any_arg_raises(self, V, loc):
        """Animal has 0 template args; any explicit template arg → CHSemanticError."""
        with pytest.raises(CHSemanticError):
            _parse("Animal<Plant>", V, loc)

    def test_zero_arg_concept_with_unconstrained_raises(self, V, loc):
        """`Animal<>` → 1 explicit slot → CHSemanticError (Animal takes 0)."""
        with pytest.raises(CHSemanticError):
            _parse("Animal<>", V, loc)

    def test_one_arg_concept_too_many_args_raises(self, V, loc):
        """Vector<A, B> → 2 args but Vector takes 1 → CHSemanticError."""
        with pytest.raises(CHSemanticError):
            _parse("Vector<A, B>", V, loc)

    def test_two_arg_concept_wrong_count_raises(self, V, loc):
        """Map<A> → 1 arg but Map takes 2 → CHSemanticError."""
        with pytest.raises(CHSemanticError):
            _parse("Map<A>", V, loc)

    def test_template_variable_with_any_arg_raises(self, V, loc):
        """Template variables may never have template arg specifications."""
        with pytest.raises(CHSemanticError):
            _parse("animal<Animal>", V, loc)

    def test_template_variable_with_unconstrained_arg_raises(self, V, loc):
        with pytest.raises(CHSemanticError):
            _parse("t<>", V, loc)


# ===========================================================================
# 7.  ConstraintGroup  (<…> as top-level or in structure operators)
# ===========================================================================


class TestConstraintGroup:
    """
    '<' nonStructureConstraint (', ' nonStructureConstraint)* '>'

    Key invariant: ConstraintGroup always contains ≥ 1 element.
    '<>' produces ConstraintGroup((Unconstrained,)), never ConstraintGroup(()).
    """

    # --- single element ---

    def test_single_concept(self, V, loc):
        r = _parse("<Animal>", V, loc)
        assert isinstance(r, ConstraintGroup)
        assert isinstance(r, StructureConstraintFormula)
        assert len(r.group_constraints) == 1
        assert isinstance(r.group_constraints[0], TemplateConstraintDescendants)
        assert r.group_constraints[0].literal == "Animal"

    def test_single_template_variable(self, V, loc):
        r = _parse("<animal>", V, loc)
        assert isinstance(r, ConstraintGroup)
        assert r.group_constraints[0].literal == "animal"

    def test_single_abstract_descendant(self, V, loc):
        r = _parse("<Animal*>", V, loc)
        assert isinstance(r.group_constraints[0], TemplateConstraintAbstractDescendants)

    def test_single_ascendant(self, V, loc):
        r = _parse("<^Animal>", V, loc)
        assert isinstance(r.group_constraints[0], TemplateConstraintAscendants)

    def test_single_non_type(self, V, loc):
        r = _parse("<Literal:int>", V, loc)
        assert isinstance(r.group_constraints[0], NonTypeTemplateConstraintFormula)

    def test_single_literal_value(self, V, loc):
        r = _parse("<3>", V, loc)
        assert isinstance(r.group_constraints[0], LiteralValueConstraintFormula)

    def test_single_and(self, V, loc):
        r = _parse("<And(A, B)>", V, loc)
        assert isinstance(r.group_constraints[0], TemplateConstraintAnd)

    def test_single_not(self, V, loc):
        r = _parse("<Not(Animal)>", V, loc)
        assert isinstance(r.group_constraints[0], TemplateConstraintNot)

    def test_repr_single(self, V, loc):
        assert repr(_parse("<Animal>", V, loc)) == "<Animal>"

    # --- multiple elements ---

    def test_two_concepts(self, V, loc):
        r = _parse("<Animal, Plant>", V, loc)
        assert isinstance(r, ConstraintGroup)
        assert len(r.group_constraints) == 2
        assert r.group_constraints[0].literal == "Animal"
        assert r.group_constraints[1].literal == "Plant"

    def test_three_mixed_elements(self, V, loc):
        r = _parse("<Animal, Literal:int, Plant*>", V, loc)
        assert len(r.group_constraints) == 3
        assert isinstance(r.group_constraints[0], TemplateConstraintDescendants)
        assert isinstance(r.group_constraints[1], NonTypeTemplateConstraintFormula)
        assert isinstance(r.group_constraints[2], TemplateConstraintAbstractDescendants)

    def test_repr_multiple(self, V, loc):
        assert repr(_parse("<Animal, Plant>", V, loc)) == "<Animal, Plant>"

    def test_unconstrained_element(self, V, loc):
        """',' at the start produces Unconstrained as first slot (comma is a delimiter)."""
        r = _parse("<Animal, Plant>", V, loc)
        assert len(r.group_constraints) == 2

    # --- empty angle brackets: ConstraintGroup cannot have zero elements ---

    def test_empty_brackets_produce_one_unconstrained_not_zero(self, V, loc):
        """
        '<>' → ConstraintGroup((Unconstrained,)).

        The grammar guarantees ≥ 1 element: _parse_non_structure_constraint()
        is called first (mandatory), and it returns Unconstrained when it sees '>'.
        The group is therefore never truly empty.
        """
        r = _parse("<>", V, loc)
        assert isinstance(r, ConstraintGroup)
        assert len(r.group_constraints) == 1, "<> must produce exactly 1 slot (Unconstrained), not 0"
        assert isinstance(r.group_constraints[0], Unconstrained)

    def test_no_brackets_vs_empty_brackets_differ(self, V, loc):
        """
        'Vector' and 'Vector<>' differ in literal_template_formulae:
          'Vector'   → () – no bracket notation at all
          'Vector<>' → (Unconstrained,) – explicit bracket notation, one slot
        Both are valid because Vector accepts exactly 1 template argument.
        """
        no_brackets = _parse("Vector", V, loc)
        empty_brackets = _parse("Vector<>", V, loc)
        assert no_brackets.literal_template_formulae == (), "No-bracket form must have empty tuple"
        assert len(empty_brackets.literal_template_formulae) == 1, "Empty-bracket form must have exactly 1 slot"
        assert isinstance(empty_brackets.literal_template_formulae[0], Unconstrained)

    def test_empty_brackets_as_template_arg_unconstrained_slot(self, V, loc):
        """Vector<> fills the single required slot with Unconstrained → valid."""
        r = _parse("Vector<>", V, loc)
        assert isinstance(r, TemplateConstraintDescendants)
        assert r.literal == "Vector"
        assert len(r.literal_template_formulae) == 1
        assert isinstance(r.literal_template_formulae[0], Unconstrained)

    def test_empty_brackets_all_hierarchy_variants(self, V, loc):
        """All five hierarchy operators work with empty brackets on a 1-arg concept."""
        r_desc = _parse("Vector<>", V, loc)
        r_aDesc = _parse("Vector<>*", V, loc)
        r_self = _parse("Vector<>.", V, loc)
        r_asc = _parse("^Vector<>", V, loc)
        r_aAsc = _parse("^Vector<>*", V, loc)
        assert isinstance(r_desc, TemplateConstraintDescendants)
        assert isinstance(r_aDesc, TemplateConstraintAbstractDescendants)
        assert isinstance(r_self, TemplateConstraintSelf)
        assert isinstance(r_asc, TemplateConstraintAscendants)
        assert isinstance(r_aAsc, TemplateConstraintAbstractAscendants)
        for r in [r_desc, r_aDesc, r_self, r_asc, r_aAsc]:
            assert len(r.literal_template_formulae) == 1
            assert isinstance(r.literal_template_formulae[0], Unconstrained)

    # --- structure constraints cannot appear inside ConstraintGroup ---

    def test_nested_constraint_group_inside_group_raises(self, V, loc):
        """
        '<' starts a structure constraint; _parse_non_structure_constraint cannot
        handle structure constraints, so '<' inside a group hits the hierarchy-literal
        path and fails because '<' is not .isalpha().
        """
        with pytest.raises(CHSyntaxError):
            _parse("<<Animal>>", V, loc)

    def test_conj_inside_group_raises(self, V, loc):
        """Conj( is a structure operator; not valid inside a ConstraintGroup slot."""
        with pytest.raises(CHSyntaxError):
            _parse("<Conj(<A>, <B>)>", V, loc)


# ===========================================================================
# 8.  Structure operators  (Conj, Disj, Neg)
# ===========================================================================


class TestStructureOperators:
    """
    Structure operators take StructureConstraintFormula arguments only.
    The structure constraint list requires ≥ 1 structure constraint;
    Conj() / Disj() / Neg() therefore raise CHSyntaxError because ')'
    does not match any structure constraint opener.
    """

    # --- StructureConjunction ---

    def test_conj_two_groups(self, V, loc):
        r = _parse("Conj(<Animal>, <Plant>)", V, loc)
        assert isinstance(r, StructureConjunction)
        assert isinstance(r, StructureConstraintFormula)
        assert len(r.structure_constraints) == 2
        assert isinstance(r.structure_constraints[0], ConstraintGroup)
        assert isinstance(r.structure_constraints[1], ConstraintGroup)

    def test_conj_three_groups(self, V, loc):
        r = _parse("Conj(<A>, <B>, <C>)", V, loc)
        assert len(r.structure_constraints) == 3

    def test_conj_single_group(self, V, loc):
        r = _parse("Conj(<Animal>)", V, loc)
        assert isinstance(r, StructureConjunction)
        assert len(r.structure_constraints) == 1

    def test_conj_with_multi_element_groups(self, V, loc):
        r = _parse("Conj(<A, B>, <C, D>)", V, loc)
        assert isinstance(r, StructureConjunction)
        assert len(r.structure_constraints[0].group_constraints) == 2
        assert len(r.structure_constraints[1].group_constraints) == 2

    def test_conj_repr(self, V, loc):
        assert repr(_parse("Conj(<Animal>, <Plant>)", V, loc)) == "Conj(<Animal>, <Plant>)"

    # --- StructureDisjunction ---

    def test_disj_two_groups(self, V, loc):
        r = _parse("Disj(<Animal>, <Plant>)", V, loc)
        assert isinstance(r, StructureDisjunction)
        assert len(r.structure_constraints) == 2

    def test_disj_single_group(self, V, loc):
        r = _parse("Disj(<Animal>)", V, loc)
        assert isinstance(r, StructureDisjunction)
        assert len(r.structure_constraints) == 1

    def test_disj_repr(self, V, loc):
        assert repr(_parse("Disj(<Animal>, <Plant>)", V, loc)) == "Disj(<Animal>, <Plant>)"

    # --- StructureNegation ---

    def test_neg_single_group(self, V, loc):
        r = _parse("Neg(<Animal>)", V, loc)
        assert isinstance(r, StructureNegation)
        assert isinstance(r.structure_constraint, ConstraintGroup)

    def test_neg_multi_element_group(self, V, loc):
        r = _parse("Neg(<Animal, Plant>)", V, loc)
        assert isinstance(r.structure_constraint, ConstraintGroup)
        assert len(r.structure_constraint.group_constraints) == 2

    def test_neg_repr(self, V, loc):
        assert repr(_parse("Neg(<Animal>)", V, loc)) == "Neg(<Animal>)"

    # --- nesting structure operators inside each other ---

    def test_conj_with_nested_disj(self, V, loc):
        r = _parse("Conj(<A>, Disj(<B>, <C>))", V, loc)
        assert isinstance(r, StructureConjunction)
        assert isinstance(r.structure_constraints[0], ConstraintGroup)
        assert isinstance(r.structure_constraints[1], StructureDisjunction)
        inner = r.structure_constraints[1]
        assert len(inner.structure_constraints) == 2

    def test_neg_wrapping_conj(self, V, loc):
        r = _parse("Neg(Conj(<A>, <B>))", V, loc)
        assert isinstance(r, StructureNegation)
        assert isinstance(r.structure_constraint, StructureConjunction)

    def test_conj_of_neg_and_group(self, V, loc):
        r = _parse("Conj(Neg(<A>), <B>)", V, loc)
        assert isinstance(r, StructureConjunction)
        assert isinstance(r.structure_constraints[0], StructureNegation)
        assert isinstance(r.structure_constraints[1], ConstraintGroup)

    def test_deeply_nested_structure(self, V, loc):
        r = _parse("Disj(Conj(<A>, Neg(<B>)), Neg(Disj(<C>, <D>)))", V, loc)
        assert isinstance(r, StructureDisjunction)
        conj = r.structure_constraints[0]
        neg_disj = r.structure_constraints[1]
        assert isinstance(conj, StructureConjunction)
        assert isinstance(neg_disj, StructureNegation)
        assert isinstance(neg_disj.structure_constraint, StructureDisjunction)

    def test_repr_nested(self, V, loc):
        assert repr(_parse("Conj(<A>, Neg(<B>))", V, loc)) == "Conj(<A>, Neg(<B>))"

    # --- structure operators cannot contain non-structure constraints ---

    def test_conj_with_non_structure_raises(self, V, loc):
        """And(A, B) is a non-structure constraint; not valid inside Conj."""
        with pytest.raises(CHSyntaxError):
            _parse("Conj(<A>, And(A, B))", V, loc)

    def test_disj_with_plain_concept_raises(self, V, loc):
        """A bare 'Animal' is a non-structure constraint; invalid inside Disj."""
        with pytest.raises(CHSyntaxError):
            _parse("Disj(<A>, Animal)", V, loc)

    def test_neg_with_non_structure_raises(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse("Neg(Animal)", V, loc)

    # --- empty structure operators raise CHSyntaxError ---

    def test_conj_empty_raises(self, V, loc):
        """
        Conj() has no structure constraint to parse for the mandatory first slot.
        _parse_structure_constraint() sees ')' which matches none of Conj/Disj/Neg/<
        and raises CHSyntaxError.
        """
        with pytest.raises(CHSyntaxError):
            _parse("Conj()", V, loc)

    def test_disj_empty_raises(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse("Disj()", V, loc)

    def test_neg_empty_raises(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse("Neg()", V, loc)

    def test_conj_whitespace_only_raises(self, V, loc):
        """Whitespace does not create a structure constraint."""
        with pytest.raises(CHSyntaxError):
            _parse("Conj( )", V, loc)

    def test_disj_whitespace_only_raises(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse("Disj( )", V, loc)

    def test_neg_whitespace_only_raises(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse("Neg( )", V, loc)

    # --- unclosed structure operators ---

    def test_unclosed_conj_raises(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse("Conj(<Animal>", V, loc)

    def test_unclosed_neg_raises(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse("Neg(<Animal>", V, loc)

    def test_trailing_content_after_structure(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse("Conj(<A>, <B>) trailing", V, loc)


# ===========================================================================
# 9.  Validator semantic validation
# ===========================================================================


class TestValidatorValidation:
    """
    The validator is called during TemplateConstraintHierarchyOperator construction.
    Errors are CHSemanticError (not CHSyntaxError).
    """

    def test_concept_no_template_args_valid(self, V, loc):
        """0-arg concept without brackets — no validator template-count check."""
        r = _parse("Animal", V, loc)
        assert r.literal == "Animal"

    def test_template_variable_valid(self, V, loc):
        r = _parse("t", V, loc)
        assert r.literal == "t"

    def test_one_arg_concept_correct_count_valid(self, V, loc):
        r = _parse("Vector<Animal>", V, loc)
        assert r.literal == "Vector"

    def test_two_arg_concept_correct_count_valid(self, V, loc):
        r = _parse("Map<Animal, Plant>", V, loc)
        assert r.literal == "Map"

    def test_zero_arg_concept_with_arg_raises_semantic(self, V, loc):
        with pytest.raises(CHSemanticError):
            _parse("Animal<Plant>", V, loc)

    def test_zero_arg_concept_with_unconstrained_raises_semantic(self, V, loc):
        with pytest.raises(CHSemanticError):
            _parse("Animal<>", V, loc)

    def test_one_arg_concept_too_many_raises_semantic(self, V, loc):
        with pytest.raises(CHSemanticError):
            _parse("Vector<A, B>", V, loc)

    def test_one_arg_concept_too_few_raises_semantic(self, V, loc):
        """Map requires 2; Map<A> gives 1 → CHSemanticError."""
        with pytest.raises(CHSemanticError):
            _parse("Map<A>", V, loc)

    def test_template_variable_with_args_raises_semantic(self, V, loc):
        with pytest.raises(CHSemanticError):
            _parse("t<Animal>", V, loc)

    def test_template_variable_with_unconstrained_arg_raises_semantic(self, V, loc):
        with pytest.raises(CHSemanticError):
            _parse("t<>", V, loc)

    def test_unknown_name_not_concept_not_variable_raises_semantic(self, V, loc):
        """
        A name that is neither concept nor template variable raises CHSemanticError.
        With the test validator this only occurs for names that are not alpha-starting
        (caught by CHSyntaxError at parse level). We test this via a custom validator.
        """

        class _RejectAllValidator(_TestValidator):
            def is_concept(self, name: str) -> bool:
                return False

            def is_template_variable(self, name: str) -> bool:
                return False

        with pytest.raises(CHSemanticError):
            _parse("Animal", _RejectAllValidator(), loc)

    def test_ascendants_concept_valid(self, V, loc):
        r = _parse("^Animal", V, loc)
        assert isinstance(r, TemplateConstraintAscendants)

    def test_self_template_variable_valid(self, V, loc):
        r = _parse("keyType.", V, loc)
        assert isinstance(r, TemplateConstraintSelf)


# ===========================================================================
# 10.  Whitespace handling
# ===========================================================================


class TestWhitespace:
    """
    The separator ', ' is consumed literally (comma then one space).
    parse_constraint() calls skip_whitespace() at entry, so leading whitespace
    before each sub-expression is fine.
    Trailing whitespace before a closing ')' or '>' is NOT automatically consumed
    by consume() and will cause a CHSyntaxError.
    """

    def test_leading_whitespace_before_concept(self, V, loc):
        assert isinstance(_parse("   Animal", V, loc), TemplateConstraintDescendants)

    def test_leading_whitespace_before_literal_type(self, V, loc):
        assert isinstance(_parse("  Literal:int", V, loc), NonTypeTemplateConstraintFormula)

    def test_leading_whitespace_before_bool(self, V, loc):
        assert isinstance(_parse("  true", V, loc), LiteralValueConstraintFormula)

    def test_leading_whitespace_before_number(self, V, loc):
        assert isinstance(_parse("  42", V, loc), LiteralValueConstraintFormula)

    def test_leading_whitespace_before_structure(self, V, loc):
        assert isinstance(_parse("  <Animal>", V, loc), ConstraintGroup)

    def test_trailing_whitespace_accepted_at_top_level(self, V, loc):
        """parse_constraint_definition does skip_whitespace before the eof check."""
        assert isinstance(_parse("Animal   ", V, loc), TemplateConstraintDescendants)

    def test_leading_whitespace_inside_and_first_operand(self, V, loc):
        """And( Animal, Plant) — leading space before first operand is fine."""
        r = _parse("And( Animal, Plant)", V, loc)
        assert len(r.sub_formulae) == 2

    def test_extra_whitespace_after_separator_is_skipped(self, V, loc):
        """After consuming ', ', parse_constraint calls skip_whitespace() so
        'And(Animal,  Plant)' (two spaces after comma) is valid."""
        r = _parse("And(Animal,  Plant)", V, loc)
        assert len(r.sub_formulae) == 2

    def test_space_before_comma_breaks_separator(self, V, loc):
        """' ,' does not match the required ', ' separator → error."""
        with pytest.raises(CHSyntaxError):
            _parse("And(Animal , Plant)", V, loc)

    def test_trailing_space_before_closing_paren_raises(self, V, loc):
        """'Animal )' — space before ')' is not consumed before consume(')') → error."""
        with pytest.raises(CHSyntaxError):
            _parse("And(Animal )", V, loc)

    def test_empty_and_with_whitespace_still_raises(self, V, loc):
        """And( ) → Unconstrained slot → not TypeTemplateConstraintFormula → CHSyntaxError."""
        with pytest.raises(CHSyntaxError):
            _parse("And( )", V, loc)

    def test_empty_conj_with_whitespace_still_raises(self, V, loc):
        """Conj( ) → ) does not start a structure constraint → CHSyntaxError."""
        with pytest.raises(CHSyntaxError):
            _parse("Conj( )", V, loc)


# ===========================================================================
# 11.  Error cases  (syntax errors at the top level)
# ===========================================================================


class TestErrorCases:
    @pytest.mark.parametrize(
        "text",
        [
            "Animal extra",
            "Animal* redundant",
            "true extra",
            "42 extra",
            '"hello" trailing',
            "Literal:int trailing",
            "And(Animal, Plant) extra",
            "<Animal> trailing",
            "Conj(<A>, <B>) trailing",
        ],
    )
    def test_trailing_content_raises(self, V, loc, text):
        with pytest.raises(CHSyntaxError):
            _parse(text, V, loc)

    @pytest.mark.parametrize(
        "text",
        ["Literal:char", "Literal:float", "Literal:Boolean", "Literal:INT", "Literal:"],
    )
    def test_invalid_literal_type_raises(self, V, loc, text):
        with pytest.raises(CHSyntaxError):
            _parse(text, V, loc)

    @pytest.mark.parametrize(
        "text",
        ["^Animal.", "^Animal*.", "Animal*.", "Animal.*"],
    )
    def test_invalid_hierarchy_combos_raise(self, V, loc, text):
        with pytest.raises(CHSyntaxError):
            _parse(text, V, loc)

    def test_bare_minus_raises(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse("-", V, loc)

    def test_unterminated_string_raises(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse('"unterminated', V, loc)

    def test_underscore_initial_raises_syntax(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse("_Animal", V, loc)

    def test_unclosed_angle_bracket_raises(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse("<Animal", V, loc)

    def test_unclosed_conj_raises(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse("Conj(<Animal>", V, loc)

    def test_structure_inside_non_structure_list_raises(self, V, loc):
        """<Animal> is a structure constraint; And only accepts non-structure → CHSyntaxError."""
        with pytest.raises(CHSyntaxError):
            _parse("And(<Animal>)", V, loc)

    def test_non_structure_inside_structure_list_raises(self, V, loc):
        with pytest.raises(CHSyntaxError):
            _parse("Conj(Animal, <B>)", V, loc)


# ===========================================================================
# 12.  Complex combinations
# ===========================================================================


class TestComplexCombinations:
    def test_and_inside_constraint_group(self, V, loc):
        """<And(A, B), C> — a boolean operator as a group element."""
        r = _parse("<And(A, B), C>", V, loc)
        assert isinstance(r, ConstraintGroup)
        assert isinstance(r.group_constraints[0], TemplateConstraintAnd)
        assert isinstance(r.group_constraints[1], TemplateConstraintDescendants)

    def test_not_inside_constraint_group(self, V, loc):
        r = _parse("<Not(Animal), Plant*>", V, loc)
        assert isinstance(r.group_constraints[0], TemplateConstraintNot)
        assert isinstance(r.group_constraints[1], TemplateConstraintAbstractDescendants)

    def test_conj_groups_with_complex_elements(self, V, loc):
        """Conj(<And(A, B)>, <^C*>) — boolean operator inside each group."""
        r = _parse("Conj(<And(A, B)>, <^C*>)", V, loc)
        assert isinstance(r, StructureConjunction)
        g0 = r.structure_constraints[0]
        g1 = r.structure_constraints[1]
        assert isinstance(g0.group_constraints[0], TemplateConstraintAnd)
        assert isinstance(g1.group_constraints[0], TemplateConstraintAbstractAscendants)

    def test_deeply_nested_boolean_in_structure(self, V, loc):
        """Neg(<Or(A, Not(B))>) — Not wraps an Or which contains a Not."""
        r = _parse("Neg(<Or(A, Not(B))>)", V, loc)
        assert isinstance(r, StructureNegation)
        inner = r.structure_constraint.group_constraints[0]
        assert isinstance(inner, TemplateConstraintOr)
        assert isinstance(inner.sub_formulae[1], TemplateConstraintNot)

    def test_vector_with_complex_arg_inside_group(self, V, loc):
        """<Vector<And(A, B)>> — Vector with an And as its template arg."""
        r = _parse("<Vector<And(A, B)>>", V, loc)
        assert isinstance(r, ConstraintGroup)
        vec = r.group_constraints[0]
        assert isinstance(vec, TemplateConstraintDescendants)
        assert vec.literal == "Vector"
        assert isinstance(vec.literal_template_formulae[0], TemplateConstraintAnd)

    def test_conj_with_empty_brackets_in_groups(self, V, loc):
        """Conj(<Vector<>>, <Animal>) — the first group contains Vector with Unconstrained arg."""
        r = _parse("Conj(<Vector<>>, <Animal>)", V, loc)
        g0 = r.structure_constraints[0]
        vec = g0.group_constraints[0]
        assert isinstance(vec.literal_template_formulae[0], Unconstrained)

    def test_all_literal_types_in_group(self, V, loc):
        r = _parse("<Literal:boolean, Literal:int, Literal:number, Literal:string>", V, loc)
        assert len(r.group_constraints) == 4
        assert all(isinstance(e, NonTypeTemplateConstraintFormula) for e in r.group_constraints)

    def test_literal_values_in_group(self, V, loc):
        r = _parse('<true, false, 3, -1, 3.14, "hello">', V, loc)
        assert len(r.group_constraints) == 6
        assert all(isinstance(e, LiteralValueConstraintFormula) for e in r.group_constraints)

    def test_conj_of_disj_and_neg(self, V, loc):
        r = _parse("Conj(Disj(<A>, <B>), Neg(<C>))", V, loc)
        assert isinstance(r, StructureConjunction)
        assert isinstance(r.structure_constraints[0], StructureDisjunction)
        assert isinstance(r.structure_constraints[1], StructureNegation)

    def test_abstract_ascendant_in_group_with_complex_conj(self, V, loc):
        """Conj(<^Animal*, Not(Plant)>, Neg(<Fungus>))."""
        r = _parse("Conj(<^Animal*, Not(Plant)>, Neg(<Fungus>))", V, loc)
        assert isinstance(r, StructureConjunction)
        g = r.structure_constraints[0]
        assert isinstance(g.group_constraints[0], TemplateConstraintAbstractAscendants)
        assert isinstance(g.group_constraints[1], TemplateConstraintNot)

    def test_template_variable_in_group_and_structure(self, V, loc):
        """Template variables are valid hierarchy operators inside ConstraintGroups."""
        r = _parse("Conj(<animal, plant>, <^fungus*>)", V, loc)
        g0 = r.structure_constraints[0]
        assert g0.group_constraints[0].literal == "animal"
        assert g0.group_constraints[1].literal == "plant"
        g1 = r.structure_constraints[1]
        assert isinstance(g1.group_constraints[0], TemplateConstraintAbstractAscendants)
        assert g1.group_constraints[0].literal == "fungus"

    def test_repr_round_trip_simple(self, V, loc):
        """repr output matches the canonical serialized form for simple formulas."""
        assert repr(_parse("And(Animal, ^Plant)", V, loc)) == "And(Animal, ^Plant)"
        assert repr(_parse("Conj(<A, B>, Neg(<C>))", V, loc)) == "Conj(<A, B>, Neg(<C>))"
        assert repr(_parse("<Vector<Animal>, Plant*>", V, loc)) == "<Vector<Animal>, Plant*>"

    def test_map_constraint_with_operator_arg(self, V, loc):
        """Map<^animal, Or(A, B)> — 2-arg concept with varied arg types."""
        r = _parse("Map<^animal, Or(A, B)>", V, loc)
        assert r.literal == "Map"
        assert isinstance(r.literal_template_formulae[0], TemplateConstraintAscendants)
        assert isinstance(r.literal_template_formulae[1], TemplateConstraintOr)
