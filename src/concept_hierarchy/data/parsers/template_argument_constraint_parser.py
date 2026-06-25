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

from __future__ import annotations

from concept_hierarchy.data.parsers.string_parser import StringParser
from concept_hierarchy.data.template_argument_constraints.constraint_formula import (
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
    TemplateConstraintHierarchyOperator,
    TemplateConstraintNot,
    TemplateConstraintOr,
    TemplateConstraintSelf,
    TypeTemplateConstraintFormula,
    Unconstrained,
)
from concept_hierarchy.data.template_argument_constraints.simplify_constraints import simplify_formula
from concept_hierarchy.errors import CHSemanticError, CHSyntaxError, LocationId
from concept_hierarchy.utils import Reference, is_integer, is_number


def parse_constraint_definition(
    constraint_def: bool | int | float | str,
    validator: TemplateConstraintFormulaValidator,
    location_id: LocationId,
    *,
    allow_unconstrained: bool,
) -> TemplateConstraintFormula:
    """
    Parse a bare constraint expression (no surrounding quotes) and return the corresponding TemplateConstraintFormula.

    Raises CHSyntaxError if there is unexpected trailing content after a valid formula,
    which usually indicates a syntax error.
    """
    parser = ConstraintParser(constraint_def, validator, location_id)
    formula = parser.parse_constraint(allow_unconstrained)
    parser.check_finished()
    return formula


class ConstraintParser(StringParser):
    """
    Recursive-descent parser for the grammar:

        constraint                 ::= structureConstraint | nonStructureConstraint
        structureConstraint        ::= constraintGroup | structureOperator
        constraintGroup            ::= '<' nonStructureConstraint (', ' nonStructureConstraint)* '>'
        structureOperator          ::= conjExpr | disjExpr | negExpr
        conjExpr                   ::= 'Conj(' structureConstraintList ')'
        disjExpr                   ::= 'Conj(' structureConstraintList ')'
        negExpr                    ::=  'Neg(' structureConstraint     ')'
        structureConstraintList    ::= structureConstraint (', ' structureConstraint)*
        nonStructureConstraint     ::= unconstrained | nonTypeConstraint | typeConstraint
        unconstrained              ::= ''
        nonTypeConstraint          ::= literalConstraint | literalValue
        literalConstraint          ::= 'Literal:' ('boolean' | 'int' | 'number' | 'string')
        literalValue               ::= stringValue | boolValue | numberValue
        stringValue                ::= '"' (char | '\\"')* '"'
        boolValue                  ::= 'true' | 'false'
        numberValue                ::= '-'? DIGITS ('.' DIGITS)?
        typeConstraint             ::= operatorCall | hierarchyLiteral
        operatorCall               ::= andExpr | orExpr | notExpr
        andExpr                    ::= 'And(' nonStructureConstraintList ')'
        orExpr                     ::=  'Or(' nonStructureConstraintList ')'
        notExpr                    ::= 'Not(' nonStructureConstraint     ')'
        nonStructureConstraintList ::= nonStructureConstraint (', ' nonStructureConstraint)*
        hierarchyLiteral           ::= ('^'? literalName ('*' | '.')?)
        literalName                ::= letterStartingName constraintGroup?
    """

    _LITERAL_TOKENS = [
        ("boolean", "bool"),
        ("number", "float"),
        ("integer", "int"),
        ("string", "string"),
    ]

    def __init__(
        self,
        constraint_def: bool | int | float | str,
        validator: TemplateConstraintFormulaValidator,
        location_id: LocationId,
    ) -> None:
        constraint_str = str(constraint_def)
        if isinstance(constraint_def, bool):
            constraint_str = "true" if constraint_def else "false"
        super().__init__(constraint_str, location_id)
        self.validator = validator

    # ------------------------------------------------------------------
    # Grammar rules
    # ------------------------------------------------------------------

    def parse_constraint(self, allow_unconstrained: bool) -> TemplateConstraintFormula:
        """Entry point for a single constraint expression."""
        self.skip_whitespace()
        if any(self.starts_with(x) for x in ["Conj(", "Disj(", "Neg(", "<"]):
            res = self._parse_structure_constraint(allow_unconstrained)
        else:
            res = self._parse_non_structure_constraint(allow_unconstrained)
        return simplify_formula(res)

    def _parse_non_structure_constraint(self, allow_unconstrained: bool) -> NonStructureConstraintFormula:
        self.skip_whitespace()

        # Empty / whitespace-only input or a closing delimiter means "no constraint".
        if self.eof() or self.peek() in (")", ">", ","):
            if not allow_unconstrained:
                raise CHSemanticError("An unconstrained constraint is not allowed here", location_id=self.location_id)
            return Unconstrained(self.location_id)

        # Literal (non-type) constraint  ->  Literal:boolean / int / number / string
        if self.starts_with("Literal:"):
            return self._parse_non_type_constraint()

        # Boolean operator wrappers
        if self.starts_with("And("):
            return self._parse_and(allow_unconstrained)
        if self.starts_with("Or("):
            return self._parse_or(allow_unconstrained)
        if self.starts_with("Not("):
            return self._parse_not(allow_unconstrained)

        # Literal *value* constraints  â†’  "...", true, false, 3, -1, 3.14
        if self.peek() == '"':
            return LiteralValueConstraintFormula("string", self.parse_string_literal(True), self.location_id)
        if self.starts_with("true") or self.starts_with("false"):
            return LiteralValueConstraintFormula("bool", self.parse_bool_literal(), self.location_id)
        if self.peek().isdigit() or (
            self.peek() == "-" and self.pos + 1 < len(self.text) and self.text[self.pos + 1].isdigit()
        ):
            start = self.pos
            raw = self.parse_number_literal()
            # Prefer integer to float when both match (e.g. bare "3")
            ref = Reference()
            if is_integer(raw, ref):
                return LiteralValueConstraintFormula("int", raw, self.location_id)
            if is_number(raw, ref):
                return LiteralValueConstraintFormula("float", raw, self.location_id)
            raise CHSyntaxError(
                f"Invalid numeric literal at position {start}: {raw!r}",
                location_id=self.location_id,
            )

        # Hierarchy literal  (T / T* / T. / ^T / ^T*)
        return self._parse_hierarchy_literal()

    def _parse_structure_constraint(self, allow_unconstrained: bool) -> StructureConstraintFormula:
        # Structure operator wrappers
        if self.starts_with("Conj("):
            return self._parse_conj()
        if self.starts_with("Disj("):
            return self._parse_disj()
        if self.starts_with("Neg("):
            return self._parse_neg()

        # Structure group
        if not self.starts_with("<"):
            raise CHSyntaxError(
                f"Expected a structure constraint at {self.remaining()} but it doesn't match any structure constraint!",
                location_id=self.location_id,
            )
        return self._parse_constraint_group(allow_unconstrained)

    # --- structure operators ----------------------------------------------

    def _parse_conj(self) -> StructureConjunction:
        self.consume("Conj(")
        args = self._parse_structure_constraint_list()
        self.consume(")")
        return StructureConjunction(self.location_id, args)

    def _parse_disj(self) -> StructureDisjunction:
        self.consume("Disj(")
        args = self._parse_structure_constraint_list()
        self.consume(")")
        return StructureDisjunction(self.location_id, args)

    def _parse_neg(self) -> StructureNegation:
        self.consume("Neg(")
        arg = self._parse_structure_constraint()
        self.consume(")")
        return StructureNegation(self.location_id, arg)

    def _parse_structure_constraint_list(self) -> tuple[StructureConstraintFormula, ...]:
        """One or more structure-constraints separated by ', '."""
        items = [self._parse_structure_constraint()]
        while self.try_consume(", "):
            items.append(self._parse_structure_constraint())
        return tuple(items)

    # --- structure group --------------------------------------------------

    def _parse_constraint_group(self, allow_unconstrained: bool) -> ConstraintGroup:
        self.consume("<")
        group_elements = [self._parse_non_structure_constraint(allow_unconstrained)]
        while self.try_consume(", "):
            group_elements.append(self._parse_non_structure_constraint(allow_unconstrained))
        self.consume(">")
        return ConstraintGroup(self.location_id, tuple(group_elements))

    # --- boolean operators ------------------------------------------------

    def _parse_and(self, allow_unconstrained: bool) -> TemplateConstraintAnd:
        self.consume("And(")
        args = self._parse_non_structure_constraint_list(allow_unconstrained)
        self.consume(")")
        return TemplateConstraintAnd(self.location_id, args)

    def _parse_or(self, allow_unconstrained: bool) -> TemplateConstraintOr:
        self.consume("Or(")
        args = self._parse_non_structure_constraint_list(allow_unconstrained)
        self.consume(")")
        return TemplateConstraintOr(self.location_id, args)

    def _parse_not(self, allow_unconstrained: bool) -> TemplateConstraintNot:
        self.consume("Not(")
        arg = self._parse_non_structure_constraint(allow_unconstrained)
        self.consume(")")
        return TemplateConstraintNot(self.location_id, arg)

    def _parse_non_structure_constraint_list(
        self, allow_unconstrained: bool
    ) -> tuple[NonStructureConstraintFormula, ...]:
        """One or more non-structure-constraints separated by ', '."""
        items = [self._parse_non_structure_constraint(allow_unconstrained)]
        while self.try_consume(", "):
            items.append(self._parse_non_structure_constraint(allow_unconstrained))
        return tuple(items)

    # --- non-type ---------------------------------------------------------

    def _parse_non_type_constraint(self) -> NonTypeTemplateConstraintFormula:
        self.consume("Literal:")
        for token, ctype in ConstraintParser._LITERAL_TOKENS:
            if self.starts_with(token):
                self.consume(token)
                return NonTypeTemplateConstraintFormula(ctype, self.location_id)
        raise CHSyntaxError(
            f"Unknown literal type constraint at position {self.pos}: {self.remaining()!r}. Allowed only "
            f"{', '.join(x[0] for x in ConstraintParser._LITERAL_TOKENS)}",
            location_id=self.location_id,
        )

    # --- hierarchy literals -----------------------------------------------

    def _parse_type_constraint(self, allow_unconstrained: bool) -> TypeTemplateConstraintFormula:
        """
        Like parse_constraint(), but restricted to TypeTemplateConstraintFormula.

        Not called from _parse_literal_name (which now accepts any constraint, including LiteralValueConstraintFormula,
        so that e.g. Vector<3> is valid).
        Retained as a utility for callers that explicitly require a type constraint and want a clear error
        rather than a silent wrong type.
        """
        if self.starts_with("Literal:"):
            raise CHSyntaxError(
                f"A 'Literal:...' constraint is not valid as a template argument at position {self.pos}. "
                f"Only type constraints (T, T*, T., ^T, ^T*, And(...), Or(...), Not(...)) are allowed here.",
                location_id=self.location_id,
            )
        formula = self.parse_constraint(allow_unconstrained)
        if not isinstance(formula, TypeTemplateConstraintFormula):
            raise CHSyntaxError(
                f"Expected a TypeTemplateConstraintFormula as a template argument, got {type(formula).__name__!r}",
                location_id=self.location_id,
            )
        return formula

    def _parse_type_constraint_list(self, allow_unconstrained: bool) -> tuple[TypeTemplateConstraintFormula, ...]:
        """One or more type-template-constraints separated by ', '."""
        items = [self._parse_type_constraint(allow_unconstrained)]
        while self.try_consume(", "):
            items.append(self._parse_type_constraint(allow_unconstrained))
        return tuple(items)

    def _parse_hierarchy_literal(self) -> TemplateConstraintHierarchyOperator:
        """
        Handles all five hierarchy-operator variants:

            T       ->  TemplateConstraintDescendants          (no abstract)
            T*      ->  TemplateConstraintAbstractDescendants  (include abstract)
            T.      ->  TemplateConstraintSelf                 (exact type / abstract ok)
            ^T      ->  TemplateConstraintAscendants           (no abstract)
            ^T*     ->  TemplateConstraintAbstractAscendants   (include abstract)
        """
        is_ascendant = self.try_consume("^")

        ch_type_name, t_arg_formulae = self._parse_literal_name()

        if is_ascendant:
            if self.peek() == ".":
                raise CHSyntaxError(
                    '"^{0}." is an invalid constraint formula! Choose either\n\t"^{0}" to mean the ascendants of {0},'
                    '\n\t"{0}." to mean only {0},\n\tor "^{0}*" to mean the ascendants including abstract ones.'
                    "".format(ch_type_name),
                    location_id=self.location_id,
                )
            if self.peek(2) == "*.":
                raise CHSyntaxError(
                    '"^{0}." is an invalid constraint formula! Choose either\n\t"^{0}" to mean the ascendants of {0},'
                    '\n\t"{0}." to mean only {0},\n\tor "^{0}*" to mean the ascendants including abstract ones.'
                    "".format(ch_type_name),
                    location_id=self.location_id,
                )
            elif self.peek() == "*":
                self.pos += 1
                return TemplateConstraintAbstractAscendants(
                    ch_type_name, t_arg_formulae, self.validator, self.location_id
                )
            else:
                return TemplateConstraintAscendants(ch_type_name, t_arg_formulae, self.validator, self.location_id)
        else:
            if self.peek(2) in ["*.", ".*"]:
                raise CHSyntaxError(
                    '"{0}.*" and "{0}*." are invalid constraint formulae! Choose either\n\t"{0}" to mean the '
                    'descendants of {0},\n\t"{0}." to mean only {0}, or\n\t"{0}*" to mean the descendants including '
                    "abstract ones.".format(ch_type_name),
                    location_id=self.location_id,
                )
            elif self.peek() == "*":
                self.pos += 1
                return TemplateConstraintAbstractDescendants(
                    ch_type_name, t_arg_formulae, self.validator, self.location_id
                )
            elif self.peek() == ".":
                self.pos += 1
                return TemplateConstraintSelf(ch_type_name, t_arg_formulae, self.validator, self.location_id)
            else:
                return TemplateConstraintDescendants(ch_type_name, t_arg_formulae, self.validator, self.location_id)

    def _parse_literal_name(self) -> tuple[str, tuple[NonStructureConstraintFormula, ...] | None]:
        """
        Parse a letterStartingName with an optional '<' template-constraint-args '>'.

        Returns
        -------
        base_name : str
            The bare concept/variable name, e.g. ``'Collection'``.
        template_arg_formulae : Optional[Tuple[TemplateConstraintFormula, ...]]
            One parsed formula per template argument (empty tuple when the brackets are empty ``<>``,
            and ``None`` when there are no angle brackets).

        Examples
        --------
            'Animal'                  ->  ('Animal',     [])
            'Animal<>'                ->  ('Animal',     [])
            'Collection<Element>'     ->  ('Collection', [Descendants('Element')])
            'Map<KeyType, ^ValType>'  ->  ('Map',        [Descendants('KeyType'), Ascendants('ValType')])
            'Foo<And(A, B), ^C*>'     ->  ('Foo',        [And([Desc('A'), Desc('B')]), AbsAsc('C')])
        """
        if self.pos >= len(self.text) or not self.text[self.pos].isalpha():
            raise CHSyntaxError(
                f"Expected a letter-starting identifier at position {self.pos}, got {self.remaining()!r}",
                location_id=self.location_id,
            )

        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == "_"):
            self.pos += 1
        literal_name = self.text[start : self.pos]

        template_arg_formulae: tuple[NonStructureConstraintFormula, ...] = ()
        if self.peek() == "<":
            template_arg_formulae = self._parse_constraint_group().group_constraints

        return literal_name, template_arg_formulae
