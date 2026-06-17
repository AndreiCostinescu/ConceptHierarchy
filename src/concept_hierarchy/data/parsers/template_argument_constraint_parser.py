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
from concept_hierarchy.data.parsers.type_parser import TypeParser
from concept_hierarchy.data.template_argument_constraints.constraint_formula import (
    LiteralValueConstraintFormula,
    NonTypeTemplateConstraintFormula,
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
)
from concept_hierarchy.errors import CHSyntaxError, LocationId
from concept_hierarchy.utils import Reference, is_integer, is_number


def parse_constraint_string(
    text: str, validator: TemplateConstraintFormulaValidator, location_id: LocationId
) -> TemplateConstraintFormula:
    """
    Parse a bare constraint expression (no surrounding quotes) and return the corresponding TemplateConstraintFormula.

    Raises RuntimeError if there is unexpected trailing content after a valid formula,
    which usually indicates a syntax error.
    """
    parser = _ConstraintParser(text, validator, location_id)
    formula = parser.parse_constraint()
    parser.skip_whitespace()
    if parser.pos != len(text):
        raise CHSyntaxError(
            f"Unexpected trailing content at position {parser.pos}: {text[parser.pos :]!r}", location_id=location_id
        )
    return formula


class _ConstraintParser(StringParser):
    """
    Recursive-descent parser for the grammar:

        constraint             ::= nonTypeConstraint | literalValueConstraint | operatorCall | hierarchyLiteral
        nonTypeConstraint      ::= 'Literal:' ('boolean' | 'int' | 'number' | 'string')
        literalValueConstraint ::= stringValue | boolValue | numberValue
        stringValue            ::= '"' (char | '\\"')* '"'
        boolValue              ::= 'true' | 'false'
        numberValue            ::= '-'? DIGITS ('.' DIGITS)?
        operatorCall           ::= andExpr | orExpr | notExpr
        andExpr                ::= 'And(' constraintList ')'
        orExpr                 ::= 'Or('  constraintList ')'
        notExpr                ::= 'Not(' constraint    ')'
        constraintList         ::= constraint (', ' constraint)*
        hierarchyLiteral       ::= ('^'? literalName ('*' | '.')?)
        literalName            ::= UPPER_NAME ('<' constraint (',' constraint)* '>')?
    """

    _LITERAL_TOKENS = [
        ("boolean", "bool"),
        ("number", "float"),
        ("int", "int"),
        ("string", "string"),
    ]

    def __init__(self, text: str, validator: TemplateConstraintFormulaValidator, location_id: LocationId) -> None:
        super().__init__(text)
        self.validator = validator
        self.location_id = location_id

    # ------------------------------------------------------------------
    # Grammar rules
    # ------------------------------------------------------------------

    def parse_constraint(self) -> TemplateConstraintFormula:
        """Entry point for a single constraint expression."""
        # Literal (non-type) constraint  â†’  Literal:boolean / int / number / string
        if self.starts_with("Literal:"):
            return self._parse_non_type_constraint()

        # Boolean operator wrappers
        if self.starts_with("And("):
            return self._parse_and()
        if self.starts_with("Or("):
            return self._parse_or()
        if self.starts_with("Not("):
            return self._parse_not()

        # Literal *value* constraints  â†’  "...", true, false, 3, -1, 3.14
        if self.peek() == '"':
            return LiteralValueConstraintFormula("string", self._parse_string_literal(True), self.location_id)
        if self.starts_with("true") or self.starts_with("false"):
            return LiteralValueConstraintFormula("bool", self._parse_bool_literal(), self.location_id)
        if self.peek().isdigit() or (
            self.peek() == "-" and self.pos + 1 < len(self.text) and self.text[self.pos + 1].isdigit()
        ):
            start = self.pos
            raw = self._parse_number_literal()
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

    # --- non-type ---------------------------------------------------------

    def _parse_non_type_constraint(self) -> NonTypeTemplateConstraintFormula:
        self.consume("Literal:")
        for token, ctype in _ConstraintParser._LITERAL_TOKENS:
            if self.starts_with(token):
                self.consume(token)
                return NonTypeTemplateConstraintFormula(ctype, self.location_id)
        raise CHSyntaxError(
            f"Unknown literal type constraint at position {self.pos}: {self.remaining()!r}. Allowed only "
            f"{', '.join(x[0] for x in _ConstraintParser._LITERAL_TOKENS)}",
            location_id=self.location_id,
        )

    # --- boolean operators ------------------------------------------------

    def _parse_and(self) -> TemplateConstraintAnd:
        self.consume("And(")
        args = self._parse_constraint_list()
        self.consume(")")
        return TemplateConstraintAnd(args, self.location_id)

    def _parse_or(self) -> TemplateConstraintOr:
        self.consume("Or(")
        args = self._parse_constraint_list()
        self.consume(")")
        return TemplateConstraintOr(args, self.location_id)

    def _parse_not(self) -> TemplateConstraintNot:
        self.consume("Not(")
        arg = self.parse_constraint()
        self.consume(")")
        return TemplateConstraintNot(arg, self.location_id)

    def _parse_constraint_list(self) -> list[TemplateConstraintFormula]:
        """One or more constraints separated by ', '."""
        items = [self.parse_constraint()]
        while self.starts_with(", "):
            self.consume(", ")
            items.append(self.parse_constraint())
        return items

    # --- hierarchy literals -----------------------------------------------

    def _parse_type_constraint(self) -> TypeTemplateConstraintFormula:
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
        formula = self.parse_constraint()
        if not isinstance(formula, TypeTemplateConstraintFormula):
            raise RuntimeError(
                f"Expected a TypeTemplateConstraintFormula as a template argument, got {type(formula).__name__!r}"
            )
        return formula

    def _parse_hierarchy_literal(self) -> TemplateConstraintFormula:
        """
        Handles all five hierarchy-operator variants:

            T       ->  TemplateConstraintDescendants          (no abstract)
            T*      ->  TemplateConstraintAbstractDescendants  (include abstract)
            T.      ->  TemplateConstraintSelf                 (exact type / abstract ok)
            ^T      ->  TemplateConstraintAscendants           (no abstract)
            ^T*     ->  TemplateConstraintAbstractAscendants   (include abstract)
        """
        is_ascendant = self.peek() == "^"
        if is_ascendant:
            self.pos += 1  # consume '^'

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

    def _parse_literal_name(self) -> tuple[str, tuple[TemplateConstraintFormula, ...] | None]:
        """
        Parse an upperCaseName with an optional '<' template-constraint-args '>'.

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
        if self.pos >= len(self.text) or not self.text[self.pos].isupper():
            raise RuntimeError(f"Expected an uppercase identifier at position {self.pos}, got {self.remaining()!r}")

        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == "_"):
            self.pos += 1
        type_parse_res = TypeParser(self.text[start : self.pos]).parse_types()
        assert len(type_parse_res) == 1
        ch_type = type_parse_res[0]
        if ch_type.full_name != ch_type.clean_name:
            raise RuntimeError(
                f"Error in processing constraints: expected a single type, but mismatch between clean "
                f"{ch_type.clean_name!r} and and full_name {ch_type.full_name!r}!"
            )

        template_arg_formulae: list[TemplateConstraintFormula] | None = None

        if self.peek() == "<":
            self.pos += 1  # consume '<'
            self.skip_whitespace()
            if self.peek() != ">":  # non-empty argument list
                template_arg_formulae = [self.parse_constraint()]
                assert template_arg_formulae is not None
                while self.starts_with(", "):
                    self.consume(", ")
                    template_arg_formulae.append(self.parse_constraint())
            self.consume(">")

        return ch_type.clean_name, None if template_arg_formulae is None else tuple(template_arg_formulae)
