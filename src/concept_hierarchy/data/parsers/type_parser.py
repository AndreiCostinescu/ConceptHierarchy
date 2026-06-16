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
type_parser.py — Contains the parser for parsing a string into a Concept Hierarchy (valid or invalid) type.
"""

from __future__ import annotations

import functools

from concept_hierarchy.data.parsers.string_parser import StringParser
from concept_hierarchy.data.types.parsed_type import (
    VARIADIC_GROUP_IDENTIFIER_CHARACTERS,
    ParsedType,
    TemplateArgumentLiteral,
    TemplateArgumentValue,
    TemplateArgumentVariadicGroup,
    TemplateArgumentWithVariadicId,
)
from concept_hierarchy.utils import is_integer


class TypeParser(StringParser):
    """
    Recursive-descent parser for type expressions.

    with_exp = with expansion operator
    with_grp = with variadic group
    with_var = with variadic identifier in types or literals

    Grammar::

        domain              ::= (named_type (',' named_type)* | ε) EOF
        named_type          ::=         name  type_extra
        named_type_with_exp ::=         name (type_extra | '...')
        type_extra          ::= ('<' (entry_list_with_grp | entry_list_with_var) '>')? ('(' entry_list_with_exp ')')?
        group               ::= '[' (literal | named_type_with_exp) (', ' (literal | named_type_with_exp))* ']' | '[]'
        entry_list_with_grp ::= ((group | literal | named_type) (',' (group | literal | named_type))*) | ε
        entry_list_with_exp ::= (named_type_with_exp (',' named_type_with_exp)*) | ε
        entry_list_with_var ::= (var_id? (literal | named_type) (',' var_id? (literal | named_type))*) | ε
        var_id              ::= [!$]+
        name                ::= [non-delimiter, non-whitespace, non-variadic, non-quote characters]+
        literal             ::= 'true' | 'false' | number | '"' character* '"'

    Usage::

        types = TypeParser("!Map<[A, B], $List<int>>").parse()
    """

    _VARIADIC_CHARS: frozenset[str] = frozenset(VARIADIC_GROUP_IDENTIFIER_CHARACTERS)  # "" not a member

    def __init__(self, domain: str, allow_trailing_comma: bool = False) -> None:
        super().__init__(domain)
        self.allow_trailing_comma = allow_trailing_comma

    def parse(self) -> tuple[ParsedType, ...]:
        """
        Parse a complete domain expression and verify the entire input is consumed.
        Returns one ``ParsedType`` per comma-separated top-level entry.

        domain ::= (named_type (',' named_type)* | ε) EOF
        """
        results: tuple[ParsedType, ...] = self._parse_entry_list_type_only(
            stop_chars=frozenset(), allow_variadic_identifiers=False, allow_template_expansion_operator=False
        )
        self.check_finished()
        return results

    def _parse_entry_list_type_only(
        self, stop_chars: frozenset[str], *, allow_variadic_identifiers: bool, allow_template_expansion_operator: bool
    ) -> tuple[ParsedType, ...]:
        results: tuple[TemplateArgumentValue, ...] = self._parse_entry_list(
            stop_chars=stop_chars,
            allow_variadic_identifiers=allow_variadic_identifiers,
            allow_template_expansion_operator=allow_template_expansion_operator,
            allow_variadic_groups=False,
            allow_literals=False,
        )
        type_results = []
        for t in results:
            if not isinstance(t, ParsedType):
                raise RuntimeError(f"Expected a parsed type, but received a {type(t)!r}: {t!r}")
            type_results.append(t)
        return tuple(type_results)

    def _parse_entry_list_no_variadic_groups(
        self, stop_chars: frozenset[str], *, allow_variadic_identifiers: bool, allow_template_expansion_operator: bool
    ) -> tuple[ParsedType | TemplateArgumentLiteral, ...]:
        results: tuple[TemplateArgumentValue, ...] = self._parse_entry_list(
            stop_chars=stop_chars,
            allow_variadic_identifiers=allow_variadic_identifiers,
            allow_template_expansion_operator=allow_template_expansion_operator,
            allow_variadic_groups=False,
            allow_literals=True,
        )
        non_variadic_group_results = []
        for t in results:
            if isinstance(t, TemplateArgumentVariadicGroup):
                raise RuntimeError(f"Expected a parsed type or a literal value, but received a {type(t)!r}: {t!r}")
            non_variadic_group_results.append(t)
        return tuple(non_variadic_group_results)

    def _parse_entry_list(
        self,
        stop_chars: frozenset[str],
        *,
        allow_variadic_identifiers: bool,
        allow_template_expansion_operator: bool,
        allow_variadic_groups: bool = True,
        allow_literals: bool = True,
    ) -> tuple[TemplateArgumentValue, ...]:
        """Comma-separated list, stopping (without consuming) at any character in *stop_chars* or EOF."""
        self.skip_whitespace()
        if self.eof() or self.peek() in stop_chars:
            return ()

        entries: list[TemplateArgumentValue] = []
        expect_value = False
        while True:
            self.skip_whitespace()
            if self.eof() or self.peek() in stop_chars:
                if expect_value:
                    raise SyntaxError(f"Expected a value, but finished parsing at index {self.pos} of {self.text!r}!")
                break
            pos_before_entry_parse = self.pos
            entry = self._parse_entry(allow_variadic_groups=allow_variadic_groups, allow_literals=allow_literals)
            if not allow_variadic_identifiers and (
                isinstance(entry, TemplateArgumentWithVariadicId) and entry.has_variadic_identifier
            ):
                raise SyntaxError(
                    f"Variadic group identifiers are only allowed in template arguments! Found at position "
                    f"{pos_before_entry_parse} of {self.text!r}"
                )
            if not allow_template_expansion_operator and (
                isinstance(entry, ParsedType) and entry.has_variadic_template_expansion
            ):
                raise SyntaxError(
                    f"Template expansion operator is only allowed in a variadic group or function arguments! Found at "
                    f"{pos_before_entry_parse} of {self.text!r}"
                )
            entries.append(entry)
            self.skip_whitespace()
            if not self.try_consume(","):
                break
            elif not self.allow_trailing_comma:
                expect_value = True
        return tuple(entries)

    def _parse_entry(self, *, allow_variadic_groups: bool, allow_literals: bool) -> TemplateArgumentValue:
        """Parse a variadic group, a literal value, or a named type entry."""
        self.skip_whitespace()

        # _VARIADIC_CHARS is a frozenset, so "" (EOF peek) is NOT a member —
        # unlike the string form where "" in "!$" is True.
        variadic_id = ""
        while not self.eof() and self.text[self.pos] in TypeParser._VARIADIC_CHARS:
            variadic_id += self.text[self.pos]
            self.pos += 1
        if variadic_id == "":
            variadic_id = None

        self.skip_whitespace()

        if self.peek() == "[":
            if not allow_variadic_groups:
                raise SyntaxError(
                    f"Variadic groups are only allowed in template arguments! Found at {self.pos} of {self.text!r}"
                )
            if variadic_id:
                raise SyntaxError(
                    f"Variadic group identifier {variadic_id!r} before a bracket group at position {self.pos}."
                )
            return self._parse_variadic_group()

        if allow_literals and self._at_literal():
            return self._parse_literal_type(variadic_id)

        return self._parse_named_type(variadic_id)

    def _at_literal(self) -> bool:
        """
        Return ``True`` iff the current position begins a literal value, without advancing the parser.

        Recognises:
         - ``"``           → string literal
         - ``-`` or digit  → numeric literal  (names must start with a letter per the grammar)
         - ``true`` / ``false`` followed by a non-identifier character (or EOF) → boolean literal
        """
        c = self.peek()
        if c == '"':
            return True
        if c == "-" or c.isdigit():
            return True
        for keyword in ("true", "false"):
            if self.starts_with(keyword):
                end = self.pos + len(keyword)
                after = self.text[end] if end < len(self.text) else ""
                if not (after.isalnum() or after == "_"):
                    return True
        return False

    def _parse_literal_type(self, variadic_id: str | None) -> TemplateArgumentLiteral:
        """
        Parse a literal value and wrap it in a ``TemplateArgumentLiteral``.

        literalValue ::= boolean | number | '\\"' character* '\\"'

        ``full_name`` / ``literal_value``:
         - Bool / number: the source text as-is (``"true"``, ``"42"``, ``"3.14"``).
         - String: the canonical JSON-re-encoded form with surrounding double quotes,
           e.g. input ``"say \\"hi\\""`` → ``full_name = '"say \\"hi\\""'``.
           Using the canonical form ensures embedded quotes survive round-trips and that
           string literals are unambiguously distinguishable from named types in the registry.

        ``clean_name``:
         - Bool / number: identical to ``full_name``.
         - String: the **decoded** content without surrounding quotes,
           e.g. ``'say "hi"'`` — all JSON escape sequences are resolved via
           ``StringParser._parse_string_literal``.
        """
        if self.peek() == '"':
            # Decode via the parent-class helper (handles \", \\, \uXXXX, etc.)
            clean_name = self._parse_string_literal(surround_result_with_quotes=False)
            literal_type = "string"
        elif self.starts_with("true") or self.starts_with("false"):
            clean_name = self._parse_bool_literal()
            literal_type = "bool"
        else:
            clean_name = self._parse_number_literal()
            literal_type = "int" if is_integer(clean_name) else "float"

        return TemplateArgumentLiteral(
            variadic_group_identifier=variadic_id,
            literal_value=clean_name,
            literal_type=literal_type,
        )

    def _parse_variadic_group(self) -> TemplateArgumentVariadicGroup:
        """
        group ::= '[' entry_list_with_exp ']'

        The bracket entries will be saved under the type's variadic group entry.
        ``full_name`` will be the string representation of the variadic group and ``clean_name`` will be empty.
        """
        self.consume("[")
        entries = self._parse_entry_list_no_variadic_groups(
            stop_chars=frozenset({"]"}), allow_variadic_identifiers=False, allow_template_expansion_operator=True
        )  # groupElement ::= literalValue | expandableName
        try:
            self.consume("]")
        except RuntimeError as e:
            raise SyntaxError(str(e)) from e
        if len(entries) == 0:
            variadic_group: tuple[ParsedType | TemplateArgumentLiteral, ...] = ()
        else:
            variadic_group = tuple(entries)
        return TemplateArgumentVariadicGroup(variadic_group=variadic_group)

    def _parse_named_type(self, variadic_id: str | None) -> ParsedType:
        """
        named_type ::= (var_id)? name (type_extra | '...')
        type_extra ::= ('<' (entry_list_with_grp | entry_list_with_var) '>')? ('(' entry_list_with_exp ')')?

        *variadic* (already consumed by ``_parse_entry``) is prepended to the assembled ``full_name``.
        """
        clean_name = self._parse_type_name()
        assert clean_name != ""
        has_variadic_template_expansion = clean_name.endswith("...") and not clean_name.endswith("....")
        self.skip_whitespace()

        if has_variadic_template_expansion:
            if variadic_id:
                raise SyntaxError(
                    f"The variadic template expansion cannot be used with variadic identifiers (here: {variadic_id})!"
                )
            clean_name = clean_name[:-3]

        # ── Template arguments ──────────────────────────────────────────
        template_arguments: list[TemplateArgumentValue] | None = None
        if self.try_consume("<"):
            entries = self._parse_entry_list(
                stop_chars=frozenset({">"}), allow_variadic_identifiers=True, allow_template_expansion_operator=False
            )  # templateArgWithGrp/Var ::= variadicGroup | literal | namedType
            try:
                self.consume(">")
            except RuntimeError as e:
                raise SyntaxError(str(e)) from e
            template_arguments = []
            has_variadic_group_as_entry = False
            has_argument_with_variadic_identifier = False
            for e in entries:
                if isinstance(e, TemplateArgumentVariadicGroup):
                    has_variadic_group_as_entry = True
                elif isinstance(e, TemplateArgumentWithVariadicId):
                    has_argument_with_variadic_identifier |= e.has_variadic_identifier
                template_arguments.append(e)
            if has_variadic_group_as_entry and has_argument_with_variadic_identifier:
                raise SyntaxError(
                    f"Can not define template argument values combining variadic groups and types with variadic "
                    f"identifiers! Found at {self.text!r}"
                )

        # ── Function-call arguments ─────────────────────────────────────
        function_arguments: list[ParsedType] | None = None
        if self.try_consume("("):
            entries = self._parse_entry_list_type_only(
                stop_chars=frozenset({")"}), allow_variadic_identifiers=False, allow_template_expansion_operator=True
            )  # funcArg ::= expandableName only; literals are forbidden
            try:
                self.consume(")")
            except RuntimeError as e:
                raise SyntaxError(str(e)) from e
            function_arguments = []
            for e in entries:
                function_arguments.append(e)

        # ── Variadic template expansion operator ────────────────────────
        if self.peek(3) == "..." or (
            has_variadic_template_expansion and (template_arguments is not None or function_arguments is not None)
        ):
            raise SyntaxError(
                f"The variadic template expansion cannot be used with function arguments, template arguments or with "
                f"variadic identifiers! Found: variadic={variadic_id!r}, template_args={template_arguments!r}, "
                f"and func_args={function_arguments!r}!"
            )

        return ParsedType(
            variadic_group_identifier=variadic_id,
            name=clean_name,
            has_variadic_template_expansion=has_variadic_template_expansion,
            template_arguments=tuple(template_arguments) if template_arguments is not None else template_arguments,
            function_arguments=tuple(function_arguments) if function_arguments is not None else function_arguments,
        )

    def _parse_type_name(self) -> str:
        """
        Consume a bare type name: a non-empty run of characters that are neither delimiters (``< > ( ) [ ] ,``),
        whitespace, variadic prefix characters (``! $``), nor a string-literal opener (``"``).
        """
        _STOP = frozenset(',<>()[]" \t\n\r') | TypeParser._VARIADIC_CHARS
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] not in _STOP:
            self.pos += 1
        name = self.text[start : self.pos]
        if not name:
            raise SyntaxError(f"Expected a type name at position {self.pos}; got {self.remaining()[:20]!r}")
        return name


# Memoized public entry-point
@functools.lru_cache(maxsize=None)
def _parse_type_cached(domain: str) -> tuple[ParsedType, ...]:
    return TypeParser(domain).parse()


def parse_type(
    domain: str,
    expected_number_of_values: int | None = None,
) -> tuple[ParsedType, ...]:
    """
    Parse a type string; results are memoized on *domain*.

    Returns one ``ParsedType`` per comma-separated top-level entry.
    Raises ``RuntimeError`` if the result count doesn't match *expected_number_of_values* when supplied.
    """
    results = _parse_type_cached(domain)
    if expected_number_of_values is not None and len(results) != expected_number_of_values:
        raise RuntimeError(
            f"Parsing {domain!r} failed: expected {expected_number_of_values} "
            f"values, got {len(results)} (result = {results})"
        )
    return results
