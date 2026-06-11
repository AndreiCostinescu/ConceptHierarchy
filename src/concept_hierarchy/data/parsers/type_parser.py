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
import json
from dataclasses import dataclass

from frozendict import frozendict

from concept_hierarchy.data.parsers.string_parser import StringParser
from concept_hierarchy.utils import is_integer

VARIADIC_GROUP_IDENTIFIER_CHARACTERS = "!$"


@dataclass(frozen=True)
class ParsedType:
    """
    Fully parsed representation of one entry in a type expression.

    ``full_name`` is:
    - ``str``   — ordinary named type or the variadic-group-as-string

    The variadic prefix (e.g. ``"!"``, ``"$"``, ``"!$"``) is prepended to the string forms
    but is absent from bracket groups.
    """

    full_name: str
    """
    The type name including:
     - template and function arguments, 
     - variadic identifier, and 
     - variadic expansion operator.
    OR the literal name:
     - for string literals: the json-encoded string value
     - for boolean literals: the ``"true"`` / ``"false"`` string value
     - for number (integer included) literals: the number represented as a string 
    """

    clean_name: str
    """
    Either
     - the base name of the type, without:
        - template and function arguments,
        - function arguments,
        - variadic identifiers, and
        - without the variadic expansion operator,
     - empty if this is a variadic group
     - OR the string literal that was classified as a literal value:
        - for string literals: the **decoded** version of the string (different than ``full_name``)
        - for boolean literals: same as ``full_name``
        - for number (integer included) literals: same as ``full_name``
    """

    variadic_group_identifier: str
    """
    Variadic prefix characters, e.g. ``""``, ``"!"``, ``"$"``, ``"!$"``.
    Can only be used in template arguments, not in function arguments, or variadic group elements.
    """

    has_variadic_template_expansion: bool
    """
    Whether the type ends with the `...` variadic template expansion operator.
    Can not be used when the type has template or function arguments or a variadic group identifier.
    """

    template_args: tuple[str | tuple[str, ...], ...] | None
    """
    Full names of template arguments. 
    ``None`` when no ``<…>`` was written; ``()`` when ``<>`` was written but empty.
    This must precede function arguments! And can not be used when `has_variadic_template_expansion` is True.
    """

    func_args: tuple[str | tuple[str, ...], ...] | None
    """
    Full names of function-call arguments.  
    ``None`` when no ``(…)`` was written; ``()`` when ``()`` was written but empty.
    This must come after template arguments! And can not be used when `has_variadic_template_expansion` is True.
    """

    variadic_group: tuple[ParsedType, ...] | None
    """Parsed arguments of a variadic group."""

    sub_types: tuple[ParsedType, ...]
    """Parsed template arguments (same order as ``template_args``)."""

    sub_func_types: tuple[ParsedType, ...]
    """Parsed function arguments (same order as ``func_args``)."""

    registry: frozendict
    """
    All types encountered while parsing this entry (including sub-types and sub-function-types), keyed by ``full_name``,
    mapped to ``(clean_name, template_args, function_args)``. Bracket groups are registered under their tuple key.
    """

    # ── Literal-value fields (optional; default None for non-literals) ───────
    literal_type: str | None = None
    """
    The kind of literal: ``"bool"``, ``"int"``, ``"float"``, or ``"string"``.
    ``None`` for named types and variadic groups.

    For string literals ``full_name`` carries the canonical JSON-encoded form including surrounding 
    double-quote characters (e.g. ``'"say \\"hi\\""'``), so string literals are unambiguously 
    distinguishable from a named type in the registry. 
    The ``clean_name`` member differs from ``full_name`` only for string literals, for which 
    it contains the **decoded** json-string value
    """

    @property
    def type_composition(self) -> tuple:
        """Construct the ``type_composition`` tree."""
        sub_comp: tuple = ()
        if self.is_variadic_group and self.sub_types != ():
            raise RuntimeError("This type is both a variadic group and has sub-types... Impossible! {self!r}")
        for sub in self.variadic_group or self.sub_types:
            sub_comp += sub.type_composition
        if self.is_variadic_group:
            return ((None, None, sub_comp),)  # identify variadic groups by the names being None
        return ((self.full_name, self.clean_name, sub_comp),)

    @property
    def is_templated(self) -> bool:
        if (self.template_args is None) and (self.sub_types != ()):
            raise RuntimeError(f"If this is not a templated type, then sub_types must be empty! Got {self!r}")
        return self.template_args is not None

    @property
    def has_arguments(self) -> bool:
        if (self.func_args is None) and (self.sub_func_types != ()):
            raise RuntimeError(f"If this is not a templated type, then sub_func_types must be empty! Got {self!r}")
        return self.func_args is not None

    @property
    def is_variadic_group(self) -> bool:
        if (self.variadic_group is not None) and (self.clean_name != ""):
            raise RuntimeError(f"If this is a variadic group, then clean_name must be empty. Got {self!r}")
        return self.variadic_group is not None

    @property
    def has_variadic_identifier(self) -> bool:
        return self.variadic_group_identifier != ""

    @property
    def is_literal(self) -> bool:
        """Whether this ``ParsedType`` represents a literal value (bool, int, float, or string)."""
        return self.literal_type is not None


class TypeParser(StringParser):
    """
    Recursive-descent parser for type expressions.

    with_exp = with expansion operator
    with_grp = with variadic group
    with_var = with variadic identifier in types

    Grammar::

        domain              ::= (named_type (',' named_type)* | ε) EOF
        named_type          ::=         name  type_extra
        named_type_with_exp ::=         name (type_extra | '...')
        named_type_with_var ::= var_id? name  type_extra
        type_extra          ::= ('<' (entry_list_with_grp | entry_list_with_var) '>')? ('(' entry_list_with_exp ')')?
        group               ::= '[' (literal | named_type_with_exp) (', ' (literal | named_type_with_exp))* ']' | '[]'
        entry_list_with_grp ::= ((group | literal | named_type) (',' (group | literal | named_type))*) | ε
        entry_list_with_exp ::= (named_type_with_exp (',' named_type_with_exp)*) | ε
        entry_list_with_var ::= ((literal | named_type_with_var) (',' (literal | named_type_with_var))*) | ε
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
        results = self._parse_entry_list(
            stop_chars=frozenset(),
            allow_variadic_identifiers=False,
            allow_template_expansion_operator=False,
            allow_variadic_groups=False,
            allow_literals=False,
        )
        self.check_finished()
        return results

    def _parse_entry_list(
        self,
        stop_chars: frozenset[str],
        *,
        allow_variadic_identifiers: bool,
        allow_template_expansion_operator: bool,
        allow_variadic_groups: bool,
        allow_literals: bool,
    ) -> tuple[ParsedType, ...]:
        """Comma-separated list, stopping (without consuming) at any character in *stop_chars* or EOF."""
        self.skip_whitespace()
        if self.eof() or self.peek() in stop_chars:
            return ()

        entries: list[ParsedType] = []
        expect_value = False
        while True:
            self.skip_whitespace()
            if self.eof() or self.peek() in stop_chars:
                if expect_value:
                    raise SyntaxError(f"Expected a value, but finished parsing at index {self.pos} of {self.text!r}!")
                break
            pos_before_entry_parse = self.pos
            entry = self._parse_entry(allow_variadic_groups=allow_variadic_groups, allow_literals=allow_literals)
            if not allow_variadic_identifiers and entry.has_variadic_identifier:
                raise SyntaxError(
                    f"Variadic group identifiers are only allowed in template arguments! Found at position "
                    f"{pos_before_entry_parse} of {self.text!r}"
                )
            if not allow_template_expansion_operator and entry.has_variadic_template_expansion:
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

    def _parse_entry(self, *, allow_variadic_groups: bool, allow_literals: bool) -> ParsedType:
        """Parse a variadic group, a literal value, or a named type entry."""
        self.skip_whitespace()

        # _VARIADIC_CHARS is a frozenset, so "" (EOF peek) is NOT a member —
        # unlike the string form where "" in "!$" is True.
        variadic = ""
        while not self.eof() and self.text[self.pos] in TypeParser._VARIADIC_CHARS:
            variadic += self.text[self.pos]
            self.pos += 1

        self.skip_whitespace()

        if self.peek() == "[":
            if not allow_variadic_groups:
                raise SyntaxError(
                    f"Variadic groups are only allowed in template arguments! Found at {self.pos} of {self.text!r}"
                )
            if variadic:
                raise SyntaxError(
                    f"Variadic group identifier {variadic!r} before a bracket group at position {self.pos}."
                )
            return self._parse_variadic_group()

        # Literals are only recognised when no variadic prefix was consumed, because variadicId prefixes only namedType,
        # never literal (grammar: templateArgWithVar ::= literal | (variadicId? namedType)).
        if allow_literals and not variadic and self._at_literal():
            return self._parse_literal_type()

        return self._parse_named_type(variadic)

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

    def _parse_literal_type(self) -> ParsedType:
        """
        Parse a literal value and wrap it in a ``ParsedType``.

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
            # Decode via the parent-class helper (handles \", \\, \uXXXX, etc.),
            # then re-encode canonically with json.dumps so that embedded quotes
            # are properly escaped in full_name.
            clean_name = self._parse_string_literal(surround_result_with_quotes=False)
            full_name = json.dumps(clean_name)  # e.g. '"say \\"hi\\""'
            literal_type = "string"
        elif self.starts_with("true") or self.starts_with("false"):
            clean_name = self._parse_bool_literal()
            full_name = clean_name
            literal_type = "bool"
        else:
            clean_name = self._parse_number_literal()
            full_name = clean_name
            literal_type = "int" if is_integer(clean_name) else "float"

        registry: dict = {full_name: (clean_name, None, None)}
        return ParsedType(
            full_name=full_name,
            clean_name=clean_name,
            variadic_group_identifier="",
            has_variadic_template_expansion=False,
            template_args=None,
            func_args=None,
            variadic_group=None,
            sub_types=(),
            sub_func_types=(),
            registry=frozendict(registry),
            literal_type=literal_type,
        )

    def _parse_variadic_group(self) -> ParsedType:
        """
        group ::= '[' entry_list_with_exp ']'

        The bracket entries will be saved under the type's variadic group entry.
        ``full_name`` will be the string representation of the variadic group and ``clean_name`` will be empty.
        """
        self.consume("[")
        entries = self._parse_entry_list(
            stop_chars=frozenset({"]"}),
            allow_variadic_identifiers=False,
            allow_template_expansion_operator=True,
            allow_variadic_groups=False,
            allow_literals=True,  # groupElement ::= literalValue | expandableName
        )
        try:
            self.consume("]")
        except RuntimeError as e:
            raise SyntaxError(str(e)) from e

        merged: dict = {}

        if len(entries) == 0:
            group_id: tuple[str, ...] = ()
            variadic_group: tuple[ParsedType, ...] = ()
        else:
            group_id = tuple(e.full_name for e in entries)
            variadic_group = tuple(entries)
            for e in entries:
                merged.update(e.registry)

        clean_name = ""
        # A bracket-group argument must be rendered as [A, B] in the parent name.
        full_name = f"[{', '.join(group_id)}]"

        return ParsedType(
            full_name=full_name,
            clean_name=clean_name,
            variadic_group_identifier="",
            has_variadic_template_expansion=False,
            template_args=None,
            func_args=None,
            variadic_group=variadic_group,
            sub_types=(),
            sub_func_types=(),
            registry=frozendict(merged),
            literal_type=None,
        )

    def _parse_named_type(self, variadic: str) -> ParsedType:
        """
        named_type ::= (var_id)? name (type_extra | '...')
        type_extra ::= ('<' (entry_list_with_grp | entry_list_with_var) '>')? ('(' entry_list_with_exp ')')?

        *variadic* (already consumed by ``_parse_entry``) is prepended to the assembled ``full_name``.
        """
        clean_name = self._parse_type_name()
        assert clean_name != ""
        has_variadic_template_expansion = clean_name.endswith("...") and not clean_name.endswith("....")
        self.skip_whitespace()

        full_name_no_prefix = clean_name
        if has_variadic_template_expansion:
            if variadic:
                raise SyntaxError(
                    f"The variadic template expansion cannot be used with variadic identifiers (here: {variadic})!"
                )
            clean_name = clean_name[:-3]

        # ── Template arguments ──────────────────────────────────────────
        sub_types: list[ParsedType] = []
        template_args: tuple | None = None
        if self.try_consume("<"):
            entries = self._parse_entry_list(
                stop_chars=frozenset({">"}),
                allow_variadic_identifiers=True,
                allow_template_expansion_operator=False,
                allow_variadic_groups=True,
                allow_literals=True,  # templateArgWithGrp/Var ::= variadicGroup | literal | namedType
            )
            try:
                self.consume(">")
            except RuntimeError as e:
                raise SyntaxError(str(e)) from e
            template_args = ()

            if len(entries) == 0:
                full_name_no_prefix += "<>"
            else:
                has_variadic_group_as_entry = False
                has_argument_with_variadic_identifier = False
                parts: list[str] = []
                for e in entries:
                    if e.is_variadic_group:
                        has_variadic_group_as_entry = True
                        template_args += (tuple(x.full_name for x in e.variadic_group),)
                    else:
                        has_argument_with_variadic_identifier |= e.has_variadic_identifier
                        template_args += (e.full_name,)
                    parts.append(e.full_name)
                    sub_types.append(e)
                if has_variadic_group_as_entry and has_argument_with_variadic_identifier:
                    raise SyntaxError(
                        f"Can not define template argument values combining variadic groups and types with variadic "
                        f"identifiers! Found at {self.text!r}"
                    )
                full_name_no_prefix += f"<{', '.join(parts)}>"

        # ── Function-call arguments ─────────────────────────────────────
        sub_func_types: list[ParsedType] = []
        func_args: tuple | None = None
        if self.try_consume("("):
            entries = self._parse_entry_list(
                stop_chars=frozenset({")"}),
                allow_variadic_identifiers=False,
                allow_template_expansion_operator=True,
                allow_variadic_groups=False,
                allow_literals=False,  # funcArg ::= expandableName only; literals are forbidden
            )
            try:
                self.consume(")")
            except RuntimeError as e:
                raise SyntaxError(str(e)) from e
            func_args = ()

            if len(entries) == 0:
                full_name_no_prefix += "()"
            else:
                parts: list[str] = []
                for e in entries:
                    if e.is_variadic_group:
                        raise SyntaxError(
                            f"Variadic groups are not allowed in function arguments! Found in {self.text!r}!"
                        )
                    parts.append(e.full_name)
                    func_args += (e.full_name,)
                    sub_func_types.append(e)
                full_name_no_prefix += f"({', '.join(parts)})"

        # ── Variadic template expansion operator ────────────────────────
        if self.peek(3) == "..." or (
            has_variadic_template_expansion and (template_args is not None or func_args is not None)
        ):
            raise SyntaxError(
                f"The variadic template expansion cannot be used with function arguments, template arguments or with "
                f"variadic identifiers! Found: variadic={variadic!r}, template_args={template_args!r}, and "
                f"func_args={func_args!r}!"
            )

        # ── Assemble with variadic prefix ───────────────────────────────
        full_name: str = variadic + full_name_no_prefix
        clean_name: str = clean_name

        registry: dict = {}
        for sub in sub_types:
            registry.update(sub.registry)
        for sub in sub_func_types:
            registry.update(sub.registry)
        registry[full_name] = (clean_name, template_args, func_args)

        return ParsedType(
            full_name=full_name,
            clean_name=clean_name,
            variadic_group_identifier=variadic,
            has_variadic_template_expansion=has_variadic_template_expansion,
            template_args=template_args,
            func_args=func_args,
            variadic_group=None,
            sub_types=tuple(sub_types),
            sub_func_types=tuple(sub_func_types),
            registry=frozendict(registry),
            literal_type=None,
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
