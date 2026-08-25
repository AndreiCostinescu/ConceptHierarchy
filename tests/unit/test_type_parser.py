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

# Copyright 2026 ConceptHierarchy Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Comprehensive pytest suite for TypeParser / ParsedType.
"""

from __future__ import annotations

import json

import pytest
from frozendict import frozendict

from concept_hierarchy.data.parsers.type_parser import ParsedType, TypeParser, parse_type
from concept_hierarchy.data.types.parsed_type import TemplateArgumentLiteral, TemplateArgumentVariadicGroup
from concept_hierarchy.errors import CHSyntaxError

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse(text: str) -> tuple[ParsedType, ...]:
    return TypeParser(text, None).parse_types()


def _p1(text: str) -> ParsedType:
    """Parse *text* and assert exactly one top-level entry is produced."""
    (t,) = _parse(text)
    return t


# ─────────────────────────────────────────────────────────────────────────────
# 1. Empty / whitespace inputs
# ─────────────────────────────────────────────────────────────────────────────


class TestEmpty:
    def test_empty_string(self):
        assert _parse("") == ()

    def test_spaces_only(self):
        assert _parse("   ") == ()

    def test_tabs_and_newlines(self):
        assert _parse("  \t\n  ") == ()

    def test_single_newline(self):
        assert _parse("\n") == ()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Simple named types
# ─────────────────────────────────────────────────────────────────────────────


class TestSimpleNamedTypes:
    def test_single_type_fields(self):
        t = _p1("MyType")
        assert t.full_name == "MyType"
        assert t.clean_name == "MyType"
        assert t.has_variadic_identifier is False
        assert t.has_variadic_template_expansion is False
        assert t.template_args is None
        assert t.func_args is None
        assert t.template_arguments is None
        assert t.function_arguments is None

    def test_single_lowercase(self):
        t = _p1("mytype")
        assert t.full_name == "mytype"
        assert t.clean_name == "mytype"

    def test_single_char(self):
        t = _p1("T")
        assert t.full_name == "T"

    def test_name_with_digits(self):
        t = _p1("uint8")
        assert t.full_name == "uint8"
        assert t.clean_name == "uint8"

    def test_name_starting_with_digit(self):
        # digits are not stop-chars in _parse_type_name; parsed as a plain name
        t = _p1("42type")
        assert t.full_name == "42type"

    def test_name_with_underscore(self):
        t = _p1("my_type")
        assert t.full_name == "my_type"

    def test_name_with_dots(self):
        # dots are not stop-chars; accepted as part of the name
        t = _p1("std.vector")
        assert t.full_name == "std.vector"

    def test_multiple_top_level_types(self):
        result = _parse("A, B, C")
        assert len(result) == 3
        assert result[0].full_name == "A"
        assert result[1].full_name == "B"
        assert result[2].full_name == "C"

    def test_two_types_no_space(self):
        result = _parse("A,B")
        assert len(result) == 2
        assert result[0].full_name == "A"
        assert result[1].full_name == "B"

    def test_leading_and_trailing_whitespace(self):
        t = _p1("  MyType  ")
        assert t.full_name == "MyType"

    def test_is_templated_false(self):
        assert _p1("T").is_templated is False

    def test_has_arguments_false(self):
        assert _p1("T").has_arguments is False

    def test_has_variadic_identifier_false(self):
        assert _p1("T").has_variadic_identifier is False

    # "true" / "false" at the TOP level are parsed as plain named types,
    # because allow_literals=False at domain level.
    def test_true_at_top_level_is_named_type(self):
        t = _p1("true")
        assert t.full_name == "true"

    def test_false_at_top_level_is_named_type(self):
        t = _p1("false")
        assert t.full_name == "false"

    def test_number_at_top_level_is_named_type(self):
        t = _p1("42")
        assert t.full_name == "42"

    def test_negative_number_at_top_level_is_named_type(self):
        t = _p1("-1")
        assert t.full_name == "-1"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Template arguments
# ─────────────────────────────────────────────────────────────────────────────


class TestTemplateArgs:
    def test_empty_template_args(self):
        t = _p1("T<>")
        assert t.full_name == "T<>"
        assert t.clean_name == "T"
        assert t.template_args == ()
        assert t.template_arguments == ()

    def test_empty_template_args_is_templated(self):
        assert _p1("T<>").is_templated is True

    def test_single_arg(self):
        t = _p1("T<A>")
        assert t.full_name == "T<A>"
        assert t.clean_name == "T"
        assert t.template_args == ("A",)
        assert len(t.template_arguments) == 1
        assert t.template_arguments[0].full_name == "A"

    def test_two_args(self):
        t = _p1("Map<K, V>")
        assert t.full_name == "Map<K, V>"
        assert t.clean_name == "Map"
        assert t.template_args == ("K", "V")
        assert len(t.template_arguments) == 2

    def test_three_args(self):
        t = _p1("Triple<A, B, C>")
        assert t.template_args == ("A", "B", "C")

    def test_nested_template_args(self):
        t = _p1("Map<List<int>, Set<String>>")
        assert t.full_name == "Map<List<int>, Set<String>>"
        assert t.clean_name == "Map"
        assert t.template_args == ("List<int>", "Set<String>")
        inner = t.template_arguments[0]
        assert inner.full_name == "List<int>"
        assert inner.clean_name == "List"
        assert isinstance(inner, ParsedType)
        assert inner.template_args == ("int",)

    def test_deep_nesting(self):
        t = _p1("A<B<C<D>>>")
        assert t.full_name == "A<B<C<D>>>"
        assert t.template_args == ("B<C<D>>",)
        inner = t.template_arguments[0]
        assert isinstance(inner, ParsedType)
        assert inner.template_args == ("C<D>",)
        inner = inner.template_arguments[0]
        assert isinstance(inner, ParsedType)
        assert inner.template_args == ("D",)

    def test_template_args_none_without_brackets(self):
        assert _p1("T").template_args is None

    def test_whitespace_inside_template_args(self):
        t = _p1("T< A , B >")
        assert t.template_args == ("A", "B")

    def test_template_and_func_args_combined(self):
        t = _p1("T<A, B>(f, g)")
        assert t.full_name == "T<A, B>(f, g)"
        assert t.clean_name == "T"
        assert t.template_args == ("A", "B")
        assert t.func_args == ("f", "g")

    def test_empty_template_and_empty_func(self):
        t = _p1("T<>()")
        assert t.full_name == "T<>()"
        assert t.template_args == ()
        assert t.func_args == ()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Function arguments
# ─────────────────────────────────────────────────────────────────────────────


class TestFunctionArgs:
    def test_empty_func_args(self):
        t = _p1("T()")
        assert t.full_name == "T()"
        assert t.clean_name == "T"
        assert t.func_args == ()
        assert t.function_arguments == ()

    def test_single_func_arg(self):
        t = _p1("T(a)")
        assert t.full_name == "T(a)"
        assert t.func_args == ("a",)
        assert len(t.function_arguments) == 1
        assert t.function_arguments[0].full_name == "a"

    def test_multiple_func_args(self):
        t = _p1("T(a, b, c)")
        assert t.func_args == ("a", "b", "c")
        assert len(t.function_arguments) == 3

    def test_func_args_none_without_parens(self):
        assert _p1("T").func_args is None

    def test_has_arguments_true(self):
        assert _p1("T(a)").has_arguments is True

    def test_func_arg_with_expansion_operator(self):
        t = _p1("T(a...)")
        arg = t.function_arguments[0]
        assert arg.full_name == "a..."
        assert arg.clean_name == "a"
        assert arg.has_variadic_template_expansion is True
        assert t.func_args == ("a...",)

    def test_expansion_of_bare_dots_in_func_args(self):
        # "..." as a standalone func arg: clean_name="" after stripping "..."
        t = _p1("T(...)")
        assert t.func_args == ("...",)
        arg = t.function_arguments[0]
        assert arg.full_name == "..."
        assert arg.clean_name == ""
        assert arg.has_variadic_template_expansion is True

    def test_func_arg_with_own_template_args(self):
        t = _p1("T(a<int>)")
        assert t.func_args == ("a<int>",)
        arg = t.function_arguments[0]
        assert arg.clean_name == "a"
        assert arg.template_args == ("int",)

    def test_func_arg_number_is_parsed_as_name(self):
        # allow_literals=False for func args; digits are not stop-chars
        t = _p1("T(42)")
        arg = t.function_arguments[0]
        assert arg.full_name == "42"

    def test_func_arg_bool_is_parsed_as_name(self):
        t = _p1("T(true)")
        arg = t.function_arguments[0]
        assert arg.full_name == "true"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Variadic groups  (bracket notation inside template args)
# ─────────────────────────────────────────────────────────────────────────────


class TestVariadicGroups:
    def test_empty_group(self):
        t = _p1("T<[]>")
        assert t.template_args == ((),)
        grp = t.template_arguments[0]
        assert isinstance(grp, TemplateArgumentVariadicGroup)
        assert grp.variadic_group == ()
        assert grp.full_name == "[]"
        assert grp.clean_name == ""

    def test_single_element_group(self):
        t = _p1("T<[A]>")
        assert t.template_args == (("A",),)
        grp = t.template_arguments[0]
        assert isinstance(grp, TemplateArgumentVariadicGroup)
        assert len(grp.variadic_group) == 1
        assert grp.variadic_group[0].full_name == "A"

    def test_two_element_group(self):
        t = _p1("T<[A, B]>")
        assert t.template_args == (("A", "B"),)
        grp = t.template_arguments[0]
        assert grp.full_name == "[A, B]"
        assert isinstance(grp, TemplateArgumentVariadicGroup)
        assert len(grp.variadic_group) == 2

    def test_two_groups_in_template_args(self):
        t = _p1("T<[A, B], [C, D]>")
        assert t.template_args == (("A", "B"), ("C", "D"))

    def test_group_mixed_with_plain_named_type(self):
        # "group mode": a variadic group and a plain named type may coexist
        t = _p1("T<[A, B], C>")
        assert t.template_args == (("A", "B"), "C")

    def test_group_element_with_expansion(self):
        t = _p1("T<[A...]>")
        grp = t.template_arguments[0]
        assert isinstance(grp, TemplateArgumentVariadicGroup)
        elem = grp.variadic_group[0]
        assert elem.has_variadic_template_expansion is True
        assert elem.clean_name == "A"
        assert elem.full_name == "A..."

    def test_group_element_with_own_template_args(self):
        t = _p1("T<[List<int>]>")
        grp = t.template_arguments[0]
        assert isinstance(grp, TemplateArgumentVariadicGroup)
        elem = grp.variadic_group[0]
        assert elem.full_name == "List<int>"
        assert elem.template_args == ("int",)

    def test_group_is_variadic_group_true(self):
        t = _p1("T<[A]>")
        grp = t.template_arguments[0]
        assert isinstance(grp, TemplateArgumentVariadicGroup)

    def test_group_clean_name_is_empty(self):
        t = _p1("T<[A, B]>")
        grp = t.template_arguments[0]
        assert grp.clean_name == ""


# ─────────────────────────────────────────────────────────────────────────────
# 6. Variadic identifiers  (! / $ prefixes on template args)
# ─────────────────────────────────────────────────────────────────────────────


class TestVariadicIdentifiers:
    def test_dollar_prefix(self):
        t = _p1("Map<$T>")
        arg = t.template_arguments[0]
        assert isinstance(arg, ParsedType)
        assert arg.variadic_group_identifier == "$"
        assert arg.clean_name == "T"
        assert arg.full_name == "$T"

    def test_bang_prefix(self):
        t = _p1("Map<!T>")
        arg = t.template_arguments[0]
        assert isinstance(arg, ParsedType)
        assert arg.variadic_group_identifier == "!"
        assert arg.full_name == "!T"

    def test_bang_dollar_combined(self):
        t = _p1("Map<!$T>")
        arg = t.template_arguments[0]
        assert isinstance(arg, ParsedType)
        assert arg.variadic_group_identifier == "!$"
        assert arg.full_name == "!$T"

    def test_two_var_id_args(self):
        t = _p1("Map<$A, !B>")
        t_arg1 = t.template_arguments[0]
        assert isinstance(t_arg1, ParsedType)
        t_arg2 = t.template_arguments[1]
        assert isinstance(t_arg2, ParsedType)
        assert t_arg1.variadic_group_identifier == "$"
        assert t_arg2.variadic_group_identifier == "!"

    def test_plain_and_var_id_mix_is_valid(self):
        # "var_id mode" allows a plain named type alongside var_id types
        t = _p1("Map<A, $B>")
        t_arg1 = t.template_arguments[0]
        assert isinstance(t_arg1, ParsedType)
        t_arg2 = t.template_arguments[1]
        assert isinstance(t_arg2, ParsedType)
        assert t_arg1.has_variadic_identifier is False
        assert t_arg2.variadic_group_identifier == "$"
        assert t.template_args == ("A", "$B")

    def test_has_variadic_identifier_true(self):
        t = _p1("Map<$T>")
        arg = t.template_arguments[0]
        assert isinstance(arg, ParsedType)
        assert arg.has_variadic_identifier is True

    def test_has_variadic_identifier_false_for_plain(self):
        t = _p1("Map<T>")
        arg = t.template_arguments[0]
        assert isinstance(arg, ParsedType)
        assert arg.has_variadic_identifier is False

    def test_var_id_full_name_includes_prefix(self):
        t = _p1("F<$A, !B>")
        assert t.template_args == ("$A", "!B")

    def test_var_id_on_literal_values(self):
        t = _p1('Map<!1, $2, !$!3.14, !"stringLiteral">')
        assert len(t.template_arguments) == 4
        arg = t.template_arguments[0]
        assert isinstance(arg, TemplateArgumentLiteral)
        assert arg.has_variadic_identifier is True
        assert arg.variadic_group_identifier == "!"
        assert arg.literal_type == "int"
        assert arg.full_name == "!1"
        arg = t.template_arguments[1]
        assert isinstance(arg, TemplateArgumentLiteral)
        assert arg.has_variadic_identifier is True
        assert arg.variadic_group_identifier == "$"
        assert arg.literal_type == "int"
        assert arg.full_name == "$2"
        arg = t.template_arguments[2]
        assert isinstance(arg, TemplateArgumentLiteral)
        assert arg.has_variadic_identifier is True
        assert arg.variadic_group_identifier == "!$!"
        assert arg.literal_type == "float"
        assert arg.full_name == "!$!3.14"
        arg = t.template_arguments[3]
        assert isinstance(arg, TemplateArgumentLiteral)
        assert arg.has_variadic_identifier is True
        assert arg.variadic_group_identifier == "!"
        assert arg.literal_type == "string"
        assert arg.full_name == '!"stringLiteral"'

    def test_var_id_with_spaces_raises(self):
        with pytest.raises(CHSyntaxError, match=r"Expected a type name at position 5; got ' 1, \$ 2>'"):
            _p1("Map<! 1, $ 2>")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Variadic expansion operator  (...)
# ─────────────────────────────────────────────────────────────────────────────


class TestVariadicExpansion:
    def test_expansion_in_func_arg(self):
        arg = _p1("T(a...)").function_arguments[0]
        assert arg.has_variadic_template_expansion is True
        assert arg.clean_name == "a"
        assert arg.full_name == "a..."

    def test_expansion_in_variadic_group_element(self):
        t = _p1("T<[a...]>")
        arg = t.template_arguments[0]
        assert isinstance(arg, TemplateArgumentVariadicGroup)
        elem = arg.variadic_group[0]
        assert elem.has_variadic_template_expansion is True
        assert elem.clean_name == "a"

    def test_four_dots_is_not_expansion(self):
        # "A...." ends with "..." but also ends with "...." → not expansion
        arg = _p1("T(A....)").function_arguments[0]
        assert arg.has_variadic_template_expansion is False
        assert arg.full_name == "A...."
        assert arg.clean_name == "A...."

    def test_five_dots_is_not_expansion(self):
        arg = _p1("T(A.....)").function_arguments[0]
        assert arg.has_variadic_template_expansion is False
        assert arg.clean_name == "A....."

    def test_expansion_not_allowed_at_top_level(self):
        with pytest.raises(
            CHSyntaxError,
            match=r"The template expansion operator is only allowed in a type application without variadic groups, in"
            r" a variadic group, or in function arguments!",
        ):
            _parse("A...")

    def test_expansion_allowed_in_template_args(self):
        arg = _p1("T<A...>").template_arguments[0]
        assert isinstance(arg, ParsedType)
        assert arg.has_variadic_identifier is False
        assert arg.has_variadic_template_expansion is True
        assert arg.clean_name == "A"

    def test_expansion_with_variadic_identifier_allowed(self):
        arg = _p1("T<!a...>").template_arguments[0]
        assert isinstance(arg, ParsedType)
        assert arg.has_variadic_identifier is True
        assert arg.variadic_group_identifier == "!"
        assert arg.has_variadic_template_expansion is True
        assert arg.clean_name == "a"

    def test_expansion_with_dollar_identifier_allowed(self):
        arg = _p1("T<$a...>").template_arguments[0]
        assert isinstance(arg, ParsedType)
        assert arg.has_variadic_identifier is True
        assert arg.variadic_group_identifier == "$"
        assert arg.has_variadic_template_expansion is True
        assert arg.clean_name == "a"

    def test_literal_expansion_forbidden(self):
        with pytest.raises(
            CHSyntaxError,
            match="Template argument literals can not use the variadic template variable expansion operator '...'. "
            "Found at 4 of 'T<!2...>'",
        ):
            _p1("T<!2...>")

    def test_expansion_with_variadic_groups_forbidden(self):
        with pytest.raises(
            CHSyntaxError,
            match=r"Can not define template argument values combining variadic groups and types with variadic "
            r"identifiers! Found at 'T<!a..., \[a...\]>",
        ):
            _p1("T<!a..., [a...]>")

    def test_expansion_with_variadic_ids_in_variadic_group_forbidden(self):
        with pytest.raises(
            CHSyntaxError,
            match=r"Variadic group identifiers are only allowed in template arguments! Found at position 3 of "
            r"'T<\[!a...\]>'",
        ):
            _p1("T<[!a...]>")

    def test_expansion_followed_by_template_args_forbidden(self):
        with pytest.raises(
            CHSyntaxError,
            match=r"The variadic template expansion cannot be used with function arguments, template arguments",
        ):
            _p1("T(a...<int>)")

    def test_expansion_followed_by_func_args_forbidden(self):
        with pytest.raises(
            CHSyntaxError,
            match=r"The variadic template expansion cannot be used with function arguments, template arguments",
        ):
            _p1("T(a...(b))")

    def test_expansion_allowed_multiple_in_same_list(self):
        # Each entry is checked separately; first "A..." already triggers the error
        args = _p1("T<A..., B...>").template_arguments
        assert isinstance(args[0], ParsedType)
        assert args[0].has_variadic_identifier is False
        assert args[0].variadic_group_identifier is None
        assert args[0].has_variadic_template_expansion is True
        assert args[0].clean_name == "A"
        assert isinstance(args[1], ParsedType)
        assert args[1].has_variadic_identifier is False
        assert args[1].variadic_group_identifier is None
        assert args[1].has_variadic_template_expansion is True
        assert args[1].clean_name == "B"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Boolean literals
# ─────────────────────────────────────────────────────────────────────────────


class TestBoolLiterals:
    def test_true_literal_fields(self):
        arg = _p1("T<true>").template_arguments[0]
        assert arg.full_name == "true"
        assert arg.clean_name == "true"
        assert isinstance(arg, TemplateArgumentLiteral)
        assert arg.literal_type == "bool"

    def test_false_literal_fields(self):
        arg = _p1("T<false>").template_arguments[0]
        assert arg.full_name == "false"
        assert arg.clean_name == "false"
        assert isinstance(arg, TemplateArgumentLiteral)
        assert arg.literal_type == "bool"

    def test_true_and_false_together(self):
        t = _p1("T<true, false>")
        assert t.template_args == ("true", "false")
        t_arg1 = t.template_arguments[0]
        assert isinstance(t_arg1, TemplateArgumentLiteral)
        t_arg2 = t.template_arguments[1]
        assert isinstance(t_arg2, TemplateArgumentLiteral)
        assert t_arg1.literal_type == "bool"
        assert t_arg2.literal_type == "bool"

    def test_true_prefix_is_not_bool__word_boundary(self):
        # "trueType" starts with "true" but the word-boundary check sees 'T'
        arg = _p1("T<trueType>").template_arguments[0]
        assert arg.full_name == "trueType"
        assert isinstance(arg, ParsedType)

    def test_false_prefix_is_not_bool(self):
        arg = _p1("T<falsehood>").template_arguments[0]
        assert arg.full_name == "falsehood"
        assert isinstance(arg, ParsedType)

    def test_true_underscore_suffix_is_not_bool(self):
        arg = _p1("T<true_val>").template_arguments[0]
        assert isinstance(arg, ParsedType)
        assert arg.full_name == "true_val"

    def test_true_digit_suffix_is_not_bool(self):
        arg = _p1("T<true1>").template_arguments[0]
        assert isinstance(arg, ParsedType)
        assert arg.full_name == "true1"

    def test_truefalse_is_single_named_type(self):
        arg = _p1("T<truefalse>").template_arguments[0]
        assert isinstance(arg, ParsedType)
        assert arg.full_name == "truefalse"

    def test_bool_in_func_args_parses_as_named_type(self):
        # allow_literals=False for func args: "true" becomes a plain name
        arg = _p1("T(true)").function_arguments[0]
        assert arg.full_name == "true"
        assert isinstance(arg, ParsedType)

    def test_bool_template_args_entry(self):
        t = _p1("T<true>")
        assert t.template_args == ("true",)

    def test_bool_template_args_with_identifier_entry(self):
        t = _p1("T<!$true>")
        assert t.template_args == ("!$true",)
        lit = t.template_arguments[0]
        assert isinstance(lit, TemplateArgumentLiteral)
        assert lit.clean_name == "true"
        assert lit.has_variadic_identifier is True
        assert lit.variadic_group_identifier == "!$"

    def test_bool_template_args_with_expansion_entry(self):
        with pytest.raises(
            CHSyntaxError,
            match=r"Template argument literals can not use the variadic template variable expansion operator '...'. "
            r"Found at 6 of 'T<true...>'",
        ):
            _p1("T<true...>")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Number literals
# ─────────────────────────────────────────────────────────────────────────────


class TestNumberLiterals:
    def test_zero_int(self):
        arg = _p1("T<0>").template_arguments[0]
        assert arg.full_name == "0"
        assert arg.clean_name == "0"
        assert isinstance(arg, TemplateArgumentLiteral)
        assert arg.literal_type == "int"

    def test_positive_int(self):
        arg = _p1("T<42>").template_arguments[0]
        assert arg.full_name == "42"
        assert isinstance(arg, TemplateArgumentLiteral)
        assert arg.literal_type == "int"

    def test_negative_int(self):
        arg = _p1("T<-1>").template_arguments[0]
        assert arg.full_name == "-1"
        assert isinstance(arg, TemplateArgumentLiteral)
        assert arg.literal_type == "int"

    def test_float(self):
        arg = _p1("T<3.14>").template_arguments[0]
        assert arg.full_name == "3.14"
        assert isinstance(arg, TemplateArgumentLiteral)
        assert arg.literal_type == "float"

    def test_negative_float(self):
        arg = _p1("T<-3.14>").template_arguments[0]
        assert arg.full_name == "-3.14"
        assert isinstance(arg, TemplateArgumentLiteral)
        assert arg.literal_type == "float"

    def test_zero_float(self):
        arg = _p1("T<0.0>").template_arguments[0]
        assert arg.full_name == "0.0"
        assert isinstance(arg, TemplateArgumentLiteral)
        assert arg.literal_type == "float"

    def test_negative_zero_float(self):
        arg = _p1("T<-0.0>").template_arguments[0]
        assert arg.full_name == "-0.0"
        assert isinstance(arg, TemplateArgumentLiteral)
        assert arg.literal_type == "float"

    def test_large_int(self):
        arg = _p1("T<1234567890>").template_arguments[0]
        assert arg.full_name == "1234567890"
        assert isinstance(arg, TemplateArgumentLiteral)
        assert arg.literal_type == "int"

    def test_dot_only_is_float_with_no_decimal_digits(self):
        # "1." is consumed as "1" + "." → literal_type float
        arg = _p1("T<1.>").template_arguments[0]
        assert arg.full_name == "1."
        assert isinstance(arg, TemplateArgumentLiteral)
        assert arg.literal_type == "float"

    def test_multiple_numbers(self):
        t = _p1("T<1, 2, 3>")
        for i, sub in enumerate(t.template_arguments):
            assert sub.full_name == str(i + 1)
            assert isinstance(sub, TemplateArgumentLiteral)
            assert sub.literal_type == "int"

    def test_int_template_args_entry(self):
        t = _p1("T<42>")
        assert t.template_args == ("42",)

    def test_float_template_args_entry(self):
        t = _p1("T<3.14>")
        assert t.template_args == ("3.14",)

    def test_lone_minus_raises(self):
        with pytest.raises(CHSyntaxError, match=r"Expected a numeric literal at position"):
            _p1("T<->")

    def test_double_minus_raises(self):
        with pytest.raises(CHSyntaxError, match=r"Expected a numeric literal at position"):
            _p1("T<--1>")

    def test_only_double_minus_raises(self):
        with pytest.raises(CHSyntaxError, match=r"Expected a numeric literal at position"):
            _p1("T<-->")

    def test_double_dot_after_number_raises(self):
        with pytest.raises(
            CHSyntaxError, match=r"Expected '>' at position 3, got '.'\n  Full input : 'T<1..>'\n  Remaining  : '..>'"
        ):
            _p1("T<1..>")

    def test_double_dot_before_number_raises(self):
        with pytest.raises(CHSyntaxError, match=r"Expected a numeric literal at position"):
            _p1("T<..1>")

    def test_number_between_dots_raises(self):
        with pytest.raises(
            CHSyntaxError, match=r"Expected '>' at position 4, got '.'\n  Full input : 'T<.1.>'\n  Remaining  : '.>'"
        ):
            arg = _p1("T<.1.>").template_args[0]
            print(arg)

    def test_neg_dot_raises(self):
        with pytest.raises(CHSyntaxError, match=r"Expected a numeric literal at position"):
            _p1("T<-.>")

    def test_single_dot_raises(self):
        with pytest.raises(CHSyntaxError, match=r"Expected a numeric literal at position"):
            _p1("T<.>")

    def test_literal_expansion_raises(self):
        with pytest.raises(
            CHSyntaxError,
            match=r"Template argument literals can not use the variadic template variable expansion operator '...'. "
            r"Found at 3 of 'T<1...>'",
        ):
            _p1("T<1...>")

    def test_too_many_dots_raises(self):
        with pytest.raises(
            CHSyntaxError,
            match=r"Expected '>' at position 3, got '.'\n  Full input : 'T<1....>'\n  Remaining  : '....>'",
        ):
            _p1("T<1....>")

    def test_literal_combination_raises(self):
        with pytest.raises(
            CHSyntaxError,
            match=r"Expected '>' at position 3, got 't'\n  Full input : 'T<1true>'\n  Remaining  : 'true>'",
        ):
            _p1("T<1true>")

    def test_number_in_func_args_parsed_as_name(self):
        # allow_literals=False; digits are not stop-chars → "42" is a name
        arg = _p1("T(42)").function_arguments[0]
        assert arg.full_name == "42"
        assert isinstance(arg, ParsedType)


# ─────────────────────────────────────────────────────────────────────────────
# 10. String literals — quoting / unquoting
# ─────────────────────────────────────────────────────────────────────────────


class TestStringLiterals:
    """
    For string literals:
      full_name  = json.dumps(decoded_content)   — canonical JSON with surrounding quotes
      clean_name = decoded_content               — the actual Python string value

    The double-quote character ``"`` is in ``_STOP`` for ``_parse_type_name``,
    so a string literal at the domain (top) level causes a CHSyntaxError,
    while inside ``<>`` or ``[]`` it is parsed correctly.
    """

    # ── Basic cases ─────────────────────────────────────────────────────────

    def test_simple_string_fields(self):
        arg = _p1('T<"hello">').template_arguments[0]
        assert arg.full_name == '"hello"'
        assert arg.clean_name == "hello"
        assert isinstance(arg, TemplateArgumentLiteral)
        assert arg.literal_type == "string"

    def test_empty_string(self):
        arg = _p1('T<"">').template_arguments[0]
        assert arg.full_name == '""'
        assert arg.clean_name == ""
        assert isinstance(arg, TemplateArgumentLiteral)
        assert arg.literal_type == "string"

    def test_string_with_space(self):
        arg = _p1('T<"hello world">').template_arguments[0]
        assert arg.clean_name == "hello world"
        assert arg.full_name == '"hello world"'

    def test_string_with_digits(self):
        arg = _p1('T<"abc123">').template_arguments[0]
        assert arg.clean_name == "abc123"

    def test_string_in_template_args_tuple(self):
        t = _p1('T<"hello">')
        assert t.template_args == ('"hello"',)

    def test_full_name_is_canonical_json(self):
        # full_name must equal json.dumps(clean_name) for any string literal
        arg = _p1('T<"hello">').template_arguments[0]
        assert arg.full_name == json.dumps(arg.clean_name)

    # ── Backslash escaping ───────────────────────────────────────────────────

    def test_single_backslash(self):
        # Parser input contains "\\" (two backslash chars) → decoded: one backslash
        #   r'T<"\\">' raw string = T < " \ \ " >
        backslash = "\\"
        arg = _p1(r'T<"\\">').template_arguments[0]
        assert arg.clean_name == backslash
        assert arg.full_name == json.dumps(backslash)  # '"\\\\"'

    def test_double_backslash(self):
        # r'T<"\\\\">' has four backslash chars inside → decoded: two backslashes
        two_backslashes = "\\\\"
        arg = _p1(r'T<"\\\\">').template_arguments[0]
        assert arg.clean_name == two_backslashes
        assert arg.full_name == json.dumps(two_backslashes)

    def test_backslash_before_other_char(self):
        # "\n" escape → newline (tested separately); here we ensure the general
        # mechanism holds: backslash pairs are consumed by \\.  in the regex
        arg = _p1(r'T<"\n">').template_arguments[0]
        assert arg.clean_name == "\n"
        assert arg.full_name == json.dumps("\n")  # '"\\n"'

    def test_backslash_at_end_of_input_unmatched(self):
        # "hello\" in the input → the backslash escapes the closing quote,
        # leaving no terminating '"' → Unterminated
        with pytest.raises(CHSyntaxError, match=r"Unterminated quoted string"):
            _p1(r'T<"hello\">')  # raw: T<"hello\"> — backslash + closing "

    # ── Quote escaping ───────────────────────────────────────────────────────

    def test_escaped_quote_alone(self):
        # r'T<"\"">': chars after T< are: " \ " "
        # → one escaped double-quote inside → decoded: "
        arg = _p1(r'T<"\"">').template_arguments[0]
        assert arg.clean_name == '"'
        assert arg.full_name == json.dumps('"')  # '"\""'

    def test_embedded_quotes_say_hi(self):
        # r'T<"say \"hi\"">': parser sees: "say \"hi\""
        arg = _p1(r'T<"say \"hi\"">').template_arguments[0]
        assert arg.clean_name == 'say "hi"'
        assert arg.full_name == json.dumps('say "hi"')

    def test_only_quotes_inside_string(self):
        # r'T<"\"\"">': decoded: "" (two double-quote chars)
        arg = _p1(r'T<"\"\"">').template_arguments[0]
        assert arg.clean_name == '""'
        assert arg.full_name == json.dumps('""')

    def test_quote_at_end_of_string(self):
        arg = _p1(r'T<"hello\"">').template_arguments[0]
        assert arg.clean_name == 'hello"'
        assert arg.full_name == json.dumps('hello"')

    def test_quote_at_start_of_string(self):
        arg = _p1(r'T<"\"hello">').template_arguments[0]
        assert arg.clean_name == '"hello'
        assert arg.full_name == json.dumps('"hello')

    # ── Standard JSON escape sequences ──────────────────────────────────────

    def test_newline_escape(self):
        arg = _p1(r'T<"\n">').template_arguments[0]
        assert arg.clean_name == "\n"
        assert arg.full_name == '"\\n"'

    def test_tab_escape(self):
        arg = _p1(r'T<"\t">').template_arguments[0]
        assert arg.clean_name == "\t"
        assert arg.full_name == '"\\t"'

    def test_carriage_return_escape(self):
        arg = _p1(r'T<"\r">').template_arguments[0]
        assert arg.clean_name == "\r"
        assert arg.full_name == '"\\r"'

    def test_backspace_escape(self):
        arg = _p1(r'T<"\b">').template_arguments[0]
        assert arg.clean_name == "\b"
        assert arg.full_name == '"\\b"'

    def test_form_feed_escape(self):
        arg = _p1(r'T<"\f">').template_arguments[0]
        assert arg.clean_name == "\f"
        assert arg.full_name == '"\\f"'

    def test_forward_slash_escape(self):
        # JSON allows \/ as an alternative encoding for /
        arg = _p1(r'T<"\/">').template_arguments[0]
        assert arg.clean_name == "/"
        # json.dumps("/") does NOT escape slashes in Python's stdlib
        assert arg.full_name == json.dumps("/")

    # ── Unicode escapes ──────────────────────────────────────────────────────

    def test_unicode_escape_ascii(self):
        # \u0041 = 'A'
        arg = _p1(r'T<"\u0041">').template_arguments[0]
        assert arg.clean_name == "A"
        assert arg.full_name == json.dumps("A")  # '"A"'

    def test_unicode_escape_latin(self):
        # \u00e9 = é
        arg = _p1(r'T<"\u00e9">').template_arguments[0]
        assert arg.clean_name == "\u00e9"

    def test_unicode_escape_cjk(self):
        # \u4e2d = 中
        arg = _p1(r'T<"\u4e2d">').template_arguments[0]
        assert arg.clean_name == "\u4e2d"

    def test_unicode_escape_for_quote(self):
        # \u0022 is the Unicode code point for double-quote
        arg = _p1(r'T<"\u0022">').template_arguments[0]
        assert arg.clean_name == '"'
        # canonical encoding uses \" not \u0022
        assert arg.full_name == json.dumps('"')

    def test_unicode_zero(self):
        # \u0000 = null character (below #x0020; must be escaped in JSON)
        arg = _p1(r'T<"\u0000">').template_arguments[0]
        assert arg.clean_name == "\u0000"
        assert arg.full_name == json.dumps("\u0000")

    # ── Combinations ────────────────────────────────────────────────────────

    def test_two_string_literals(self):
        t = _p1('T<"hello", "world">')
        assert t.template_arguments[0].clean_name == "hello"
        assert t.template_arguments[1].clean_name == "world"
        assert t.template_args == ('"hello"', '"world"')

    def test_string_mixed_with_other_literal_types(self):
        t = _p1('T<42, "hello", true>')
        arg = t.template_arguments[0]
        assert isinstance(arg, TemplateArgumentLiteral)
        assert arg.literal_type == "int"
        arg = t.template_arguments[1]
        assert isinstance(arg, TemplateArgumentLiteral)
        assert arg.literal_type == "string"
        arg = t.template_arguments[2]
        assert isinstance(arg, TemplateArgumentLiteral)
        assert arg.literal_type == "bool"

    def test_string_with_backslash_and_quote_combined(self):
        # r'T<"a\\\"b">': input chars after T< are: " a \ \ \ " b "
        # \\  → one backslash;  \"  → one quote  → decoded: a\"b
        arg = _p1(r'T<"a\\\"b">').template_arguments[0]
        assert arg.clean_name == 'a\\"b'  # a + backslash + " + b
        assert arg.full_name == json.dumps('a\\"b')

    def test_all_basic_escapes_in_one_string(self):
        arg = _p1(r'T<"\n\t\r\b\f\\\"">').template_arguments[0]
        expected = '\n\t\r\b\f\\"'
        assert arg.clean_name == expected
        assert arg.full_name == json.dumps(expected)

    # ── Error cases ──────────────────────────────────────────────────────────

    def test_unterminated_string_in_template_arg(self):
        with pytest.raises(CHSyntaxError, match=r"Unterminated quoted string"):
            _p1('T<"hello>')  # missing closing "

    def test_unterminated_string_at_eof(self):
        with pytest.raises(CHSyntaxError, match=r"Unterminated quoted string"):
            _p1('T<"hello')  # neither " nor > to close

    def test_unclosed_template_after_complete_string_literal(self):
        # The literal itself is fine; the enclosing <> is unclosed
        with pytest.raises(CHSyntaxError, match=r"Expected '>'"):
            _p1('T<"hello"')  # missing closing >

    def test_invalid_escape_sequence(self):
        with pytest.raises(CHSyntaxError, match=r"Invalid escape sequence in quoted string"):
            _p1(r'T<"\q">')

    def test_invalid_escape_x_sequence(self):
        # \x is not valid JSON (unlike Python); JSON only allows \uXXXX
        with pytest.raises(CHSyntaxError, match=r"Invalid escape sequence in quoted string"):
            _p1(r'T<"\x41">')

    def test_string_literal_not_allowed_in_func_args(self):
        # '"' is in _STOP; _parse_type_name returns empty → CHSyntaxError
        with pytest.raises(CHSyntaxError, match=r"Expected a type name at position"):
            _p1('T("hello")')

    def test_string_literal_not_allowed_at_top_level(self):
        # Same mechanism: '"' causes _parse_type_name to see an empty name
        with pytest.raises(CHSyntaxError, match=r"Expected a type name at position"):
            _parse('"hello"')

    def test_string_literal_not_allowed_as_second_top_level(self):
        with pytest.raises(CHSyntaxError, match=r"Expected a type name at position"):
            _parse('T, "hello"')


# ─────────────────────────────────────────────────────────────────────────────
# 11. Literals inside variadic groups
# ─────────────────────────────────────────────────────────────────────────────


class TestLiteralsInVariadicGroups:
    def test_bool_in_group(self):
        t = _p1("T<[true, false]>")
        grp = t.template_arguments[0]
        assert isinstance(grp, TemplateArgumentVariadicGroup)
        assert grp.variadic_group[0].literal_type == "bool"
        assert grp.variadic_group[0].full_name == "true"
        assert grp.variadic_group[1].literal_type == "bool"
        assert grp.variadic_group[1].full_name == "false"

    def test_int_in_group(self):
        t = _p1("T<[1, 2]>")
        grp = t.template_arguments[0]
        assert isinstance(grp, TemplateArgumentVariadicGroup)
        assert grp.variadic_group[0].literal_type == "int"
        assert grp.variadic_group[1].literal_type == "int"

    def test_float_in_group(self):
        t = _p1("T<[3.14, -1.0]>")
        grp = t.template_arguments[0]
        assert isinstance(grp, TemplateArgumentVariadicGroup)
        assert grp.variadic_group[0].literal_type == "float"
        assert grp.variadic_group[1].literal_type == "float"

    def test_string_in_group(self):
        t = _p1('T<["hello", A]>')
        grp = t.template_arguments[0]
        assert isinstance(grp, TemplateArgumentVariadicGroup)
        s = grp.variadic_group[0]
        assert s.literal_type == "string"
        assert s.clean_name == "hello"
        assert s.full_name == '"hello"'
        a = grp.variadic_group[1]
        assert not isinstance(a, TemplateArgumentLiteral)

    def test_string_with_backslash_in_group(self):
        # r'T<["\\"]>': string literal containing one backslash
        grp = _p1(r'T<["\\"]>').template_arguments[0]
        assert isinstance(grp, TemplateArgumentVariadicGroup)
        arg = grp.variadic_group[0]
        assert arg.literal_type == "string"
        assert arg.clean_name == "\\"

    def test_string_with_embedded_quote_in_group(self):
        grp = _p1(r'T<["\""]>').template_arguments[0]
        assert isinstance(grp, TemplateArgumentVariadicGroup)
        arg = grp.variadic_group[0]
        assert arg.literal_type == "string"
        assert arg.clean_name == '"'

    def test_mixed_literals_and_named_type(self):
        t = _p1('T<[42, "hi", true, A]>')
        grp = t.template_arguments[0]
        assert isinstance(grp, TemplateArgumentVariadicGroup)
        elems = grp.variadic_group
        arg = elems[0]
        assert isinstance(arg, TemplateArgumentLiteral)
        assert arg.literal_type == "int"
        assert arg.full_name == "42"
        arg = elems[1]
        assert isinstance(arg, TemplateArgumentLiteral)
        assert arg.literal_type == "string"
        assert arg.full_name == '"hi"'
        assert arg.clean_name == "hi"
        arg = elems[2]
        assert isinstance(arg, TemplateArgumentLiteral)
        assert arg.literal_type == "bool"
        assert arg.full_name == "true"
        arg = elems[3]
        assert isinstance(arg, ParsedType)
        assert arg.full_name == "A"

    def test_group_full_name_encodes_literal_full_names(self):
        # The group's full_name uses each element's full_name, so string
        # literals appear with their surrounding quotes.
        t = _p1('T<[42, "hi"]>')
        grp = t.template_arguments[0]
        assert grp.full_name == '[42, "hi"]'

    def test_template_args_tuple_for_group_with_literals(self):
        t = _p1('T<[42, "hi", true, A]>')
        # Template args: one variadic group entry → tuple-of-tuples
        assert t.template_args == (("42", '"hi"', "true", "A"),)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Nested / complex expressions
# ─────────────────────────────────────────────────────────────────────────────


class TestNestedTypes:
    def test_deeply_nested_three_levels(self):
        t = _p1("A<B<C<D>>>")
        assert t.full_name == "A<B<C<D>>>"
        assert t.template_args == ("B<C<D>>",)
        b = t.template_arguments[0]
        assert isinstance(b, ParsedType)
        assert b.template_args == ("C<D>",)
        c = b.template_arguments[0]
        assert isinstance(c, ParsedType)
        assert c.template_args == ("D",)

    def test_nested_with_literals(self):
        t = _p1('Outer<Inner<42, "hi">>')
        inner = t.template_arguments[0]
        assert inner.clean_name == "Inner"
        assert isinstance(inner, ParsedType)
        arg = inner.template_arguments[0]
        assert isinstance(arg, TemplateArgumentLiteral)
        assert arg.literal_type == "int"
        arg = inner.template_arguments[1]
        assert isinstance(arg, TemplateArgumentLiteral)
        assert arg.literal_type == "string"
        assert arg.clean_name == "hi"

    def test_group_nested_in_complex_type(self):
        t = _p1("T<[A, B]>(c)")
        assert t.template_args == (("A", "B"),)
        assert t.func_args == ("c",)
        assert t.full_name == "T<[A, B]>(c)"

    def test_multiple_top_level_complex_types(self):
        result = _parse("Map<K, V>, Set<T>")
        assert len(result) == 2
        assert result[0].clean_name == "Map"
        assert result[0].template_args == ("K", "V")
        assert result[1].clean_name == "Set"

    def test_two_groups_and_literals(self):
        t = _p1('T<[1, 2], ["a", "b"]>')
        assert len(t.template_arguments) == 2
        g0 = t.template_arguments[0]
        assert isinstance(g0, TemplateArgumentVariadicGroup)
        g1 = t.template_arguments[1]
        assert isinstance(g1, TemplateArgumentVariadicGroup)
        assert g0.variadic_group[0].literal_type == "int"
        assert g1.variadic_group[0].literal_type == "string"
        assert g1.variadic_group[0].clean_name == "a"

    def test_full_name_roundtrip_complex(self):
        # Parsing should reconstruct the exact full_name
        text = "Map<List<int>, Set<String>>"
        t = _p1(text)
        assert t.full_name == text


# ─────────────────────────────────────────────────────────────────────────────
# 13. Registry
# ─────────────────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_simple_type_registry(self):
        t = _p1("MyType")
        assert t.registry == frozendict({"MyType": ("MyType", None, None)})

    def test_template_arg_in_registry(self):
        t = _p1("Map<K, V>")
        reg = t.registry
        assert "K" in reg
        assert "V" in reg
        assert "Map<K, V>" in reg
        assert reg["Map<K, V>"] == ("Map", ("K", "V"), None)
        assert reg["K"] == ("K", None, None)
        assert reg["V"] == ("V", None, None)

    def test_nested_types_in_registry(self):
        t = _p1("Map<List<int>>")
        reg = t.registry
        assert "int" in reg
        assert "List<int>" in reg
        assert "Map<List<int>>" in reg

    def test_func_args_in_registry(self):
        t = _p1("T(a, b)")
        reg = t.registry
        assert "T(a, b)" in reg
        assert "a" in reg
        assert "b" in reg
        assert reg["T(a, b)"] == ("T", None, ("a", "b"))

    def test_int_literal_registry_key(self):
        arg = _p1("T<42>").template_arguments[0]
        assert "42" in arg.registry
        # key = full_name; value[0] = clean_name; for ints they are the same
        assert arg.registry["42"][0] == "42"
        assert arg.registry["42"][1] is None
        assert arg.registry["42"][2] is None

    def test_bool_literal_registry_key(self):
        arg = _p1("T<true>").template_arguments[0]
        assert "true" in arg.registry
        assert arg.registry["true"][0] == "true"

    def test_string_literal_registry_key_is_quoted(self):
        # The registry key for a string literal is the json.dumps form (with quotes)
        arg = _p1('T<"hello">').template_arguments[0]
        assert '"hello"' in arg.registry
        assert arg.registry['"hello"'][0] == "hello"  # clean_name (decoded)
        assert arg.registry['"hello"'][1] is None
        assert arg.registry['"hello"'][2] is None

    def test_string_literal_with_embedded_quote_registry_key(self):
        # key must be the canonical json.dumps form
        arg = _p1(r'T<"say \"hi\"">').template_arguments[0]
        expected_key = json.dumps('say "hi"')  # '"say \\"hi\\""'
        assert expected_key in arg.registry
        assert arg.registry[expected_key][0] == 'say "hi"'

    def test_registry_key_not_named_type_for_string(self):
        # A named type "hello" and string literal "hello" must be distinguishable;
        # the literal's key is '"hello"' (with quotes), not 'hello'
        arg = _p1('T<"hello">').template_arguments[0]
        assert "hello" not in arg.registry  # without quotes → absent
        assert '"hello"' in arg.registry  # with quotes → present


# ─────────────────────────────────────────────────────────────────────────────
# 14. Trailing-comma opt-in
# ─────────────────────────────────────────────────────────────────────────────


class TestTrailingComma:
    def test_trailing_comma_allowed_in_template_args(self):
        t = TypeParser("T<A,>", allow_trailing_comma=True).parse_types()[0]
        assert t.template_args == ("A",)

    def test_trailing_comma_allowed_in_func_args(self):
        t = TypeParser("T(a,)", allow_trailing_comma=True).parse_types()[0]
        assert t.func_args == ("a",)

    def test_trailing_comma_allowed_at_top_level(self):
        result = TypeParser("A, B,", allow_trailing_comma=True).parse_types()
        assert len(result) == 2

    def test_trailing_comma_with_literal(self):
        t = TypeParser("T<42,>", allow_trailing_comma=True).parse_types()[0]
        assert t.template_args == ("42",)

    def test_trailing_comma_disallowed_by_default_in_template_args(self):
        with pytest.raises(CHSyntaxError, match=r"Expected a value, but finished parsing"):
            _p1("T<A,>")

    def test_trailing_comma_disallowed_by_default_in_func_args(self):
        with pytest.raises(CHSyntaxError, match=r"Expected a value, but finished parsing"):
            _p1("T(a,)")

    def test_trailing_comma_disallowed_by_default_at_top_level(self):
        with pytest.raises(CHSyntaxError, match=r"Expected a value, but finished parsing"):
            _parse("A,")

    def test_double_trailing_comma_still_errors_even_with_allow(self):
        # "T<A,,>" – the second comma causes _parse_type_name to see a stop-char
        with pytest.raises(CHSyntaxError, match=r"Expected a type name at position"):
            TypeParser("T<A,,>", allow_trailing_comma=True).parse_types()


# ─────────────────────────────────────────────────────────────────────────────
# 15. parse_type public API
# ─────────────────────────────────────────────────────────────────────────────


class TestParseTypePublicAPI:
    def test_simple_parse(self):
        result = parse_type("MyType")
        assert len(result) == 1
        assert result[0].full_name == "MyType"

    def test_expected_count_match(self):
        result = parse_type("A, B", expected_number_of_values=2)
        assert len(result) == 2

    def test_expected_count_none_accepts_any(self):
        result = parse_type("A, B, C")
        assert len(result) == 3

    def test_expected_count_mismatch_raises_runtime_error(self):
        with pytest.raises(
            RuntimeError,
            match=r"Parsing 'A' failed: expected 2 values, got 1",
        ):
            parse_type("A", expected_number_of_values=2)

    def test_expected_count_too_many_raises(self):
        with pytest.raises(
            RuntimeError,
            match=r"Parsing 'A, B' failed: expected 1 values, got 2",
        ):
            parse_type("A, B", expected_number_of_values=1)

    def test_results_are_memoized(self):
        r1 = parse_type("MyType")
        r2 = parse_type("MyType")
        assert r1 is r2  # same tuple object via lru_cache

    def test_empty_domain_returns_empty_tuple(self):
        assert parse_type("") == ()


# ─────────────────────────────────────────────────────────────────────────────
# 16. General error cases
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorCasesGeneral:
    # ── Unexpected trailing content ──────────────────────────────────────────

    def test_stray_closing_angle_bracket(self):
        with pytest.raises(CHSyntaxError, match=r"Unexpected trailing content after input"):
            _parse("T>")

    def test_stray_closing_paren(self):
        with pytest.raises(CHSyntaxError, match=r"Unexpected trailing content after input"):
            _parse("T)")

    def test_two_types_without_separator(self):
        # "A B" — after parsing "A", the parser sees " B" which is not consumed
        with pytest.raises(CHSyntaxError, match=r"Unexpected trailing content after input"):
            _parse("A B")

    # ── Empty / missing type name ────────────────────────────────────────────

    def test_stray_closing_bracket_at_start(self):
        # ']' is in _STOP → _parse_type_name immediately returns empty → CHSyntaxError
        with pytest.raises(CHSyntaxError, match=r"Expected a type name at position"):
            _parse("]")

    def test_opening_angle_without_name(self):
        with pytest.raises(CHSyntaxError, match=r"Expected a type name at position"):
            _p1("<T>")

    def test_opening_paren_without_name(self):
        with pytest.raises(CHSyntaxError, match=r"Expected a type name at position"):
            _p1("(a)")

    def test_leading_comma(self):
        with pytest.raises(CHSyntaxError, match=r"Expected a type name at position"):
            _parse(",T")

    def test_leading_comma_in_template_args(self):
        with pytest.raises(CHSyntaxError, match=r"Expected a type name at position"):
            _p1("T<,>")

    # ── Unclosed delimiters ──────────────────────────────────────────────────

    def test_unclosed_template_args(self):
        with pytest.raises(CHSyntaxError, match=r"Expected '>'"):
            _p1("T<")

    def test_unclosed_func_args(self):
        with pytest.raises(CHSyntaxError, match=r"Expected '\)'"):
            _p1("T(")

    def test_unclosed_variadic_group_closed_by_angle(self):
        # '[A' in template args; ']' expected but '>' found
        with pytest.raises(CHSyntaxError, match=r"Expected '\]'"):
            _p1("T<[A>")

    def test_unclosed_variadic_group_at_eof(self):
        with pytest.raises(CHSyntaxError, match=r"Expected '\]'"):
            _p1("T<[A")

    # ── Trailing comma without a following value ─────────────────────────────

    def test_trailing_comma_in_template_args(self):
        with pytest.raises(CHSyntaxError, match=r"Expected a value, but finished parsing"):
            _p1("T<A,>")

    def test_trailing_comma_in_func_args(self):
        with pytest.raises(CHSyntaxError, match=r"Expected a value, but finished parsing"):
            _p1("T(a,)")

    def test_trailing_comma_at_top_level(self):
        with pytest.raises(CHSyntaxError, match=r"Expected a value, but finished parsing"):
            _parse("A,")

    # ── Variadic identifier misuse ───────────────────────────────────────────

    def test_var_id_at_top_level(self):
        with pytest.raises(
            CHSyntaxError,
            match=r"Variadic group identifiers are only allowed in template arguments",
        ):
            _parse("!T")

    def test_dollar_var_id_at_top_level(self):
        with pytest.raises(
            CHSyntaxError,
            match=r"Variadic group identifiers are only allowed in template arguments",
        ):
            _parse("$T")

    def test_var_id_in_func_args(self):
        with pytest.raises(
            CHSyntaxError,
            match=r"Variadic group identifiers are only allowed in template arguments",
        ):
            _p1("T(!a)")

    def test_var_id_in_variadic_group(self):
        with pytest.raises(
            CHSyntaxError,
            match=r"Variadic group identifiers are only allowed in template arguments",
        ):
            _p1("T<[$a]>")

    def test_var_id_immediately_before_bracket_group(self):
        with pytest.raises(
            CHSyntaxError,
            match=r"Variadic group identifier '!' before a bracket group",
        ):
            _p1("T<![A]>")

    def test_dollar_var_id_immediately_before_bracket_group(self):
        with pytest.raises(
            CHSyntaxError,
            match=r"Variadic group identifier '\$' before a bracket group",
        ):
            _p1("T<$[A]>")

    # ── Variadic group misuse ────────────────────────────────────────────────

    def test_group_at_top_level(self):
        with pytest.raises(
            CHSyntaxError,
            match=r"Variadic groups are only allowed in template arguments",
        ):
            _parse("[A]")

    def test_group_in_func_args(self):
        with pytest.raises(
            CHSyntaxError,
            match=r"Variadic groups are only allowed in template arguments",
        ):
            _p1("T([A])")

    def test_nested_group_inside_group(self):
        with pytest.raises(
            CHSyntaxError,
            match=r"Variadic groups are only allowed in template arguments",
        ):
            _p1("T<[[A]]>")

    # ── Mixed variadic modes ─────────────────────────────────────────────────

    def test_mixing_group_and_var_id__first_group(self):
        with pytest.raises(
            CHSyntaxError,
            match=r"Can not define template argument values combining variadic groups and types with variadic "
            r"identifiers",
        ):
            _p1("T<[A, B], $C>")

    def test_mixing_group_and_var_id__first_var_id(self):
        with pytest.raises(
            CHSyntaxError,
            match=r"Can not define template argument values combining variadic groups and types with variadic "
            r"identifiers",
        ):
            _p1("T<$A, [B, C]>")

    # ── Expansion operator misuse ─────────────────────────────────────────────

    def test_expansion_not_allowed_at_top_level(self):
        with pytest.raises(
            CHSyntaxError,
            match=r"The template expansion operator is only allowed in a type application without variadic groups, in a"
            r" variadic group, or in function arguments! Found at 0 of 'A...'",
        ):
            _parse("A...")


# ─────────────────────────────────────────────────────────────────────────────
# 17. ParsedType properties: is_templated / has_arguments
# ─────────────────────────────────────────────────────────────────────────────


class TestParsedTypeProperties:
    def test_is_templated_true(self):
        assert _p1("T<A>").is_templated is True

    def test_is_templated_empty_brackets(self):
        assert _p1("T<>").is_templated is True

    def test_is_templated_false_plain(self):
        assert _p1("T").is_templated is False

    def test_has_arguments_true(self):
        assert _p1("T(a)").has_arguments is True

    def test_has_arguments_empty_parens(self):
        assert _p1("T()").has_arguments is True

    def test_has_arguments_false_plain(self):
        assert _p1("T").has_arguments is False

    def test_is_literal_for_bool_in_sub_types(self):
        arg = _p1("T<true>").template_arguments[0]
        assert isinstance(arg, TemplateArgumentLiteral)

    def test_is_literal_for_int_in_sub_types(self):
        arg = _p1("T<42>").template_arguments[0]
        assert isinstance(arg, TemplateArgumentLiteral)

    def test_is_literal_for_string_in_sub_types(self):
        arg = _p1('T<"hi">').template_arguments[0]
        assert isinstance(arg, TemplateArgumentLiteral)


# ─────────────────────────────────────────────────────────────────────────────
# 18. is_variadic_group property
# ─────────────────────────────────────────────────────────────────────────────


class TestIsVariadicGroupProperty:
    def test_is_variadic_group_does_not_raise_on_plain_named_type(self):
        t = _p1("MyType")
        assert isinstance(t, ParsedType)

    def test_is_variadic_group_true_for_actual_group(self):
        grp = _p1("T<[A, B]>").template_arguments[0]
        assert isinstance(grp, TemplateArgumentVariadicGroup)

    def test_is_variadic_group_false_for_expansion_type(self):
        arg = _p1("T(a...)").function_arguments[0]
        assert isinstance(arg, ParsedType)

    def test_is_variadic_group_false_for_empty_string_literal(self):
        arg = _p1('T<"">').template_arguments[0]
        assert isinstance(arg, TemplateArgumentLiteral)
