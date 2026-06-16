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
string_parser.py — Contains the base class for parsing string data containig string parsing utilities.
"""

import json
import re


class StringParser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def remaining(self) -> str:
        return self.text[self.pos :]

    def peek(self, n: int = 1) -> str:
        return self.text[self.pos : self.pos + n]

    def starts_with(self, prefix: str) -> bool:
        return self.text[self.pos :].startswith(prefix)

    def consume(self, expected: str) -> None:
        if not self.starts_with(expected):
            got = self.text[self.pos : self.pos + len(expected)]
            raise RuntimeError(
                f"Expected {expected!r} at position {self.pos}, got {got!r}\n"
                f"  Full input : {self.text!r}\n"
                f"  Remaining  : {self.remaining()!r}"
            )
        self.pos += len(expected)

    def try_consume(self, s: str) -> bool:
        if self.text[self.pos :].startswith(s):
            self.pos += len(s)
            return True
        return False

    def skip_whitespace(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in " \t\n\r":
            self.pos += 1

    def eof(self) -> bool:
        return self.pos >= len(self.text)

    def check_finished(self):
        self.skip_whitespace()
        if not self.eof():
            raise SyntaxError(f"Unexpected trailing content after input: {self.remaining()!r}")

    # --- Low-level lexers ---

    def _parse_string_literal(self, surround_result_with_quotes: bool) -> str:
        """
        Parse a JSON-style double-quoted string; return the unescaped content.

        The returned raw_value can include the surrounding double-quote characters.
        If surround_result_with_quotes is True, the returned value will be
            "hello"         ->  raw_value = '"hello"'
            "say \\"hi\\""  ->  raw_value = '"say \\"hi\\""'
        """
        if self.peek() != '"':
            raise SyntaxError(f"Expected '\"' at position {self.pos}; got {self.remaining()[:10]!r}")
        self.consume('"')
        m = re.match(r'((?:[^"\\]|\\.)*)"', self.remaining())
        if not m:
            raise SyntaxError(f"Unterminated quoted string at position {self.pos}")
        raw = m.group(1)
        self.pos += len(m.group())
        # Decode JSON escape sequences (\\, \", \n, \uXXXX, etc.) so that the
        # stored string matches actual Python dict keys / values.
        try:
            content = json.loads('"' + raw + '"')
        except json.JSONDecodeError as exc:
            raise SyntaxError(f"Invalid escape sequence in quoted string at position {self.pos}: {exc}") from exc
        if not surround_result_with_quotes:
            return content
        return '"' + content + '"'

    def _parse_bool_literal(self) -> str:
        """
        Parse 'true' or 'false' (JSON boolean syntax).

        A word-boundary check guards against accidentally consuming a prefix of some longer lower-case token
        """
        for keyword in ("true", "false"):
            if self.starts_with(keyword):
                end = self.pos + len(keyword)
                if end < len(self.text) and (self.text[end].isalnum() or self.text[end] == "_"):
                    raise RuntimeError(
                        f"Expected boolean literal ('true'/'false') at position {self.pos} "
                        f"but found a longer identifier: {self.text[self.pos : end + 1]!r}"
                    )
                self.pos = end
                return keyword
        raise RuntimeError(f"Expected 'true' or 'false' at position {self.pos}, got {self.remaining()!r}")

    def _parse_number_literal(self) -> str:
        """
        Parse an integer or floating-point literal, including optional leading '-'.

        Strategy: consume the widest valid numeric token, then delegate type
        detection to is_integer / is_number so that the exact same rules apply
        here as everywhere else in the system.

            "3"     ->  int     "3"
            "-1"    ->  int     "-1"
            "3.14"  ->  float   "3.14"
            "3.0"   ->  float   "3.0"
        """
        start = self.pos
        if self.peek() == "-":
            self.pos += 1
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
        if self.pos < len(self.text) and self.text[self.pos] == ".":
            self.pos += 1
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self.pos += 1

        raw = self.text[start : self.pos]
        if not raw or raw == "-":
            raise RuntimeError(f"Expected a numeric literal at position {start}, got {self.remaining()!r}")

        return raw

    def _parse_natural(self) -> str:
        m = re.match(r"\d+", self.remaining())
        if not m:
            raise SyntaxError(f"Expected a natural number at position {self.pos}; got {self.remaining()[:10]!r}")
        self.pos += len(m.group())
        return m.group()

    def _parse_upper_case_name(self) -> str:
        m = re.match(r"[A-Z][A-Za-z0-9_]*", self.remaining())
        if not m:
            raise SyntaxError(f"Expected upperCaseName at position {self.pos}; got {self.remaining()[:20]!r}")
        self.pos += len(m.group())
        return m.group()

    def _consume_balanced(self, open_ch: str, close_ch: str) -> str:
        """
        Consume a balanced delimited sequence (handles nesting).
        Returns the full matched text including the delimiters.
        """
        if self.peek() != open_ch:
            raise SyntaxError(f"Expected {open_ch!r} at position {self.pos}; got {self.remaining()[:10]!r}")
        depth = 0
        start = self.pos
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    self.pos += 1
                    return self.text[start : self.pos]
            self.pos += 1
        raise SyntaxError(f"Unmatched {open_ch!r} starting at position {start}")
