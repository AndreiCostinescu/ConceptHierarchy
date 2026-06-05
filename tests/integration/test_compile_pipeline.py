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

"""Integration tests: full compile pipeline → C++ output."""

import json
import textwrap

import pytest

from concept_hierarchy.compiler import compile_hierarchy


ANIMAL_KINGDOM = json.dumps(
    {
        "name": "AnimalKingdom",
        "concepts": [
            {
                "name": "Animal",
                "attributes": {"age": "int", "name": "string"},
                "description": "Base animal.",
            },
            {
                "name": "Dog",
                "parent": "Animal",
                "attributes": {"breed": "string"},
            },
        ],
    }
)


class TestCompileHierarchy:
    def test_returns_string(self):
        code = compile_hierarchy(ANIMAL_KINGDOM, target="cpp")
        assert isinstance(code, str)

    def test_contains_animal_struct(self):
        code = compile_hierarchy(ANIMAL_KINGDOM, target="cpp")
        assert "struct Animal" in code

    def test_inheritance_present(self):
        code = compile_hierarchy(ANIMAL_KINGDOM, target="cpp")
        assert "struct Dog" in code
        assert "Animal" in code  # Dog inherits from Animal

    def test_attribute_present(self):
        code = compile_hierarchy(ANIMAL_KINGDOM, target="cpp")
        assert "breed" in code

    def test_pragma_once(self):
        code = compile_hierarchy(ANIMAL_KINGDOM, target="cpp")
        assert "#pragma once" in code

    def test_unsupported_target_raises(self):
        with pytest.raises(ValueError, match="java"):
            compile_hierarchy(ANIMAL_KINGDOM, target="java")

    def test_write_to_file(self, tmp_path):
        out = tmp_path / "out.hpp"
        compile_hierarchy(ANIMAL_KINGDOM, target="cpp", output_path=str(out))
        assert out.exists()
        assert "struct Animal" in out.read_text()
