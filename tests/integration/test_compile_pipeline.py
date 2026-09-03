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

import pytest

from concept_hierarchy.compiler import ch_compile_from_json

ANIMAL_KINGDOM = {
    "name": "AnimalKingdom",
    "concepts": {
        "Concept": {},
        "Animal": {
            "directParents": ["Concept"],
            "description": "Base animal",
            "data": {"properties": {"age": "Integer", "name": "String"}},
        },
        "Dog": {"directParents": ["Animal"], "data": {"properties": {"breed": "String"}}},
        "ValueDomain": {"directParents": ["Concept"], "data": {}, "abstract": True},
        "Integer": {"directParents": ["ValueDomain"], "data": {"instantiation": "integer"}},
        "String": {"directParents": ["ValueDomain"], "data": {"instantiation": "string"}},
    },
}


class TestCompileHierarchy:
    def test_returns_string(self):
        code = ch_compile_from_json(ANIMAL_KINGDOM, target="cpp")
        assert isinstance(code, str)

    def test_contains_animal_struct(self):
        code = ch_compile_from_json(ANIMAL_KINGDOM, target="cpp")
        assert "struct Animal" in code

    def test_inheritance_present(self):
        code = ch_compile_from_json(ANIMAL_KINGDOM, target="cpp")
        assert "struct Dog" in code
        assert "Animal" in code  # Dog inherits from Animal

    def test_attribute_present(self):
        code = ch_compile_from_json(ANIMAL_KINGDOM, target="cpp")
        assert "breed" in code

    def test_pragma_once(self):
        code = ch_compile_from_json(ANIMAL_KINGDOM, target="cpp")
        assert "#pragma once" in code

    def test_unsupported_target_raises(self):
        with pytest.raises(ValueError, match="java"):
            ch_compile_from_json(ANIMAL_KINGDOM, target="java")

    def test_write_to_file(self, tmp_path):
        out = tmp_path / "out.hpp"
        ch_compile_from_json(ANIMAL_KINGDOM, target="cpp", output_path=str(out))
        assert out.exists()
        assert "struct Animal" in out.read_text()


ALIASED_KINGDOM = {
    "name": "AliasedKingdom",
    "concepts": {
        "Concept": {},
        "Animal": {"directParents": ["Concept"], "data": {"properties": {"age": "Integer"}}},
        # a concept alias, and a type alias over a templatable ValueDomain
        "Beast": "Animal",
        "Critter": "Beast",
        "IntBox": "Box<Integer>",
        "ValueDomain": {"directParents": ["Concept"], "data": {}, "abstract": True},
        "Numeric": {"directParents": ["ValueDomain"], "data": {}},
        "Number": {"directParents": ["Numeric"], "data": {"instantiation": "number"}},
        "Integer": {"directParents": ["Number"], "data": {"instantiation": "integer"}},
        "Box": {"directParents": ["ValueDomain"], "data": {"templateContext": {"order": ["T"], "T": "ValueDomain"}}},
    },
}


class TestAliasEmission:
    """
    Aliases survive into the compiled output as ``using`` declarations and never as data copies (§8 of
    ``documentation/TODO_ALIASES_IMPLEMENTATION.md``) -- which is why the alias containers live on the
    definition the backend reads.
    """

    def test_a_concept_alias_emits_a_using_declaration(self):
        code = ch_compile_from_json(ALIASED_KINGDOM, target="cpp")
        assert "using Beast = Animal;" in code

    def test_a_type_alias_emits_a_using_declaration(self):
        code = ch_compile_from_json(ALIASED_KINGDOM, target="cpp")
        assert "using IntBox = Box<Integer>;" in code

    def test_an_alias_does_not_emit_a_second_struct(self):
        code = ch_compile_from_json(ALIASED_KINGDOM, target="cpp")
        assert "struct Beast" not in code
        assert "struct IntBox" not in code

    def test_the_aliased_concept_is_still_emitted_once(self):
        code = ch_compile_from_json(ALIASED_KINGDOM, target="cpp")
        assert code.count("struct Animal") == 1

    def test_a_using_declaration_follows_the_thing_it_names(self):
        """``using Beast = Animal;`` is only valid C++ after ``Animal`` has been declared."""
        code = ch_compile_from_json(ALIASED_KINGDOM, target="cpp")
        assert code.index("struct Animal") < code.index("using Beast = Animal;")
        assert code.index("struct Box") < code.index("using IntBox = Box<Integer>;")

    def test_a_chain_is_emitted_against_the_canonical_name(self):
        """
        ``Critter`` aliases ``Beast`` aliases ``Animal``. The containers hold the canonical target, so the
        declaration names ``Animal`` -- which also means the emission does not depend on the aliases being
        ordered among themselves.
        """
        code = ch_compile_from_json(ALIASED_KINGDOM, target="cpp")
        assert "using Critter = Animal;" in code
        assert "using Critter = Beast;" not in code

    def test_a_variable_alias_emits_nothing(self):
        """
        §8 says "each container", but ``using`` introduces a *type* name in C++ and this backend emits no
        global variables at all -- so a variable alias has nothing to be an alias *of* yet. Pinned so that
        the omission stays a decision rather than an oversight.
        """
        hierarchy: dict[str, dict[str, int | str | dict]] = {
            **ALIASED_KINGDOM,
            "instances": {"origin": 0, "start": "origin"},
        }
        hierarchy["concepts"] = {
            **hierarchy["concepts"],
            "Integer": {
                "directParents": ["Number"],
                "data": {"defaultSerialization": "integer", "instantiation": "integer"},
            },
        }
        code = ch_compile_from_json(hierarchy, target="cpp")
        assert "start" not in code
        assert "using Beast = Animal;" in code, "the concept aliases are still emitted"
