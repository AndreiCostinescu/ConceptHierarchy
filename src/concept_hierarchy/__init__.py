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
concept_hierarchy — Compiler for the ConceptHierarchy programming language.

Takes a JSON definition of a ConceptHierarchy, validates its syntax and
semantics, and generates an implementation in a target language (currently C++).

Typical usage
-------------
    from concept_hierarchy import ch_compile

    result = ch_compile("my_hierarchy.json", target="cpp")
"""

from concept_hierarchy.compiler import ch_check, ch_check_from_json, ch_compile, ch_compile_from_json  # noqa: F401

__version__ = "0.1.0"
__all__ = ["ch_check", "ch_check_from_json", "ch_compile", "ch_compile_from_json", "__version__"]
