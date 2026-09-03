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
compiler.py — Top-level orchestrator for the ConceptHierarchy compiler pipeline.

Pipeline
--------
1. Load    : load Concept Hierarchy json content into an internal model
2. Validate: syntax + semantic checks
3. Codegen : generate target-language source from the validated model
"""

import os

from concept_hierarchy.codegen.generator import generate
from concept_hierarchy.definitions.concept_hierarchy import ConceptHierarchyDefinition
from concept_hierarchy.errors import ConceptHierarchyError
from concept_hierarchy.validator.checker import check_model


def compile_impl(
    ch: ConceptHierarchyDefinition,
    target: str = "cpp",
    output_path: str | None = None,
) -> str:
    # Read file source, validate and interpret data!
    check_model(ch)

    # Code generation
    code = generate(ch, target=target)

    # Optional file output
    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(code)

    return code


def ch_compile(
    source: str,
    target: str = "cpp",
    output_path: str | None = None,
) -> str:
    """Compile a ConceptHierarchy JSON definition to a target language.

    Parameters
    ----------
    source:
        Path to a JSON file *or* a raw JSON string containing the hierarchy definition.
    target:
        Target language for code generation.  Currently only ``"cpp"`` is supported.
    output_path:
        If given, the generated source is written to this file path in addition to being returned.

    Returns
    -------
    str
        The generated source code as a string.

    Raises
    ------
    concept_hierarchy.errors.SyntaxError
        When the JSON definition violates the ConceptHierarchy syntax rules.
    concept_hierarchy.errors.SemanticError
        When the definition is syntactically valid but semantically incorrect
        (e.g. undefined references, circular dependencies).
    ValueError
        When an unsupported *target* language is requested.
    """
    # Create ConceptHierarchy definition container from file source
    model = ConceptHierarchyDefinition.create_by_parser(source)
    return compile_impl(model, target, output_path)


def ch_compile_from_json(
    data: object,
    target: str = "cpp",
    output_path: str | None = None,
) -> str:
    """Compile a ConceptHierarchy JSON definition to a target language.

    Parameters
    ----------
    data:
        The JSON content containing the hierarchy definition.
    target:
        Target language for code generation.  Currently only ``"cpp"`` is supported.
    output_path:
        If given, the generated source is written to this file path in addition to being returned.

    Returns
    -------
    str
        The generated source code as a string.

    Raises
    ------
    concept_hierarchy.errors.SyntaxError
        When the JSON definition violates the ConceptHierarchy syntax rules.
    concept_hierarchy.errors.SemanticError
        When the definition is syntactically valid but semantically incorrect
        (e.g. undefined references, circular dependencies).
    ValueError
        When an unsupported *target* language is requested.
    """
    # Create ConceptHierarchy definition container from json-deserialized data
    model = ConceptHierarchyDefinition.create_from_data(data)
    return compile_impl(model, target, output_path)


def check_impl(ch: ConceptHierarchyDefinition, *, raise_errors: bool = False) -> bool:
    if raise_errors:
        check_model(ch)  # To let the errors pass through
    else:
        try:
            check_model(ch)  # parse definition and validate it
        except ConceptHierarchyError as e:
            print("There was an error in validating the Concept Hierarchy definition:\n", e.print(1), sep="")
            return False
    print("The Concept Hierarchy definition is valid!")
    return True


def ch_check(source: str) -> bool:
    """Check that a ConceptHierarchy JSON definition is syntactically and semantically valid.

    Parameters
    ----------
    source:
        Path to a JSON file *or* a raw JSON string containing the hierarchy definition.

    Returns
    -------
    bool
        Whether the Concept Hierarchy definition is valid.

    Raises
    ------
    concept_hierarchy.errors.SyntaxError
        When the JSON definition violates the ConceptHierarchy syntax rules.
    concept_hierarchy.errors.SemanticError
        When the definition is syntactically valid but semantically incorrect
        (e.g. undefined references, circular dependencies).
    RuntimeError
        Upon logical coding errors of the compiler itself...
    """
    source = os.path.abspath(source)
    ch = ConceptHierarchyDefinition(source, os.path.dirname(source))
    return check_impl(ch)


def ch_check_from_json(data: object) -> bool:
    """Check that a ConceptHierarchy JSON definition is syntactically and semantically valid.

    Parameters
    ----------
    data:
        The JSON content containing the hierarchy definition.

    Returns
    -------
    bool
        Whether the Concept Hierarchy definition is valid.

    Raises
    ------
    concept_hierarchy.errors.SyntaxError
        When the JSON definition violates the ConceptHierarchy syntax rules.
    concept_hierarchy.errors.SemanticError
        When the definition is syntactically valid but semantically incorrect
        (e.g. undefined references, circular dependencies).
    RuntimeError
        Upon logical coding errors of the compiler itself...
    """
    ch = ConceptHierarchyDefinition.create_from_data(data)
    return check_impl(ch)
