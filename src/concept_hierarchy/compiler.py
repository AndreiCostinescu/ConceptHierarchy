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
1. Parse   : load & parse JSON into an internal AST/model
2. Validate: syntax + semantic checks
3. Codegen : generate target-language source from the validated model
"""

from __future__ import annotations

import json
import os
from typing import Optional

from concept_hierarchy.parser.parser import parse
from concept_hierarchy.validator.syntax import check_syntax
from concept_hierarchy.validator.semantics import check_semantics
from concept_hierarchy.codegen.generator import generate


def compile_hierarchy(
    source: str,
    target: str = "cpp",
    output_path: Optional[str] = None,
) -> str:
    """Compile a ConceptHierarchy JSON definition to a target language.

    Parameters
    ----------
    source:
        Path to a JSON file *or* a raw JSON string containing the hierarchy
        definition.
    target:
        Target language for code generation.  Currently only ``"cpp"`` is
        supported.
    output_path:
        If given, the generated source is written to this file path in
        addition to being returned.

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
    # ------------------------------------------------------------------
    # 1. Load raw JSON
    # ------------------------------------------------------------------
    if os.path.isfile(source):
        with open(source, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    else:
        raw = json.loads(source)

    # ------------------------------------------------------------------
    # 2. Parse into internal model
    # ------------------------------------------------------------------
    model = parse(raw)

    # ------------------------------------------------------------------
    # 3. Validate
    # ------------------------------------------------------------------
    check_syntax(model)
    check_semantics(model)

    # ------------------------------------------------------------------
    # 4. Code generation
    # ------------------------------------------------------------------
    code = generate(model, target=target)

    # ------------------------------------------------------------------
    # 5. Optional file output
    # ------------------------------------------------------------------
    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(code)

    return code
