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
generator.py — Dispatch code generation to the appropriate backend.
"""

from __future__ import annotations

from concept_hierarchy.errors import CodegenError
from concept_hierarchy.models import ConceptHierarchyModel

_BACKENDS = {
    "cpp": "concept_hierarchy.backends.cpp:CppBackend",
}


def generate(model: ConceptHierarchyModel, target: str = "cpp") -> str:
    """Generate source code for *model* in *target* language.

    Parameters
    ----------
    model:
        A validated :class:`~concept_hierarchy.models.ConceptHierarchyModel`.
    target:
        One of the supported backends (currently ``"cpp"``).

    Returns
    -------
    str
        Complete generated source as a single string.

    Raises
    ------
    ValueError
        For unsupported *target* values.
    concept_hierarchy.errors.CodegenError
        When generation fails for a supported target.
    """
    if target not in _BACKENDS:
        supported = ", ".join(sorted(_BACKENDS))
        raise ValueError(
            f"Unsupported target {target!r}. Supported backends: {supported}."
        )

    module_path, class_name = _BACKENDS[target].rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    backend_cls = getattr(module, class_name)
    backend = backend_cls()

    try:
        return backend.generate(model)
    except Exception as exc:
        raise CodegenError(f"Code generation failed for target {target!r}: {exc}") from exc
