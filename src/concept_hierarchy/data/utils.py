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
Internal helpers shared by ``jsonschema_parser`` and ``value_instantiation_validator``.

Not part of the public API.
"""

from __future__ import annotations

from concept_hierarchy.errors import ConceptHierarchyError


class StopValidation(Exception):
    """Raised internally to unwind the recursive validators as soon as the
    first error is found, when ``collect_all=False`` (fail-fast mode)."""


def record(errors: list[ConceptHierarchyError], collect_all: bool, err: ConceptHierarchyError) -> None:
    """
    Append ``err`` to ``errors`` and, in fail-fast mode, immediately stop validation by raising :class:`StopValidation`.
    """
    errors.append(err)
    if not collect_all:
        raise StopValidation()
