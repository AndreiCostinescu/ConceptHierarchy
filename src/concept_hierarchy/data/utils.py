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
Internal helpers shared by ``jsonschema_parser`` and ``value_instantiation_parser``.

Not part of the public API.
"""

from __future__ import annotations

from concept_hierarchy.errors import ConceptHierarchyError


class StopValidation(Exception):
    """Raised internally to unwind the recursive validators as soon as the
    first error is found, when ``collect_all_errors=False`` (fail-fast mode)."""


def record(errors: list[ConceptHierarchyError], collect_all_errors: bool, err: ConceptHierarchyError) -> None:
    """
    Append ``err`` to ``errors`` and, in fail-fast mode, immediately stop validation by raising :class:`StopValidation`.
    """
    errors.append(err)
    if not collect_all_errors:
        raise StopValidation()


UNINITIALIZED = object()


class _Missing:
    """Sentinel for "no value present" / "no default specified", distinguishable from a legitimate JSON ``null``."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<MISSING>"

    def __bool__(self) -> bool:  # pragma: no cover - defensive
        return False


MISSING = _Missing()


def lazy_properties(cls):
    # All annotated class vars without a class-level default are lazy.
    # (hasattr returns True for defaults, class methods, etc.)
    field_names = [name for name in cls.__annotations__ if not hasattr(cls, name)]

    original_init = cls.__init__

    for name in field_names:
        private_name = f"_{name}"

        def make_property(_private_name, pub_name):
            def getter(self):
                val = getattr(self, _private_name, UNINITIALIZED)
                if val is UNINITIALIZED:
                    raise RuntimeError(f"{pub_name} not initialized")
                return val

            def setter(self, value):
                setattr(self, _private_name, value)

            return property(getter, setter)

        def make_is_initialized(_private_name):
            def is_initialized(self) -> bool:
                return getattr(self, _private_name, UNINITIALIZED) is not UNINITIALIZED

            return is_initialized

        setattr(cls, name, make_property(private_name, name))
        setattr(cls, f"is_{name}_initialized", make_is_initialized(private_name))

    def new_init(self, *args, **kwargs):
        for field_name in field_names:
            setattr(self, f"_{field_name}", UNINITIALIZED)
        original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    return cls
