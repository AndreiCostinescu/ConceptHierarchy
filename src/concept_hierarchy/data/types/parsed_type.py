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
parsed_type.py — Contains the classes to store all the available data after parsing types.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeAlias

from frozendict import frozendict

VARIADIC_GROUP_IDENTIFIER_CHARACTERS = "!$"


TypeComposition: TypeAlias = tuple[tuple[str | None, str | None, tuple], ...]


class TemplateArgumentValue(ABC):
    """Fully parsed representation of a template argument value in a type expression."""

    _full_name: str | None
    _type_composition: TypeComposition | None = None
    """
    All types encountered while parsing this template argument value (including sub-types and sub-function-types), 
    keyed by ``full_name``, mapped to ``(clean_name, template_args, function_args)``. 
    Variadic groups are registered under their tuple key.
    """
    _registry: frozendict | None

    def __init__(self):
        # use object.__setattr__() to bypass the frozen restriction from subclasses!
        object.__setattr__(self, "_full_name", None)
        object.__setattr__(self, "_type_composition", None)
        object.__setattr__(self, "_registry", None)

    @property
    @abstractmethod
    def full_name(self) -> str:
        pass

    @property
    @abstractmethod
    def clean_name(self) -> str:
        pass

    @property
    @abstractmethod
    def type_composition(self) -> TypeComposition:
        pass

    @property
    @abstractmethod
    def registry(self) -> frozendict:
        pass


@dataclass(frozen=True)
class TemplateArgumentWithVariadicId(TemplateArgumentValue, ABC):
    variadic_group_identifier: str | None
    """
    Variadic prefix characters, e.g. ``""``, ``"!"``, ``"$"``, ``"!$"``.
    Can only be used in template arguments, not in function arguments, or variadic group elements.
    """

    @property
    def has_variadic_identifier(self) -> bool:
        return self.variadic_group_identifier is not None


@dataclass(frozen=True)
class TemplateArgumentLiteral(TemplateArgumentWithVariadicId):
    literal_value: str

    literal_type: str
    """
    The kind of literal: ``"bool"``, ``"int"``, ``"float"``, or ``"string"``.
    ``None`` for named types and variadic groups.

    For string literals ``full_name`` carries the canonical JSON-encoded form including surrounding 
    double-quote characters (e.g. ``'"say \\"hi\\""'``), so string literals are unambiguously 
    distinguishable from a named type in the registry. 
    The ``clean_name`` member differs from ``full_name`` only for string literals, for which 
    it contains the **decoded** json-string value
    """

    def __post_init__(self):
        super().__init__()  # Initialize parent cache fields

    @property
    def full_name(self) -> str:
        """Re-encode canonically with json.dumps so that embedded quotes are properly escaped in full_name."""
        if self._full_name is None:
            if self.literal_type == "string":
                object.__setattr__(self, "_full_name", json.dumps(self.literal_value))
            else:
                object.__setattr__(self, "_full_name", self.literal_value)
        return self._full_name

    @property
    def clean_name(self) -> str:
        return self.literal_value

    @property
    def type_composition(self) -> TypeComposition:
        if self._type_composition is None:
            object.__setattr__(self, "type_composition", ((self.full_name, self.clean_name, ()),))
        return self._type_composition

    @property
    def registry(self) -> frozendict:
        if self._registry is None:
            object.__setattr__(self, "_registry", frozendict({self.full_name: (self.clean_name, None, None)}))
        return self._registry


@dataclass(frozen=True)
class TemplateArgumentVariadicGroup(TemplateArgumentValue):
    variadic_group: tuple[TemplateArgumentLiteral | ParsedType, ...]

    def __post_init__(self):
        super().__init__()  # Initialize parent cache fields

    @property
    def full_name(self) -> str:
        if self._full_name is None:
            object.__setattr__(self, "_full_name", f"[{', '.join(x.full_name for x in self.variadic_group)}]")
        return self._full_name

    @property
    def clean_name(self) -> str:
        return ""

    @property
    def type_composition(self) -> TypeComposition:
        if self._type_composition:
            sub_comp: tuple = ()
            for sub in self.variadic_group:
                sub_comp += sub.type_composition
            object.__setattr__(self, "_type_composition", ((None, None, sub_comp),))
        return self._type_composition

    @property
    def registry(self) -> frozendict:
        if self._registry is None:
            merged: dict = {}
            for entry in self.variadic_group:
                merged.update(entry.registry)
            object.__setattr__(self, "_registry", frozendict(merged))
        return self._registry


@dataclass(frozen=True)
class ParsedType(TemplateArgumentWithVariadicId):
    """
    Fully parsed representation of a type in a type expression.

    ``full_name`` is:
    - ``str``   — ordinary named type or the variadic-group-as-string

    The variadic prefix (e.g. ``"!"``, ``"$"``, ``"!$"``) is prepended to the string forms
    but is absent from bracket groups.
    """

    name: str
    """
    The base name of the type without template and function arguments, variadic identifiers, 
    and without the variadic expansion operator,
    """

    has_variadic_template_expansion: bool
    """
    Whether the type ends with the `...` variadic template expansion operator.
    Can not be used when the type has template or function arguments or a variadic group identifier.
    """

    template_args: tuple[str | tuple[str, ...], ...] | None
    """
    Full names of template arguments. 
    ``None`` when no ``<…>`` was written; ``()`` when ``<>`` was written but empty.
    This must precede function arguments! And can not be used when `has_variadic_template_expansion` is True.
    """

    func_args: tuple[str, ...] | None
    """
    Full names of function-call arguments.  
    ``None`` when no ``(…)`` was written; ``()`` when ``()`` was written but empty.
    This must come after template arguments! And can not be used when `has_variadic_template_expansion` is True.
    """

    template_argument_values: tuple[TemplateArgumentValue, ...]
    """Parsed template argument values (same order as ``template_args``)."""

    sub_func_types: tuple[ParsedType, ...]
    """Parsed function arguments (same order as ``func_args``)."""

    def __post_init__(self):
        """
        Dataclass-specific method (interpreted by dataclass) for doing stuff
        after the dataclass __init__ method is called.
        """
        super().__init__()  # Initialize parent cache fields

    @property
    def full_name(self) -> str:
        """
        The type name including:
         - template and function arguments,
         - variadic identifier, and
         - variadic expansion operator.
        """
        if self._full_name is None:
            t_args_str, f_args_str = "", ""
            if self.template_args is not None:
                t_args_str = (
                    "<"
                    + ", ".join((x if isinstance(x, str) else ("[" + ", ".join(x) + "]")) for x in self.template_args)
                    + ">"
                )
            if self.func_args is not None:
                f_args_str = "(" + ", ".join(self.func_args) + ")"
            object.__setattr__(
                self,
                "_full_name",
                self.variadic_group_identifier
                + self.name
                + ("..." if self.has_variadic_template_expansion else "")
                + t_args_str
                + f_args_str,
            )
        return self._full_name

    @property
    def clean_name(self) -> str:
        return self.name

    @property
    def type_composition(self) -> TypeComposition:
        """Construct the ``type_composition`` tree."""
        if self._type_composition is None:
            sub_comp: tuple = ()
            for sub in self.template_argument_values:
                sub_comp += sub.type_composition
            object.__setattr__(self, "_type_composition", ((self.full_name, self.clean_name, sub_comp),))
        return self._type_composition

    @property
    def registry(self) -> frozendict:
        if self._registry is None:
            registry: dict = {}
            for sub in self.template_argument_values:
                registry.update(sub.registry)
            for sub in self.sub_func_types:
                registry.update(sub.registry)
            registry[self.full_name] = (self.clean_name, self.template_args, self.func_args)
            object.__setattr__(self, "_registry", frozendict(registry))
        return self._registry

    @property
    def is_templated(self) -> bool:
        if (self.template_args is None) and (self.template_argument_values != ()):
            raise RuntimeError(f"If this is not a templated type, then sub_types must be empty! Got {self!r}")
        return self.template_args is not None

    @property
    def has_arguments(self) -> bool:
        if (self.func_args is None) and (self.sub_func_types != ()):
            raise RuntimeError(f"If this is not a templated type, then sub_func_types must be empty! Got {self!r}")
        return self.func_args is not None
