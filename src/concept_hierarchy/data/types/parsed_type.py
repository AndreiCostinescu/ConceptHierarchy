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

from frozendict import frozendict

VARIADIC_GROUP_IDENTIFIER_CHARACTERS = "!$"


class TemplateArgumentValue(ABC):
    """Fully parsed representation of a template argument value in a type expression."""

    _full_name: str | None
    """
    All types encountered while parsing this template argument value (including sub-types and sub-function-types), 
    keyed by ``full_name``, mapped to ``(clean_name, template_args, function_args)``. 
    Variadic groups are registered under their tuple key.
    """
    _registry: frozendict | None = None

    def __init__(self):
        # use object.__setattr__() to bypass the frozen restriction from subclasses!
        object.__setattr__(self, "_full_name", None)
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
    def registry(self) -> frozendict:
        pass


@dataclass(frozen=True)
class TemplateArgumentWithVariadicId(TemplateArgumentValue, ABC):
    variadic_group_identifier: str | None
    """
    Variadic prefix characters, e.g. ``""``, ``"!"``, ``"$"``, ``"!$"``.
    Can only be used in template arguments, not in function arguments, or variadic group elements.
    """

    def __post_init__(self):
        """
        Dataclass-specific method (interpreted by dataclass) for doing stuff
        after the dataclass __init__ method is called.
        """
        super().__init__()  # Initialize parent cache fields

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

    @property
    def full_name(self) -> str:
        """Re-encode canonically with json.dumps so that embedded quotes are properly escaped in full_name."""
        if self._full_name is None:
            if self.literal_type == "string":
                val = json.dumps(self.literal_value)
            else:
                val = self.literal_value
            if self.has_variadic_identifier:
                val = self.variadic_group_identifier + val
            object.__setattr__(self, "_full_name", val)
        return self._full_name

    @property
    def clean_name(self) -> str:
        return self.literal_value

    @property
    def registry(self) -> frozendict:
        if self._registry is None:
            object.__setattr__(self, "_registry", frozendict({self.full_name: (self.clean_name, None, None)}))
        return self._registry


@dataclass(frozen=True)
class TemplateArgumentVariadicGroup(TemplateArgumentValue):
    variadic_group: tuple[ParsedType | TemplateArgumentLiteral, ...]

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
    def registry(self) -> frozendict:
        if self._registry is None:
            merged: dict = {}
            for entry in self.variadic_group:
                merged.update(entry.registry)
            object.__setattr__(self, "_registry", frozendict(merged))
        return self._registry


class ParsedTypeCache:
    _t_args: tuple[str | tuple[str, ...], ...] | None = None
    _f_args: tuple[str, ...] | None = None

    def __init__(self):
        object.__setattr__(self, "_t_args", None)
        object.__setattr__(self, "_f_args", None)


@dataclass(frozen=True)
class ParsedType(TemplateArgumentWithVariadicId, ParsedTypeCache):
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

    template_arguments: tuple[TemplateArgumentValue, ...] | None
    """
    Parsed template argument value. 
    ``None`` when no ``<…>`` was written; ``()`` when ``<>`` was written but empty.
    This must precede function arguments! And can not be used when `has_variadic_template_expansion` is True.
    """

    function_arguments: tuple[ParsedType, ...] | None
    """Parsed function arguments (same order as ``func_args``)."""

    def __post_init__(self):
        super().__post_init__()
        ParsedTypeCache.__init__(self)

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
            if self.is_templated:
                t_args_str = (
                    "<"
                    + ", ".join((x if isinstance(x, str) else ("[" + ", ".join(x) + "]")) for x in self.template_args)
                    + ">"
                )
            if self.has_arguments:
                f_args_str = "(" + ", ".join(self.func_args) + ")"
            val = self.name + ("..." if self.has_variadic_template_expansion else "") + t_args_str + f_args_str
            if self.has_variadic_identifier:
                val = self.variadic_group_identifier + val
            object.__setattr__(self, "_full_name", val)
        return self._full_name

    @property
    def clean_name(self) -> str:
        return self.name

    @property
    def registry(self) -> frozendict:
        if self._registry is None:
            registry: dict = {}
            for sub in self.template_arguments or ():
                registry.update(sub.registry)
            for sub in self.function_arguments or ():
                registry.update(sub.registry)
            registry[self.full_name] = (self.clean_name, self.template_args, self.func_args)
            object.__setattr__(self, "_registry", frozendict(registry))
        return self._registry

    @property
    def is_templated(self) -> bool:
        return self.template_arguments is not None

    @property
    def has_arguments(self) -> bool:
        return self.function_arguments is not None

    def get_t_values(self) -> tuple[TemplateArgumentValue, ...]:
        if not self.is_templated:
            raise RuntimeError(f"There are no template values for {self.full_name}")
        assert self.template_arguments is not None
        return self.template_arguments

    def get_f_args(self) -> tuple[ParsedType, ...]:
        if self.function_arguments is None:
            raise RuntimeError(f"There are no function arguments for {self.full_name}")
        assert self.function_arguments is not None
        return self.function_arguments

    @property
    def template_args(self) -> tuple[str | tuple[str, ...], ...] | None:
        if self.template_arguments is not None and self._t_args is None:
            res = []
            for arg in self.template_arguments:
                if isinstance(arg, TemplateArgumentVariadicGroup):
                    res.append(tuple([x.full_name for x in arg.variadic_group]))
                else:
                    res.append(arg.full_name)
            object.__setattr__(self, "_t_args", tuple(res))
        return self._t_args

    @property
    def func_args(self) -> tuple[str, ...] | None:
        if self.function_arguments is not None and self._f_args is None:
            object.__setattr__(self, "_f_args", tuple([x.full_name for x in self.function_arguments]))
        return self._f_args
