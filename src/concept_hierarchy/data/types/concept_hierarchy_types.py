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

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TypeAlias

from frozendict import frozendict

TypeComposition: TypeAlias = tuple[tuple[str | None, str | None, tuple], ...]


class ConceptHierarchyTemplateArgument(ABC):
    def __init__(self, clean_name: str, **kwargs):
        self.clean_name = clean_name
        self.used_templates: set[str] = set()
        """Keeps track of the used template variables in this value only."""

        self._full_name: str | None = None
        self._type_composition: TypeComposition | None = None
        """
        tuple of multiple (full_name, clean_name, template_argument_sub_type_composition)
        """
        self._registry: frozendict | None = None
        """
        All types encountered while parsing this template argument value (including sub-types and sub-function-types), 
        keyed by ``full_name``, mapped to ``(clean_name, template_args, function_args)``. 
        Variadic groups are registered under their tuple key.
        """

    def __str__(self) -> str:
        return self.full_name

    def __repr__(self) -> str:
        return self.full_name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConceptHierarchyTemplateArgument):
            return False
        return type(self) is type(other) and self.full_name == other.full_name

    @property
    def depends_on_templates(self) -> bool:
        return len(self.used_templates) > 0

    @property
    @abstractmethod
    def full_name(self) -> str:
        pass

    @property
    @abstractmethod
    def type_composition(self) -> TypeComposition:
        pass

    @property
    @abstractmethod
    def registry(self) -> frozendict:
        pass


class Instantiated(ConceptHierarchyTemplateArgument, ABC):
    def __init__(self, clean_name: str, **kwargs):
        super().__init__(clean_name=clean_name, **kwargs)


class TemplateDependent(ConceptHierarchyTemplateArgument, ABC):
    def __init__(self, clean_name: str, **kwargs):
        super().__init__(clean_name=clean_name, **kwargs)


class VariadicArgument(ConceptHierarchyTemplateArgument, ABC):
    pass


class ConceptHierarchyType(ConceptHierarchyTemplateArgument, ABC):
    def __init__(self, clean_name: str, template_arguments: tuple[ConceptHierarchyTemplateArgument, ...], **kwargs):
        super().__init__(clean_name=clean_name, **kwargs)
        # Because this is in canonical form (i.e. only variadic groups allowed; no variadic identifiers),
        #  a () value for template_arguments means that the type has no template arguments!
        self.template_arguments: tuple[ConceptHierarchyTemplateArgument, ...] = template_arguments
        for t_arg in self.template_arguments:
            self.used_templates |= t_arg.used_templates

    @property
    def is_templated(self) -> bool:
        return self.template_arguments != ()

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
                t_args_str = "<" + ", ".join(x.full_name for x in self.template_arguments) + ">"
            """
            if self.has_arguments:
                f_args_str = "(" + ", ".join(self.func_args) + ")"
            """
            self._full_name = self.clean_name + t_args_str + f_args_str
        return self._full_name

    @property
    def type_composition(self) -> TypeComposition:
        """Construct the ``type_composition`` tree."""
        if self._type_composition is None:
            sub_comp: tuple = ()
            for sub in self.template_arguments:
                sub_comp += sub.type_composition
            object.__setattr__(self, "_type_composition", ((self.full_name, self.clean_name, sub_comp),))
        return self._type_composition

    @property
    def registry(self) -> frozendict:
        if self._registry is None:
            registry: dict = {}
            for sub in self.template_arguments or ():
                registry.update(sub.registry)
            """
            for sub in self.function_arguments or ():
                registry.update(sub.registry)
            registry[self.full_name] = (self.clean_name, self.template_args, self.func_args)
            """
            registry[self.full_name] = (self.clean_name, self.template_arguments, ())
            self._registry = frozendict(registry)
        return self._registry


class InstantiatedType(Instantiated, ConceptHierarchyType):
    def __init__(self, clean_name: str, template_arguments: tuple[ConceptHierarchyTemplateArgument, ...], **kwargs):
        super().__init__(clean_name=clean_name, template_arguments=template_arguments, **kwargs)
        if self.depends_on_templates:
            raise RuntimeError(
                f"Tried to create an Instantiated template argument value {self!r} that depends on templates "
                f"{self.used_templates}!"
            )


class TemplateDependentType(TemplateDependent, ConceptHierarchyType):
    def __init__(self, clean_name: str, template_arguments: tuple[ConceptHierarchyTemplateArgument, ...], **kwargs):
        super().__init__(clean_name=clean_name, template_arguments=template_arguments, **kwargs)
        if not self.depends_on_templates:
            raise RuntimeError(
                f"Tried to create a TemplateDependent template argument value {self!r} that does not depend on "
                f"templates!"
            )


class TemplateVariable(TemplateDependent, ABC):
    def __init__(self, clean_name: str, unique_defining_location: str, **kwargs):
        super().__init__(clean_name=clean_name, **kwargs)
        self.unique_defining_location = unique_defining_location
        """
        The unique defining location is to prevent variable capture in substitution procedures. See: 
        https://opencs.aalto.fi/en/courses/programming-languages/part-5/3-alpha-equivalence-and-capture-avoiding-subst
        """
        assert self.unique_defining_location is not None
        self.used_templates.add(self.clean_name)

    @property
    def full_name(self) -> str:
        if self._full_name is None:
            self._full_name = self.unique_defining_location + ":" + self.clean_name
        return self._full_name

    @property
    def type_composition(self) -> TypeComposition:
        if self._type_composition is None:
            self._type_composition = ((self.full_name, self.clean_name, ()),)
        return self._type_composition

    @property
    def registry(self) -> frozendict:
        if self._registry is None:
            self._registry = frozendict({self.full_name: (self.clean_name, None, None)})
        return self._registry

    @property
    @abstractmethod
    def is_variadic(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_variadic_expanded(self) -> bool:
        pass


class NonVariadicTemplateVariable(TemplateVariable):
    @property
    def is_variadic(self) -> bool:
        return False

    @property
    def is_variadic_expanded(self) -> bool:
        return False


class VariadicTemplateVariable(TemplateVariable, VariadicArgument):
    @property
    def is_variadic(self) -> bool:
        return True

    @property
    def is_variadic_expanded(self) -> bool:
        return False


class ExpandedVariadicTemplateVariable(VariadicTemplateVariable):
    @property
    def full_name(self) -> str:
        if self._full_name is None:
            self._full_name = self.clean_name + "..."
        return self._full_name

    @property
    def is_variadic(self) -> bool:
        return True

    @property
    def is_variadic_expanded(self) -> bool:
        return True


class LiteralValue(Instantiated):
    def __init__(self, clean_name: str, literal_type: str, **kwargs):
        super().__init__(clean_name=clean_name, **kwargs)
        self.literal_type = literal_type

    @property
    def full_name(self) -> str:
        if self._full_name is None:
            if self.literal_type == "string":
                self._full_name = json.dumps(self.clean_name)
            else:
                self._full_name = self.clean_name
        return self._full_name

    @property
    def type_composition(self) -> TypeComposition:
        if self._type_composition is None:
            self._type_composition = ((self.full_name, self.clean_name, ()),)
        return self._type_composition

    @property
    def registry(self) -> frozendict:
        if self._registry is None:
            self._registry = frozendict({self.full_name: (self.clean_name, None, None)})
        return self._registry


class ConceptHierarchyVariadicGroup(VariadicArgument, ABC):
    def __init__(
        self,
        clean_name: str,
        variadic_group: tuple[ConceptHierarchyType | LiteralValue | TemplateVariable, ...],
        **kwargs,
    ):
        super().__init__(clean_name=clean_name, **kwargs)
        self.variadic_group = variadic_group
        for group_element in variadic_group:
            self.used_templates |= group_element.used_templates

    @property
    def full_name(self) -> str:
        if self._full_name is None:
            self._full_name = "[" + ", ".join(x.full_name for x in self.variadic_group) + "]"
        return self._full_name

    @property
    def type_composition(self) -> TypeComposition:
        if self._type_composition:
            sub_comp: tuple = ()
            for sub in self.variadic_group:
                sub_comp += sub.type_composition
            self._type_composition = ((None, None, sub_comp),)
        return self._type_composition

    @property
    def registry(self) -> frozendict:
        if self._registry is None:
            merged: dict = {}
            for entry in self.variadic_group:
                merged.update(entry.registry)
            self._registry = frozendict(merged)
        return self._registry


class InstantiatedVariadicGroup(Instantiated, ConceptHierarchyVariadicGroup):
    def __init__(
        self,
        clean_name: str,
        variadic_group: tuple[ConceptHierarchyType | LiteralValue | TemplateVariable, ...],
        **kwargs,
    ):
        super().__init__(clean_name=clean_name, variadic_group=variadic_group, **kwargs)
        if self.depends_on_templates:
            raise RuntimeError(
                f"Tried to create an Instantiated template argument value {self!r} that depends on templates "
                f"{self.used_templates}!"
            )


class TemplateDependentVariadicGroup(TemplateDependent, ConceptHierarchyVariadicGroup):
    def __init__(
        self,
        clean_name: str,
        variadic_group: tuple[ConceptHierarchyType | LiteralValue | TemplateVariable, ...],
        **kwargs,
    ):
        super().__init__(clean_name=clean_name, variadic_group=variadic_group, **kwargs)
        if not self.depends_on_templates:
            raise RuntimeError(
                f"Tried to create a TemplateDependent template argument value {self!r} that does not depend on "
                f"templates!"
            )


TypeValue: TypeAlias = InstantiatedType | TemplateDependentType | NonVariadicTemplateVariable | VariadicTemplateVariable
