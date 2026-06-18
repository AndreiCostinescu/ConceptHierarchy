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

from frozendict import frozendict

from concept_hierarchy.data.contexts.template_context import TemplateContext
from concept_hierarchy.models import ConceptHierarchyModel

from .types.parsed_type import ParsedType, TemplateArgumentValue
from .utils import lazy_properties


class ConceptHierarchyData:
    def __init__(self, name: str):
        self.name = name


class ConceptData(ConceptHierarchyData):
    def __init__(self, name: str, parents: frozendict[str, ConceptData]):
        super().__init__(name)
        self.parents = parents


@lazy_properties
class DomainConceptData(ConceptData):
    # all the data is (available) for this concept; this class does not store any data for subconcepts
    property_types: frozendict[str, ParsedType]
    property_constraints: frozendict[str, object]  # replace object with Expression
    function_types: frozendict[str, ParsedType]

    def __init__(self, name: str, parents: frozendict[str, ConceptData]):
        super().__init__(name, parents)


@lazy_properties
class TypeData(ConceptData):
    template_context: TemplateContext
    parent_template_variable_substitution: frozendict[tuple[str, str], TemplateArgumentValue]
    instantiable: bool

    def __init__(self, name: str, parents: frozendict[str, ConceptData]):
        super().__init__(name, parents)


class ValueDomainData(TypeData):
    def __init__(self, name: str, parents: frozendict[str, ConceptData]):
        super().__init__(name, parents)


class FunctionData(ValueDomainData):
    def __init__(self, name: str, parents: frozendict[str, ConceptData]):
        super().__init__(name, parents)


@lazy_properties
class GlobalVariableData(ConceptHierarchyData):
    value_type: ParsedType
    value: object  # replace object with Expression

    def __init__(self, name: str):
        super().__init__(name)


class ConceptHierarchy:
    def __init__(self, ch_def: ConceptHierarchyModel):
        self.ch = ch_def
        self.concepts: dict[str, ConceptData] = {}
        self.instances: dict[str, GlobalVariableData] = {}
        self.domain_concepts: dict[str, DomainConceptData] = {}
        self.value_domains: dict[str, ValueDomainData] = {}  # this includes functions and their FunctionData
        self.functions: dict[str, FunctionData] = {}
