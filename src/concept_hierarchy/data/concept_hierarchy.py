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

from enum import Enum

from frozendict import frozendict

from concept_hierarchy.data.contexts.template_context import TemplateContext
from concept_hierarchy.data.types.concept_hierarchy_types import InstantiatedType, TypeValue
from concept_hierarchy.data.utils import UNINITIALIZED
from concept_hierarchy.models import ConceptHierarchyModel

from .types.concept_hierarchy_types import ConceptHierarchyTemplateArgument
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
    property_types: frozendict[str, InstantiatedType]
    property_constraints: frozendict[str, object]  # replace object with Expression
    function_types: frozendict[str, InstantiatedType]

    def __init__(self, name: str, parents: frozendict[str, ConceptData]):
        super().__init__(name, parents)


@lazy_properties
class TypeData(ConceptData):
    template_context: TemplateContext
    parent_template_variable_substitution: frozendict[tuple[str, str], ConceptHierarchyTemplateArgument]
    """Contains all substitution values of the template arguments of all parents (also non direct parents!)"""
    instantiable: bool
    """
    An abstract ValueDomain can not be instantiated. 
    An abstract Function can not be instantiated and does not have to define its interface (but it can)!
    """

    def __init__(self, name: str, parents: frozendict[str, ConceptData]):
        super().__init__(name, parents)


class ValueDomainArgumentReference(Enum):
    NO_REF = "NoRef"
    REF = "Reference"


class ValueDomainData(TypeData):
    def __init__(self, name: str, parents: frozendict[str, ConceptData]):
        super().__init__(name, parents)


class FunctionArgumentReference(Enum):
    NO_REF = "NoRef"
    REF = "Reference"
    EMPTY_REF = "EmptyReference"


class FunctionResultModifier(Enum):
    GET = "Get"
    MOD = "Modify"


class FunctionArgumentModifier(Enum):
    GET = "Get"
    MOD = "Modify"
    GET_MOD = "GetModify"


@lazy_properties
class FunctionData(ValueDomainData):
    evaluation_interface: tuple[str, ...]
    evaluation_argument_types: frozendict[str, TypeValue]
    evaluation_argument_modifier_type: frozendict[str, FunctionArgumentModifier]
    evaluation_argument_reference_type: frozendict[str, FunctionArgumentReference]
    evaluation_argument_default_value: frozendict[str, object]  # replace object with Expression
    evaluation_result_type: TypeValue | None | object
    evaluation_result_modifier_type: FunctionResultModifier | None
    evaluation_result_reference_type: ValueDomainArgumentReference | None

    procedure: object  # replace object with expression

    sub_scope_vars: frozendict[str, frozendict[str, TypeValue]]
    """Maps evaluation argument names to new variables available in their scope and their type."""
    new_vars_in_scope: frozendict[str, TypeValue]
    """Maps the new variables introduced after the evaluation of this Function to their type."""

    def __init__(self, name: str, parents: frozendict[str, ConceptData]):
        super().__init__(name, parents)

    @property
    def knows_what_it_returns(self):
        return self.evaluation_result_type is not UNINITIALIZED

    @property
    def returns_something(self):
        assert self.knows_what_it_returns
        return self.evaluation_result_type is not None


@lazy_properties
class GlobalVariableData(ConceptHierarchyData):
    value_type: InstantiatedType
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
