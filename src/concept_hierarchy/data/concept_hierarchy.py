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
from concept_hierarchy.data.expressions.expression import Expression
from concept_hierarchy.data.expressions.expression_utils import (
    FunctionArgumentAccessor,
    FunctionArgumentProvenance,
    FunctionResultAccessor,
    ValueDomainArgumentProvenance,
)
from concept_hierarchy.data.jsonschema import CHSchemaNode
from concept_hierarchy.data.type_template_variables.constraint_formula import ConstraintGroup
from concept_hierarchy.data.types.concept_hierarchy_types import (
    ConceptHierarchyTemplateArgument,
    InstantiatedType,
    TypeValue,
)
from concept_hierarchy.data.utils import UNINITIALIZED
from concept_hierarchy.models import ConceptHierarchyModel

from .utils import lazy_properties


class ConceptHierarchyData:
    def __init__(self, name: str):
        self.name = name


@lazy_properties
class ConceptData(ConceptHierarchyData):
    instantiable: bool
    """
    An abstract ValueDomain can not be instantiated. 
    An abstract Function can not be instantiated and does not have to define its interface (but it can)!
    """

    def __init__(self, name: str, parents: frozendict[str, ConceptData]):
        super().__init__(name)
        self.parents = parents


@lazy_properties
class DomainConceptData(ConceptData):
    # all the data is (available) for this concept; this class does not store any data for subconcepts
    property_types: frozendict[str, InstantiatedType]
    property_constraints: frozendict[str, Expression]
    function_types: frozendict[str, InstantiatedType]

    def __init__(self, name: str, parents: frozendict[str, ConceptData]):
        super().__init__(name, parents)


@lazy_properties
class TypeData(ConceptData):
    template_context: TemplateContext
    parent_template_variable_substitution: frozendict[tuple[str, str], ConceptHierarchyTemplateArgument]
    """Contains all substitution values of the template arguments of all parents (also non direct parents!)"""

    def __init__(self, name: str, parents: frozendict[str, ConceptData]):
        super().__init__(name, parents)


@lazy_properties
class ValueDomainData(TypeData):
    instantiation: tuple[tuple[ConstraintGroup | None, CHSchemaNode], ...]
    """`None` appears as the first element of the instantiation when the type does not have template arguments"""

    def __init__(self, name: str, parents: frozendict[str, ConceptData]):
        super().__init__(name, parents)


@lazy_properties
class FunctionData(ValueDomainData):
    evaluation_interface: tuple[str, ...]
    evaluation_argument_types: frozendict[str, TypeValue]
    evaluation_argument_access_type: frozendict[str, FunctionArgumentAccessor]
    evaluation_argument_provenance_type: frozendict[str, FunctionArgumentProvenance]
    evaluation_argument_default_value: frozendict[str, Expression]
    evaluation_result_type: TypeValue | None
    evaluation_result_access_type: FunctionResultAccessor | None
    evaluation_result_provenance_type: ValueDomainArgumentProvenance | None

    procedure: Expression

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
    is_alias: bool
    value_type: InstantiatedType
    value: Expression

    def __init__(self, name: str, is_alias: bool):
        super().__init__(name)
        self.is_alias = is_alias


class ConceptHierarchy:
    def __init__(self, ch_def: ConceptHierarchyModel):
        self.ch = ch_def
        self.concepts: dict[str, ConceptData] = {}
        self.instances: dict[str, GlobalVariableData] = {}
        self.domain_concepts: dict[str, DomainConceptData] = {}
        self.value_domains: dict[str, ValueDomainData] = {}  # this includes functions and their FunctionData
        self.functions: dict[str, FunctionData] = {}
