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
from concept_hierarchy.data.type_template_variables.constraint_formula import (
    ConstraintGroup,
    NonStructureConstraintFormula,
)
from concept_hierarchy.data.types.concept_hierarchy_types import (
    ConceptHierarchyTemplateArgument,
    InstantiatedType,
    TypeValue,
)
from concept_hierarchy.data.utils import UNINITIALIZED
from concept_hierarchy.definitions.concept_definition_domain_concept import (
    FunctionDefinitionKeywords,
    PropertyDefinitionKeywords,
)
from concept_hierarchy.definitions.concept_hierarchy import ConceptHierarchyDefinition

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

    def __init__(self, name: str, parents: frozendict[str, ConceptData], all_parents: frozendict[str, ConceptData]):
        super().__init__(name)
        self.direct_parents = parents
        self.all_parents = all_parents


@lazy_properties
class DomainConceptData(ConceptData):
    # all the data is (available) for this concept; this class does not store any data for subconcepts
    property_types: frozendict[str, InstantiatedType]
    property_constraints: frozendict[str, Expression]
    static_properties: frozenset[str]
    function_types: frozendict[str, InstantiatedType]
    static_functions: frozenset[str]
    function_expressions: frozendict[str, frozendict[str, Expression]]
    """
    Format: { func_name: { forThis/forSub: Expression } }. 
    Only contains a Function's default value as expression data.
    """

    management: frozendict[str, Expression]
    """The registered management functions of this concept alone."""

    available_property_data: frozendict[str, frozendict[str, str]]
    """
    Every property available to this concept -- defined here or inherited -- mapped to, per definition
    keyword, the concept that provides that keyword's value.

    This is the *derived* view. The same-named field on ``DomainConceptDefinition`` is the view of what the
    concept's own JSON declares and specializes, and does not include anything inherited.
    """
    available_function_data: frozendict[str, frozendict[str, str]]
    """As :attr:`available_property_data`, for the concept's ``functions`` members."""

    def __init__(self, name: str, parents: frozendict[str, ConceptData], all_parents: frozendict[str, ConceptData]):
        super().__init__(name, parents, all_parents)
        self._all_available_property_types = None
        self._all_available_function_types = None

    @property
    def all_available_property_types(self) -> frozendict[str, InstantiatedType]:
        if self._all_available_property_types is None:
            assert self.available_property_data is not UNINITIALIZED
            types_of_data: dict[str, InstantiatedType] = {}
            for prop_name, prop_data in self.available_property_data.items():
                domain_concept_defining_type = prop_data[PropertyDefinitionKeywords.VALUE_DOMAIN]
                if domain_concept_defining_type == self.name:
                    types_of_data[prop_name] = self.property_types[prop_name]
                else:
                    parent = self.all_parents[domain_concept_defining_type]
                    assert isinstance(parent, DomainConceptData)
                    types_of_data[prop_name] = parent.property_types[prop_name]
            self._all_available_property_types = frozendict(types_of_data)
        return self._all_available_property_types

    @property
    def all_available_function_types(self) -> frozendict[str, InstantiatedType]:
        if self._all_available_function_types is None:
            assert self.available_function_data is not UNINITIALIZED
            types_of_data: dict[str, InstantiatedType] = {}
            for func_name, func_data in self.available_function_data.items():
                domain_concept_defining_type = func_data[FunctionDefinitionKeywords.VALUE_DOMAIN]
                if domain_concept_defining_type == self.name:
                    types_of_data[func_name] = self.function_types[func_name]
                else:
                    parent = self.all_parents[domain_concept_defining_type]
                    assert isinstance(parent, DomainConceptData)
                    types_of_data[func_name] = parent.function_types[func_name]
            self._all_available_function_types = frozendict(types_of_data)
        return self._all_available_function_types


@lazy_properties
class TypeData(ConceptData):
    template_context: TemplateContext
    parent_template_variable_substitution: frozendict[tuple[str, str], ConceptHierarchyTemplateArgument]
    """Contains all substitution values of the template arguments of all parents (also non direct parents!)"""

    def __init__(self, name: str, parents: frozendict[str, ConceptData], all_parents: frozendict[str, ConceptData]):
        super().__init__(name, parents, all_parents)


@lazy_properties
class ValueDomainData(TypeData):
    instantiation: tuple[tuple[ConstraintGroup | None, CHSchemaNode], ...]
    """`None` appears as the first element of the instantiation when the type does not have template arguments"""

    def __init__(self, name: str, parents: frozendict[str, ConceptData], all_parents: frozendict[str, ConceptData]):
        super().__init__(name, parents, all_parents)


@lazy_properties
class FunctionData(ValueDomainData):
    evaluation_argument_types: frozendict[str, TypeValue]
    evaluation_argument_access_type: frozendict[str, FunctionArgumentAccessor]
    evaluation_argument_provenance_type: frozendict[str, FunctionArgumentProvenance]
    evaluation_default_arguments: frozenset[str]
    """
    The whole collection of default arguments (also inherited ones from parent Functions).
    Needed separately from `evaluation_argument_default_value_expressions`.
    Because during parsing of expressions, the list of a Function's default arguments is needed.
    So keep this `evaluation_default_arguments` member, and then, during expression parsing, 
    populate the `evaluation_argument_default_value_expressions` member.
    """
    evaluation_argument_default_value_expressions: frozendict[str, Expression]
    evaluation_result_type: TypeValue | None
    evaluation_result_access_type: FunctionResultAccessor | None
    evaluation_result_provenance_type: ValueDomainArgumentProvenance | None

    default_argument_dependencies: frozendict[str, frozenset[str]]
    """
    Maps default argument names to the dependencies on the value of other Function arguments in the default expression.
    Every default argument name is in the frozendict; if it has no dependencies, its corresponding frozenset is empty. 
    """

    procedure: Expression

    sub_scope_vars: frozendict[str, frozendict[str, tuple[TypeValue, bool]]]
    """Maps evaluation argument names to new variables available in their scope and their type."""
    new_vars_in_scope: frozendict[str, tuple[TypeValue, bool]]
    """Maps the new variables introduced after the evaluation of this Function to their type."""

    def __init__(self, name: str, parents: frozendict[str, ConceptData], all_parents: frozendict[str, ConceptData]):
        super().__init__(name, parents, all_parents)

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
    value: Expression

    def __init__(self, name: str):
        super().__init__(name)


class ConceptHierarchy:
    def __init__(self, ch_def: ConceptHierarchyDefinition):
        self.ch = ch_def
        self.concepts: dict[str, ConceptData] = {}
        self.instances: dict[str, GlobalVariableData] = {}
        self.domain_concepts: dict[str, DomainConceptData] = {}
        self.value_domains: dict[str, ValueDomainData] = {}  # this includes functions and their FunctionData
        self.functions: dict[str, FunctionData] = {}

        self.x_template_variable_constraint: NonStructureConstraintFormula | None = None
