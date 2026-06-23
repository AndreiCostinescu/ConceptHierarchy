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

from concept_hierarchy.data.concept_hierarchy import (
    DomainConceptData,
    FunctionData,
    TypeData,
    ValueDomainData,
)
from concept_hierarchy.data.contexts.context import ConceptHierarchyContext
from concept_hierarchy.definitions.concept_definition_domain_concept import DomainConceptDefinition
from concept_hierarchy.definitions.concept_definition_functions import FunctionDefinition
from concept_hierarchy.definitions.concept_definition_hidden_implementation import HiddenImplementationDefinition
from concept_hierarchy.definitions.concept_definition_value_domain import ValueDomainDefinition


def check_expressions_in_domain_concept_definition(
    c: DomainConceptDefinition, datum: DomainConceptData, context: ConceptHierarchyContext
):
    """
    The expressions to verify are:
    - domain concept properties
        - constraints + set valueDomain if not set
        - confidence duration value
        - hook procedures
            - checks that functions exist,
            - function arguments exist,
            - function argument type is a parent type of the property type
        - computation procedures
        - assumption Variation values
        - default values
        - specialization values:
            - check all of the above
    - domain concept function procedures:
        - CustomFunction instantiations
        - Function procedures
        - specialization values:
            - default Function procedures / CustomFunction instantiations
    - domain concept management:
        - initialization function procedure
        - consolidation function procedure
    """
    pass


def check_expressions_in_hidden_implementation_definition(
    c: HiddenImplementationDefinition, datum: TypeData, context: ConceptHierarchyContext
):
    """No expressions in HiddenImplementationDefinitions."""
    pass


def check_expressions_in_value_domain_definition(
    c: ValueDomainDefinition, datum: ValueDomainData, context: ConceptHierarchyContext
):
    """
    Check default values of value domain instantiations (plus type-check value domain instantiations themselves)
    """
    pass


def check_expressions_in_function_definition(
    c: FunctionDefinition, datum: FunctionData, context: ConceptHierarchyContext
):
    """
    Check Function evaluation default argument values, variations, procedure, and inversions

    :param c: Function concept for which to check expressions
    :param datum: the output data container
    :param context: the concept hierarchy in which the check is made
    """
    pass


def check_expressions_in_concept_hierarchy(context: ConceptHierarchyContext):
    # First process template types
    for c_name, c in context.ch.concepts.items():
        if isinstance(c, HiddenImplementationDefinition):
            check_expressions_in_hidden_implementation_definition(c, context.model.value_domains[c_name], context)
    # Then check types in the Concept Hierarchy (property default values, Function argument default values, etc.)
    for c_name, c in context.ch.concepts.items():
        if isinstance(c, DomainConceptDefinition):
            check_expressions_in_domain_concept_definition(c, context.model.domain_concepts[c_name], context)
        if isinstance(c, ValueDomainDefinition):
            check_expressions_in_value_domain_definition(c, context.model.value_domains[c_name], context)
        if isinstance(c, FunctionDefinition):
            check_expressions_in_function_definition(c, context.model.functions[c_name], context)
