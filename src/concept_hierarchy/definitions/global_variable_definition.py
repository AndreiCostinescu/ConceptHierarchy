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

from concept_hierarchy.definitions.definition import ConceptHierarchyDefinition, LocationOfCheckData
from concept_hierarchy.definitions.utils import check_ch_name
from concept_hierarchy.errors import CHSyntaxError, PathPart


class GlobalVariableDefinition(ConceptHierarchyDefinition):
    global_variable_name: str = "Global Variable"

    def __init__(self, name: str, definition_data: object, definition_location_str: str):
        self.orig_data = None
        self.value = None
        self.value_type = None
        self.deserialize_with_value = False
        self.is_literal = False
        self.is_instance = False
        super().__init__(name, definition_data, definition_location_str)

    def check(self):
        super().check()
        if not check_ch_name(self.name, allow_starting_with_underscore=True):
            raise CHSyntaxError(
                f"Names of global variables must be valid non-digit-starting string names, not {self.name}",
                location_id=self.location_id(),
                part=PathPart.KEY,
            )

    def definition_type(self) -> str:
        return GlobalVariableDefinition.global_variable_name

    def definition_location(self) -> list[str]:
        location_res = super().definition_location() + [self.name]
        if self.from_reference is not None:
            location_res.append("ref:" + self.from_reference)
        return location_res

    def location_of_impl(self, *keywords: str) -> LocationOfCheckData:
        # processes name-of-variable keyword (after processing parent keywords: "concepts"/"instances")
        # if there will be subclasses of this; extend this code with logic on when to raise StopLocationOfCheck
        return self.check_location_id(
            super().location_of_impl(*keywords),
            GlobalVariableDefinition.definition_location(self),
            location_check=self.name,
            previous_location=self.definition_location_str,
            allow_start_at_this_location=True,
        )

    def check_syntax(self):
        self.value = self.orig_data

    def check_semantics(self):
        """
        self.value_type, self.is_literal = determine_value_type(self.value, GlobalValueDomainInstance.template_context)
        self.deserialize_with_value = False
        self.is_instance = self.value_type.is_subtype_of("InstanceBase")
        if self.is_literal:
            pass
        elif self.is_instance:
            # temporary fix until values are correctly processed in python as well
            #  (1: creation/literal; 2: sub-type-creation; 3) function-call)
            self.deserialize_with_value = True
        else:
            # collect the used types in the definition to pass to valueDomains/generationUtils.cpp
            self.value = Expression.process_expression(
                self.value_type,
                ExpressionRef.NO_REF,
                ExpressionMod.GET,
                self.value,
                GlobalValueDomainInstance.template_context,
                GlobalValueDomainInstance.global_variable_context,
            )
        """
        pass
