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

from concept_hierarchy.data.types.parsed_type import ParsedType


class VariableContext:
    @staticmethod
    def create_from(var: VariableContext) -> VariableContext:
        new_context = {}
        new_context.update(var.context)
        return VariableContext(new_context)

    def __init__(
        self,
        context: frozendict[str, ParsedType | dict] | dict[str, ParsedType | dict] | None = None,
    ):
        self.context = frozendict(context if context is not None else {})

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "{}".format(self.context)

    @property
    def empty(self):
        return len(self.context) == 0

    def has_variable(self, variable_name) -> bool:
        return variable_name in self.context

    def get(self, var_name: str) -> ParsedType:
        var_type_datum = self.context.get(var_name)
        if isinstance(var_type_datum, ParsedType):
            return var_type_datum
        assert isinstance(var_type_datum, dict)
        return var_type_datum.get("inferredValueDomain", var_type_datum["valueDomain"])

    def set_inferred_type_for(
        self, var_name: str, inferred_value_domain: ParsedType, allow_new_variables: bool = False
    ) -> VariableContext:
        if var_name not in self.context:
            if allow_new_variables:
                return self.add_variable(var_name, inferred_value_domain)
            raise RuntimeError(
                "Can't create a new variable with the set_inferred_type_for method; "
                "{} does not exist in context {}".format(var_name, self.context)
            )
        new_context = {}
        new_context.update(self.context)
        var_type_datum = self.context[var_name]
        if isinstance(var_type_datum, ParsedType):
            new_context[var_name] = {"valueDomain": var_type_datum, "inferredValueDomain": inferred_value_domain}
        else:
            new_context[var_name]["inferredValueDomain"] = inferred_value_domain
        return VariableContext(new_context)

    def add_variable(self, var_name, value_domain: ParsedType) -> VariableContext:
        if var_name in self.context:
            raise RuntimeError(
                "Variable {} already exists in VariableContext {}! Can't add again!".format(var_name, self)
            )
        new_context = {var_name: value_domain}
        new_context.update(self.context)
        return VariableContext(new_context)

    def add_variables(self, new_variables: dict[str, ParsedType | dict]) -> VariableContext:
        for var_name in new_variables:
            if var_name in self.context:
                raise RuntimeError(
                    "Variable {} already exists in VariableContext {}! Can't add again!".format(var_name, self)
                )
        new_context = {}
        new_context.update(self.context)
        new_context.update(new_variables)
        return VariableContext(new_context)

    def add_context(self, context: VariableContext) -> VariableContext:
        for var_name in context.context:
            if var_name in self.context:
                raise RuntimeError(
                    f"Variable {var_name!r} already exists in VariableContext {self!r}! Can't add again!"
                )
        new_context = {}
        new_context.update(self.context)
        new_context.update(context.context)
        return VariableContext(new_context)
