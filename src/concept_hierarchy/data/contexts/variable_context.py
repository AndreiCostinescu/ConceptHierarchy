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

from concept_hierarchy.data.types.concept_hierarchy_types import (
    TYPE_VALUE_IS_INSTANCE_CHECK,
    TypeValue,
)


class VariableStackFrame:
    def __init__(
        self,
        variables: frozendict[str, TypeValue | dict] | dict[str, TypeValue | dict] | None = None,
    ):
        self.variables = frozendict(variables if variables is not None else {})

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "{}".format(self.variables)

    @property
    def empty(self):
        return len(self.variables) == 0

    def clone(self) -> VariableStackFrame:
        new_variables = {}
        new_variables.update(self.variables)
        return VariableStackFrame(new_variables)

    def has_variable(self, variable_name) -> bool:
        return variable_name in self.variables

    def get(self, var_name: str) -> TypeValue:
        var_type_datum = self.variables.get(var_name)
        if isinstance(var_type_datum, TYPE_VALUE_IS_INSTANCE_CHECK):
            return var_type_datum
        assert isinstance(var_type_datum, dict)
        return var_type_datum.get("inferredValueDomain", var_type_datum["valueDomain"])

    def set_inferred_type_for(
        self, var_name: str, inferred_value_domain: TypeValue, allow_new_variables: bool = False
    ) -> VariableStackFrame:
        if var_name not in self.variables:
            if allow_new_variables:
                return self.add_variable(var_name, inferred_value_domain)
            raise RuntimeError(
                "Can't create a new variable with the set_inferred_type_for method; "
                "{} does not exist in context {}".format(var_name, self.variables)
            )
        new_variables = {}
        new_variables.update(self.variables)
        var_type_datum = self.variables[var_name]
        if isinstance(var_type_datum, TYPE_VALUE_IS_INSTANCE_CHECK):
            new_variables[var_name] = {"valueDomain": var_type_datum, "inferredValueDomain": inferred_value_domain}
        else:
            new_variables[var_name]["inferredValueDomain"] = inferred_value_domain
        return VariableStackFrame(new_variables)

    def add_variable(self, var_name, value_domain: TypeValue) -> VariableStackFrame:
        if var_name in self.variables:
            raise RuntimeError(
                "Variable {} already exists in VariableStackFrame {}! Can't add again!".format(var_name, self)
            )
        new_variables = {var_name: value_domain}
        new_variables.update(self.variables)
        return VariableStackFrame(new_variables)

    def add_variables(self, new_variables: dict[str, TypeValue | dict]) -> VariableStackFrame:
        for var_name in new_variables:
            if var_name in self.variables:
                raise RuntimeError(
                    "Variable {} already exists in VariableStackFrame {}! Can't add again!".format(var_name, self)
                )
        new_variables = {}
        new_variables.update(self.variables)
        new_variables.update(new_variables)
        return VariableStackFrame(new_variables)

    def add_frame(self, frame: VariableStackFrame) -> VariableStackFrame:
        for var_name in frame.variables:
            if var_name in self.variables:
                raise RuntimeError(
                    f"Variable {var_name!r} already exists in VariableStackFrame {self!r}! Can't add again!"
                )
        new_variables = {}
        new_variables.update(self.variables)
        new_variables.update(frame.variables)
        return VariableStackFrame(new_variables)


class VariableContext:
    def __init__(self, stack_frames: list[VariableStackFrame]):
        self.stack_frames = stack_frames

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "\n".join(f"{x}" for x in self.stack_frames)

    def clone(self) -> VariableContext:
        return VariableContext([x.clone() for x in self.stack_frames])

    @property
    def empty(self):
        return len(self.stack_frames) == 0

    def has_variable(self, variable_name) -> bool:
        for frame in reversed(self.stack_frames):
            if variable_name in frame.variables:
                return True
        return False

    def get(self, var_name: str) -> TypeValue:
        for frame in reversed(self.stack_frames):
            if var_name in frame.variables:
                return frame.get(var_name)
        assert not self.has_variable(var_name)
        raise RuntimeError(f"Variable {var_name} not in scope:\n{self}")

    def set_inferred_type_for(
        self, var_name: str, inferred_value_domain: TypeValue, allow_new_variables: bool = False
    ) -> VariableContext:
        for replace_index, frame in enumerate(reversed(self.stack_frames)):
            if var_name in frame.variables:
                new_frame = frame.set_inferred_type_for(var_name, inferred_value_domain, allow_new_variables)
                new_stack_frames = [
                    x.clone() if index != replace_index else new_frame for index, x in enumerate(self.stack_frames)
                ]
                return VariableContext(new_stack_frames)
        assert not self.has_variable(var_name)
        raise RuntimeError(f"Variable {var_name} not in scope:\n{self}")

    def add_variable(self, var_name, value_domain: TypeValue) -> VariableContext:
        new_frame = self.stack_frames[-1].add_variable(var_name, value_domain)
        new_context = self.clone()
        new_context.stack_frames[-1] = new_frame
        return new_context

    def add_variables(self, new_variables: dict[str, TypeValue | dict]) -> VariableContext:
        new_frame = self.stack_frames[-1].add_variables(new_variables)
        new_context = self.clone()
        new_context.stack_frames[-1] = new_frame
        return new_context

    def push_variable_context(self, variables: VariableStackFrame) -> None:
        self.stack_frames.append(variables)

    def pop_variable_context(self) -> None:
        self.stack_frames.pop()

    def add_context(self, context: VariableContext) -> VariableContext:
        new_stack_frames = self.clone().stack_frames + context.clone().stack_frames
        return VariableContext(new_stack_frames)
