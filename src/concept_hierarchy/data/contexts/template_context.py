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

from frozendict import frozendict


class TemplateContext:
    @staticmethod
    def create_from(var: "TemplateContext") -> "TemplateContext":
        new_context = {}
        new_context.update(var.context)
        return TemplateContext(new_context)

    def __init__(self, template_context: dict[str, bool] | frozendict[str, bool] | None = None):
        self.context = frozendict(template_context) if template_context is not None else frozendict()

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "{}".format(self.context)

    def is_variadic(self, variable_name):
        return self.context[variable_name]

    def has_template_variable(self, variable_name) -> bool:
        return variable_name in self.context

    def add_template_variable(self, variable_name, is_variadic) -> "TemplateContext":
        if variable_name in self.context:
            raise RuntimeError(
                "Template variable {} already exists in TemplateContext {}! Can't add again!".format(
                    variable_name, self
                )
            )
        new_context = {variable_name: is_variadic}
        new_context.update(self.context)
        return TemplateContext(new_context)

    def add_template_variables(self, new_variables: dict[str, bool]) -> "TemplateContext":
        for var_name in new_variables:
            if var_name in self.context:
                raise RuntimeError(
                    "Template variable {} already exists in TemplateContext {}! Can't add again!".format(var_name, self)
                )
        new_context = {}
        new_context.update(self.context)
        new_context.update(new_variables)
        return TemplateContext(new_context)

    def add_context(self, context: "TemplateContext") -> "TemplateContext":
        for var_name in context.context:
            if var_name in self.context:
                raise RuntimeError(
                    "Template variable {} already exists in TemplateContext {}! Can't add again!".format(var_name, self)
                )
        new_context = {}
        new_context.update(self.context)
        new_context.update(context.context)
        return TemplateContext(new_context)
