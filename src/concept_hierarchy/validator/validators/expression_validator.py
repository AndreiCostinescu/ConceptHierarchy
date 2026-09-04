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

from contextlib import contextmanager
from typing import Iterator

from frozendict import frozendict

from concept_hierarchy.data.contexts.context import ConceptHierarchyContext
from concept_hierarchy.data.contexts.template_context import TemplateContext
from concept_hierarchy.data.contexts.variable_context import VariableContext, VariableStackFrame
from concept_hierarchy.data.expressions.expression import Expression
from concept_hierarchy.data.expressions.expression_utils import (
    FunctionArgumentAccessor,
    FunctionArgumentProvenance,
    FunctionResultAccessor,
    ValueDomainArgumentProvenance,
)
from concept_hierarchy.data.expressions.instantiated_value import ParsedValue
from concept_hierarchy.data.jsonschema import CHSchemaNode
from concept_hierarchy.data.parsers.expression_parser import (
    ExpressionParserValidator,
    GroundedArgumentDefault,
    expansion_cycle_expression,
    parse_expression,
)
from concept_hierarchy.data.parsers.value_instantiation_parser import parse_value
from concept_hierarchy.data.type_template_variables.constraint_formula import ConstraintGroup
from concept_hierarchy.data.types.concept_hierarchy_types import (
    ConceptHierarchyTemplateArgument,
    InstantiatedType,
    TemplateDependentType,
    TypeValue,
)
from concept_hierarchy.data.validators.template_argument_constraints_validator import TypeTemplateInstantiationValidator
from concept_hierarchy.data.validators.type_validator import parse_convert_type, parse_convert_type_in_template_context
from concept_hierarchy.errors import ConceptHierarchyError, LocationId


class ExpressionValidator(ExpressionParserValidator):
    def __init__(self, context: ConceptHierarchyContext):
        self.context = context
        self.instantiated_types: dict[str, InstantiatedType] = {}
        self.parsed_types: dict[str, TypeValue] = {}
        self.resolved_instantiation_schemas: dict[tuple[str, int], CHSchemaNode] = {}
        """(application full name, constraint group index) -> schema substituted for that application."""

        self._default_sites: dict[int, tuple[CHSchemaNode, dict | None, int]] = {}
        """Registered default sites, by node identity: how to parse each one when it is first needed."""

        self._grounded_function_defaults: dict[tuple[str, str], GroundedArgumentDefault] = {}
        """
        Every Function argument default already grounded, keyed ``(application full_name, argument)``.

        Both the memo and the on-path guard: an entry is written *before* its default is parsed, marked
        `GroundedArgumentDefault.in_progress`, so a default that reaches itself finds it there. Keyed by
        application rather than by the declaring Function so that two call sites evaluating the same
        application share the work, and two different applications of it do not.
        """

        self._resolving: set[int] = set()
        """The default sites on the current resolution path. Re-entering one is an expansion cycle."""

    def is_concept(self, candidate_concept_name: str) -> bool:
        return self.context.ch.is_concept(candidate_concept_name)

    def is_template_variable(self, candidate_template_variable_name: str) -> bool:
        return self.context.template_context.has_template_variable(candidate_template_variable_name)

    def is_literal_template_variable(self, candidate_literal_template_variable_name: str) -> bool:
        return self.context.template_context.is_literal_template_variable(candidate_literal_template_variable_name)

    def get_literal_template_var_constraint(self, literal_template_variable_name: str) -> str:
        return self.context.template_context.constraint_sort(literal_template_variable_name)

    def is_variable(self, candidate_variable_name: str) -> bool:
        return self.context.variable_context.has_variable(
            self.context.ch.canonical_variable_name(candidate_variable_name)
        )

    def get_variable_type(self, variable_name: str) -> TypeValue:
        # A global variable may be written under any of its names; the variable context is keyed by the
        # canonical one. Local names can not collide with an alias -- a Function argument, a property or a
        # concept function sharing a global variable's name is rejected in `check_after_parsing_concepts`.
        return self.context.variable_context.get(self.context.ch.canonical_variable_name(variable_name))

    def is_type_abstract(self, candidate_type: InstantiatedType | TemplateDependentType) -> bool:
        return not self.context.model.value_domains[candidate_type.clean_name].instantiable

    def is_a_subtype_of_b(self, a: InstantiatedType, b: InstantiatedType, location_id: LocationId) -> bool:
        return self.context.type_application_constraints_validator.is_a_subtype_of_b(a, b, location_id)

    def create_instantiated_type(self, instantiated_type_name: str, location_id: LocationId) -> InstantiatedType:
        if instantiated_type_name not in self.instantiated_types:
            self.instantiated_types[instantiated_type_name] = parse_convert_type(
                instantiated_type_name, self.context.type_validator, location_id
            )
            self.parsed_types[instantiated_type_name] = self.instantiated_types[instantiated_type_name]
        return self.instantiated_types[instantiated_type_name]

    def create_possibly_template_dependent_type(self, type_name: str, location_id: LocationId) -> TypeValue:
        if type_name not in self.parsed_types:
            self.parsed_types[type_name] = parse_convert_type_in_template_context(
                type_name, self.context.type_validator, location_id
            )
            if isinstance(self.parsed_types[type_name], InstantiatedType):
                self.instantiated_types[type_name] = self.parsed_types[type_name]
        return self.parsed_types[type_name]

    def get_substituted_value_domain_instantiation_schema(
        self, type_name: InstantiatedType | TemplateDependentType
    ) -> TypeValue:
        raise NotImplementedError

    def get_substituted_function_interface(
        self, ch_type: InstantiatedType
    ) -> tuple[tuple[str, ...], set[str], InstantiatedType | None, ValueDomainArgumentProvenance | None]:
        raise NotImplementedError

    def get_properties_of_concepts(self, concepts: list[str]) -> dict[str, InstantiatedType]:
        raise NotImplementedError

    def get_type_of_instance_property(self, instance_type: InstantiatedType, prop_name: str) -> InstantiatedType:
        raise NotImplementedError

    def get_if_has_instantiation_schema(
        self, type_data: TypeValue
    ) -> tuple[tuple[ConstraintGroup, CHSchemaNode], ...] | None:
        """
        If the type is fully instantiated, return the schema.
        If the type is a TemplateDependentType, return all instantiation schemas.
        Otherwise, return None

        :param type_data: the type to check for an instantiation schema
        :return:
        """
        if not isinstance(type_data, (InstantiatedType, TemplateDependentType)):
            return None

        assert self.is_concept(type_data.clean_name)
        concept_name = type_data.clean_name
        assert concept_name in self.context.model.value_domains
        value_domain_instantiation = self.context.model.value_domains[concept_name].instantiation
        return value_domain_instantiation

    def validate_value_against_schema(
        self,
        schema: CHSchemaNode,
        value: object,
        location_id: LocationId,
        template_substitution: dict[str, ConceptHierarchyTemplateArgument] | None,
        expansion_depth: int,
    ) -> tuple[ParsedValue, list[ConceptHierarchyError]]:
        return parse_value(
            value,
            schema,
            self.context.instantiation_values_validator,
            location_id,
            template_substitution,
            expansion_depth,
        )

    def register_default_site(
        self, node: CHSchemaNode, template_substitution: dict | None, expansion_depth: int
    ) -> None:
        # Keyed by identity, which needs two things to be safe, both of which hold:
        #
        # - the id can not be recycled after this in the traversal.
        #   Enforced, because this registry stores the node itself and so keeps it alive for as long as the entry exists
        # - the node can not be *copied* after being registered, which would leave the copy unregistered.
        #   `CHSchemaNode` is NOW only ever copied by `substitute_schema` and `_copy_schema`, both called from
        #   `build_resolved_instantiation_schema` -- which registers afterward, on the tree it returns,
        #   and is itself guarded by the resolved-schema cache. Registration is therefore the last thing
        #   that happens to a node. **If another copy is ever introduced between the two, this breaks
        #   silently**: the copy's default is simply never resolved.
        #
        # Identity is also the only correct key: `CHSchemaNode` is an `eq=True` dataclass and hence
        # unhashable, and two distinct sites can compare equal anyway (which would be wrong).
        self._default_sites[id(node)] = (node, template_substitution, expansion_depth)

    def resolve_default_site(self, node: CHSchemaNode) -> Expression | None:
        if node.parsed_default_expr is not None:
            return node.parsed_default_expr
        site = self._default_sites.get(id(node))
        if site is None:
            # Not a site of a schema resolved for a ground application; nothing to resolve here.
            return None
        _node, template_substitution, expansion_depth = site
        if id(node) in self._resolving:
            # Reached while it is still being resolved: the expansion needs this very default in order to
            # produce it. Memoized, because a site that is cyclic once is cyclic always.
            node.parsed_default_expr = expansion_cycle_expression(node)
            return node.parsed_default_expr
        self._resolving.add(id(node))
        try:
            # The depth counted is how many *applications* deep this is, recorded when the schema was
            # built -- deliberately not the length of the current resolution path. See
            # `register_default_site`: a long path within one schema is finite by construction and must
            # not be bounded, while a path that keeps generating new applications is not and must be.
            # Counts application builds, not the length of this path -- see `register_default_site`: a
            # long path within the schemas already built is finite by construction and must not be
            # bounded, while one that keeps generating applications is not and must be.
            node.parsed_default_expr = parse_expression(
                node.default_expr,
                node.custom_type,
                node.provenance,
                FunctionArgumentAccessor.GET,
                TemplateContext(),
                self,
                node.location_id + ["default"],
                template_substitution=template_substitution,
                expansion_depth=expansion_depth,
            )
        finally:
            self._resolving.discard(id(node))
        return node.parsed_default_expr

    def get_resolved_instantiation_schema(self, cache_key: tuple[str, int]) -> CHSchemaNode | None:
        return self.resolved_instantiation_schemas.get(cache_key)

    def put_resolved_instantiation_schema(self, cache_key: tuple[str, int], schema: CHSchemaNode) -> None:
        self.resolved_instantiation_schemas[cache_key] = schema

    def get_default_expansion_depth_limit(self) -> int:
        return self.context.ch.get_expansion_depth_limit_for_default_instantiation_expressions()

    def get_default_serialization_concept_name_for(self, json_value_type: str) -> str | None:
        return self.context.ch.default_serializations.get(json_value_type, None)

    def get_function_return_interface(
        self, f_name
    ) -> tuple[TypeValue, FunctionResultAccessor, ValueDomainArgumentProvenance] | None:
        assert f_name in self.context.model.functions
        if not self.context.model.functions[f_name].returns_something:
            return None
        f = self.context.model.functions[f_name]
        return f.evaluation_result_type, f.evaluation_result_access_type, f.evaluation_result_provenance_type

    def is_function_argument(self, f_name, f_arg_name) -> bool:
        assert f_name in self.context.model.functions
        return f_arg_name in self.context.model.functions[f_name].evaluation_argument_types

    def get_function_arguments(self, f_name) -> set[str]:
        assert f_name in self.context.model.functions
        return set(self.context.model.functions[f_name].evaluation_argument_types)

    def get_required_function_arguments(self, f_name) -> set[str]:
        assert f_name in self.context.model.functions
        f = self.context.model.functions[f_name]
        return set(f.evaluation_argument_types) - set(f.evaluation_default_arguments)

    def get_function_argument_interface(
        self, f_name, f_arg_name
    ) -> tuple[TypeValue, FunctionArgumentAccessor, FunctionArgumentProvenance]:
        assert f_name in self.context.model.functions
        f = self.context.model.functions[f_name]
        assert f_arg_name in f.evaluation_argument_types
        return (
            f.evaluation_argument_types[f_arg_name],
            f.evaluation_argument_access_type[f_arg_name],
            f.evaluation_argument_provenance_type[f_arg_name],
        )

    def get_default_argument_dependencies(self, f_name) -> frozendict[str, frozenset[str]] | None:
        assert f_name in self.context.model.functions
        f = self.context.model.functions[f_name]
        if f.is_default_argument_dependencies_initialized():
            return f.default_argument_dependencies
        return None

    def get_function_argument_default(self, f_name: str, f_arg_name: str) -> Expression | None:
        assert f_name in self.context.model.functions, f_name
        return self.context.model.functions[f_name].evaluation_argument_default_value_expressions.get(f_arg_name)

    @contextmanager
    def function_argument_scope(self, arguments: dict[str, TypeValue]) -> Iterator[None]:
        previous = self.context.variable_context
        assert previous is not None, "there is no variable context to scope"
        # Frame 0 is the global variables -- `check_expressions_in_concept_hierarchy` starts from an empty
        # context and `init_expressions` pushes them first, before anything Function- or ValueDomain-local.
        # Those stay; every frame above them is dropped for the duration.
        globals_frame = previous.stack_frames[0] if previous.stack_frames else VariableStackFrame()
        self.context.set_variable_context(VariableContext([globals_frame, VariableStackFrame(arguments)]))
        try:
            yield
        finally:
            self.context.set_variable_context(previous)

    def get_grounded_function_default(self, application: str, argument: str) -> GroundedArgumentDefault | None:
        return self._grounded_function_defaults.get((application, argument))

    def put_grounded_function_default(self, application: str, argument: str, grounded: GroundedArgumentDefault) -> None:
        self._grounded_function_defaults[(application, argument)] = grounded

    def get_template_context(self, type_name_clean) -> TemplateContext:
        assert type_name_clean in self.context.model.value_domains
        return self.context.model.value_domains[type_name_clean].template_context

    def get_type_template_instantiation_validator(self) -> TypeTemplateInstantiationValidator:
        return self.context.type_application_constraints_validator
