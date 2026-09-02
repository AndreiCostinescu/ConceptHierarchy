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

from concept_hierarchy.data.concept_hierarchy import TypeData
from concept_hierarchy.data.contexts.context import ConceptHierarchyContext
from concept_hierarchy.data.contexts.template_context import TemplateContext
from concept_hierarchy.data.type_template_variables.constraint_formula import (
    HierarchyCheckType,
    LiteralValueConstraintFormula,
    NonStructureConstraintFormula,
    StructureConstraintFormula,
    TemplateConstraintFormulaValidator,
    TemplateConstraintHierarchyOperator,
    TemplateConstraintSelf,
)
from concept_hierarchy.data.type_template_variables.template_substitution import substitute
from concept_hierarchy.data.types.concept_hierarchy_types import (
    ConceptHierarchyTemplateArgument,
    ConceptHierarchyType,
    InstantiatedType,
    LiteralValue,
    NonVariadicTemplateVariable,
    TemplateVariable,
    VariadicArgument,
    VariadicTemplateVariable,
)
from concept_hierarchy.data.validators.template_argument_constraints_validator import (
    TemplateContextDeterminator,
    TypeTemplateInstantiationValidator,
    validate_complete_instantiation_of_concept,
)
from concept_hierarchy.definitions.concept_definition_hidden_implementation import HiddenImplementationDefinition
from concept_hierarchy.errors import LocationId


def create_exact_match_constraint_from_value(
    value: ConceptHierarchyTemplateArgument, validator: TemplateConstraintFormulaValidator, location_id: LocationId
) -> NonStructureConstraintFormula:
    if isinstance(value, LiteralValue):
        return LiteralValueConstraintFormula(value.literal_type, value.full_name, location_id)
    elif isinstance(value, ConceptHierarchyType):
        template_constraints = []
        for t_arg in value.template_arguments:
            template_constraints.append(create_exact_match_constraint_from_value(t_arg, validator, location_id))
        return TemplateConstraintSelf(value.clean_name, tuple(template_constraints), validator, location_id)
    elif isinstance(value, VariadicArgument):
        raise RuntimeError("[Feature-Request] Did not implement support for using variadic constraints!")
    else:
        assert isinstance(value, TemplateVariable)
        return TemplateConstraintSelf(value.clean_name, (), validator, location_id)


class TypeApplicationValidator(TypeTemplateInstantiationValidator):
    def __init__(self, context: ConceptHierarchyContext):
        self.context = context

    def is_concept(self, name: str) -> bool:
        return self.context.ch.is_concept(name)

    def is_a_subconcept_of_b(self, a: str, b: str, include_self: bool) -> bool:
        return self.context.ch.is_a_subconcept_of_b(a, b, include_self=include_self)

    def is_a_subtype_of_b(
        self,
        a: InstantiatedType,
        b: InstantiatedType,
        location_id: LocationId | None = None,
    ) -> bool:
        if not self.is_a_subconcept_of_b(a.clean_name, b.clean_name, include_self=True):
            return False
        b_subst_t_args: tuple[tuple[str, ConceptHierarchyTemplateArgument], ...] = self.create_substitution_for(
            b.clean_name, a, location_id
        )
        assert b_subst_t_args is not None
        if len(b_subst_t_args) != len(b.template_arguments):
            raise RuntimeError(
                f"Mismatch between the number of substituted template argument values {b_subst_t_args!r} and the number"
                f" of template arguments of {b.full_name!r}."
            )
        b_subst_t_args_to_check = tuple(x[1] for x in b_subst_t_args)
        return b_subst_t_args_to_check == b.template_arguments

    def concept_check(self, a_type: ConceptHierarchyType, b_name: str, check_type: HierarchyCheckType) -> bool:
        # perform the subconcept check!
        a_name = a_type.clean_name
        a_is_abstract = self.context.ch.concepts[a_name].abstract
        match check_type:
            case HierarchyCheckType.SELF:
                include_abstract = True
                return a_name == b_name and (not a_is_abstract or include_abstract)
            case HierarchyCheckType.DESCENDANTS_OF | HierarchyCheckType.NON_ABSTRACT_DESCENDANTS_OF:
                include_abstract = check_type != HierarchyCheckType.NON_ABSTRACT_DESCENDANTS_OF
                include_self = True
                return self.context.ch.is_a_subconcept_of_b(a_name, b_name, include_self=include_self) and (
                    not a_is_abstract or include_abstract
                )
            case HierarchyCheckType.ASCENDANTS_OF | HierarchyCheckType.NON_ABSTRACT_ASCENDANTS_OF:
                include_abstract = check_type != HierarchyCheckType.NON_ABSTRACT_DESCENDANTS_OF
                include_self = False
                return self.context.ch.is_a_subconcept_of_b(b_name, a_name, include_self=include_self) and (
                    not a_is_abstract or include_abstract
                )
            case _:
                raise RuntimeError(f"Unknown hierarchy check type: {check_type!r}")

    def create_substitution_for(
        self,
        parent_type_name: str,
        sub_type: ConceptHierarchyType,
        location_id: LocationId,
        template_context: TemplateContext | None = None,
    ) -> tuple[tuple[str, ConceptHierarchyTemplateArgument], ...] | None:
        sub_type_def_data = self.context.ch.concepts[sub_type.clean_name]
        parent_def_data = self.context.ch.concepts[parent_type_name]
        if not isinstance(parent_def_data, HiddenImplementationDefinition) or not parent_def_data.is_templatable():
            return ()
        # now, parent is definitely templatable
        if not isinstance(sub_type_def_data, HiddenImplementationDefinition):
            # sub_type is not a ValueDomain or a Function...
            # so a substitution between the sub_type and parent type does not exist
            return None

        if template_context is None:
            template_context = TemplateContext()

        substitution: dict[str, ConceptHierarchyTemplateArgument] = {}
        for t_arg_name, t_arg_val in zip(sub_type_def_data.template_argument_order, sub_type.template_arguments):
            substitution[t_arg_name] = t_arg_val

        sub_type_model_data = self.context.model.concepts[sub_type.clean_name]
        assert isinstance(sub_type_model_data, TypeData)
        all_substitutions_of_sub_type = sub_type_model_data.parent_template_variable_substitution

        substituted_t_args_of_parent: list[ConceptHierarchyTemplateArgument] = []
        for literal_t_arg in parent_def_data.template_argument_order:
            if (parent_type_name, literal_t_arg) not in all_substitutions_of_sub_type:
                assert sub_type.clean_name == parent_type_name, (
                    f"sub_type.clean_name = {sub_type.clean_name}, parent_type_name = {parent_type_name}"
                )
                if literal_t_arg in parent_def_data.variadic_template_arguments:
                    substitution_type = VariadicTemplateVariable(literal_t_arg, parent_def_data.name)
                else:
                    substitution_type = NonVariadicTemplateVariable(literal_t_arg, parent_def_data.name)
            else:
                substitution_type = all_substitutions_of_sub_type[parent_type_name, literal_t_arg]
            subst_val, subst_context = substitute(
                substitution_type,
                substitution,
                sub_type_model_data.template_context,
                template_context,
                self,
                location_id,
            )
            # The substitution must not *derive* any new constraint on the template variables that remain in
            # the substituted value; those variables stay in `template_context` and are validated below.
            if not subst_context.is_unconstrained:
                raise RuntimeError(
                    f"Expected full substitution for value {substitution_type}, but produced {subst_context!r}"
                )
            substituted_t_args_of_parent.append(subst_val)
        subst_tuple = tuple(substituted_t_args_of_parent)
        errors = validate_complete_instantiation_of_concept(
            parent_type_name, subst_tuple, TemplateContextDeterminator(template_context), self, location_id
        )
        if errors:
            raise RuntimeError(
                f"Wrong substitution {substituted_t_args_of_parent!r} because it doesn't satisfy all constraints of "
                f"{parent_type_name}"
            )
        return tuple(zip(parent_def_data.template_argument_order, subst_tuple))

    def get_template_argument_names_of(self, concept_name: str) -> tuple[str, ...]:
        if concept_name not in self.context.ch.concepts:
            raise RuntimeError(f"{concept_name} is not a concept!")
        c_data = self.context.ch.concepts[concept_name]
        if not isinstance(c_data, HiddenImplementationDefinition):
            return ()
        return c_data.template_argument_order

    def get_constraint_formula_of(self, name: str) -> StructureConstraintFormula | None:
        if name not in self.context.model.concepts:
            raise RuntimeError("fConcept {name} is not a concept!")
        model_data = self.context.model.concepts[name]
        if isinstance(model_data, TypeData) and not model_data.template_context.empty:
            return model_data.template_context.constraint
        return None

    def create_type_constraint_from_value(
        self,
        value: ConceptHierarchyTemplateArgument,
        location_id: LocationId,
        op: HierarchyCheckType = HierarchyCheckType.SELF,
        template_context: TemplateContext | None = None,
    ) -> NonStructureConstraintFormula:
        formula_validator = self.context.template_constraint_formula_validator
        if template_context is None:
            subst_formula = create_exact_match_constraint_from_value(value, formula_validator, location_id)
        else:
            # `value` may reference template variables; they are only recognised as such while they are in
            # the formula validator's scope. Registering the context's variables (rather than the ones the
            # value happens to mention) keeps the out-of-scope check intact: a reference to a variable that
            # is not in `template_context` still fails as "not a concept and not a template variable".
            previous_scope = formula_validator.get_existing_template_variables()
            formula_validator.update_existing_template_variables(set(template_context.variables))
            try:
                subst_formula = create_exact_match_constraint_from_value(value, formula_validator, location_id)
            finally:
                formula_validator.update_existing_template_variables(previous_scope)
        assert isinstance(subst_formula, TemplateConstraintHierarchyOperator)
        subst_formula = subst_formula.change_hierarchy_operator(
            op, self.context.template_constraint_formula_validator, location_id
        )
        return subst_formula

    def get_template_context(self) -> TemplateContext:
        return self.context.template_context
