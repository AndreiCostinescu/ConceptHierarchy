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

from concept_hierarchy.data.type_template_variables.constraint_formula import (
    ConstraintGroup,
    NonStructureConstraintFormula,
    StructureConjunction,
    StructureConstraintFormula,
    StructureDisjunction,
    StructureNegation,
    TemplateConstraintAnd,
    TemplateConstraintFormula,
    TypeTemplateConstraintFormula,
    Unconstrained,
)
from concept_hierarchy.data.type_template_variables.simplify_constraints import (
    create_empty_structure_constraint,
    create_unconstrained_structure_constraint,
    simplify_structure_constraint,
)
from concept_hierarchy.errors import LocationId


class TemplateContext:
    @staticmethod
    def create_from(context: TemplateContext) -> TemplateContext:
        new_variadic_variables: set[str] = set()
        new_variadic_variables.update(context.variadic_variables)
        return TemplateContext(context.variables, new_variadic_variables, context._constraint)

    def __init__(
        self,
        template_variables: tuple[str, ...] = (),
        variadic_variables: set[str] | frozenset[str] | None = None,
        constraint_on_variables: StructureConstraintFormula | None = None,
    ):
        """the order is very important here in variables; this is the order in the structure constraint"""
        self.variables = template_variables
        self.variadic_variables = frozenset(variadic_variables) if variadic_variables is not None else frozenset()
        self.nr_variables = len(self.variables)
        self._constraint = simplify_structure_constraint(constraint_on_variables)
        self.check_invariant(self.nr_variables, self._constraint)

    @staticmethod
    def check_invariant(
        nr_variables: int,
        constraint_to_check: StructureConstraintFormula | None = None,
    ) -> None:
        if (nr_variables == 0) != (constraint_to_check is None):
            if nr_variables == 0:
                raise RuntimeError(
                    f"TemplateContext has no variables but there is a constraint on them: {constraint_to_check}"
                )
            raise RuntimeError(
                f"TemplateContext has {nr_variables} template variables but there is no constraint on them"
            )
        if constraint_to_check is None:
            return

        def check_constraint_invariant(x: StructureConstraintFormula) -> None:
            if isinstance(x, ConstraintGroup):
                if len(x.group_constraints) != nr_variables:
                    raise RuntimeError(
                        f"There is a mismatch between the size of the constraint group ({len(x.group_constraints)}) and"
                        f" the number of template variables ({nr_variables})!"
                    )
            elif isinstance(x, StructureConjunction):
                for sub_x in x.structure_constraints:
                    check_constraint_invariant(sub_x)
            elif isinstance(x, StructureDisjunction):
                for sub_x in x.structure_constraints:
                    check_constraint_invariant(sub_x)
            elif isinstance(x, StructureNegation):
                check_constraint_invariant(x.structure_constraint)
            else:
                raise RuntimeError(f"Don't know how to process the constraint formula as a structure constraint: {x!r}")

        check_constraint_invariant(constraint_to_check)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return (
            f"TemplateContext(vars: {self.variables!r}, variadic: {sorted(self.variadic_variables)!r}, "
            f"constraint: {(None if self.empty else self.constraint)!r})"
        )

    def set_constraint(self, constraint: StructureConstraintFormula | None) -> None:
        self._constraint = constraint
        self.check_invariant(self.nr_variables, self.constraint)

    @property
    def constraint(self) -> StructureConstraintFormula:
        if self._constraint is None:
            raise RuntimeError(
                "There is no constraint on an empty TemplateContext. First check .empty() before calling this method!"
            )
        return self._constraint

    @property
    def empty(self) -> bool:
        return len(self.variables) == 0

    @property
    def is_unconstrained(self) -> bool:
        return True if self.empty else self.constraint.is_unconstrained

    @property
    def is_empty_constraint(self) -> bool:
        return False if self.empty else self.constraint.is_empty

    def is_variadic(self, variable_name) -> bool:
        return variable_name in self.variadic_variables

    def has_template_variable(self, variable_name) -> bool:
        return variable_name in self.variables

    def is_literal_template_variable(self, variable_name) -> bool:
        assert self.has_template_variable(variable_name)
        assert not self.empty
        return (
            self.constraint.variable_constraint_types[self.variables.index(variable_name)]
            != TypeTemplateConstraintFormula.TYPE
        )

    def constraint_sort(self, variable_name) -> str:
        return self.constraint.variable_constraint_types[self.variables.index(variable_name)]

    def add_and_constraint_to(
        self, variable_name: str, new_variable_constraint: NonStructureConstraintFormula, location_id: LocationId
    ) -> StructureConstraintFormula:
        """
        This function simplifies the created constraint
         because this function is used to overwrite a TemplateContext's constraint.

        :param variable_name:
        :param new_variable_constraint:
        :param location_id:
        :return:
        """
        if not self.has_template_variable(variable_name):
            raise RuntimeError(
                f"Can't add constraint to non-template variable: {variable_name!r}; available ones are "
                f"{self.variables!r}"
            )
        assert self._constraint is not None
        if new_variable_constraint.is_unconstrained:
            return self._constraint

        var_index = self.variables.index(variable_name)
        if new_variable_constraint.is_empty or self.is_unconstrained:
            # basically create empty or return the other constraint
            return self.create_unconstrained_except_with_constraint_at_index(
                location_id, var_index, new_variable_constraint
            )

        res: StructureConstraintFormula | None = None
        if isinstance(self._constraint, ConstraintGroup):
            existing_variable_constraint = self._constraint.group_constraints[var_index]
            if isinstance(existing_variable_constraint, TypeTemplateConstraintFormula) and isinstance(
                new_variable_constraint, TypeTemplateConstraintFormula
            ):
                new_variable_constraint = TemplateConstraintAnd(
                    location_id, (existing_variable_constraint, new_variable_constraint)
                )
                res = self.replace_constraint_at_index_with(var_index, new_variable_constraint)
        if res is None:
            new_constraint = self.create_unconstrained_except_with_constraint_at_index(
                location_id, var_index, new_variable_constraint
            )
            res = StructureConjunction(location_id, (self._constraint, new_constraint))
        return simplify_structure_constraint(res)

    def add_template_variable(
        self, variable_name: str, is_variadic: bool, constraint: TemplateConstraintFormula, location_id: LocationId
    ) -> TemplateContext:
        if variable_name in self.variables:
            raise RuntimeError(
                "Template variable {} already exists in TemplateContext {}! Can't add again!".format(
                    variable_name, self
                )
            )
        new_variables = self.variables + (variable_name,)
        new_variadic_variables = self.variadic_variables | ({variable_name} if is_variadic else set())
        if isinstance(constraint, NonStructureConstraintFormula):
            constraint = ConstraintGroup(location_id, (constraint,))
        assert isinstance(constraint, StructureConstraintFormula)
        self.check_invariant(1, constraint)  # removes Neg from structure and pushes it inward
        if self.empty:
            new_constraint = constraint
        else:
            new_constraint = self.extend_constraint(constraint, location_id)
        return TemplateContext(new_variables, new_variadic_variables, new_constraint)

    def delete_template_variable(self, template_variable_name: str, location_id: LocationId) -> TemplateContext:
        # FIXME: update procedure to work not only with the last template variable!
        if template_variable_name != self.variables[-1]:
            raise RuntimeError(
                f"Can't remove template variable {template_variable_name} which is not the last-added template "
                f"variable. All template variables: {self.variables}"
            )
        new_variables = self.variables[:-1]
        new_variadic_variables: set[str] = set(self.variadic_variables.copy())
        new_variadic_variables.discard(template_variable_name)
        # FIXME: update procedure to not depend on the constraint being a ConstraintGroup!
        if not isinstance(self._constraint, ConstraintGroup):
            raise RuntimeError(
                f"Can't remove template variable from non-ConstraintGroup constraint {self._constraint!r}"
            )
        if len(self.variables) == 1:
            new_constraint = None
        else:
            # FIXME: make sure that the remaining variables to not depend on the variable being removed!
            new_constraint = ConstraintGroup(location_id, self._constraint.group_constraints[:-1])
        return TemplateContext(new_variables, new_variadic_variables, new_constraint)

    def add_context(self, context: TemplateContext, location_id: LocationId) -> TemplateContext:
        for var_name in context.variables:
            if var_name in self.variables:
                raise RuntimeError(
                    "Template variable {} already exists in TemplateContext {}! Can't add again!".format(var_name, self)
                )
        new_variables = self.variables + context.variables
        new_variadic_variables = self.variadic_variables | context.variadic_variables
        new_constraint = self.extend_constraint(context.constraint, location_id)
        return TemplateContext(new_variables, new_variadic_variables, new_constraint)

    def extend_constraint(
        self, constraint_to_extend: StructureConstraintFormula, location_id: LocationId
    ) -> StructureConstraintFormula:
        """
        Does not simplify the result formula!
        Because (so far) this function is only used in the process of creating a new TemplateContext.
        And the template constraint is simplified in the TemplateContext constructor.
        """

        def extend(x: ConstraintGroup, y: ConstraintGroup) -> ConstraintGroup:
            return ConstraintGroup(location_id, x.group_constraints + y.group_constraints)

        def extend_other(x: ConstraintGroup, y: StructureConstraintFormula) -> StructureConstraintFormula:
            if isinstance(y, ConstraintGroup):
                return extend(x, y)
            elif isinstance(y, StructureConjunction):
                new_conjunctions = []
                for sub_y in y.structure_constraints:
                    new_conjunctions.append(extend_other(x, sub_y))
                return StructureConjunction(location_id, tuple(new_conjunctions))
            elif isinstance(y, StructureDisjunction):
                new_disjunctions = []
                for sub_y in y.structure_constraints:
                    new_disjunctions.append(extend_other(x, sub_y))
                return StructureDisjunction(location_id, tuple(new_disjunctions))
            elif isinstance(y, StructureNegation):
                raise RuntimeError(
                    f"There shouldn't be any structure negations left in the constraint, but found one: "
                    f"{constraint_to_extend}"
                )
            else:
                raise RuntimeError(f"Don't know how to process the constraint formula as a structure constraint: {x!r}")

        def extend_this(x: StructureConstraintFormula, y: StructureConstraintFormula) -> StructureConstraintFormula:
            if isinstance(x, ConstraintGroup):
                return extend_other(x, y)
            elif isinstance(x, StructureConjunction):
                new_conjunctions = []
                for sub_x in x.structure_constraints:
                    new_conjunctions.append(extend_this(sub_x, y))
                return StructureConjunction(x.location_id, tuple(new_conjunctions))
            elif isinstance(x, StructureDisjunction):
                new_disjunctions = []
                for sub_x in x.structure_constraints:
                    new_disjunctions.append(extend_this(sub_x, y))
                return StructureDisjunction(x.location_id, tuple(new_disjunctions))
            elif isinstance(x, StructureNegation):
                raise RuntimeError(
                    f"There shouldn't be any structure negations left in the constraint, but found one: "
                    f"{self._constraint!r}"
                )
            else:
                raise RuntimeError(f"Don't know how to process the constraint formula as a structure constraint: {x!r}")

        return extend_this(self._constraint, constraint_to_extend)

    def create_unconstrained_except_with_constraint_at_index(
        self, location_id: LocationId, var_index: int, var_constraint: NonStructureConstraintFormula
    ) -> ConstraintGroup:
        return ConstraintGroup(
            location_id,
            tuple(
                Unconstrained(location_id) if index != var_index else var_constraint
                for index in range(self.nr_variables)
            ),
        )

    def create_unconstrained_except_with_constraint_at_name(
        self, location_id: LocationId, var_name: str, var_constraint: NonStructureConstraintFormula
    ) -> ConstraintGroup:
        var_index = self.variables.index(var_name)
        return self.create_unconstrained_except_with_constraint_at_index(location_id, var_index, var_constraint)

    def replace_constraint_at_index_with(
        self, var_index, var_constraint: NonStructureConstraintFormula
    ) -> ConstraintGroup:
        """Do not simplify the created constraint because it is simplified at the (so far only) call-site."""
        assert isinstance(self._constraint, ConstraintGroup)
        new_constraint = tuple(
            x if index != var_index else var_constraint for index, x in enumerate(self._constraint.group_constraints)
        )
        return ConstraintGroup(var_constraint.location_id, new_constraint)

    def create_unconstrained_context(self, location_id: LocationId) -> TemplateContext:
        new_constraint = create_unconstrained_structure_constraint(self.nr_variables, location_id)
        return TemplateContext(self.variables, self.variadic_variables, new_constraint)

    def create_empty_context(self, location_id: LocationId) -> TemplateContext:
        new_constraint = create_empty_structure_constraint(
            self.nr_variables, self._constraint.variable_constraint_types, location_id
        )
        return TemplateContext(self.variables, self.variadic_variables, new_constraint)

    def make_constraint_neg(self) -> StructureConstraintFormula:
        return simplify_structure_constraint(StructureNegation(self._constraint.location_id, self._constraint))

    def merge_in_place(self, other: TemplateContext, location_id: LocationId) -> None:
        self.merge_constraints_and([other], location_id)

    def _check_contexts_to_merge(self, sub_template_contexts: list[TemplateContext]) -> None:
        for sub_template_context in sub_template_contexts:
            if (
                (self.variables != sub_template_context.variables)
                or (self.variadic_variables != sub_template_context.variadic_variables)
                or ((self._constraint is None) != (sub_template_context._constraint is None))
            ):
                raise RuntimeError(f"Can not merge unrelated template contexts: {self!r} and {sub_template_context!r}")

    def merge_constraints_and(self, sub_template_contexts: list[TemplateContext], location_id: LocationId) -> None:
        self._check_contexts_to_merge(sub_template_contexts)
        if self._constraint is None or self.is_empty_constraint:
            return
        new_constraints: list[StructureConstraintFormula] = []
        contexts_to_merge = sub_template_contexts
        for to_merge in contexts_to_merge:
            if to_merge.is_unconstrained:
                continue
            if to_merge.is_empty_constraint:
                self._constraint = create_empty_structure_constraint(
                    self.nr_variables, None if self.empty else self._constraint.variable_constraint_types, location_id
                )
                return
            new_constraints.append(to_merge.constraint)
        if new_constraints:
            if self.is_unconstrained:
                if len(new_constraints) == 1:
                    self._constraint = new_constraints[0]
                else:
                    self._constraint = StructureConjunction(location_id, tuple(new_constraints))
            else:
                new_constraints = [self._constraint] + new_constraints
                self._constraint = StructureConjunction(location_id, tuple(new_constraints))
            self._constraint = simplify_structure_constraint(self._constraint)

    def merge_constraints_or(self, sub_template_contexts: list[TemplateContext], location_id: LocationId) -> None:
        self._check_contexts_to_merge(sub_template_contexts)
        if self._constraint is None or self.is_unconstrained:
            return
        new_constraints: list[StructureConstraintFormula] = []
        contexts_to_merge = sub_template_contexts
        for to_merge in contexts_to_merge:
            if to_merge.is_unconstrained:
                self._constraint = create_unconstrained_structure_constraint(self.nr_variables, location_id)
                return
            if to_merge.is_empty_constraint:
                continue
            new_constraints.append(to_merge.constraint)
        if new_constraints:
            if self.is_empty_constraint:
                if len(new_constraints) == 1:
                    self._constraint = new_constraints[0]
                else:
                    self._constraint = StructureDisjunction(location_id, tuple(new_constraints))
            else:
                new_constraints = [self._constraint] + new_constraints
                self._constraint = StructureDisjunction(location_id, tuple(new_constraints))
            self._constraint = simplify_structure_constraint(self.constraint)
