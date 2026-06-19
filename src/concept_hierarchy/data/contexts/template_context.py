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

from concept_hierarchy.data.template_argument_constraints.constraint_formula import (
    ConstraintGroup,
    Empty,
    NonStructureConstraintFormula,
    StructureConjunction,
    StructureConstraintFormula,
    StructureDisjunction,
    StructureNegation,
    TemplateConstraintAnd,
    TypeTemplateConstraintFormula,
    Unconstrained,
)
from concept_hierarchy.errors import LocationId


class TemplateContext:
    @staticmethod
    def create_from(var: TemplateContext) -> TemplateContext:
        new_variadic_variables: set[str] = set()
        new_variadic_variables.update(var.variadic_variables)
        return TemplateContext(var.variables, new_variadic_variables, var.constraint)

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
        self.constraint = constraint_on_variables
        self.check_invariant(self.nr_variables, self.constraint)

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
            f"TemplateContext(vars: {self.variables!r}, variadic: {self.variadic_variables!r}, "
            f"constraint: {self.constraint!r})"
        )

    @property
    def empty(self) -> bool:
        return len(self.variables) == 0

    @property
    def is_unconstrained(self) -> bool:
        return self.constraint.is_unconstrained

    @property
    def is_empty_constraint(self) -> bool:
        return self.constraint.is_empty

    def is_variadic(self, variable_name) -> bool:
        return variable_name in self.variadic_variables

    def has_template_variable(self, variable_name) -> bool:
        return variable_name in self.variables

    def add_and_constraint_to(
        self, variable_name: str, new_variable_constraint: NonStructureConstraintFormula, location_id: LocationId
    ) -> StructureConstraintFormula:
        if not self.has_template_variable(variable_name):
            raise RuntimeError(
                f"Can't add constraint to non-template variable: {variable_name!r}; available ones are "
                f"{self.variables!r}"
            )
        assert self.constraint is not None
        if new_variable_constraint.is_unconstrained:
            return self.constraint

        var_index = self.variables.index(variable_name)
        if new_variable_constraint.is_empty:
            # basically create empty!
            return self.create_unconstrained_except_with_constraint_at_index(
                location_id, var_index, new_variable_constraint
            )

        if isinstance(self.constraint, ConstraintGroup):
            existing_variable_constraint = self.constraint.group_constraints[var_index]
            if isinstance(existing_variable_constraint, TypeTemplateConstraintFormula) and isinstance(
                new_variable_constraint, TypeTemplateConstraintFormula
            ):
                new_variable_constraint = TemplateConstraintAnd(
                    location_id, (existing_variable_constraint, new_variable_constraint)
                )
                return self.replace_constraint_at_index_with(var_index, new_variable_constraint)
        new_constraint = self.create_unconstrained_except_with_constraint_at_index(
            location_id, var_index, new_variable_constraint
        )
        return StructureConjunction(location_id, (self.constraint, new_constraint))

    def add_template_variable(
        self, variable_name: str, is_variadic: bool, constraint: StructureConstraintFormula, location_id: LocationId
    ) -> TemplateContext:
        if variable_name in self.variables:
            raise RuntimeError(
                "Template variable {} already exists in TemplateContext {}! Can't add again!".format(
                    variable_name, self
                )
            )
        new_variables = self.variables + (variable_name,)
        new_variadic_variables = self.variadic_variables | ({variable_name} if is_variadic else set())
        self.check_invariant(1, constraint)  # removes Neg from structure and pushes it inward
        new_constraint = self.extend_constraint(constraint, location_id)
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
                    f"{self.constraint!r}"
                )
            else:
                raise RuntimeError(f"Don't know how to process the constraint formula as a structure constraint: {x!r}")

        return extend_this(self.constraint, constraint_to_extend)

    def create_unconstrained(self, location_id: LocationId) -> ConstraintGroup:
        return ConstraintGroup(location_id, tuple(Unconstrained(location_id) for _ in self.variables))

    def create_empty(self, location_id: LocationId) -> ConstraintGroup:
        return ConstraintGroup(location_id, tuple(Empty(location_id) for _ in self.variables))

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

    def replace_constraint_at_index_with(
        self, var_index, var_constraint: NonStructureConstraintFormula
    ) -> ConstraintGroup:
        assert isinstance(self.constraint, ConstraintGroup)
        new_constraint = tuple(
            x if index != var_index else var_constraint for index, x in enumerate(self.constraint.group_constraints)
        )
        return ConstraintGroup(var_constraint.location_id, new_constraint)

    def create_unconstrained_context(self, location_id: LocationId) -> TemplateContext:
        return TemplateContext(self.variables, self.variadic_variables, self.create_unconstrained(location_id))

    def make_constraint_neg(self) -> StructureConstraintFormula:
        return StructureNegation(self.constraint.location_id, self.constraint)

    def merge_in_place(self, other: TemplateContext, location_id: LocationId) -> None:
        if (
            (self.variables != other.variables)
            or (self.variadic_variables != other.variadic_variables)
            or ((self.constraint is None) != (other.constraint is None))
        ):
            raise RuntimeError(f"Can not merge unrelated template contexts: {self!r} and {other!r}")
        if self.constraint is None:
            return
        # make and constraint!
        if other.is_unconstrained:
            return
        if other.is_empty_constraint:
            self.constraint = self.create_empty(location_id)
            return
        assert self.constraint is not None and other.constraint is not None
        self.constraint = StructureConjunction(location_id, (self.constraint, other.constraint))

    """
    This does not perform any compatibility checks between the constraints! 
    It assumes that there are the same template variables are used in the constraint!
    """

    def merge_constraints_and(self, sub_template_contexts: list[TemplateContext], location_id: LocationId) -> None:
        if self.is_empty_constraint:
            return
        new_constraints: list[StructureConstraintFormula] = []
        contexts_to_merge = sub_template_contexts
        for to_merge in contexts_to_merge:
            if to_merge.is_unconstrained:
                continue
            if to_merge.is_empty_constraint:
                self.constraint = self.create_empty(location_id)
                return
            new_constraints.append(to_merge.constraint)
        if new_constraints:
            if self.is_unconstrained:
                if len(new_constraints) == 1:
                    self.constraint = new_constraints[0]
                else:
                    self.constraint = StructureConjunction(location_id, tuple(new_constraints))
            else:
                new_constraints = [self.constraint] + new_constraints
                self.constraint = StructureConjunction(location_id, tuple(new_constraints))

    """
    This does not perform any compatibility checks between the constraints! 
    It assumes that there are the same template variables are used in the constraint!
    """

    def merge_constraints_or(self, sub_template_contexts: list[TemplateContext], location_id: LocationId) -> None:
        if self.is_unconstrained:
            return
        new_constraints: list[StructureConstraintFormula] = []
        contexts_to_merge = sub_template_contexts
        for to_merge in contexts_to_merge:
            if to_merge.is_unconstrained:
                self.constraint = self.create_unconstrained(location_id)
                return
            if to_merge.is_empty_constraint:
                continue
            new_constraints.append(to_merge.constraint)
        if new_constraints:
            if self.is_empty_constraint:
                if len(new_constraints) == 1:
                    self.constraint = new_constraints[0]
                else:
                    self.constraint = StructureDisjunction(location_id, tuple(new_constraints))
            else:
                new_constraints = [self.constraint] + new_constraints
                self.constraint = StructureDisjunction(location_id, tuple(new_constraints))
