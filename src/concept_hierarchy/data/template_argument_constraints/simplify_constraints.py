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

from concept_hierarchy.data.template_argument_constraints.constraint_formula import (
    NonTypeTemplateConstraintFormula,
    StructureConjunction,
    StructureDisjunction,
    StructureNegation,
    TemplateConstraintAnd,
    TemplateConstraintNot,
    TemplateConstraintOr,
    TypeTemplateConstraintFormula,
)
from concept_hierarchy.errors import LocationId
from concept_hierarchy.utils import remove_indices

from .constraint_formula import (
    ConstraintGroup,
    Empty,
    NonStructureConstraintFormula,
    StructureConstraintFormula,
    TemplateConstraintFormula,
    Unconstrained,
)


def create_empty_structure_constraint(
    nr_variables: int, constraint_types: tuple[str, ...], location_id: LocationId
) -> ConstraintGroup | None:
    if nr_variables == 0:
        return None
    assert nr_variables == len(constraint_types)
    return ConstraintGroup(
        location_id, tuple(Empty(location_id, constraint_type) for constraint_type in constraint_types)
    )


def create_unconstrained_structure_constraint(nr_variables: int, location_id: LocationId) -> ConstraintGroup | None:
    if nr_variables == 0:
        return None
    return ConstraintGroup(location_id, tuple(Unconstrained(location_id) for _ in range(nr_variables)))


def simplify_formula(f: TemplateConstraintFormula) -> TemplateConstraintFormula:
    """
    Entry point for constraint formula simplification.

    Dispatches to the appropriate simplifier based on whether ``f`` is a
    non-structure (literal / type) formula or a structure formula (i.e.
    joint constraints on multiple template variables represented as a tuple
    of per-slot constraints).

    The function applies a fixed set of algebraic identities (idempotence,
    identity elements, annihilation, complement, double negation, flattening).
    Absorption, distributivity, and deMorgan laws are not applied.

    Parameters
    ----------
    f:
        The constraint formula to simplify.
        Must be either a ``NonStructureConstraintFormula`` or a
        ``StructureConstraintFormula``.

    Returns
    -------
    TemplateConstraintFormula
        A logically equivalent formula in a simplified form.
    """
    if isinstance(f, NonStructureConstraintFormula):
        return simplify_non_structure_constraint(f)
    assert isinstance(f, StructureConstraintFormula)
    return simplify_structure_constraint(f)


def _collect_different_data(data: tuple) -> tuple:
    to_remove_indices: set[int] = set()
    for index, f_element in enumerate(data[:-1]):
        if index in to_remove_indices:
            continue
        for index_to_remove, following_element in enumerate(data[index + 1 :]):
            if f_element == following_element:
                to_remove_indices.add(index + 1 + index_to_remove)
    if len(to_remove_indices) > 0:
        return remove_indices(data, to_remove_indices)
    return data


def _structure_elements_contains_a_and_neg_a(elements: tuple[StructureConstraintFormula, ...]) -> bool:
    for element in elements:
        if isinstance(element, StructureNegation):
            continue
        for other_element in elements:
            if not isinstance(other_element, StructureNegation):
                continue
            if element == other_element.structure_constraint:
                return True
    return False


def _formula_contains_a_and_not_a(sub_formulae: tuple[NonStructureConstraintFormula, ...]) -> bool:
    for formula in sub_formulae:
        if isinstance(formula, TemplateConstraintNot):
            continue
        for other_formula in sub_formulae:
            if not isinstance(other_formula, TemplateConstraintNot):
                continue
            if formula == other_formula.sub_formula:
                return True
    return False


def simplify_structure_constraint(f: StructureConstraintFormula | None) -> StructureConstraintFormula | None:
    """
    Simplify a structure template constraint formula.

    Structure constraints describe tuples of per-slot constraints (represented
    as ``ConstraintGroup``) and can be combined with ``StructureConjunction``,
    ``StructureDisjunction``, and ``StructureNegation``.  This function applies
    the same lattice identities as ``simplify_non_structure_constraint`` at the
    structure level, and additionally handles the reduction of conjunctions of
    ``ConstraintGroup`` nodes into a single merged group.

    The following rules are applied in order:

    1. **None passthrough** – ``None`` is returned immediately, representing the
       absence of a structure constraint (zero-variable concepts).

    2. **Terminal normalisation** – If the formula evaluates to ``is_empty`` or
       ``is_unconstrained`` but is not already a canonical ``ConstraintGroup``
       sentinel, it is replaced with the appropriate canonical form via
       ``create_empty_structure_constraint`` or
       ``create_unconstrained_structure_constraint``.  These properties recurse
       through children, so e.g. ``Neg(Empty)`` and ``Neg(Unconstrained)`` are
       caught here.

    3. **Recursive child simplification** (``StructureConjunction`` /
       ``StructureDisjunction``) – Each child structure constraint is simplified
       independently.  If any child changed (by object identity), a new
       composite node is constructed and the function restarts from the top.

    4. **Flattening** – Nested associative nodes of the same kind are merged:
       ``Conj(Conj(A, B), C) → Conj(A, B, C)``,
       ``Disj(Disj(A, B), C) → Disj(A, B, C)``.
       Only triggers a restart when at least one nested node was actually
       expanded.

    5. **Identity element removal** – Children that contribute nothing to the
       result are filtered out:

       * ``is_unconstrained`` children are dropped from ``StructureConjunction``
         (the fully-unconstrained group is the identity for structure
         intersection);
       * ``is_empty`` children are dropped from ``StructureDisjunction``
         (the empty group is the identity for structure union).

       If *all* children are filtered, the result is the identity value itself.
       Triggers a restart when any child was removed.

    6. **Deduplication** – Duplicate children in a ``StructureConjunction`` or
       ``StructureDisjunction`` are removed, exploiting idempotence.  Triggers
       a restart when any duplicate was removed.

    7. **Singleton unwrapping** – ``StructureConjunction(X)`` and
       ``StructureDisjunction(X)`` are replaced by ``X``.

    8. **Contradiction / tautology elimination** –
       ``Conj(A, Neg(A)) → Empty``  (for any ordering),
       ``Disj(A, Neg(A)) → Unconstrained`` (for any ordering).

    9. **Double-negation elimination** – ``Neg(Neg(X)) → X``.

    10. **Single-nonUnconstrained-negation-propagation** -
        ``Neg(<, , X, , >)``   =>   ``<, , Not(X), , >``
        A single non-unconstrained value can be consumed by negation.

    11. **ConstraintGroup merging** – A ``StructureConjunction`` whose *all*
        children are ``ConstraintGroup`` nodes is converted into a single
        ``ConstraintGroup`` with element-wise ``And`` constraints:
        ``Conj(<X, Y>, <Z, W>) → <And(X, Z), And(Y, W)>``.
        This rule fires only when the two template-variable slots are independent,
        which is structurally guaranteed when all children are plain
        ``ConstraintGroup`` nodes rather than shared-variable
        ``StructureConjunction`` subtrees.

    12. **Intra-group simplification** – The per-slot constraints inside a
        ``ConstraintGroup`` are each passed through
        ``simplify_non_structure_constraint``.

    Parameters
    ----------
    f:
        The structure formula to simplify, or ``None`` for zero-variable types.

    Returns
    -------
    StructureConstraintFormula | None
        A logically equivalent, simplified structure formula, or ``None`` if
        the input was ``None``.  The returned object may be the same object as
        ``f``, a canonical ``ConstraintGroup`` sentinel, or a newly constructed
        node.

    Notes
    -----
    *Rule 10 is intentionally placed after rules 3–9.*  Child simplification
    (rule 3) may reduce a ``StructureConjunction`` child to a ``ConstraintGroup``
    (e.g. by firing rule 10 recursively), which then makes the parent eligible
    for rule 10 on the restarted call.

    *The dual of rule 10 for ``StructureDisjunction``* — reducing
    ``Disj(<X, Y>, <Z, W>) → <Or(X, Z), Or(Y, W)>`` — is not a valid
    simplification and must not be added.  A ``ConstraintGroup`` imposes a
    joint conjunction across its slots, so ``Disj(<X, Y>, <Z, W>)`` expands by
    distributivity to four conjuncts including the cross-slot terms
    ``(X ∨ W)`` and ``(Y ∨ Z)`` that cannot be represented inside a flat
    ``ConstraintGroup``.  The element-wise ``Or`` would silently drop those
    cross-terms, yielding an over-approximation.
    """
    if f is None:
        return None
    if f.is_empty:
        if not isinstance(f, ConstraintGroup):
            return create_empty_structure_constraint(f.nr_variables, f.variable_constraint_types, f.location_id)
        return f
    elif f.is_unconstrained:
        if not isinstance(f, ConstraintGroup):
            return create_unconstrained_structure_constraint(f.nr_variables, f.location_id)
        return f

    # Rule: simplify composite constraints first
    if isinstance(f, StructureConjunction):
        new_constraints: list[StructureConstraintFormula] = []
        has_simplification = False
        for constraint in f.structure_constraints:
            new_constraint = simplify_structure_constraint(constraint)
            new_constraints.append(new_constraint)
            has_simplification |= new_constraint is not constraint  # object-id comparison, not equality
        if has_simplification:
            return simplify_structure_constraint(StructureConjunction(f.location_id, tuple(new_constraints)))
    elif isinstance(f, StructureDisjunction):
        new_constraints: list[StructureConstraintFormula] = []
        has_simplification = False
        for constraint in f.structure_constraints:
            new_constraint = simplify_structure_constraint(constraint)
            new_constraints.append(new_constraint)
            has_simplification |= new_constraint is not constraint  # object-id comparison, not equality
        if has_simplification:
            return simplify_structure_constraint(StructureDisjunction(f.location_id, tuple(new_constraints)))

    # Rule: expand conjunctions inside conjunctions and disjunctions inside disjunctions
    if isinstance(f, StructureConjunction):
        new_conjunctions: list[StructureConstraintFormula] = []
        expanded = False
        for index, f_element in enumerate(f.structure_constraints):
            if isinstance(f_element, StructureConjunction):
                new_conjunctions.extend(f_element.structure_constraints)
                expanded = True
            else:
                new_conjunctions.append(f_element)
        if expanded:
            return simplify_structure_constraint(StructureConjunction(f.location_id, tuple(new_conjunctions)))
    elif isinstance(f, StructureDisjunction):
        new_disjunctions: list[StructureConstraintFormula] = []
        expanded = False
        for index, f_element in enumerate(f.structure_constraints):
            if isinstance(f_element, StructureDisjunction):
                new_disjunctions.extend(f_element.structure_constraints)
                expanded = True
            else:
                new_disjunctions.append(f_element)
        if expanded:
            return simplify_structure_constraint(StructureDisjunction(f.location_id, tuple(new_disjunctions)))

    # Rule: type-unconstrained formulae in And can be removed and empty formulae in Or can be removed
    if isinstance(f, StructureConjunction):
        new_constraints: list[StructureConstraintFormula] = []
        removed = False
        for constraint in f.structure_constraints:
            if constraint.is_unconstrained:
                removed = True
            else:
                new_constraints.append(constraint)
        if removed:
            if new_constraints:
                return simplify_structure_constraint(StructureConjunction(f.location_id, tuple(new_constraints)))
            else:
                return create_unconstrained_structure_constraint(f.nr_variables, f.location_id)
    elif isinstance(f, StructureDisjunction):
        new_constraints: list[StructureConstraintFormula] = []
        removed = False
        for constraint in f.structure_constraints:
            if constraint.is_empty:
                removed = True
            else:
                new_constraints.append(constraint)
        if removed:
            if new_constraints:
                return simplify_structure_constraint(StructureDisjunction(f.location_id, tuple(new_constraints)))
            else:
                return create_empty_structure_constraint(f.nr_variables, f.variable_constraint_types, f.location_id)

    # Rule: Equal terms in conjunctions and disjunctions can be discarded
    if isinstance(f, StructureConjunction):
        new_structure_constraints = _collect_different_data(f.structure_constraints)
        if new_structure_constraints is not f.structure_constraints:
            return simplify_structure_constraint(StructureConjunction(f.location_id, new_structure_constraints))
    elif isinstance(f, StructureDisjunction):
        new_structure_constraints = _collect_different_data(f.structure_constraints)
        if new_structure_constraints is not f.structure_constraints:
            return simplify_structure_constraint(StructureDisjunction(f.location_id, new_structure_constraints))

    # Rule: expand single-entry conjunctions and disjunctions
    if isinstance(f, StructureConjunction):
        if len(f.structure_constraints) == 1:
            return simplify_structure_constraint(f.structure_constraints[0])
    elif isinstance(f, StructureDisjunction):
        if len(f.structure_constraints) == 1:
            return simplify_structure_constraint(f.structure_constraints[0])

    # Rule: conj (a , neg(a)) is empty (consider possibly conj (neg(a) , a) order)
    if isinstance(f, StructureConjunction):
        if _structure_elements_contains_a_and_neg_a(f.structure_constraints):
            return create_empty_structure_constraint(f.nr_variables, f.variable_constraint_types, f.location_id)

    # Rule: disj (a , neg(a)) is unconstrained (consider possibly disj (neg(a) , a) order)
    if isinstance(f, StructureDisjunction):
        if _structure_elements_contains_a_and_neg_a(f.structure_constraints):
            return create_unconstrained_structure_constraint(f.nr_variables, f.location_id)

    # Rule: consume neg(neg(...))
    if isinstance(f, StructureNegation) and isinstance(f.structure_constraint, StructureNegation):
        return simplify_structure_constraint(f.structure_constraint.structure_constraint)

    # Rule: Neg(<, , X, , >) -> <, , Not(X), , >  single non-unconstrained value can be consumed by negation
    """
    Full negation distribution rule is:
    Neg(<C1, C2, C3, ...>) = Disj(<Not(C1), C2, C3, ...>, <C1, Not(C2), C3, ...>, <C1, C2, Not(C3), ...>)
    When C1 is unconstrained (for example), <Not(C1), C2, C3, ...> = <Empty, C2, C3, ...> = <Empty, Empty, Empty, ...>
    Thus, unconstrained group constraints turn to Empty-Group, which is ignored in a disjunction.
    Thus, if only one entry is not unconstrained, all the other disjuncts will be empty in the disjunction (and ignored)
     and the only remaining disjunct will be the negation/not of the non-unconstrained-entry.
    """
    if isinstance(f, StructureNegation) and isinstance(f.structure_constraint, ConstraintGroup):
        non_unconstrained_indices: list[int] = []
        for index, constraint in enumerate(f.structure_constraint.group_constraints):
            if not constraint.is_type_unconstrained:
                non_unconstrained_indices.append(index)
        if len(non_unconstrained_indices) == 1:
            new_constraints: tuple[NonStructureConstraintFormula, ...] = tuple(
                (x if index not in non_unconstrained_indices else TemplateConstraintNot(f.location_id, x))
                for index, x in enumerate(f.structure_constraint.group_constraints)
            )
            return simplify_structure_constraint(ConstraintGroup(f.location_id, new_constraints))

    # Rule??: StructureConjunction with only ConstraintGroups is a ConstraintGroup with element-wise AND constraints
    if isinstance(f, StructureConjunction) and all(isinstance(x, ConstraintGroup) for x in f.structure_constraints):
        element_wise_new_constraints: list[list[NonStructureConstraintFormula]] = [[] for _ in range(f.nr_variables)]
        for element in f.structure_constraints:
            assert isinstance(element, ConstraintGroup)
            for index, element_constraint in enumerate(element.group_constraints):
                element_wise_new_constraints[index].append(element_constraint)
        return simplify_structure_constraint(
            ConstraintGroup(
                f.location_id,
                tuple(TemplateConstraintAnd(f.location_id, tuple(x)) for x in element_wise_new_constraints),
            )
        )

    # Rule: simplify inside a ConstraintGroup
    if isinstance(f, ConstraintGroup):
        new_group_constraints: list[NonStructureConstraintFormula] = []
        has_simplification = False
        for group_constraint in f.group_constraints:
            new_group_constraint = simplify_non_structure_constraint(group_constraint)
            new_group_constraints.append(new_group_constraint)
            has_simplification |= new_group_constraint is not group_constraint
        if has_simplification:
            return ConstraintGroup(f.location_id, tuple(new_group_constraints))
    return f


def simplify_non_structure_constraint(f: NonStructureConstraintFormula) -> NonStructureConstraintFormula:
    """
    Simplify a non-structure (literal / type) template constraint formula.

    Applies a fixed set of algebraic identities in a bottom-up pass, restarting
    the pass whenever a structural change occurs.  The function is not guaranteed
    to produce a *minimal* formula, but it is guaranteed to terminate and to
    return a formula that is logically equivalent to the input under the CH
    constraint lattice.

    The function applies a fixed set of algebraic identities (idempotence,
    identity elements, annihilation, complement, double negation, flattening).
    Absorption, distributivity, and deMorgan laws are not applied.

    The following rules are applied in order:

    1. **Terminal normalisation** – Three mutually exclusive terminal states are
       recognised and, if reached by a non-canonical node, replaced with their
       canonical sentinel:

       * ``is_empty`` → ``Empty(location_id, constraint_type)``
       * ``is_unconstrained`` → ``Unconstrained(location_id)``
       * ``is_type_unconstrained`` → ``TypeTemplateConstraintFormula.any_type(location_id)``
         when ``constraint_type == "type"``, otherwise
         ``NonTypeTemplateConstraintFormula(constraint_type, location_id)``.

       This ensures that downstream code can rely on ``isinstance`` checks rather
       than the property alone.  Because ``is_empty`` and ``is_unconstrained``
       are checked first, no ``is_type_unconstrained`` node will be ``is_empty``
       or ``is_unconstrained`` by the time rule 1c fires.

    2. **Recursive child simplification** (``And`` / ``Or``) – Each sub-formula
       is simplified independently.  If any child changed (by object identity),
       a new composite node is constructed and the function restarts from the
       top so that subsequent rules see the simplified children.

    3. **Flattening** – Nested associative nodes of the same kind are merged into
       a single flat node:
       ``And(And(A, B), C) → And(A, B, C)``,
       ``Or(Or(A, B), C)  → Or(A, B, C)``.
       Only triggers a restart when at least one nested node was actually
       expanded; a no-change pass falls through.

    4. **Identity element removal** – Children that contribute nothing to the
       result are filtered out:

       * ``is_type_unconstrained`` children are dropped from ``And``
         (the whole-domain constraint is the identity for intersection);
       * ``is_empty`` children are dropped from ``Or``
         (the empty constraint is the identity for union).

       If *all* children are filtered (e.g. ``And(Unconstrained_type,
       Unconstrained_type)``), the result is the identity value itself.
       Triggers a restart when any child was removed.

    5. **Deduplication** – Duplicate children inside ``And`` or ``Or`` are
       removed (``And(X, X, Y) → And(X, Y)``), exploiting idempotence.
       Triggers a restart when any duplicate was removed.

    6. **Singleton unwrapping** – ``And(X)`` and ``Or(X)`` are replaced by
       ``X``, removing redundant wrappers.  This rule fires after deduplication
       and identity removal, so a restart from one of those rules will naturally
       reduce a formerly multi-child node to a singleton before this rule is
       reached on the restarted call.

    7. **Contradiction / tautology elimination** –
       ``And(A, Not(A)) → Empty`` (for any ordering of ``A`` and ``Not(A)``),
       ``Or(A, Not(A))  → TypeTemplateConstraintFormula.any_type(location_id)``
       when ``constraint_type == "type"``, otherwise
       ``NonTypeTemplateConstraintFormula(constraint_type, location_id)``.

    8. **Double-negation elimination** – ``Not(Not(X)) → X``.

    Parameters
    ----------
    f:
        The non-structure formula to simplify.

    Returns
    -------
    NonStructureConstraintFormula
        A logically equivalent, simplified formula.  The returned object may be
        the same object as ``f`` (when no rule fired), a canonical sentinel
        (``Empty`` / ``Unconstrained`` / type-unconstrained), or a newly
        constructed node.

    Notes
    -----
    *Rule ordering matters.*  Flattening (rule 3) must precede contradiction
    detection (rule 7) because a contradiction such as
    ``And(And(A, B), Not(A))`` is only detected after flattening produces the
    flat ``And(A, B, Not(A))``.  Similarly, identity removal (rule 4) and
    deduplication (rule 5) must precede singleton unwrapping (rule 6) because
    they are the steps that can reduce a multi-child node to a single child.

    *Object-identity change detection* (``new_constraint is not constraint``)
    in rule 2 is intentional: a newly constructed node that is structurally
    equal to the original still forces a restart, because its *children* may
    have been replaced and subsequent rules must see the updated tree.

    *Partial negation* – Not all constraint types have a representable
    complement.  For literal value constraints (e.g. the integer ``3``),
    ``Not(3)`` is left as an opaque ``TemplateConstraintNot`` node; no attempt
    is made to propagate the negation inward.  Rule 7 therefore only fires when
    an exact syntactic complement ``Not(A)`` of a sibling ``A`` is present in
    the same ``And`` / ``Or``.

    *``is_unconstrained`` vs ``is_type_unconstrained``* – ``Unconstrained``
    (rule 1b) means the slot carries no constraint whatsoever — it is not even
    bound to a specific domain.  ``is_type_unconstrained`` (rules 1c and 4)
    means the domain is fixed (e.g. ``"type"`` or ``"literal:int"``) but the
    constraint covers the whole domain.  The two are checked in the order
    1b → 1c so that a fully unconstrained node is never mistakenly normalised
    as a type-unconstrained one.
    """
    if f.is_empty:
        if not isinstance(f, Empty):
            return Empty(f.location_id, f.constraint_type)
        return f
    if f.is_unconstrained:
        if not isinstance(f, Unconstrained):
            return Unconstrained(f.location_id)
        return f
    if f.is_type_unconstrained:
        any_type = TypeTemplateConstraintFormula.any_type(f.location_id)
        if f.constraint_type == "type":
            if f != any_type:
                return any_type
        elif not isinstance(f, NonTypeTemplateConstraintFormula):
            return NonTypeTemplateConstraintFormula(f.constraint_type, f.location_id)
        return f

    # Rule: simplify composite constraints first
    if isinstance(f, TemplateConstraintAnd):
        new_constraints: list[NonStructureConstraintFormula] = []
        has_simplification = False
        for constraint in f.sub_formulae:
            new_constraint = simplify_non_structure_constraint(constraint)
            new_constraints.append(new_constraint)
            has_simplification |= new_constraint is not constraint  # object-id comparison, not equality
        if has_simplification:
            return simplify_non_structure_constraint(TemplateConstraintAnd(f.location_id, tuple(new_constraints)))
    elif isinstance(f, TemplateConstraintOr):
        new_constraints: list[NonStructureConstraintFormula] = []
        has_simplification = False
        for constraint in f.sub_formulae:
            new_constraint = simplify_non_structure_constraint(constraint)
            new_constraints.append(new_constraint)
            has_simplification |= new_constraint is not constraint  # object-id comparison, not equality
        if has_simplification:
            return simplify_non_structure_constraint(TemplateConstraintOr(f.location_id, tuple(new_constraints)))

    # Rule: expand and-formulae inside and-formula and or-formulae inside or-formula
    if isinstance(f, TemplateConstraintAnd):
        new_sub_formulae: list[NonStructureConstraintFormula] = []
        expanded = False
        for index, sub_f in enumerate(f.sub_formulae):
            if isinstance(sub_f, TemplateConstraintAnd):
                new_sub_formulae.extend(sub_f.sub_formulae)
                expanded = True
            else:
                new_sub_formulae.append(sub_f)
        if expanded:
            return simplify_non_structure_constraint(TemplateConstraintAnd(f.location_id, tuple(new_sub_formulae)))
    elif isinstance(f, TemplateConstraintOr):
        new_sub_formulae: list[NonStructureConstraintFormula] = []
        expanded = False
        for index, sub_f in enumerate(f.sub_formulae):
            if isinstance(sub_f, TemplateConstraintOr):
                new_sub_formulae.extend(sub_f.sub_formulae)
                expanded = True
            else:
                new_sub_formulae.append(sub_f)
        if expanded:
            return simplify_non_structure_constraint(TemplateConstraintOr(f.location_id, tuple(new_sub_formulae)))

    # Rule: type-unconstrained formulae in And can be removed and empty formulae in Or can be removed
    if isinstance(f, TemplateConstraintAnd):
        new_constraints: list[NonStructureConstraintFormula] = []
        removed = False
        for constraint in f.sub_formulae:
            if constraint.is_type_unconstrained:
                removed = True
            else:
                new_constraints.append(constraint)
        if removed:
            if new_constraints:
                return simplify_non_structure_constraint(TemplateConstraintAnd(f.location_id, tuple(new_constraints)))
            elif f.constraint_type == "type":
                return TypeTemplateConstraintFormula.any_type(f.location_id)
            else:
                return NonTypeTemplateConstraintFormula(f.constraint_type, f.location_id)
    elif isinstance(f, TemplateConstraintOr):
        new_constraints: list[NonStructureConstraintFormula] = []
        removed = False
        for constraint in f.sub_formulae:
            if constraint.is_empty:
                removed = True
            else:
                new_constraints.append(constraint)
        if removed:
            if new_constraints:
                return simplify_non_structure_constraint(TemplateConstraintOr(f.location_id, tuple(new_constraints)))
            else:
                return Empty(f.location_id, f.constraint_type)

    # Rule: Equal terms in and- and or-formulae can be discarded
    if isinstance(f, TemplateConstraintAnd):
        new_sub_formulae: tuple[NonStructureConstraintFormula, ...] = _collect_different_data(f.sub_formulae)
        if new_sub_formulae is not f.sub_formulae:
            return simplify_non_structure_constraint(TemplateConstraintAnd(f.location_id, new_sub_formulae))
    elif isinstance(f, TemplateConstraintOr):
        new_sub_formulae: tuple[NonStructureConstraintFormula, ...] = _collect_different_data(f.sub_formulae)
        if new_sub_formulae is not f.sub_formulae:
            return simplify_non_structure_constraint(TemplateConstraintOr(f.location_id, new_sub_formulae))

    # Rule: expand single-entry and- and or-formulae
    if isinstance(f, TemplateConstraintAnd):
        if len(f.sub_formulae) == 1:
            return simplify_non_structure_constraint(f.sub_formulae[0])
    elif isinstance(f, TemplateConstraintOr):
        if len(f.sub_formulae) == 1:
            return simplify_non_structure_constraint(f.sub_formulae[0])

    # Rule: and (a , not(a)) is empty (consider possibly and (not(a) , a) order)
    if isinstance(f, TemplateConstraintAnd):
        if _formula_contains_a_and_not_a(f.sub_formulae):
            return Empty(f.location_id, f.constraint_type)

    # Rule: or (a , not(a)) is unconstrained (consider possibly or (not(a) , a) order)
    if isinstance(f, TemplateConstraintOr):
        if _formula_contains_a_and_not_a(f.sub_formulae):
            # differentiate between unconstrained types and unconstrained literal types!
            if f.constraint_type == "type":
                return TypeTemplateConstraintFormula.any_type(f.location_id)
            return NonTypeTemplateConstraintFormula(f.constraint_type, f.location_id)

    # Rule: consume not(not(...))
    if isinstance(f, TemplateConstraintNot) and isinstance(f.sub_formula, TemplateConstraintNot):
        return simplify_non_structure_constraint(f.sub_formula.sub_formula)

    return f
