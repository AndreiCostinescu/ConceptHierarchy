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

from abc import ABC, abstractmethod
from enum import Enum

from concept_hierarchy.errors import CHSemanticError, LocationId
from concept_hierarchy.utils import Reference, is_integer, is_number


class TemplateConstraintFormulaValidator(ABC):
    """
    Abstract interface that exposes concept-hierarchy knowledge to formula classes.

    An instance is passed to every ``TemplateConstraintHierarchyOperator`` at
    construction time so that semantic validation can happen immediately, before
    any further compiler phases run.  The validator is responsible for three
    checks carried out in the operator's ``__init__``:

    1. The referenced name is known as either a concept or a template variable.
    2. Template variables are never given explicit template-argument constraints.
    3. If explicit template-argument constraints are given for a concept, their
       count matches the concept's declared number of template parameters exactly.

    Implementors must supply:

    * ``is_concept`` / ``is_template_variable`` — mutually exclusive membership
      tests that classify a raw name string.
    * ``get_nr_template_arguments`` — the number of template parameters declared
      by a concept (0 for non-parameterised leaf concepts).
    * ``full_type_name`` — the canonical display name with template parameters
      filled in, used in error messages (e.g. ``"Map<K, V>"``).
    """

    @abstractmethod
    def full_type_name(self, name: str) -> str:
        pass

    @abstractmethod
    def get_nr_template_arguments(self, concept_name: str) -> int:
        pass

    @abstractmethod
    def is_concept(self, name: str) -> bool:
        pass

    @abstractmethod
    def is_template_variable(self, name: str):
        pass


"""
TemplateConstraintFormulae are either:
- NonStructureConstraintFormula:
    - Unconstrained (both for types and non-types)
    - TypeTemplateConstraintFormula
        - And, Or, Not
        - HierarchyOperators
            - X, X., X*, ^X, ^X*
    - NonTypeTemplateConstraintFormula (\equiv LiteralConstraintFormula)
        - LiteralValueConstraintFormula
- StructureConstraintFormula
    - ConstraintGroup
    - StructureConjunction
    - StructureDisjunction
    - StructureNegation
"""


class TemplateConstraintFormula(ABC):
    """
    Root abstract base class for every template-argument constraint formula.

    Every formula in the tree carries the ``location_id`` of the source
    location where it was written, so that semantic errors produced during
    later compiler phases can be reported with precise source context.

    The two disjoint sub-hierarchies are:

    * ``NonStructureConstraintFormula`` — restricts what a *single* slot may
      hold (a type, a literal value, or nothing).
    * ``StructureConstraintFormula`` — expresses constraints on the *relationship
      between* multiple template-argument slots simultaneously.
    """

    def __init__(self, location_id: LocationId):
        self.location_id = location_id

    def __str__(self):
        return self.__repr__()


class NonStructureConstraintFormula(TemplateConstraintFormula):
    """
    A constraint that governs a single template-argument slot in isolation.

    Non-structure constraints answer the question "what is an acceptable value
    for *this one* slot?", without reference to any other slot.  They form the
    operand type for ``And``, ``Or``, and ``Not``, and they appear as the
    per-slot entries inside a ``ConstraintGroup``.

    The three concrete families are:

    * ``Unconstrained`` — no restriction; any value is accepted.
    * ``TypeTemplateConstraintFormula`` — restricts the slot to a type from the
      concept hierarchy.
    * ``NonTypeTemplateConstraintFormula`` — restricts the slot to a primitive
      literal value domain (int, float, bool, or string).
    """

    pass


class Unconstrained(NonStructureConstraintFormula):
    """
    The vacuously true constraint: any type or literal value satisfies it.

    Most importantly, ``Not(Unconstrained)`` means that no value satisfies it.

    ``Unconstrained`` is equivalent to a ``Concept`` constraint, which accepts
      any type.
      The constraint ``Not(Concept)`` means that no value can satisfy it.

    ``Unconstrained`` is used wherever a slot is left deliberately unrestricted.
    It arises in two situations:

    * Explicitly, when a concept is referenced with empty angle brackets, e.g.
      ``Vector<>`` — the single slot of ``Vector`` receives an ``Unconstrained``
      formula, meaning any instantiation of that slot is accepted.
    * Implicitly, as the result of parsing an absent sub-expression, e.g. the
      body of ``And()`` or a slot delimiter ``>`` or ``,`` in the input stream.

    Its ``repr`` is the empty string, so it is invisible in serialised output.
    """

    def __init__(self, location_id: LocationId):
        super().__init__(location_id)

    def __repr__(self):
        return ""


class TypeTemplateConstraintFormula(NonStructureConstraintFormula, ABC):
    """
    A non-structure constraint that restricts a slot to a *type* in the
    concept hierarchy, as opposed to a primitive literal value.

    All ``And``, ``Or``, ``Not``, and hierarchy-operator constraints belong to
    this family.  The ``And`` and ``Or`` constructors enforce that every operand
    is a ``TypeTemplateConstraintFormula``; mixing type constraints with literal
    values or ``Unconstrained`` inside a boolean operator is a runtime error.
    """

    pass


class TemplateConstraintAnd(TypeTemplateConstraintFormula):
    """
    Type-constraint intersection: the template argument must satisfy *all*
    sub-formulae simultaneously.  Written ``And(F₁, F₂, …)``.

    Every operand must be a ``TypeTemplateConstraintFormula``; passing
    ``Unconstrained`` or a ``NonTypeTemplateConstraintFormula`` raises a
    ``RuntimeError`` at construction time.  At least one operand is required.

    Example: ``And(Animal, Not(Predator))`` matches any concrete type that is
    both a descendant of ``Animal`` and not a descendant of ``Predator``.
    """

    def __init__(self, location_id: LocationId, sub_formulae: tuple[NonStructureConstraintFormula, ...]):
        super().__init__(location_id)
        # make tuple to be immutable
        assert sub_formulae
        self.sub_formulae: tuple[NonStructureConstraintFormula, ...] = sub_formulae
        for f in self.sub_formulae:
            if not isinstance(f, TypeTemplateConstraintFormula):
                raise RuntimeError(
                    f"At {self.location_id}, the formula {f} is not a TypeTemplateConstraintFormula, "
                    "which is required by TemplateConstraintAnd!"
                )

    def __repr__(self):
        return "And(" + ", ".join([repr(f) for f in self.sub_formulae]) + ")"


class TemplateConstraintOr(TypeTemplateConstraintFormula):
    """
    Type-constraint union: the template argument must satisfy *at least one*
    sub-formula.  Written ``Or(F₁, F₂, …)``.

    Every operand must be a ``TypeTemplateConstraintFormula``; the same
    restrictions and error behaviour as ``TemplateConstraintAnd`` apply.
    At least one operand is required.

    Example: ``Or(Mammal, Bird)`` matches any concrete type that is a
    descendant of either ``Mammal`` or ``Bird``.
    """

    def __init__(self, location_id: LocationId, sub_formulae: tuple[NonStructureConstraintFormula, ...]):
        super().__init__(location_id)
        # make tuple to be immutable
        assert sub_formulae
        self.sub_formulae: tuple[NonStructureConstraintFormula, ...] = sub_formulae
        for f in self.sub_formulae:
            if not isinstance(f, TypeTemplateConstraintFormula):
                raise RuntimeError(
                    f"At {self.location_id}, the formula {f} is not a TypeTemplateConstraintFormula, "
                    "which is required by TemplateConstraintOr!"
                )

    def __repr__(self):
        return "Or(" + ", ".join([repr(f) for f in self.sub_formulae]) + ")"


class TemplateConstraintNot(TypeTemplateConstraintFormula):
    """
    Type-constraint complement: the template argument must *not* satisfy the
    sub-formula.  Written ``Not(F)``.

    The single operand must be a ``TypeTemplateConstraintFormula``; passing
    ``Unconstrained`` or a ``NonTypeTemplateConstraintFormula`` raises a
    ``RuntimeError`` at construction time.

    Example: ``Not(Abstract)`` matches any concrete type that is not a
    descendant of the abstract concept ``Abstract``.
    """

    def __init__(self, sub_formula: TemplateConstraintFormula, location_id: LocationId):
        super().__init__(location_id)
        self.sub_formula: TemplateConstraintFormula = sub_formula
        if not isinstance(self.sub_formula, TypeTemplateConstraintFormula):
            raise RuntimeError(
                f"At {self.location_id}, the formula {self.sub_formula} is not a TypeTemplateConstraintFormula, "
                "which is required by TemplateConstraintNot!"
            )

    def __repr__(self):
        return "Not(" + repr(self.sub_formula) + ")"


class HierarchyCheckType(Enum):
    """
    Identifies which positional relationship to the named concept a
    ``TemplateConstraintHierarchyOperator`` tests.

    Members
    -------
    DESCENDANTS_OF
        Matches concrete (non-abstract) subtypes of the literal, excluding the
        literal itself unless it is concrete.  Syntax: ``T``.
    ABSTRACT_DESCENDANTS_OF
        Matches all subtypes of the literal, whether abstract or concrete,
        including the literal itself.  Syntax: ``T*``.
    SELF
        Matches exactly the named concept — no subtypes or supertypes.
        Syntax: ``T.``.
    ASCENDANTS_OF
        Matches concrete (non-abstract) supertypes of the literal, excluding
        the literal itself unless it is concrete.  Syntax: ``^T``.
    ABSTRACT_ASCENDANTS_OF
        Matches all supertypes of the literal, whether abstract or concrete,
        including the literal itself.  Syntax: ``^T*``.
    """

    DESCENDANTS_OF = (0,)
    ABSTRACT_DESCENDANTS_OF = (1,)
    ASCENDANTS_OF = (2,)
    ABSTRACT_ASCENDANTS_OF = (3,)
    SELF = 4


# The literal is just the concept name: context.ch.is_concept(self.literal)
# If the concept is a template ValueDomain, and it doesn't have any template restrictions,
#  then it accepts all template-instantiations
class TemplateConstraintHierarchyOperator(TypeTemplateConstraintFormula, ABC):
    """
    Abstract base for the five hierarchy-positional type constraints, each of
    which names a specific concept (or template variable) as its reference point
    and selects an up- or downward region of the hierarchy relative to it.

    The ``literal`` attribute holds the bare name of the reference concept or
    template variable.  The optional ``literal_template_formulae`` tuple
    further constrains the reference concept's own template arguments when it is
    itself parameterised (e.g. ``Collection<Animal>`` constrains the element
    type).  When no angle brackets are written, ``literal_template_formulae``
    is the empty tuple, meaning the reference concept is accepted in any of its
    instantiations.

    Semantic validation is performed eagerly at construction time via the
    supplied ``validator``:

    * The ``literal`` must be recognised as either a concept or a template
      variable; anything else raises ``CHSemanticError``.
    * Template variables may not carry template-argument constraints, because a
      template variable stands for an unknown concept whose own template
      signature is not known at constraint-definition time.
    * For parameterised concepts, the number of entries in
      ``literal_template_formulae`` must equal the concept's declared parameter
      count exactly; a mismatch raises ``CHSemanticError``.

    Concrete subclasses select the hierarchy direction via ``HierarchyCheckType``
    and provide the appropriate ``repr``.
    """

    def __init__(
        self,
        literal: str,
        literal_template_formulae: tuple[NonStructureConstraintFormula, ...],
        validator: TemplateConstraintFormulaValidator,
        location_id: LocationId,
        hierarchy_op: HierarchyCheckType,
    ):
        super().__init__(location_id)
        self.literal = literal
        self.literal_template_formulae: tuple[TemplateConstraintFormula, ...] = literal_template_formulae
        self.hierarchy_op: HierarchyCheckType = hierarchy_op

        # validate self.literal type
        if not validator.is_template_variable(self.literal) and not validator.is_concept(self.literal):
            raise CHSemanticError(
                f"{self.literal} is not a concept and not a template variable!",
                location_id=self.location_id,
            )

        # don't allow constraints like "T<ValueDomain>" where T is a template variable!
        if validator.is_template_variable(self.literal) and self.has_specification_of_template_constraints:
            raise CHSemanticError(
                "Can not define a constraint literal value that is a template variable ({0}) and also "
                "specify constraints on template arguments: {0}<{1}>".format(
                    self.literal, ", ".join(str(t_constraint) for t_constraint in self.literal_template_formulae)
                ),
                location_id=self.location_id,
            )
        # Check that either no template_constraint_formulae are specified
        #  or the same number of formulae as the literal has template arguments!
        elif not validator.is_template_variable(self.literal) and self.has_specification_of_template_constraints:
            assert validator.is_concept(self.literal)
            nr_template_arguments = validator.get_nr_template_arguments(self.literal)
            # check if the concept also has template arguments if the constraint formula has template constraints!
            if nr_template_arguments == 0:
                t_arg_constraints_str = ", ".join(str(t_constraint) for t_constraint in self.literal_template_formulae)
                literal_str = self.literal + (("<" + t_arg_constraints_str + ">") if t_arg_constraints_str else "")
                raise CHSemanticError(
                    f"Can not define a constraint literal value {self.literal} that is a non-template "
                    f"ValueDomain with template arguments: {literal_str}!",
                    location_id=location_id,
                )
            if (
                self.has_specification_of_template_constraints
                and len(self.literal_template_formulae) != nr_template_arguments
            ):
                t_arg_constraints_str = ", ".join(str(t_constraint) for t_constraint in self.literal_template_formulae)
                raise CHSemanticError(
                    f"The number {len(self.literal_template_formulae)} of template argument constraints "
                    f"{t_arg_constraints_str} on literal {self.literal} does not match the number of template arguments"
                    f" in the ValueDomain's definition: {validator.full_type_name(self.literal)}!",
                    location_id=self.location_id,
                )

    @property
    def has_specification_of_template_constraints(self) -> bool:
        return self.literal_template_formulae != ()

    def print_constraints_of_template_arguments(self):
        return ", ".join(str(t_constraint) for t_constraint in self.literal_template_formulae)

    def print_literal(self):
        res = self.print_constraints_of_template_arguments()
        if res:
            return self.literal + "<" + res + ">"
        return self.literal


class TemplateConstraintDescendants(TemplateConstraintHierarchyOperator):
    """
    Matches any *concrete* (non-abstract) descendant of the named concept in
    the hierarchy.  Written as the bare concept name: ``T``.

    A template argument is accepted if its assigned concept is a transitive
    subtype of ``literal`` and is not abstract.  The literal concept itself is
    accepted only if it is concrete.

    Example: ``Animal`` matches ``Dog``, ``Cat``, etc., but not the abstract
    concept ``Vertebrate`` even if ``Vertebrate`` is a subtype of ``Animal``.
    """

    def __init__(
        self,
        literal: str,
        literal_template_formulae: tuple[NonStructureConstraintFormula, ...],
        validator: TemplateConstraintFormulaValidator,
        location_id: LocationId,
    ):
        super().__init__(literal, literal_template_formulae, validator, location_id, HierarchyCheckType.DESCENDANTS_OF)

    def __repr__(self):
        return self.print_literal()


class TemplateConstraintAbstractDescendants(TemplateConstraintHierarchyOperator):
    """
    Matches *any* descendant of the named concept — concrete or abstract —
    including the literal concept itself.  Written as ``T*``.

    Use this variant instead of ``TemplateConstraintDescendants`` when abstract
    intermediate concepts in the hierarchy should also be valid instantiations.

    Example: ``Animal*`` matches ``Dog``, ``Cat``, and also the abstract concept
    ``Vertebrate`` if it is declared as a subtype of ``Animal``.
    """

    def __init__(
        self,
        literal: str,
        literal_template_formulae: tuple[NonStructureConstraintFormula, ...],
        validator: TemplateConstraintFormulaValidator,
        location_id: LocationId,
    ):
        super().__init__(
            literal, literal_template_formulae, validator, location_id, HierarchyCheckType.ABSTRACT_DESCENDANTS_OF
        )

    def __repr__(self):
        return self.print_literal() + "*"


class TemplateConstraintSelf(TemplateConstraintHierarchyOperator):
    """
    Matches *exactly* the named concept — no subtypes or supertypes are
    accepted.  Written as ``T.`` (concept name followed by a dot).

    Use this variant to require a template argument to be instantiated with
    precisely the specified concept, ruling out any more- or less-specific type.

    Example: ``Animal.`` accepts only an argument assigned the concept
    ``Animal`` itself; ``Dog`` would be rejected even though it is a subtype.
    """

    def __init__(
        self,
        literal: str,
        literal_template_formulae: tuple[NonStructureConstraintFormula, ...],
        validator: TemplateConstraintFormulaValidator,
        location_id: LocationId,
    ):
        super().__init__(literal, literal_template_formulae, validator, location_id, HierarchyCheckType.SELF)

    def __repr__(self):
        return self.print_literal() + "."


class TemplateConstraintAscendants(TemplateConstraintHierarchyOperator):
    """
    Matches any *concrete* (non-abstract) ancestor of the named concept in
    the hierarchy.  Written as ``^T``.

    A template argument is accepted if its assigned concept is a transitive
    supertype of ``literal`` and is not abstract.  This is the upward-directed
    counterpart of ``TemplateConstraintDescendants``.

    Example: ``^Dog`` matches ``Animal``, ``Mammal``, etc., but not an abstract
    common ancestor if that ancestor is declared abstract.
    """

    def __init__(
        self,
        literal: str,
        literal_template_formulae: tuple[NonStructureConstraintFormula, ...],
        validator: TemplateConstraintFormulaValidator,
        location_id: LocationId,
    ):
        super().__init__(literal, literal_template_formulae, validator, location_id, HierarchyCheckType.ASCENDANTS_OF)

    def __repr__(self):
        return "^" + self.print_literal()


class TemplateConstraintAbstractAscendants(TemplateConstraintHierarchyOperator):
    """
    Matches *any* ancestor of the named concept — concrete or abstract —
    including the literal concept itself.  Written as ``^T*``.

    Use this variant instead of ``TemplateConstraintAscendants`` when abstract
    supertypes in the hierarchy should also be valid instantiations.

    Example: ``^Dog*`` matches ``Animal``, ``Mammal``, and also any abstract
    concept that appears above ``Dog`` in the hierarchy.
    """

    def __init__(
        self,
        literal: str,
        literal_template_formulae: tuple[NonStructureConstraintFormula, ...],
        validator: TemplateConstraintFormulaValidator,
        location_id: LocationId,
    ):
        super().__init__(
            literal, literal_template_formulae, validator, location_id, HierarchyCheckType.ABSTRACT_ASCENDANTS_OF
        )

    def __repr__(self):
        return "^" + self.print_literal() + "*"


class NonTypeTemplateConstraintFormula(NonStructureConstraintFormula):
    """
    Restricts a template-argument slot to a primitive (non-type) value domain
    without pinning a specific value.  Written ``Literal:boolean``,
    ``Literal:int``, ``Literal:number``, or ``Literal:string`` in the CH
    language; stored internally with the normalised type strings ``"bool"``,
    ``"int"``, ``"float"``, or ``"string"``.

    Use this class when any value of the given primitive kind is acceptable.
    To require a specific value (e.g. exactly ``3`` or exactly ``"tag"``), use
    the subclass ``LiteralValueConstraintFormula`` instead.

    Example: a concept ``FixedVector<N: int>`` might constrain its length
    parameter with ``Literal:int`` to accept any integer instantiation.
    """

    def __init__(self, constraint_type: str, location_id: LocationId):
        super().__init__(location_id)
        self.constraint_type = constraint_type
        if self.constraint_type not in {"int", "float", "bool", "string"}:
            raise RuntimeError(f"Unknown literal constraint type: {self.constraint_type}")

    def __repr__(self):
        return "Literal:" + self.constraint_type


class LiteralValueConstraintFormula(NonTypeTemplateConstraintFormula):
    """
    A constraint that matches exactly one concrete literal value.

    In contrast to ``NonTypeTemplateConstraintFormula`` ("any integer"), this
    class pins the value: e.g. ``3``, ``-1``, ``3.14``, ``true``, ``false``,
    ``"hello"``.

    Typical use-case is as a template argument constraint, e.g.::
        Vector<3>     ->  LiteralValueConstraintFormula("int",    "3")
        Flags<true>   ->  LiteralValueConstraintFormula("bool",   "true")
        Tag<"x">      ->  LiteralValueConstraintFormula("string", '"x"')

    The ``raw_value`` string is always the canonical serialised form:

    * ``int`` / ``float`` — the digit string exactly as written (e.g. ``"3"``,
      ``"-1"``, ``"3.14"``).
    * ``bool`` — ``"true"`` or ``"false"`` (JSON convention).
    * ``string`` — the full quoted form, including surrounding ``"`` and any
      ``\\"`` escapes.

    The parsed Python value is also stored in ``value`` for efficient exact
    comparison during instantiation checking, without reparsing on every call.
    """

    def __init__(self, constraint_type: str, raw_value: str, location_id: LocationId):
        super().__init__(constraint_type, location_id)
        self.raw_value = raw_value  # canonical serialized form; used for repr & equality
        self.value: bool | int | float | str | None = None

        # Parse and store the typed Python value so check() can do exact comparison without reparsing on every call.
        if constraint_type == "int":
            ref = Reference()
            if not is_integer(raw_value, ref):
                raise RuntimeError(f"Invalid integer literal value: {raw_value!r}")
            self.value: int = ref.ref
        elif constraint_type == "float":
            ref = Reference()
            if not is_number(raw_value, ref):
                raise RuntimeError(f"Invalid float literal value: {raw_value!r}")
            self.value: float = ref.ref
        elif constraint_type == "bool":
            if raw_value not in ("true", "false"):
                raise RuntimeError(f"Boolean literal must be 'true' or 'false', got: {raw_value!r}")
            self.value: bool = raw_value == "true"
        else:  # string
            assert self.constraint_type == "string"
            if not (raw_value.startswith('"') and raw_value.endswith('"')):
                raise RuntimeError(f"String literal must be surrounded by double quotes, got: {raw_value!r}")
            self.value: str = raw_value  # keep as raw quoted string; comparison is raw-to-raw

    def __repr__(self) -> str:
        return self.raw_value


class StructureConstraintFormula(TemplateConstraintFormula):
    """
    Abstract base for constraints that express relationships *between* multiple
    template-argument slots simultaneously, rather than restricting each slot
    in isolation.

    Whereas a ``NonStructureConstraintFormula`` answers "what may this one slot
    hold?", a ``StructureConstraintFormula`` answers "given all the slots
    together, which cross-slot assignments are permissible?".  Structure
    constraints can only appear as operands of other structure constraints
    (``StructureConjunction``, ``StructureDisjunction``, ``StructureNegation``)
    or stand alone as a top-level constraint; they cannot appear inside ``And``,
    ``Or``, or ``Not``.

    The atomic unit is ``ConstraintGroup``, which pins one non-structure
    constraint per slot.  ``StructureConjunction``, ``StructureDisjunction``,
    and ``StructureNegation`` let these groups be combined logically.
    """

    def __init__(self, location_id: LocationId):
        super().__init__(location_id)


class ConstraintGroup(StructureConstraintFormula):
    """
    An ordered tuple of ``NonStructureConstraintFormula`` instances, one per
    template-argument slot, representing a single concrete cross-slot
    assignment.  Written ``<C_1, C_2, ..., C_n>``.

    A ``ConstraintGroup`` is the atomic unit of structural constraint: it
    simultaneously constrains every template-argument slot of a concept.  For a
    two-parameter concept ``Map<K, V>``, the group ``<Animal, Plant>`` means
    "K must match ``Animal`` **and** V must match ``Plant``, at the same time".

    ``group_constraints`` always contains at least one element; ``<>`` produces
    a group with a single ``Unconstrained`` slot rather than an empty tuple.

    ``ConstraintGroup`` also serves as the syntactic vehicle for template
    argument lists in hierarchy literals (e.g. the ``<Animal>`` in
    ``Vector<Animal>``).  In that context the parser extracts ``group_constraints``
    directly as the operator's ``literal_template_formulae``, so the
    ``ConstraintGroup`` object itself does not appear in the formula tree of a
    hierarchy literal.
    """

    def __init__(self, location_id: LocationId, group_constraints: tuple[NonStructureConstraintFormula, ...]):
        super().__init__(location_id)
        self.group_constraints = group_constraints

    def __repr__(self):
        return "<" + ", ".join(repr(x) for x in self.group_constraints) + ">"


class StructureConjunction(StructureConstraintFormula):
    """
    All the given structure constraints must hold simultaneously.
    Written ``Conj(S_1, S_2, ...)``.

    A cross-slot assignment is accepted only if it satisfies *every* operand.
    This is useful for expressing compound structural requirements that must all
    be true at once.

    Example: ``Conj(<Animal, Plant>, Neg(<Predator, Prey>))`` requires both
    that K is ``Animal`` and V is ``Plant``, *and* that the pair is not exactly
    ``(Predator, Prey)``.

    At least one operand is required; an empty ``Conj()`` is a parse error.
    """

    def __init__(self, location_id: LocationId, structure_constraints: tuple[StructureConstraintFormula, ...]):
        super().__init__(location_id)
        self.structure_constraints = structure_constraints

    def __repr__(self):
        return "Conj(" + ", ".join(repr(x) for x in self.structure_constraints) + ")"


class StructureDisjunction(StructureConstraintFormula):
    """
    At least one of the given structure constraints must hold.
    Written ``Disj(S_1, S_2, ...)``.

    A cross-slot assignment is accepted if it satisfies *any* operand.  This
    lets the author enumerate several acceptable cross-slot configurations.

    Example: ``Disj(<Animal, Plant>, <Fungus, Protist>)`` accepts the slot
    assignment ``(Animal, Plant)`` or ``(Fungus, Protist)``, but not any other
    combination such as ``(Animal, Protist)``.

    At least one operand is required; an empty ``Disj()`` is a parse error.
    """

    def __init__(self, location_id: LocationId, structure_constraints: tuple[StructureConstraintFormula, ...]):
        super().__init__(location_id)
        self.structure_constraints = structure_constraints

    def __repr__(self):
        return "Disj(" + ", ".join(repr(x) for x in self.structure_constraints) + ")"


class StructureNegation(StructureConstraintFormula):
    """
    The given structure constraint must *not* hold.  Written ``Neg(S)``.

    A cross-slot assignment is accepted only if it does **not** satisfy the
    single operand.  ``StructureNegation`` is typically nested inside
    ``StructureConjunction`` to exclude specific forbidden combinations while
    still permitting others.

    Example: ``Neg(<Predator, Prey>)`` accepts any slot assignment except the
    pair ``(Predator, Prey)``. This example is equivalent to
    ``Disj(<Not(Predator), Unconstrained>, <Unconstrained, Not(Prey)>)``.
    """

    def __init__(self, location_id: LocationId, structure_constraint: StructureConstraintFormula):
        super().__init__(location_id)
        self.structure_constraint = structure_constraint

    def __repr__(self):
        return "Neg(" + repr(self.structure_constraint) + ")"
