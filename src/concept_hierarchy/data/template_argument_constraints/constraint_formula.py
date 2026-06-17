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

from concept_hierarchy.errors import LocationId
from concept_hierarchy.utils import Reference, is_integer, is_number


class TemplateConstraintFormulaValidator(ABC):
    @abstractmethod
    def validate(
        self,
        ch_type_name: str,
        template_constraint_arguments: tuple[TemplateConstraintFormula, ...] | None,
        location_id: LocationId,
    ):
        pass


class TemplateConstraintFormula(ABC):
    def __init__(self, location_id: LocationId):
        self.location_id = location_id

    def __str__(self):
        return self.__repr__()


class TypeTemplateConstraintFormula(TemplateConstraintFormula, ABC):
    def __init__(self, location_id: LocationId):
        super().__init__(location_id)


class TemplateConstraintAnd(TypeTemplateConstraintFormula):
    def __init__(self, sub_formulae: list[TemplateConstraintFormula], location_id: LocationId):
        super().__init__(location_id)
        # make tuple to be immutable
        self.sub_formulae: tuple[TemplateConstraintFormula] = tuple(sub_formulae or ())  # handle None case by 'or'
        for f in self.sub_formulae:
            if not isinstance(f, TypeTemplateConstraintFormula):
                raise RuntimeError(
                    f"At {self.location_id}, the formula {f} is not a TypeTemplateConstraintFormula, "
                    "which is required by TemplateConstraintAnd!"
                )

    def __repr__(self):
        return "And(" + ", ".join([repr(f) for f in self.sub_formulae]) + ")"


class TemplateConstraintOr(TypeTemplateConstraintFormula):
    def __init__(self, sub_formulae: list[TemplateConstraintFormula], location_id: LocationId):
        super().__init__(location_id)
        # make tuple to be immutable
        self.sub_formulae: tuple[TemplateConstraintFormula, ...] = tuple(sub_formulae or ())  # handle None case by 'or'
        for f in self.sub_formulae:
            if not isinstance(f, TypeTemplateConstraintFormula):
                raise RuntimeError(
                    f"At {self.location_id}, the formula {f} is not a TypeTemplateConstraintFormula, "
                    "which is required by TemplateConstraintOr!"
                )

    def __repr__(self):
        return "Or(" + ", ".join([repr(f) for f in self.sub_formulae]) + ")"


class TemplateConstraintNot(TypeTemplateConstraintFormula):
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
    DESCENDANTS_OF = (0,)
    ABSTRACT_DESCENDANTS_OF = (1,)
    ASCENDANTS_OF = (2,)
    ABSTRACT_ASCENDANTS_OF = (3,)
    SELF = 4


# The literal is just the concept name: context.ch.is_concept(self.literal)
# If the concept is a template ValueDomain, and it doesn't have any template restrictions,
#  then it accepts all template-instantiations
class TemplateConstraintHierarchyOperator(TypeTemplateConstraintFormula, ABC):
    def __init__(
        self,
        literal: str,
        literal_template_formulae: tuple[TemplateConstraintFormula, ...] | None,
        validator: TemplateConstraintFormulaValidator,
        location_id: LocationId,
        hierarchy_op: HierarchyCheckType,
    ):
        super().__init__(location_id)
        self.literal = literal
        self.has_specification_of_template_constraints = literal_template_formulae is not None
        self.literal_template_formulae: tuple[TemplateConstraintFormula, ...] = literal_template_formulae or ()
        self.hierarchy_op: HierarchyCheckType = hierarchy_op

        # validate self.literal type
        validator.validate(
            self.literal,
            self.literal_template_formulae if self.has_specification_of_template_constraints else None,
            self.location_id,
        )

    def print_constraints_of_template_arguments(self):
        return ", ".join(str(t_constraint) for t_constraint in self.literal_template_formulae)

    def print_literal(self):
        res = self.print_constraints_of_template_arguments()
        if res:
            return self.literal + "<" + res + ">"
        return self.literal


class TemplateConstraintDescendants(TemplateConstraintHierarchyOperator):
    def __init__(
        self,
        literal: str,
        literal_template_formulae: tuple[TemplateConstraintFormula, ...],
        validator: TemplateConstraintFormulaValidator,
        location_id: LocationId,
    ):
        super().__init__(literal, literal_template_formulae, validator, location_id, HierarchyCheckType.DESCENDANTS_OF)

    def __repr__(self):
        return self.print_literal()


class TemplateConstraintAbstractDescendants(TemplateConstraintHierarchyOperator):
    def __init__(
        self,
        literal: str,
        literal_template_formulae: tuple[TemplateConstraintFormula, ...],
        validator: TemplateConstraintFormulaValidator,
        location_id: LocationId,
    ):
        super().__init__(
            literal, literal_template_formulae, validator, location_id, HierarchyCheckType.ABSTRACT_DESCENDANTS_OF
        )

    def __repr__(self):
        return self.print_literal() + "*"


class TemplateConstraintSelf(TemplateConstraintHierarchyOperator):
    def __init__(
        self,
        literal: str,
        literal_template_formulae: tuple[TemplateConstraintFormula, ...],
        validator: TemplateConstraintFormulaValidator,
        location_id: LocationId,
    ):
        super().__init__(literal, literal_template_formulae, validator, location_id, HierarchyCheckType.SELF)

    def __repr__(self):
        return self.print_literal() + "."


class TemplateConstraintAscendants(TemplateConstraintHierarchyOperator):
    def __init__(
        self,
        literal: str,
        literal_template_formulae: tuple[TemplateConstraintFormula, ...],
        validator: TemplateConstraintFormulaValidator,
        location_id: LocationId,
    ):
        super().__init__(literal, literal_template_formulae, validator, location_id, HierarchyCheckType.ASCENDANTS_OF)

    def __repr__(self):
        return "^" + self.print_literal()


class TemplateConstraintAbstractAscendants(TemplateConstraintHierarchyOperator):
    def __init__(
        self,
        literal: str,
        literal_template_formulae: tuple[TemplateConstraintFormula, ...],
        validator: TemplateConstraintFormulaValidator,
        location_id: LocationId,
    ):
        super().__init__(
            literal, literal_template_formulae, validator, location_id, HierarchyCheckType.ABSTRACT_ASCENDANTS_OF
        )

    def __repr__(self):
        return "^" + self.print_literal() + "*"


class NonTypeTemplateConstraintFormula(TemplateConstraintFormula):
    def __init__(self, constraint_type: str, location_id: LocationId):
        super().__init__(location_id)
        self.constraint_type = constraint_type
        if self.constraint_type not in ["int", "float", "bool", "string"]:
            raise RuntimeError("Unknown constraint type: {}".format(self.constraint_type))

    def __repr__(self):
        return "Literal:" + self.constraint_type


class LiteralValueConstraintFormula(TemplateConstraintFormula):
    """
    A constraint that matches exactly one concrete literal value.

    In contrast to NonTypeTemplateConstraintFormula ("any integer"), this
    class pins the value: e.g. 3, -1, 3.14, true, false, "hello".

    Typical use-case is as a template argument constraint, e.g.
        Vector<3>     ->  LiteralValueConstraintFormula("int",    "3")
        Flags<true>   ->  LiteralValueConstraintFormula("bool",   "true")
        Tag<"x">      ->’  LiteralValueConstraintFormula("string", '"x"')

    The raw_value string is always the canonical serialized form:
      int/float  --  the digit string exactly as written  (e.g. "3", "-1", "3.14")
      bool       --  "true" or "false"  (JSON convention)
      string     --  the full quoted form, including surrounding " and any \" escapes
    """

    def __init__(self, constraint_type: str, raw_value: str, location_id: LocationId):
        super().__init__(location_id)
        if constraint_type not in ("int", "float", "bool", "string"):
            raise RuntimeError(f"Unknown literal constraint type: {constraint_type!r}")
        self.constraint_type = constraint_type
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
