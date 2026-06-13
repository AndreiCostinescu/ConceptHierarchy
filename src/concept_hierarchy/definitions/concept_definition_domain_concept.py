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

from concept_hierarchy.definitions.concept_definition import ConceptDefinition
from concept_hierarchy.definitions.utils import check_ch_name
from concept_hierarchy.errors import CHSemanticError, CHSyntaxError, LocationId, PathPart, PathSegment
from concept_hierarchy.models import ConceptHierarchyModel


class MultipleSpecializationException(Exception):
    def __init__(
        self,
        unspecialized_keyword: str,
        specialization_for: str,
        which_specialize: list[bool],
        individual_specializations: list,
    ):
        self.keyword = unspecialized_keyword
        self.specialization_for = specialization_for
        self.which = which_specialize
        self.individual_specializations = individual_specializations


INHERIT_FROM_KEYWORD = "inheritFrom:"


class DomainDataDefinition(ABC):
    def __init__(
        self,
        concept_hierarchy_data: ConceptHierarchyModel,
        location_id: LocationId,
        domain_data_type: str,
        specialization_keywords: list[str],
        name: str,
        defining_concept: str | None = None,
    ):
        self.concept_hierarchy_data = concept_hierarchy_data
        self._location_id = location_id
        self.domain_data_type = domain_data_type
        self.name = name  # type: str
        self.defining_concept = defining_concept  # type: str | None  # this should be set only when defining the property, not when specializing it!
        # For merging data from the "forSub" of multiple parent concepts,
        #  one could determine where the specialization occurred (by storing in 'inherited_data' the concept in which
        #  the specialization happened) and checking if the concept that defined the specialization is the same for all
        #  multiple values
        self.inherited_keywords = {}  # type: dict[str, bool]
        self.is_definition_of_keyword = {}  # type: dict[str, bool]  # if this is the first-definition of the keyword's value
        # maps keywords to the direct-parent-concept from which to inherit the data
        self.disambiguate_from = {}  # type: dict[str, str]
        # keywords that can be specialized:
        self.specialization_keywords = specialization_keywords

    @abstractmethod
    def convert_to_dict(self):
        pass

    @abstractmethod
    def get_type_plural(self):
        pass

    @abstractmethod
    def is_data_set(self, prop_keyword: str):
        pass

    def is_inherited_data(self, datum_keyword: str) -> bool:
        if self.is_definition_of_keyword[datum_keyword]:
            raise RuntimeError(
                f"{self.domain_data_type} keyword {datum_keyword} is defined in this concept, not specialized!"
            )
        return self.inherited_keywords.get(datum_keyword, True)

    # whether the data is defined in this definition
    def has_defined_data(self, datum_keyword: str) -> bool:
        return self.is_data_set(datum_keyword) and self.is_definition_of_keyword[datum_keyword]

    # whether the data is not defined, but there is data, which is not specialized (i.e. is inherited) in this concept
    def has_inherited_data(self, datum_keyword: str) -> bool:
        return (
            self.is_data_set(datum_keyword)
            and not self.is_definition_of_keyword[datum_keyword]
            and self.inherited_keywords[datum_keyword]
        )

    # whether the data is not defined, but there is data, which is specialized in this concept
    def has_specialized_data(self, datum_keyword: str) -> bool:
        return (
            self.is_data_set(datum_keyword)
            and not self.is_definition_of_keyword[datum_keyword]
            and not self.inherited_keywords[datum_keyword]
        )

    @abstractmethod
    def get_data(self, prop_keyword: str):
        pass

    @abstractmethod
    def set_data(self, prop_keyword: str, datum: object, definition: bool = False, inherit_from_allowed: bool = True):
        pass

    @staticmethod
    def get_merged_data_for_keyword(
        data_defs: dict[str, DomainDataDefinition], specialization_keyword: str, specialization_for: str
    ) -> bool | tuple[object, bool]:
        # For merging data from the "forSub" of multiple parent concepts,
        #  one could determine where the specialization occurred
        #   (by storing in 'inherited_data' the concept in which the specialization happened)
        #  and checking if the concept is the same for all multiple values
        count_set_data = 0
        which_specialize = []
        who_specializes = []
        specialized_data = []
        keyword_inherited_data = None
        for specializing_parent, data_def in data_defs.items():
            res = data_def.is_data_set(specialization_keyword)  # don't differentiate between inherited and specialized
            which_specialize.append(res)
            who_specializes.append(specializing_parent)
            if res:
                count_set_data += 1
                keyword_inherited_data = data_def.get_data(specialization_keyword)
                assert not isinstance(keyword_inherited_data, tuple)
                specialized_data.append(keyword_inherited_data)
            else:
                specialized_data.append(None)
        same_inherited_value = all(
            x == keyword_inherited_data for i, x in enumerate(specialized_data) if which_specialize[i]
        )
        # if count_set_data == 0:
        #   it can be that no parent concept specializes this domain_def_data
        #       and its definition is simply "unknown"/default
        #   if so, don't set the specialized data
        if count_set_data > 1 and not same_inherited_value:
            raise MultipleSpecializationException(
                specialization_keyword, specialization_for, who_specializes, specialized_data
            )
        elif count_set_data != 0:
            # either one specialization or multiple specializations with the same inherited value
            return keyword_inherited_data, True
        return False

    def merge_defs(self, data_defs: dict[str, DomainDataDefinition]):
        if len(data_defs) == 0:
            return
        for specialization_keyword in self.specialization_keywords:
            if not self.is_data_set(specialization_keyword):
                if specialization_keyword in self.disambiguate_from:
                    concept_to_inherit_from = self.disambiguate_from[specialization_keyword]
                    if concept_to_inherit_from not in data_defs:
                        raise RuntimeError(
                            f"Logical Error: concept to inherit from {concept_to_inherit_from} not in data_defs!"
                        )
                    p_def = data_defs[concept_to_inherit_from]
                    if p_def.is_data_set(specialization_keyword):
                        self.set_data(specialization_keyword, p_def.get_data(specialization_keyword), False, False)
                        assert (
                            self.is_data_set(specialization_keyword)
                            and not self.is_definition_of_keyword[specialization_keyword]
                        )
                else:
                    res = DomainDataDefinition.get_merged_data_for_keyword(
                        data_defs, specialization_keyword, self.get_type_plural()
                    )
                    if isinstance(res, tuple) and len(res) == 2 and res[1]:
                        res_data = res[0]
                        assert not (isinstance(res_data, str) and res_data.startswith(INHERIT_FROM_KEYWORD))
                        # that the keyword's is inherited will be set at the end of the set_data function
                        self.set_data(specialization_keyword, res_data, False, False)
                self.inherited_keywords[specialization_keyword] = True

    def location_id(self, *location_ids: PathSegment):
        return self._location_id + [*location_ids]


class PropertyDefinition(DomainDataDefinition):
    VALUE_DOMAIN = "valueDomain"
    DESCRIPTION = "description"
    CONSTRAINT = "constraint"
    STATIC = "static"
    DEFAULT = "default"
    ASSUMPTIONS = "assumptions"
    HOOKS = "hooks"
    COMPUTATIONS = "computations"
    CONFIDENCE = "confidenceHalfDecayTime"
    DEFAULT_INSTANCE_NAMING = "nameDefaultInstanceValuesWithThisInstanceName"
    SPECIALIZATION_KEYWORDS = [
        CONSTRAINT,
        DEFAULT,
        ASSUMPTIONS,
        HOOKS,
        COMPUTATIONS,
        CONFIDENCE,
        DEFAULT_INSTANCE_NAMING,
    ]
    ALL_DEFINITION_KEYWORDS = [VALUE_DOMAIN, DESCRIPTION, STATIC] + SPECIALIZATION_KEYWORDS

    def __init__(
        self,
        concept_hierarchy_data: ConceptHierarchyModel,
        location_id: LocationId,
        prop_name: str,
        defining_concept: str = None,
    ):
        super().__init__(
            concept_hierarchy_data,
            location_id,
            "Property",
            PropertyDefinition.SPECIALIZATION_KEYWORDS,
            prop_name,
            defining_concept,
        )
        self.value_domain = None  # type: str | None
        self.description = None  # type: str | None
        self.constraint = None  # type: str | dict | None  # either a sub-ValueDomain or a Variation definition
        self.static = False  # type: bool | None  # whether the property is static
        self.default = None  # type: object | None  # serialized value; how to differentiate between None as "null" serialized value and no-default-value specified?
        self.default_specified = False  # type: bool  # whether the None value above represents a not-specified-default-value or the default value of "null"
        self.assumed = None  # type: object | None  # assumed values
        self.hooks = None  # type: dict | None  # consistency: hooks
        self.computation = None  # type: dict | None  # consistency: computations
        self.confidence = None  # type: tuple[float, str] | float | None  # type: confidence half-decay time
        self.name_default_instance_values_with_instance_name = None  # type: bool | None

    def __repr__(self):
        return "PropertyDefinition(name: {}, data: {})".format(self.name, repr(self.convert_to_dict()))

    def convert_to_dict(self):
        res = {}
        if self.value_domain is not None:
            res[PropertyDefinition.VALUE_DOMAIN] = self.value_domain
        if self.description is not None:
            res[PropertyDefinition.DESCRIPTION] = self.description
        if self.constraint is not None:
            res[PropertyDefinition.CONSTRAINT] = self.constraint
        if self.static is not None:
            res[PropertyDefinition.STATIC] = self.static
        if self.default_specified:
            res[PropertyDefinition.DEFAULT] = self.default
        if self.assumed is not None:
            res[PropertyDefinition.ASSUMPTIONS] = self.assumed
        if self.hooks is not None:
            res[PropertyDefinition.HOOKS] = self.hooks
        if self.computation is not None:
            res[PropertyDefinition.COMPUTATIONS] = self.computation
        if self.confidence is not None:
            res[PropertyDefinition.CONFIDENCE] = self.confidence
        if self.name_default_instance_values_with_instance_name is not None:
            res[PropertyDefinition.DEFAULT_INSTANCE_NAMING] = self.name_default_instance_values_with_instance_name
        return res

    def get_type_plural(self):
        return "properties"

    def is_data_set(self, prop_keyword: str):
        if prop_keyword == PropertyDefinition.VALUE_DOMAIN:
            return self.value_domain is not None
        elif prop_keyword == PropertyDefinition.DESCRIPTION:
            return self.description is not None
        elif prop_keyword == PropertyDefinition.CONSTRAINT:
            return self.constraint is not None
        elif prop_keyword == PropertyDefinition.STATIC:
            return self.static is not None
        elif prop_keyword == PropertyDefinition.DEFAULT:
            return self.default_specified
        elif prop_keyword == PropertyDefinition.ASSUMPTIONS:
            return self.assumed is not None
        elif prop_keyword == PropertyDefinition.HOOKS:
            return self.hooks is not None
        elif prop_keyword == PropertyDefinition.COMPUTATIONS:
            return self.computation is not None
        elif prop_keyword == PropertyDefinition.CONFIDENCE:
            return self.confidence is not None
        elif prop_keyword == PropertyDefinition.DEFAULT_INSTANCE_NAMING:
            return self.name_default_instance_values_with_instance_name is not None
        raise RuntimeError(f"Unknown property definition key: {prop_keyword}")

    def get_data(self, prop_keyword: str):
        if not self.is_data_set(prop_keyword):
            raise RuntimeError(
                "Non-existent property definition data {} to get for property {}".format(prop_keyword, self.name)
            )
        if prop_keyword == PropertyDefinition.VALUE_DOMAIN:
            return self.value_domain
        elif prop_keyword == PropertyDefinition.DESCRIPTION:
            return self.description
        elif prop_keyword == PropertyDefinition.CONSTRAINT:
            return self.constraint
        elif prop_keyword == PropertyDefinition.STATIC:
            return self.static
        elif prop_keyword == PropertyDefinition.DEFAULT:
            return self.default
        elif prop_keyword == PropertyDefinition.ASSUMPTIONS:
            return self.assumed
        elif prop_keyword == PropertyDefinition.HOOKS:
            return self.hooks
        elif prop_keyword == PropertyDefinition.COMPUTATIONS:
            return self.computation
        elif prop_keyword == PropertyDefinition.CONFIDENCE:
            return self.confidence
        elif prop_keyword == PropertyDefinition.DEFAULT_INSTANCE_NAMING:
            return self.name_default_instance_values_with_instance_name
        raise RuntimeError("Unknown property definition key:", prop_keyword)

    def set_data(self, prop_keyword: str, datum: object, definition: bool = False, inherit_from_allowed: bool = True):
        if not definition and (
            prop_keyword in PropertyDefinition.ALL_DEFINITION_KEYWORDS
            and prop_keyword not in PropertyDefinition.SPECIALIZATION_KEYWORDS
        ):
            raise RuntimeError(
                "{} should not be modifiable after creation!".format(
                    ", ".join(
                        [
                            x
                            for x in PropertyDefinition.ALL_DEFINITION_KEYWORDS
                            if x not in PropertyDefinition.SPECIALIZATION_KEYWORDS
                        ]
                    )
                )
            )
        if isinstance(datum, str) and datum.startswith(INHERIT_FROM_KEYWORD):
            if definition:
                raise RuntimeError("Can't define inherit from in definition!")
            if not inherit_from_allowed:
                raise RuntimeError("Logic error: Inherit from not allowed here")
            self.disambiguate_from[prop_keyword] = ":".join(datum.split(":")[1:])
            return
        elif prop_keyword == PropertyDefinition.VALUE_DOMAIN:
            self.value_domain = datum
        elif prop_keyword == PropertyDefinition.DESCRIPTION:
            self.description = datum
        elif prop_keyword == PropertyDefinition.CONSTRAINT:
            self.constraint = datum
        elif prop_keyword == PropertyDefinition.STATIC:
            self.static = datum
        elif prop_keyword == PropertyDefinition.DEFAULT:
            self.default = datum
            self.default_specified = True
        elif prop_keyword == PropertyDefinition.ASSUMPTIONS:
            self.assumed = datum
        elif prop_keyword == PropertyDefinition.HOOKS:
            self.hooks = datum
        elif prop_keyword == PropertyDefinition.COMPUTATIONS:
            self.computation = datum
        elif prop_keyword == PropertyDefinition.CONFIDENCE:
            self.confidence = datum
        elif prop_keyword == PropertyDefinition.DEFAULT_INSTANCE_NAMING:
            self.name_default_instance_values_with_instance_name = datum
        else:
            raise RuntimeError("Unknown property definition key:", prop_keyword)
        self.is_definition_of_keyword[prop_keyword] = definition
        if not definition:
            self.inherited_keywords[prop_keyword] = False


class FunctionDefinition(DomainDataDefinition):
    VALUE_DOMAIN = "valueDomain"
    DESCRIPTION = "description"
    STATIC = "static"
    DEFAULT = "default"
    SPECIALIZATION_KEYWORDS = [DEFAULT]
    ALL_DEFINITION_KEYWORDS = [VALUE_DOMAIN, DESCRIPTION, STATIC] + SPECIALIZATION_KEYWORDS

    @staticmethod
    def looks_like_function_definition(x):
        return isinstance(x, dict) and ((len(x) == 1) or (len(x) == 2 and "procedure" in x and "interface" in x))

    def __init__(
        self,
        concept_hierarchy_data: ConceptHierarchyModel,
        location_id: LocationId,
        prop_name: str,
        defining_concept: str = None,
    ):
        super().__init__(
            concept_hierarchy_data,
            location_id,
            "Function",
            FunctionDefinition.SPECIALIZATION_KEYWORDS,
            prop_name,
            defining_concept,
        )
        self.value_domain = "CustomFunction" if defining_concept is not None else None  # type: str | None
        self.description = None  # type: str | None
        self.static = None  # type: bool | None  # whether the property is static
        self.default = None  # type: object | None  # serialized value; how to differentiate between None as "null" serialized value and no-default-value specified?
        self.default_specified = False  # type: bool  # whether the None value above represents a not-specified-default-value or the default value of "null"

    def __repr__(self):
        return "FunctionDefinition(name: {}, data: {})".format(self.name, repr(self.convert_to_dict()))

    def convert_to_dict(self):
        res = {}
        if self.value_domain is not None:
            res[FunctionDefinition.VALUE_DOMAIN] = self.value_domain
        if self.description is not None:
            res[FunctionDefinition.DESCRIPTION] = self.description
        if self.static is not None:
            res[FunctionDefinition.STATIC] = self.static
        if self.default_specified:
            res[FunctionDefinition.DEFAULT] = self.default
        return res

    def get_type_plural(self):
        return "functions"

    def is_data_set(self, func_keyword: str):
        if func_keyword == FunctionDefinition.VALUE_DOMAIN:
            return self.value_domain is not None
        elif func_keyword == FunctionDefinition.DESCRIPTION:
            return self.description is not None
        elif func_keyword == FunctionDefinition.STATIC:
            return self.static is not None
        elif func_keyword == FunctionDefinition.DEFAULT:
            return self.default_specified
        raise RuntimeError("Unknown function definition key:", func_keyword)

    def get_data(self, func_keyword: str):
        if not self.is_data_set(func_keyword):
            raise RuntimeError(
                "Non-existent function definition data {} to get for function {}".format(func_keyword, self.name)
            )
        if func_keyword == FunctionDefinition.VALUE_DOMAIN:
            return self.value_domain
        elif func_keyword == FunctionDefinition.DESCRIPTION:
            return self.description
        elif func_keyword == FunctionDefinition.STATIC:
            return self.static
        elif func_keyword == FunctionDefinition.DEFAULT:
            return self.default
        raise RuntimeError("Unknown function definition key:", func_keyword)

    def set_data(self, func_keyword: str, datum: object, definition: bool = False, inherit_from_allowed: bool = True):
        if not definition and (
            func_keyword in FunctionDefinition.ALL_DEFINITION_KEYWORDS
            and func_keyword not in FunctionDefinition.SPECIALIZATION_KEYWORDS
        ):
            raise RuntimeError(
                "{} should not be modifiable after creation!".format(
                    ", ".join(
                        [
                            x
                            for x in FunctionDefinition.ALL_DEFINITION_KEYWORDS
                            if x not in FunctionDefinition.SPECIALIZATION_KEYWORDS
                        ]
                    )
                )
            )
        if isinstance(datum, str) and datum.startswith(INHERIT_FROM_KEYWORD):
            if definition:
                raise RuntimeError("Can't define inherit from in definition!")
            if not inherit_from_allowed:
                raise RuntimeError("Logic error: Inherit from not allowed here")
            self.disambiguate_from[func_keyword] = ":".join(datum.split(":")[1:])
            return
        elif func_keyword == FunctionDefinition.VALUE_DOMAIN:
            if not self.concept_hierarchy_data.is_a_subconcept_of_b(datum, "CustomFunction", include_self=True):
                raise CHSemanticError(
                    f"Bad ValueDomain {datum!r} defined for concept function {self.name}",
                    self.location_id(FunctionDefinition.VALUE_DOMAIN),
                    part=PathPart.VALUE,
                )
            self.value_domain = datum
        elif func_keyword == FunctionDefinition.DESCRIPTION:
            self.description = datum
        elif func_keyword == FunctionDefinition.STATIC:
            self.static = datum
        elif func_keyword == FunctionDefinition.DEFAULT:
            self.default = datum
            self.default_specified = True
        else:
            raise RuntimeError("Unknown function definition key:", func_keyword)
        self.is_definition_of_keyword[func_keyword] = definition
        if not definition:
            self.inherited_keywords[func_keyword] = False


class DomainConceptDefinition(ConceptDefinition):
    domain_concept_name: str = "Domain Concept"
    domain_concept_properties: str = "properties"
    domain_concept_functions: str = "functions"
    domain_concept_management: str = "management"
    domain_concept_management_initialization: str = "initialization"
    domain_concept_management_consolidation: str = "consolidation"
    domain_concept_specialization: str = "_specializations"
    domain_concept_specialization_for_this: str = "_forThis"
    domain_concept_data_keys: set[str] = {
        domain_concept_properties,
        domain_concept_functions,
        domain_concept_management,
    }
    property_data_keys: set[str] = {
        PropertyDefinition.VALUE_DOMAIN,
        PropertyDefinition.CONSTRAINT,
        PropertyDefinition.DESCRIPTION,
        PropertyDefinition.STATIC,
        PropertyDefinition.DEFAULT,
        PropertyDefinition.ASSUMPTIONS,
        PropertyDefinition.HOOKS,
        PropertyDefinition.COMPUTATIONS,
        PropertyDefinition.CONFIDENCE,
        PropertyDefinition.DEFAULT_INSTANCE_NAMING,
    }
    function_data_keys: set[str] = {
        FunctionDefinition.VALUE_DOMAIN,
        FunctionDefinition.DESCRIPTION,
        FunctionDefinition.STATIC,
        FunctionDefinition.DEFAULT,
    }
    management_data_keys: set[str] = {domain_concept_management_initialization, domain_concept_management_consolidation}
    default_value_domain_type_of_domain_concept_functions: str = "CustomFunction"

    def __init__(self, name: str, definition_data: object, definition_location_str: str):
        super().__init__(name, definition_data, definition_location_str)

        self.properties: dict[str, dict] = {}
        self.functions: dict[str, dict] = {}
        self.management: dict[str, dict] = {}

        # [prop name -> data to be inherited by subconcepts (prop-def-key -> value)]
        # will be extended with the "implicit get" values => they won't just contain the specified values in the concept
        self.property_specializations_for_sub: dict[str, dict[str, str | object]] = {}
        self.function_specializations_for_sub: dict[str, dict[str, str | object]] = {}
        # [prop name -> data to be set in this concept (prop-def-key -> (value, whether it was defined or inherited))]
        self.property_specializations_for_this: dict[str, dict[str, tuple[str | object, bool]]] = {}
        self.function_specializations_for_this: dict[str, dict[str, tuple[str | object, bool]]] = {}
        # keys are (property name -> property definition keyword) / (function name -> function definition keyword)
        # value is the name of the concept in which the value of the property definition keyword is defined
        self.available_property_data: dict[str, dict[str, str]] = {}
        self.available_function_data: dict[str, dict[str, str]] = {}

        # (property name, concept that defines it, property type)
        # self.all_properties: dict[str, tuple[str, str, ValueDomainType]] = {}
        # self.own_properties: dict[str, PropertyDefinition] = {}
        # (function name, concept that defines it, function type)
        # self.all_functions: dict[str, tuple[str, str, ValueDomainType]] = {}
        # self.own_functions: dict[str, FunctionDefinition] = {}

    @classmethod
    def from_node(cls, concept_definition: ConceptDefinition):
        domain_concept = cls._from_node(concept_definition)
        domain_concept.properties = {}
        domain_concept.functions = {}
        domain_concept.management = {}

        domain_concept.property_specializations_for_sub = {}
        domain_concept.function_specializations_for_sub = {}
        domain_concept.property_specializations_for_this = {}
        domain_concept.function_specializations_for_this = {}
        domain_concept.available_property_data = {}
        domain_concept.available_function_data = {}
        return domain_concept

    def definition_type(self) -> str:
        return DomainConceptDefinition.domain_concept_name

    def definition_location(self) -> list[str]:
        return self.data_location_id

    def initialize_domain_concept_data(
        self,
        for_either_properties_or_functions: bool,
        data_container: dict[str, dict[str, object]],
        data_specialization_for_sub: dict[str, dict[str, object]],
        data_specialization_for_this: dict[str, dict[str, tuple[str | object, bool]]],
    ):
        """Modify data_container, data_specialization_for_sub, and data_specialization_for_this in place!"""
        data_container.clear()
        data_specialization_for_sub.clear()
        data_specialization_for_this.clear()
        data_type_plural = "properties" if for_either_properties_or_functions else "functions"
        def_location = (
            DomainConceptDefinition.domain_concept_properties
            if for_either_properties_or_functions
            else DomainConceptDefinition.domain_concept_functions
        )

        data_container.update(self.data.get(def_location, {}))
        if not isinstance(data_container, dict):
            raise CHSyntaxError(
                f"The definition of domain concept {data_type_plural} must be a JSON object, not "
                f"{data_container!r} as encountered at domain concept {self.name}",
                location_id=self.location_id(def_location),
                part=PathPart.VALUE,
            )
        data_specialization_for_sub.update(
            data_container.pop(DomainConceptDefinition.domain_concept_specialization, {})
        )
        for name, spec_data in data_specialization_for_sub.pop(
            DomainConceptDefinition.domain_concept_specialization_for_this, {}
        ).items():
            data_specialization_for_this[name] = {}
            for def_key, def_data in spec_data.items():
                # Assumption that data defined here is inherited.
                # Correct the False value below in :func:`initialize_domain_concept_specialization_data`
                #  if name is a property/function defined in this concept
                data_specialization_for_this[name][def_key] = (def_data, False)

    def initialize_domain_concept_specialization_data(
        self,
        for_either_properties_or_functions: bool,
        data_container: dict[str, dict[str, object]],
        data_specialization_for_sub: dict[str, dict[str, object]],
        data_specialization_for_this: dict[str, dict[str, tuple[str | object, bool]]],
        available_data: dict[str, dict[str, str]],
        specializable_keywords: list[str],
        value_domain_keyword: str,
    ):
        """Modify data_container, data_specialization_for_sub, and data_specialization_for_this in place!"""
        data_type = "property" if for_either_properties_or_functions else "function"
        def_location = (
            DomainConceptDefinition.domain_concept_properties
            if for_either_properties_or_functions
            else DomainConceptDefinition.domain_concept_functions
        )
        for name, definition_data in data_container.items():
            # create the specialization data for the own defined data (i.e. add it to the specialized data)
            if name not in data_specialization_for_sub:
                data_specialization_for_sub[name] = {}
            else:
                # Tried to specialize a property in the top-level SPECIALIZATIONS object that is defined in this concept
                raise CHSemanticError(
                    f"The {data_type} {name!r} that is defined in this concept can not be specialized at"
                    f" the top-level.\nTop-level specializations mean to specialize for subconcepts, which already is "
                    f"the default specialization when defining the {data_type}.\n\tRemove the key from "
                    f'"{DomainConceptDefinition.domain_concept_specialization}" and merge the data with the {data_type}'
                    f' definition in "{def_location}/{name}".!',
                    location_id=self.location_id(
                        def_location, DomainConceptDefinition.domain_concept_specialization, name
                    ),
                    part=PathPart.KEY,
                )
            if name not in data_specialization_for_this:
                data_specialization_for_this[name] = {}
            else:
                # mark existing keys as set, not inherited
                for spec_for_this_name, spec_for_this_data in data_specialization_for_this.items():
                    for spec_for_this_def_key, spec_for_this_def_data in spec_for_this_data.items():
                        assert isinstance(spec_for_this_def_data, tuple) and spec_for_this_def_data[1] is False
                        spec_for_this_data[spec_for_this_def_key] = (spec_for_this_def_data[0], True)
            available_data[name] = {}
            for def_key, def_data in definition_data.items():
                available_data[name][def_key] = self.name
                if def_key in specializable_keywords:
                    # don't overwrite data of existing keys with the definition in properties
                    if def_key not in data_specialization_for_this[name]:
                        # only set data in forThis if it was not already specified in the specialization
                        data_specialization_for_this[name][def_key] = (def_data, True)
                    assert def_key not in data_specialization_for_sub[name]
                    data_specialization_for_sub[name][def_key] = def_data
            # the value domain definition could be a constraint instead of a string/ValueDomain
            #  if so, the VALUE_DOMAIN entry does not appear in prop_data; so add it manually here below
            if value_domain_keyword not in available_data[name]:
                available_data[name][value_domain_keyword] = self.name
                # don't set VALUE_DOMAIN data for this or for subconcepts, because VALUE_DOMAIN is not specializable!
                assert value_domain_keyword not in specializable_keywords

    def concept_data_check(self):
        data_keys: set[str] = set(self.data.keys())
        if not (data_keys <= DomainConceptDefinition.domain_concept_data_keys):
            extra_keys = data_keys - DomainConceptDefinition.domain_concept_data_keys
            raise CHSyntaxError(
                f"Found extra keys {extra_keys!r} in the domain concept data definition of {self.name}",
                self.location_id(),
                part=PathPart.VALUE,
            )

        has_properties = DomainConceptDefinition.domain_concept_properties in self.data
        has_functions = DomainConceptDefinition.domain_concept_functions in self.data
        has_management = DomainConceptDefinition.domain_concept_management in self.data
        if not self.is_root and not has_properties and not has_functions and not has_management:
            raise CHSemanticError(
                f"Found a domain concept with no data defined {self.name}", self.location_id(), part=PathPart.VALUE
            )

        self.initialize_domain_concept_data(
            True,
            self.properties,
            self.property_specializations_for_sub,
            self.property_specializations_for_this,
        )
        for prop_name, prop_data in self.properties.items():
            assert prop_name != DomainConceptDefinition.domain_concept_specialization
            # write assertion because the data from "properties" comes directly from json deserialization of an object,
            #  which comes directly from a syntactically valid json file, where object keys are defined to be strings
            assert isinstance(prop_name, str)
            if not check_ch_name(prop_name, must_start_lowercase=True):
                raise CHSyntaxError(
                    f"Property names of domain concepts must be a lowercase-starting string, not {prop_name!r}",
                    self.location_id(DomainConceptDefinition.domain_concept_properties, prop_name),
                    part=PathPart.KEY,
                )
            if not isinstance(prop_data, (dict, str)):
                raise CHSyntaxError(
                    f"The definition of domain concept properties must be a JSON object or string "
                    f"(that defines its ValueDomain), not {prop_data!r} for {prop_name} of domain concept {self.name}!",
                    self.location_id(DomainConceptDefinition.domain_concept_properties, prop_name),
                    part=PathPart.VALUE,
                )
            if isinstance(prop_data, dict):
                prop_def_data_keys = set(prop_data.keys())
                if not (prop_def_data_keys <= DomainConceptDefinition.property_data_keys):
                    if not prop_def_data_keys.isdisjoint(DomainConceptDefinition.property_data_keys):
                        extra_keys = prop_def_data_keys - DomainConceptDefinition.property_data_keys
                        raise CHSyntaxError(
                            f"Found extra keys {extra_keys!r} in the domain concept data definition of property "
                            f"{prop_name} for {self.name}",
                            self.location_id(DomainConceptDefinition.domain_concept_properties, prop_name),
                            part=PathPart.VALUE,
                        )
                    # if prop_def does not contain any property-definition-keys, interpret as the definition of an
                    #  expression that defines the constraint & type of the property
                    self.properties[prop_name] = {PropertyDefinition.CONSTRAINT: prop_data}
            else:
                self.properties[prop_name] = {PropertyDefinition.VALUE_DOMAIN: prop_data}
        self.initialize_domain_concept_specialization_data(
            True,
            self.properties,
            self.property_specializations_for_sub,
            self.property_specializations_for_this,
            self.available_property_data,
            PropertyDefinition.SPECIALIZATION_KEYWORDS,
            PropertyDefinition.VALUE_DOMAIN,
        )
        for prop_name, prop_data in self.properties.items():
            prop_def_data_keys = set(prop_data.keys())
            if not any(
                x in prop_def_data_keys for x in [PropertyDefinition.VALUE_DOMAIN, PropertyDefinition.CONSTRAINT]
            ):
                raise CHSyntaxError(
                    f"The property definition of a domain concept property must define either its ValueDomain or "
                    f"its constraint.\n\tGot {prop_data!r} as the definition of {prop_name} of {self.name}",
                    self.location_id(DomainConceptDefinition.domain_concept_properties, prop_name),
                    part=PathPart.VALUE,
                )
            # check the type of each property definition keyword value!
            if PropertyDefinition.VALUE_DOMAIN in prop_data:
                prop_def_val = prop_data[PropertyDefinition.VALUE_DOMAIN]
                if not isinstance(prop_def_val, str):
                    raise CHSyntaxError(
                        f"The ValueDomain definition of properties must be a JSON string value, not {prop_def_val!r}",
                        self.location_id(
                            DomainConceptDefinition.domain_concept_properties,
                            prop_name,
                            PropertyDefinition.VALUE_DOMAIN,
                        ),
                        part=PathPart.VALUE,
                    )
            if PropertyDefinition.DESCRIPTION in prop_data:
                prop_def_val = prop_data[PropertyDefinition.DESCRIPTION]
                if not isinstance(prop_def_val, str):
                    raise CHSyntaxError(
                        f"The definition of a property description must be a JSON string value, not {prop_def_val!r}",
                        self.location_id(
                            DomainConceptDefinition.domain_concept_properties, prop_name, PropertyDefinition.DESCRIPTION
                        ),
                        part=PathPart.VALUE,
                    )
            # Don't check constraints, default, assumed, and confidence values because they are the serialization of
            #  ValueDomains  (i.e. any JSON value).
            if PropertyDefinition.STATIC in prop_data:
                prop_def_val = prop_data[PropertyDefinition.STATIC]
                if not isinstance(prop_def_val, bool):
                    raise CHSyntaxError(
                        f"The marker of static properties must be a JSON boolean, not {prop_def_val!r}",
                        self.location_id(
                            DomainConceptDefinition.domain_concept_properties, prop_name, PropertyDefinition.STATIC
                        ),
                        part=PathPart.VALUE,
                    )
            if PropertyDefinition.HOOKS in prop_data:
                prop_def_val = prop_data[PropertyDefinition.HOOKS]
                if not isinstance(prop_def_val, dict):
                    raise CHSyntaxError(
                        f"The definition of property hooks must be a JSON object, not {prop_def_val!r}",
                        self.location_id(
                            DomainConceptDefinition.domain_concept_properties, prop_name, PropertyDefinition.HOOKS
                        ),
                        part=PathPart.VALUE,
                    )
            if PropertyDefinition.COMPUTATIONS in prop_data:
                prop_def_val = prop_data[PropertyDefinition.COMPUTATIONS]
                if not isinstance(prop_def_val, dict):
                    raise CHSyntaxError(
                        f"The definition of property computations must be a JSON object, not {prop_def_val!r}",
                        self.location_id(
                            DomainConceptDefinition.domain_concept_properties,
                            prop_name,
                            PropertyDefinition.COMPUTATIONS,
                        ),
                        part=PathPart.VALUE,
                    )
            if PropertyDefinition.DEFAULT_INSTANCE_NAMING in prop_data:
                prop_def_val = prop_data[PropertyDefinition.DEFAULT_INSTANCE_NAMING]
                if not isinstance(prop_def_val, bool):
                    raise CHSyntaxError(
                        f"The definition of whether to name instances created as part of default values of this "
                        f"property with this instance's name must be a JSON boolean value, not {prop_def_val!r}",
                        self.location_id(
                            DomainConceptDefinition.domain_concept_properties,
                            prop_name,
                            PropertyDefinition.DEFAULT_INSTANCE_NAMING,
                        ),
                        part=PathPart.VALUE,
                    )
        # missing checks:
        #  - property names should be unique among all concepts (incl. defining a function with same name as a property)
        #    REQUIRES: all concept data to be initialized
        #    STRUCTURE CHECK
        #       - done in checker.py - check_after_parsing_concepts
        #  - property types (correctly template-instantiated type)
        #    TYPE CHECK
        #  - property constraints (correctly defined either subtype of property type of Variation-instantiation of
        #      property type)
        #    EXPRESSION CHECK
        #  - property constraints subtype of property types
        #    REQUIRES: constraint expression to be processed
        #    TYPE CHECK
        #  - hooks: correctly template-instantiated Function name, correct argument name
        #      (and type of this property must be a subtype of the argument's type)
        #      and the FunctionComposition hook expression
        #    STRUCTURE and TYPE CHECK
        #  - computations: valid FunctionComposition expression
        #    EXPRESSION CHECK
        #  - confidenceHalfDecayTime: valid Duration expression
        #    EXPRESSION CHECK
        #  - default: valid expression of property type
        #    EXPRESSION CHECK
        #  - assumptions: valid expression of type Variation of property type
        #    EXPRESSION CHECK
        #  - specialization keyword (structure and content; can't specialize ValueDomain, description or static-ness)!
        #    REQUIRES: all parent concepts to be processed (because they are the ones from which data is inherited)
        #    PROCESS AFTER: all concept data represented, all types parsed, all expressions processed
        #    STRUCTURE CHECK
        #  - whether DEFAULT_INSTANCE_NAME can be true (check that the property type contains instances in its type!)
        #    TYPE CHECK

        self.initialize_domain_concept_data(
            False, self.functions, self.function_specializations_for_sub, self.function_specializations_for_this
        )
        for func_name, func_data in self.functions.items():
            assert func_name != DomainConceptDefinition.domain_concept_specialization
            # write assertion because the data from "functions" comes directly from json deserialization of an object,
            #  which comes directly from a syntactically valid json file, where object keys are defined to be strings
            assert isinstance(func_name, str)
            if not check_ch_name(func_name, must_start_lowercase=True):
                raise CHSyntaxError(
                    f"Function names of domain concepts must be a lowercase-starting string, not {func_name!r}",
                    self.location_id(DomainConceptDefinition.domain_concept_functions, func_name),
                    part=PathPart.KEY,
                )
            if not isinstance(func_data, dict):
                raise CHSyntaxError(
                    f"The definition of domain concept functions must be a FunctionComposition value "
                    f"(serialized as a JSON object), not {func_data!r} for {func_name} of domain concept {self.name}!",
                    self.location_id(DomainConceptDefinition.domain_concept_functions, func_name),
                    part=PathPart.VALUE,
                )
            else:
                func_def_data_keys = set(func_data.keys())
                if not (func_def_data_keys <= DomainConceptDefinition.function_data_keys):
                    if not func_def_data_keys.isdisjoint(DomainConceptDefinition.function_data_keys):
                        extra_keys = func_def_data_keys - DomainConceptDefinition.function_data_keys
                        raise CHSyntaxError(
                            f"Found extra keys {extra_keys!r} in the domain concept data definition of function "
                            f"{func_name} for {self.name}",
                            self.location_id(DomainConceptDefinition.domain_concept_functions, func_name),
                            part=PathPart.VALUE,
                        )
                    # interpret as default value of the static property
                    self.functions[func_name] = {FunctionDefinition.STATIC: True, FunctionDefinition.DEFAULT: func_data}
        self.initialize_domain_concept_specialization_data(
            False,
            self.functions,
            self.function_specializations_for_sub,
            self.function_specializations_for_this,
            self.available_function_data,
            FunctionDefinition.SPECIALIZATION_KEYWORDS,
            FunctionDefinition.VALUE_DOMAIN,
        )
        for func_name, func_data in self.functions.items():
            for func_data_def_key, func_data_def_val in func_data.items():
                assert func_data_def_key in DomainConceptDefinition.function_data_keys
            if func_data == {}:
                func_data[FunctionDefinition.STATIC] = True
                func_data[FunctionDefinition.DEFAULT] = {}
            if FunctionDefinition.VALUE_DOMAIN not in func_data:
                func_data[FunctionDefinition.VALUE_DOMAIN] = (
                    DomainConceptDefinition.default_value_domain_type_of_domain_concept_functions
                )
            # check the type of each function definition keyword value!
            if FunctionDefinition.VALUE_DOMAIN in func_data:
                func_def_val = func_data[FunctionDefinition.VALUE_DOMAIN]
                if not isinstance(func_def_val, str):
                    raise CHSyntaxError(
                        f"The ValueDomain definition of functions must be a JSON string value, not {func_def_val!r}",
                        self.location_id(
                            DomainConceptDefinition.domain_concept_functions, func_name, FunctionDefinition.VALUE_DOMAIN
                        ),
                        part=PathPart.VALUE,
                    )
            # missing checks: the type check (subconcept of CustomFunction) of the defined domain concept function!
            if FunctionDefinition.DESCRIPTION in func_data:
                func_def_val = func_data[FunctionDefinition.DESCRIPTION]
                if not isinstance(func_def_val, str):
                    raise CHSyntaxError(
                        f"The definition of a function description must be a JSON string value, not {func_def_val!r}",
                        self.location_id(
                            DomainConceptDefinition.domain_concept_functions, func_name, FunctionDefinition.DESCRIPTION
                        ),
                        part=PathPart.VALUE,
                    )
            if FunctionDefinition.STATIC in func_data:
                func_def_val = func_data[FunctionDefinition.STATIC]
                if not isinstance(func_def_val, bool):
                    raise CHSyntaxError(
                        f"The marker of static functions must be a JSON boolean, not {func_def_val!r}",
                        self.location_id(
                            DomainConceptDefinition.domain_concept_functions, func_name, FunctionDefinition.STATIC
                        ),
                        part=PathPart.VALUE,
                    )
            if FunctionDefinition.DEFAULT in func_data:
                assert isinstance(func_data[FunctionDefinition.DEFAULT], dict)
        # missing checks:
        #  - check that the valueDomain of every domain concept function is a subconcept of CustomFunction
        #    REQUIRES: all concepts to be processed and the type validator to be initialized
        #    TYPE CHECK
        #       - done in checker.py - check_after_parsing_concepts
        #  - function names should be unique among all concepts (incl. defining a property with same name as a function)
        #    STRUCTURE CHECK
        #       - done in checker.py - check_after_parsing_concepts
        #  - CustomFunction expression (incl. types of argument and procedure expression)
        #    EXPRESSION CHECK
        #  - specialization keyword (structure and content; can't specialize ValueDomain, description or static-ness)!
        #    REQUIRES: all parent concepts to be processed (because they are the ones from which data is inherited)
        #    PROCESS AFTER: all concept data represented, all types parsed, all expressions processed
        #    STRUCTURE CHECK

        self.management = self.data.get(DomainConceptDefinition.domain_concept_management, {})
        if not isinstance(self.management, dict):
            raise CHSyntaxError(
                f"The definition of domain concept management data must be a JSON object, not {self.management!r} for "
                f"domain concept {self.name}!",
                self.location_id(DomainConceptDefinition.domain_concept_management),
                part=PathPart.VALUE,
            )
        management_keys = set(self.management.keys())
        if not (management_keys <= DomainConceptDefinition.management_data_keys):
            extra_keys = management_keys - DomainConceptDefinition.management_data_keys
            raise CHSyntaxError(
                f"Found extra keys {extra_keys!r} in the domain concept management data definition of {self.name}",
                self.location_id(DomainConceptDefinition.domain_concept_management),
                part=PathPart.VALUE,
            )
        for management_key, management_data in self.management.items():
            assert isinstance(management_key, str)
            if not isinstance(management_data, dict):
                raise CHSyntaxError(
                    f"The definition of {management_key!r} domain concept management data must be a JSON object, not "
                    f"{management_data!r} for domain concept {self.name}!",
                    self.location_id(DomainConceptDefinition.domain_concept_management, management_key),
                    part=PathPart.VALUE,
                )
        # missing checks:
        #  - "initialization" is a valid FunctionComposition expression (with variable context "instance")
        #    EXPRESSION CHECK
        #  - "consolidation" is a valid FunctionComposition expression (with variable context "instance")
        #    EXPRESSION CHECK
