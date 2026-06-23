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

from enum import Enum

from concept_hierarchy.definitions.concept_definition import ConceptDefinition
from concept_hierarchy.definitions.definition import LocationOfCheckData
from concept_hierarchy.definitions.utils import check_ch_name
from concept_hierarchy.errors import CHSemanticError, CHSyntaxError, LocationId, PathPart


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


class PropertyDefinition:
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


class FunctionDefinition:
    VALUE_DOMAIN = "valueDomain"
    DESCRIPTION = "description"
    STATIC = "static"
    DEFAULT = "default"
    SPECIALIZATION_KEYWORDS = [DEFAULT]
    ALL_DEFINITION_KEYWORDS = [VALUE_DOMAIN, DESCRIPTION, STATIC] + SPECIALIZATION_KEYWORDS


class ForPropertyOrFunction(Enum):
    FUNCTION = 0
    PROPERTY = 1


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

    def __init__(self, name: str, definition_data: object, definition_location_id: LocationId):
        super().__init__(name, definition_data, definition_location_id)

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

        self.is_shorthand_property_definition: set[str] = set()  # set of property names who use a shorthand definition
        self.is_shorthand_function_definition: set[str] = set()  # set of function names who use a shorthand definition

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
        domain_concept.is_shorthand_property_definition = set()
        domain_concept.is_shorthand_function_definition = set()
        return domain_concept

    def definition_type(self) -> str:
        return DomainConceptDefinition.domain_concept_name

    def definition_location(self) -> LocationId:
        return self.data_location_id

    def location_of_impl(self, *keywords: str) -> LocationOfCheckData:
        check_res = super().location_of_impl(*keywords)
        # check top-level properties, functions, management keys
        self.check_location_id(
            check_res,
            DomainConceptDefinition.definition_location(self) + [check_res.first_remaining],
            location_check=self.data,
            previous_location=ConceptDefinition.concept_definition_data,
            allow_start_at_this_location=True,
        )
        # try to consume property data
        properties_data = self.data.get(DomainConceptDefinition.domain_concept_properties, {})
        assert isinstance(properties_data, dict)
        # check property data
        self.check_location_id(
            check_res,
            DomainConceptDefinition.definition_location(self)
            + [DomainConceptDefinition.domain_concept_properties, check_res.first_remaining],
            location_check=properties_data,
            previous_location=DomainConceptDefinition.domain_concept_properties,
            allow_start_at_this_location=check_res.first_remaining
            != DomainConceptDefinition.domain_concept_specialization,
        )
        if check_res.check_successful:
            # => we now have consumed either a property name or the specializations keyword
            # try to consume property definition data and specializations data
            prop_name_or_specialization = check_res.last_consumed
            assert prop_name_or_specialization in properties_data
            prop_def_data = properties_data[prop_name_or_specialization]

            # The data for a property name can be either:
            #  - a string-value representing the value_domain type,
            #  - an expression defining the type constraint,
            #  - or the property definition dictionary
            if (
                prop_name_or_specialization in self.properties
                and prop_name_or_specialization in self.is_shorthand_property_definition
            ):
                if isinstance(prop_def_data, str):
                    if check_res.first_remaining != PropertyDefinition.VALUE_DOMAIN:
                        # search failed
                        return check_res
                    self._consume_remaining_keyword(check_res)
                    return check_res
                else:
                    assert isinstance(prop_def_data, dict)
                    if check_res.first_remaining in [PropertyDefinition.VALUE_DOMAIN, PropertyDefinition.CONSTRAINT]:
                        self._consume_remaining_keyword(check_res)
                    return check_res

            # check prop_def_key data or forSub prop_name data or forThis keyword
            self.check_location_id(
                check_res,
                check_res.current_location_id + [check_res.first_remaining],
                location_check=prop_def_data,
                previous_location=prop_name_or_specialization,
                allow_start_at_this_location=False,
            )
            if not check_res.check_successful:
                # Stop processing location
                return check_res
            prop_def_key_or_for_sub_prop_name_or_for_this = check_res.last_consumed
            # all property definition keywords except hooks have no more data: finish processing
            if (
                prop_def_key_or_for_sub_prop_name_or_for_this in PropertyDefinition.ALL_DEFINITION_KEYWORDS
                and prop_def_key_or_for_sub_prop_name_or_for_this != PropertyDefinition.HOOKS
            ):
                return check_res
            sub_data = prop_def_data[prop_def_key_or_for_sub_prop_name_or_for_this]
            assert isinstance(sub_data, dict)
            # prop_def_key can only be hooks at this point
            # check prop_name hook_f_name data, forSub prop_def_key data or forThis prop_name data
            self.check_location_id(
                check_res,
                check_res.current_location_id + [check_res.first_remaining],
                location_check=sub_data,
                previous_location=prop_def_key_or_for_sub_prop_name_or_for_this,
                allow_start_at_this_location=False,
            )
            if not check_res.check_successful:
                return check_res
            hook_f_name_or_for_sub_prop_def_key_or_for_this_prop_name = check_res.last_consumed
            # stop processing if this was the forSub property definition key (except for hooks)
            if (
                hook_f_name_or_for_sub_prop_def_key_or_for_this_prop_name in PropertyDefinition.ALL_DEFINITION_KEYWORDS
                and hook_f_name_or_for_sub_prop_def_key_or_for_this_prop_name != PropertyDefinition.HOOKS
            ):
                return check_res
            sub_sub_data = sub_data[hook_f_name_or_for_sub_prop_def_key_or_for_this_prop_name]
            assert isinstance(sub_sub_data, dict)
            # check prop_name hook_f_arg data, forSub hook_f_name data or forThis prop_def_key data
            self.check_location_id(
                check_res,
                check_res.current_location_id + [check_res.first_remaining],
                location_check=sub_sub_data,
                previous_location=hook_f_name_or_for_sub_prop_def_key_or_for_this_prop_name,
                allow_start_at_this_location=False,
            )
            if not check_res.check_successful:
                return check_res
            hook_f_arg_or_for_sub_hook_f_name_or_for_this_prop_def_key = check_res.last_consumed
            # stop if for_this_prop_def_key != hooks or if the last consumed is "hook_f_arg"
            #   check if the last consumed is "hook_f_arg" by prop_name_or_specialization != "_specializations"
            if (
                hook_f_arg_or_for_sub_hook_f_name_or_for_this_prop_def_key in PropertyDefinition.ALL_DEFINITION_KEYWORDS
                and hook_f_arg_or_for_sub_hook_f_name_or_for_this_prop_def_key != PropertyDefinition.HOOKS
            ) or prop_name_or_specialization != DomainConceptDefinition.domain_concept_specialization:
                return check_res
            sub_sub_sub_data = sub_sub_data[hook_f_arg_or_for_sub_hook_f_name_or_for_this_prop_def_key]
            assert isinstance(sub_sub_sub_data, dict)
            # check forSub hook_f_arg data or forThis hook_f_name data
            self.check_location_id(
                check_res,
                check_res.current_location_id + [check_res.first_remaining],
                location_check=sub_sub_sub_data,
                previous_location=hook_f_arg_or_for_sub_hook_f_name_or_for_this_prop_def_key,
                allow_start_at_this_location=False,
            )
            if not check_res.check_successful:
                return check_res
            for_sub_hook_f_arg_or_for_this_hook_f_name = check_res.last_consumed
            # finish processing for the forSub chain:
            #  forSub is finished if prop_def_key_or_for_sub_prop_name_or_for_this is not the "_forThis" keyword
            if (
                prop_def_key_or_for_sub_prop_name_or_for_this
                != DomainConceptDefinition.domain_concept_specialization_for_this
            ):
                return check_res
            sub_sub_sub_sub_data = sub_sub_sub_data[for_sub_hook_f_arg_or_for_this_hook_f_name]
            assert isinstance(sub_sub_sub_sub_data, dict)
            # check forThis hook_f_arg
            # final check: no matter if successful or not, this is the end-leaf of the data chain for specializations
            return self.check_location_id(
                check_res,
                check_res.current_location_id + [check_res.first_remaining],
                location_check=sub_sub_sub_sub_data,
                previous_location=for_sub_hook_f_arg_or_for_this_hook_f_name,
                allow_start_at_this_location=False,
            )

        # try to consume functions data
        functions_data = self.data.get(DomainConceptDefinition.domain_concept_functions, {})
        assert isinstance(functions_data, dict)
        # check functions data
        self.check_location_id(
            check_res,
            DomainConceptDefinition.definition_location(self)
            + [DomainConceptDefinition.domain_concept_functions, check_res.first_remaining],
            location_check=functions_data,
            previous_location=DomainConceptDefinition.domain_concept_functions,
            allow_start_at_this_location=check_res.first_remaining
            != DomainConceptDefinition.domain_concept_specialization,
        )
        if check_res.check_successful:
            # => we now have consumed either a function name or the specializations keyword
            # Watch out because function specialization data can be at the top-level (after the function name!)
            #   It can skip the func_def_key, take this into account in the processing steps below!
            #   Also, compared to the property chain above, there are no hooks that can go deeper with keywords!
            # try to consume function definition data and specializations data
            func_name_or_specialization = check_res.last_consumed
            assert func_name_or_specialization in functions_data
            func_def_data = functions_data[func_name_or_specialization]
            # check that func_def_data is not a FunctionComposition value at func_name
            if (
                func_name_or_specialization != DomainConceptDefinition.domain_concept_specialization
                and DomainConceptDefinition.looks_like_function_instantiation(
                    func_def_data, accept_specialization=False
                )
            ):
                # consume default, static, and valueDomain key
                if check_res.remaining_keywords[0] in [
                    FunctionDefinition.VALUE_DOMAIN,
                    FunctionDefinition.STATIC,
                    FunctionDefinition.DEFAULT,
                ]:
                    self._consume_remaining_keyword(check_res)
                return check_res
            # check func_def_key data or forSub func_name data or forThis keyword
            self.check_location_id(
                check_res,
                check_res.current_location_id + [check_res.first_remaining],
                location_check=func_def_data,
                previous_location=func_name_or_specialization,
                allow_start_at_this_location=False,
            )
            if not check_res.check_successful:
                # Stop processing location
                return check_res
            func_def_key_or_for_sub_func_name_or_for_this = check_res.last_consumed
            # all function definition keywords have no more data: finish processing
            if func_name_or_specialization != DomainConceptDefinition.domain_concept_specialization:
                return check_res
            sub_data = func_def_data[func_def_key_or_for_sub_func_name_or_for_this]
            # check that func_def_data is not a FunctionComposition or INHERIT_FROM_KEYWORD value at forSub func_name
            if (
                func_def_key_or_for_sub_func_name_or_for_this
                != DomainConceptDefinition.domain_concept_specialization_for_this
                and DomainConceptDefinition.looks_like_function_instantiation(sub_data, accept_specialization=True)
            ):
                if check_res.remaining_keywords[0] == FunctionDefinition.DEFAULT:
                    self._consume_remaining_keyword(check_res)
                return check_res
            assert isinstance(sub_data, dict)
            # check forSub func_def_key data or forThis func_name data
            self.check_location_id(
                check_res,
                check_res.current_location_id + [check_res.first_remaining],
                location_check=sub_data,
                previous_location=func_def_key_or_for_sub_func_name_or_for_this,
                allow_start_at_this_location=False,
            )
            if not check_res.check_successful:
                return check_res
            for_sub_func_def_key_or_for_this_func_name = check_res.last_consumed
            # stop processing if this was the forSub function definition key
            if (
                func_def_key_or_for_sub_func_name_or_for_this
                != DomainConceptDefinition.domain_concept_specialization_for_this
            ):
                return check_res
            sub_sub_data = sub_data[for_sub_func_def_key_or_for_this_func_name]
            # stop processing if this is a FunctionComposition or INHERIT_FROM_KEYWORD definition at forThis func_name
            if DomainConceptDefinition.looks_like_function_instantiation(sub_sub_data, accept_specialization=True):
                if check_res.remaining_keywords[0] == FunctionDefinition.DEFAULT:
                    self._consume_remaining_keyword(check_res)
                return check_res
            assert isinstance(sub_sub_data, dict)
            # check func_name hook_f_arg data, forSub hook_f_name data or forThis func_def_key data
            return self.check_location_id(
                check_res,
                check_res.current_location_id + [check_res.first_remaining],
                location_check=sub_sub_data,
                previous_location=for_sub_func_def_key_or_for_this_func_name,
                allow_start_at_this_location=False,
            )

        # try to consume management data
        management_data = self.data.get(DomainConceptDefinition.domain_concept_management, {})
        assert isinstance(management_data, dict)
        # check management data
        self.check_location_id(
            check_res,
            DomainConceptDefinition.definition_location(self)
            + [DomainConceptDefinition.domain_concept_management, check_res.first_remaining],
            location_check=management_data,
            previous_location=DomainConceptDefinition.domain_concept_management,
            allow_start_at_this_location=True,
        )
        # Stop after management data, because the initialization and consolidation data values are FunctionCompositions
        return check_res

    def initialize_domain_concept_data(
        self,
        for_either_properties_or_functions: ForPropertyOrFunction,
        data_container: dict[str, dict[str, object]],
        data_specialization_for_sub: dict[str, dict[str, object]],
        data_specialization_for_this: dict[str, dict[str, tuple[str | object, bool]]],
    ):
        """Modify data_container, data_specialization_for_sub, and data_specialization_for_this in place!"""
        data_container.clear()
        data_specialization_for_sub.clear()
        data_specialization_for_this.clear()
        data_type = "property" if for_either_properties_or_functions.value else "function"
        data_type_plural = "properties" if for_either_properties_or_functions.value else "functions"
        def_location = (
            DomainConceptDefinition.domain_concept_properties
            if for_either_properties_or_functions.value
            else DomainConceptDefinition.domain_concept_functions
        )

        def_location_data = self.data.get(def_location, {})
        if not isinstance(def_location_data, dict):
            raise CHSyntaxError(
                f"The definition of domain concept {data_type_plural} must be a JSON object, not "
                f"{data_container!r} as encountered at domain concept {self.name}",
                location_id=self.location_id(def_location),
                part=PathPart.VALUE,
            )
        data_container.update(def_location_data)
        def_specialization = data_container.pop(DomainConceptDefinition.domain_concept_specialization, {})
        if not isinstance(def_specialization, dict):
            raise CHSyntaxError(
                f"The specialization content of {self.definition_type()} {data_type_plural} must be a JSON object, not "
                f"{def_specialization!r}"
                f'\nThe "{DomainConceptDefinition.domain_concept_specialization}" JSON object '
                f"must contain the {data_type} name as key and, as value, a JSON object mapping the specializable "
                f"definition keys of {self.definition_type()} {data_type_plural} to  their specialization value.\nThese"
                f" values, specified in the top-level specialization definition will be passed on to subconcepts.\nTo "
                f"specialize the values for (instances of) this concept only, the "
                f'"{DomainConceptDefinition.domain_concept_specialization_for_this}" key must be specified at the top-'
                f'level "{DomainConceptDefinition.domain_concept_specialization}" JSON object.\nThe structure and '
                f'content of the "{DomainConceptDefinition.domain_concept_specialization_for_this}" JSON object is the '
                f"same as at the top-level (just that it can not contain the "
                f'"{DomainConceptDefinition.domain_concept_specialization_for_this}" key again :)',
                location_id=self.location_id(def_location, DomainConceptDefinition.domain_concept_specialization),
                part=PathPart.VALUE,
            )
        def_specialization_for_this = def_specialization.pop(
            DomainConceptDefinition.domain_concept_specialization_for_this, {}
        )
        if not isinstance(def_specialization_for_this, dict):
            raise CHSyntaxError(
                f"The specialization content for {data_type_plural} of (the instances of) this {self.definition_type()}"
                f" must be a JSON object, not {def_specialization_for_this!r}"
                f'\nThe "{DomainConceptDefinition.domain_concept_specialization}" JSON object '
                f"must contain the {data_type} name as key and, as value, a JSON object mapping the specializable "
                f"definition keys of {self.definition_type()} {data_type_plural} to  their specialization value.\nThese"
                f" values, specified in the top-level specialization definition will be passed on to subconcepts.\nTo "
                f"specialize the values for (instances of) this concept only, the "
                f'"{DomainConceptDefinition.domain_concept_specialization_for_this}" key must be specified at the top-'
                f'level "{DomainConceptDefinition.domain_concept_specialization}" JSON object.\nThe structure and '
                f'content of the "{DomainConceptDefinition.domain_concept_specialization_for_this}" JSON object is the '
                f"same as at the top-level (just that it can not contain the "
                f'"{DomainConceptDefinition.domain_concept_specialization_for_this}" key again :)',
                location_id=self.location_id(
                    def_location,
                    DomainConceptDefinition.domain_concept_specialization,
                    DomainConceptDefinition.domain_concept_specialization_for_this,
                ),
                part=PathPart.VALUE,
            )
        # check structure for specialization-forThis
        for name, spec_data in def_specialization_for_this.items():
            data_specialization_for_this[name] = {}
            if for_either_properties_or_functions == ForPropertyOrFunction.FUNCTION and (
                DomainConceptDefinition.looks_like_function_instantiation(spec_data, accept_specialization=True)
            ):
                # For functions that can be directly instantiated // or have the INHERIT_FROM_KEYWORD specified directly
                #   without specifying the DEFAULT definition keyword
                #   This is only possible because there is only one specializable def. key for domain concept functions
                if isinstance(spec_data, str) and not spec_data.startswith(INHERIT_FROM_KEYWORD):
                    raise CHSyntaxError(
                        f"When defining the specialization value of a {self.definition_type()} function, you can choose"
                        f" from which direct parent concept to use the specialized value passed on for subconcepts."
                        f'\nUse the "{INHERIT_FROM_KEYWORD}<NameOfDirectParentConcept>" syntax to specify using that '
                        f"value; got {spec_data!r}",
                        location_id=self.location_id(
                            def_location,
                            DomainConceptDefinition.domain_concept_specialization,
                            DomainConceptDefinition.domain_concept_specialization_for_this,
                            name,
                        ),
                        part=PathPart.VALUE,
                    )
                # Assumption that data defined here is inherited.
                # Correct the False value below in
                #  :func:`initialize_domain_concept_specialization_data_from_defined_data`
                #  if name is a property/function defined in this concept
                data_specialization_for_this[name][FunctionDefinition.DEFAULT] = (spec_data, False)
            else:
                if not isinstance(spec_data, dict):
                    raise CHSyntaxError(
                        f"The specialization content of {self.definition_type()} {data_type} data must be a JSON "
                        f"object. Got {spec_data!r} at {data_type} {name} in {self.name}",
                        location_id=self.location_id(
                            def_location,
                            DomainConceptDefinition.domain_concept_specialization,
                            DomainConceptDefinition.domain_concept_specialization_for_this,
                            name,
                        ),
                        part=PathPart.VALUE,
                    )
                for def_key, def_data in spec_data.items():
                    # Assumption that data defined here is inherited.
                    # Correct the False value below in
                    #  :func:`initialize_domain_concept_specialization_data_from_defined_data`
                    #  if name is a property/function defined in this concept
                    data_specialization_for_this[name][def_key] = (def_data, False)
        # check structure for specialization-forSub
        for name, spec_data in def_specialization.items():
            data_specialization_for_sub[name] = {}
            if for_either_properties_or_functions == ForPropertyOrFunction.FUNCTION and (
                DomainConceptDefinition.looks_like_function_instantiation(spec_data) or isinstance(spec_data, str)
            ):
                # For functions that can be directly instantiated // or have the INHERIT_FROM_KEYWORD specified directly
                #   without specifying the DEFAULT definition keyword
                #   This is only possible because there is only one specializable def. key for domain concept functions
                data_specialization_for_sub[name][FunctionDefinition.DEFAULT] = spec_data
            else:
                if not isinstance(spec_data, dict):
                    raise CHSyntaxError(
                        f"The specialization content of {self.definition_type()} {data_type} data must be a JSON "
                        f"object. Got {spec_data!r} at {data_type} {name} in {self.name}",
                        location_id=self.location_id(
                            def_location,
                            DomainConceptDefinition.domain_concept_specialization,
                            name,
                        ),
                        part=PathPart.VALUE,
                    )
                data_specialization_for_sub[name] = spec_data

    def initialize_domain_concept_specialization_data_from_defined_data(
        self,
        for_either_properties_or_functions: ForPropertyOrFunction,
        data_container: dict[str, dict[str, object]],
        data_specialization_for_sub: dict[str, dict[str, object]],
        data_specialization_for_this: dict[str, dict[str, tuple[str | object, bool]]],
        available_data: dict[str, dict[str, str]],
        specializable_keywords: list[str],
        value_domain_keyword: str,
    ):
        """Modify data_container, data_specialization_for_sub, and data_specialization_for_this in place!"""
        data_type = "property" if for_either_properties_or_functions.value else "function"
        def_location = (
            DomainConceptDefinition.domain_concept_properties
            if for_either_properties_or_functions.value
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
                for spec_for_this_def_key, spec_for_this_def_data in data_specialization_for_this[name].items():
                    assert isinstance(spec_for_this_def_data, tuple) and spec_for_this_def_data[1] is False, (
                        f"At concept {self.name}: {spec_for_this_def_data!r}"
                    )
                    data_specialization_for_this[name][spec_for_this_def_key] = (spec_for_this_def_data[0], True)
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

    @staticmethod
    def check_property_data_types(
        c: DomainConceptDefinition,
        prop_name: str,
        prop_data: dict[str, object],
        for_specialization: bool,
        specialize_for_sub: bool = False,
    ):
        if for_specialization:
            location_id = c.location_id(
                DomainConceptDefinition.domain_concept_properties, DomainConceptDefinition.domain_concept_specialization
            )
            if not specialize_for_sub:
                location_id.append(DomainConceptDefinition.domain_concept_specialization_for_this)
            location_id.append(prop_name)
            for def_key in PropertyDefinition.ALL_DEFINITION_KEYWORDS:
                if def_key not in PropertyDefinition.SPECIALIZATION_KEYWORDS and def_key in prop_data:
                    raise CHSemanticError(
                        f"{c.definition_type()} property definition keyword {def_key} is not specializable!"
                        f"\nPlease remove it from the specialization specification!",
                        location_id=location_id,
                        part=PathPart.KEY,
                    )
        else:
            location_id = c.location_id(DomainConceptDefinition.domain_concept_properties, prop_name)
        # Don't check constraints, default, assumed, and confidence values because they are the serialization of
        #  ValueDomains  (i.e. any JSON value).
        # only allow string values if this is a specialization, is specializable and starts with INHERIT_FROM_KEYWORD
        if PropertyDefinition.VALUE_DOMAIN in prop_data:
            prop_def_val = prop_data[PropertyDefinition.VALUE_DOMAIN]
            if not isinstance(prop_def_val, str):
                raise CHSyntaxError(
                    f"The ValueDomain definition of properties must be a JSON string value, not {prop_def_val!r}",
                    location_id=location_id + [PropertyDefinition.VALUE_DOMAIN],
                    part=PathPart.VALUE,
                )
        if PropertyDefinition.DESCRIPTION in prop_data:
            prop_def_val = prop_data[PropertyDefinition.DESCRIPTION]
            if not isinstance(prop_def_val, str):
                raise CHSyntaxError(
                    f"The definition of a property description must be a JSON string value, not {prop_def_val!r}",
                    location_id=location_id + [PropertyDefinition.DESCRIPTION],
                    part=PathPart.VALUE,
                )
        if PropertyDefinition.STATIC in prop_data:
            prop_def_val = prop_data[PropertyDefinition.STATIC]
            if not isinstance(prop_def_val, bool):
                raise CHSyntaxError(
                    f"The marker of static properties must be a JSON boolean, not {prop_def_val!r}",
                    location_id=location_id + [PropertyDefinition.STATIC],
                    part=PathPart.VALUE,
                )
        if PropertyDefinition.HOOKS in prop_data:
            prop_def_val = prop_data[PropertyDefinition.HOOKS]
            if for_specialization and not specialize_for_sub:
                assert isinstance(prop_def_val, tuple)
                prop_def_val = prop_def_val[0]
            if not isinstance(prop_def_val, dict):
                # only allow string-values if this is a specialization
                if (
                    not for_specialization
                    or not isinstance(prop_def_val, str)
                    or not prop_def_val.startswith(INHERIT_FROM_KEYWORD)
                ):
                    raise CHSyntaxError(
                        f"The structure of hooks should be a JSON object mapping template-instantiated Function types "
                        f"to a JSON object mapping arguments of the template-instantiated Function to "
                        f"FunctionComposition values of the hook.\nExpected a JSON object as the definition of property"
                        f" hooks, not {prop_def_val!r}",
                        location_id=location_id + [PropertyDefinition.HOOKS],
                        part=PathPart.VALUE,
                    )
            else:
                # this is a dictionary: check substructure
                for hook_f_name, hook_f_data in prop_def_val.items():
                    # assertion, not check because this is a key of a JSON object
                    assert isinstance(hook_f_name, str)
                    # missing checks:
                    #  - check that hook_f_name is a correctly template-instantiated Function type!
                    #    REQUIRES
                    #    TYPE CHECK
                    if not isinstance(hook_f_data, dict):
                        raise CHSyntaxError(
                            f"The structure of hooks should be a JSON object mapping template-instantiated Function "
                            f"types to a JSON object mapping arguments of the template-instantiated Function to "
                            f"FunctionComposition values of the hook.\nExpected a JSON object at Function {hook_f_name}"
                            f", got {hook_f_data!r}!",
                            location_id=location_id + [PropertyDefinition.HOOKS, hook_f_name],
                            part=PathPart.VALUE,
                        )
                    for hook_f_arg, hook_f_procedure in hook_f_data.items():
                        # assertion, not check because this is a key of a JSON object
                        assert isinstance(hook_f_arg, str)
                        # missing checks:
                        #  - check that hook_f_arg is an argument of the Function
                        #    REQUIRES all Function data to be initialized
                        #    STRUCTURE CHECK
                        #  - check that the type of hook_f_arg is a parent type of the property type
                        #       on which the hook is attached
                        #    REQUIRES types to be processable and validatable
                        #    TYPE CHECK
                        if not isinstance(hook_f_procedure, dict):
                            raise CHSyntaxError(
                                f"The structure of hooks should be a JSON object mapping template-instantiated Function"
                                f" types to a JSON object mapping arguments of the template-instantiated Function to "
                                f"FunctionComposition values of the hook.\nExpected a JSON object as "
                                f"FunctionComposition value, not {hook_f_procedure!r}",
                                location_id=location_id + [PropertyDefinition.HOOKS, hook_f_name, hook_f_arg],
                                part=PathPart.VALUE,
                            )
                        # missing checks:
                        #  - check that hook_f_procedure is a valid FunctionComposition expression
                        #    EXPRESSION CHECK
        if PropertyDefinition.COMPUTATIONS in prop_data:
            prop_def_val = prop_data[PropertyDefinition.COMPUTATIONS]
            if for_specialization and not specialize_for_sub:
                assert isinstance(prop_def_val, tuple)
                prop_def_val = prop_def_val[0]
            if not isinstance(prop_def_val, dict):
                # only allow string-values if this is a specialization
                if not for_specialization or (
                    not isinstance(prop_def_val, str) or not prop_def_val.startswith(INHERIT_FROM_KEYWORD)
                ):
                    raise CHSyntaxError(
                        f"The definition of property computations must be a JSON object, not {prop_def_val!r}",
                        location_id=location_id + [PropertyDefinition.COMPUTATIONS],
                        part=PathPart.VALUE,
                    )
        if PropertyDefinition.DEFAULT_INSTANCE_NAMING in prop_data:
            prop_def_val = prop_data[PropertyDefinition.DEFAULT_INSTANCE_NAMING]
            if for_specialization and not specialize_for_sub:
                assert isinstance(prop_def_val, tuple)
                prop_def_val = prop_def_val[0]
            if not isinstance(prop_def_val, bool):
                # only allow string-values if this is a specialization
                if not for_specialization or (
                    not isinstance(prop_def_val, str) or not prop_def_val.startswith(INHERIT_FROM_KEYWORD)
                ):
                    raise CHSyntaxError(
                        f"The definition of whether to name instances created as part of default values of this "
                        f"property with this instance's name must be a JSON boolean value, not {prop_def_val!r}",
                        location_id=location_id + [PropertyDefinition.DEFAULT_INSTANCE_NAMING],
                        part=PathPart.VALUE,
                    )

    @staticmethod
    def check_function_data_types(
        c: DomainConceptDefinition,
        func_name: str,
        func_data: dict[str, object],
        for_specialization: bool,
        specialize_for_sub: bool = False,
    ):
        if for_specialization:
            location_id = c.location_id(
                DomainConceptDefinition.domain_concept_functions, DomainConceptDefinition.domain_concept_specialization
            )
            if not specialize_for_sub:
                location_id.append(DomainConceptDefinition.domain_concept_specialization_for_this)
            location_id.append(func_name)
            for def_key in FunctionDefinition.ALL_DEFINITION_KEYWORDS:
                if def_key not in FunctionDefinition.SPECIALIZATION_KEYWORDS and def_key in func_data:
                    raise CHSemanticError(
                        f"{c.definition_type()} function definition keyword {def_key} is not specializable!"
                        f"\nPlease remove it from the specialization specification!",
                        location_id=location_id,
                        part=PathPart.KEY,
                    )
        else:
            location_id = c.location_id(DomainConceptDefinition.domain_concept_functions, func_name)
        # check the type of each function definition keyword value!
        # only allow string values if this is a specialization, is specializable and starts with INHERIT_FROM_KEYWORD
        if FunctionDefinition.VALUE_DOMAIN in func_data:
            func_def_val = func_data[FunctionDefinition.VALUE_DOMAIN]
            if not isinstance(func_def_val, str):
                raise CHSyntaxError(
                    f"The ValueDomain definition of functions must be a JSON string value, not {func_def_val!r}",
                    location_id=location_id + [FunctionDefinition.VALUE_DOMAIN],
                    part=PathPart.VALUE,
                )
        if FunctionDefinition.DESCRIPTION in func_data:
            func_def_val = func_data[FunctionDefinition.DESCRIPTION]
            if not isinstance(func_def_val, str):
                raise CHSyntaxError(
                    f"The definition of a function description must be a JSON string value, not {func_def_val!r}",
                    location_id=location_id + [FunctionDefinition.DESCRIPTION],
                    part=PathPart.VALUE,
                )
        if FunctionDefinition.STATIC in func_data:
            func_def_val = func_data[FunctionDefinition.STATIC]
            if not isinstance(func_def_val, bool):
                raise CHSyntaxError(
                    f"The marker of static functions must be a JSON boolean, not {func_def_val!r}",
                    location_id=location_id + [FunctionDefinition.STATIC],
                    part=PathPart.VALUE,
                )
        if FunctionDefinition.DEFAULT in func_data:
            func_def_val = func_data[FunctionDefinition.DEFAULT]
            if for_specialization and not specialize_for_sub:
                assert isinstance(func_def_val, tuple)
                func_def_val = func_def_val[0]
            if not isinstance(func_def_val, dict):
                # only allow string-values if this is a specialization
                if (
                    not for_specialization
                    or not isinstance(func_def_val, str)
                    or not func_def_val.startswith(INHERIT_FROM_KEYWORD)
                ):
                    raise CHSemanticError(
                        f"{c.definition_type()} function specialization values must be either:"
                        f"\n - FunctionComposition values (i.e. the procedure of a "
                        f"{DomainConceptDefinition.default_value_domain_type_of_domain_concept_functions} without any "
                        f"input arguments),\n - the instantiation of a "
                        f"{DomainConceptDefinition.default_value_domain_type_of_domain_concept_functions} (i.e. by "
                        f"specifying its procedure and optional interface)\n - or a JSON object defining the only "
                        f"specializable definition key for {c.definition_type()} functions: the "
                        f'"{FunctionDefinition.DEFAULT}" value (which can be any of the above).\n\tGot '
                        f"{func_def_val!r}!",
                        location_id=location_id,
                        part=PathPart.VALUE,
                    )

    @staticmethod
    def looks_like_function_instantiation(func_data: object, accept_specialization: bool = False) -> bool:
        if not isinstance(func_data, dict):
            return accept_specialization and isinstance(func_data, str)
        func_def_data_keys = set(func_data.keys())
        if not (func_def_data_keys <= DomainConceptDefinition.function_data_keys):
            if not func_def_data_keys.isdisjoint(DomainConceptDefinition.function_data_keys):
                return False
        return True

    def concept_data_check(self):
        data_keys: set[str] = set(self.data.keys())
        if not (data_keys <= DomainConceptDefinition.domain_concept_data_keys):
            extra_keys = data_keys - DomainConceptDefinition.domain_concept_data_keys
            raise CHSyntaxError(
                f"Found extra keys {extra_keys!r} in the domain concept data definition of {self.name}",
                location_id=self.location_id(),
                part=PathPart.VALUE,
            )

        has_properties = DomainConceptDefinition.domain_concept_properties in self.data
        has_functions = DomainConceptDefinition.domain_concept_functions in self.data
        has_management = DomainConceptDefinition.domain_concept_management in self.data
        if not self.is_root_concept and not has_properties and not has_functions and not has_management:
            raise CHSemanticError(
                f"Found a domain concept with no data defined: {self.name!r}", self.location_id(), part=PathPart.VALUE
            )

        self.initialize_domain_concept_data(
            ForPropertyOrFunction.PROPERTY,
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
                    location_id=self.location_id(DomainConceptDefinition.domain_concept_properties, prop_name),
                    part=PathPart.KEY,
                )
            if not isinstance(prop_data, (dict, str)):
                raise CHSyntaxError(
                    f"The definition of domain concept properties must be a JSON object or string "
                    f"(that defines its ValueDomain), not {prop_data!r} for {prop_name} of domain concept {self.name}!",
                    location_id=self.location_id(DomainConceptDefinition.domain_concept_properties, prop_name),
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
                            location_id=self.location_id(DomainConceptDefinition.domain_concept_properties, prop_name),
                            part=PathPart.VALUE,
                        )
                    # if prop_def does not contain any property-definition-keys, interpret as the definition of an
                    #  expression that defines the constraint & type of the property
                    self.properties[prop_name] = {PropertyDefinition.CONSTRAINT: prop_data}
                    self.is_shorthand_property_definition.add(prop_name)
            else:
                self.properties[prop_name] = {PropertyDefinition.VALUE_DOMAIN: prop_data}
                self.is_shorthand_property_definition.add(prop_name)
        self.initialize_domain_concept_specialization_data_from_defined_data(
            ForPropertyOrFunction.PROPERTY,
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
                    location_id=self.location_id(DomainConceptDefinition.domain_concept_properties, prop_name),
                    part=PathPart.VALUE,
                )
            # check the type of each property definition keyword value!
            DomainConceptDefinition.check_property_data_types(self, prop_name, prop_data, False)
        # missing checks:
        #  - property names should be unique among all concepts (incl. defining a function with same name as a property)
        #    REQUIRES: all concept data to be initialized
        #    STRUCTURE CHECK
        #       - done in checker.py - check_after_parsing_concepts
        #  - property types (correctly template-instantiated type)
        #    TYPE CHECK
        #    EXPRESSION CHECK
        #       - partially done in concept_hierarchy_type_checks.py
        #       - still need to parse constraints
        #  - property types should be subconcepts of ValueDomain
        #    TYPE CHECK
        #    EXPRESSION CHECK
        #       - partially done in concept_hierarchy_type_checks.py
        #       - still need to parse constraints
        #  - property constraints (correctly defined either subtype of property type of Variation-instantiation of
        #      property type)
        #    EXPRESSION CHECK
        #  - property constraints subtype of property types
        #    REQUIRES: constraint expression to be processed
        #    TYPE CHECK
        #    EXPRESSION CHECK
        #  - hooks: correctly template-instantiated Function name, correct argument name
        #      (and type of this property must be a subtype of the argument's type)
        #      and the FunctionComposition hook expression
        #    STRUCTURE CHECK
        #    TYPE CHECK
        #    EXPRESSION CHECK
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
        #    STRUCTURE CHECK
        #       - done in domain_concept_specialization_checks
        #  - whether CONFIDENCE can be used (check that the Duration type is defined in the ConceptHierarchy
        #    STRUCTURE CHECK
        #       - done in checker.py - check_after_parsing_concepts
        #  - whether DEFAULT_INSTANCE_NAME can be true (check that the property type contains instances in its type!)
        #    TYPE CHECK

        self.initialize_domain_concept_data(
            ForPropertyOrFunction.FUNCTION,
            self.functions,
            self.function_specializations_for_sub,
            self.function_specializations_for_this,
        )
        for func_name, func_data in self.functions.items():
            assert func_name != DomainConceptDefinition.domain_concept_specialization
            # write assertion because the data from "functions" comes directly from json deserialization of an object,
            #  which comes directly from a syntactically valid json file, where object keys are defined to be strings
            assert isinstance(func_name, str)
            if not check_ch_name(func_name, must_start_lowercase=True):
                raise CHSyntaxError(
                    f"Function names of domain concepts must be a lowercase-starting string, not {func_name!r}",
                    location_id=self.location_id(DomainConceptDefinition.domain_concept_functions, func_name),
                    part=PathPart.KEY,
                )
            if not isinstance(func_data, dict):
                raise CHSyntaxError(
                    f"The definition of domain concept functions must be a FunctionComposition value "
                    f"(serialized as a JSON object), not {func_data!r} for {func_name} of domain concept {self.name}!",
                    location_id=self.location_id(DomainConceptDefinition.domain_concept_functions, func_name),
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
                            location_id=self.location_id(DomainConceptDefinition.domain_concept_functions, func_name),
                            part=PathPart.VALUE,
                        )
                    # interpret as default value of the static property
                    self.functions[func_name] = {FunctionDefinition.STATIC: True, FunctionDefinition.DEFAULT: func_data}
                    self.is_shorthand_function_definition.add(func_name)
        self.initialize_domain_concept_specialization_data_from_defined_data(
            ForPropertyOrFunction.FUNCTION,
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
            DomainConceptDefinition.check_function_data_types(self, func_name, func_data, False)
        # missing checks:
        #  - check that the valueDomain of every domain concept function is a subconcept of CustomFunction
        #    REQUIRES: all concepts to be processed and the type validator to be initialized
        #    TYPE CHECK
        #       - done in concept_hierarchy_type_checks.py
        #  - function names should be unique among all concepts (incl. defining a property with same name as a function)
        #    STRUCTURE CHECK
        #       - done in checker.py - check_after_parsing_concepts
        #  - CustomFunction expression (incl. types of argument and procedure expression)
        #    EXPRESSION CHECK
        #  - specialization keyword (structure and content; can't specialize ValueDomain, description or static-ness)!
        #    REQUIRES: all parent concepts to be processed (because they are the ones from which data is inherited)
        #    PROCESS AFTER: all concept data represented, all types parsed, all expressions processed
        #    STRUCTURE CHECK
        #       - done in domain_concept_specialization_checks

        self.management = self.data.get(DomainConceptDefinition.domain_concept_management, {})
        if not isinstance(self.management, dict):
            raise CHSyntaxError(
                f"The definition of domain concept management data must be a JSON object, not {self.management!r} for "
                f"domain concept {self.name}!",
                location_id=self.location_id(DomainConceptDefinition.domain_concept_management),
                part=PathPart.VALUE,
            )
        management_keys = set(self.management.keys())
        if not (management_keys <= DomainConceptDefinition.management_data_keys):
            extra_keys = management_keys - DomainConceptDefinition.management_data_keys
            raise CHSyntaxError(
                f"Found extra keys {extra_keys!r} in the domain concept management data definition of {self.name}",
                location_id=self.location_id(DomainConceptDefinition.domain_concept_management),
                part=PathPart.VALUE,
            )
        for management_key, management_data in self.management.items():
            assert isinstance(management_key, str)
            if not isinstance(management_data, dict):
                raise CHSyntaxError(
                    f"The definition of {management_key!r} domain concept management data must be a JSON object, not "
                    f"{management_data!r} for domain concept {self.name}!",
                    location_id=self.location_id(DomainConceptDefinition.domain_concept_management, management_key),
                    part=PathPart.VALUE,
                )
        # missing checks:
        #  - "initialization" is a valid FunctionComposition expression (with variable context "instance")
        #    EXPRESSION CHECK
        #  - "consolidation" is a valid FunctionComposition expression (with variable context "instance")
        #    EXPRESSION CHECK
