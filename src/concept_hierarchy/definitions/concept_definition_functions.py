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

from typing import Callable

from concept_hierarchy.data.utils import UNINITIALIZED
from concept_hierarchy.definitions.concept_definition import ConceptDefinition
from concept_hierarchy.definitions.concept_definition_hidden_implementation import HiddenImplementationDefinition
from concept_hierarchy.definitions.definition import LocationOfCheckData, StopLocationOfCheck
from concept_hierarchy.definitions.utils import check_ch_name
from concept_hierarchy.errors import CHSemanticError, CHSyntaxError, LocationId, PathPart


class FunctionDefinition(HiddenImplementationDefinition):
    function_name: str = "Function"
    function_interface: str = "interface"
    function_procedure: str = "procedure"
    function_inversion: str = "inversion"
    function_variations: str = "variations"
    function_add_new_variables_in_existing_scope: str = "addNewVariablesInExistingScope"
    function_sub_scopes: str = "subScopes"
    function_result: str = "res"
    function_default_argument_values: str = "_defaultArgumentValues"
    implementation_related_keys: set[str] = HiddenImplementationDefinition.implementation_related_keys | {
        function_interface,
        function_procedure,
        function_inversion,
        function_variations,
        function_add_new_variables_in_existing_scope,
        function_sub_scopes,
    }
    evaluation_interface_keywords: set[str] = {function_result, function_default_argument_values}
    argument_reference_types: list[str] = ["NoRef", "Reference", "EmptyReference"]
    argument_modifier_types: list[str] = ["Get", "Modify", "GetModify"]
    default_function_instantiation_schema = {"type": "object", "maxProperties": 0, "additionalProperties": False}

    def __init__(self, name: str, definition_data: object, definition_location_id: LocationId):
        super().__init__(name, definition_data, definition_location_id)

        self.interface: dict = {}
        """The evaluation interface of the function. Data must be inherited!"""
        self.procedure: dict | None = None
        self.inversion: list[tuple[tuple[str, ...], dict[str, dict]]] | None = None
        """
        The first tuple entry (of each list element) is a tuple of template argument constraint formulae 
        (in the order of the template argument definition).
        The value is the procedure dictionary mapping argument names to a Function composition value
        """
        self.variations: dict[tuple[str, ...], dict] | None = None
        """
        The key is a tuple or argument names which, when they are variations of the current type,
         change this Function call to the Function call defined as value.
        The value is str | dict: either a template-instantiated Function name or a Function call
        """
        self.add_new_variables_in_existing_scope: dict[str, tuple[str, bool]] = {}
        """variable name -> (ValueDomain type, whether the new variable name is fixed or comes as an argument)"""
        self.sub_scopes: dict[str, dict[str, tuple[str, bool]]] = {}
        """ 
        argument name -> (new variable in scope of argument name -> (ValueDomain type, whether the var name is fixed)).
        Data must be inherited! 
        """

        self.all_evaluation_arguments: dict[str, str] = {}
        """Mapping from all available arguments (incl. the inherited ones) to the concept that defines them."""
        self.result_defined_in: str | None | object = UNINITIALIZED
        """Stores the concept that defines the result type or ``None`` if the function does not return anything."""

        self.all_sub_scope_data: dict[str, dict[str, tuple[str, bool]]] = {}
        """Contains all inherited data and the overwritten data from this concept."""

        self.evaluation_interface: tuple[str, ...] = ()
        self.evaluation_argument_types: dict[str, str] = {}
        self.evaluation_argument_reference_types: dict[str, str] = {}
        self.evaluation_argument_modifier_types: dict[str, str] = {}
        self.evaluation_argument_default_values: dict[str, object] = {}
        self.result_type: str | None = None
        self.result_reference_type: str | None = None
        self.result_modifier_type: str | None = None

    @classmethod
    def from_node(cls, concept_definition: ConceptDefinition):
        domain_concept = super().from_node(concept_definition)
        domain_concept.interface = {}
        domain_concept.procedure = None
        domain_concept.inversion = None
        domain_concept.variations = None
        domain_concept.add_new_variables_in_existing_scope = {}
        domain_concept.sub_scopes = {}

        domain_concept.all_evaluation_arguments = {}
        domain_concept.result_defined_in = UNINITIALIZED
        domain_concept.all_sub_scope_data = {}

        domain_concept.evaluation_interface = ()
        domain_concept.evaluation_argument_types = {}
        domain_concept.evaluation_argument_reference_types = {}
        domain_concept.evaluation_argument_modifier_types = {}
        domain_concept.evaluation_argument_default_values = {}
        domain_concept.result_type = None
        domain_concept.result_reference_type = None
        domain_concept.result_modifier_type = None

        return domain_concept

    def definition_type(self) -> str:
        return FunctionDefinition.function_name

    def location_of_impl(self, *keywords: str) -> LocationOfCheckData:
        check_res = super().location_of_impl(*keywords)
        # process top-level function data keywords: interface, procedure, inversion, variations, addVariables, subScopes
        self.check_location_id(
            check_res,
            FunctionDefinition.definition_location(self) + [check_res.first_remaining],
            location_check=self.data,
            previous_location=ConceptDefinition.concept_definition_data,
            allow_start_at_this_location=True,
        )
        # stop if found procedure; it has no more sub-data (just the expression?)
        if check_res.check_successful and check_res.last_consumed == FunctionDefinition.function_procedure:
            raise StopLocationOfCheck(check_res)

        # try to consume data inside "interface" or evaluation arguments directly
        if not check_res.check_successful or check_res.last_consumed == FunctionDefinition.function_interface:
            # use ``get``-method because it is possible that "interface" was not consumed
            interface_data = self.data.get(FunctionDefinition.function_interface, {})
            assert isinstance(interface_data, dict)
            self.check_location_id(
                check_res,
                FunctionDefinition.definition_location(self)
                + [FunctionDefinition.function_interface, check_res.first_remaining],
                location_check=interface_data,
                previous_location=FunctionDefinition.function_interface,
                allow_start_at_this_location=True,
            )
            if check_res.check_successful:
                if check_res.last_consumed == FunctionDefinition.function_default_argument_values:
                    # try to consume data inside "_defaultArgumentValues"
                    default_arguments_data = interface_data.get(check_res.last_consumed)
                    self.check_location_id(
                        check_res,
                        check_res.current_location_id + [check_res.first_remaining],
                        location_check=default_arguments_data,
                        previous_location=FunctionDefinition.function_default_argument_values,
                        allow_start_at_this_location=False,
                    )
            # either no top-level keyword was found or no interface data was matched => either way, stop checking
            raise StopLocationOfCheck(check_res)

        assert check_res.check_successful

        if check_res.last_consumed in [FunctionDefinition.function_variations, FunctionDefinition.function_inversion]:
            # Can't consume anything further for variations and inversions...
            raise StopLocationOfCheck(check_res)

        # try to consume data inside "subScopes"
        if check_res.last_consumed == FunctionDefinition.function_sub_scopes:
            sub_scopes_data = self.data[FunctionDefinition.function_sub_scopes]
            assert isinstance(sub_scopes_data, dict)
            self.check_location_id(
                check_res,
                check_res.current_location_id + [check_res.first_remaining],
                location_check=sub_scopes_data,
                previous_location=FunctionDefinition.function_sub_scopes,
                allow_start_at_this_location=False,
            )
            if not check_res.check_successful:
                raise StopLocationOfCheck(check_res)
            sub_scopes_argument_data = sub_scopes_data[check_res.last_consumed]
            assert isinstance(sub_scopes_argument_data, dict)
            self.check_location_id(
                check_res,
                check_res.current_location_id + [check_res.first_remaining],
                location_check=sub_scopes_argument_data,
                previous_location=check_res.last_consumed,
                allow_start_at_this_location=False,
            )
            raise StopLocationOfCheck(check_res)

        # try to consume "addNewVariablesInExistingScope" data
        if check_res.last_consumed == FunctionDefinition.function_add_new_variables_in_existing_scope:
            add_new_vars_in_scope = self.data[FunctionDefinition.function_add_new_variables_in_existing_scope]
            assert isinstance(add_new_vars_in_scope, dict)
            self.check_location_id(
                check_res,
                check_res.current_location_id + [check_res.first_remaining],
                location_check=add_new_vars_in_scope,
                previous_location=FunctionDefinition.function_add_new_variables_in_existing_scope,
                allow_start_at_this_location=False,
            )
            raise StopLocationOfCheck(check_res)

        return check_res

    def check_inversion_arguments(
        self, arg_inversion_mapping: dict[str, dict], location_id_functor: Callable[[str], LocationId]
    ):
        for arg_name, arg_inv_procedure in arg_inversion_mapping.items():
            # assertion, not check because this is a key of a JSON object
            assert isinstance(arg_name, str)
            if arg_name not in self.evaluation_argument_types:
                raise CHSemanticError(
                    f"Argument {arg_name} is not an argument of the {self.definition_type()}! "
                    f"Can not define an inversion for it!",
                    location_id_functor(arg_name),
                    part=PathPart.KEY,
                )
            if not isinstance(arg_inv_procedure, dict):
                raise CHSyntaxError(
                    f"{self.definition_type()} inversion procedure definitions must be JSON objects representing "
                    f"Function compositions!\n\tGot {arg_inv_procedure!r}",
                    location_id_functor(arg_name),
                    part=PathPart.VALUE,
                )

    def check_new_var_dict_def(
        self, new_var_dict_def, for_arg_name: str | None, location_id_functor: Callable[[str], LocationId]
    ):
        for new_var_name, new_var_def in new_var_dict_def.items():
            # assertion, not check
            assert isinstance(new_var_name, str)
            if not isinstance(new_var_def, (str, list)) or (
                isinstance(new_var_def, list) and not (1 <= len(new_var_def) <= 2)
            ):
                raise CHSyntaxError(
                    f"The definition of the addition of a new variable must be a JSON string defining the type of the "
                    f"variable or a 2-element JSON array containing on the first position the type of the new variable "
                    f"and on the optional second position, a boolean value!\nWhen false, the name of the newly added "
                    f"variable will be the name of the JSON object's key.\nWhen true, the JSON object key must be an "
                    f"evaluation argument of the {self.definition_type()} (of String type), whose value at "
                    f"evaluation-time determines the name of the new variable.\n\tGot {new_var_def!r}",
                    location_id_functor(new_var_name),
                    part=PathPart.VALUE,
                )
            if isinstance(new_var_def, str):
                new_var_dict_def[new_var_name] = (new_var_def, False)
            else:
                if not isinstance(new_var_def[0], str) or (
                    len(new_var_def) > 1 and not isinstance(new_var_def[1], bool)
                ):
                    raise CHSyntaxError(
                        f"The definition of the addition of a new variable must be a JSON string defining the type of "
                        f"the variable or a 2-element JSON array containing on the first position the type of the new "
                        f"variable and on the optional second position, a boolean value!\nWhen false, the name of the "
                        f"newly added variable will be the name of the JSON object's key.\nWhen true, the JSON object "
                        f"key must be an evaluation argument of the {self.definition_type()} (of String type), whose "
                        f"value at evaluation-time determines the name of the new variable.\n\tGot {new_var_def!r}",
                        location_id_functor(new_var_name),
                        part=PathPart.VALUE,
                    )
                if len(new_var_def) > 1:
                    if new_var_def[1] is True:
                        if (
                            new_var_name not in self.evaluation_argument_types
                            or self.evaluation_argument_types[new_var_name] != "String"
                        ):
                            raise CHSemanticError(
                                f"When using the [|NewVariableType|, true] new variable definition syntax, the variable"
                                f" name must be an evaluation argument of the {self.definition_type()}, that has String"
                                f" type!\n\t{new_var_name} is not a String argument of {self.name}!",
                                location_id_functor(new_var_name),
                                part=PathPart.KEY,
                            )
                        elif for_arg_name is not None and for_arg_name == new_var_name:
                            raise CHSemanticError(
                                f"Can't define a new variable, whose name depends on the runtime (call-time) value of "
                                f"an argument as a sub-scope variable available in the scope of that same argument."
                                f"\n\tFound at {for_arg_name}",
                                location_id_functor(new_var_name),
                                part=PathPart.KEY,
                            )
                    new_var_dict_def[new_var_name] = (new_var_def[0], new_var_def[1])
                else:
                    new_var_dict_def[new_var_name] = (new_var_def[0], False)

    def process_type_reference_and_modifier_of_argument(self, arg_name, arg_type_def) -> tuple[str, str, str]:
        if not isinstance(arg_type_def, (str, list)) or (
            isinstance(arg_type_def, list)
            and (not (1 <= len(arg_type_def) <= 3) or any(not isinstance(x, str) for x in arg_type_def))
        ):
            if "default" in arg_name and isinstance(arg_type_def, dict):
                raise CHSyntaxError(
                    f"It seems you were trying to define default argument values for the {self.definition_type()} "
                    f"{self.name!r}. {arg_type_def!r} does not look like a {self.definition_type()} argument definition"
                    f'.\nUse the "{FunctionDefinition.function_default_argument_values}" keyword for that!',
                    location_id=self.location_id(FunctionDefinition.function_interface, arg_name),
                )
            raise CHSyntaxError(
                f"{self.definition_type()} argument type definitions must be either a JSON string "
                f"(defining the argument's type) or an array of at least 1 and at most 3 string items "
                f"(defining the argument's type, its modifier type, and its reference type)!\n\t"
                f"Got {arg_type_def!r}",
                location_id=self.location_id(FunctionDefinition.function_interface, arg_name),
                part=PathPart.VALUE,
            )
        elif isinstance(arg_type_def, str):
            arg_type = arg_type_def
            arg_ref_type = self.argument_reference_types[0]
            arg_mod_type = self.argument_modifier_types[0]
        else:
            arg_type = arg_type_def[0]
            if len(arg_type_def) > 1:
                arg_ref_type = arg_type_def[1]
                if len(arg_type_def) > 2:
                    arg_mod_type = arg_type_def[2]
                else:
                    arg_mod_type = self.argument_modifier_types[0]
            else:
                arg_ref_type = self.argument_reference_types[0]
                arg_mod_type = self.argument_modifier_types[0]
        if arg_ref_type not in self.argument_reference_types:
            raise CHSyntaxError(
                f"{self.definition_type()} argument reference definitions must be "
                f"{self.argument_reference_types!r}, not {arg_ref_type}",
                location_id=self.location_id(FunctionDefinition.function_interface, arg_name),
                part=PathPart.VALUE,
            )
        if arg_mod_type not in self.argument_modifier_types:
            raise CHSyntaxError(
                f"{self.definition_type()} argument modifier definitions must be "
                f"{self.argument_modifier_types!r}, not {arg_mod_type}",
                location_id=self.location_id(FunctionDefinition.function_interface, arg_name),
                part=PathPart.VALUE,
            )
        return arg_type, arg_ref_type, arg_mod_type

    def concept_data_check(self):
        super().concept_data_check()

        if len(self.parents) != 1:
            raise CHSemanticError(
                f"{self.definition_type()}s can have at most one parent concept! {self.name!r} has {self.parents}",
                location_id=ConceptDefinition.definition_location(self) + [ConceptDefinition.concept_direct_parents],
            )

        # check "interface"; abstract Functions may not have an interface
        if FunctionDefinition.function_interface not in self.data and not self.abstract:
            raise CHSyntaxError(
                f"A {self.definition_type()} must define its evaluation interface as a JSON object in the "
                f'"{FunctionDefinition.function_interface}" key of its definition data. {self.definition_type()} '
                f"{self.name} does not!",
                location_id=self.location_id(),
                part=PathPart.VALUE,
            )
        self.interface = self.data.get(FunctionDefinition.function_interface, {})
        if not isinstance(self.interface, dict):
            raise CHSyntaxError(
                f"The evaluation interface of a {self.definition_type()} must be defined as a JSON object, not "
                f"{self.interface!r}!",
                location_id=self.location_id(FunctionDefinition.function_interface),
                part=PathPart.VALUE,
            )
        else:
            interface_extra_data = {k: v for k, v in self.interface.items() if k in self.evaluation_interface_keywords}
            for arg_name, arg_type_def in self.interface.items():
                if arg_name in interface_extra_data:
                    continue
                # assertion, not check because this is a key of a JSON object
                assert isinstance(arg_name, str)
                if not check_ch_name(arg_name, must_start_lowercase=True):
                    raise CHSyntaxError(
                        f"{self.definition_type()} argument name {arg_name!r} must be a lowercase-starting string! "
                        f"Got {arg_name}!",
                        location_id=self.location_id(FunctionDefinition.function_interface, arg_name),
                    )
                type_def_res = self.process_type_reference_and_modifier_of_argument(arg_name, arg_type_def)
                self.evaluation_argument_types[arg_name] = type_def_res[0]
                self.evaluation_argument_reference_types[arg_name] = type_def_res[1]
                self.evaluation_argument_modifier_types[arg_name] = type_def_res[2]
            if FunctionDefinition.function_result in self.interface:
                type_def_res = self.process_type_reference_and_modifier_of_argument(
                    FunctionDefinition.function_result, self.interface[FunctionDefinition.function_result]
                )
                self.result_type = type_def_res[0]
                self.result_reference_type = type_def_res[1]
                self.result_modifier_type = type_def_res[2]
            if FunctionDefinition.function_default_argument_values in self.interface:
                self.evaluation_argument_default_values = self.interface[
                    FunctionDefinition.function_default_argument_values
                ]
                if not isinstance(self.evaluation_argument_default_values, dict):
                    raise CHSyntaxError(
                        f"The default argument values of a {self.definition_type()} must be a JSON object, not "
                        f"{self.evaluation_argument_default_values!r}",
                        location_id=self.location_id(
                            FunctionDefinition.function_interface, FunctionDefinition.function_default_argument_values
                        ),
                        part=PathPart.VALUE,
                    )
                # check whether the default argument is an argument later, after all concept data has been initialized
                #  because there can be default arguments on parent Function arguments, which are not defined here
        # missing checks:
        # - function evaluation argument types
        #   TYPE CHECK
        #       - done in concept_hierarchy_type_checks.py
        # - function evaluation result types
        #   TYPE CHECK
        #       - done in concept_hierarchy_type_checks.py
        # - function evaluation default arguments
        #   EXPRESSION CHECK
        # - function arguments not doubly-defined
        #   REQUIRES: all concepts initialized (arguments can be inherited from parent functions)
        #   STRUCTURE CHECK
        #       - done in checker.py - check_after_parsing_concepts
        # - function result type not doubly-defined
        #   REQUIRES: all concepts initialized (arguments can be inherited from parent functions)
        #   STRUCTURE CHECK
        #       - done in checker.py - check_after_parsing_concepts
        # - whether default argument values are truly defined arguments
        #   (can't check here because this Function can define default values for arguments of the parent Function)
        #   REQUIRES: all concepts initialized (arguments can be inherited from parent functions)
        #   STRUCTURE CHECK
        #       - done in checker.py - check_after_parsing_concepts

        # check "procedure"
        self.procedure = self.data.get(FunctionDefinition.function_procedure, None)
        if self.procedure is not None:
            if not isinstance(self.procedure, dict):
                raise CHSyntaxError(
                    f"The procedure of a {self.definition_type()} must be a JSON object, not {self.procedure!r}",
                    location_id=self.location_id(FunctionDefinition.function_procedure),
                    part=PathPart.VALUE,
                )
        # missing checks:
        # - procedure is valid FunctionComposition expression
        #   EXPRESSION CHECK

        # check "inversion"
        inversion_data = self.data.get(FunctionDefinition.function_inversion, None)
        if inversion_data is not None:
            if not isinstance(inversion_data, (dict, list)):
                raise CHSyntaxError(
                    f"The inversion of a {self.definition_type()} must be a JSON object, not {inversion_data!r}",
                    location_id=self.location_id(FunctionDefinition.function_inversion),
                    part=PathPart.VALUE,
                )
            if isinstance(inversion_data, dict):
                self.check_inversion_arguments(
                    inversion_data, lambda x: self.location_id(FunctionDefinition.function_inversion, x)
                )
                self.inversion = [(tuple("" for _ in self.template_argument_order), inversion_data)]
            elif isinstance(inversion_data, list) and len(inversion_data) == 0:
                raise CHSyntaxError(
                    f"Can not specify an empty list of {self.definition_type()} inversions for {self.name!r}!",
                    location_id=self.location_id(FunctionDefinition.function_inversion),
                    part=PathPart.VALUE,
                )
            else:
                already_defined_specializations: set[tuple[str, ...]] = set()
                self.inversion = []
                for inversion_index, inversion_def in enumerate(inversion_data):
                    if not isinstance(inversion_def, list) or not len(inversion_def) == 2:
                        raise CHSyntaxError(
                            f"The template specialization syntax for {self.definition_type()} inversion definition "
                            f"should be a JSON array of 2-element JSON arrays containing:"
                            f"\n\tat the first element, a constraint formula specification for each "
                            f"{self.definition_type()} template argument, and\n\tat the second element, the inversion "
                            f"Function composition procedure for that template constraint specialization!\n"
                            f"Got {inversion_def!r}",
                            location_id=self.location_id(FunctionDefinition.function_inversion, inversion_index),
                        )
                    if not isinstance(inversion_def[0], list):
                        raise CHSyntaxError(
                            f"Invalid entry in template-specific inversion definition:\n\tthe first entry of "
                            f"the 2-element array must be a JSON array of string (template argument constraint "
                            f"formulae) values for each template argument.\n\t"
                            f"Got {inversion_def[0]!r}",
                            self.location_id(FunctionDefinition.function_inversion, inversion_index, 0),
                        )
                    if len(inversion_def[0]) != len(self.template_argument_order):
                        raise CHSemanticError(
                            f"Invalid entry in template-specific inversion definition:\n\tthe first entry of "
                            f"the 2-element array must be a JSON array of string (template argument constraint "
                            f"formulae) values of length {len(self.template_argument_order)}.\n\t\tAn entry for "
                            f"each template argument!\n\tGot {inversion_def[0]!r} of length {len(inversion_def[0])}",
                            self.location_id(FunctionDefinition.function_inversion, inversion_index, 0),
                        )
                    else:
                        for constraint_index, template_arg_constraint in enumerate(inversion_def[0]):
                            if not isinstance(template_arg_constraint, str):
                                raise CHSyntaxError(
                                    f"Invalid entry in template-specific inversion definition:\n\tthe first entry of "
                                    f"the 2-element array must be a JSON array of string (template argument constraint "
                                    f"formulae) values for each template argument.\n\tGot {template_arg_constraint!r}",
                                    self.location_id(
                                        FunctionDefinition.function_inversion, inversion_index, 0, constraint_index
                                    ),
                                )
                    inversion_specialization_key = tuple(inversion_def[0])
                    if inversion_specialization_key in already_defined_specializations:
                        raise CHSemanticError(
                            f"Doubly-defined template-specific {self.definition_type()} inversion key "
                            f"{inversion_specialization_key!r}",
                            location_id=self.location_id(FunctionDefinition.function_inversion, inversion_index),
                        )
                    procedure_def = inversion_def[1]
                    if not isinstance(procedure_def, dict):
                        raise CHSyntaxError(
                            f"The {self.definition_type()} inversion procedure definition in the template-"
                            f"specialization syntax of a {self.definition_type()} must be a JSON object, not "
                            f"{procedure_def!r}",
                            location_id=self.location_id(
                                FunctionDefinition.function_inversion,
                                inversion_index,
                                FunctionDefinition.function_procedure,
                            ),
                            part=PathPart.VALUE,
                        )
                    self.check_inversion_arguments(
                        procedure_def,
                        lambda x: self.location_id(
                            FunctionDefinition.function_inversion,
                            inversion_index,
                            FunctionDefinition.function_procedure,
                            x,
                        ),
                    )
                    # add data to ordered inversion list
                    already_defined_specializations.add(inversion_specialization_key)
                    self.inversion.append((inversion_specialization_key, procedure_def))
        # missing checks:
        #  - inversions are valid FunctionComposition expressions
        #    EXPRESSION CHECK
        #  - template constraints of inversions
        #    TYPE CHECK
        #       (valid template constraint def. + warn if definition intersected with t-arg-constraint is empty)

        # check "variations"
        self.variations = self.data.get(FunctionDefinition.function_variations, None)
        if self.variations is not None:
            if not isinstance(self.variations, (dict, list)):
                raise CHSyntaxError(
                    f"The definition of {self.definition_type()} variation relations must be a JSON array or object, "
                    f"not {self.variations!r}",
                    location_id=self.location_id(FunctionDefinition.function_variations),
                    part=PathPart.VALUE,
                )
            if isinstance(self.variations, dict):
                for arg_name, var_def in self.variations.items():
                    # assertion, not check
                    assert isinstance(arg_name, str)
                    if arg_name not in self.evaluation_argument_types:
                        raise CHSemanticError(
                            f"{arg_name} is not a {self.definition_type()} argument of {self.name}! "
                            f"Can't define a {self.definition_type()} variation relation for it",
                            location_id=self.location_id(FunctionDefinition.function_variations, arg_name),
                            part=PathPart.KEY,
                        )
                    if not isinstance(var_def, (dict, str)):
                        raise CHSyntaxError(
                            f"The definition of a {self.definition_type()} variation relation must be a JSON string "
                            f"(template-instantiated Function name) or object (Function evaluation mapping in which its"
                            f" arguments are set to this {self.definition_type()}'s arguments!).\n\tGot {var_def!r}",
                            location_id=self.location_id(FunctionDefinition.function_variations, arg_name),
                            part=PathPart.VALUE,
                        )

                    if isinstance(var_def, str):
                        #  update with explicit function call
                        self.variations[(arg_name,)] = {var_def: {x: x for x in self.evaluation_argument_types}}
                    else:
                        self.variations[(arg_name,)] = var_def
            else:
                curated_variations_definition = {}
                for var_index, var_entry in enumerate(self.variations):
                    if not isinstance(var_entry, list) or len(var_entry) != 2:
                        raise CHSyntaxError(
                            f"The array syntax of {self.definition_type()} variation relation definition must be a "
                            f"2-element JSON array.\n\tOn the first position, the JSON string argument name or a JSON "
                            f"array of string argument names must be given.\n\tOn the second position, the variation "
                            f"relation definition (as a JSON string or object value) must be specified.\n"
                            f"Got {var_entry!r}",
                            location_id=self.location_id(FunctionDefinition.function_variations, var_index),
                        )
                    if not isinstance(var_entry[0], (str, list)):
                        raise CHSyntaxError(
                            f"The array syntax of {self.definition_type()} variation relation definition must be a "
                            f"2-element JSON array.\n\tOn the first position, the JSON string argument name or a JSON "
                            f"array of string argument names must be given.\n\tOn the second position, the variation "
                            f"relation definition (as a JSON string or object value) must be specified.\n"
                            f"Got {var_entry!r}",
                            location_id=self.location_id(FunctionDefinition.function_variations, var_index),
                        )
                    elif isinstance(var_entry[0], str):
                        arg_name = var_entry[0]
                        if arg_name not in self.evaluation_argument_types:
                            raise CHSemanticError(
                                f"{arg_name} is not a {self.definition_type()} argument of {self.name}! "
                                f"Can't define a {self.definition_type()} variation relation for it",
                                location_id=self.location_id(FunctionDefinition.function_variations, var_index, 0),
                            )
                        arg_tuple_id = (arg_name,)
                    else:
                        arg_tuple_id = tuple(var_entry[0])
                        for arg_index, arg_name in enumerate(arg_tuple_id):
                            if arg_name not in self.evaluation_argument_types:
                                raise CHSemanticError(
                                    f"{arg_name} is not a {self.definition_type()} argument of {self.name}! "
                                    f"Can't define a {self.definition_type()} variation relation for it",
                                    location_id=self.location_id(
                                        FunctionDefinition.function_variations, var_index, 0, arg_index
                                    ),
                                )
                    if not isinstance(var_entry[1], (str, dict)):
                        raise CHSyntaxError(
                            f"The definition of a {self.definition_type()} variation relation must be a JSON string "
                            f"(template-instantiated Function name) or object (Function evaluation mapping in which its"
                            f" arguments are set to this {self.definition_type()}'s arguments!)."
                            f"\n\tGot {var_entry[1]!r}",
                            location_id=self.location_id(FunctionDefinition.function_variations, var_index, 1),
                        )
                    elif isinstance(var_entry[1], str):
                        var_def = {var_entry[1]: {x: x for x in self.evaluation_argument_types}}
                    else:
                        var_def = var_entry[1]
                    curated_variations_definition[arg_tuple_id] = var_def
                self.variations = curated_variations_definition
        # missing checks:
        #  - variation data is a valid Function! instantiation
        #    EXPRESSION CHECK

        # check "addNewVariablesInExistingScope"
        self.add_new_variables_in_existing_scope = self.data.get(
            FunctionDefinition.function_add_new_variables_in_existing_scope, {}
        )
        if not isinstance(self.add_new_variables_in_existing_scope, dict):
            raise CHSyntaxError(
                f"The definition of new variables to be added to the scope in which the {self.definition_type()} was "
                f"called must be a JSON object, not {self.add_new_variables_in_existing_scope!r}.",
                location_id=self.location_id(FunctionDefinition.function_add_new_variables_in_existing_scope),
                part=PathPart.VALUE,
            )
        self.check_new_var_dict_def(
            self.add_new_variables_in_existing_scope,
            None,
            lambda x: self.location_id(FunctionDefinition.function_add_new_variables_in_existing_scope),
        )
        # missing checks:
        #  - types of new variables are valid
        #    TYPE CHECK
        #       - done in concept_hierarchy_type_checks.py

        # check "subScopes"
        self.sub_scopes = self.data.get(FunctionDefinition.function_sub_scopes, {})
        if not isinstance(self.sub_scopes, dict):
            raise CHSyntaxError(
                f"The definition of new variables to be added in the scope of {self.definition_type()} arguments must "
                f"be a JSON object, not {self.sub_scopes!r}",
                location_id=self.location_id(FunctionDefinition.function_sub_scopes),
                part=PathPart.VALUE,
            )
        for arg_name, new_var_def_data in self.sub_scopes.items():
            # assertion, not check
            assert isinstance(arg_name, str)
            if arg_name not in self.evaluation_argument_types:
                raise CHSemanticError(
                    f"{arg_name} is not an evaluation argument of {self.name}! Can't define new variables in the "
                    f"argument's subscope!",
                    location_id=self.location_id(FunctionDefinition.function_sub_scopes, arg_name),
                    part=PathPart.KEY,
                )
            if not isinstance(new_var_def_data, dict):
                raise CHSyntaxError(
                    f"The definition of new variables to be added to the scope of a {self.definition_type()} argument "
                    f"must be a JSON object.\n\tGot {new_var_def_data!r}",
                    location_id=self.location_id(FunctionDefinition.function_sub_scopes, arg_name),
                    part=PathPart.VALUE,
                )
            self.check_new_var_dict_def(
                new_var_def_data, arg_name, lambda x: self.location_id(FunctionDefinition.function_sub_scopes, arg_name)
            )
        # missing checks:
        #  - types of new variables are valid
        #    TYPE CHECK
        #        - done in concept_hierarchy_type_checks.py

    @property
    def returns_something(self):
        return self.result_type is not None

    @property
    def has_interface_defined(self):
        """Stores whether the concept definition has the ``"interface"`` keyword."""
        return FunctionDefinition.function_interface in self.data
