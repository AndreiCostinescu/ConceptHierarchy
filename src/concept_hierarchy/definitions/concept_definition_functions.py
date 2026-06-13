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

from types import NoneType
from typing import Callable

from concept_hierarchy.definitions.concept_definition import ConceptDefinition
from concept_hierarchy.definitions.concept_definition_hidden_implementation import HiddenImplementationDefinition
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

    def __init__(self, name: str, definition_data: object, definition_location_str: str):
        super().__init__(name, definition_data, definition_location_str)

        self.interface: dict = {}
        self.procedure: dict | None = None
        # The key is a tuple of template argument constraint formulae (in the order of the template argument definition)
        # The value is the procedure dictionary mapping argument names to a Function composition value
        self.inversion: dict[tuple[str | None, ...], dict[str, dict]] | None = None
        # The key is a tuple or argument names which, when they are variations of the current type,
        #  change this Function call to the Function call defined as value.
        # The value is str | dict: either a template-instantiated Function name or a Function call
        self.variations: dict[tuple[str, ...], dict] | None = None
        # variable name -> (ValueDomain type, whether the new variable name is fixed or comes as an argument)
        self.add_new_variables_in_existing_scope: dict[str, tuple[str, bool]] = {}
        # argument name -> (new variable in scope of argument name -> (ValueDomain type, whether the var name is fixed))
        self.sub_scopes: dict[str, dict[str, tuple[str, bool]]] = {}

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
                    if (
                        new_var_name not in self.evaluation_argument_types
                        or self.evaluation_argument_types[new_var_name] != "String"
                    ):
                        raise CHSemanticError(
                            f"When using the [|NewVariableType|, true] new variable definition syntax, the variable "
                            f"name must be an evaluation argument of the {self.definition_type()}, that has String "
                            f"type!\n\t{new_var_name} is not a String argument of {self.name}!",
                            location_id_functor(new_var_name),
                            part=PathPart.KEY,
                        )
                    elif new_var_def[1] and for_arg_name is not None and for_arg_name == new_var_name:
                        raise CHSemanticError(
                            f"Can't define a new variable, whose name depends on the runtime (call-time) value of an "
                            f"argument as a sub-scope variable available in the scope of that same argument."
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
            raise CHSyntaxError(
                f"{self.definition_type()} argument type definitions must be either a JSON string "
                f"(defining the argument's type) or an array of at least 1 and at most 3 string items "
                f"(defining the argument's type, its modifier type, and its reference type)!\n\t"
                f"Got {arg_type_def!r}",
                self.location_id(FunctionDefinition.function_interface, arg_name),
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
                self.location_id(FunctionDefinition.function_interface, arg_name),
                part=PathPart.VALUE,
            )
        if arg_mod_type not in self.argument_modifier_types:
            raise CHSyntaxError(
                f"{self.definition_type()} argument modifier definitions must be "
                f"{self.argument_modifier_types!r}, not {arg_mod_type}",
                self.location_id(FunctionDefinition.function_interface, arg_name),
                part=PathPart.VALUE,
            )
        return arg_type, arg_ref_type, arg_mod_type

    def concept_data_check(self):
        super().concept_data_check()

        # check "interface"
        if FunctionDefinition.function_interface not in self.data:
            raise CHSyntaxError(
                f"A {self.definition_type()} must define its evaluation interface as a JSON object in the "
                f'"{FunctionDefinition.function_interface}" key of its definition data. {self.definition_type()} '
                f"{self.name} does not!",
                self.location_id(),
                part=PathPart.VALUE,
            )
        self.interface = self.data[FunctionDefinition.function_interface]
        if not isinstance(self.interface, dict):
            raise CHSyntaxError(
                f"The evaluation interface of a {self.definition_type()} must be defined as a JSON object, not "
                f"{self.interface!r}!",
                self.location_id(FunctionDefinition.function_interface),
                part=PathPart.VALUE,
            )
        else:
            interface_extra_data = {k: v for k, v in self.interface.items() if k in self.evaluation_interface_keywords}
            for arg_name, arg_type_def in self.interface.items():
                if arg_name in interface_extra_data:
                    continue
                # assertion, not check because this is a key of a JSON object
                assert isinstance(arg_name, str)
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
                        self.location_id(
                            FunctionDefinition.function_interface, FunctionDefinition.function_default_argument_values
                        ),
                        part=PathPart.VALUE,
                    )
                # check whether the default argument is an argument later, after all concept data has been initialized
                #  because there can be default arguments on parent Function arguments, which are not defined here
        # missing checks:
        # - function evaluation argument types
        #   TYPE CHECK
        # - function evaluation default arguments
        #   EXPRESSION CHECK
        # - whether default argument values are truly defined arguments
        #   (can't check here because this Function can define default values for arguments of the parent Function)
        #   REQUIRES: all concepts initialized (arguments can be inherited from parent functions)
        #   STRUCTURE CHECK

        # check "procedure"
        self.procedure = self.data.get(FunctionDefinition.function_procedure, None)
        if self.procedure is not None:
            if not isinstance(self.procedure, dict):
                raise CHSyntaxError(
                    f"The procedure of a {self.definition_type()} must be a JSON object, not {self.procedure!r}",
                    self.location_id(FunctionDefinition.function_procedure),
                    part=PathPart.VALUE,
                )
        # missing checks:
        # - procedure is valid FunctionComposition expression
        #   EXPRESSION CHECK

        # check "inversion"
        self.inversion = self.data.get(FunctionDefinition.function_inversion, None)
        if self.inversion is not None:
            if not isinstance(self.inversion, (dict, list)):
                raise CHSyntaxError(
                    f"The inversion of a {self.definition_type()} must be a JSON object, not {self.inversion!r}",
                    self.location_id(FunctionDefinition.function_inversion),
                    part=PathPart.VALUE,
                )
            if isinstance(self.inversion, dict):
                self.check_inversion_arguments(
                    self.inversion, lambda x: self.location_id(FunctionDefinition.function_inversion, x)
                )
                self.inversion = {tuple(None for _ in self.template_argument_order): self.inversion}
            else:
                curated_inversion_definition = {}
                for inversion_index, inversion_def in enumerate(self.inversion):
                    if (
                        not isinstance(inversion_def, dict)
                        or FunctionDefinition.function_procedure not in inversion_def
                    ):
                        raise CHSyntaxError(
                            f"The template specialization syntax for {self.definition_type()} inversion definition "
                            f"should be a JSON array of JSON objects containing:"
                            f"\n\tfor each {self.definition_type()} template argument a template constraint formula, "
                            f"and\n\tthe inversion Function composition procedure (at the "
                            f'"{FunctionDefinition.function_procedure}" key of the object) for that template '
                            f"specialization!\nGot {inversion_def!r}",
                            self.location_id(FunctionDefinition.function_inversion, inversion_index),
                        )
                    for t_arg, t_arg_constraint in inversion_def.items():
                        # assertion, not check because this is a key of a JSON object
                        assert isinstance(t_arg, str)
                        if t_arg not in self.template_arguments:
                            raise CHSemanticError(
                                f"{t_arg} is not a template argument of the {self.definition_type()} {self.name}! "
                                f"Can't define an inversion for this specialization!",
                                self.location_id(FunctionDefinition.function_inversion, inversion_index, t_arg),
                                part=PathPart.KEY,
                            )
                        if not isinstance(t_arg_constraint, (str, NoneType)):
                            raise CHSyntaxError(
                                f"Specialization of template constraint formulae for {self.definition_type()} inversion"
                                f" procedure must be a JSON string or null (if that template argument is not to be "
                                f"specialized).\n\tGot {t_arg_constraint!r}",
                                self.location_id(FunctionDefinition.function_inversion, inversion_index, t_arg),
                                part=PathPart.VALUE,
                            )
                    procedure_def = inversion_def[FunctionDefinition.function_procedure]
                    if not isinstance(procedure_def, dict):
                        raise CHSyntaxError(
                            f"The {self.definition_type()} inversion procedure definition in the template-"
                            f"specialization syntax of a {self.definition_type()} must be a JSON object, not "
                            f"{procedure_def!r}",
                            self.location_id(
                                FunctionDefinition.function_inversion,
                                inversion_index,
                                FunctionDefinition.function_procedure,
                            ),
                            part=PathPart.VALUE,
                        )
                    self.check_inversion_arguments(
                        inversion_def[FunctionDefinition.function_procedure],
                        lambda x: self.location_id(
                            FunctionDefinition.function_inversion,
                            inversion_index,
                            FunctionDefinition.function_procedure,
                            x,
                        ),
                    )
                    specialization_id = tuple(inversion_def.get(x, None) for x in self.template_argument_order)
                    curated_inversion_definition[specialization_id] = inversion_def[
                        FunctionDefinition.function_procedure
                    ]
                self.inversion = curated_inversion_definition
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
                    self.location_id(FunctionDefinition.function_variations),
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
                            self.location_id(FunctionDefinition.function_variations, arg_name),
                            part=PathPart.KEY,
                        )
                    if not isinstance(var_def, (dict, str)):
                        raise CHSyntaxError(
                            f"The definition of a {self.definition_type()} variation relation must be a JSON string "
                            f"(template-instantiated Function name) or object (Function evaluation mapping in which its"
                            f" arguments are set to this {self.definition_type()}'s arguments!).\n\tGot {var_def!r}",
                            self.location_id(FunctionDefinition.function_variations, arg_name),
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
                            self.location_id(FunctionDefinition.function_variations, var_index),
                        )
                    if not isinstance(var_entry[0], (str, list)):
                        raise CHSyntaxError(
                            f"The array syntax of {self.definition_type()} variation relation definition must be a "
                            f"2-element JSON array.\n\tOn the first position, the JSON string argument name or a JSON "
                            f"array of string argument names must be given.\n\tOn the second position, the variation "
                            f"relation definition (as a JSON string or object value) must be specified.\n"
                            f"Got {var_entry!r}",
                            self.location_id(FunctionDefinition.function_variations, var_index),
                        )
                    elif isinstance(var_entry[0], str):
                        arg_name = var_entry[0]
                        if arg_name not in self.evaluation_argument_types:
                            raise CHSemanticError(
                                f"{arg_name} is not a {self.definition_type()} argument of {self.name}! "
                                f"Can't define a {self.definition_type()} variation relation for it",
                                self.location_id(FunctionDefinition.function_variations, var_index, 0),
                            )
                        arg_tuple_id = (arg_name,)
                    else:
                        arg_tuple_id = tuple(var_entry[0])
                        for arg_index, arg_name in enumerate(arg_tuple_id):
                            if arg_name not in self.evaluation_argument_types:
                                raise CHSemanticError(
                                    f"{arg_name} is not a {self.definition_type()} argument of {self.name}! "
                                    f"Can't define a {self.definition_type()} variation relation for it",
                                    self.location_id(FunctionDefinition.function_variations, var_index, 0, arg_index),
                                )
                    if not isinstance(var_entry[1], (str, dict)):
                        raise CHSyntaxError(
                            f"The definition of a {self.definition_type()} variation relation must be a JSON string "
                            f"(template-instantiated Function name) or object (Function evaluation mapping in which its"
                            f" arguments are set to this {self.definition_type()}'s arguments!)."
                            f"\n\tGot {var_entry[1]!r}",
                            self.location_id(FunctionDefinition.function_variations, var_index, 1),
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
                self.location_id(FunctionDefinition.function_add_new_variables_in_existing_scope),
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

        # check "subScopes"
        self.sub_scopes = self.data.get(FunctionDefinition.function_sub_scopes, {})
        if not isinstance(self.sub_scopes, dict):
            raise CHSyntaxError(
                f"The definition of new variables to be added in the scope of {self.definition_type()} arguments must "
                f"be a JSON object, not {self.sub_scopes!r}",
                self.location_id(FunctionDefinition.function_sub_scopes),
                part=PathPart.VALUE,
            )
        for arg_name, new_var_def_data in self.sub_scopes.items():
            # assertion, not check
            assert isinstance(arg_name, str)
            if arg_name not in self.evaluation_argument_types:
                raise CHSemanticError(
                    f"{arg_name} is not an evaluation argument of {self.name}! Can't define new variables in the "
                    f"argument's subscope!",
                    self.location_id(FunctionDefinition.function_sub_scopes, arg_name),
                    part=PathPart.KEY,
                )
            if not isinstance(new_var_def_data, dict):
                raise CHSyntaxError(
                    f"The definition of new variables to be added to the scope of a {self.definition_type()} argument "
                    f"must be a JSON object.\n\tGot {new_var_def_data!r}",
                    self.location_id(FunctionDefinition.function_sub_scopes, arg_name),
                    part=PathPart.VALUE,
                )
            self.check_new_var_dict_def(
                new_var_def_data, arg_name, lambda x: self.location_id(FunctionDefinition.function_sub_scopes, arg_name)
            )
        # missing checks:
        #  - types of new variables are valid
        #    TYPE CHECK
