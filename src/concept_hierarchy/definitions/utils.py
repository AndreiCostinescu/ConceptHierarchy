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

from concept_hierarchy.errors import CHSyntaxError, LocationId, PathPart
from concept_hierarchy.utils import capitalize


def check_ch_name(
    ch_name: str,
    name_type: str | None = None,
    *,
    must_start_uppercase: bool = False,
    must_start_lowercase: bool = False,
    allow_empty: bool = False,
    allow_starting_with_digit: bool = False,
    allow_starting_with_underscore: bool = False,
    allowed_special_characters: list[str] = None,
    location_id: LocationId | None = None,
):
    if not isinstance(ch_name, str):
        raise RuntimeError(f"Given argument ch_name must be a string, not {ch_name!r}")
    return_false_instead_of_error = name_type is None
    if allowed_special_characters is None:
        allowed_special_characters = []
    if not allow_empty and ch_name == "":
        if return_false_instead_of_error:
            return False
        raise CHSyntaxError(f"The name of a {name_type} cannot be empty!", location_id=location_id, part=PathPart.KEY)
    illegal_characters_found = []
    for char in ch_name:
        if not (char.isalnum() or char == "_" or char in allowed_special_characters):
            illegal_characters_found.append(char)
    if not allow_starting_with_digit and ch_name[0].isnumeric():
        if return_false_instead_of_error:
            return False
        raise CHSyntaxError(
            f"{capitalize(name_type)} names may not start with a digit!", location_id=location_id, part=PathPart.KEY
        )
    if not allow_starting_with_underscore and ch_name.startswith("_"):
        if return_false_instead_of_error:
            return False
        raise CHSyntaxError(
            f"{capitalize(name_type)} names may not start with an underscore '_' character!",
            location_id=location_id,
            part=PathPart.KEY,
        )
    if illegal_characters_found:
        if return_false_instead_of_error:
            return False
        raise CHSyntaxError(
            f"{capitalize(name_type)} names can not contain the character(s) "
            + ", ".join("'" + c + "'" for c in illegal_characters_found)
            + ". Delete or replace character by '_'!",
            location_id=location_id,
            part=PathPart.KEY,
        )
    if must_start_lowercase and not ch_name[0].islower():
        if return_false_instead_of_error:
            return False
        raise CHSyntaxError(
            f"{capitalize(name_type)} name {ch_name!r} must start with a lowercase letter!",
            location_id=location_id,
            part=PathPart.KEY,
        )
    if must_start_uppercase and not ch_name[0].isupper():
        if return_false_instead_of_error:
            return False
        raise CHSyntaxError(
            f"{capitalize(name_type)} name {ch_name!r} must start with an uppercase letter!",
            location_id=location_id,
            part=PathPart.KEY,
        )
    return True
