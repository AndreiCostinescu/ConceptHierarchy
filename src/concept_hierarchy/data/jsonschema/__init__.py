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

"""ch_schema -- shorthand-aware draft-07 JSON-Schema validation for ConceptHierarchy.

Typical usage::

    from ch_schema import parse_schema, validate_value, CHSchemaContext, CHValueContext

    node, errors = parse_schema(my_schema, my_schema_context)
    if not errors:
        value_errors = validate_value(my_value, node, my_value_context)

See :mod:`ch_schema.schema_validator` and :mod:`ch_schema.value_validator` for details,
and :mod:`ch_schema.context` for the two context protocols that need to be implemented by callers.
"""

from concept_hierarchy.data.jsonschema.ast_nodes import CHSchemaNode

__all__ = ["CHSchemaNode"]
