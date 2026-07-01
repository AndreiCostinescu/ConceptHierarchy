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

"""
Unit tests for checking whether the validation of template-instantiated types functions correctly.
"""

from copy import deepcopy

import pytest

from concept_hierarchy.data.concept_hierarchy import ConceptHierarchy
from concept_hierarchy.errors import CHSemanticError
from concept_hierarchy.models import ConceptHierarchyModel
from concept_hierarchy.validator.checker import ConceptHierarchyChecker


class TestTemplateArgumentParsing:
    _model_data = {
        "Concept": {},
        "DomainConcept": {"directParents": ["Concept"], "data": {"properties": {"example": "NonTemplate1"}}},
        "ValueDomain": {"directParents": ["Concept"], "data": {"abstract": True}},
        "NonTemplateValueDomain": {"directParents": ["ValueDomain"], "data": {}},
        "TemplateValueDomain": {"directParents": ["ValueDomain"], "data": {}},
        "NonTemplate1": {"directParents": ["NonTemplateValueDomain"], "data": {}},
        "SubNonTemplate1": {"directParents": ["NonTemplate1"], "data": {}},
        "NonTemplate2": {"directParents": ["NonTemplateValueDomain"], "data": {}},
        "Template1_DefaultConstraints": {
            "directParents": ["TemplateValueDomain"],
            "data": {"templateContext": ["T"]},
        },
        "Template1": {
            "directParents": ["TemplateValueDomain"],
            "data": {
                "templateContext": {
                    "order": ["T"],
                    "T": "And(Concept, Not(ValueDomain))",
                }
            },
        },
        "Template2_DefaultConstraints": {
            "directParents": ["TemplateValueDomain"],
            "data": {"templateContext": ["T1", "T2"]},
        },
        "Template2": {
            "directParents": ["TemplateValueDomain"],
            "data": {
                "templateContext": {
                    "order": ["T1", "T2"],
                    "T1": "And(Concept, Not(ValueDomain))",
                    "T2": "NonTemplateValueDomain",
                }
            },
        },
        "Template3_DefaultConstraints": {
            "directParents": ["TemplateValueDomain"],
            "data": {"templateContext": {"order": ["T"]}},
        },
        "Template3_NonTemplate": {
            "directParents": ["TemplateValueDomain"],
            "data": {
                "templateContext": {
                    "order": ["T"],
                    "T": "NonTemplateValueDomain",
                }
            },
        },
        "Template3_Template": {
            "directParents": ["TemplateValueDomain"],
            "data": {
                "templateContext": {
                    "order": ["T"],
                    "T": "TemplateValueDomain",
                }
            },
        },
        "SubTemplate1_Templated": {
            "directParents": ["Template1"],
            "data": {
                "templateContext": {"order": ["T1", "T2"], "T1": "Concept", "substitution": {"Template1:T": "T1"}},
            },
        },
    }

    def clone_model(self):
        return deepcopy(self._model_data)

    def shallow_copy_for_adding_or_removing_concepts(self):
        return self._model_data.copy()

    def get_external_data(self, x, y):
        pass

    def get_model(self, model_data) -> ConceptHierarchy:
        model = ConceptHierarchyModel.create_from_data(model_data)
        checker = ConceptHierarchyChecker(model, self.get_external_data)
        res = checker.model
        checker.check()
        return res

    def test_model_succeeds(self):
        self.get_model(self._model_data)

    def test_model_fails_with_wrong_substitution_of(self):
        """
        SubTemplate1_template substitutes Template1:T, which must be a DomainConcept; but T2 is not a domain concept!
        """
        new_model = self.clone_model()
        new_model["SubTemplate1_Templated"]["data"]["templateContext"]["substitution"]["Template1:T"] = "T2"
        with pytest.raises(
            CHSemanticError,
            match="Merging template context with determined constraints during substitution-instantiation of Template1 "
            "lead to no possible template-instantiation of SubTemplate1_Templated",
        ):
            self.get_model(new_model)

    def test_sub_templated_concept_without_substitution_fails(self):
        new_model = self.shallow_copy_for_adding_or_removing_concepts()
        new_model["SubTemplatedConceptWithoutSubstitution"] = {"directParents": ["Template1"], "data": {}}
        with pytest.raises(
            CHSemanticError,
            match=r'Missing "templateContext" definition in SubTemplatedConceptWithoutSubstitution, because it '
            r'must define a substitution for "Template1:T"!',
        ):
            self.get_model(new_model)

    def test_sub_templated_template_concept_without_substitution_fails(self):
        new_model = self.shallow_copy_for_adding_or_removing_concepts()
        new_model["SubTemplatedTemplateConceptWithoutSubstitution"] = {
            "directParents": ["Template1"],
            "data": {
                "templateContext": ["T1"],
            },
        }
        with pytest.raises(
            CHSemanticError,
            match=r'Missing "substitution" definition in SubTemplatedTemplateConceptWithoutSubstitution, '
            r'because it must define a substitution for "Template1:T"!',
        ):
            self.get_model(new_model)

    def test_sub_templated_template_concept_with_ambiguous_shorthand_substitution_fails(self):
        new_model = self.shallow_copy_for_adding_or_removing_concepts()
        new_model["SubTemplatedTemplateConceptWithAmbiguousShorthandSubstitution"] = {
            "directParents": ["Template1", "Template3_Template"],
            "data": {
                "templateContext": {"order": ["T"], "substitution": {"T": "T"}},
            },
        }
        with pytest.raises(
            CHSemanticError,
            match=r'\["concepts": "SubTemplatedTemplateConceptWithAmbiguousShorthandSubstitution": "data": '
            r'"templateContext": "substitution" \(key\)\] \n    The substitution specification of template '
            r"argument T is ambiguous in SubTemplatedTemplateConceptWithAmbiguousShorthandSubstitution because "
            r"the parent concepts \[\'Template1\', \'Template3_Template\'\] define the template argument with the"
            r' same name. Use the "<ParentConceptName>:<ParentTemplateArgumentName>" syntax to define the '
            r"unambiguous substitution value for all parent template arguments",
        ):
            self.get_model(new_model)

    def test_sub_templated_concept_with_ambiguous_shorthand_substitution_fails(self):
        new_model = self.shallow_copy_for_adding_or_removing_concepts()
        new_model["SubTemplatedConceptWithAmbiguousShorthandSubstitution"] = {
            "directParents": ["Template1", "Template3_NonTemplate"],
            "data": {
                "templateContext": {"substitution": {"T": "NonTemplate1"}},
            },
        }
        with pytest.raises(
            CHSemanticError,
            match=r'\["concepts": "SubTemplatedConceptWithAmbiguousShorthandSubstitution": "data": "templateContext":'
            r' "substitution" \(key\)\] \n    The substitution specification of template argument T is ambiguous'
            r" in SubTemplatedConceptWithAmbiguousShorthandSubstitution because the parent concepts "
            r"\[\'Template1\', \'Template3_NonTemplate\'\] define the template argument with the same name. Use "
            r'the "<ParentConceptName>:<ParentTemplateArgumentName>" syntax to define the unambiguous '
            r"substitution value for all parent template arguments",
        ):
            self.get_model(new_model)
