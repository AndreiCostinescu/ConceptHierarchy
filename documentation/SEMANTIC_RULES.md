# Concept Hierarchy — Semantic Rules

This document lists every semantic constraint enforced by the validator. 
Each rule corresponds to a `CHSemanticError` raise in the source code. 
<!-- Rules are grouped by the subsystem that enforces them.-->

---

## 1. Hierarchy Structure

### 1.1 No cycles in the concept graph
Concepts must form a directed acyclic graph (DAG). If the topological sort detects a cycle, validation fails.

- **Source:** `validator/checker.py` — `check_structure`
- **Trigger:** `topological_sort` raises `RuntimeError` with a "Non-hierarchy structure detected" prefix

### 1.2 Exactly one root
The concept graph must have exactly one root (a concept with no parents). If topological sort yields multiple roots *and* the expected root name is among them, validation fails. (If the expected root name is not among the determined roots, an empty concept is created (with the expected name) and the previous roots become its direct children.)

- **Source:** `validator/checker.py` — `check_structure`

### 1.3 Root must have the canonical name
The single root concept must be named with `ConceptHierarchyModel.root_concept_name`. Any other name is rejected.

- **Source:** `validator/checker.py` — `check_structure`

### 1.4 Every parent must be defined
Each concept's `directParents` list may only reference concepts that are themselves defined in the hierarchy.

- **Source:** `validator/checker.py` — `check_structure`

### 1.5 No cycles in concept/instance references
If any subset of concepts or instances form a reference cycle (e.g. A references B and B references A), validation fails. Detected by iterative resolution that stalls before completion.

- **Source:** `validator/checker.py` — `check_cycles_in_references_based_on_defined`

### 1.6 References are valid identifiers
The referenced value of a concept must be a concept appearing in the concept definitions. 
Similarly. for the reference value of an instance, which also must be a defined instance.

- **Source:** `validator/checker.py` — `check_cycles_in_references_based_on_defined`

---

## 2. Naming and Uniqueness

### 2.1 Global variable names must not clash with concept names
A global variable (instance) and a concept may not share a name, as this creates lookup ambiguity.

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["instances", <name>]`

### 2.2 Template argument names must not clash with concept names
A Function's or ValueDomain's template argument name may not coincide with any concept name in the hierarchy, since this creates ambiguity in constraint formulae, instantiations, and substitutions.

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "data", "templateArguments"]`

### 2.3 Function evaluation argument names must not clash with global variable names
A Function's evaluation interface argument name may not coincide with any global variable name. The conflict would cause ambiguity in FunctionComposition procedures, inversions, and variations.

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "data", "interface"]`

### 2.4 Domain concept property names must not clash with global variable names
A domain concept's property name may not coincide with any global variable name. The conflict would cause ambiguity in property hooks, computations, concept functions, and management functions.

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "data", "properties", <prop>]`

### 2.5 Domain concept function names must not clash with global variable names
Same constraint as 2.4, applied to domain concept functions.

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "data", "functions", <func>]`

### 2.6 Property names must be unique across the entire hierarchy
The same property name may not be defined in more than one domain concept. If it appears in two places, it must be moved to a shared ancestor or one occurrence must be renamed.

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "data", "properties", <prop>]`

### 2.7 Function names must be unique across the entire hierarchy
The same function name may not be defined in more than one domain concept.

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "data", "functions", <func>]`

### 2.8 Property names and function names must be disjoint
A name that is used as a property in one concept may not be used as a function in another concept (and vice versa).

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "data", "properties"|"functions", <name>]`

---

## 3. Concept Classification

### 3.1 Every non-root concept must be classifiable
After the hierarchy structure is validated, each concept is classified as one of: `Function`, `ValueDomain`, or `DomainConcept`. A concept that cannot be assigned to any category is rejected.

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "directParents"]`

### 3.2 Every domain concept must carry data
A domain concept that is not the root concept must define at least one of: `properties`, `functions`, or `management`.

- **Source:** `definitions/concept_definition_domain_concept.py` — `concept_data_check`
- **Location:** `["concepts", <concept>]` (the concept's own location)

---

## 4. External Data

### 4.1 External data files must be resolvable
When a concept definition references an external data file, that file must exist and be loadable. A `RuntimeError` from the loader (indicating the file could not be found) is re-raised as a semantic error.

- **Source:** `definitions/concept_definition.py` — `data` property
- **Location:** the concept's data location

---

## 5. Hidden Implementation (Functions and ValueDomains)

### 5.1 Implementation file path must not include a file extension
The `implementation` field specifies a file path without extension. Paths containing a `.` character are rejected.

- **Source:** `definitions/concept_definition_hidden_implementation.py` — `concept_data_check`
- **Location:** `["concepts", <concept>, "data", "implementation"]`

### 5.2 Abstract concepts must not define an instantiation structure
If a ValueDomain or Function is marked `abstract: true`, it may not simultaneously define an `instantiation` field.

- **Source:** `definitions/concept_definition_value_domain.py` — `check_instantiation`
- **Location:** `["concepts", <concept>, "data", "instantiation"]`

### 5.3 Template argument constraints must be defined on declared template arguments
A constraint entry in the `templateArguments` object must use a key that is already declared in the `order` list of the same definition.

- **Source:** `definitions/concept_definition_hidden_implementation.py` — `concept_data_check`
- **Location:** `["concepts", <concept>, "data", "templateArguments", <arg>]`

### 5.4 Variadic group identifiers may only be defined on declared variadic template arguments
Each key in `variadicGroupIdentifiers` must be an argument present in `template_argument_order` that has been defined with the `...` variadic marker. Using the `...` suffix on the key (the variadic marker) in this position is also disallowed.

- **Source:** `definitions/concept_definition_hidden_implementation.py` — `concept_data_check`
- **Location:** `["concepts", <concept>, "data", "templateArguments", "variadicGroupIdentifiers", <arg>]`

### 5.5 The empty variadic group identifier is only allowed for all-variadic-template concepts
An empty string `""` may be used as a variadic group identifier only when every template argument of the concept is variadic.

- **Source:** `definitions/concept_definition_hidden_implementation.py` — `concept_data_check`
- **Location:** `["concepts", <concept>, "data", "templateArguments", "variadicGroupIdentifiers", <arg>]`

---

## 6. Domain Concept Functions (concept-level functions, not Function concepts)

### 6.1 The `ValueDomain` of a domain concept function must be a subconcept of `CustomFunction`
When setting the value domain for a function defined on a domain concept, the specified ValueDomain must be `CustomFunction` itself or one of its subconcepts.

- **Source:** `definitions/concept_definition_domain_concept.py` — `set_data`
- **Location:** `["concepts", <concept>, "data", "functions" <function>, "valueDomain"]`

---

## 7. ValueDomain Definitions (ValueDomain concepts)

### 7.1 The `defaultSerialization` values of all `ValueDomain` must be unique
This value is used to determine the type of an expression, when an expected expression type was not specified.
Thus, 1) this value must be a non-composite json type, and 2) its value must be unique across all ValueDomains to avoid ambiguity in this case!

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "data", "defaultSerialization"]`

---

## 7. Function Definitions (Function concepts)

### 7.1 Inversion may only be defined for arguments that exist in the evaluation interface
Each key of an `inversion` mapping must name an actual evaluation argument of the Function.

- **Source:** `definitions/concept_definition_functions.py` — `check_inversion_arguments`
- **Location:** `["concepts", <function>, "data", "inversion", <arg>]`

### 7.2 Template argument keys in template-specialised inversions must be declared template arguments
In the array syntax for template-specialised inversions, every non-`"procedure"` key in the specialisation object must be a declared template argument of the Function.

- **Source:** `definitions/concept_definition_functions.py` — `concept_data_check`
- **Location:** `["concepts", <function>, "data", "inversion", <index>, <t_arg>]`

### 7.3 Variation relations may only be defined for arguments that exist in the evaluation interface (object syntax)
In the object syntax for `variations`, each key must name an actual evaluation argument.

- **Source:** `definitions/concept_definition_functions.py` — `concept_data_check`
- **Location:** `["concepts", <function>, "data", "variations", <arg>]`

### 7.4 Variation relations may only be defined for arguments that exist in the evaluation interface (array syntax — string arg)
In the array syntax, when the first element is a string, that string must name an actual evaluation argument.

- **Source:** `definitions/concept_definition_functions.py` — `concept_data_check`
- **Location:** `["concepts", <function>, "data", "variations", <index>, 0]`

### 7.5 Variation relations may only be defined for arguments that exist in the evaluation interface (array syntax — array of args)
In the array syntax, when the first element is an array of argument names, every name in that array must be an actual evaluation argument.

- **Source:** `definitions/concept_definition_functions.py` — `concept_data_check`
- **Location:** `["concepts", <function>, "data", "variations", <index>, 0, <arg_index>]`

### 7.6 Sub-scope new-variable definitions may only target existing evaluation arguments
In `subScopes`, each key must name an actual evaluation argument of the Function.

- **Source:** `definitions/concept_definition_functions.py` — `concept_data_check`
- **Location:** `["concepts", <function>, "data", "subScopes", <arg>]`

### 7.7 The `[type, true]` dynamic-name variable syntax requires the key to be a `String`-typed evaluation argument
When a new-variable definition uses the `[type, true]` form (meaning the variable name is determined at call time), the JSON key must be the name of an evaluation argument of type `String`.

- **Source:** `definitions/concept_definition_functions.py` — `check_new_var_dict_def`
- **Location:** `["concepts", <function>, "data", "addNewVariablesInExistingScope", <new_var_name>]` and `["concepts", <function>, "data", "subScopes", <arg>, <new_var_name>]`

### 7.8 A dynamically-named sub-scope variable may not be scoped to the argument that determines its name
A variable whose name depends on the runtime value of argument `X` cannot be placed in the sub-scope of that same argument `X`.

- **Source:** `definitions/concept_definition_functions.py` — `check_new_var_dict_def`
- **Location:** `["concepts", <function>, "data", "subScopes", <arg>, <new_var_name>]`