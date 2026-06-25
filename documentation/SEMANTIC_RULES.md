# Concept Hierarchy — Semantic Rules

This document lists every semantic constraint enforced by the validator.
Each rule corresponds to a `CHSemanticError` raise in the source code.
<!-- Rules are grouped by the subsystem that enforces them.-->
<!-- Last verified against commit 8478adc028419291076736e877b91ff1bdcc426a, plus uncommitted working-tree changes present at verification time in:
     data/validators/value_instantiation_validator.py, definitions/concept_definition_domain_concept.py,
     definitions/concept_definition_functions.py, validator/concept_hierarchy_type_checks.py, validator/expression_checks.py,
     and the new/untracked: data/expressions/ module (expression.py, expression_errors.py, function_composition.py),
     data/parsers/expression_parser.py, data/value_domain_type.py.
     (examples/animal_kingdom.json also modified but is not a source-of-rules file; "documentation/SEMANTIC_RULES - Copy.md" is a stray backup, ignore it.)
     Next update: review `git diff 8478adc028419291076736e877b91ff1bdcc426a..HEAD` AND the current working-tree diff for CHSemanticError raise-site changes. -->

---

## 1. Hierarchy Structure

### 1.1 No cycles in the concept graph
Concepts must form a directed acyclic graph (DAG). If the topological sort detects a cycle, validation fails.

- **Source:** `validator/checker.py` — `check_structure`
- **Trigger:** `topological_sort` raises `RuntimeError` with a "Non-hierarchy structure detected" prefix

### 1.2 Every parent must be defined
Each concept's `directParents` list may only reference concepts that are themselves defined in the hierarchy.

- **Source:** `validator/checker.py` — `check_structure`
- **Location:** `["concepts", <concept>, "directParents", <index>]`

### 1.3 Exactly one root
The concept graph must have exactly one root (a concept with no parents). If topological sort yields multiple roots *and* the expected root name is among them, validation fails. (If the expected root name is not among the determined roots, an empty concept is created (with the expected name) and the previous roots become its direct children.)

- **Source:** `validator/checker.py` — `check_structure`

### 1.4 Root must have the canonical name
The single root concept must be named with `ConceptHierarchyModel.root_concept_name`. Any other name is rejected.

- **Source:** `validator/checker.py` — `check_structure`

### 1.5 References are valid identifiers
The referenced value of a concept must itself be a concept appearing in the concept definitions. Similarly, the reference value of an instance must be a defined instance.

- **Source:** `validator/checker.py` — `check_cycles_in_references_based_on_defined`
- **Location:** `[..., <reference>]`

### 1.6 No cycles in concept/instance references
If any subset of concepts or instances form a reference cycle (e.g. A references B and B references A), validation fails. Detected by iterative resolution that stalls before completion.

- **Source:** `validator/checker.py` — `check_cycles_in_references_based_on_defined`

### 1.7 Global variable names must not clash with concept names
A global variable (instance) and a concept may not share a name, as this creates lookup ambiguity.

- **Source:** `validator/checker.py` — `check_structure`
- **Location:** `["instances", <name>]`

---

## 2. Concept Classification and Parent-Type Rules

These checks run in `check_after_parsing_concepts` once every concept's own data has been parsed; they validate the relationship between a concept and its parents/siblings.

### 2.1 Every non-root concept must be classifiable
Each concept is classified as one of: `Function`, `ValueDomain`, or `DomainConcept`, based on its parents. A concept that cannot be assigned to any category is rejected.

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "directParents"]`

### 2.2 Template argument names must not clash with concept names
A template argument name may not coincide with any concept name in the hierarchy, since this creates ambiguity in constraint formulae, instantiations, and substitutions.

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "data", "templateArguments", "order", <index>]`

### 2.3 Parent template arguments must be substituted
For a templated `Function`/`ValueDomain`, every template argument of every parent must be substituted: either via the `"DirectParentName:ArgumentName"` syntax or the shorthand `"ArgumentName"` syntax (when unambiguous). Missing the `templateArguments`/`substitution` structure entirely, or omitting a specific parent argument, is rejected.

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "data"]` or `[..., "templateArguments", "substitution"]`

### 2.4 Shorthand template substitution must be unambiguous
When two or more parent concepts define a template argument with the same name, the shorthand `"ArgumentName"` substitution syntax is ambiguous; the `"ParentName:ArgumentName"` syntax must be used instead.

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "data", "templateArguments", "substitution"]`

### 2.5 No extra template substitution keys
The substitution definition may not contain keys that do not correspond to a template argument of any parent concept.

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "data", "templateArguments", "substitution"]`

### 2.6 Parents of a ValueDomain must be ValueDomains
A `ValueDomain` (other than direct children of `Concept`) may only have `ValueDomain` concepts as parents.

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "directParents", <index>]`

### 2.7 The `defaultSerialization` value of every ValueDomain must be unique
This value is used to determine the type of an expression when an expected expression type was not specified. Its value must therefore be unique across all `ValueDomain`s to avoid ambiguity.

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "data", "defaultSerialization"]`

### 2.8 Parents of a Function must be Functions
A `Function` (other than a direct child of `ValueDomain`) may only have `Function` concepts as parents.

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "directParents", <index>]`

### 2.9 A Function's result type may only be defined once
If a parent `Function` already defines a result type (or explicitly defines that it returns nothing), a subconcept may not redefine the result.

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "data", "interface", "res"]`

### 2.10 Function evaluation arguments must not be redefined incompatibly
If an evaluation argument with the same name is already defined by a parent `Function`, redefining it in a subconcept (with the same or a different type/modifier/reference definition) is rejected — it must be removed instead.

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "data", "interface", <arg>]`

### 2.11 Function evaluation argument names must not clash with global variable names
A Function's evaluation interface argument name may not coincide with any global variable name. The conflict would cause ambiguity in FunctionComposition procedures, inversions, and variations.

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "data", "interface", <arg>]`

### 2.12 Default argument values must reference an actual evaluation argument
Each key of `_defaultArgumentValues` must name an evaluation argument that exists on this Function or one of its parents.

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "data", "interface", "_defaultArgumentValues", <arg>]`

### 2.13 Parents of a Domain Concept must be Domain Concepts
A `DomainConcept` may only have `DomainConcept`s as parents.

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "directParents", <index>]`

### 2.14 Domain concept property names must not clash with global variable names
A domain concept's property name may not coincide with any global variable name. The conflict would cause ambiguity in property hooks, computations, concept functions, and management functions.

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "data", "properties", <prop>]`

### 2.15 Property names must be unique across the entire hierarchy
The same property name may not be defined in more than one domain concept. If it appears in two places, it must be moved to a shared ancestor or one occurrence must be renamed.

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "data", "properties", <prop>]`

### 2.16 Property names and function names must be disjoint
A name used as a property in one concept may not be used as a function in another concept (and vice versa).

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "data", "properties"|"functions", <name>]`

### 2.17 The `CONFIDENCE` keyword requires the `Duration` concept
A property may only use the `confidenceHalfDecayTime` definition keyword if a `Duration` concept (subconcept of `ValueDomain`) is defined in the hierarchy.

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "data", "properties", <prop>, "confidenceHalfDecayTime"]`

### 2.18 Domain concept function names must not clash with global variable names
Same constraint as 2.14, applied to domain concept functions.

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "data", "functions", <func>]`

### 2.19 Function names must be unique across the entire hierarchy
The same function name may not be defined in more than one domain concept.

- **Source:** `validator/checker.py` — `check_after_parsing_concepts`
- **Location:** `["concepts", <concept>, "data", "functions", <func>]`

---

## 3. Specialization (Inheritance Overrides for Properties and Functions)

`_specializations` lets a domain concept set, override ("for sub" vs. `_forThis`), or disambiguate property/function definition data inherited from its parents. These checks validate that mechanism.

### 3.1 Specialization target must be an available property/function
A name used as a key in `_specializations` must name a property or function that is either defined in this concept or inherited (available) from a parent.

- **Source:** `validator/domain_concept_specialization_checks.py` — `process_specialization_for_domain_concepts` (helper)
- **Location:** `[..., "_specializations", <name>]` (key)

### 3.2 `inheritFrom:` must reference a direct parent
The `"inheritFrom:<Parent>"` value used to disambiguate an inherited value must name a concept that is actually a direct parent of the current concept.

- **Source:** `validator/domain_concept_specialization_checks.py`
- **Location:** `[..., "_specializations", <name>, <def_key>]` (value)

### 3.3 `inheritFrom:` cannot target data defined in this concept
The `"inheritFrom:<Parent>"` syntax may only be used to disambiguate *inherited* data; it is rejected if the property/function is itself defined (not just inherited) in the current concept.

- **Source:** `validator/domain_concept_specialization_checks.py`
- **Location:** `[..., "_specializations", <name>, <def_key>]` (value)

### 3.4 Ambiguous inheritance must be disambiguated
If a property/function definition key is available from more than one parent and was not set or disambiguated (via `inheritFrom:`) in this concept, validation fails.

- **Source:** `validator/domain_concept_specialization_checks.py`

### 3.5 Own data must not be re-specialized at the top level
A property or function that is defined directly in this concept cannot also appear as a key of the top-level (for-sub) `_specializations` object — the concept's own definition already *is* the specialization passed to subconcepts.

- **Source:** `definitions/concept_definition_domain_concept.py` — `initialize_domain_concept_specialization_data_from_defined_data`
- **Location:** `["concepts", <concept>, "data", "properties"|"functions", "_specializations", <name>]` (key)

### 3.6 Only specializable property definition keywords may be specialized
A property definition keyword (e.g. `valueDomain`, `description`, `static`) that is not in the specializable keyword set may not appear inside a specialization object.

- **Source:** `definitions/concept_definition_domain_concept.py` — `check_property_data_types`
- **Location:** `[..., "_specializations"["_forThis"], <prop>]` (key)

### 3.7 Only specializable function definition keywords may be specialized
Same constraint as 3.6, for domain concept functions (whose only specializable keyword is `default`).

- **Source:** `definitions/concept_definition_domain_concept.py` — `check_function_data_types`
- **Location:** `[..., "_specializations"["_forThis"], <func>]` (key)

### 3.8 Function specialization values must have a recognized shape
A domain concept function specialization value must be a `FunctionComposition` procedure, a `CustomFunction` instantiation, or a JSON object using only the `default` keyword.

- **Source:** `definitions/concept_definition_domain_concept.py` — `check_function_data_types`
- **Location:** `[..., "_specializations"["_forThis"], <func>, "default"]` (value)

---

## 4. Concept Classification (Domain Concepts)

### 4.1 Every domain concept must carry data
A domain concept that is not the root concept must define at least one of: `properties`, `functions`, or `management`.

- **Source:** `definitions/concept_definition_domain_concept.py` — `concept_data_check`
- **Location:** the concept's own location

---

## 5. Hidden Implementation and Templates (Functions and ValueDomains)

### 5.1 Implementation file path must not include a file extension
The `implementation` field specifies a file path without extension. Paths containing a `.` character are rejected.

- **Source:** `definitions/concept_definition_hidden_implementation.py` — `check_implementation`
- **Location:** `["concepts", <concept>, "data", "implementation"]`

### 5.2 Template argument names must be unique
A template argument name may not be declared twice in the `order` list of the same definition.

- **Source:** `definitions/concept_definition_hidden_implementation.py` — `check_template_argument_definition_list`
- **Location:** `["concepts", <concept>, "data", "templateArguments"["order"], <index>]`

### 5.3 Template argument constraints must be defined on declared template arguments
A constraint entry in the `templateArguments` object must use a key that is already declared in the `order` list of the same definition.

- **Source:** `definitions/concept_definition_hidden_implementation.py` — `check_template_arguments`
- **Location:** `["concepts", <concept>, "data", "templateArguments", <arg>]` (key)

### 5.4 Variadic group identifiers may only be defined on declared template arguments
Each key in `variadicGroupIdentifiers` must name an argument present in `template_argument_order`.

- **Source:** `definitions/concept_definition_hidden_implementation.py` — `check_template_arguments`
- **Location:** `["concepts", <concept>, "data", "templateArguments", "variadicGroupIdentifiers", <arg>]` (key)

### 5.5 Variadic group identifiers may only be defined on variadic template arguments
A key in `variadicGroupIdentifiers` must name an argument that was declared with the `...` variadic marker.

- **Source:** `definitions/concept_definition_hidden_implementation.py` — `check_template_arguments`
- **Location:** `["concepts", <concept>, "data", "templateArguments", "variadicGroupIdentifiers", <arg>]` (key)

### 5.6 The empty variadic group identifier is only allowed for all-variadic-template concepts
An empty string `""` may be used as a variadic group identifier only when every template argument of the concept is variadic.

- **Source:** `definitions/concept_definition_hidden_implementation.py` — `check_template_arguments`
- **Location:** `["concepts", <concept>, "data", "templateArguments", "variadicGroupIdentifiers", <arg>]` (value)

### 5.7 Variadic group identifiers must be defined for either all or none of the variadic template arguments
If one variadic template argument has a `variadicGroupIdentifiers` entry, every variadic template argument of the concept must also have one.

- **Source:** `definitions/concept_definition_hidden_implementation.py` — `check_template_arguments`
- **Location:** `["concepts", <concept>, "data", "templateArguments", "variadicGroupIdentifiers"]`

### 5.8 Duplicate variadic group identifiers are not allowed
The same variadic group identifier string may not be assigned to two different variadic template arguments.

- **Source:** `definitions/concept_definition_hidden_implementation.py` — `check_template_arguments`
- **Location:** `["concepts", <concept>, "data", "templateArguments", "variadicGroupIdentifiers", <arg>]` (value)

---

## 6. ValueDomain Definitions (ValueDomain concepts)

### 6.1 Abstract concepts must not define an instantiation structure
If a `ValueDomain` is marked `abstract: true`, it may not simultaneously define an `instantiation` field.

- **Source:** `definitions/concept_definition_value_domain.py` — `check_instantiation`
- **Location:** `["concepts", <concept>, "data", "instantiation"]` (key)

### 6.2 Template-specific instantiation constraint lists must match the ValueDomain's template-argument arity
In the array syntax for template-specialised `instantiation` entries, the first element (the list of per-template-argument constraint formulae) must have exactly one entry per declared template argument of the `ValueDomain`.

- **Source:** `definitions/concept_definition_value_domain.py` — `check_instantiation`
- **Location:** `["concepts", <concept>, "data", "instantiation", <index>, 0]`

### 6.3 Template-specific instantiation specialization keys must be unique
The same tuple of template-argument constraint formulae may not be used as the specialization key of more than one `instantiation` entry — it would be ambiguous which deserialization structure applies.

- **Source:** `definitions/concept_definition_value_domain.py` — `check_instantiation`
- **Location:** `["concepts", <concept>, "data", "instantiation", <index>]`

---

## 7. Function Definitions (Function concepts)

### 7.1 A Function may have at most one parent concept
Multiple inheritance is not supported for `Function` concepts.

- **Source:** `definitions/concept_definition_functions.py` — `concept_data_check`
- **Location:** `["concepts", <concept>, "directParents"]`

### 7.2 Inversion may only be defined for arguments that exist in the evaluation interface
Each key of an `inversion` mapping must name an actual evaluation argument of the Function.

- **Source:** `definitions/concept_definition_functions.py` — `check_inversion_arguments`
- **Location:** `["concepts", <function>, "data", "inversion", <arg>]` (key)

### 7.3 Template-specific inversion constraint lists must match the Function's template-argument arity
In the array syntax for template-specialised `inversion` entries, the first element (the list of per-template-argument constraint formulae) must have exactly one entry per declared template argument of the Function.

- **Source:** `definitions/concept_definition_functions.py` — `concept_data_check`
- **Location:** `["concepts", <function>, "data", "inversion", <index>, 0]`

### 7.4 Template-specific inversion specialization keys must be unique
The same tuple of template-argument constraint formulae may not be used as the specialization key of more than one `inversion` entry — it would be ambiguous which inversion procedure applies.

- **Source:** `definitions/concept_definition_functions.py` — `concept_data_check`
- **Location:** `["concepts", <function>, "data", "inversion", <index>]`

### 7.5 Variation relations may only be defined for arguments that exist in the evaluation interface (object syntax)
In the object syntax for `variations`, each key must name an actual evaluation argument.

- **Source:** `definitions/concept_definition_functions.py` — `concept_data_check`
- **Location:** `["concepts", <function>, "data", "variations", <arg>]` (key)

### 7.6 Variation relations may only be defined for arguments that exist in the evaluation interface (array syntax — string arg)
In the array syntax, when the first element is a string, that string must name an actual evaluation argument.

- **Source:** `definitions/concept_definition_functions.py` — `concept_data_check`
- **Location:** `["concepts", <function>, "data", "variations", <index>, 0]`

### 7.7 Variation relations may only be defined for arguments that exist in the evaluation interface (array syntax — array of args)
In the array syntax, when the first element is an array of argument names, every name in that array must be an actual evaluation argument.

- **Source:** `definitions/concept_definition_functions.py` — `concept_data_check`
- **Location:** `["concepts", <function>, "data", "variations", <index>, 0, <arg_index>]`

### 7.8 Sub-scope new-variable definitions may only target existing evaluation arguments
In `subScopes`, each key must name an actual evaluation argument of the Function.

- **Source:** `definitions/concept_definition_functions.py` — `concept_data_check`
- **Location:** `["concepts", <function>, "data", "subScopes", <arg>]` (key)

### 7.9 The `[type, true]` dynamic-name variable syntax requires the key to be a `String`-typed evaluation argument
When a new-variable definition uses the `[type, true]` form (meaning the variable name is determined at call time), the JSON key must be the name of an evaluation argument of type `String`.

- **Source:** `definitions/concept_definition_functions.py` — `check_new_var_dict_def`
- **Location:** `["concepts", <function>, "data", "addNewVariablesInExistingScope", <new_var_name>]` and `["concepts", <function>, "data", "subScopes", <arg>, <new_var_name>]` (key)

### 7.10 A dynamically-named sub-scope variable may not be scoped to the argument that determines its name
A variable whose name depends on the runtime value of argument `X` cannot be placed in the sub-scope of that same argument `X`.

- **Source:** `definitions/concept_definition_functions.py` — `check_new_var_dict_def`
- **Location:** `["concepts", <function>, "data", "subScopes", <arg>, <new_var_name>]` (key)

---

## 8. External Data

### 8.1 External data files must be resolvable
When a concept definition references an external data file, that file must exist and be loadable. A `RuntimeError` from the loader (indicating the file could not be found) is re-raised as a semantic error.

- **Source:** `definitions/concept_definition.py` — `data` property
- **Location:** the concept's data location

---

## 9. Type Checking (Properties, Functions, Template Substitutions)

These checks run in `check_types_in_concept_hierarchy` (after structure and specialization are validated) and resolve/validate the *types* used throughout the hierarchy.

### 9.1 A property's `valueDomain` definition must parse as a valid type
The string value of a property's `valueDomain` keyword must be parseable into a concept-hierarchy type (a defined `ValueDomain`/concept with correctly-instantiated template arguments).

- **Source:** `validator/concept_hierarchy_type_checks.py` — `check_types_in_domain_concept_definition`
- **Location:** `["concepts", <concept>, "data", "properties", <prop>, "valueDomain"]`

### 9.2 A property's type must be a subtype of ValueDomain
The resolved type of a property must be a `ValueDomain` or a subconcept thereof.

- **Source:** `validator/concept_hierarchy_type_checks.py` — `check_types_in_domain_concept_definition`
- **Location:** `["concepts", <concept>, "data", "properties", <prop>, "valueDomain"]`

### 9.3 `DEFAULT_INSTANCE_NAMING` requires the `InstanceBase` concept
The `nameDefaultInstanceValuesWithThisInstanceName` property keyword may only be used if an `InstanceBase` concept (subconcept of `ValueDomain`) is defined in the hierarchy.

- **Source:** `validator/concept_hierarchy_type_checks.py` — `check_types_in_domain_concept_definition`
- **Location:** `["concepts", <concept>, "data", "properties", <prop>, "nameDefaultInstanceValuesWithThisInstanceName"]` (key)

### 9.4 `DEFAULT_INSTANCE_NAMING` requires an instance-typed property
A property may only set `nameDefaultInstanceValuesWithThisInstanceName: true` if its resolved type contains a subtype of `InstanceBase`.

- **Source:** `validator/concept_hierarchy_type_checks.py` — `check_types_in_domain_concept_definition`
- **Location:** `["concepts", <concept>, "data", "properties", <prop>, "nameDefaultInstanceValuesWithThisInstanceName"]` (value)

### 9.5 A domain concept function's `valueDomain` definition must parse as a valid type
Same as 9.1, applied to domain concept functions.

- **Source:** `validator/concept_hierarchy_type_checks.py` — `check_types_in_domain_concept_definition`
- **Location:** `["concepts", <concept>, "data", "functions", <func>, "valueDomain"]`

### 9.6 The ValueDomain of a domain concept function must be a subconcept of CustomFunction
When setting the value domain for a function defined on a domain concept, the specified `ValueDomain` must be `CustomFunction` itself or one of its subconcepts.

- **Source:** `validator/concept_hierarchy_type_checks.py` — `check_types_in_domain_concept_definition`
- **Location:** `["concepts", <concept>, "data", "functions", <func>, "valueDomain"]`

### 9.7 Template argument substitution values must parse and validate
Each value substituted for a parent's template argument (in `templateArguments.substitution`) must successfully parse into a template argument value and satisfy the template constraints of any fully-instantiated types it contains.

- **Source:** `validator/concept_hierarchy_type_checks.py` — `check_types_in_hidden_implementation_definition`
- **Location:** `["concepts", <concept>, "data", "templateArguments", "substitution", <key>]`

### 9.8 Template substitution must satisfy the parent's template constraints
The full set of substitution values for a parent's template arguments must, together, satisfy that parent's own template constraint formula.

- **Source:** `validator/concept_hierarchy_type_checks.py` — `check_types_in_hidden_implementation_definition`
- **Location:** `["concepts", <concept>, "data", "templateArguments", "substitution"]`

### 9.9 A Function's evaluation argument type must parse as a valid type
Each evaluation argument's declared type must be parseable into a concept-hierarchy type.

- **Source:** `validator/concept_hierarchy_type_checks.py` — `check_types_in_function_definition`
- **Location:** `["concepts", <function>, "data", "interface", <arg>]`

### 9.10 A Function's result type must parse as a valid type
The declared result type (`res`) must be parseable into a concept-hierarchy type.

- **Source:** `validator/concept_hierarchy_type_checks.py` — `check_types_in_function_definition`
- **Location:** `["concepts", <function>, "data", "interface", "res"]`

### 9.11 A sub-scope new-variable's type must parse as a valid type
Each new variable added to the sub-scope of an evaluation argument (`subScopes`) must declare a type that parses successfully.

- **Source:** `validator/concept_hierarchy_type_checks.py` — `check_types_in_function_definition`
- **Location:** `["concepts", <function>, "data", "subScopes", <arg>, <new_var_name>]`

### 9.12 A new variable added to the existing scope must declare a valid type
Each entry of `addNewVariablesInExistingScope` must declare a type that parses successfully.

- **Source:** `validator/concept_hierarchy_type_checks.py` — `check_types_in_function_definition`
- **Location:** `["concepts", <function>, "data", "addNewVariablesInExistingScope", <new_var_name>]`

### 9.13 A substituted template argument must produce a fully-instantiated type that satisfies its own instantiation constraints
When substituting template variables inside a value (e.g. while computing whether one type is a subtype of another, via `ConstraintValidator.is_subtype`), if the substitution fully instantiates a sub-value's type, that resulting type must itself satisfy the template-instantiation constraints of its own concept.

- **Source:** `validator/concept_hierarchy_type_checks.py` — `substitute_non_template_variable`
- **Location:** the location of the substituted sub-value within the type expression being substituted

### 9.14 Merging substitution-derived constraints with a parent's template context must leave it satisfiable
After computing the template-argument substitution for a parent and merging the constraints it implies into the current concept's own `TemplateContext` (`merge_in_place`), the merged context must not be empty (i.e. the combined constraints must not be mutually contradictory) — otherwise no concrete type could ever instantiate this concept.

- **Source:** `validator/concept_hierarchy_type_checks.py` — `check_types_in_hidden_implementation_definition`
- **Location:** the substitution's own location

---

## 10. Type Expression Validation (Type Parsing and Template Instantiation)

These checks live in `data/validators/type_validator.py` and validate a single parsed type expression (e.g. `Sequence<Integer>`) in isolation, independent of where it is used.

### 10.1 A type identifier must be a known concept or template variable
`ParsedType.clean_name` must resolve either to a template variable in scope or to a defined concept.

- **Source:** `data/validators/type_validator.py` — `validate_type_and_parse_to_variadic_groups`, `validate_type`

### 10.2 The variadic expansion operator (`...`) may only be used on template arguments
Using `...` directly on a concept name (rather than on a template variable) is invalid.

- **Source:** `data/validators/type_validator.py` — `validate_type_and_parse_to_variadic_groups`

### 10.3 Template arguments must be specified exactly when (and only when) the type is templated
A templated concept's instantiation must specify template arguments; a non-templated concept's instantiation must not.

- **Source:** `data/validators/type_validator.py` — `validate_type_and_parse_to_variadic_groups`

### 10.4 The number of supplied template arguments must match the type's declared arity
Too few (non-variadic-eligible) arguments, or a count that doesn't reconcile with the number of variadic template arguments, is rejected.

- **Source:** `data/validators/type_validator.py` — `validate_type_and_parse_to_variadic_groups`

### 10.5 Variadic group identifiers used in an instantiation must be defined by the type
A template-argument value tagged with a variadic identifier (e.g. `Foo!`) must reference an identifier actually declared via `variadicGroupIdentifiers` for that type.

- **Source:** `data/validators/type_validator.py` — `validate_type_and_parse_to_variadic_groups`

### 10.6 The empty variadic identifier may only be omitted in allowed contexts
A template-argument value with no variadic identifier is rejected unless the type defines the empty identifier or has non-variadic arguments to disambiguate against.

- **Source:** `data/validators/type_validator.py` — `validate_type_and_parse_to_variadic_groups`

### 10.7 Variadic groups and variadic identifiers may not be mixed in the same instantiation
A template instantiation must consistently use either the explicit variadic-group-list syntax or the per-value variadic-identifier syntax, not both.

- **Source:** `data/validators/type_validator.py` — `validate_type_and_parse_to_variadic_groups`

### 10.8 Every non-variadic template argument must receive a value
When using variadic identifiers, a value must be supplied for each non-variadic template argument of the type.

- **Source:** `data/validators/type_validator.py` — `validate_type_and_parse_to_variadic_groups`

### 10.9 The variadic expansion operator may only be used where allowed, and only on variadic template variables
Using `...` on a non-variadic template variable, or using it on a variadic template variable in a context that disallows expansion, is rejected.

- **Source:** `data/validators/type_validator.py` — `validate_type`

### 10.10 A template variable may not itself be given template arguments
Writing `T<...>` where `T` is a template variable (rather than a concept) is invalid.

- **Source:** `data/validators/type_validator.py` — `validate_type`

### 10.11 A parsed type name must resolve to either a template variable or a concept
Fallback check after the template-variable branch: if the name is not a concept either, the type is invalid.

- **Source:** `data/validators/type_validator.py` — `validate_type`

### 10.12 Literal values cannot be used as variadic template argument values
A literal (e.g. a number or string) may not be substituted for a variadic template argument slot.

- **Source:** `data/validators/type_validator.py` — `validate_template_argument_value`

### 10.13 Non-variadic template variables and ordinary types cannot be used as variadic argument values
Only a variadic template variable (or a variadic group) may be substituted where a variadic template argument is expected.

- **Source:** `data/validators/type_validator.py` — `validate_template_argument_value`

### 10.14 A variadic group cannot be used as a non-variadic template argument value
The inverse of 10.12/10.13: a variadic group may not be substituted into a non-variadic template argument slot.

- **Source:** `data/validators/type_validator.py` — `validate_template_argument_value`

### 10.15 A parsed type expression must resolve to the expected category
Helper entry points (`parse_convert_type`, `parse_convert_type_in_template_context`) require the parsed-and-converted result to be an `InstantiatedType` (or, in the template-context variant, also a `TemplateDependentType`/template variable); anything else is rejected.

- **Source:** `data/validators/type_validator.py` — `parse_convert_type`, `parse_convert_type_in_template_context`

---

## 11. Template Constraint Formulae and Their Validation

`data/template_argument_constraints/constraint_formula.py` validates constraint *formulae* themselves (e.g. `Sequence<ValueDomain>`); `data/validators/template_argument_constraints_validator.py` validates that a concrete template-argument *value* satisfies such a formula.

### 11.1 A constraint formula's literal must be a concept or template variable
The literal name used in a `TemplateConstraintHierarchyOperator` formula (e.g. the `Sequence` in `DescendantsOf<Sequence>`) must resolve to a concept or a template variable.

- **Source:** `data/template_argument_constraints/constraint_formula.py` — `TemplateConstraintHierarchyOperator.__init__`

### 11.2 A template-variable constraint literal cannot also constrain template arguments
Writing `T<SomeConstraint>` where `T` is a template variable is invalid — a template variable has no template arguments of its own to constrain.

- **Source:** `data/template_argument_constraints/constraint_formula.py` — `TemplateConstraintHierarchyOperator.__init__`

### 11.3 Template argument constraints require a templated literal
Specifying template-argument constraints (e.g. `Foo<X>`) on a literal `Foo` that itself has no template arguments is invalid.

- **Source:** `data/template_argument_constraints/constraint_formula.py` — `TemplateConstraintHierarchyOperator.__init__`

### 11.4 The number of template-argument constraints must match the literal's arity
If `Foo` has *n* template arguments, a constraint `Foo<C1, ..., Cm>` must specify exactly *n* constraints.

- **Source:** `data/template_argument_constraints/constraint_formula.py` — `TemplateConstraintHierarchyOperator.__init__`

### 11.5 The `Empty` constraint never matches
A constraint formula explicitly built as `Empty` always fails validation for any value — this is reported as a semantic error during instantiation-constraint validation.

- **Source:** `data/validators/template_argument_constraints_validator.py` — `_delegate_constraint_check`

### 11.6 A conjunction (`And`) constraint requires all sub-formulae to be satisfied
If any sub-formula of an `And` constraint fails against the value, the whole constraint fails (with the individual failures attached as causes).

- **Source:** `data/validators/template_argument_constraints_validator.py` — `_validate_and`

### 11.7 A disjunction (`Or`) constraint requires at least one sub-formula to be satisfied
If every sub-formula of an `Or` constraint fails against the value, the whole constraint fails (with the individual failures attached as causes).

- **Source:** `data/validators/template_argument_constraints_validator.py` — `_validate_or`

### 11.8 A negated (`Not`) structure constraint must actually constrain something to succeed
If the negated sub-formula matches without producing any constraint on template variables, the negation cannot meaningfully fail it, so it is rejected as unsatisfiable.

- **Source:** `data/validators/template_argument_constraints_validator.py` — `_validate_not`, `validate_complete_instantiation_of_type`

### 11.9 A value must satisfy the hierarchy-operator constraint placed on it
A type value must pass the subconcept/superconcept check (`DescendantsOf`/`AscendantsOf`/etc.) against the constraint's literal concept; a literal JSON value can never satisfy a type-hierarchy constraint.

- **Source:** `data/validators/template_argument_constraints_validator.py` — `_validate_type`

### 11.10 Substituted template arguments of a matched type must themselves satisfy the constraint's nested constraints
Once a value's outer concept matches a hierarchy-operator constraint with its own template-argument constraints, the value's (substituted) template arguments must satisfy those nested constraints too.

- **Source:** `data/validators/template_argument_constraints_validator.py` — `_validate_type`

### 11.11 A non-type (literal) constraint cannot be satisfied by a type value
A constraint on a literal kind (`int`/`float`/`bool`/`string`) fails if the value being checked is a `ConceptHierarchyType` rather than a literal.

- **Source:** `data/validators/template_argument_constraints_validator.py` — `_validate_literal`

### 11.12 A literal value must match its literal-kind or literal-value constraint
A literal substituted for a non-type template argument must satisfy the declared literal-type check (e.g. `is_integer`) or, for an exact-value constraint, must equal the expected value.

- **Source:** `data/validators/template_argument_constraints_validator.py` — `_validate_literal`

### 11.13 A template argument's constraint definition may not contain the Unconstrained specifier
When parsing a constraint *definition* (as opposed to a constraint reference inside another formula), an empty/whitespace-only formula — which would otherwise parse to `Unconstrained` — is rejected; template arguments must always declare an explicit constraint.

- **Source:** `data/parsers/template_argument_constraint_parser.py` — `TemplateArgumentConstraintParser._parse_non_structure_constraint` (raised when `allow_unconstrained=False`)

### 11.14 A ValueDomain's (and Function's because Functions are ValueDomains) per-template-argument constraint formula must parse to a structure constraint
When building a `ValueDomain`'s or a `Function`'s template-argument constraint context, each declared template argument's constraint formula (parsed with `allow_unconstrained=False`, see 11.13) must resolve to a `NonStructureConstraintFormula`; anything else is rejected.

- **Source:** `validator/value_domain_template_constraint_checks.py` — `check_value_domain_template_constraint_formulae`
- **Location:** the template argument's own location (or `[]` for the default constraint, which should never raise an error)

### 11.15 A logical composition (`And`/`Or`) constraint may not combine sub-formulae of different constraint kinds
Every sub-formula of an `And`/`Or` constraint must agree on its `constraint_type` (e.g. all `"type"`, or all the same literal kind such as `"int"`); `Unconstrained` sub-formulae are ignored for this check, but mixing, say, a type constraint with an `int` literal constraint inside the same `And`/`Or` is rejected as meaningless.

- **Source:** `data/template_argument_constraints/constraint_formula.py` — `TemplateConstraintAnd.__init__`, `TemplateConstraintOr.__init__`

### 11.16 A ValueDomain's per-template-argument constraints must not collectively prevent any instantiation
After building the `TemplateContext` from a `ValueDomain`'s declared template-argument constraint formulae, that context must not be empty — i.e. the constraints (each individually valid per 11.14) must not combine to leave no satisfiable instantiation at all.

- **Source:** `validator/value_domain_template_constraint_checks.py` — `check_value_domain_template_constraint_formulae`
- **Location:** `["concepts", <concept>, "data", "templateArguments"]`

---

## 12. Instantiation Value Validation (JSON-Schema-Based)

`data/validators/value_instantiation_validator.py` validates a concrete JSON value against a `ValueDomain`'s `instantiation` schema (a JSON-Schema-Draft-07-derived AST). All errors below carry a path into the *value*, not the schema.

### 12.1 A value is rejected outright by a `false` boolean schema
If a schema node's canonical form is the boolean `false`, no value is acceptable at that location.

- **Source:** `data/validators/value_instantiation_validator.py` — `_validate`

### 12.2 A value must satisfy its node's own JSON-Schema keywords
`type`, `enum`, `const`, numeric/string/array size constraints, `pattern`, `format`, etc. are checked via `Draft7Validator`; any violation is reported at the value's path.

- **Source:** `data/validators/value_instantiation_validator.py` — `_validate`

### 12.3 Required object properties must be present
Each key listed in a schema node's `required` must be present in the value being validated (checked explicitly so the error points at the missing key itself).

- **Source:** `data/validators/value_instantiation_validator.py` — `_validate`
- **Location:** points at the missing key (`part=KEY`)

### 12.4 Additional object properties are rejected when disallowed
If a schema node sets `additionalProperties: false`, any value key not matched by `properties`/`patternProperties` is rejected.

- **Source:** `data/validators/value_instantiation_validator.py` — `_validate_object`
- **Location:** points at the offending key (`part=KEY`)

### 12.5 Property names must satisfy `propertyNames`
If a schema node declares `propertyNames`, every key of the value (as a string) must itself satisfy that sub-schema (or, for a custom type, pass `check_value`).

- **Source:** `data/validators/value_instantiation_validator.py` — `_validate_object`

### 12.6 Additional array items are rejected when disallowed
If a schema node (using the tuple-validation array form) sets `additionalItems: false`, any item beyond the declared positional items is rejected.

- **Source:** `data/validators/value_instantiation_validator.py` — `_validate_array`

### 12.7 A `contains` schema must match at least one array element
If a schema node declares `contains`, at least one element of the array value must satisfy that sub-schema.

- **Source:** `data/validators/value_instantiation_validator.py` — `_validate_array`

### 12.8 A value must not match a `not` schema
If a schema node declares `not`, the value is rejected if it *does* satisfy that sub-schema.

- **Source:** `data/validators/value_instantiation_validator.py` — `_validate`

### 12.9 A value must match at least one branch of `anyOf`
If every branch of an `anyOf` schema fails, the value is rejected (with each branch's failures attached as causes).

- **Source:** `data/validators/value_instantiation_validator.py` — `_validate_any_of`

### 12.10 A value must match exactly one branch of `oneOf`
Matching zero branches, or matching more than one, is rejected.

- **Source:** `data/validators/value_instantiation_validator.py` — `_validate_one_of`

### 12.11 A custom-typed schema node's value must satisfy the value context's check
Values at a custom-type node (e.g. an `InstanceBase` or `Reference` type) are delegated to `CHValueContext.check_value`, which may itself return a `CHSemanticError`.

- **Source:** `data/validators/value_instantiation_validator.py` — `_validate`

### 12.12 A `requireAllKeysFromProperties` object must define every key listed in `properties`
This is distinct from the standard draft-07 `required` keyword (12.3): when a schema node sets the custom `requireAllKeysFromProperties` flag, every key declared in that node's `properties` map is treated as required, even though `required` itself may list none/some of them.

- **Source:** `data/validators/value_instantiation_validator.py` — `_validate_object`
- **Location:** the containing object's path (`part=VALUE`)

---

## 13. JSON-Schema Definition Parsing (Custom Concept-Hierarchy Schema Syntax)

These checks run while *parsing* a `ValueDomain`/property JSON-Schema-derived type definition into a `CHSchemaNode` (i.e. while a concept's schema is being built, not while a value is later validated against it) — see `data/parsers/jsonschema_parser.py`.

### 13.1 A `props`/`funcs` concept restriction must name a defined concept
The custom `"props(...)"`/`"funcs(...)"` concept-data-constraint syntax restricts which concept(s) a property/function reference applies to; every concept name listed inside the parentheses must be a concept that actually exists in the hierarchy.

- **Source:** `data/parsers/jsonschema_parser.py` — `_parse_custom_concept_data_constraint`
- **Location:** the constraint's own location

### 13.2 A literal template variable used for a numeric/size schema keyword must have the matching literal kind
`minProperties`, `maxProperties`, `minLength`, `maxLength`, `minItems`, `maxItems` (integer-kind), and `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `multipleOf` (number-kind) may be given as a string naming a template variable instead of a literal; that template variable must be declared with the matching literal kind (`integer`/`number`), or it is rejected.

- **Source:** `data/parsers/jsonschema_parser.py` — `_finish_builtin_node`
- **Location:** the keyword's own location (e.g. `[..., "minItems"]`)
