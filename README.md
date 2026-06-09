![Logo](Logo.png) 
# ConceptHierarchy

[![CI](https://github.com/AndreiCostinescu/ConceptHierarchy/actions/workflows/ci.yml/badge.svg)](https://github.com/AndreiCostinescu/ConceptHierarchy/actions/workflows/ci.yml)
[![Lint](https://github.com/AndreiCostinescu/ConceptHierarchy/actions/workflows/lint.yml/badge.svg)](https://github.com/AndreiCostinescu/ConceptHierarchy/actions/workflows/lint.yml)

**ConceptHierarchy** is the compiler for the Concept Hierarchy knowledge programming language.

It takes a JSON definition of a concept hierarchy, validates its syntax and
semantics, and generates an implementation in a target programming language
(currently **C++**).

## Installation

```bash
pip install ConceptHierarchy
```

## Quick start

1. Write a hierarchy definition in JSON:

```json
{
    "Concept": {},
    "Animal": {
        "directParents": ["Concept"],
        "data": {
            "properties": {
                "age": "Integer",
                "name": "String"
            }
        }
    },
    "Dog": {
        "directParents": ["Animal"],
        "data": {
            "properties": {
                "breed": "String"
            }
        }
    },
    "ValueDomain": {
        "directParents": ["Concept"],
        "data": {
            "abstract": true
        }
    },
    "Integer": {
        "directParents": ["ValueDomain"],
        "data": {
            "instantiation": "integer"
        }
    },
    "String": {
        "directParents": ["ValueDomain"],
        "data": {
            "instantiation": "string"
        }
    }
}
```

2. Compile it:

```bash
concept-hierarchy compile animal_kingdom.json --target cpp --output animal_kingdom.hpp
```

Or use the Python API:

```python
from concept_hierarchy import ch_compile

code = ch_compile("animal_kingdom.json", target="cpp")
print(code)
```

## Project layout

```
src/concept_hierarchy/
├── __init__.py                                     # Public API: exports version and the ch_check and ch_compile functions
├── cli.py                                          # CLI entry point (compile / validate subcommands)
├── compiler.py                                     # Pipeline orchestrator: parse → validate → codegen
├── errors.py                                       # CHSyntaxError / CHSemanticError with location tracking
├── models.py                                       # ConceptHierarchyModel: root data structure from parsing
├── utils.py                                        # Path helpers, string utilities, tab constant
│
├── definitions/                                    # Parsed and validated definition objects
│   ├── definition.py                               # Abstract base: name validation and location tracking
│   ├── concept_definition.py                       # ConceptDefinition: parents, description, raw data
│   ├── concept_definition_hidden_implementation.py # HiddenImplementationDefinition: template args, abstract flag, impl path
│   ├── concept_definition_value_domain.py          # ValueDomainDefinition: instantiation and serialization config
│   ├── concept_definition_functions.py             # FunctionDefinition: interface, procedure, inversion, variations, scopes
│   ├── concept_definition_domain_concept.py        # DomainConceptDefinition + PropertyDefinition: properties and functions
│   ├── global_variable_definition.py               # GlobalVariableDefinition: global instance variables
│   └── utils.py                                    # check_ch_name: identifier naming rules
│
├── validator/                                      # Semantic validation
│   └── checker.py                                  # check_model: full syntax and semantic checks on the model
│
├── codegen/                                        # Code generation dispatch
│   └── generator.py                                # generate(model, target): routes to the appropriate backend
│
└── backends/                                       # Language-specific code generators
    ├── base.py                                     # BaseBackend: abstract interface (generate(model) → str)
    └── cpp.py                                      # C++ backend: emits .hpp header with class hierarchies
```

## Supported backends

| Target | Flag           | Output               |
|--------|----------------|----------------------|
| C++    | `--target cpp` | Header file (`.hpp`) |

## Contributing

Please read the [contribution guide](CONTRIBUTING.md) for full details on DCO
sign-off, GPG commit signing, coding standards, and the PR checklist.

**Quickstart:**

```bash
git clone https://github.com/AndreiCostinescu/ConceptHierarchy.git
cd ConceptHierarchy
make setup
```

`make setup` installs dev dependencies (including ruff and pre-commit) and
registers the git hooks so formatting and linting run automatically on every
commit.

```bash
make lint      # check formatting + linting
make format    # auto-fix formatting and safe lint issues
pytest         # run the test suite
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).