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