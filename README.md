# ConceptHierarchy

[![Lint](https://github.com/AndreiCostinescu/ConceptHierarchy/actions/workflows/lint.yml/badge.svg)](https://github.com/AndreiCostinescu/ConceptHierarchy/actions/workflows/lint.yml)

**ConceptHierarchy** is a compiler for the ConceptHierarchy programming language.

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
            "fromJsonLiteral": "integer"
        }
    },
    "String": {
        "directParents": ["ValueDomain"],
        "data": {
            "fromJsonLiteral": "string"
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
from concept_hierarchy import compile_hierarchy

code = compile_hierarchy("animal_kingdom.json", target="cpp")
print(code)
```

## Project layout

```
src/concept_hierarchy/
├── __init__.py          # Public API
├── compiler.py          # Pipeline orchestrator
├── cli.py               # Command-line interface
├── models.py            # Internal AST / data model
├── errors.py            # Exception hierarchy
├── parser/              # JSON → model
├── validator/           # Syntax & semantic checks
├── codegen/             # Dispatcher
└── backends/            # Language backends (cpp, …)
```

## Supported backends

| Target | Flag       | Output              |
|--------|------------|---------------------|
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