# ConceptHierarchy

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

## Development

All commands below should be run from the **repository root** (the directory that contains `pyproject.toml`).

1. Clone the repo and install the package in editable mode with dev dependencies:

```bash
git clone https://github.com/AndreiCostinescu/ConceptHierarchy.git
cd ConceptHierarchy
pip install -e ".[dev]"
```

2. Run the test suite:

```bash
pytest
```

3. Run with coverage:

```bash
pytest --cov=concept_hierarchy --cov-report=term-missing
```

4. Run type checks:

```bash
mypy src/
```

5. Test across all supported Python versions (requires the interpreters to be installed):

```bash
tox
```

6. Build a distribution:

```bash
python -m build
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).