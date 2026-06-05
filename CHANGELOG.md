# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-05
### Added
- Initial project structure.
- JSON parser producing an immutable internal model (using `frozendict`).
- Syntax validator: identifier naming rules.
- Semantic validator: undefined parents, duplicate names, cycle detection.
- Naive C++ code-generation backend
- `concept-hierarchy compile` CLI command.
- Python 3.7–3.12 compatibility.
