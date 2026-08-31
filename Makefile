.PHONY: setup lint format

setup:
	-git config --unset-all core.hooksPath
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .