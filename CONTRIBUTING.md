# Contributing to ConceptHierarchy

Thank you for considering a contribution!  This document explains the two
legal/process requirements every contributor must satisfy, plus the day-to-day
development workflow.

---

## Table of contents

1. [Developer Certificate of Origin (DCO)](#1-developer-certificate-of-origin-dco)
2. [GPG / SSH commit signing](#2-gpg--ssh-commit-signing)
3. [Development setup](#3-development-setup)
4. [Development workflow](#4-development-workflow)
5. [Coding standards](#5-coding-standards)
6. [Starting development](#6-starting-development)
7. [Running tests](#7-running-tests)

---

## 1. Developer Certificate of Origin (DCO)

Every commit must include a Signed-off-by: line. 
Just use git commit -s and Git adds it automatically. 
This is the Developer Certificate of Origin — 
it's your certification that you wrote the code and have the right to contribute it under the Apache 2.0 license.
**Every commit** — *not just once per contributor* — must include the
`Signed-off-by:` trailer.

Add it automatically by passing `-s` to every `git commit`:

```bash
git commit -s -m "feat: add something useful"
```

This appends:

```
Signed-off-by: Your Name <you@example.com>
```

The DCO workflow in `.github/workflows/dco.yml` checks every commit in your
PR.  A PR with even one unsigned commit will be blocked from merging.

**Amending an existing commit:**

```bash
git commit --amend -s          # last commit only
git rebase --signoff HEAD~N    # last N commits
git push --force-with-lease
```

---

## 2. GPG / SSH commit signing

In addition to the DCO sign-off, the `main` branch requires cryptographic
commit signing (configured via GitHub branch protection).

**Generate a GPG key (if you don't have one):**

```bash
gpg --full-generate-key      # choose RSA 4096, no expiry recommended
gpg --list-secret-keys --keyid-format LONG
```

**Add it to GitHub:**

```bash
gpg --armor --export <YOUR_KEY_ID>   # copy output → GitHub → Settings → SSH and GPG keys
```

**Configure Git to always sign:**

```bash
git config --global user.signingkey <YOUR_KEY_ID>
git config --global commit.gpgsign true
```

SSH signing is also accepted; see the
[GitHub docs on SSH commit signing](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification).

---

## 3. Development setup

```bash
# Clone the repo
git clone https://github.com/AndreiCostinescu/ConceptHierarchy.git
cd ConceptHierarchy

# Install dev dependencies and register all git hooks
python dev.py setup python

# Verify your identity is configured
git config user.name   # should be "Your Name"
git config user.email  # should be "you@example.com"
```

`python dev.py setup python` does the following (`dev.py` is a tiny bootstrap that
fetches development tools from [DevTools](https://github.com/AndreiCostinescu/DevTools) 
inside `.devtools` and delegates everything else to it — no `make` required):

1. `git submodule update --init --recursive` — fetches `.devtools` (and any
   other submodules) on a fresh clone.
2. Unsets `core.hooksPath` if previously configured (pre-commit requires this).
3. `pip install -e ".[dev]"` — installs the package in editable mode with all
   dev dependencies.
4. `pre-commit install` — registers the pre-commit framework for both hook
   stages in `.git/hooks/`.
5. Syncs `ruff.toml` from `.devtools/config/` into the repo root.

After setup, every `git commit` automatically runs:

- **ruff format** — enforces consistent formatting.
- **ruff check --fix** — auto-fixes import order and style issues; fails the
  commit if any unfixable violations remain.
- **License header check** — rejects staged Python files missing the Apache
  2.0 header.
- **DCO check** — rejects commits missing a `Signed-off-by:` trailer, saving
  you a failed CI run.

You can also run linting and formatting checks manually at any time:

```bash
python dev.py lint python      # check formatting and linting without modifying files
python dev.py format python    # auto-fix formatting and safe lint issues
```

---

## 4. Development workflow

```bash
# Create a feature branch
git checkout -b feat/my-feature

# Make changes, then commit with sign-off (-s) and GPG sign (-S, or auto via config)
git commit -s -m "feat: describe the change"

# Push and open a PR against main
git push origin feat/my-feature
```

PRs require:
- All CI tests passing (Python 3.10–3.13).
- DCO check passing (every commit signed off).
- Lint CI passing (ruff formatting and linting checks).

---

## 5. Coding standards

- **Style:** follow PEP 8.  `ruff` and `pre-commit` handle code formatting, 
  so make sure you run `python dev.py setup python` once after cloning the repository.
- **Type hints:** use them on all public functions.  Stay compatible with
  Python 3.10.
- **License headers:** every new `.py` file must begin with the Apache 2.0
  header (copy from any existing file).
- **New backends:** add a module under `src/concept_hierarchy/backends/`,
  subclass `BaseBackend`, and register the entry in
  `src/concept_hierarchy/codegen/generator.py`.

---

## 6. Starting development

All commands below should be run from the **repository root** (the directory that contains `pyproject.toml`).

1. Clone the repo and run the one-time setup:

```bash
git clone https://github.com/AndreiCostinescu/ConceptHierarchy.git
cd ConceptHierarchy
python dev.py setup python
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

---

## 7. Running tests

```bash
# Single Python version
pytest tests/ -v

# All supported versions (requires tox and the interpreters installed)
tox

# With coverage
pytest tests/ --cov=concept_hierarchy --cov-report=term-missing
```
