# Contributing to ConceptHierarchy

Thank you for considering a contribution!  This document explains the two
legal/process requirements every contributor must satisfy, plus the day-to-day
development workflow.

---

## Table of contents

1. [Developer Certificate of Origin (DCO)](#2-developer-certificate-of-origin-dco)
2. [GPG / SSH commit signing](#3-gpg--ssh-commit-signing)
3. [One-time local setup](#4-one-time-local-setup)
4. [Development workflow](#5-development-workflow)
5. [Coding standards](#6-coding-standards)
6. [Running tests](#7-running-tests)

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

## 3. One-time local setup

```bash
# Clone the repo
git clone https://github.com/your-org/ConceptHierarchy.git
cd ConceptHierarchy

# Install the project with dev dependencies
pip install -e ".[dev]"

# Install the local commit-msg hook (enforces DCO before push)
git config core.hooksPath .githooks

# Verify your identity is configured
git config user.name   # should be "Your Name"
git config user.email  # should be "you@example.com"
```

The `commit-msg` hook rejects commits locally before they even reach GitHub,
saving you a failed CI run.

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
- All CI tests passing (Python 3.7–3.12).
- DCO check passing (every commit signed off).
- CLA signed (once per contributor).
- At least one approving review.

---

## 5. Coding standards

- **Style:** follow PEP 8.  `black` and `isort` are recommended but not
  enforced by CI yet.
- **Type hints:** use them on all public functions.  Stay compatible with
  Python 3.7 (`from __future__ import annotations` is already imported
  everywhere).
- **License headers:** every new `.py` file must begin with the Apache 2.0
  header (copy from any existing file).
- **New backends:** add a module under `src/concept_hierarchy/backends/`,
  subclass `BaseBackend`, and register the entry in
  `src/concept_hierarchy/codegen/generator.py`.

---

## 6. Running tests

```bash
# Single Python version
pytest tests/ -v

# All supported versions (requires tox and the interpreters installed)
tox

# With coverage
pytest tests/ --cov=concept_hierarchy --cov-report=term-missing
```
