# Branch Protection Configuration

This file documents the recommended GitHub branch protection settings for the
`main` branch.  These cannot be committed to the repo — you must apply them
manually in **Settings → Branches → Add branch protection rule** (or via the
GitHub CLI commands below).

---

## Required status checks for `main`

Enable **"Require status checks to pass before merging"** and add:

| Check name | Provided by |
|---|---|
| `test (3.7)` … `test (3.12)` | `ci.yml` |
| `DCO` | `dco.yml` |

Also enable:
- ✅ **Require branches to be up to date before merging**
- ✅ **Require signed commits** ← enforces GPG/SSH signing at the Git level
- ✅ **Do not allow bypassing the above settings** (even for admins, recommended)

---

## Apply via GitHub CLI

```bash
gh api repos/{owner}/ConceptHierarchy/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["test (3.7)","test (3.8)","test (3.9)","test (3.10)","test (3.11)","test (3.12)","DCO"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":1,"dismiss_stale_reviews":true}' \
  --field restrictions=null \
  --field required_linear_history=true \
  --field required_signatures=true
```

Replace `{owner}` with your GitHub username or organisation name.

---

## Signed commits vs DCO — what is each?

| Mechanism | What it proves | How |
|---|---|---|
| **DCO** (`Signed-off-by:`) | Contributor certifies they have the right to submit the code (legal) | `git commit -s` |
| **GPG / SSH commit signing** | The commit was actually made by the key owner (identity) | `git commit -S` (capital S) |

Both are enforced in this repo:
- DCO is checked by the `dco.yml` workflow on every PR.
- GPG/SSH signing is enforced by the GitHub branch protection rule **"Require signed commits"**.

Contributors need to set up both:
```bash
# Sign-off (DCO) — always pass -s
git commit -s -m "feat: my change"

# GPG signing — configure once, then use -S or set commit.gpgsign=true
git config --global commit.gpgsign true
git config --global user.signingkey <YOUR_KEY_ID>
```
