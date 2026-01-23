---
name: gh-resolve-issue
description: Resolve GitHub issues with branch creation, implementation, and PR
user-invocable: true
---

# GitHub Issue Resolution

[ISSUE RESOLUTION MODE ACTIVATED]

Receive a GitHub issue number and resolve it end-to-end.

## Arguments

`$ARGUMENTS` = Issue number (e.g., `42`)

## Workflow

### 1. Analyze Issue

```bash
gh issue view $ISSUE_NUMBER --json title,body,comments,milestone,labels
```

Check for TDD marker: `<!-- TDD: enabled -->` in issue body.

If milestone exists, view related issues for context:
```bash
gh issue list --milestone "<milestone-name>" --json number,title,state
```

### 2. Verify Plan File (If Exists)

Look for plan file references in issue body:
- `Plan: /path/to/plan.md`
- `See: .omc/plans/xxx.md`

If found, read and verify scope alignment before proceeding.

### 3. Create Branch

From `main` or `master`:

```bash
git checkout main && git pull
git checkout -b {type}/{issue-number}-{short-description}
```

Branch naming:
- `type`: `fix` (bug), `feat` (feature/enhancement), `refactor`, `docs`
- `short-description`: slugified title (lowercase, hyphens, max 50 chars)

Examples: `fix/42-login-validation`, `feat/15-add-dark-mode`

### 4. Analyze Codebase

**Narrow scope** (1-2 files): Use symbolic tools directly
**Broad scope** (multiple modules): Spawn Explorer agents in parallel

### 5. Implement Solution

**If TDD enabled:**
1. RED: Write failing tests first
2. GREEN: Implement minimal code to pass
3. REFACTOR: Clean up while keeping tests green

**If TDD not enabled:**
Implement directly according to the plan.

Always execute and verify runnable code.

### 6. Write Tests

Achieve at least 80% coverage for new code.

### 7. Validate

Run tests, lint, and build verification.

### 8. Create PR

```bash
git add <specific-files>  # Never use git add -A
git commit -m "feat: resolve #$ISSUE_NUMBER - description"
git push -u origin HEAD
gh pr create --title "..." --body "Closes #$ISSUE_NUMBER"
```

### 9. Update Issue

Mark completed checkbox items in the issue body.

## Verification Principles

- Execute code to confirm it works (not just "expected to work")
- Provide evidence (actual output/results)
- Never present assumptions as facts
- Distinguish partial completion from full completion
