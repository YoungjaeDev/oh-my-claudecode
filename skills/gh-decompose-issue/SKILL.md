---
name: gh-decompose-issue
description: Break down large work items into context-completable GitHub issues
user-invocable: true
---

# Decompose Work

[DECOMPOSE MODE ACTIVATED]

Break down large work items into manageable, independent issues.

## Arguments

`$ARGUMENTS` = Work description or parent issue number

## Workflow

### 1. Check Current Issues

```bash
gh issue list --limit 20
```

### 2. Discover Components

Scan for available agents and detect test frameworks.

### 3. Check TDD Applicability

- Test framework detected + code work: Ask about TDD approach
- No test framework or non-code work: Skip TDD

If TDD selected: Add `<!-- TDD: enabled -->` marker to each issue.

### 4. Analyze and Decompose

Split into **context-completable units** - each issue completable in a single session without context switching.

### 5. Identify Dependencies

Map prerequisite relationships between tasks.

### 6. Suggest Milestone

Propose a milestone name to group related issues.

### 7. Create Issues

Ask user confirmation, then create with `gh issue create`.

## Issue Sizing Principle

### Context-Completable Units

| Good (Completable) | Bad (Fragmented) |
|--------------------|------------------|
| "Add user auth with login/logout/session" | "Add login button", "Add logout button", "Add session" (3 issues) |
| "Implement CRUD API for products" | "Add create", "Add read", "Add update", "Add delete" (4 issues) |

### Issue Content Depth

Since issues are larger, include:
1. **Implementation order** - numbered steps
2. **File-by-file changes** - specific modifications
3. **Code snippets** - key patterns to implement
4. **Edge cases** - known gotchas

## Issue Template

```markdown
**Purpose**: [Why this is needed]

## Implementation Steps

1. [ ] Step 1 - specific details
2. [ ] Step 2 - specific details
3. [ ] Step 3 - specific details

## Files to Modify

- `path/file` - what to change
- `path/file2` - what to change

## Completion Criteria

- [ ] Implementation complete
- [ ] Execution verified
- [ ] Tests pass

## Dependencies

- None or #issue-number

## Execution Strategy

**Pattern**: main-only | sequential | parallel | delegation
```

## Execution Patterns

```
main-only:     +------+
               | main |
               +------+

sequential:    +---------+     +------+
               | explore | --> | main |
               +---------+     +------+

parallel:      +---------+
               | explore |--+
               +---------+  |  +------+
               | explore |--+->| main |
               +---------+  |  +------+
               | explore |--+
               +---------+

delegation:    +------+     +------------+
               | main | --> | specialist |
               +------+     +------------+
```

## Verification

| Work Type | Verification Method |
|-----------|---------------------|
| Python | `python -m py_compile` + execution |
| TypeScript | `tsc --noEmit` or build |
| API | Endpoint call test |
| CLI | Run basic commands |
