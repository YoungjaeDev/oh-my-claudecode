---
name: gh-post-merge
description: Clean up branch and update CLAUDE.md after PR merge
user-invocable: true
---

# Post-Merge Cleanup

[POST-MERGE MODE ACTIVATED]

Perform branch cleanup and configuration updates after a PR has been merged.

## Arguments

`$ARGUMENTS` = PR number (optional, infers from context if not provided)

## Workflow

### 1. Identify PR

```bash
# If no argument, show recent merged PRs
gh pr list --state merged --limit 5

# Get PR details
gh pr view <PR_NUMBER> --json number,title,baseRefName,headRefName,body,state
```

Verify `state` is MERGED.

### 2. Check Local Changes

```bash
git status --porcelain
```

- **Untracked files (`??`)**: Ignore, proceed
- **Modified/Staged files**: Prompt for action (stash/discard/abort)

### 3. Switch to Base Branch

```bash
git fetch origin
git checkout <baseRefName>
git pull origin <baseRefName>
```

### 4. Clean Up Branch (Optional)

Prompt user before deleting:
```bash
git branch -d <headRefName>
```

### 5. Update Configuration Files

Check which files exist:
- `CLAUDE.md` - Claude Code instructions
- `AGENTS.md` - Cross-tool AI agent instructions
- `GEMINI.md` - Gemini CLI instructions
- `.claude/rules/*.md` - Modular rule files

### Content Placement Rules

| Content Type | Placement |
|--------------|-----------|
| Project-wide constraints | Golden Rules > Immutable |
| Project-wide recommendations | Golden Rules > Do's |
| Project-wide prohibitions | Golden Rules > Don'ts |
| Module-specific rules | `.claude/rules/[module].md` |
| New commands | Commands section |

### Content to Remove

- Temporary notes (`TODO: remove after #N`)
- Resolved known issues
- Workaround descriptions for fixed bugs

### Content to Add

- Code conventions discovered during resolution
- Guidelines to prevent common mistakes
- New patterns or architecture decisions

### 6. Commit Changes (Optional)

If configuration files were modified:
```bash
git add CLAUDE.md AGENTS.md GEMINI.md 2>/dev/null || true
git commit -m "docs: update configuration after PR #N merge"
```

## Configuration File Structure

**Root Config (CLAUDE.md, AGENTS.md, GEMINI.md)**:
1. Project Context - Business goal + tech stack
2. Commands - Package manager and run commands
3. Golden Rules - Immutable / Do's / Don'ts
4. Modular Rules - `See @.claude/rules/[module].md`
5. Project-Specific - Data locations, tracking

**Modular Rules (.claude/rules/*.md)**:
```markdown
---
paths: src/[module]/**/*.py
---
# [Module] Rules
Role description
## Key Components
## Do's
## Don'ts
```
