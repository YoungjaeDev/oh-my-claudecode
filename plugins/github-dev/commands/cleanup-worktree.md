---
description: Clean up git worktrees with optional branch and remote deletion
---

# Cleanup Worktree

Remove git worktrees and their associated branches. Supports selective, targeted, and bulk cleanup. Follow project guidelines in `@CLAUDE.md`.

## Usage

```bash
/github-dev:cleanup-worktree           # List worktrees, select which to remove
/github-dev:cleanup-worktree <branch>  # Remove worktree for specific branch
/github-dev:cleanup-worktree --all     # Remove all non-main worktrees (with confirmation)
/github-dev:cleanup-worktree --prune   # Clean dangling references only
```

## Flags

| Flag | Description |
|------|-------------|
| `--all` | Remove all non-main worktrees (prompts for confirmation) |
| `--prune` | Only run `git worktree prune` to clean dangling references |
| `--delete-remote` | Also delete the remote tracking branch |
| `--force` | Force removal even if worktree has uncommitted changes |

## Workflow

1. **List Current Worktrees**
   ```bash
   git worktree list
   ```
   - Parse output to identify all worktrees and their branches
   - Identify the **main worktree** (first entry, the primary checkout) — this is never removed
   - If no non-main worktrees exist, inform user and exit

2. **Handle `--prune` Flag**
   - If `--prune` passed, run `git worktree prune` and exit
   - Report number of dangling references cleaned

3. **Select Target Worktrees**

   **If `<branch>` argument provided:**
   - Find worktree matching the branch name
   - If not found, show available worktrees and exit with error

   **If `--all` flag:**
   - Select all non-main worktrees
   - Display list and prompt user for confirmation using `AskUserQuestion`

   **If no arguments (interactive):**
   - Display numbered list of non-main worktrees with branch names and paths
   - Use `AskUserQuestion` to let user select which worktree(s) to remove

4. **Check Dirty State (per worktree)**
   ```bash
   git -C "$WORKTREE_PATH" status --porcelain
   ```
   - If clean: proceed to removal
   - If dirty (uncommitted changes):
     - Warn user: "Worktree at `$WORKTREE_PATH` has uncommitted changes"
     - Prompt with `AskUserQuestion`:
       - **Force remove**: proceed with `--force`
       - **Skip**: skip this worktree
       - **Abort**: stop entire cleanup

5. **Remove Worktree**
   ```bash
   # Normal removal
   git worktree remove "$WORKTREE_PATH"

   # Force removal (if dirty and user confirmed)
   git worktree remove --force "$WORKTREE_PATH"
   ```

6. **Delete Local Branch**
   ```bash
   # Safe delete (fails if unmerged)
   git branch -d "$BRANCH_NAME"
   ```
   - If fails (unmerged branch), prompt user:
     - **Force delete**: `git branch -D "$BRANCH_NAME"`
     - **Keep branch**: skip branch deletion

7. **Delete Remote Branch (if `--delete-remote`)**
   ```bash
   git push origin --delete "$BRANCH_NAME"
   ```
   - Skip silently if remote branch does not exist

8. **Prune Dangling References**
   ```bash
   git worktree prune
   ```

9. **Output Summary**
   ```
   Worktree cleanup complete!

   Removed:
     - fix/42-login-bug (../fix-42-login-bug)

   Branches deleted (local): fix/42-login-bug
   Branches deleted (remote): origin/fix/42-login-bug   # if --delete-remote

   Remaining worktrees:
     /home/user/project (main) [main worktree]
   ```

## Notes

- The main worktree (primary checkout) is **never** removed
- Use `--prune` periodically to clean up references from manually deleted worktree directories
- After PR merge, prefer `/github-dev:post-merge` for branch cleanup; use this command for worktree-specific removal
- This command pairs with `/github-dev:create-worktree` for the full worktree lifecycle
