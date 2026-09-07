---
id: codex-plugin-surfaces
aliases: [codex-hook-trust, codex-skills-only, codex-plugin-install]
last_verified: 2026-09-08
status: active
volatility: volatile
sources: 4
---

# What Codex actually registers from an installed plugin

`codex plugin add <name>@my-claude-plugins` registers the plugin's `skills/` and nothing else.

- `commands/` and `agents/` in `plugin.json` are ignored. Skill logic moved into an agent definition disappears on Codex without an error.
- Bundled hooks are copied into the cache but never run until the user registers them in `~/.codex/hooks.json` and approves them in `/hooks`. Until then there is no signal at all; `codex exec` (headless) can never grant trust, so plugin hooks never fire in CI.
- `/hooks` approval is recorded in `~/.codex/config.toml`, not in `hooks.json`: an entry `[hooks.state."<source path>:<event>:<i>:<j>"]` carries a `trusted_hash` (sha256). For example:

  ```toml
  [hooks.state."/path/to/hook.sh:UserPromptSubmit:0:0"]
  trusted_hash = "<sha256>"
  ```

  A missing `config.toml` entry skips the hook silently: no error, no warning.
- A `hook: UserPromptSubmit` (or other event) line in `codex exec` output does not identify which hook source ran. The line can come from an already-trusted, unrelated hook (for example a pre-existing `~/.codex/hooks.json` entry), not the plugin's own hook. The 2026-07-27 investigation made exactly this misjudgment once: confirm by command path, not by event name.
- `trusted_hash` is presumably a content hash, so editing a hook script after approval probably invalidates trust and forces re-approval via `/hooks`. **Unverified**: not confirmed in the 2026-07-27 session.
- A per-turn convention (for example "read the wiki MOC before answering lore questions") therefore has only two carriers on Codex: the manually registered hook, or the repo's own `AGENTS.md`. A skill cannot carry it because a skill loads only on the turn it is selected.
- Skill `description` over 1024 characters is skipped silently; `: ` inside an unquoted description collapses the YAML on both runtimes. `scripts/check-skill-contract.mjs` guards both.
- The 2.30.0 rename moved the cache paths: hook entries pointing at `core-config/<ver>/hooks/...` or `llm-wiki/<ver>/hooks/...` must be re-pointed at `core/1.0.0/...` and `wiki/1.0.0/...` and re-trusted.

## Sources

- GitHub issue #169 (measured on codex-cli 0.145.0, 2026-07-27)
- PR #213 Codex review thread on `plugins/llm-wiki/CLAUDE.md` (2026-09-04)
- `plugins/wiki/CLAUDE.md` "Codex hooks (descriptor shipped, manual wiring)" (amended 2026-09-08 with the `config.toml` trust schema)
- `plugins/core/CLAUDE.md` "Codex CLI parity (`prompt_inject.sh codex`)" (amended 2026-09-08 with the same schema)

> Evidence: https://github.com/YoungjaeDev/my-claude-plugins/issues/169
> See-also: [[bundle-rename-is-a-new-entry]]
