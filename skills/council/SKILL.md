---
name: council
description: Consult multiple AI models and synthesize collective wisdom through multi-round deliberation (LLM Council)
user-invocable: true
argument-hint: [--quick] <question>
---

# LLM Council Skill

Inspired by Andrej Karpathy's LLM Council: query multiple AI models with the same question, anonymize their responses, and synthesize collective wisdom through multi-round deliberation.

## Core Philosophy

- Collective intelligence > single expert opinion
- Anonymization prevents model favoritism
- Multi-round deliberation resolves conflicts and fills gaps
- Diverse perspectives lead to better answers

## Usage

```
/oh-my-claudecode:council <question>
/oh-my-claudecode:council --quick <question>
```

### Arguments

**Flags:**
- `--quick`: Quick mode - single round, no schema enforcement, reduced reasoning

**Default behavior (no flags):**
- Maximum reasoning depth (Codex: reasoningEffort=xhigh, model=gpt-5.1-codex-max)
- Full multi-round deliberation (up to 3 rounds)
- YAML schema enforced

**Quick mode (`--quick`):**
- All 4 models queried (Opus, Sonnet, Codex, Gemini)
- Single round only (Round 1 -> direct Synthesis, no Round 1.5 analysis)
- YAML schema not enforced (free-form responses accepted)
- Codex: reasoningEffort=high (instead of xhigh)

### Examples

```
# Standard council consultation (full multi-round, max reasoning)
/oh-my-claudecode:council What's the best way to implement caching in this API?

# Quick mode for simpler questions
/oh-my-claudecode:council --quick Should we use tabs or spaces for indentation?

# Architecture review
/oh-my-claudecode:council Review the current authentication flow and suggest improvements
```

## Pre-flight Checks

Before querying models, verify environment parity.

### CLI Availability Check

```bash
command -v claude && echo "Claude Code: OK" || echo "Claude Code: Missing"
command -v codex && echo "Codex CLI: OK" || echo "Codex CLI: Missing"
command -v gemini && echo "Gemini CLI: OK" || echo "Gemini CLI: Missing"
```

### Guidelines Files Check

```bash
[ -f ./CLAUDE.md ] && echo "CLAUDE.md: OK" || echo "CLAUDE.md: Missing"
[ -f ./AGENTS.md ] && echo "AGENTS.md: OK" || echo "AGENTS.md: Missing"
[ -f ./GEMINI.md ] && echo "GEMINI.md: OK" || echo "GEMINI.md: Missing"
```

### MCP Configuration Check

| CLI | Config Location | Check Command |
|-----|-----------------|---------------|
| Claude Code | `.mcp.json`, `~/.claude.json` | `claude mcp list` |
| Codex CLI | `~/.codex/config.toml` | `codex mcp --help` |
| Gemini CLI | `~/.gemini/settings.json`, `.gemini/settings.json` | `gemini mcp list` |

### Serena Integration Check

Check if Serena LSP is available for enhanced code understanding:

```bash
# Check for Serena MCP in configuration
grep -q "serena" .mcp.json 2>/dev/null && echo "Serena: Available" || echo "Serena: Not configured"
```

If Serena is available, use `find_symbol` tool for accurate code references.

### Warning Handling

**Warning conditions (proceed with caution):**
- Guidelines file missing: model runs without project context
- MCP not configured: model has limited tool access
- CLI not installed: model excluded from council

**If warnings detected:**
Use `AskUserQuestion` tool to confirm whether to proceed or fix issues first:

```
AskUserQuestion(
  question="Pre-flight checks detected issues:\n- Codex CLI: Missing\n- GEMINI.md: Missing\n\nProceed with available models (3/4), or fix issues first?",
  options=["Proceed with available models", "Stop and fix issues"]
)
```

## Context Gathering

Before Round 1, collect relevant context automatically.

### Auto-collect

```
- git status / git diff (current changes)
- Directory structure (tree -L 2)

Model-specific guidelines (project root):
- ./CLAUDE.md (Claude Opus/Sonnet)
- ./AGENTS.md (Codex)
- ./gemini.md (Gemini)
- .claude/guidelines/work-guidelines.md (All models - style and response guidelines)
- .claude/rules/*.md (All models - module-specific rules, if directory exists)
```

### Conditional Code Exploration

When relevant files are unclear from the question, spawn Explore agents:

**Trigger conditions:**
- Question mentions code/architecture/structure without specific files
- Question asks about "this", "the code", "current implementation" ambiguously
- UI/UX questions that need component/style file identification

**Skip exploration when:**
- User provides specific file paths or permalinks
- Question is conceptual (no code context needed)
- Files are obvious from recent git diff

**Exploration via Task tool:**

```
Task(subagent_type="oh-my-claudecode:explore", model="haiku", run_in_background=true):
  prompt: |
    Find files related to: [USER_QUESTION]

    Return results in this format:
    - /absolute/path/file.ext:LINE-LINE (brief context)

    Focus on:
    - Direct implementation files
    - Related tests
    - Configuration if relevant
```

### Serena Integration

If Serena MCP is available, use `find_symbol` for accurate symbol lookups:

```
mcp__serena__find_symbol(symbol_name="UserService")
```

This provides more accurate code references than grep-based exploration.

### File Path Inclusion Format (MANDATORY)

```
Relevant files for this question:
- /absolute/path/to/file.py:45-78 (authentication logic)
- /absolute/path/to/model.py:12-35 (User model definition)
- /absolute/path/to/screenshot.png (UI reference)

Use your file access tools to READ these files directly.
```

### Model-specific File Access

| Model | File Access Method |
|-------|-------------------|
| Claude Opus/Sonnet | Read tool (images supported) |
| Codex | sandbox read-only file access |
| Gemini | MCP tools or Bash file read (MCP supported since 2025) |

### Sensitive Data Filtering

Exclude from prompts:
- Files: `.env*`, `secrets*`, `*credentials*`, `*.pem`, `*.key`
- Patterns: `sk-[a-zA-Z0-9]+`, Bearer tokens, passwords
- Directories: `node_modules/`, `__pycache__/`, `.git/`

### Prompt Size Management

- Large files (>500 lines): include only relevant sections or diff
- Max 5 files per prompt
- Prefer git diff over full file content
- If timeout occurs: reduce context, retry

## Council Member Output Schema

All council members MUST return responses in this structured YAML format:

```yaml
council_member:
  model: "opus" | "sonnet" | "codex" | "gemini"
  response:
    summary: "1-2 sentence core answer"
    detailed_answer: "full response content"
    key_points:
      - point: "key insight"
        evidence: "file:line or reasoning"
    code_references:  # optional
      - file: "/absolute/path/to/file.py"
        lines: "42-58"
        context: "why this is relevant"
    caveats:  # optional
      - "potential limitation or edge case"
    beyond_question:  # optional, evidence-based only
      - insight: "improvement opportunity"
        evidence: "file:line or codebase reference"
        rationale: "why this is relevant to the question context"
  # Round 2+ additional fields
  gaps:
    - "aspect not fully addressed"
  conflicts:
    - "disagrees with [model] on [topic]: [reason]"
```

**Schema enforcement:**
- Sub-agents that fail to follow this schema will have their results flagged
- Missing required fields trigger re-query in next round

**Beyond the Question (Evidence-Based Only):**
Council members may suggest improvements beyond the direct question, but ONLY with:
- Specific file:line references from the codebase
- Evidence from actual code analysis
- Clear connection to the question context

Generic best practices without codebase evidence are NOT accepted.

## Progress Tracking

Use TodoWrite to show progress at each stage:

**Round 1 start:**

```
TodoWrite([
  { content: "[Council] Query Opus", status: "in_progress" },
  { content: "[Council] Query Sonnet", status: "in_progress" },
  { content: "[Council] Query Codex", status: "in_progress" },
  { content: "[Council] Query Gemini", status: "in_progress" },
  { content: "[Council] Analyze responses", status: "pending" },
  { content: "[Council] Synthesize", status: "pending" }
])
```

**Update rules:**
- Model response received -> mark that model's todo as "completed"
- All models done -> "[Council] Analyze responses" to "in_progress"
- Round 2 needed -> add re-query todos for specific models
- Analysis done -> "[Council] Synthesize" to "in_progress"

## Execution

### Round 1: Collect Initial Responses

Query all 4 models **in parallel** using Task tool with `run_in_background: true`:

**Claude Opus:**

```
Task(subagent_type="oh-my-claudecode:architect", model="opus", run_in_background=true):
  prompt: |
    You are participating in an LLM Council deliberation as Claude Opus.

    ## Guidelines
    Read and follow ./CLAUDE.md project guidelines.
    Read and follow: .claude/guidelines/work-guidelines.md for style guidelines.
    You have access to MCP tools. Use them actively to gather accurate information.

    ## Question
    [USER_QUESTION]

    ## Context Files (READ directly using exact paths)
    [FILE_LIST_WITH_LINE_NUMBERS]

    ## Current Changes
    [git diff summary]

    ## Instructions
    Provide your best answer following the Council Member Output Schema.
    Be concise but thorough. Focus on accuracy and actionable insights.

    ## Output (YAML format required)
    [COUNCIL_MEMBER_SCHEMA]
```

**Claude Sonnet:**

```
Task(subagent_type="oh-my-claudecode:architect-medium", model="sonnet", run_in_background=true):
  prompt: [Same structure as Opus]
```

**Codex (via MCP):**

```
Task(subagent_type="oh-my-claudecode:architect-medium", model="sonnet", run_in_background=true):
  prompt: |
    You are participating in an LLM Council deliberation as Codex.

    ## Tool Usage
    Use mcp__codex-cli__codex tool with:
    - sandbox: "read-only"
    - workingDirectory: "{PROJECT_ROOT}"
    - reasoningEffort: "xhigh"  (or "high" with --quick)
    - model: "gpt-5.1-codex-max"

    ## Guidelines
    Read and follow ./AGENTS.md project guidelines.
    Read and follow: .claude/guidelines/work-guidelines.md for style guidelines.
    You have access to MCP tools. Use them actively to gather accurate information.

    ## Question
    [USER_QUESTION]

    ## Context Files
    [FILE_LIST_WITH_LINE_NUMBERS]

    ## Instructions
    Parse Codex's response and return structured YAML following the schema.

    ## Output (YAML format required)
    [COUNCIL_MEMBER_SCHEMA]
```

**Gemini (via CLI):**

```
Task(subagent_type="oh-my-claudecode:architect-medium", model="sonnet", run_in_background=true):
  prompt: |
    You are participating in an LLM Council deliberation as Gemini.

    ## Tool Usage
    Use Bash tool to invoke Gemini CLI:
    ```bash
    cat <<'EOF' | gemini -p -
    [GEMINI_PROMPT_WITH_CONTEXT]
    EOF
    ```
    Note: Gemini CLI supports MCP (since 2025). If MCP is configured,
    Gemini can access project files directly via MCP tools.

    ## Guidelines
    Read and follow ./gemini.md project guidelines.
    Read and follow: .claude/guidelines/work-guidelines.md for style guidelines.
    You have access to MCP tools. Use them actively to gather accurate information.

    ## Question
    [USER_QUESTION]

    ## Context Files (READ directly using exact paths)
    [FILE_LIST_WITH_LINE_NUMBERS]

    ## Instructions
    Parse Gemini's response and return structured YAML following the schema.

    ## Output (YAML format required)
    [COUNCIL_MEMBER_SCHEMA]
```

### Model Timeouts

| Model | Timeout | Reason |
|-------|---------|--------|
| Opus/Sonnet | 300000ms (5min) | Direct execution |
| Codex | 480000ms (8min) | MCP tool + deep reasoning |
| Gemini | 600000ms (10min) | CLI invocation + long thinking |

**Important:** TaskOutput must use matching timeout: `TaskOutput(task_id, block=true, timeout=600000)`

### Round 1.5: Coordinator Analysis (MANDATORY)

**DO NOT SKIP**: After collecting responses, the coordinator MUST perform this analysis before synthesis. Skipping Round 1.5 defeats the purpose of multi-round deliberation.

**1. Anonymize Responses:**

```
1. Assign labels in response arrival order: Response A, B, C, D
2. Create internal mapping:
   label_to_model = {
     "Response A": "[first arrived]",
     "Response B": "[second arrived]",
     "Response C": "[third arrived]",
     "Response D": "[fourth arrived]"
   }
3. Present responses by label only (hide model names until synthesis)
```

**2. Gap Analysis:**

```yaml
gaps_detected:
  - model: "opus"
    gap: "performance benchmarks not addressed"
    severity: "medium"
  - model: "gemini"
    gap: "security implications missing"
    severity: "high"
```

**3. Conflict Detection:**

```yaml
conflicts_detected:
  - topic: "recommended approach"
    positions:
      - model: "opus"
        position: "use library A"
        evidence: "official docs recommend"
      - model: "codex"
        position: "use library B"
        evidence: "better performance"
    resolution_needed: true
```

**4. Convergence Check (REQUIRED before synthesis):**

```yaml
convergence_status:
  agreement_count: 3  # models with same core conclusion
  gaps_remaining: 2
  conflicts_remaining: 1
  decision: "proceed_to_round_2" | "terminate_and_synthesize"
```

**Decision logic:**
- If `agreement_count >= 3` -> `terminate_and_synthesize` (strong consensus)
- If `gaps_remaining == 0` AND `conflicts_remaining == 0` -> `terminate_and_synthesize`
- If `conflicts_remaining > 0` AND round < 3 -> `proceed_to_round_2`
- If `gaps_remaining > 0` AND round < 3 -> `proceed_to_round_2`
- Otherwise -> `terminate_and_synthesize`

### Round 2: Targeted Re-queries (Conditional)

If convergence criteria not met, re-query only models with gaps/conflicts:

**Re-query prompt template:**

```
## Previous Round Summary
Round 1 produced the following positions:

### Response A
- Position: [summary]
- Key points: [list]

### Response B
- Position: [summary]
- Key points: [list]

[... other responses ...]

## Gaps Identified
- [gap 1]
- [gap 2]

## Conflicts Detected
- Topic: [topic]
  - Position A: [description]
  - Position B: [description]

## Re-query Focus
Please address specifically:
1. [specific gap or conflict to resolve]
2. [specific gap or conflict to resolve]

Provide evidence and reasoning for your position.

## Output (YAML format required)
[COUNCIL_MEMBER_SCHEMA with gaps/conflicts fields]
```

### Round 2.5: Coordinator Analysis

Same as Round 1.5. Check convergence again.

### Round 3: Final Cross-Validation (Conditional)

If still not converged after Round 2:
- Focused on resolving remaining conflicts
- Models see other models' positions (still anonymized)
- Final opportunity for consensus

### Synthesis

After convergence or max rounds:

1. **Reveal** the label-to-model mapping
2. **Analyze** all responses:
   - Consensus points (where models agree)
   - Resolved conflicts (with reasoning)
   - Remaining disagreements (with analysis)
   - Unique insights (valuable points from individual models)
3. **Produce** final verdict combining best elements

## Termination Criteria

### Hard Limits (Mandatory Termination)

| Condition | Value |
|-----------|-------|
| Max rounds | 3 |
| Max total time | 20 min |
| Max per-model timeout | 10 min (Gemini) |
| Min successful models | 2/4 (proceed with partial results) |
| All models failed | immediate termination |

### Soft Limits (Convergence - any triggers termination)

| Condition | Threshold |
|-----------|-----------|
| Strong consensus | 3+ models agree on core conclusion |
| All gaps resolved | 0 remaining |
| All conflicts resolved | 0 remaining |
| Conflicts irreconcilable | Cannot be resolved with more queries |

## Output Format

```markdown
## LLM Council Deliberation

### Question
[Original user question]

### Deliberation Process
| Round | Models Queried | Convergence | Status |
|-------|---------------|-------------|--------|
| 1 | All (4) | 65% | Gaps detected |
| 2 | Codex, Gemini | 85% | Conflict on approach |
| 3 | Codex | 95% | Converged |

### Individual Responses (Anonymized)

#### Response A
[Content]

**Key Points:**
- [point 1] (evidence: file:line)
- [point 2] (evidence: file:line)

#### Response B
[Content]

#### Response C
[Content]

#### Response D
[Content]

### Model Reveal
| Label | Model |
|-------|-------|
| Response A | codex |
| Response B | opus |
| Response C | sonnet |
| Response D | gemini |

### Coordinator Analysis

#### Gaps Addressed
| Gap | Resolved By | Round |
|-----|-------------|-------|
| Performance benchmarks | Codex | 2 |
| Security considerations | Opus | 1 |

#### Conflicts Resolved
| Topic | Final Position | Reasoning |
|-------|---------------|-----------|
| Library choice | Library A | Official docs + 3 model consensus |

#### Remaining Disagreements
| Topic | Positions | Analysis |
|-------|-----------|----------|
| [topic] | A: [pos], B: [pos] | [why unresolved] |

### Council Synthesis

#### Consensus
[Points where all/most models agree - with evidence]

#### Key Insights by Model
| Model | Unique Contribution |
|-------|-------------------|
| Codex | [insight] |
| Opus | [insight] |

### Final Verdict
[Synthesized answer combining collective wisdom with confidence level and caveats]

### Code References
| File | Lines | Context |
|------|-------|---------|
| /path/to/file.py | 45-78 | Authentication logic |
```

## Error Handling

| Error | Response |
|-------|----------|
| Model timeout | Continue with successful responses, note failures |
| All models fail | Report error, suggest retry |
| Parse failure | Use fallback extraction, flag for re-query |
| Empty response | Exclude from synthesis, note in output |
| Schema violation | Flag and request re-query in next round |

## User Interaction

Use `AskUserQuestion` tool when clarification is needed:

**Before Round 1:**
- Question is ambiguous or too broad
- Missing critical context (e.g., "review this code" but no file specified)
- Multiple interpretations possible

**During Deliberation:**
- Strong disagreement between models that cannot be resolved
- New information discovered that changes the question scope

**After Synthesis:**
- Remaining disagreements require user input to decide
- Actionable next steps require user confirmation

**Example questions:**

```
AskUserQuestion(
  question="Your question mentions 'the API' - which specific endpoint or service?",
  options=["Auth API (/api/auth/*)", "User API (/api/users/*)", "Both", "Let me specify"]
)

AskUserQuestion(
  question="Models disagree on X vs Y approach. Which aligns better with your constraints?",
  options=["X - better performance", "Y - better maintainability", "Need more analysis"]
)
```

**Important:** Never assume or guess when context is unclear. Ask first, then proceed.

## Guidelines

- Respond in the same language as the user's question
- No emojis in code or documentation
- If context is needed, gather it before querying models
- For code-related questions, include relevant file snippets with line numbers
- Respect `@CLAUDE.md` project conventions
- **Never assume unclear context - use AskUserQuestion to clarify**
