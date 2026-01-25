---
name: ask-codex
description: Request Codex review with Claude cross-check (dual-validation consensus)
user-invocable: true
---

# Codex Review (OMC)

Request code review from Codex MCP, cross-checked by Claude for consensus.

**Core Principle:** Codex has limited context/tools, so Claude validates all feedback. Disagreements trigger re-queries (max 3 rounds) until consensus.

---

## Arguments

`$ARGUMENTS` parsing:

| Input | Mode | Action |
|-------|------|--------|
| (none) | General | AskUserQuestion for review type |
| `'text in quotes'` | Directed | Review in specified direction |
| `path/to/file` | File-focused | Review that file + dependencies |

**Detection:** If argument is an existing file/directory path, use File-focused mode. Otherwise, treat as Directed review text.

---

## Context Gathering

### Auto-collect

| Item | Source | Priority |
|------|--------|----------|
| Files from conversation | Read/Edit/Write history | Required |
| Git changes | `git diff`, `git status` | Required |
| CLAUDE.md | Project root | Required |
| Directory structure | `tree -L 2 -I 'node_modules\|__pycache__\|.git'` | Recommended |
| Recent commits | `git log --oneline -10` | Optional |

### Exclude (sensitive data)

**Files:** `.env*`, `secrets*`, `*credentials*`, `*token*`, `*.pem`, `*.key`
**Patterns:** `sk-[a-zA-Z0-9]+`, `AKIA[A-Z0-9]+`, `Bearer [...]`, `password[:=]...`
**Dirs:** `node_modules/`, `__pycache__/`, `.git/`, binaries, media files

### Size Management

- Prefer `git diff` over full file content
- Large files (>500 lines): include only relevant sections (50 lines around changes)
- Max 5 files per prompt
- On timeout: reduce CLAUDE.md to key sections, use `tree -L 1`

---

## Execution

### 1. Detect MCP Server

Try both common Codex MCP server names:

1. `mcp__codex-cli__codex` (primary)
2. `mcp__codex__codex` (fallback)

Use Task tool to invoke:

```typescript
Task(
  tool_name="mcp__codex-cli__codex",  // or mcp__codex__codex
  arguments={
    prompt: [REVIEW_PROMPT],
    sandbox: "read-only",
    workingDirectory: [PROJECT_ROOT]
  }
)
```

**Prompt template:**

```
## Role
You are a code review expert reviewing work in progress.

## Review Type
[Implementation direction / Code quality / Architecture]

## Project Context
### CLAUDE.md
[Project guidelines]

### Work Guidelines
Read and follow: .claude/guidelines/work-guidelines.md
If .claude/rules/ exists, also read relevant rule files for module-specific guidance.
(Use your file access to read these files directly)

### Directory Structure
[tree output]

## Review Target
### Current Work Summary
[Description of ongoing work]

### Changes
[git diff or file list]

### File Contents
[Key files with line numbers]

## Output Format (required)

### Strengths
- [item]: [description] (file:line)

### Suggestions
| Item | Location | Issue | Solution |
|------|----------|-------|----------|

### Risks
- [severity]: [description] (file:line)

### Questions
- [question]

### Summary
[1-2 sentence conclusion]

### Beyond the Question (Evidence-Based)
If you identify improvements beyond the direct question:
- Alternative approaches FOUND IN this codebase (with file:line)
- Architectural patterns ALREADY USED that could apply
- Potential optimizations BASED ON actual code analysis

DO NOT suggest generic best practices without codebase evidence.
```

### 2. Parse Response

Extract sections: Strengths, Suggestions, Risks, Questions, Summary

### 3. Claude Cross-check

Validate Codex feedback against:
- Project context (did Codex understand the codebase?)
- CLAUDE.md compliance
- Technical accuracy (is the suggestion implementable?)
- Existing patterns (does it match current codebase style?)
- Already-resolved issues (is Codex pointing out something already fixed?)

Identify any incorrect claims with evidence.

### 4. Resolve Disagreements

If discrepancies found, invoke Codex again with context:

```
## Previous Review Summary
[Codex 1st response key points]

## Claude Cross-check Results
[Discrepancies with evidence]

## Re-review Request
Reconsider only these items:
1. [item 1]
2. [item 2]

Provide corrections in the same output format.
```

**Exit conditions:**
- Consensus reached
- 3 rounds completed
- Codex accepts Claude's evidence

### 5. Final Output (YAML Schema)

Output structured response:

```yaml
codex_review:
  metadata:
    review_type: "string"  # Implementation / Code Quality / Architecture
    rounds: number  # 1-3
    status: "string"  # Full consensus / Partial / Claude judgment
    timestamp: "ISO8601"

  valid_feedback:
    - item: "string"
      description: "string"
      location: "file:line"
      source: "Codex | Claude | Both"

  suggestions:
    - item: "string"
      issue: "string"
      solution: "string"
      location: "file:line"
      source: "Codex | Claude | Both"

  risks:
    - severity: "High | Medium | Low"
      description: "string"
      location: "file:line"

  corrections:
    - codex_claim: "string"
      actual: "string"
      evidence: "string"

  action_items:
    - priority: "High | Medium | Low"
      description: "string"
      rationale: "string"

  summary: "string"
```

**Markdown presentation:**

```markdown
## Codex Review Result (Codex + Claude Consensus)

### Process
- Review type: [type]
- Rounds: [N]
- Status: [Full consensus / Partial / Claude judgment]

### Valid Feedback
| Item | Description | Location | Source |
|------|-------------|----------|--------|

### Suggestions
| Item | Issue | Solution | Source |
|------|-------|----------|--------|

### Risks
| Severity | Description | Location |
|----------|-------------|----------|

### Corrections (Codex errors)
| Codex Claim | Actual | Evidence |
|-------------|--------|----------|

### Action Items
[Recommended next steps - use AskUserQuestion if choices needed]

### Summary
[Final conclusion]
```

---

## Error Handling

| Error | Response |
|-------|----------|
| No context | "No reviewable content found. Specify review direction." |
| Codex MCP failure | "Codex MCP invocation failed. Check MCP server status with: ls ~/.config/Claude/claude_desktop_config.json" |
| Timeout | "Response timeout. Reducing prompt size and retrying." |
| Both MCP names fail | "Codex MCP server not found. Expected mcp__codex-cli__codex or mcp__codex__codex. Check MCP configuration." |

---

## Guidelines

- Respond in user's language (한국어 사용자는 한국어로 응답)
- No emojis in code or documentation
- **Never assume unclear context - use AskUserQuestion**
- Code modifications require user confirmation
- Follow `@CLAUDE.md` project conventions
- Use Task tool for MCP invocation (NOT Bash)

---

## Example Usage

```bash
# General review (interactive)
/oh-my-claudecode:ask-codex

# Directed review (지정된 방향)
/oh-my-claudecode:ask-codex 'Review error handling approach'

# File-focused review (파일 중심)
/oh-my-claudecode:ask-codex src/components/auth/login.tsx

# Architecture review (아키텍처 검토)
/oh-my-claudecode:ask-codex 'Evaluate system architecture for scalability'
```

---

## Implementation Notes

### Context Collection Strategy

```typescript
// 1. Argument parsing
const args = parseArguments($ARGUMENTS);
const mode = detectMode(args);  // General | Directed | File-focused

// 2. Gather context
const context = {
  files: getConversationFiles(),
  gitDiff: await bash("git diff"),
  gitStatus: await bash("git status"),
  claudeMd: await read("CLAUDE.md"),
  tree: await bash("tree -L 2 -I 'node_modules|__pycache__|.git'"),
  recentCommits: await bash("git log --oneline -10")
};

// 3. Filter sensitive data
context = filterSensitiveData(context);

// 4. Size management
if (context.size > MAX_SIZE) {
  context = reduceContext(context);
}
```

### Consensus Loop

```typescript
let round = 1;
let consensus = false;
let codexResponse;

while (round <= 3 && !consensus) {
  // Invoke Codex
  codexResponse = await Task({
    tool_name: detectCodexMCP(),  // Try both names
    arguments: { prompt: buildPrompt(context, round) }
  });

  // Claude cross-check
  const validation = await validateResponse(codexResponse, context);

  if (validation.discrepancies.length === 0) {
    consensus = true;
  } else if (round < 3) {
    // Prepare re-review prompt
    context.previousRound = {
      response: codexResponse,
      issues: validation.discrepancies
    };
  }

  round++;
}

// Output final result
outputConsensusResult(codexResponse, validation, round);
```

### MCP Detection

```typescript
async function detectCodexMCP() {
  const servers = ["mcp__codex-cli__codex", "mcp__codex__codex"];

  for (const server of servers) {
    try {
      // Test invocation
      await Task({ tool_name: server, arguments: { prompt: "test" } });
      return server;
    } catch (e) {
      continue;
    }
  }

  throw new Error("Codex MCP server not found");
}
```

---

## Korean Comments (한국어 주석)

```typescript
// 주요 실행 흐름:
// 1. 인자 파싱: $ARGUMENTS로부터 모드 결정
// 2. 컨텍스트 수집: git diff, CLAUDE.md, 디렉토리 구조 등
// 3. Codex 호출: Task 도구로 MCP 서버 호출
// 4. 크로스체크: Claude가 Codex 응답 검증
// 5. 합의 도출: 불일치 발견 시 최대 3라운드 반복
// 6. 결과 출력: YAML + Markdown 형식으로 구조화된 결과 제공
```
