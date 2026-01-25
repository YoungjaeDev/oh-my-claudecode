---
name: ask-gemini
description: Request Gemini review with Claude cross-check for consensus validation
user-invocable: true
---

# Gemini Review Skill

Request code review from Gemini CLI, cross-checked by Claude for consensus.

**Core Principle:** Gemini has limited context/tools, so Claude validates all feedback. Disagreements trigger re-queries (max 3 rounds) until consensus.

---

## Arguments

Parse `$ARGUMENTS`:

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

## Execution Protocol

### 1. Invoke Gemini CLI

Use heredoc to avoid quote/escape issues:

```bash
cat <<'EOF' | gemini -p -
[REVIEW_PROMPT]
EOF
```

**Bash tool parameters:**
- `timeout`: 300000 (5 minutes)
- `description`: "Gemini code review request"

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

## Output Format (YAML structured response)

```yaml
review:
  strengths:
    - item: "[strength name]"
      description: "[detailed description]"
      location: "[file:line]"

  suggestions:
    - item: "[suggestion name]"
      location: "[file:line]"
      issue: "[what's wrong]"
      solution: "[how to fix]"

  risks:
    - severity: "[high|medium|low]"
      description: "[risk description]"
      location: "[file:line]"

  questions:
    - "[question text]"

  summary: "[1-2 sentence conclusion]"

  beyond_the_question:
    # Evidence-based improvements only
    - type: "[alternative_approach|pattern|optimization]"
      description: "[what you found]"
      evidence: "[file:line reference from THIS codebase]"
```

**Important:**
- DO NOT suggest generic best practices without codebase evidence
- All "beyond_the_question" items MUST reference actual files in this project
- Include file:line references for all specific feedback
```

### 2. Parse Response

Extract YAML sections or fallback to markdown parsing:
- Strengths
- Suggestions
- Risks
- Questions
- Summary
- Beyond the Question

### 3. Claude Cross-check

Validate Gemini feedback against:

| Validation Area | Check |
|----------------|-------|
| Project context | Did Gemini understand the codebase? |
| CLAUDE.md compliance | Does suggestion align with project guidelines? |
| Technical accuracy | Is the suggestion implementable? |
| Existing patterns | Does it match current codebase style? |
| Already-resolved issues | Is Gemini pointing out something already fixed? |

**Output cross-check results:**
```yaml
cross_check:
  consensus_items:
    - item: "[what both agree on]"
      confidence: "[high|medium]"

  discrepancies:
    - gemini_claim: "[what Gemini said]"
      claude_evidence: "[what Claude found]"
      file_reference: "[file:line]"
      severity: "[high|medium|low]"
```

### 4. Resolve Disagreements

If discrepancies found (max 3 rounds):

**Re-query prompt:**
```bash
cat <<'EOF' | gemini -p -
## Previous Review Summary
[Gemini's key points from round N]

## Claude Cross-check Results
[Discrepancies with evidence in YAML format]

```yaml
discrepancies:
  - item: "[discrepancy 1]"
    gemini_said: "[original claim]"
    claude_found: "[counter-evidence]"
    evidence: "[file:line]"
```

## Re-review Request
Reconsider only these items. Provide corrections in the same YAML format.

If you maintain your position, explain why Claude's evidence doesn't apply.
If you accept Claude's evidence, update your recommendation.
EOF
```

**Exit conditions:**
- Consensus reached ✓
- 3 rounds completed (Claude judgment prevails)
- Gemini accepts Claude's evidence ✓

### 5. Final Output

```yaml
gemini_review_result:
  process:
    review_type: "[type]"
    rounds: N
    status: "[full_consensus|partial_consensus|claude_judgment]"

  valid_feedback:
    - item: "[feedback item]"
      description: "[details]"
      location: "[file:line]"
      source: "[gemini|claude|consensus]"

  suggestions:
    - item: "[suggestion]"
      issue: "[problem]"
      solution: "[fix]"
      priority: "[high|medium|low]"
      source: "[gemini|claude|consensus]"

  risks:
    - severity: "[high|medium|low]"
      description: "[risk details]"
      location: "[file:line]"

  corrections:
    # Where Gemini was wrong
    - gemini_claim: "[original claim]"
      actual: "[correct information]"
      evidence: "[file:line]"

  action_items:
    - action: "[recommended next step]"
      priority: "[high|medium|low]"
      requires_user_input: true/false

  summary: "[final conclusion]"
```

Present to user in formatted markdown tables for readability.

---

## Error Handling

| Error | Response | Action |
|-------|----------|--------|
| No context | "No reviewable content found" | Use AskUserQuestion for clarification |
| CLI not installed | "Gemini CLI not installed" | Show install command: `npm install -g @google/gemini-cli && gemini auth` |
| CLI failure | "Gemini CLI failed" | Check auth status with `gemini auth` |
| Timeout | "Response timeout" | Reduce prompt size, retry with minimal context |
| Parse failure | "Cannot parse Gemini response" | Fallback to markdown parsing |

---

## Implementation Guidelines

```typescript
// 인수 파싱 (Argument parsing)
function parseArguments(args: string): ReviewMode {
  if (!args.trim()) return { mode: 'general' };

  // 파일 경로 확인 (Check if file path)
  if (fs.existsSync(args.trim())) {
    return { mode: 'file-focused', path: args.trim() };
  }

  // 따옴표로 묶인 텍스트 (Quoted text)
  return { mode: 'directed', query: args.trim() };
}

// 컨텍스트 수집 (Collect context)
async function gatherContext(): Promise<ReviewContext> {
  // 병렬로 정보 수집 (Collect in parallel)
  const [gitDiff, gitStatus, claudeMd, tree] = await Promise.all([
    execBash('git diff'),
    execBash('git status'),
    readFile('CLAUDE.md'),
    execBash('tree -L 2 -I "node_modules|__pycache__|.git"')
  ]);

  return { gitDiff, gitStatus, claudeMd, tree };
}

// 민감 정보 필터링 (Filter sensitive data)
function filterSensitive(content: string): string {
  const patterns = [
    /sk-[a-zA-Z0-9]+/g,
    /AKIA[A-Z0-9]+/g,
    /Bearer [^\s]+/g,
    /password[:=][^\s]+/gi
  ];

  let filtered = content;
  patterns.forEach(pattern => {
    filtered = filtered.replace(pattern, '[REDACTED]');
  });

  return filtered;
}

// Gemini 호출 (Invoke Gemini)
async function invokeGemini(prompt: string): Promise<string> {
  return await execBash({
    command: `cat <<'EOF' | gemini -p -\n${prompt}\nEOF`,
    timeout: 300000,
    description: "Gemini code review request"
  });
}

// YAML 파싱 (Parse YAML response)
function parseYamlResponse(response: string): ReviewData {
  try {
    const yamlMatch = response.match(/```yaml\n([\s\S]+?)\n```/);
    if (yamlMatch) {
      return yaml.parse(yamlMatch[1]);
    }
  } catch (e) {
    // Fallback to markdown parsing
  }
  return parseMarkdownResponse(response);
}

// 교차 검증 (Cross-check)
async function crossCheck(geminiReview: ReviewData): Promise<CrossCheckResult> {
  // Claude가 코드베이스 실제 상태와 비교
  // (Claude compares against actual codebase state)
  const discrepancies = [];

  for (const suggestion of geminiReview.suggestions) {
    const isValid = await validateSuggestion(suggestion);
    if (!isValid) {
      discrepancies.push({
        gemini_claim: suggestion.item,
        claude_evidence: await getCounterEvidence(suggestion),
        severity: 'high'
      });
    }
  }

  return { discrepancies, consensus_items: [...] };
}

// 합의 루프 (Consensus loop - max 3 rounds)
async function achieveConsensus(
  initialReview: ReviewData,
  maxRounds: number = 3
): Promise<FinalResult> {
  let currentReview = initialReview;
  let round = 1;

  while (round <= maxRounds) {
    const crossCheckResult = await crossCheck(currentReview);

    if (crossCheckResult.discrepancies.length === 0) {
      return { status: 'full_consensus', review: currentReview, rounds: round };
    }

    if (round === maxRounds) {
      return { status: 'claude_judgment', review: currentReview, rounds: round };
    }

    // 재질의 (Re-query)
    const reQueryPrompt = buildReQueryPrompt(currentReview, crossCheckResult);
    const geminiResponse = await invokeGemini(reQueryPrompt);
    currentReview = parseYamlResponse(geminiResponse);

    round++;
  }
}
```

---

## Examples

```bash
# General review (전체 리뷰)
/oh-my-claudecode:ask-gemini

# Directed review (방향성 리뷰)
/oh-my-claudecode:ask-gemini 'Review error handling approach'

# File-focused review (파일 중심 리뷰)
/oh-my-claudecode:ask-gemini src/components/auth/login.tsx

# Using $ARGUMENTS variable
REVIEW_TARGET="$ARGUMENTS"
if [[ -z "$REVIEW_TARGET" ]]; then
  # AskUserQuestion for review type
  REVIEW_TYPE="general"
elif [[ -f "$REVIEW_TARGET" ]] || [[ -d "$REVIEW_TARGET" ]]; then
  # File-focused mode
  REVIEW_TYPE="file-focused"
else
  # Directed mode
  REVIEW_TYPE="directed"
fi
```

---

## Notes

- **Language**: Respond in user's language (한국어 사용자에게는 한국어로)
- **No emojis** in code or documentation
- **Never assume unclear context** - use AskUserQuestion
- Code modifications require user confirmation
- Follow `@CLAUDE.md` project conventions
- **Maximum 3 rounds** for consensus (prevent infinite loops)
- **YAML preferred** for structured output (easier parsing)
- **Evidence-based only** - no generic suggestions without codebase proof
