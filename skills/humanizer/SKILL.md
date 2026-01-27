---
name: humanizer
description: Remove signs of AI-generated writing and add authentic human voice
---

# Humanizer Skill

Transform AI-generated text into natural, human-sounding writing by detecting and fixing common AI patterns.

## The Insight

LLMs use statistical algorithms to guess what should come next. The result tends toward the most statistically likely result that applies to the widest variety of cases. This creates detectable patterns.

Based on Wikipedia's "Signs of AI writing" guide, maintained by WikiProject AI Cleanup.

## Recognition Patterns

Trigger on: "humanize", "make it human", "remove AI slop", "인간적으로", "AI 느낌 빼", "자연스럽게"

## Your Task

When given text to humanize:

1. **Identify AI patterns** - Scan for the patterns listed below
2. **Rewrite problematic sections** - Replace AI-isms with natural alternatives
3. **Preserve meaning** - Keep the core message intact
4. **Maintain voice** - Match the intended tone (formal, casual, technical, etc.)
5. **Add soul** - Don't just remove bad patterns; inject actual personality

---

## The 24 AI Writing Patterns

### Category 1: Content Patterns (6 patterns)

| # | Pattern | Description | Example |
|---|---------|-------------|---------|
| 1 | **Significance inflation** | Overstating importance with grandiose framing | BEFORE: "marking a pivotal moment in the evolution of..." / AFTER: "This changed how we..." |
| 2 | **Notability name-dropping** | Citing prestigious sources without substance | BEFORE: "cited in NYT, BBC, FT, and The Hindu" / AFTER: Remove unless directly relevant |
| 3 | **Superficial -ing analyses** | Using -ing verbs to sound analytical | BEFORE: "symbolizing... reflecting... showcasing..." / AFTER: Use concrete observations |
| 4 | **Promotional language** | Marketing-speak disguised as description | BEFORE: "nestled within the breathtaking region" / AFTER: "located in the valley" |
| 5 | **Vague attributions** | Citing unnamed "experts" or "studies" | BEFORE: "Experts believe it plays a crucial role" / AFTER: Name the expert or remove |
| 6 | **Formulaic challenges** | Template resilience narratives | BEFORE: "Despite challenges... continues to thrive" / AFTER: Be specific about what challenges and outcomes |

### Category 2: Language Patterns (6 patterns)

| # | Pattern | Description | Example |
|---|---------|-------------|---------|
| 7 | **AI vocabulary** | Overused AI-specific words | BEFORE: "Additionally... testament... landscape... showcasing" / AFTER: Use plain alternatives |
| 8 | **Copula avoidance** | Avoiding "is/has" with fancier verbs | BEFORE: "serves as... features... boasts" / AFTER: "is... has..." |
| 9 | **Negative parallelisms** | Fake tension constructions | BEFORE: "It's not just X, it's Y" / AFTER: "It's Y" (just say the thing) |
| 10 | **Rule of three** | Forced triplets for rhythm | BEFORE: "innovation, inspiration, and insights" / AFTER: Use one or two specific terms |
| 11 | **Synonym cycling** | Rotating synonyms unnaturally | BEFORE: "protagonist... main character... central figure... hero" / AFTER: Pick one and stick with it |
| 12 | **False ranges** | Meaninglessly broad scopes | BEFORE: "from the Big Bang to dark matter" / AFTER: Be specific about actual scope |

### Category 3: Style Patterns (6 patterns)

| # | Pattern | Description | Example |
|---|---------|-------------|---------|
| 13 | **Em dash overuse** | Excessive em dashes for drama | BEFORE: "institutions—not the people—yet this continues—" / AFTER: Use commas or periods |
| 14 | **Boldface overuse** | Bolding everything important | BEFORE: "**OKRs**, **KPIs**, **BMC**" / AFTER: Bold sparingly or not at all |
| 15 | **Inline-header lists** | Redundant bold headers in lists | BEFORE: "**Performance:** Performance improved" / AFTER: "Performance improved" |
| 16 | **Title Case Headings** | Capitalizing Every Word | BEFORE: "Strategic Negotiations And Partnerships" / AFTER: "Strategic negotiations and partnerships" |
| 17 | **Emojis** | Using emojis in professional writing | BEFORE: "🚀 Launch Phase: 💡 Key Insight:" / AFTER: Remove emojis in formal contexts |
| 18 | **Curly quotes** | Using typographic quotes | BEFORE: Using " " / AFTER: Use straight quotes " or ' |

### Category 4: Communication Patterns (3 patterns)

| # | Pattern | Description | Example |
|---|---------|-------------|---------|
| 19 | **Chatbot artifacts** | Leftover conversational phrases | BEFORE: "I hope this helps! Let me know if..." / AFTER: Remove entirely |
| 20 | **Cutoff disclaimers** | Hedging about knowledge limits | BEFORE: "While details are limited in available sources..." / AFTER: Just state what you know |
| 21 | **Sycophantic tone** | Excessive praise of the reader | BEFORE: "Great question! You're absolutely right!" / AFTER: Answer directly |

### Category 5: Filler and Hedging (3 patterns)

| # | Pattern | Description | Example |
|---|---------|-------------|---------|
| 22 | **Filler phrases** | Unnecessary padding words | BEFORE: "In order to", "Due to the fact that" / AFTER: "To", "Because" |
| 23 | **Excessive hedging** | Stacking uncertainty qualifiers | BEFORE: "could potentially possibly" / AFTER: "might" or just state it |
| 24 | **Generic conclusions** | Empty optimistic endings | BEFORE: "The future looks bright" / AFTER: Specific next steps or remove |

---

## Adding Soul (Beyond Pattern Detection)

Humanization is not just about removing bad patterns. Add authentic voice:

1. **Opinions**: Take a stance. "This approach works well" not "This approach could potentially work"
2. **Varied rhythm**: Mix sentence lengths. Short punchy. Then a longer, more flowing sentence with subordinate clauses.
3. **Complexity acknowledgment**: "It's complicated" is human. Neat summaries are suspicious.
4. **Specific details**: Names, numbers, dates. Not "various stakeholders" but "the engineering team"
5. **Imperfection**: Occasional fragments. Starting with "And" or "But". Real humans do this.

### Signs of soulless writing (even if technically "clean"):
- Every sentence is the same length and structure
- No opinions, just neutral reporting
- No acknowledgment of uncertainty or mixed feelings
- No first-person perspective when appropriate
- No humor, no edge, no personality
- Reads like a Wikipedia article or press release

---

## Full Example

**Before (AI-sounding):**
> Great question! Here is an essay on this topic. I hope this helps!
>
> AI-assisted coding serves as an enduring testament to the transformative potential of large language models, marking a pivotal moment in the evolution of software development. In today's rapidly evolving technological landscape, these groundbreaking tools—nestled at the intersection of research and practice—are reshaping how engineers ideate, iterate, and deliver, underscoring their vital role in modern workflows.

**After (Humanized):**
> AI coding assistants speed up some tasks. In a 2024 study by Google, developers using Codex completed simple functions 55% faster than a control group, but showed no improvement on debugging or architectural decisions.
>
> The tools are good at boilerplate: config files, test scaffolding, repetitive refactors. They are bad at knowing when they are wrong. I have mass-accepted suggestions that compiled, passed lint, and still did the wrong thing because I stopped paying attention.

---

## Process

1. Read the input text carefully
2. Identify all instances of the patterns above
3. Rewrite each problematic section
4. Ensure the revised text:
   - Sounds natural when read aloud
   - Varies sentence structure naturally
   - Uses specific details over vague claims
   - Maintains appropriate tone for context
   - Uses simple constructions (is/are/has) where appropriate
5. Present the humanized version

## Output Format

Provide:
1. The rewritten text
2. A brief summary of changes made (optional, if helpful)

---

## Reference

This skill is based on [Wikipedia:Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing), maintained by WikiProject AI Cleanup.
