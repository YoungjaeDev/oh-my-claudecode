---
name: gh-create-label
description: Analyze project and create appropriate GitHub issue labels
user-invocable: true
---

# Create Issue Labels

[LABEL CREATION MODE ACTIVATED]

Analyze project structure and create appropriate GitHub issue labels.

## Workflow

### 1. Analyze Project

Examine `package.json`, `README.md`, and code structure.

### 2. Identify Tech Stack

Detect frameworks, libraries, and tools in use.

### 3. Classify Project Areas

Frontend, backend, API, infrastructure, etc.

### 4. Create Labels

Generate essential labels based on type, area, and complexity.

## Label Categories

### Type Labels

```bash
gh label create "type: feature" --color "0e8a16" --description "New feature addition"
gh label create "type: bug" --color "d73a4a" --description "Something isn't working"
gh label create "type: enhancement" --color "a2eeef" --description "Improvement to existing feature"
gh label create "type: documentation" --color "0075ca" --description "Documentation updates"
gh label create "type: refactor" --color "fbca04" --description "Code refactoring"
```

### Area Labels

```bash
gh label create "frontend" --color "1d76db" --description "Frontend-related work"
gh label create "backend" --color "5319e7" --description "Backend-related work"
gh label create "api" --color "006b75" --description "API-related work"
gh label create "infrastructure" --color "d4c5f9" --description "Infrastructure and DevOps"
gh label create "database" --color "f9d0c4" --description "Database-related work"
```

### Complexity Labels

```bash
gh label create "complexity: easy" --color "7057ff" --description "Simple task, good for beginners"
gh label create "complexity: medium" --color "fef2c0" --description "Moderate complexity"
gh label create "complexity: hard" --color "b60205" --description "Complex task, requires expertise"
```

### Priority Labels

```bash
gh label create "priority: high" --color "d93f0b" --description "High priority, address soon"
gh label create "priority: medium" --color "fbca04" --description "Medium priority"
gh label create "priority: low" --color "0e8a16" --description "Low priority, address when possible"
```

## Customization

Adjust labels based on project needs. Common additions:

- **AI/ML projects**: `model`, `training`, `dataset`, `inference`
- **Web apps**: `ui`, `ux`, `accessibility`, `performance`
- **Libraries**: `breaking-change`, `deprecation`, `compatibility`

## Output

List created labels with their colors and descriptions.
