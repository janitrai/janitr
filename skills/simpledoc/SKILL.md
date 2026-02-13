---
name: simpledoc
description: Create or update documentation in this repo following SimpleDoc conventions. Use when creating docs, plans, logs, or any markdown files.
---

# SimpleDoc Documentation Skill

**Attention agent!** Complete every item below before touching documentation work:

1. **Read this file in full for the current session.** No shortcuts.
2. **Verify that git is initialized and configured.** You will need the name and email of the current user in order to populate the `author` field in the YAML frontmatter. Run the following one-liner to verify:

```bash
printf '%s <%s>\n' "$(git config --local --get user.name 2>/dev/null || git config --global --get user.name)" "$(git config --local --get user.email 2>/dev/null || git config --global --get user.email)"
```

If the name and email are not available for some reason, ask the user to provide them, and also setup git configuration for them.

## SimpleDoc Specification

SimpleDoc defines two types of files:

1. **Date-prefixed files**: SHOULD be used for most documents, e.g. `docs/2025-12-22-an-awesome-doc.md`.
2. **Capitalized files**: SHOULD be used for general documents that are not tied to a specific time, e.g. `README.md`.

### 1. Date-prefixed files

- MUST put date-prefixed files in a top level `docs/` folder, or a subfolder `docs/<topic>/`. Subfolders MAY be nested indefinitely.
- MUST use ISO 8601 date prefixes (`YYYY-MM-DD`) — the date MUST contain dashes.
- After the date prefix, lowercase filenames SHOULD use dashes (`-`) as word delimiters (kebab-case). Avoid spaces and underscores.
- The date prefix MAY be the entire filename (for example, daily logs like `docs/logs/2026-02-04.md`).
- MUST NOT use capital letters in filename for Latin, Greek, Cyrillic and other writing systems that have lowercase/uppercase distinction.
- MAY use non-ASCII characters.
- Date-prefixed files SHOULD contain YAML frontmatter with at least `title`, `author` and `date` fields:
  ```yaml
  ---
  title: Implementation Plan
  author: John Doe <john.doe@example.com>
  date: 2025-12-22
  ---
  ```
- If present in YAML frontmatter, author SHOULD be of `Name <email>` per the RFC 5322 name-addr mailbox format and date SHOULD be ISO 8601 `YYYY-MM-DD` format.

### 2. Capitalized files

- For general documents not tied to a specific time, e.g. `README.md`, `AGENTS.md`, `INSTALL.md`, `HOW_TO_DEBUG.md`.
- Multi-word filenames SHOULD use underscores (`CODE_OF_CONDUCT.md`).

## Preferences in Documentation Style

- Tone: casual, clear, technically precise but not academic
- Planning docs: concrete and actionable, include checklists
- Keep docs concise — no fluff, no filler paragraphs
- Use ISO timestamps where relevant
- Prefer bullet points over prose for technical content

## Before You Start

1. Run `date +%Y-%m-%d` and use the output for both filename prefix and `date` field.
2. Identify where the document belongs:
   - Keep general documentation at the root of `docs/`.
   - Use dedicated subdirectories for specialized content (plans, logs, reports).
3. Check for existing, related docs to avoid duplicates and to link to prior work.

## File Naming

- Format: `YYYY-MM-DD-descriptive-title.md`. The date MUST use dashes; the rest SHOULD be lowercase with hyphens (avoid underscores).
- Choose names that reflect the problem or topic, not the team or author.
- Example: `2025-06-20-api-migration-guide.md`.
- Place the file in the appropriate folder before committing.

### Timeless vs. Dated

- **Timeless general documents** describe enduring processes or repo-wide rules. They do not carry a date prefix and keep their canonical names.
- **All other content** (design notes, incidents, feature guides, migrations, meeting notes, plans, etc.) must use the date-prefixed naming pattern above.
- When adding or reviewing documentation, decide which bucket applies.

## Required Front Matter

Every doc **must** start with YAML front matter:

```yaml
---
date: 2025-10-24 # From `date +%Y-%m-%d`
author: Name <email@example.com>
title: Short Descriptive Title
tags: [tag1, tag2] # Optional but recommended
---
```

## Daily Logs (SimpleLog)

Default location: `docs/logs/YYYY-MM-DD.md`.

### Create a daily log entry

```bash
npx -y @simpledoc/simpledoc log "Entry text here"
```

For multiline:
```bash
cat <<'EOF' | npx -y @simpledoc/simpledoc log --stdin
Multiline entry here
- point one
- point two
EOF
```

### Manual edits (if needed)

- Keep the YAML frontmatter intact (`title`, `author`, `date`, `tz`, `created`, optional `updated`).
- Ensure a blank line separates entries.
- Session sections must be `## HH:MM` (local time of the first entry in that section).

### Ongoing logging

Log anything worth noting: significant changes, decisions, errors, workarounds, progress. Log each entry after completing the step. No exceptions.

## Final Checks Before Submitting

- [ ] Filename follows the `YYYY-MM-DD-…` pattern and lives in the correct directory.
- [ ] Front matter is complete and accurate.
- [ ] Links to related documentation exist where applicable.
- [ ] Run `npx -y @simpledoc/simpledoc check` to verify SimpleDoc conventions.
