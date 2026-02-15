---
name: huggingface
description: Sync Janitr experiment artifacts from remote to local and publish to Hugging Face experiments repo with the repo scripts.
---

# Hugging Face Experiments Skill

## Purpose

Use the repo scripts to move experiment artifacts from the remote machine to a local clone of `janitr/experiments`, then commit and push.

## Canonical Paths

- Remote experiments clone: `/home/bob/offline/janitr-experiments`
- Local experiments clone: `~/offline/janitr-experiments`
- Remote-to-local sync script: `scripts/sync_experiments_from_remote.sh`
- Remote artifact staging script: `scripts/sync_to_experiments_repo.py`

## Rules

- Default sync is full repo sync. Do not pass `--run-id` unless explicitly requested.
- Prefer the provided scripts over bare `rsync` commands.
- Keep run naming format as `yyyy-mm-dd-<petname>` when creating new run directories.

## Workflow

1. Stage artifacts into the remote experiments clone:

```bash
python3 scripts/sync_to_experiments_repo.py \
  --dest-root ~/offline/janitr-experiments \
  --run-id 2026-02-15-flying-narwhal
```

2. Sync remote experiments repo to local (default full sync, no run filter):

```bash
bash scripts/sync_experiments_from_remote.sh \
  --remote bob@<remote-host> \
  --remote-path /home/bob/offline/janitr-experiments \
  --dest ~/offline/janitr-experiments
```

3. Optional dry-run before real sync:

```bash
bash scripts/sync_experiments_from_remote.sh \
  --remote bob@<remote-host> \
  --remote-path /home/bob/offline/janitr-experiments \
  --dest ~/offline/janitr-experiments \
  --dry-run
```

4. Commit and push from local experiments clone:

```bash
cd ~/offline/janitr-experiments
git status
git add runs
git commit -m "add experiment run(s)"
git push
```

## Optional: Sync a Single Run

Use only when explicitly asked:

```bash
bash scripts/sync_experiments_from_remote.sh \
  --remote bob@<remote-host> \
  --remote-path /home/bob/offline/janitr-experiments \
  --dest ~/offline/janitr-experiments \
  --run-id 2026-02-15-flying-narwhal
```
