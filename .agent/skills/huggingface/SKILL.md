---
name: huggingface
description: Sync Janitr experiment artifacts from remote to local and publish to Hugging Face experiments repo with single-index and manifest validation.
---

# Hugging Face Experiments Skill

## Purpose

Use the repo scripts to move experiment artifacts from the remote machine to a local clone of `janitr/experiments`, then commit and push.

## Canonical Paths

- Remote experiments clone: `/home/bob/offline/janitr-experiments`
- Local experiments clone: `~/offline/janitr-experiments`
- Remote-to-local sync script: `scripts/sync_experiments_from_remote.sh`
- Remote artifact staging script: `scripts/sync_to_experiments_repo.py`
- Dataset sync script: `scripts/sync_datasets_to_experiments_repo.py`
- Index rebuild script: `scripts/rebuild_experiment_index.py`
- Manifest validation script: `scripts/validate_experiment_runs.py`

## Rules

- Default sync is full repo sync. Do not pass `--run-id` unless explicitly requested.
- Prefer the provided scripts over bare `rsync` commands.
- Keep run naming format as `yyyy-mm-dd-<petname>` when creating new run directories.
- Runs root must contain exactly one index file: `runs/INDEX.json`.
- Always rebuild index after run changes and validate before commit/push.
- Keep model artifacts at or below 30MB.
- Use `--delete` when syncing remote to local to prevent stale/empty local run dirs.
- Dataset snapshots use auto names `yyyy-mm-dd-<petname>` by default.
- Dataset sync is deduplicated by content hash; no new snapshot is created when x-posts+x-replies files are unchanged.

## Workflow

1. Stage artifacts into the remote experiments clone:

```bash
python3 scripts/sync_to_experiments_repo.py \
  --dest-root ~/offline/janitr-experiments \
  --run-id 2026-02-15-flying-narwhal
```

2. Sync remote experiments repo to local (default full sync, no run filter, mirror mode):

```bash
bash scripts/sync_experiments_from_remote.sh \
  --remote bob@<remote-host> \
  --remote-path /home/bob/offline/janitr-experiments \
  --dest ~/offline/janitr-experiments \
  --delete
```

3. Optional dry-run before real sync:

```bash
bash scripts/sync_experiments_from_remote.sh \
  --remote bob@<remote-host> \
  --remote-path /home/bob/offline/janitr-experiments \
  --dest ~/offline/janitr-experiments \
  --delete \
  --dry-run
```

4. Rebuild single index and validate manifests:

```bash
python3 scripts/rebuild_experiment_index.py \
  --runs-root ~/offline/janitr-experiments/runs

python3 scripts/validate_experiment_runs.py \
  --runs-root ~/offline/janitr-experiments/runs \
  --require-runs-root
```

5. Commit and push from local experiments clone:

```bash
cd ~/offline/janitr-experiments
git status
git add runs
git commit -m "add experiment run(s)"
git push
```

## Dataset Snapshots

Use this to stage current x-posts + x-replies datasets into the experiments repo:

```bash
python3 scripts/sync_datasets_to_experiments_repo.py \
  --dest-root ~/offline/janitr-experiments
```

Defaults:

- x-posts dataset source: `data/sample.jsonl`
- x-replies dataset source: `data/replies.jsonl`
- snapshot id: auto-generated `yyyy-mm-dd-<petname>`

Behavior:

- Writes snapshot under `datasets/snapshots/<snapshot_id>/`.
- Updates `datasets/INDEX.json`.
- If content is unchanged (same hashes), does not create a duplicate snapshot.

## Optional: Sync a Single Run

Use only when explicitly asked:

```bash
bash scripts/sync_experiments_from_remote.sh \
  --remote bob@<remote-host> \
  --remote-path /home/bob/offline/janitr-experiments \
  --dest ~/offline/janitr-experiments \
  --run-id 2026-02-15-flying-narwhal
```
