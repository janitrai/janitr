#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Sync janitr experiment artifacts from a remote machine to local.

Usage:
  scripts/sync_experiments_from_remote.sh \
    --remote USER@HOST \
    [--remote-path /home/bob/offline/janitr-experiments] \
    [--dest ~/offline/janitr-experiments] \
    [--run-id RUN_ID] \
    [--dry-run] \
    [--delete]

Options:
  --remote       Required. Remote SSH target, e.g. bob@203.0.113.10
  --remote-path  Remote experiments root. Default: /home/bob/offline/janitr-experiments
  --dest         Local destination root. Default: ~/offline/janitr-experiments
  --run-id       Optional. Pull only one run under runs/<RUN_ID>
  --dry-run      Show what would be copied, without writing.
  --delete       Mirror mode. Delete local files not present remotely.

Notes:
  - Uses rsync over SSH.
  - Excludes .git so your local repo metadata is not overwritten.
EOF
}

REMOTE=""
REMOTE_PATH="/home/bob/offline/janitr-experiments"
DEST_PATH="$HOME/offline/janitr-experiments"
RUN_ID=""
DRY_RUN=0
DELETE_MODE=0

while (($#)); do
  case "$1" in
    --remote)
      REMOTE="${2:-}"
      shift 2
      ;;
    --remote-path)
      REMOTE_PATH="${2:-}"
      shift 2
      ;;
    --dest)
      DEST_PATH="${2:-}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --delete)
      DELETE_MODE=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$REMOTE" ]]; then
  echo "Missing required --remote USER@HOST" >&2
  usage >&2
  exit 2
fi

mkdir -p "$DEST_PATH"

RSYNC_ARGS=(
  -avz
  --progress
  --exclude=.git/
)

if [[ "$DRY_RUN" -eq 1 ]]; then
  RSYNC_ARGS+=(--dry-run)
fi

if [[ "$DELETE_MODE" -eq 1 ]]; then
  RSYNC_ARGS+=(--delete)
fi

if [[ -n "$RUN_ID" ]]; then
  SRC="${REMOTE}:${REMOTE_PATH%/}/runs/${RUN_ID}/"
  DST="${DEST_PATH%/}/runs/${RUN_ID}/"
  mkdir -p "$DST"
else
  SRC="${REMOTE}:${REMOTE_PATH%/}/"
  DST="${DEST_PATH%/}/"
fi

echo "[sync] source:      $SRC"
echo "[sync] destination: $DST"
echo "[sync] dry-run:     $DRY_RUN"
echo "[sync] delete:      $DELETE_MODE"

rsync "${RSYNC_ARGS[@]}" "$SRC" "$DST"

echo "[sync] done"
