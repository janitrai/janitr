#!/usr/bin/env python3
"""Sync selected Janitr model artifacts into an offline experiments repo.

This script is intended for a safe handoff workflow:
- Codex prepares artifacts locally.
- User runs this script to copy artifacts into their own local experiments repo.
- User reviews/commits/pushes from that repo manually.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path("models/benchmarks/2026-02-14_expanded_holdout/transformer")
DEFAULT_DEST_ROOT = Path("~/offline/janitr-experiments").expanduser()
DEFAULT_PROFILE = "transformer_int8_benchmark_2026_02_14"


@dataclass(frozen=True)
class FileMapping:
    source_rel: str
    dest_rel: str
    required: bool = True


BASE_MAPPINGS: list[FileMapping] = [
    FileMapping("student.int8.onnx", "model/model.int8.onnx"),
    FileMapping("student/student_config.json", "model/student_config.json"),
    FileMapping("student/tokenizer/tokenizer.json", "model/tokenizer/tokenizer.json"),
    FileMapping(
        "student/tokenizer/tokenizer_config.json",
        "model/tokenizer/tokenizer_config.json",
    ),
    FileMapping("student/tokenizer/vocab.txt", "model/tokenizer/vocab.txt"),
    FileMapping("thresholds.transformer.int8.json", "model/thresholds.int8.json"),
    FileMapping("thresholds.transformer.json", "model/thresholds.torch.json"),
    FileMapping("student_holdout_eval_int8.json", "eval/holdout_eval_int8.json"),
    FileMapping("student_holdout_eval.json", "eval/holdout_eval_torch.json"),
]

OPTIONAL_FP32_MAPPINGS: list[FileMapping] = [
    FileMapping("student.onnx", "model/model.onnx", required=False),
    FileMapping("student/pytorch_model.bin", "model/pytorch_model.bin", required=False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy benchmark transformer artifacts into an offline experiments repo "
            "under runs/<run_id> with a generated manifest."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"Source artifact root (default: {DEFAULT_SOURCE_ROOT})",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=DEFAULT_DEST_ROOT,
        help=f"Destination experiments repo root (default: {DEFAULT_DEST_ROOT})",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier under runs/. Defaults to UTC timestamp + profile.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=DEFAULT_PROFILE,
        help=f"Profile name for metadata (default: {DEFAULT_PROFILE})",
    )
    parser.add_argument(
        "--include-fp32",
        action="store_true",
        help="Include larger FP32 artifacts (student.onnx, pytorch_model.bin) when present.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite destination run directory if it already exists.",
    )
    parser.add_argument(
        "--allow-non-git",
        action="store_true",
        help="Allow destination root that is not a git repository.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print planned copies without writing files.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_id(profile: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}__{profile}"


def ensure_dest_repo(dest_root: Path, allow_non_git: bool) -> None:
    if not dest_root.exists():
        raise FileNotFoundError(
            f"Destination root does not exist: {dest_root}\n"
            "Create/clone your experiments repo first."
        )
    if allow_non_git:
        return
    if not (dest_root / ".git").exists():
        raise RuntimeError(
            f"Destination is not a git repository: {dest_root}\n"
            "Use --allow-non-git to bypass this check."
        )


def resolve_mappings(source_root: Path, include_fp32: bool) -> list[FileMapping]:
    mappings = list(BASE_MAPPINGS)
    if include_fp32:
        mappings.extend(OPTIONAL_FP32_MAPPINGS)
    for mapping in mappings:
        source_path = source_root / mapping.source_rel
        if mapping.required and not source_path.exists():
            raise FileNotFoundError(f"Required source artifact missing: {source_path}")
    return mappings


def make_manifest(
    *,
    run_id: str,
    profile: str,
    source_root: Path,
    run_dir: Path,
    copied_files: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "profile": profile,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_root": str(source_root.resolve()),
        "destination_run_dir": str(run_dir.resolve()),
        "files": copied_files,
    }


def main() -> int:
    args = parse_args()

    source_root = args.source_root.expanduser()
    dest_root = args.dest_root.expanduser()
    run_id = args.run_id or build_run_id(args.profile)
    run_dir = dest_root / "runs" / run_id

    try:
        ensure_dest_repo(dest_root=dest_root, allow_non_git=args.allow_non_git)
        mappings = resolve_mappings(
            source_root=source_root,
            include_fp32=args.include_fp32,
        )
    except Exception as exc:  # pragma: no cover - simple CLI error path
        print(f"[sync] error: {exc}", file=sys.stderr)
        return 2

    if run_dir.exists():
        if not args.force:
            print(
                f"[sync] error: destination run already exists: {run_dir}\n"
                "Use --force to overwrite.",
                file=sys.stderr,
            )
            return 2
        if not args.dry_run:
            shutil.rmtree(run_dir)

    copied_files: list[dict[str, object]] = []
    planned = []
    for mapping in mappings:
        src = source_root / mapping.source_rel
        if not src.exists():
            continue
        dst = run_dir / mapping.dest_rel
        planned.append((src, dst))

    print(f"[sync] source_root: {source_root}")
    print(f"[sync] dest_root:   {dest_root}")
    print(f"[sync] run_id:      {run_id}")
    print(f"[sync] files:       {len(planned)}")

    if args.dry_run:
        for src, dst in planned:
            print(f"[dry-run] {src} -> {dst}")
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in planned:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied_files.append(
            {
                "source": str(src.resolve()),
                "destination": str(dst.relative_to(dest_root)),
                "size_bytes": src.stat().st_size,
                "sha256": sha256_file(dst),
            }
        )
        print(f"[sync] copied {src} -> {dst}")

    manifest = make_manifest(
        run_id=run_id,
        profile=args.profile,
        source_root=source_root,
        run_dir=run_dir,
        copied_files=copied_files,
    )
    manifest_path = run_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[sync] wrote {manifest_path}")

    rel_run_dir = run_dir.relative_to(dest_root)
    print("")
    print("[sync] next steps:")
    print(f"  cd {dest_root}")
    print(f"  git add {rel_run_dir}")
    print(f'  git commit -m "add experiment run {run_id}"')
    print("  git push")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
