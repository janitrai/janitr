#!/usr/bin/env python3
"""Build unlabeled tweet text corpus for optional teacher DAPT (MLM)."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from transformer_common import DATA_DIR, clean_text, load_jsonl

DEFAULT_INPUTS = [
    DATA_DIR / "train.jsonl",
    DATA_DIR / "valid.jsonl",
    DATA_DIR / "holdout.jsonl",
]
DEFAULT_OUTPUT = DATA_DIR / "transformer" / "unlabeled_corpus.txt"


def iter_input_paths(paths_value: str | None) -> list[Path]:
    if not paths_value:
        return list(DEFAULT_INPUTS)
    out: list[Path] = []
    for item in paths_value.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(Path(item))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=str,
        default=None,
        help="Comma-separated JSONL input files. Defaults to train/valid/holdout.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--strip-urls", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--no-lowercase", action="store_true")
    args = parser.parse_args()

    paths = iter_input_paths(args.inputs)
    if not paths:
        raise SystemExit("No input files provided.")
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Input not found: {path}")

    seen: set[str] = set()
    kept = 0
    skipped_short = 0
    skipped_empty = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as out_f:
        for path in paths:
            for sample in load_jsonl(path):
                text_raw = sample.get("text") or sample.get("raw_text") or ""
                text = clean_text(
                    text_raw,
                    normalize=not args.no_normalize,
                    lowercase=not args.no_lowercase,
                    strip_urls=args.strip_urls,
                )
                if not text:
                    skipped_empty += 1
                    continue
                if len(text) < args.min_chars:
                    skipped_short += 1
                    continue

                digest = hashlib.blake2b(
                    text.encode("utf-8"), digest_size=16
                ).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)

                out_f.write(text + "\n")
                kept += 1
                if args.max_rows > 0 and kept >= args.max_rows:
                    break
            if args.max_rows > 0 and kept >= args.max_rows:
                break

    print(
        f"Wrote {kept} unique lines to {args.output} "
        f"(skipped_empty={skipped_empty}, skipped_short={skipped_short})"
    )


if __name__ == "__main__":
    main()
