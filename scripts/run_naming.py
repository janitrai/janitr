#!/usr/bin/env python3
"""Helpers for human-friendly dated run naming."""

from __future__ import annotations

import re
import secrets
from datetime import datetime, timezone
from pathlib import Path

ADJECTIVES = [
    "agile",
    "amber",
    "apex",
    "arcane",
    "aurora",
    "bold",
    "brisk",
    "calm",
    "crisp",
    "daring",
    "eager",
    "fierce",
    "flying",
    "frosty",
    "gentle",
    "glossy",
    "golden",
    "granite",
    "keen",
    "lively",
    "lunar",
    "mellow",
    "misty",
    "noble",
    "onyx",
    "plucky",
    "rapid",
    "scarlet",
    "silent",
    "solar",
    "spry",
    "steady",
    "swift",
    "tidy",
    "vivid",
    "wild",
]

ANIMALS = [
    "albatross",
    "badger",
    "beacon",
    "bison",
    "cougar",
    "coyote",
    "dragonfly",
    "eagle",
    "falcon",
    "fox",
    "gazelle",
    "gecko",
    "hawk",
    "ibex",
    "jaguar",
    "koala",
    "lemur",
    "lynx",
    "narwhal",
    "ocelot",
    "otter",
    "owl",
    "panther",
    "puma",
    "quokka",
    "raven",
    "seal",
    "sparrow",
    "stoat",
    "swiftlet",
    "tiger",
    "viper",
    "walrus",
    "wolf",
    "wren",
    "yak",
]

DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-")


def today_iso_date_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def sanitize_name_token(value: str) -> str:
    token = value.strip().lower().replace("_", "-")
    token = re.sub(r"[^a-z0-9-]+", "-", token)
    token = re.sub(r"-{2,}", "-", token).strip("-")
    return token


def generate_petname() -> str:
    rng = secrets.SystemRandom()
    return f"{rng.choice(ADJECTIVES)}-{rng.choice(ANIMALS)}"


def resolve_run_name(name: str | None) -> str:
    date_prefix = today_iso_date_utc()
    if not name:
        return f"{date_prefix}-{generate_petname()}"

    cleaned = sanitize_name_token(name)
    if not cleaned:
        return f"{date_prefix}-{generate_petname()}"
    if DATE_PREFIX_RE.match(cleaned):
        return cleaned
    return f"{date_prefix}-{cleaned}"


def apply_run_name_template(path: Path, run_name: str) -> Path:
    raw = str(path)
    if "{run_name}" not in raw:
        return path
    return Path(raw.replace("{run_name}", run_name))
