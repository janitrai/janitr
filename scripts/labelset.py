from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_LABELS_MD_PATH = REPO_ROOT / "docs" / "LABELS.md"


def load_v2026_labels_from_labels_md(
    labels_md_path: Path = DEFAULT_LABELS_MD_PATH,
) -> list[str]:
    """
    Extract the canonical v2026 label set from the YAML block in docs/LABELS.md.

    This intentionally avoids a YAML dependency; the v2026 block is a simple mapping
    from group -> list of labels, so we just extract all `- label_name` entries from
    the fenced ```yaml block.
    """

    lines = labels_md_path.read_text(encoding="utf-8").splitlines()

    in_yaml_block = False
    labels: list[str] = []
    seen: set[str] = set()

    for line in lines:
        if not in_yaml_block:
            if line.strip() == "```yaml":
                in_yaml_block = True
            continue

        if line.strip() == "```":
            break

        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        m = re.match(r"^-\s*([a-z0-9_]+)\s*(?:#.*)?$", stripped)
        if not m:
            continue

        label = m.group(1)
        if label in seen:
            continue

        labels.append(label)
        seen.add(label)

    if not in_yaml_block:
        raise RuntimeError(
            f"Could not find a fenced ```yaml block in {labels_md_path} "
            "to extract the v2026 label set."
        )

    if not labels:
        raise RuntimeError(
            f"Found fenced ```yaml block in {labels_md_path}, but extracted 0 labels. "
            "Expected list items like `- scam` / `- topic_crypto`."
        )

    return labels
