#!/usr/bin/env python3
"""
Generate size-reduced fastText models from a reference .bin.
This does not retrain; it only applies post-training reduction.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = REPO_ROOT / "models" / "scam_detector.bin"
DEFAULT_VALID = REPO_ROOT / "data" / "valid.txt"
DEFAULT_OUT_DIR = REPO_ROOT / "models" / "reduced"
DEFAULT_RESULTS = DEFAULT_OUT_DIR / "reduction_results.csv"
DEFAULT_THRESHOLD = 0.90

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import evaluate as eval_mod  # type: ignore


@dataclass(frozen=True)
class ReductionSpec:
    name: str
    quant_args: dict[str, Any]
    pca_dim: int | None = None


def parse_int_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_bool_list(value: str) -> list[bool]:
    if not value:
        return []
    out: list[bool] = []
    for item in value.split(","):
        token = item.strip().lower()
        if token in {"1", "true", "t", "yes", "y"}:
            out.append(True)
        elif token in {"0", "false", "f", "no", "n"}:
            out.append(False)
        else:
            raise ValueError(f"Invalid boolean token: {item}")
    return out


def build_specs(
    profile: str,
    cutoffs: list[int],
    dsubs: list[int],
    qouts: list[bool],
    qnorms: list[bool],
    pca_dims: list[int],
) -> list[ReductionSpec]:
    specs: list[ReductionSpec] = []
    if profile == "grid":
        for cutoff in cutoffs:
            for dsub in dsubs:
                for qout in qouts:
                    for qnorm in qnorms:
                        name_parts = [f"cutoff{cutoff}", f"dsub{dsub}"]
                        if qout:
                            name_parts.append("qout")
                        if qnorm:
                            name_parts.append("qnorm")
                        name = "quant-" + "-".join(name_parts)
                        specs.append(
                            ReductionSpec(
                                name=name,
                                quant_args={
                                    "cutoff": cutoff,
                                    "dsub": dsub,
                                    "qout": qout,
                                    "qnorm": qnorm,
                                },
                            )
                        )
    else:
        specs.extend(
            [
                ReductionSpec(
                    "quant-default",
                    {"cutoff": 0, "dsub": 2, "qout": False, "qnorm": False},
                ),
                ReductionSpec(
                    "quant-dsub4",
                    {"cutoff": 0, "dsub": 4, "qout": False, "qnorm": False},
                ),
                ReductionSpec(
                    "quant-dsub8",
                    {"cutoff": 0, "dsub": 8, "qout": False, "qnorm": False},
                ),
                ReductionSpec(
                    "quant-cutoff100k",
                    {"cutoff": 100000, "dsub": 2, "qout": False, "qnorm": False},
                ),
                ReductionSpec(
                    "quant-cutoff50k",
                    {"cutoff": 50000, "dsub": 2, "qout": False, "qnorm": False},
                ),
                ReductionSpec(
                    "quant-cutoff20k",
                    {"cutoff": 20000, "dsub": 2, "qout": False, "qnorm": False},
                ),
                ReductionSpec(
                    "quant-cutoff10k",
                    {"cutoff": 10000, "dsub": 2, "qout": False, "qnorm": False},
                ),
                ReductionSpec(
                    "quant-cutoff5k",
                    {"cutoff": 5000, "dsub": 2, "qout": False, "qnorm": False},
                ),
                ReductionSpec(
                    "quant-cutoff1k",
                    {"cutoff": 1000, "dsub": 2, "qout": False, "qnorm": False},
                ),
            ]
        )
    for dim in pca_dims:
        specs.append(
            ReductionSpec(
                name=f"pca{dim}-quant-cutoff50k",
                pca_dim=dim,
                quant_args={"cutoff": 50000, "dsub": 4, "qout": True, "qnorm": True},
            )
        )
    return specs


def evaluate_model(model, valid_path: Path, threshold: float) -> dict[str, float]:
    rows: list[tuple[str, float]] = []
    total = 0
    with open(valid_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parsed = eval_mod.parse_line(line)
            if parsed is None:
                continue
            actual, text = parsed
            if actual not in eval_mod.CLASSES:
                continue
            scores = eval_mod.get_probs(model, text)
            rows.append((actual, scores["scam"]))
            total += 1
    if total == 0:
        raise SystemExit("No valid rows found for evaluation.")
    confusion = eval_mod.build_confusion(rows, threshold)
    return eval_mod.summarize(confusion)


def run_spec(
    spec: ReductionSpec,
    model_path: Path,
    valid_path: Path,
    out_dir: Path,
    threshold: float,
) -> dict[str, Any]:
    import fasttext  # type: ignore

    try:
        import fasttext.util as ft_util  # type: ignore
    except Exception:
        ft_util = None

    model = fasttext.load_model(str(model_path))
    if spec.pca_dim is not None:
        if ft_util is None or not hasattr(ft_util, "reduce_model"):
            raise SystemExit("fasttext.util.reduce_model is not available; omit --pca-dims.")
        ft_util.reduce_model(model, spec.pca_dim)

    model.quantize(**spec.quant_args)
    out_path = out_dir / f"{spec.name}.ftz"
    model.save_model(str(out_path))
    size_mb = out_path.stat().st_size / (1024 * 1024)

    metrics = evaluate_model(model, valid_path, threshold)
    return {
        "name": spec.name,
        "size_mb": round(size_mb, 2),
        "pca_dim": spec.pca_dim or "",
        "cutoff": spec.quant_args.get("cutoff", ""),
        "dsub": spec.quant_args.get("dsub", ""),
        "qout": int(bool(spec.quant_args.get("qout", False))),
        "qnorm": int(bool(spec.quant_args.get("qnorm", False))),
        "precision": round(metrics["scam_precision"], 4),
        "recall": round(metrics["scam_recall"], 4),
        "f1": round(metrics["scam_f1"], 4),
        "fpr": round(metrics["fpr"], 4),
        "fnr": round(metrics["fnr"], 4),
        "model_path": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reduce a fastText .bin model into smaller .ftz candidates"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Reference .bin model")
    parser.add_argument("--valid", type=Path, default=DEFAULT_VALID, help="Validation data file")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help="CSV output path (use '-' to skip writing)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Predict scam when p(scam) >= threshold",
    )
    parser.add_argument(
        "--profile",
        choices=("compact", "grid"),
        default="compact",
        help="compact = curated set, grid = full cartesian sweep",
    )
    parser.add_argument("--cutoffs", default="0,200000,100000,50000,20000")
    parser.add_argument("--dsubs", default="2,4,8")
    parser.add_argument("--qout-options", default="0,1")
    parser.add_argument("--qnorm-options", default="0,1")
    parser.add_argument("--pca-dims", default="", help="Optional PCA dims, e.g. 50,25")
    parser.add_argument("--only", default="", help="Comma-separated spec names to run")
    parser.add_argument("--list", action="store_true", help="List configs and exit")
    args = parser.parse_args()

    cutoffs = parse_int_list(args.cutoffs)
    dsubs = parse_int_list(args.dsubs)
    qouts = parse_bool_list(args.qout_options)
    qnorms = parse_bool_list(args.qnorm_options)
    pca_dims = parse_int_list(args.pca_dims)

    specs = build_specs(args.profile, cutoffs, dsubs, qouts, qnorms, pca_dims)

    if args.list:
        for spec in specs:
            print(f"{spec.name} {spec.quant_args} pca_dim={spec.pca_dim}")
        return

    if not args.model.exists():
        raise SystemExit(f"Model file not found: {args.model}")
    if not args.valid.exists():
        raise SystemExit(f"Validation file not found: {args.valid}")

    try:
        import fasttext  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "fasttext is not installed. Install with: pip install fasttext-wheel"
        ) from exc

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.only:
        allow = {name.strip() for name in args.only.split(",") if name.strip()}
        specs = [spec for spec in specs if spec.name in allow]
        if not specs:
            raise SystemExit("No matching specs found for --only")

    results: list[dict[str, Any]] = []
    for spec in specs:
        result = run_spec(spec, args.model, args.valid, args.out_dir, args.threshold)
        results.append(result)
        print(
            f"{result['name']}: size={result['size_mb']}MB "
            f"prec={result['precision']} rec={result['recall']} fpr={result['fpr']}"
        )

    if str(args.results) != "-":
        args.results.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "name",
            "size_mb",
            "precision",
            "recall",
            "f1",
            "fpr",
            "fnr",
            "cutoff",
            "dsub",
            "qout",
            "qnorm",
            "pca_dim",
            "model_path",
        ]
        with open(args.results, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"Saved {len(results)} results")
    if str(args.results) != "-":
        print(f"CSV: {args.results}")


if __name__ == "__main__":
    main()
