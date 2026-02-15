#!/usr/bin/env python3
"""Shared helpers for Janitr tiny-transformer training/evaluation scripts."""

from __future__ import annotations

import json
import math
import random
import re
import subprocess
import unicodedata
from datetime import datetime, timezone
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
CONFIG_DIR = REPO_ROOT / "config"

TRAINING_CLASSES = ["clean", "topic_crypto", "scam"]
TOPIC_LABELS = ["topic_crypto"]
SCAM_CLEAN_CLASSES = ["clean", "scam"]

SCAM_RAW_LABELS = {
    "phishing",
    "malware",
    "fake_support",
    "recovery_scam",
    "job_scam",
    "romance_scam",
    "impersonation",
    "account_compromise",
    "spam",
    "reply_spam",
    "dm_spam",
    "promo",
    "affiliate",
    "lead_gen",
    "engagement_bait",
    "follow_train",
    "giveaway",
    "bot",
    "scam",
    "crypto_scam",
}

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class PreparedRecord:
    id: str
    text: str
    text_normalized: str
    labels: list[str]
    raw_labels: list[str]
    collapsed_label: str
    y_scam_clean: int
    y_topics: list[int]
    has_url: bool
    author_handle: str | None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def require_cuda(*, context: str) -> "Any":
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            f"{context} requires torch with CUDA support installed."
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit(
            f"{context} requires CUDA, but torch.cuda.is_available() is False."
        )

    device = torch.device("cuda")
    dev_name = torch.cuda.get_device_name(device)
    print(
        f"{context}: using CUDA device '{dev_name}' (torch={torch.__version__}, cuda={torch.version.cuda})"
    )
    return device


def clean_text(
    text: str | None,
    *,
    normalize: bool = True,
    lowercase: bool = True,
    strip_urls: bool = False,
) -> str:
    if not text:
        return ""
    out = text
    if normalize:
        out = unicodedata.normalize("NFKC", out)
    out = ZERO_WIDTH_RE.sub("", out)
    if strip_urls:
        out = URL_RE.sub(" ", out)
    if lowercase:
        out = out.lower()
    out = WHITESPACE_RE.sub(" ", out).strip()
    return out


def normalize_label(label: str) -> str:
    return label.strip().lower()


def extract_raw_labels(sample: dict[str, Any]) -> list[str]:
    raw: list[str] = []
    labels_value = sample.get("labels")
    if isinstance(labels_value, list):
        for item in labels_value:
            if isinstance(item, str):
                norm = normalize_label(item)
                if norm:
                    raw.append(norm)
    label_value = sample.get("label")
    if isinstance(label_value, str):
        norm = normalize_label(label_value)
        if norm:
            raw.append(norm)
    # Keep first occurrence order.
    seen: set[str] = set()
    deduped: list[str] = []
    for label in raw:
        if label in seen:
            continue
        seen.add(label)
        deduped.append(label)
    return deduped


def collapse_training_targets(raw_labels: Sequence[str]) -> tuple[str, int, list[int]]:
    raw_set = set(raw_labels)
    is_scam = bool(raw_set & SCAM_RAW_LABELS)
    topic_bits = [
        1 if topic in raw_set or topic.replace("topic_", "") in raw_set else 0
        for topic in TOPIC_LABELS
    ]

    if is_scam:
        collapsed = "scam"
    elif any(topic_bits):
        collapsed = "topic_crypto"
    else:
        collapsed = "clean"

    y_scam_clean = 1 if collapsed == "scam" else 0
    return collapsed, y_scam_clean, topic_bits


def extract_author_handle(sample: dict[str, Any]) -> str | None:
    for key in ("authorHandle", "author_handle", "handle", "author", "username"):
        value = sample.get(key)
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value.lower()
    return None


def record_from_sample(
    sample: dict[str, Any],
    *,
    normalize: bool = True,
    lowercase: bool = True,
    strip_urls: bool = False,
) -> PreparedRecord | None:
    text_raw = sample.get("text") or sample.get("raw_text") or ""
    if not isinstance(text_raw, str):
        text_raw = str(text_raw)

    text_normalized = clean_text(
        text_raw,
        normalize=normalize,
        lowercase=lowercase,
        strip_urls=strip_urls,
    )
    if not text_normalized:
        return None

    raw_labels = extract_raw_labels(sample)
    collapsed, y_scam_clean, y_topics = collapse_training_targets(raw_labels)
    sample_id = sample.get("id")
    if not isinstance(sample_id, str) or not sample_id:
        sample_id = f"row_{abs(hash(text_normalized))}"

    has_url = bool(URL_RE.search(text_raw))
    return PreparedRecord(
        id=sample_id,
        text=text_raw,
        text_normalized=text_normalized,
        labels=[collapsed],
        raw_labels=raw_labels,
        collapsed_label=collapsed,
        y_scam_clean=y_scam_clean,
        y_topics=y_topics,
        has_url=has_url,
        author_handle=extract_author_handle(sample),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_prepared_rows(path: Path) -> list[PreparedRecord]:
    rows: list[PreparedRecord] = []
    for payload in load_jsonl(path):
        rows.append(
            PreparedRecord(
                id=str(payload["id"]),
                text=str(payload.get("text", "")),
                text_normalized=str(
                    payload.get("text_normalized") or payload.get("text", "")
                ),
                labels=list(payload.get("labels", [])),
                raw_labels=list(payload.get("raw_labels", [])),
                collapsed_label=str(payload["collapsed_label"]),
                y_scam_clean=int(payload["y_scam_clean"]),
                y_topics=[int(x) for x in payload.get("y_topics", [0])],
                has_url=bool(payload.get("has_url", False)),
                author_handle=(
                    str(payload["author_handle"]).lower()
                    if payload.get("author_handle")
                    else None
                ),
            )
        )
    return rows


def softmax(logits: np.ndarray) -> np.ndarray:
    shift = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shift)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def decision_from_probs(
    scam_prob: float,
    topic_prob: float,
    *,
    scam_threshold: float,
    topic_threshold: float,
) -> str:
    if scam_prob >= scam_threshold:
        return "scam"
    if topic_prob >= topic_threshold:
        return "topic_crypto"
    return "clean"


def one_vs_all_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    classes: Sequence[str] = TRAINING_CLASSES,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    total = len(y_true)
    for cls in classes:
        tp = fp = fn = support = 0
        for gold, pred in zip(y_true, y_pred):
            gold_pos = gold == cls
            pred_pos = pred == cls
            if gold_pos:
                support += 1
            if gold_pos and pred_pos:
                tp += 1
            elif (not gold_pos) and pred_pos:
                fp += 1
            elif gold_pos and (not pred_pos):
                fn += 1

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        negatives = total - support
        fpr = safe_div(fp, negatives)
        fnr = safe_div(fn, support)
        metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "fnr": fnr,
            "support": float(support),
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
        }
    return metrics


def micro_macro_from_metrics(
    metrics: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    tps = sum(m["tp"] for m in metrics.values())
    fps = sum(m["fp"] for m in metrics.values())
    fns = sum(m["fn"] for m in metrics.values())

    micro_precision = safe_div(tps, tps + fps)
    micro_recall = safe_div(tps, tps + fns)
    micro_f1 = safe_div(
        2 * micro_precision * micro_recall, micro_precision + micro_recall
    )

    macro_precision = safe_div(
        sum(m["precision"] for m in metrics.values()), len(metrics)
    )
    macro_recall = safe_div(sum(m["recall"] for m in metrics.values()), len(metrics))
    macro_f1 = safe_div(sum(m["f1"] for m in metrics.values()), len(metrics))

    return {
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
        },
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
    }


def exact_match_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for gold, pred in zip(y_true, y_pred) if gold == pred) / len(y_true)


def format_one_vs_all_metrics(
    per_class: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        label: {
            "precision": vals["precision"],
            "recall": vals["recall"],
            "f1": vals["f1"],
            "fpr": vals["fpr"],
            "fnr": vals["fnr"],
            "support": vals["support"],
        }
        for label, vals in per_class.items()
    }


def summarize_label_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    classes: Sequence[str] = TRAINING_CLASSES,
) -> dict[str, Any]:
    per_class = one_vs_all_metrics(y_true, y_pred, classes=classes)
    mm = micro_macro_from_metrics(per_class)
    return {
        "metrics": format_one_vs_all_metrics(per_class),
        "exact_match": exact_match_accuracy(y_true, y_pred),
        "micro": mm["micro"],
        "macro": mm["macro"],
    }


def predict_labels_from_probs(
    scam_probs: Sequence[float] | np.ndarray,
    topic_probs: Sequence[float] | np.ndarray,
    *,
    scam_threshold: float,
    topic_threshold: float,
) -> list[str]:
    preds: list[str] = []
    for s_prob, t_prob in zip(scam_probs, topic_probs):
        preds.append(
            decision_from_probs(
                float(s_prob),
                float(t_prob),
                scam_threshold=scam_threshold,
                topic_threshold=topic_threshold,
            )
        )
    return preds


def tune_thresholds_for_scam_fpr(
    *,
    y_true: Sequence[str],
    scam_probs: Sequence[float] | np.ndarray,
    topic_probs: Sequence[float] | np.ndarray,
    target_scam_fpr: float,
    step: float,
    classes: Sequence[str] = TRAINING_CLASSES,
) -> tuple[float, float, dict[str, Any]]:
    best: tuple[float, float, float, float, float, dict[str, Any]] | None = None

    scam_arr = np.asarray(scam_probs, dtype=np.float64)
    topic_arr = np.asarray(topic_probs, dtype=np.float64)
    scam_grid = np.arange(0.5, 1.0001, step)
    topic_grid = np.arange(0.5, 1.0001, step)

    for scam_thr in scam_grid:
        for topic_thr in topic_grid:
            preds = predict_labels_from_probs(
                scam_arr,
                topic_arr,
                scam_threshold=float(scam_thr),
                topic_threshold=float(topic_thr),
            )
            summary = summarize_label_predictions(y_true, preds, classes=classes)
            scam_metrics = summary["metrics"]["scam"]
            if float(scam_metrics["fpr"]) > target_scam_fpr:
                continue
            candidate = (
                float(scam_metrics["recall"]),
                float(scam_metrics["precision"]),
                float(summary["macro"]["f1"]),
                float(scam_thr),
                float(topic_thr),
                summary,
            )
            if best is None or candidate[:3] > best[:3]:
                best = candidate

    if best is not None:
        return best[3], best[4], best[5]

    # Conservative fallback when no pair satisfies target FPR.
    scam_thr = 0.99
    topic_thr = 0.99
    preds = predict_labels_from_probs(
        scam_arr,
        topic_arr,
        scam_threshold=scam_thr,
        topic_threshold=topic_thr,
    )
    summary = summarize_label_predictions(y_true, preds, classes=classes)
    summary["note"] = (
        "No thresholds met target scam FPR; used conservative fallback 0.99/0.99."
    )
    return scam_thr, topic_thr, summary


def binary_pr_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    # Hand-rolled AUPRC to avoid extra runtime dependencies.
    if not y_true:
        return 0.0
    positives = sum(y_true)
    if positives == 0:
        return 0.0

    pairs = sorted(zip(scores, y_true), key=lambda p: p[0], reverse=True)
    tp = 0
    fp = 0
    prev_score = math.inf
    points: list[tuple[float, float]] = [(0.0, 1.0)]

    for score, label in pairs:
        if score != prev_score:
            precision = safe_div(tp, tp + fp)
            recall = safe_div(tp, positives)
            points.append((recall, precision))
            prev_score = score
        if label:
            tp += 1
        else:
            fp += 1

    points.append((safe_div(tp, positives), safe_div(tp, tp + fp)))
    points.sort(key=lambda p: p[0])

    auc = 0.0
    for (r0, p0), (r1, p1) in zip(points, points[1:]):
        auc += (r1 - r0) * ((p0 + p1) / 2)
    return auc


def calibration_bins(
    y_true: Sequence[int],
    conf: Sequence[float],
    *,
    bins: int = 10,
) -> list[dict[str, float]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    out: list[dict[str, float]] = []

    y_arr = np.array(list(y_true), dtype=np.float64)
    c_arr = np.array(list(conf), dtype=np.float64)
    for idx in range(bins):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == bins - 1:
            mask = (c_arr >= lo) & (c_arr <= hi)
        else:
            mask = (c_arr >= lo) & (c_arr < hi)
        count = int(mask.sum())
        if count == 0:
            out.append(
                {
                    "bin": float(idx),
                    "lo": float(lo),
                    "hi": float(hi),
                    "count": 0.0,
                    "mean_conf": 0.0,
                    "empirical_acc": 0.0,
                }
            )
            continue
        mean_conf = float(c_arr[mask].mean())
        empirical_acc = float(y_arr[mask].mean())
        out.append(
            {
                "bin": float(idx),
                "lo": float(lo),
                "hi": float(hi),
                "count": float(count),
                "mean_conf": mean_conf,
                "empirical_acc": empirical_acc,
            }
        )
    return out


def expected_calibration_error(calib_bins: Sequence[dict[str, float]]) -> float:
    total = sum(item["count"] for item in calib_bins)
    if total == 0:
        return 0.0
    ece = 0.0
    for item in calib_bins:
        weight = item["count"] / total
        ece += weight * abs(item["mean_conf"] - item["empirical_acc"])
    return ece


def brier_score(y_true: Sequence[int], prob: Sequence[float]) -> float:
    if not y_true:
        return 0.0
    y_arr = np.array(list(y_true), dtype=np.float64)
    p_arr = np.array(list(prob), dtype=np.float64)
    return float(np.mean((p_arr - y_arr) ** 2))


def choose_dtype_args(dtype: str) -> dict[str, bool]:
    dt = dtype.lower()
    return {
        "fp16": dt == "fp16",
        "bf16": dt == "bf16",
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stable_object_hash(payload: Any) -> str:
    blob = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def current_git_commit(repo_root: Path = REPO_ROOT) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = proc.stdout.strip()
    return commit or None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_label_map(labels: Sequence[str]) -> str:
    payload = {"labels": [str(label) for label in labels]}
    return stable_object_hash(payload)


def hash_prepared_rows(rows: Sequence[PreparedRecord]) -> str:
    payload = [
        {
            "id": row.id,
            "text_normalized": row.text_normalized,
            "collapsed_label": row.collapsed_label,
            "y_scam_clean": int(row.y_scam_clean),
            "y_topics": [int(v) for v in row.y_topics],
        }
        for row in rows
    ]
    return stable_object_hash(payload)


def parse_seed_csv(seed_csv: str) -> list[int]:
    seeds = [int(item.strip()) for item in seed_csv.split(",") if item.strip()]
    if not seeds:
        raise ValueError("No seeds provided.")
    return seeds


def tokenizer_backend_vocab_size(tokenizer: Any) -> int:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        get_vocab_size = getattr(backend, "get_vocab_size", None)
        if callable(get_vocab_size):
            try:
                return int(get_vocab_size(with_added_tokens=False))
            except TypeError:
                return int(get_vocab_size(False))
    return int(getattr(tokenizer, "vocab_size", len(tokenizer)))


def tokenizer_sample_unk_ratio(
    tokenizer: Any,
    sample_texts: Sequence[str],
    *,
    max_length: int,
    sample_size: int,
) -> tuple[float, int]:
    if not sample_texts:
        return 0.0, 0

    take = min(len(sample_texts), sample_size)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is None:
        return 0.0, take

    special_ids = {
        token_id
        for token_id in (
            getattr(tokenizer, "pad_token_id", None),
            getattr(tokenizer, "cls_token_id", None),
            getattr(tokenizer, "sep_token_id", None),
            getattr(tokenizer, "mask_token_id", None),
        )
        if token_id is not None
    }

    unk_count = 0
    total_count = 0
    for text in sample_texts[:take]:
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_attention_mask=False,
        )
        input_ids = enc.get("input_ids", [])
        for token_id in input_ids:
            if token_id in special_ids:
                continue
            total_count += 1
            if token_id == unk_id:
                unk_count += 1

    if total_count == 0:
        return 0.0, take
    return float(unk_count / total_count), take


def assert_tokenizer_sanity(
    *,
    tokenizer: Any,
    expected_vocab_size: int,
    context: str,
    sample_texts: Sequence[str] | None = None,
    max_length: int = 96,
    max_unk_ratio: float = 0.05,
    sample_size: int = 512,
) -> dict[str, float | int]:
    backend_vocab = tokenizer_backend_vocab_size(tokenizer)
    loaded_vocab = int(len(tokenizer))

    if backend_vocab != expected_vocab_size:
        raise SystemExit(
            f"{context}: tokenizer backend vocab size {backend_vocab} != expected {expected_vocab_size} "
            f"(len={loaded_vocab}). This usually indicates incorrect tokenizer loading."
        )

    stats: dict[str, float | int] = {
        "backend_vocab_size": backend_vocab,
        "loaded_vocab_size": loaded_vocab,
    }

    if sample_texts is not None:
        unk_ratio, sample_count = tokenizer_sample_unk_ratio(
            tokenizer,
            sample_texts,
            max_length=max_length,
            sample_size=sample_size,
        )
        stats["sample_count"] = sample_count
        stats["sample_unk_ratio"] = unk_ratio
        if unk_ratio > max_unk_ratio:
            raise SystemExit(
                f"{context}: tokenizer UNK ratio {unk_ratio:.4f} exceeds max {max_unk_ratio:.4f} "
                f"on sample_size={sample_count}."
            )

    return stats
