#!/usr/bin/env python3
"""Evaluate tiny-transformer model (torch or ONNX) on Janitr splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast

from student_runtime import TinyStudentModel, load_student_from_dir

from transformer_common import (
    CONFIG_DIR,
    DATA_DIR,
    MODELS_DIR,
    TRAINING_CLASSES,
    PreparedRecord,
    assert_tokenizer_sanity,
    binary_pr_auc,
    brier_score,
    calibration_bins,
    expected_calibration_error,
    load_prepared_rows,
    predict_labels_from_probs,
    save_json,
    sigmoid,
    softmax,
    summarize_label_predictions,
    tune_thresholds_for_scam_fpr,
)

DEFAULT_STUDENT_DIR = MODELS_DIR / "student"
DEFAULT_VALID = DATA_DIR / "transformer" / "valid.prepared.jsonl"
DEFAULT_HOLDOUT = DATA_DIR / "transformer" / "holdout.prepared.jsonl"
DEFAULT_TRAIN = DATA_DIR / "transformer" / "train.prepared.jsonl"
DEFAULT_THRESHOLD_OUT = CONFIG_DIR / "thresholds.transformer.json"


class EvalDataset(Dataset):
    def __init__(
        self, rows: list[PreparedRecord], tokenizer: BertTokenizerFast, max_length: int
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        enc = self.tokenizer(
            row.text_normalized,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_attention_mask=True,
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        }


def collate(batch: list[dict]) -> dict:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
    }


def infer_probs_torch(
    rows: list[PreparedRecord],
    model: TinyStudentModel,
    tokenizer: BertTokenizerFast,
    *,
    max_length: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    ds = EvalDataset(rows, tokenizer=tokenizer, max_length=max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    use_amp = device.type == "cuda"

    scam_probs: list[float] = []
    topic_probs: list[float] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention = batch["attention_mask"].to(device)
            with torch.autocast(
                device_type=device.type, enabled=use_amp, dtype=torch.bfloat16
            ):
                scam_logits, topic_logits = model(
                    input_ids=input_ids, attention_mask=attention
                )
            scam_prob = softmax(scam_logits.detach().cpu().float().numpy())[:, 1]
            topic_prob = sigmoid(
                topic_logits.detach().cpu().float().numpy().reshape(-1)
            )
            scam_probs.extend(scam_prob.tolist())
            topic_probs.extend(topic_prob.tolist())

    return np.array(scam_probs, dtype=np.float64), np.array(
        topic_probs, dtype=np.float64
    )


def infer_probs_onnx(
    rows: list[PreparedRecord],
    onnx_path: Path,
    tokenizer: BertTokenizerFast,
    *,
    max_length: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    ds = EvalDataset(rows, tokenizer=tokenizer, max_length=max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)

    scam_probs: list[float] = []
    topic_probs: list[float] = []
    for batch in loader:
        out = session.run(
            ["scam_logits", "topic_logits"],
            {
                "input_ids": batch["input_ids"].numpy(),
                "attention_mask": batch["attention_mask"].numpy(),
            },
        )
        scam_prob = softmax(out[0])[:, 1]
        topic_prob = sigmoid(out[1].reshape(-1))
        scam_probs.extend(scam_prob.tolist())
        topic_probs.extend(topic_prob.tolist())
    return np.array(scam_probs, dtype=np.float64), np.array(
        topic_probs, dtype=np.float64
    )


def compute_metrics(
    *,
    rows: list[PreparedRecord],
    scam_probs: np.ndarray,
    topic_probs: np.ndarray,
    scam_threshold: float,
    topic_threshold: float,
    train_handles: set[str],
) -> dict:
    y_true = [row.collapsed_label for row in rows]
    preds = predict_labels_from_probs(
        scam_probs,
        topic_probs,
        scam_threshold=scam_threshold,
        topic_threshold=topic_threshold,
    )
    summary = summarize_label_predictions(y_true, preds, classes=TRAINING_CLASSES)

    y_scam = [1 if label == "scam" else 0 for label in y_true]
    scam_auc = binary_pr_auc(y_scam, scam_probs.tolist())
    scam_bins = calibration_bins(y_scam, scam_probs.tolist(), bins=10)

    metrics = {
        **summary,
        "pr_auc": {
            "scam": scam_auc,
        },
        "calibration": {
            "scam_bins": scam_bins,
            "scam_ece": expected_calibration_error(scam_bins),
            "scam_brier": brier_score(y_scam, scam_probs.tolist()),
        },
    }

    subgroup_masks = {
        "short_posts_lt_40": [len(row.text_normalized) < 40 for row in rows],
        "with_url": [row.has_url for row in rows],
        "without_url": [not row.has_url for row in rows],
        "seen_handles": [
            (row.author_handle in train_handles)
            if row.author_handle is not None
            else False
            for row in rows
        ],
        "unseen_handles": [
            (row.author_handle not in train_handles)
            if row.author_handle is not None
            else True
            for row in rows
        ],
    }

    subgroup_metrics: dict[str, dict] = {}
    for name, mask in subgroup_masks.items():
        idxs = [idx for idx, flag in enumerate(mask) if flag]
        if not idxs:
            subgroup_metrics[name] = {
                "samples": 0,
                "metrics": None,
            }
            continue

        sub_rows = [rows[idx] for idx in idxs]
        sub_scam = scam_probs[idxs]
        sub_topic = topic_probs[idxs]

        sub_y_true = [row.collapsed_label for row in sub_rows]
        sub_preds = predict_labels_from_probs(
            sub_scam,
            sub_topic,
            scam_threshold=scam_threshold,
            topic_threshold=topic_threshold,
        )
        sub_summary = summarize_label_predictions(
            sub_y_true, sub_preds, classes=TRAINING_CLASSES
        )
        subgroup_metrics[name] = {
            "samples": len(sub_rows),
            "metrics": sub_summary["metrics"],
        }

    metrics["subgroups"] = subgroup_metrics
    return metrics


def assert_student_provenance(config_payload: dict, *, allow_missing: bool) -> None:
    source = config_payload.get("source_artifacts")
    if source is None:
        if allow_missing:
            return
        raise SystemExit(
            "student_config.json is missing source_artifacts provenance. "
            "Re-train student with current distillation pipeline or pass --allow-missing-provenance."
        )
    required = {
        "teacher_id",
        "calibration_id",
        "logits_cache_train_id",
        "logits_cache_valid_id",
        "seeds",
        "label_map_hash",
        "split_hashes",
    }
    missing = sorted(required - set(source.keys()))
    if missing:
        raise SystemExit(f"student_config source_artifacts missing keys: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-dir", type=Path, default=DEFAULT_STUDENT_DIR)
    parser.add_argument("--valid", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument(
        "--onnx", type=Path, default=None, help="If set, evaluate this ONNX model"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--max-unk-ratio", type=float, default=0.05)
    parser.add_argument("--tokenizer-sanity-sample-size", type=int, default=512)
    parser.add_argument("--target-scam-fpr", type=float, default=0.02)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--allow-missing-provenance", action="store_true")
    parser.add_argument(
        "--thresholds-out",
        type=Path,
        default=DEFAULT_THRESHOLD_OUT,
        help="Write tuned thresholds JSON",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=MODELS_DIR / "student_holdout_eval.json",
    )
    args = parser.parse_args()

    for path in (args.student_dir, args.valid, args.holdout, args.train):
        if not path.exists():
            raise SystemExit(f"Missing required path: {path}")

    valid_rows = load_prepared_rows(args.valid)
    holdout_rows = load_prepared_rows(args.holdout)
    train_rows = load_prepared_rows(args.train)
    train_handles = {row.author_handle for row in train_rows if row.author_handle}

    model, tokenizer, student_cfg = load_student_from_dir(args.student_dir)
    assert_student_provenance(student_cfg, allow_missing=args.allow_missing_provenance)
    max_length = int(student_cfg["architecture"].get("max_length", args.max_length))
    tokenizer_stats = assert_tokenizer_sanity(
        tokenizer=tokenizer,
        expected_vocab_size=int(student_cfg["architecture"]["vocab_size"]),
        context="evaluate_transformer.py",
        sample_texts=[row.text_normalized for row in train_rows],
        max_length=max_length,
        max_unk_ratio=args.max_unk_ratio,
        sample_size=args.tokenizer_sanity_sample_size,
    )
    print(
        "Tokenizer sanity: "
        f"backend_vocab={tokenizer_stats['backend_vocab_size']} "
        f"len={tokenizer_stats['loaded_vocab_size']} "
        f"unk_ratio={float(tokenizer_stats.get('sample_unk_ratio', 0.0)):.4f} "
        f"sample_count={int(tokenizer_stats.get('sample_count', 0))}"
    )

    if args.onnx is None:
        valid_scam, valid_topic = infer_probs_torch(
            valid_rows,
            model,
            tokenizer,
            max_length=max_length,
            batch_size=args.batch_size,
        )
        holdout_scam, holdout_topic = infer_probs_torch(
            holdout_rows,
            model,
            tokenizer,
            max_length=max_length,
            batch_size=args.batch_size,
        )
        engine = "torch"
    else:
        if not args.onnx.exists():
            raise SystemExit(f"ONNX model not found: {args.onnx}")
        valid_scam, valid_topic = infer_probs_onnx(
            valid_rows,
            onnx_path=args.onnx,
            tokenizer=tokenizer,
            max_length=max_length,
            batch_size=args.batch_size,
        )
        holdout_scam, holdout_topic = infer_probs_onnx(
            holdout_rows,
            onnx_path=args.onnx,
            tokenizer=tokenizer,
            max_length=max_length,
            batch_size=args.batch_size,
        )
        engine = "onnx"

    scam_thr, topic_thr, valid_threshold_metrics = tune_thresholds_for_scam_fpr(
        y_true=[row.collapsed_label for row in valid_rows],
        scam_probs=valid_scam,
        topic_probs=valid_topic,
        target_scam_fpr=args.target_scam_fpr,
        step=args.threshold_step,
        classes=TRAINING_CLASSES,
    )

    threshold_payload = {
        "version": 1,
        "engine": engine,
        "thresholds": {
            "scam": scam_thr,
            "topic_crypto": topic_thr,
        },
        "tune_target_scam_fpr": args.target_scam_fpr,
        "valid_metrics_at_threshold": valid_threshold_metrics,
    }
    save_json(args.thresholds_out, threshold_payload)

    holdout_metrics = compute_metrics(
        rows=holdout_rows,
        scam_probs=holdout_scam,
        topic_probs=holdout_topic,
        scam_threshold=scam_thr,
        topic_threshold=topic_thr,
        train_handles=train_handles,
    )

    report = {
        "engine": engine,
        "onnx_path": str(args.onnx) if args.onnx else None,
        "thresholds": threshold_payload,
        "holdout": holdout_metrics,
    }
    save_json(args.out, report)

    scam = holdout_metrics["metrics"]["scam"]
    print(f"Engine: {engine}")
    print(f"Thresholds: scam={scam_thr:.4f}, topic_crypto={topic_thr:.4f}")
    print(
        "Holdout scam metrics: "
        f"precision={scam['precision']:.4f} recall={scam['recall']:.4f} fpr={scam['fpr']:.4f}"
    )
    print(f"Wrote thresholds to {args.thresholds_out}")
    print(f"Wrote evaluation report to {args.out}")


if __name__ == "__main__":
    main()
