#!/usr/bin/env python3
"""Export distilled tiny student to ONNX and validate torch/ORT parity."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast

from student_runtime import load_student_from_dir

from transformer_common import (
    DATA_DIR,
    MODELS_DIR,
    assert_tokenizer_sanity,
    decision_from_probs,
    load_prepared_rows,
    sigmoid,
    softmax,
)

DEFAULT_STUDENT_DIR = MODELS_DIR / "student"
DEFAULT_TRAIN = DATA_DIR / "transformer" / "train.prepared.jsonl"
DEFAULT_VALID = DATA_DIR / "transformer" / "valid.prepared.jsonl"
DEFAULT_OUT = MODELS_DIR / "student.onnx"


class EvalDataset(Dataset):
    def __init__(self, rows, tokenizer: BertTokenizerFast, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        enc = self.tokenizer(
            row.text_normalized,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_attention_mask=True,
        )
        return {
            "collapsed_label": row.collapsed_label,
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        }


def collate(batch: list[dict]) -> dict:
    return {
        "collapsed_label": [x["collapsed_label"] for x in batch],
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-dir", type=Path, default=DEFAULT_STUDENT_DIR)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--parity-samples", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--max-unk-ratio", type=float, default=0.05)
    parser.add_argument("--tokenizer-sanity-sample-size", type=int, default=512)
    parser.add_argument("--max-mean-delta", type=float, default=0.01)
    parser.add_argument("--min-label-agreement", type=float, default=0.99)
    args = parser.parse_args()

    for path in (args.student_dir, args.train, args.valid):
        if not Path(path).exists():
            raise SystemExit(f"Missing required input: {path}")
    if args.parity_samples < 1000:
        raise SystemExit(
            f"parity-samples must be >= 1000 per plan acceptance criteria, got {args.parity_samples}."
        )

    model, tokenizer, payload = load_student_from_dir(args.student_dir)
    arch = payload["architecture"]
    thresholds = payload.get("thresholds", {"scam": 0.5, "topic_crypto": 0.5})
    max_length = int(arch.get("max_length", args.max_length))

    rows = load_prepared_rows(args.valid) + load_prepared_rows(args.train)
    tokenizer_stats = assert_tokenizer_sanity(
        tokenizer=tokenizer,
        expected_vocab_size=int(arch["vocab_size"]),
        context="export_transformer_student_onnx.py",
        sample_texts=[row.text_normalized for row in rows],
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

    dummy_input_ids = torch.ones((1, max_length), dtype=torch.long)
    dummy_attention = torch.ones((1, max_length), dtype=torch.long)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention),
        str(args.out),
        input_names=["input_ids", "attention_mask"],
        output_names=["scam_logits", "topic_logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "scam_logits": {0: "batch"},
            "topic_logits": {0: "batch"},
        },
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"Exported ONNX model to {args.out}")

    if len(rows) < args.parity_samples:
        raise SystemExit(
            f"Need at least {args.parity_samples} samples for parity, but only found {len(rows)}."
        )
    rows = rows[: args.parity_samples]
    dataset = EvalDataset(rows, tokenizer=tokenizer, max_length=max_length)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )

    ort_session = ort.InferenceSession(
        str(args.out), providers=["CPUExecutionProvider"]
    )

    deltas: list[float] = []
    matches = 0
    total = 0

    scam_thr = float(thresholds.get("scam", 0.5))
    topic_thr = float(thresholds.get("topic_crypto", 0.5))

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"]
            attention = batch["attention_mask"]

            torch_scam, torch_topic = model(
                input_ids=input_ids, attention_mask=attention
            )
            torch_scam_prob = softmax(torch_scam.numpy())[:, 1]
            torch_topic_prob = sigmoid(torch_topic.numpy().reshape(-1))

            ort_out = ort_session.run(
                ["scam_logits", "topic_logits"],
                {
                    "input_ids": input_ids.numpy(),
                    "attention_mask": attention.numpy(),
                },
            )
            ort_scam_prob = softmax(ort_out[0])[:, 1]
            ort_topic_prob = sigmoid(ort_out[1].reshape(-1))

            delta = np.abs(torch_scam_prob - ort_scam_prob) + np.abs(
                torch_topic_prob - ort_topic_prob
            )
            deltas.extend((delta / 2.0).tolist())

            for ts, tt, os, ot in zip(
                torch_scam_prob, torch_topic_prob, ort_scam_prob, ort_topic_prob
            ):
                torch_label = decision_from_probs(
                    float(ts),
                    float(tt),
                    scam_threshold=scam_thr,
                    topic_threshold=topic_thr,
                )
                ort_label = decision_from_probs(
                    float(os),
                    float(ot),
                    scam_threshold=scam_thr,
                    topic_threshold=topic_thr,
                )
                matches += int(torch_label == ort_label)
                total += 1

    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    label_agreement = float(matches / total) if total else 1.0

    print(f"Parity mean abs prob delta: {mean_delta:.6f}")
    print(f"Parity label agreement: {label_agreement:.4%} ({matches}/{total})")

    if mean_delta > args.max_mean_delta:
        raise SystemExit(
            f"Parity check failed: mean abs probability delta {mean_delta:.6f} > {args.max_mean_delta:.6f}"
        )
    if label_agreement < args.min_label_agreement:
        raise SystemExit(
            f"Parity check failed: label agreement {label_agreement:.4%} < {args.min_label_agreement:.4%}"
        )


if __name__ == "__main__":
    main()
