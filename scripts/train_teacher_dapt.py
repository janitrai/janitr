#!/usr/bin/env python3
"""Run optional domain-adaptive pretraining (MLM) for the teacher checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from transformer_common import DATA_DIR, MODELS_DIR, choose_dtype_args, require_cuda, set_seed

DEFAULT_CORPUS = DATA_DIR / "transformer" / "unlabeled_corpus.txt"
DEFAULT_OUTPUT_DIR = MODELS_DIR / "teacher_dapt"
DEFAULT_MODEL = "cardiffnlp/twitter-roberta-large-2022-154m"


class LineDataset(Dataset):
    def __init__(self, lines: list[str], tokenizer, max_length: int) -> None:
        self.encodings = tokenizer(
            lines,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_attention_mask=True,
        )

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx], dtype=torch.long),
        }


def load_lines(path: Path, max_rows: int) -> list[str]:
    rows: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(text)
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=0, help="0 disables max_steps")
    parser.add_argument("--per-device-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--max-rows", type=int, default=0, help="0 keeps all rows")
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "fp32"],
        default="bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    if not args.corpus.exists():
        raise SystemExit(f"Corpus not found: {args.corpus}")

    set_seed(args.seed)
    require_cuda(context="train_teacher_dapt.py")

    lines = load_lines(args.corpus, max_rows=args.max_rows)
    if not lines:
        raise SystemExit("No usable lines found in corpus.")
    print(f"Loaded {len(lines)} MLM lines from {args.corpus}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    use_grad_ckpt = args.gradient_checkpointing and not args.no_gradient_checkpointing
    if use_grad_ckpt:
        model.gradient_checkpointing_enable()

    train_dataset = LineDataset(lines, tokenizer, max_length=args.max_length)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    dtype_args = choose_dtype_args(args.dtype)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        do_train=True,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
        **dtype_args,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved DAPT model to {args.output_dir}")


if __name__ == "__main__":
    main()
