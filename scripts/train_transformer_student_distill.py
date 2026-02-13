#!/usr/bin/env python3
"""Distill calibrated teacher outputs into a tiny student transformer."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import BertWordPieceTokenizer
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import BertConfig, BertModel, BertTokenizerFast

from transformer_common import (
    DATA_DIR,
    MODELS_DIR,
    TRAINING_CLASSES,
    PreparedRecord,
    decision_from_probs,
    exact_match_accuracy,
    load_prepared_rows,
    micro_macro_from_metrics,
    one_vs_all_metrics,
    require_cuda,
    save_json,
    set_seed,
    sigmoid,
    softmax,
)

DEFAULT_TRAIN = DATA_DIR / "transformer" / "train.prepared.jsonl"
DEFAULT_VALID = DATA_DIR / "transformer" / "valid.prepared.jsonl"
DEFAULT_HOLDOUT = DATA_DIR / "transformer" / "holdout.prepared.jsonl"
DEFAULT_CACHE_TRAIN = MODELS_DIR / "teacher_logits_train.npz"
DEFAULT_CACHE_VALID = MODELS_DIR / "teacher_logits_valid.npz"
DEFAULT_OUT_DIR = MODELS_DIR / "student"

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


class TinyStudentModel(nn.Module):
    def __init__(self, config: BertConfig, teacher_hidden_size: int) -> None:
        super().__init__()
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.head_scam_clean = nn.Linear(config.hidden_size, 2)
        self.head_topic = nn.Linear(config.hidden_size, 1)
        self.teacher_projections = nn.ModuleList(
            [nn.Linear(teacher_hidden_size, config.hidden_size) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> dict:
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        cls = self.dropout(out.last_hidden_state[:, 0])
        scam_logits = self.head_scam_clean(cls)
        topic_logits = self.head_topic(cls).squeeze(-1)
        return {
            "scam_logits": scam_logits,
            "topic_logits": topic_logits,
            "hidden_states": out.hidden_states if output_hidden_states else None,
        }


class DistillTrainDataset(Dataset):
    def __init__(
        self,
        rows: list[PreparedRecord],
        tokenizer: BertTokenizerFast,
        max_length: int,
        cache: dict[str, np.ndarray],
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

        ids = cache["ids"].tolist()
        self.id_to_idx = {str(sample_id): idx for idx, sample_id in enumerate(ids)}
        self.cache = cache

        missing = [row.id for row in rows if row.id not in self.id_to_idx]
        if missing:
            raise RuntimeError(
                f"{len(missing)} rows from prepared split are missing in teacher cache; first={missing[0]}"
            )

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
        cache_idx = self.id_to_idx[row.id]

        return {
            "id": row.id,
            "collapsed_label": row.collapsed_label,
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "y_scam_clean": torch.tensor(row.y_scam_clean, dtype=torch.long),
            "y_topic": torch.tensor(float(row.y_topics[0]), dtype=torch.float32),
            "teacher_scam_logits": torch.tensor(
                self.cache["scam_logits_cal"][cache_idx], dtype=torch.float32
            ),
            "teacher_topic_logit": torch.tensor(
                float(self.cache["topic_logits_cal"][cache_idx]), dtype=torch.float32
            ),
            "teacher_hidden_cls": torch.tensor(
                self.cache["teacher_hidden_cls"][cache_idx], dtype=torch.float32
            ),
        }


class EvalDataset(Dataset):
    def __init__(self, rows: list[PreparedRecord], tokenizer: BertTokenizerFast, max_length: int) -> None:
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
            "id": row.id,
            "collapsed_label": row.collapsed_label,
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "y_scam_clean": torch.tensor(row.y_scam_clean, dtype=torch.long),
            "y_topic": torch.tensor(float(row.y_topics[0]), dtype=torch.float32),
        }


def collate(batch: list[dict]) -> dict:
    out = {
        "id": [x["id"] for x in batch],
        "collapsed_label": [x["collapsed_label"] for x in batch],
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "y_scam_clean": torch.stack([x["y_scam_clean"] for x in batch]),
        "y_topic": torch.stack([x["y_topic"] for x in batch]),
    }
    if "teacher_scam_logits" in batch[0]:
        out["teacher_scam_logits"] = torch.stack([x["teacher_scam_logits"] for x in batch])
        out["teacher_topic_logit"] = torch.stack([x["teacher_topic_logit"] for x in batch])
        out["teacher_hidden_cls"] = torch.stack([x["teacher_hidden_cls"] for x in batch])
    return out


def train_or_load_tokenizer(
    *,
    rows: list[PreparedRecord],
    tokenizer_dir: Path,
    vocab_size: int,
    min_frequency: int,
) -> BertTokenizerFast:
    def ensure_target_vocab(tokenizer: BertTokenizerFast) -> BertTokenizerFast:
        current_size = int(len(tokenizer))
        if current_size > vocab_size:
            raise RuntimeError(
                f"Tokenizer vocab ({current_size}) exceeds requested vocab_size ({vocab_size})."
            )
        if current_size < vocab_size:
            pad_tokens = [f"[UNUSED_{i}]" for i in range(vocab_size - current_size)]
            tokenizer.add_tokens(pad_tokens)
            tokenizer.save_pretrained(str(tokenizer_dir))
        return tokenizer

    vocab_path = tokenizer_dir / "vocab.txt"
    if vocab_path.exists():
        tokenizer = BertTokenizerFast.from_pretrained(str(tokenizer_dir))
        if int(len(tokenizer)) == vocab_size:
            return tokenizer
        print(
            f"Tokenizer at {tokenizer_dir} has vocab={len(tokenizer)} (target={vocab_size}); rebuilding."
        )

    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    wp = BertWordPieceTokenizer(lowercase=True, strip_accents=False, clean_text=False)
    wp.train_from_iterator(
        (row.text_normalized for row in rows),
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    wp.save_model(str(tokenizer_dir))

    tokenizer = BertTokenizerFast(
        vocab_file=str(vocab_path),
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )
    tokenizer.save_pretrained(str(tokenizer_dir))
    return ensure_target_vocab(tokenizer)


def compute_weights(rows: list[PreparedRecord], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    scam_counts = Counter(row.y_scam_clean for row in rows)
    clean_count = max(1, scam_counts.get(0, 0))
    scam_count = max(1, scam_counts.get(1, 0))
    total = clean_count + scam_count

    ce_weight = torch.tensor(
        [total / (2 * clean_count), total / (2 * scam_count)], dtype=torch.float32, device=device
    )

    topic_pos = sum(row.y_topics[0] for row in rows)
    topic_neg = len(rows) - topic_pos
    pos_weight = torch.tensor(
        [float(topic_neg) / max(1.0, float(topic_pos))], dtype=torch.float32, device=device
    )
    return ce_weight, pos_weight


def evaluate_model(
    *,
    model: TinyStudentModel,
    loader: DataLoader,
    device: torch.device,
    scam_threshold: float,
    topic_threshold: float,
    dtype: str,
) -> dict:
    model.eval()
    use_amp = device.type == "cuda" and dtype in {"fp16", "bf16"}
    amp_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16

    y_true: list[str] = []
    y_pred: list[str] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                out = model(input_ids=input_ids, attention_mask=attention_mask)

            scam_logits = out["scam_logits"].detach().cpu().numpy()
            topic_logits = out["topic_logits"].detach().cpu().numpy()
            scam_probs = softmax(scam_logits)[:, 1]
            topic_probs = sigmoid(topic_logits)

            for gold, s_prob, t_prob in zip(batch["collapsed_label"], scam_probs, topic_probs):
                pred = decision_from_probs(
                    float(s_prob),
                    float(t_prob),
                    scam_threshold=scam_threshold,
                    topic_threshold=topic_threshold,
                )
                y_true.append(gold)
                y_pred.append(pred)

    per_class = one_vs_all_metrics(y_true, y_pred, classes=TRAINING_CLASSES)
    mm = micro_macro_from_metrics(per_class)
    return {
        "metrics": {
            label: {
                "precision": v["precision"],
                "recall": v["recall"],
                "f1": v["f1"],
                "fpr": v["fpr"],
                "fnr": v["fnr"],
                "support": v["support"],
            }
            for label, v in per_class.items()
        },
        "exact_match": exact_match_accuracy(y_true, y_pred),
        "micro": mm["micro"],
        "macro": mm["macro"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--cache-train", type=Path, default=DEFAULT_CACHE_TRAIN)
    parser.add_argument("--cache-valid", type=Path, default=DEFAULT_CACHE_VALID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--distill-temp", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.5, help="Hard loss weight")
    parser.add_argument("--hidden-loss-weight", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--min-frequency", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--num-hidden-layers", type=int, default=4)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--scam-threshold", type=float, default=0.5)
    parser.add_argument("--topic-threshold", type=float, default=0.5)
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "fp32"],
        default="bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    for path in (args.train, args.valid, args.holdout, args.cache_train, args.cache_valid):
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    train_rows = load_prepared_rows(args.train)
    valid_rows = load_prepared_rows(args.valid)
    holdout_rows = load_prepared_rows(args.holdout)

    with np.load(args.cache_train, allow_pickle=True) as cache_train_npz:
        cache_train = {key: cache_train_npz[key] for key in cache_train_npz.files}
    teacher_hidden_size = int(cache_train["teacher_hidden_cls"].shape[-1])

    tokenizer_dir = args.output_dir / "tokenizer"
    tokenizer = train_or_load_tokenizer(
        rows=train_rows,
        tokenizer_dir=tokenizer_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    vocab_size_actual = int(len(tokenizer))
    if vocab_size_actual != args.vocab_size:
        raise SystemExit(
            f"Tokenizer vocab size mismatch: expected {args.vocab_size}, got {vocab_size_actual}."
        )
    config = BertConfig(
        vocab_size=vocab_size_actual,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=max(128, args.max_length + 8),
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        type_vocab_size=1,
        pad_token_id=tokenizer.pad_token_id,
    )

    device = require_cuda(context="train_transformer_student_distill.py")

    model = TinyStudentModel(config, teacher_hidden_size=teacher_hidden_size).to(device)

    train_ds = DistillTrainDataset(
        rows=train_rows,
        tokenizer=tokenizer,
        max_length=args.max_length,
        cache=cache_train,
    )
    valid_ds = EvalDataset(valid_rows, tokenizer=tokenizer, max_length=args.max_length)
    holdout_ds = EvalDataset(holdout_rows, tokenizer=tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)
    holdout_loader = DataLoader(
        holdout_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    ce_weight, topic_pos_weight = compute_weights(train_rows, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = device.type == "cuda" and args.dtype in {"fp16", "bf16"}
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and args.dtype == "fp16"))
    amp_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    best_state: dict[str, torch.Tensor] | None = None
    best_valid_macro_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        progress = tqdm(train_loader, desc=f"student epoch {epoch}/{args.epochs}", leave=False)
        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y_scam = batch["y_scam_clean"].to(device)
            y_topic = batch["y_topic"].to(device)

            teacher_scam_logits = batch["teacher_scam_logits"].to(device)
            teacher_topic_logit = batch["teacher_topic_logit"].to(device)
            teacher_hidden = batch["teacher_hidden_cls"].to(device)

            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                student_scam_logits = out["scam_logits"]
                student_topic_logits = out["topic_logits"]

                hard_ce = F.cross_entropy(student_scam_logits, y_scam, weight=ce_weight)
                hard_topic = F.binary_cross_entropy_with_logits(
                    student_topic_logits,
                    y_topic,
                    pos_weight=topic_pos_weight,
                )
                hard_loss = hard_ce + hard_topic

                t = args.distill_temp
                teacher_scam_probs = F.softmax(teacher_scam_logits / t, dim=-1)
                student_scam_log_probs = F.log_softmax(student_scam_logits / t, dim=-1)
                soft_scam = F.kl_div(student_scam_log_probs, teacher_scam_probs, reduction="batchmean") * (
                    t * t
                )

                teacher_topic_probs = torch.sigmoid(teacher_topic_logit / t)
                soft_topic = (
                    F.binary_cross_entropy_with_logits(
                        student_topic_logits / t,
                        teacher_topic_probs,
                    )
                    * (t * t)
                )
                soft_loss = soft_scam + soft_topic

                student_hidden_states = out["hidden_states"]
                hidden_losses: list[torch.Tensor] = []
                for idx in range(config.num_hidden_layers):
                    student_cls = student_hidden_states[idx + 1][:, 0, :]
                    teacher_cls = teacher_hidden[:, idx, :]
                    proj_teacher = model.teacher_projections[idx](teacher_cls)
                    hidden_losses.append(F.mse_loss(student_cls, proj_teacher))
                hidden_loss = torch.stack(hidden_losses).mean()

                loss = (
                    args.alpha * hard_loss
                    + (1.0 - args.alpha) * soft_loss
                    + args.hidden_loss_weight * hidden_loss
                )
                loss_scaled = loss / args.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        valid_metrics = evaluate_model(
            model=model,
            loader=valid_loader,
            device=device,
            scam_threshold=args.scam_threshold,
            topic_threshold=args.topic_threshold,
            dtype=args.dtype,
        )
        valid_macro_f1 = float(valid_metrics["macro"]["f1"])
        print(
            f"epoch={epoch} train_loss={running_loss / max(1, len(train_loader)):.4f} "
            f"valid_macro_f1={valid_macro_f1:.4f}"
        )

        if valid_macro_f1 > best_valid_macro_f1:
            best_valid_macro_f1 = valid_macro_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(str(tokenizer_dir))

    valid_metrics = evaluate_model(
        model=model,
        loader=valid_loader,
        device=device,
        scam_threshold=args.scam_threshold,
        topic_threshold=args.topic_threshold,
        dtype=args.dtype,
    )
    holdout_metrics = evaluate_model(
        model=model,
        loader=holdout_loader,
        device=device,
        scam_threshold=args.scam_threshold,
        topic_threshold=args.topic_threshold,
        dtype=args.dtype,
    )

    config_payload = {
        "architecture": {
            "hidden_size": args.hidden_size,
            "num_hidden_layers": args.num_hidden_layers,
            "num_attention_heads": args.num_attention_heads,
            "intermediate_size": args.intermediate_size,
            "dropout": args.dropout,
            "max_length": args.max_length,
            "vocab_size": vocab_size_actual,
            "teacher_hidden_size": teacher_hidden_size,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "distill_temp": args.distill_temp,
            "alpha": args.alpha,
            "hidden_loss_weight": args.hidden_loss_weight,
            "dtype": args.dtype,
            "seed": args.seed,
        },
        "thresholds": {
            "scam": args.scam_threshold,
            "topic_crypto": args.topic_threshold,
        },
    }
    save_json(args.output_dir / "student_config.json", config_payload)

    eval_payload = {
        "valid": valid_metrics,
        "holdout": holdout_metrics,
        "config": config_payload,
    }
    save_json(args.output_dir / "student_eval.json", eval_payload)
    save_json(MODELS_DIR / "student_eval.json", eval_payload)

    print(f"Saved student model to {args.output_dir}")
    print(f"Saved student eval to {args.output_dir / 'student_eval.json'}")


if __name__ == "__main__":
    main()
