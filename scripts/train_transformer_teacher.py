#!/usr/bin/env python3
"""Fine-tune a transformer teacher with Janitr's dual-head objective."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from run_naming import apply_run_name_template, resolve_run_name
from transformer_common import (
    DATA_DIR,
    MODELS_DIR,
    TRAINING_CLASSES,
    PreparedRecord,
    current_git_commit,
    decision_from_probs,
    hash_prepared_rows,
    load_prepared_rows,
    parse_seed_csv,
    require_cuda,
    save_json,
    set_seed,
    sigmoid,
    stable_object_hash,
    summarize_label_predictions,
    softmax,
    utc_now_iso,
    write_jsonl,
)

DEFAULT_MODEL = "cardiffnlp/twitter-roberta-large-2022-154m"
DEFAULT_TRAIN = DATA_DIR / "transformer" / "train.prepared.jsonl"
DEFAULT_VALID = DATA_DIR / "transformer" / "valid.prepared.jsonl"
DEFAULT_HOLDOUT = DATA_DIR / "transformer" / "holdout.prepared.jsonl"
DEFAULT_OUTPUT = MODELS_DIR / "teacher"


class JanitrTeacherModel(nn.Module):
    def __init__(self, model_name_or_path: str, gradient_checkpointing: bool) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name_or_path,
            output_hidden_states=True,
        )
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        hidden_size = int(self.encoder.config.hidden_size)
        dropout_prob = float(getattr(self.encoder.config, "hidden_dropout_prob", 0.1))
        self.dropout = nn.Dropout(dropout_prob)
        self.head_scam_clean = nn.Linear(hidden_size, 2)
        self.head_topic = nn.Linear(hidden_size, 1)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...] | None]:
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        cls = out.last_hidden_state[:, 0]
        pooled = self.dropout(cls)
        scam_logits = self.head_scam_clean(pooled)
        topic_logits = self.head_topic(pooled)
        return {
            "scam_logits": scam_logits,
            "topic_logits": topic_logits,
            "hidden_states": out.hidden_states if output_hidden_states else None,
        }


class PreparedDataset(Dataset):
    def __init__(self, rows: list[PreparedRecord], tokenizer, max_length: int) -> None:
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
            "text": row.text_normalized,
            "collapsed_label": row.collapsed_label,
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "y_scam_clean": torch.tensor(row.y_scam_clean, dtype=torch.long),
            "y_topic": torch.tensor(float(row.y_topics[0]), dtype=torch.float32),
        }


def collate(batch: list[dict]) -> dict:
    out: dict[str, list] = {
        "id": [],
        "text": [],
        "collapsed_label": [],
        "input_ids": [],
        "attention_mask": [],
        "y_scam_clean": [],
        "y_topic": [],
    }
    for item in batch:
        for key in ("id", "text", "collapsed_label"):
            out[key].append(item[key])
        out["input_ids"].append(item["input_ids"])
        out["attention_mask"].append(item["attention_mask"])
        out["y_scam_clean"].append(item["y_scam_clean"])
        out["y_topic"].append(item["y_topic"])

    return {
        "id": out["id"],
        "text": out["text"],
        "collapsed_label": out["collapsed_label"],
        "input_ids": torch.stack(out["input_ids"]),
        "attention_mask": torch.stack(out["attention_mask"]),
        "y_scam_clean": torch.stack(out["y_scam_clean"]),
        "y_topic": torch.stack(out["y_topic"]),
    }


@dataclass
class LossWeights:
    ce_weight: torch.Tensor
    bce_pos_weight: torch.Tensor


def compute_loss_weights(
    rows: list[PreparedRecord], device: torch.device
) -> LossWeights:
    scam_clean_counts = Counter(row.y_scam_clean for row in rows)
    topic_pos = sum(row.y_topics[0] for row in rows)
    topic_neg = len(rows) - topic_pos

    clean_count = max(1, scam_clean_counts.get(0, 0))
    scam_count = max(1, scam_clean_counts.get(1, 0))
    total = clean_count + scam_count

    ce_weight = torch.tensor(
        [total / (2 * clean_count), total / (2 * scam_count)],
        dtype=torch.float32,
        device=device,
    )
    pos_weight = torch.tensor(
        [float(topic_neg) / max(1.0, float(topic_pos))],
        dtype=torch.float32,
        device=device,
    )
    return LossWeights(ce_weight=ce_weight, bce_pos_weight=pos_weight)


def get_amp_dtype(dtype: str) -> torch.dtype:
    if dtype == "fp16":
        return torch.float16
    return torch.bfloat16


def train_one_seed(
    *,
    seed: int,
    model_name_or_path: str,
    train_rows: list[PreparedRecord],
    valid_rows: list[PreparedRecord],
    holdout_rows: list[PreparedRecord],
    out_dir: Path,
    batch_size: int,
    grad_accum_steps: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    max_length: int,
    dtype: str,
    gradient_checkpointing: bool,
    eval_batch_size: int,
    scam_threshold: float,
    topic_threshold: float,
) -> dict:
    set_seed(seed)

    device = require_cuda(context=f"train_transformer_teacher.py seed={seed}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = JanitrTeacherModel(
        model_name_or_path=model_name_or_path,
        gradient_checkpointing=gradient_checkpointing,
    ).to(device)

    train_ds = PreparedDataset(train_rows, tokenizer, max_length=max_length)
    valid_ds = PreparedDataset(valid_rows, tokenizer, max_length=max_length)
    holdout_ds = PreparedDataset(holdout_rows, tokenizer, max_length=max_length)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=eval_batch_size, shuffle=False, collate_fn=collate
    )
    holdout_loader = DataLoader(
        holdout_ds, batch_size=eval_batch_size, shuffle=False, collate_fn=collate
    )

    weights = compute_loss_weights(train_rows, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    use_amp = device.type == "cuda" and dtype in {"fp16", "bf16"}
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and dtype == "fp16"))
    amp_dtype = get_amp_dtype(dtype)

    best_valid_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        progress = tqdm(
            train_loader, desc=f"seed {seed} epoch {epoch}/{epochs}", leave=False
        )
        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y_scam = batch["y_scam_clean"].to(device)
            y_topic = batch["y_topic"].to(device)

            amp_ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if use_amp
                else nullcontext()
            )
            with amp_ctx:
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                scam_logits = out["scam_logits"]
                topic_logits = out["topic_logits"].squeeze(-1)

                loss_scam = F.cross_entropy(
                    scam_logits, y_scam, weight=weights.ce_weight
                )
                loss_topic = F.binary_cross_entropy_with_logits(
                    topic_logits,
                    y_topic,
                    pos_weight=weights.bce_pos_weight,
                )
                loss = loss_scam + loss_topic
                loss_scaled = loss / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            if step % grad_accum_steps == 0 or step == len(train_loader):
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        valid_metrics, _ = evaluate_loader(
            model=model,
            loader=valid_loader,
            device=device,
            dtype=dtype,
            scam_threshold=scam_threshold,
            topic_threshold=topic_threshold,
        )
        valid_f1 = valid_metrics["metrics"]["scam"]["f1"]
        print(
            f"[seed={seed}] epoch={epoch} train_loss={running_loss / max(1, len(train_loader)):.4f} "
            f"valid_scam_f1={valid_f1:.4f}"
        )
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), seed_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(str(seed_dir))
    with (seed_dir / "teacher_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name_or_path": model_name_or_path,
                "max_length": max_length,
                "scam_threshold": scam_threshold,
                "topic_threshold": topic_threshold,
            },
            f,
            indent=2,
        )

    valid_metrics, valid_preds = evaluate_loader(
        model=model,
        loader=valid_loader,
        device=device,
        dtype=dtype,
        scam_threshold=scam_threshold,
        topic_threshold=topic_threshold,
    )
    holdout_metrics, holdout_preds = evaluate_loader(
        model=model,
        loader=holdout_loader,
        device=device,
        dtype=dtype,
        scam_threshold=scam_threshold,
        topic_threshold=topic_threshold,
    )

    write_jsonl(seed_dir / "valid_preds.jsonl", valid_preds)
    write_jsonl(seed_dir / "holdout_preds.jsonl", holdout_preds)

    return {
        "seed": seed,
        "seed_dir": str(seed_dir),
        "valid_metrics": valid_metrics,
        "holdout_metrics": holdout_metrics,
        "valid_preds": valid_preds,
        "holdout_preds": holdout_preds,
    }


def evaluate_loader(
    *,
    model: JanitrTeacherModel,
    loader: DataLoader,
    device: torch.device,
    dtype: str,
    scam_threshold: float,
    topic_threshold: float,
) -> tuple[dict, list[dict]]:
    model.eval()
    use_amp = device.type == "cuda" and dtype in {"fp16", "bf16"}
    amp_dtype = get_amp_dtype(dtype)

    y_true: list[str] = []
    y_pred: list[str] = []
    pred_rows: list[dict] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            amp_ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if use_amp
                else nullcontext()
            )
            with amp_ctx:
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            scam_logits = out["scam_logits"].detach().cpu().float().numpy()
            topic_logits = (
                out["topic_logits"].detach().cpu().float().numpy().reshape(-1)
            )

            scam_probs = softmax(scam_logits)[:, 1]
            topic_probs = sigmoid(topic_logits)

            for i in range(len(batch["id"])):
                gold = batch["collapsed_label"][i]
                pred = decision_from_probs(
                    float(scam_probs[i]),
                    float(topic_probs[i]),
                    scam_threshold=scam_threshold,
                    topic_threshold=topic_threshold,
                )
                y_true.append(gold)
                y_pred.append(pred)
                pred_rows.append(
                    {
                        "id": batch["id"][i],
                        "text": batch["text"][i],
                        "collapsed_label": gold,
                        "pred_label": pred,
                        "scam_logits": [float(x) for x in scam_logits[i].tolist()],
                        "topic_logit": float(topic_logits[i]),
                        "scam_prob": float(scam_probs[i]),
                        "topic_prob": float(topic_probs[i]),
                    }
                )

    metrics = summarize_label_predictions(y_true, y_pred, classes=TRAINING_CLASSES)
    return metrics, pred_rows


def merge_ensemble_predictions(per_seed_preds: list[list[dict]]) -> list[dict]:
    if not per_seed_preds:
        return []
    merged: list[dict] = []
    num_seeds = len(per_seed_preds)
    length = len(per_seed_preds[0])
    for preds in per_seed_preds:
        if len(preds) != length:
            raise RuntimeError(
                "Seed prediction lengths do not match for ensemble merge."
            )

    for idx in range(length):
        base = per_seed_preds[0][idx]
        scam_logits = np.zeros(2, dtype=np.float64)
        topic_logit = 0.0
        for seed_preds in per_seed_preds:
            row = seed_preds[idx]
            scam_logits += np.array(row["scam_logits"], dtype=np.float64)
            topic_logit += float(row["topic_logit"])
        scam_logits /= num_seeds
        topic_logit /= num_seeds

        merged.append(
            {
                "id": base["id"],
                "text": base["text"],
                "collapsed_label": base["collapsed_label"],
                "scam_logits": [float(x) for x in scam_logits.tolist()],
                "topic_logit": float(topic_logit),
            }
        )
    return merged


def eval_ensemble_predictions(
    rows: list[dict],
    *,
    scam_threshold: float,
    topic_threshold: float,
) -> tuple[dict, list[dict]]:
    y_true: list[str] = []
    y_pred: list[str] = []
    out_rows: list[dict] = []

    for row in rows:
        scam_prob = float(
            softmax(np.array([row["scam_logits"]], dtype=np.float64))[0][1]
        )
        topic_prob = float(sigmoid(np.array([row["topic_logit"]], dtype=np.float64))[0])
        pred = decision_from_probs(
            scam_prob,
            topic_prob,
            scam_threshold=scam_threshold,
            topic_threshold=topic_threshold,
        )
        y_true.append(str(row["collapsed_label"]))
        y_pred.append(pred)
        out = dict(row)
        out["pred_label"] = pred
        out["scam_prob"] = scam_prob
        out["topic_prob"] = topic_prob
        out_rows.append(out)

    metrics = summarize_label_predictions(y_true, y_pred, classes=TRAINING_CLASSES)
    return metrics, out_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--teacher-init-path",
        type=str,
        default=None,
        help="Optional local checkpoint path (e.g., models/teacher_dapt). Overrides --model-name when set.",
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name. Defaults to yyyy-mm-dd-<petname>. Use {run_name} in --output-dir to template.",
    )
    parser.add_argument("--seeds", type=str, default="13,17,21")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--scam-threshold", type=float, default=0.5)
    parser.add_argument("--topic-threshold", type=float, default=0.5)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "fp32"],
        default="bf16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "fp16",
    )
    args = parser.parse_args()

    for path in (args.train, args.valid, args.holdout):
        if not path.exists():
            raise SystemExit(
                f"Prepared split missing: {path}. Run prepare_transformer_data.py first."
            )

    train_rows = load_prepared_rows(args.train)
    valid_rows = load_prepared_rows(args.valid)
    holdout_rows = load_prepared_rows(args.holdout)

    try:
        seeds = parse_seed_csv(args.seeds)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if len(train_rows) <= 5000 and len(seeds) < 3:
        raise SystemExit(
            f"Dataset has {len(train_rows)} train rows; expected 3 teacher seeds per plan, got {len(seeds)}."
        )
    if args.teacher_init_path is None and args.model_name != DEFAULT_MODEL:
        raise SystemExit(
            f"Teacher must be {DEFAULT_MODEL} for this pipeline; got {args.model_name}."
        )

    run_name = resolve_run_name(args.run_name)
    model_name_or_path = args.teacher_init_path or args.model_name
    out_dir = apply_run_name_template(args.output_dir, run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_hashes = {
        "train": hash_prepared_rows(train_rows),
        "valid": hash_prepared_rows(valid_rows),
        "holdout": hash_prepared_rows(holdout_rows),
    }
    teacher_identity_payload = {
        "model_name_or_path": model_name_or_path,
        "seeds": seeds,
        "thresholds": {
            "scam": args.scam_threshold,
            "topic_crypto": args.topic_threshold,
        },
        "max_length": args.max_length,
        "split_hashes": split_hashes,
        "run_name": run_name,
        "code_commit": current_git_commit(),
    }
    teacher_id = f"teacher-{stable_object_hash(teacher_identity_payload)[:16]}"

    per_seed_results: list[dict] = []
    for seed in seeds:
        result = train_one_seed(
            seed=seed,
            model_name_or_path=model_name_or_path,
            train_rows=train_rows,
            valid_rows=valid_rows,
            holdout_rows=holdout_rows,
            out_dir=out_dir,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_length=args.max_length,
            dtype=args.dtype,
            gradient_checkpointing=(
                args.gradient_checkpointing and not args.no_gradient_checkpointing
            ),
            eval_batch_size=args.eval_batch_size,
            scam_threshold=args.scam_threshold,
            topic_threshold=args.topic_threshold,
        )
        per_seed_results.append(result)

    ensemble_valid = merge_ensemble_predictions(
        [x["valid_preds"] for x in per_seed_results]
    )
    ensemble_holdout = merge_ensemble_predictions(
        [x["holdout_preds"] for x in per_seed_results]
    )

    ensemble_valid_metrics, ensemble_valid_out = eval_ensemble_predictions(
        ensemble_valid,
        scam_threshold=args.scam_threshold,
        topic_threshold=args.topic_threshold,
    )
    ensemble_holdout_metrics, ensemble_holdout_out = eval_ensemble_predictions(
        ensemble_holdout,
        scam_threshold=args.scam_threshold,
        topic_threshold=args.topic_threshold,
    )

    write_jsonl(MODELS_DIR / "teacher_valid_preds.jsonl", ensemble_valid_out)
    write_jsonl(MODELS_DIR / "teacher_holdout_preds.jsonl", ensemble_holdout_out)

    summary = {
        "run_name": run_name,
        "teacher_id": teacher_id,
        "model_name_or_path": model_name_or_path,
        "seeds": seeds,
        "dtype": args.dtype,
        "gradient_checkpointing": args.gradient_checkpointing
        and not args.no_gradient_checkpointing,
        "thresholds": {
            "scam": args.scam_threshold,
            "topic_crypto": args.topic_threshold,
        },
        "per_seed": [
            {
                "seed": item["seed"],
                "seed_dir": item["seed_dir"],
                "valid": item["valid_metrics"],
                "holdout": item["holdout_metrics"],
            }
            for item in per_seed_results
        ],
        "ensemble": {
            "valid": ensemble_valid_metrics,
            "holdout": ensemble_holdout_metrics,
        },
    }
    save_json(out_dir / "training_summary.json", summary)

    teacher_manifest = {
        "version": 1,
        "run_name": run_name,
        "teacher_id": teacher_id,
        "created_at": utc_now_iso(),
        "code_commit": current_git_commit(),
        "model_name_or_path": model_name_or_path,
        "teacher_init_path": args.teacher_init_path,
        "seeds": seeds,
        "thresholds": {
            "scam": args.scam_threshold,
            "topic_crypto": args.topic_threshold,
        },
        "splits": {
            "train": {
                "path": str(args.train),
                "rows": len(train_rows),
                "hash": split_hashes["train"],
            },
            "valid": {
                "path": str(args.valid),
                "rows": len(valid_rows),
                "hash": split_hashes["valid"],
            },
            "holdout": {
                "path": str(args.holdout),
                "rows": len(holdout_rows),
                "hash": split_hashes["holdout"],
            },
        },
    }
    save_json(out_dir / "teacher_manifest.json", teacher_manifest)

    print(f"Run name: {run_name}")
    print(f"Wrote teacher summary to {out_dir / 'training_summary.json'}")
    print(f"Wrote teacher manifest to {out_dir / 'teacher_manifest.json'}")
    print(f"Wrote ensemble valid preds to {MODELS_DIR / 'teacher_valid_preds.jsonl'}")


if __name__ == "__main__":
    main()
