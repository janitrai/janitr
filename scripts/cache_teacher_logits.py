#!/usr/bin/env python3
"""Cache calibrated teacher logits (and hidden states) for student distillation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from transformer_common import (
    DATA_DIR,
    MODELS_DIR,
    TRAINING_CLASSES,
    PreparedRecord,
    current_git_commit,
    hash_label_map,
    hash_prepared_rows,
    load_json,
    load_prepared_rows,
    require_cuda,
    save_json,
    set_seed,
    stable_object_hash,
    utc_now_iso,
)

DEFAULT_TRAIN = DATA_DIR / "transformer" / "train.prepared.jsonl"
DEFAULT_VALID = DATA_DIR / "transformer" / "valid.prepared.jsonl"
DEFAULT_TEACHER_DIR = MODELS_DIR / "teacher"
DEFAULT_CALIBRATION = MODELS_DIR / "teacher_calibration.json"


class JanitrTeacherModel(nn.Module):
    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name_or_path,
            output_hidden_states=True,
        )
        hidden_size = int(self.encoder.config.hidden_size)
        dropout_prob = float(getattr(self.encoder.config, "hidden_dropout_prob", 0.1))
        self.dropout = nn.Dropout(dropout_prob)
        self.head_scam_clean = nn.Linear(hidden_size, 2)
        self.head_topic = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        cls = self.dropout(out.last_hidden_state[:, 0])
        scam_logits = self.head_scam_clean(cls)
        topic_logits = self.head_topic(cls).squeeze(-1)
        return {
            "scam_logits": scam_logits,
            "topic_logits": topic_logits,
            "hidden_states": out.hidden_states,
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
            "y_scam_clean": int(row.y_scam_clean),
            "y_topic": int(row.y_topics[0]),
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        }


def collate(batch: list[dict]) -> dict:
    return {
        "id": [x["id"] for x in batch],
        "text": [x["text"] for x in batch],
        "collapsed_label": [x["collapsed_label"] for x in batch],
        "y_scam_clean": np.array([x["y_scam_clean"] for x in batch], dtype=np.int64),
        "y_topic": np.array([x["y_topic"] for x in batch], dtype=np.int64),
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
    }


def list_seed_dirs(teacher_dir: Path, explicit_seeds: str | None) -> list[Path]:
    if explicit_seeds:
        dirs = [
            teacher_dir / f"seed_{int(s.strip())}"
            for s in explicit_seeds.split(",")
            if s.strip()
        ]
    else:
        dirs = sorted(path for path in teacher_dir.glob("seed_*") if path.is_dir())
    if not dirs:
        raise SystemExit(f"No teacher seed directories found under {teacher_dir}")
    for path in dirs:
        if not (path / "pytorch_model.bin").exists():
            raise SystemExit(
                f"Teacher checkpoint missing at {path / 'pytorch_model.bin'}"
            )
    return dirs


def layer_indices(num_hidden_layers: int, target_layers: int = 4) -> list[int]:
    return [int(round(x)) for x in np.linspace(1, num_hidden_layers, target_layers)]


def load_teacher(
    seed_dir: Path, device: torch.device
) -> tuple[JanitrTeacherModel, any, int, str]:
    config_path = seed_dir / "teacher_config.json"
    if not config_path.exists():
        raise SystemExit(f"Missing teacher config at {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    model_name = str(config["model_name_or_path"])
    max_length = int(config.get("max_length", 96))

    tokenizer = AutoTokenizer.from_pretrained(str(seed_dir))
    model = JanitrTeacherModel(model_name)
    state = torch.load(seed_dir / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, tokenizer, max_length, model_name


def cache_split(
    *,
    split_name: str,
    rows: list[PreparedRecord],
    seed_dirs: list[Path],
    device: torch.device,
    batch_size: int,
    out_path: Path,
    scam_temp: float,
    topic_temp: float,
    dtype: str,
    teacher_id: str,
    calibration_id: str,
    seed_list: list[int],
    label_map_hash: str,
    split_hash: str,
) -> None:
    models: list[JanitrTeacherModel] = []
    tokenizers = []
    max_lengths: list[int] = []

    for seed_dir in seed_dirs:
        model, tokenizer, max_length, _model_name = load_teacher(
            seed_dir, device=device
        )
        models.append(model)
        tokenizers.append(tokenizer)
        max_lengths.append(max_length)

    max_length = min(max_lengths)
    dataset = PreparedDataset(rows, tokenizers[0], max_length=max_length)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
    )

    num_layers = int(models[0].encoder.config.num_hidden_layers)
    hid_indices = layer_indices(num_layers, target_layers=4)

    ids: list[str] = []
    texts: list[str] = []
    labels: list[str] = []
    y_scam_clean: list[int] = []
    y_topic: list[int] = []
    scam_logits_raw: list[np.ndarray] = []
    topic_logits_raw: list[float] = []
    hidden_cls: list[np.ndarray] = []

    use_amp = device.type == "cuda" and dtype in {"fp16", "bf16"}
    amp_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Caching {split_name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            seed_scam: list[np.ndarray] = []
            seed_topic: list[np.ndarray] = []
            seed_hidden: list[np.ndarray] = []

            for model in models:
                with torch.autocast(
                    device_type=device.type, enabled=use_amp, dtype=amp_dtype
                ):
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                scam = out["scam_logits"].detach().cpu().float().numpy()
                topic = out["topic_logits"].detach().cpu().float().numpy()
                hidden_states = out["hidden_states"]

                selected = []
                for idx in hid_indices:
                    layer_tensor = hidden_states[idx].detach().cpu().float().numpy()
                    selected.append(layer_tensor[:, 0, :])  # CLS vectors
                selected_arr = np.stack(selected, axis=1)

                seed_scam.append(scam)
                seed_topic.append(topic)
                seed_hidden.append(selected_arr)

            avg_scam = np.mean(np.stack(seed_scam, axis=0), axis=0)
            avg_topic = np.mean(np.stack(seed_topic, axis=0), axis=0)
            avg_hidden = np.mean(np.stack(seed_hidden, axis=0), axis=0)

            ids.extend(batch["id"])
            texts.extend(batch["text"])
            labels.extend(batch["collapsed_label"])
            y_scam_clean.extend(batch["y_scam_clean"].tolist())
            y_topic.extend(batch["y_topic"].tolist())
            scam_logits_raw.extend(avg_scam)
            topic_logits_raw.extend(avg_topic.tolist())
            hidden_cls.extend(avg_hidden)

    scam_logits = np.array(scam_logits_raw, dtype=np.float32)
    topic_logits = np.array(topic_logits_raw, dtype=np.float32)
    hidden = np.array(hidden_cls, dtype=np.float32)

    scam_logits_cal = scam_logits / float(scam_temp)
    topic_logits_cal = topic_logits / float(topic_temp)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        ids=np.array(ids, dtype=object),
        texts=np.array(texts, dtype=object),
        collapsed_label=np.array(labels, dtype=object),
        y_scam_clean=np.array(y_scam_clean, dtype=np.int64),
        y_topic=np.array(y_topic, dtype=np.int64),
        scam_logits_raw=scam_logits.astype(np.float16),
        topic_logits_raw=topic_logits.astype(np.float16),
        scam_logits_cal=scam_logits_cal.astype(np.float16),
        topic_logits_cal=topic_logits_cal.astype(np.float16),
        teacher_hidden_cls=hidden.astype(np.float16),
        hidden_layer_indices=np.array(hid_indices, dtype=np.int64),
        scam_temp=np.array([scam_temp], dtype=np.float32),
        topic_temp=np.array([topic_temp], dtype=np.float32),
    )
    metadata = {
        "version": 1,
        "logits_cache_id": f"logits-{stable_object_hash({'teacher_id': teacher_id, 'calibration_id': calibration_id, 'seed_list': seed_list, 'label_map_hash': label_map_hash, 'split': split_name, 'split_hash': split_hash})[:16]}",
        "teacher_id": teacher_id,
        "calibration_id": calibration_id,
        "seeds": seed_list,
        "label_map_hash": label_map_hash,
        "split": split_name,
        "split_hash": split_hash,
        "rows": len(rows),
        "code_commit": current_git_commit(),
        "created_at": utc_now_iso(),
        "path": str(out_path),
    }
    meta_path = Path(f"{out_path}.meta.json")
    if meta_path.exists():
        existing = load_json(meta_path)
        for key in (
            "teacher_id",
            "calibration_id",
            "seeds",
            "label_map_hash",
            "split",
            "split_hash",
        ):
            if existing.get(key) != metadata.get(key):
                raise SystemExit(
                    f"Existing cache metadata mismatch for {meta_path} key={key}: "
                    f"existing={existing.get(key)} new={metadata.get(key)}"
                )
    save_json(meta_path, metadata)

    print(f"Cached {split_name} logits to {out_path}")
    print(f"Wrote cache metadata to {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER_DIR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "fp32"],
        default="bf16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "fp16",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=MODELS_DIR / "teacher_logits_train.npz",
    )
    parser.add_argument(
        "--valid-out",
        type=Path,
        default=MODELS_DIR / "teacher_logits_valid.npz",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    for path in (args.train, args.valid, args.teacher_dir):
        if not path.exists():
            raise SystemExit(f"Missing required path: {path}")

    teacher_manifest_path = args.teacher_dir / "teacher_manifest.json"
    if not teacher_manifest_path.exists():
        raise SystemExit(
            f"Teacher manifest missing: {teacher_manifest_path}. Re-run train_transformer_teacher.py first."
        )

    if not args.calibration.exists():
        raise SystemExit(
            f"Calibration file missing: {args.calibration}. Run calibrate_teacher.py first."
        )

    teacher_manifest = load_json(teacher_manifest_path)
    teacher_id = str(teacher_manifest.get("teacher_id", "")).strip()
    if not teacher_id:
        raise SystemExit(
            f"Teacher manifest at {teacher_manifest_path} is missing teacher_id."
        )
    expected_seeds = [int(seed) for seed in teacher_manifest.get("seeds", [])]
    if not expected_seeds:
        raise SystemExit(
            f"Teacher manifest at {teacher_manifest_path} has empty seed list."
        )

    calibration = load_json(args.calibration)
    calibration_meta = calibration.get("meta", {})
    calibration_id = str(calibration_meta.get("calibration_id", "")).strip()
    calibration_teacher_id = str(calibration_meta.get("teacher_id", "")).strip()
    if not calibration_id or not calibration_teacher_id:
        raise SystemExit(
            "Calibration metadata is missing calibration_id/teacher_id. Re-run calibrate_teacher.py."
        )
    if calibration_teacher_id != teacher_id:
        raise SystemExit(
            f"Calibration teacher_id mismatch: calibration={calibration_teacher_id} teacher_manifest={teacher_id}"
        )
    scam_temp = float(calibration["temps"]["scam_clean_head"])
    topic_temp = float(calibration["temps"]["topic_crypto_head"])

    train_rows = load_prepared_rows(args.train)
    valid_rows = load_prepared_rows(args.valid)
    split_hashes = {
        "train": hash_prepared_rows(train_rows),
        "valid": hash_prepared_rows(valid_rows),
    }
    label_map_hash = hash_label_map(TRAINING_CLASSES)

    device = require_cuda(context="cache_teacher_logits.py")

    seed_dirs = list_seed_dirs(args.teacher_dir, args.seeds)
    selected_seeds = [int(path.name.split("_", 1)[1]) for path in seed_dirs]
    if selected_seeds != expected_seeds:
        raise SystemExit(
            f"Teacher seed mismatch. selected={selected_seeds} manifest={expected_seeds}. "
            "Refusing to mix incompatible seed sets."
        )
    print("Teacher seeds:", ", ".join(path.name for path in seed_dirs))

    cache_split(
        split_name="train",
        rows=train_rows,
        seed_dirs=seed_dirs,
        device=device,
        batch_size=args.batch_size,
        out_path=args.train_out,
        scam_temp=scam_temp,
        topic_temp=topic_temp,
        dtype=args.dtype,
        teacher_id=teacher_id,
        calibration_id=calibration_id,
        seed_list=selected_seeds,
        label_map_hash=label_map_hash,
        split_hash=split_hashes["train"],
    )
    cache_split(
        split_name="valid",
        rows=valid_rows,
        seed_dirs=seed_dirs,
        device=device,
        batch_size=args.batch_size,
        out_path=args.valid_out,
        scam_temp=scam_temp,
        topic_temp=topic_temp,
        dtype=args.dtype,
        teacher_id=teacher_id,
        calibration_id=calibration_id,
        seed_list=selected_seeds,
        label_map_hash=label_map_hash,
        split_hash=split_hashes["valid"],
    )


if __name__ == "__main__":
    main()
