#!/usr/bin/env python3
"""Calibrate teacher logits using temperature scaling on validation split."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from transformer_common import (
    DATA_DIR,
    MODELS_DIR,
    PreparedRecord,
    brier_score,
    calibration_bins,
    current_git_commit,
    expected_calibration_error,
    hash_prepared_rows,
    load_json,
    load_prepared_rows,
    sigmoid,
    softmax,
    stable_object_hash,
    utc_now_iso,
    write_jsonl,
    load_jsonl,
    save_json,
)

DEFAULT_PREPARED_VALID = DATA_DIR / "transformer" / "valid.prepared.jsonl"
DEFAULT_PREDS = MODELS_DIR / "teacher_valid_preds.jsonl"
DEFAULT_OUT = MODELS_DIR / "teacher_calibration.json"
DEFAULT_OUT_PREDS = MODELS_DIR / "teacher_valid_preds_calibrated.jsonl"
DEFAULT_TEACHER_DIR = MODELS_DIR / "teacher"


def nll_scam(logits: np.ndarray, y_true: np.ndarray, temp: float) -> float:
    probs = softmax(logits / temp)
    p = np.clip(probs[:, 1], 1e-7, 1 - 1e-7)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def nll_topic(logits: np.ndarray, y_true: np.ndarray, temp: float) -> float:
    probs = sigmoid(logits / temp)
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def best_temperature(
    scorer,
    logits: np.ndarray,
    y_true: np.ndarray,
    *,
    min_temp: float,
    max_temp: float,
    step: float,
) -> tuple[float, float]:
    best_t = 1.0
    best_loss = math.inf
    t = min_temp
    while t <= max_temp + 1e-9:
        loss = scorer(logits, y_true, t)
        if loss < best_loss:
            best_loss = loss
            best_t = t
        t += step
    return best_t, best_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-valid", type=Path, default=DEFAULT_PREPARED_VALID)
    parser.add_argument("--preds", type=Path, default=DEFAULT_PREDS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out-calibrated-preds", type=Path, default=DEFAULT_OUT_PREDS)
    parser.add_argument("--teacher-dir", type=Path, default=None)
    parser.add_argument("--min-temp", type=float, default=0.5)
    parser.add_argument("--max-temp", type=float, default=5.0)
    parser.add_argument("--temp-step", type=float, default=0.05)
    args = parser.parse_args()

    if not args.prepared_valid.exists():
        raise SystemExit(f"Prepared valid split not found: {args.prepared_valid}")
    if not args.preds.exists():
        raise SystemExit(f"Teacher valid preds not found: {args.preds}")
    teacher_dir = args.teacher_dir
    if teacher_dir is None:
        inferred = args.preds.parent / "teacher"
        teacher_dir = inferred if inferred.exists() else DEFAULT_TEACHER_DIR
    teacher_manifest_path = teacher_dir / "teacher_manifest.json"
    if not teacher_manifest_path.exists():
        raise SystemExit(
            f"Teacher manifest missing: {teacher_manifest_path}. Re-run train_transformer_teacher.py first."
        )

    valid_rows = load_prepared_rows(args.prepared_valid)
    teacher_manifest = load_json(teacher_manifest_path)
    teacher_id = str(teacher_manifest.get("teacher_id", "")).strip()
    if not teacher_id:
        raise SystemExit(
            f"Teacher manifest at {teacher_manifest_path} is missing teacher_id."
        )
    valid_hash_expected = (
        teacher_manifest.get("splits", {}).get("valid", {}).get("hash")
    )
    valid_hash_actual = hash_prepared_rows(valid_rows)
    if valid_hash_expected and str(valid_hash_expected) != valid_hash_actual:
        raise SystemExit(
            "Prepared valid split hash does not match teacher manifest. "
            "Calibration would mix incompatible artifacts."
        )
    valid_by_id = {row.id: row for row in valid_rows}

    preds_raw = load_jsonl(args.preds)
    aligned: list[tuple[PreparedRecord, dict]] = []
    for row in preds_raw:
        sample_id = str(row.get("id", ""))
        gt = valid_by_id.get(sample_id)
        if gt is None:
            continue
        aligned.append((gt, row))

    if not aligned:
        raise SystemExit("No aligned rows between prepared-valid and teacher preds.")

    scam_logits = np.array(
        [item[1]["scam_logits"] for item in aligned], dtype=np.float64
    )
    topic_logits = np.array(
        [item[1]["topic_logit"] for item in aligned], dtype=np.float64
    )
    y_scam = np.array([item[0].y_scam_clean for item in aligned], dtype=np.float64)
    y_topic = np.array([item[0].y_topics[0] for item in aligned], dtype=np.float64)

    temp_scam, scam_nll = best_temperature(
        nll_scam,
        scam_logits,
        y_scam,
        min_temp=args.min_temp,
        max_temp=args.max_temp,
        step=args.temp_step,
    )
    temp_topic, topic_nll = best_temperature(
        nll_topic,
        topic_logits,
        y_topic,
        min_temp=args.min_temp,
        max_temp=args.max_temp,
        step=args.temp_step,
    )

    scam_prob_raw = softmax(scam_logits)[:, 1]
    topic_prob_raw = sigmoid(topic_logits)
    scam_prob_cal = softmax(scam_logits / temp_scam)[:, 1]
    topic_prob_cal = sigmoid(topic_logits / temp_topic)

    scam_bins_raw = calibration_bins(
        y_scam.astype(int).tolist(), scam_prob_raw.tolist()
    )
    scam_bins_cal = calibration_bins(
        y_scam.astype(int).tolist(), scam_prob_cal.tolist()
    )
    topic_bins_raw = calibration_bins(
        y_topic.astype(int).tolist(), topic_prob_raw.tolist()
    )
    topic_bins_cal = calibration_bins(
        y_topic.astype(int).tolist(), topic_prob_cal.tolist()
    )

    payload = {
        "meta": {
            "version": 1,
            "calibration_id": f"calib-{stable_object_hash({'teacher_id': teacher_id, 'valid_hash': valid_hash_actual, 'preds': str(args.preds), 'min_temp': args.min_temp, 'max_temp': args.max_temp, 'temp_step': args.temp_step})[:16]}",
            "teacher_id": teacher_id,
            "valid_split_hash": valid_hash_actual,
            "code_commit": current_git_commit(),
            "created_at": utc_now_iso(),
        },
        "temps": {
            "scam_clean_head": float(temp_scam),
            "topic_crypto_head": float(temp_topic),
        },
        "validation": {
            "samples": len(aligned),
            "nll": {
                "scam_clean_head": float(scam_nll),
                "topic_crypto_head": float(topic_nll),
            },
            "ece": {
                "scam_raw": expected_calibration_error(scam_bins_raw),
                "scam_calibrated": expected_calibration_error(scam_bins_cal),
                "topic_raw": expected_calibration_error(topic_bins_raw),
                "topic_calibrated": expected_calibration_error(topic_bins_cal),
            },
            "brier": {
                "scam_raw": brier_score(
                    y_scam.astype(int).tolist(), scam_prob_raw.tolist()
                ),
                "scam_calibrated": brier_score(
                    y_scam.astype(int).tolist(), scam_prob_cal.tolist()
                ),
                "topic_raw": brier_score(
                    y_topic.astype(int).tolist(), topic_prob_raw.tolist()
                ),
                "topic_calibrated": brier_score(
                    y_topic.astype(int).tolist(), topic_prob_cal.tolist()
                ),
            },
            "bins": {
                "scam_raw": scam_bins_raw,
                "scam_calibrated": scam_bins_cal,
                "topic_raw": topic_bins_raw,
                "topic_calibrated": topic_bins_cal,
            },
        },
    }

    save_json(args.out, payload)

    calibrated_rows: list[dict] = []
    for (gt, row), s_prob, t_prob in zip(aligned, scam_prob_cal, topic_prob_cal):
        out_row = dict(row)
        out_row["calibrated_scam_prob"] = float(s_prob)
        out_row["calibrated_topic_prob"] = float(t_prob)
        out_row["y_scam_clean"] = int(gt.y_scam_clean)
        out_row["y_topic_crypto"] = int(gt.y_topics[0])
        calibrated_rows.append(out_row)
    write_jsonl(args.out_calibrated_preds, calibrated_rows)

    print(f"Saved teacher calibration to {args.out}")
    print(
        "Best temperatures: "
        f"scam_clean_head={temp_scam:.3f}, topic_crypto_head={temp_topic:.3f}"
    )
    print(f"Saved calibrated validation predictions to {args.out_calibrated_preds}")


if __name__ == "__main__":
    main()
