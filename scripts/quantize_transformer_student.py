#!/usr/bin/env python3
"""Quantize exported student ONNX model for browser deployment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from transformers import BertTokenizerFast

from transformer_common import DATA_DIR, MODELS_DIR, load_prepared_rows

DEFAULT_IN = MODELS_DIR / "student.onnx"
DEFAULT_OUT = MODELS_DIR / "student.int8.onnx"
DEFAULT_VALID = DATA_DIR / "transformer" / "valid.prepared.jsonl"
DEFAULT_STUDENT_DIR = MODELS_DIR / "student"


class StudentCalibrationReader(CalibrationDataReader):
    def __init__(
        self,
        tokenizer: BertTokenizerFast,
        texts: list[str],
        max_length: int,
        batch_size: int,
    ) -> None:
        self._inputs: list[dict[str, np.ndarray]] = []
        for idx in range(0, len(texts), batch_size):
            chunk = texts[idx : idx + batch_size]
            enc = tokenizer(
                chunk,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_attention_mask=True,
            )
            self._inputs.append(
                {
                    "input_ids": np.array(enc["input_ids"], dtype=np.int64),
                    "attention_mask": np.array(enc["attention_mask"], dtype=np.int64),
                }
            )
        self._iter = iter(self._inputs)

    def get_next(self):
        return next(self._iter, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mode", choices=["dynamic", "static"], default="dynamic")
    parser.add_argument("--student-dir", type=Path, default=DEFAULT_STUDENT_DIR)
    parser.add_argument("--valid", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--calib-batch-size", type=int, default=32)
    parser.add_argument("--calib-max-samples", type=int, default=512)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input ONNX model not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "dynamic":
        quantize_dynamic(
            model_input=str(args.input),
            model_output=str(args.output),
            weight_type=QuantType.QInt8,
        )
    else:
        tokenizer_dir = args.student_dir / "tokenizer"
        if not tokenizer_dir.exists():
            raise SystemExit(
                f"Tokenizer directory not found for static quantization: {tokenizer_dir}"
            )
        if not args.valid.exists():
            raise SystemExit(f"Prepared valid split not found: {args.valid}")

        tokenizer = BertTokenizerFast.from_pretrained(str(tokenizer_dir))
        rows = load_prepared_rows(args.valid)
        texts = [row.text_normalized for row in rows[: args.calib_max_samples]]
        if not texts:
            raise SystemExit("No calibration texts available for static quantization.")

        calib_reader = StudentCalibrationReader(
            tokenizer=tokenizer,
            texts=texts,
            max_length=args.max_length,
            batch_size=args.calib_batch_size,
        )
        quantize_static(
            model_input=str(args.input),
            model_output=str(args.output),
            calibration_data_reader=calib_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
        )

    src_size = args.input.stat().st_size / (1024 * 1024)
    dst_size = args.output.stat().st_size / (1024 * 1024)
    print(f"Quantized {args.input} -> {args.output} ({args.mode})")
    print(f"Size: {src_size:.2f} MB -> {dst_size:.2f} MB")


if __name__ == "__main__":
    main()
