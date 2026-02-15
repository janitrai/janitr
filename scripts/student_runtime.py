#!/usr/bin/env python3
"""Shared tiny-student runtime helpers (architecture + loading)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import BertConfig, BertModel, BertTokenizerFast

from transformer_common import load_json


class TinyStudentModel(nn.Module):
    def __init__(self, config: BertConfig, teacher_hidden_size: int) -> None:
        super().__init__()
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.head_scam_clean = nn.Linear(config.hidden_size, 2)
        self.head_topic = nn.Linear(config.hidden_size, 1)
        self.teacher_projections = nn.ModuleList(
            [
                nn.Linear(teacher_hidden_size, config.hidden_size)
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> (
        dict[str, torch.Tensor | tuple[torch.Tensor, ...] | None]
        | tuple[torch.Tensor, torch.Tensor]
    ):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        cls = self.dropout(out.last_hidden_state[:, 0])
        scam_logits = self.head_scam_clean(cls)
        topic_logits = self.head_topic(cls).squeeze(-1)
        if output_hidden_states or return_dict:
            return {
                "scam_logits": scam_logits,
                "topic_logits": topic_logits,
                "hidden_states": out.hidden_states if output_hidden_states else None,
            }
        return scam_logits, topic_logits


def build_student_bert_config(*, arch: dict[str, Any], pad_token_id: int) -> BertConfig:
    return BertConfig(
        vocab_size=int(arch["vocab_size"]),
        hidden_size=int(arch["hidden_size"]),
        num_hidden_layers=int(arch["num_hidden_layers"]),
        num_attention_heads=int(arch["num_attention_heads"]),
        intermediate_size=int(arch["intermediate_size"]),
        max_position_embeddings=max(128, int(arch["max_length"]) + 8),
        hidden_dropout_prob=float(arch["dropout"]),
        attention_probs_dropout_prob=float(arch["dropout"]),
        type_vocab_size=1,
        pad_token_id=pad_token_id,
    )


def load_student_from_dir(
    student_dir: Path,
) -> tuple[TinyStudentModel, BertTokenizerFast, dict[str, Any]]:
    config_payload = load_json(student_dir / "student_config.json")
    arch = config_payload["architecture"]
    tokenizer = BertTokenizerFast.from_pretrained(str(student_dir / "tokenizer"))
    config = build_student_bert_config(arch=arch, pad_token_id=tokenizer.pad_token_id)
    model = TinyStudentModel(
        config, teacher_hidden_size=int(arch["teacher_hidden_size"])
    )
    state = torch.load(student_dir / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer, config_payload
