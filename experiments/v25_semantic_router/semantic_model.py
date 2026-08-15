"""Qwen candidate encoder and full-order Set Transformer for V25."""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from v25_common import clean_text


def candidate_prompt(record: dict, context_titles: list[str] | None = None) -> str:
    context_titles = context_titles or []
    fields = [
        "任务：判断候选告警是否是该故障工单的根因。只学习因果关系，不依据RID或IP。",
        f"候选标题：{record['title']}",
        f"候选原因：{record['reason']}",
        f"位置结构：{record['location']}",
        f"设备：{record['device']} 厂商：{record['vendor']}",
        f"设备类型：{record['device_type']} 单板类型：{record['board_type']}",
        f"补充原因：{record['cause']} 制式：{record['radio']} 部署：{record['deployment']}",
        f"近30分钟序列：{record['timeline']}",
        f"目标告警：{record['target_summary']}",
        "直接相邻告警：" + " | ".join(record.get("neighbor_titles", [])),
        "工单高分上下文：" + " | ".join(clean_text(value, 120) for value in context_titles),
        "结论：",
    ]
    return "\n".join(fields)


def order_prompts(records: list[dict], v11: np.ndarray, context_count: int = 8) -> list[str]:
    ranked = np.argsort(-v11, kind="stable")[:context_count]
    titles = [records[int(index)]["title"] for index in ranked]
    return [candidate_prompt(record, titles) for record in records]


def resolve_device(requested: str = "auto"):
    import torch

    if requested != "auto":
        return torch.device(requested)
    # torch_npu registers the ``torch.npu`` backend at import time. Some entry
    # points resolve the device before importing PEFT/Accelerate, so relying on
    # those packages to perform this registration silently falls back to CPU.
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_qwen(model_path: Path, config: dict, dapt_adapter: Path | None = None, trainable=True):
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if dapt_adapter is not None:
        model = PeftModel.from_pretrained(model, dapt_adapter, is_trainable=False)
        model = model.merge_and_unload()
    if trainable:
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    return tokenizer, model


def last_token_embedding(model, tokenizer, prompts, device, max_length: int, gradients=False):
    import torch

    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    scope = contextlib.nullcontext() if gradients else torch.no_grad()
    with scope:
        output = model(**encoded, output_hidden_states=True, return_dict=True)
        hidden = output.hidden_states[-1]
        indices = encoded["attention_mask"].sum(dim=1) - 1
        pooled = hidden[torch.arange(len(indices), device=device), indices]
    return pooled.float()


def make_set_ranker(embedding_dim: int, numeric_dim: int, config: dict):
    import torch
    from torch import nn

    class SetRanker(nn.Module):
        def __init__(self):
            super().__init__()
            hidden = config["set_hidden_dim"]
            self.semantic = nn.Linear(embedding_dim, hidden)
            self.numeric = nn.Sequential(
                nn.Linear(numeric_dim, hidden), nn.GELU(), nn.Dropout(config["dropout"])
            )
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=config["set_heads"],
                dim_feedforward=hidden * 4,
                dropout=config["dropout"],
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, config["set_layers"])
            self.score = nn.Linear(hidden, 1)
            self.count = nn.Linear(hidden, 8)

        def forward(self, embeddings, numeric):
            values = self.semantic(embeddings) + self.numeric(numeric)
            values = self.encoder(values.unsqueeze(0)).squeeze(0)
            return {
                "logits": self.score(values).squeeze(-1),
                "count_logits": self.count(values.mean(dim=0)),
            }

    return SetRanker()


def multitask_loss(output, labels, base_mask, count: int, weights: dict):
    import torch
    import torch.nn.functional as functional

    logits = output["logits"]
    labels = labels.float()
    bce = functional.binary_cross_entropy_with_logits(logits, labels)
    positive, negative = logits[labels > 0.5], logits[labels <= 0.5]
    if len(positive) and len(negative):
        pairwise = functional.softplus(-(positive[:, None] - negative[None, :])).mean()
    else:
        pairwise = logits.sum() * 0.0
    count_target = torch.tensor(min(max(count, 1), 8) - 1, device=logits.device)
    count_loss = functional.cross_entropy(output["count_logits"].unsqueeze(0), count_target.unsqueeze(0))
    probability = torch.sigmoid(logits)
    preserve = functional.binary_cross_entropy(probability, base_mask.float())
    total = (
        weights["bce"] * bce
        + weights["pairwise"] * pairwise
        + weights["count"] * count_loss
        + weights["preserve"] * preserve
    )
    return total, {
        "bce": float(bce.detach().cpu()),
        "pairwise": float(pairwise.detach().cpu()),
        "count": float(count_loss.detach().cpu()),
        "preserve": float(preserve.detach().cpu()),
    }
