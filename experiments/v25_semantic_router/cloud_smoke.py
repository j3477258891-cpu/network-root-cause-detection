"""One-example Qwen LoRA forward/backward and NPU memory preflight."""

from __future__ import annotations

import argparse
from pathlib import Path

from semantic_model import last_token_embedding, load_qwen, order_prompts, resolve_device
from v25_common import Bundle, read_json, seed_everything, write_json


def peak_memory(device):
    import torch

    if device.type == "npu" and hasattr(torch, "npu"):
        return int(torch.npu.max_memory_allocated())
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated())
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    import torch
    import transformers
    import peft

    config = read_json(args.config)
    seed_everything(config["seeds"][0])
    device = resolve_device(args.device)
    if device.type == "cpu":
        raise SystemExit("V25 Qwen smoke requires NPU or CUDA")
    bundle = Bundle.load(args.data_root)
    tokenizer, model = load_qwen(args.model, config["model"], trainable=True)
    model.to(device).train()
    order = bundle.records["train"][0]
    start, stop = map(int, bundle.ptr("train")[:2])
    prompt = order_prompts(order["alarms"], bundle.arrays["train_v11"][start:stop])[0]
    embedding = last_token_embedding(
        model, tokenizer, [prompt], device, config["model"]["max_length"], gradients=True
    )
    loss = embedding.square().mean()
    loss.backward()
    finite = bool(torch.isfinite(loss).item()) and all(
        value.grad is None or bool(torch.isfinite(value.grad).all().item())
        for value in model.parameters() if value.requires_grad
    )
    report = {
        "version": config["version"],
        "device": str(device),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "trainable_parameters": sum(value.numel() for value in model.parameters() if value.requires_grad),
        "loss": float(loss.detach().cpu()),
        "finite_gradients": finite,
        "peak_memory_bytes": peak_memory(device),
    }
    write_json(args.output, report)
    print(report)
    if not finite:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
