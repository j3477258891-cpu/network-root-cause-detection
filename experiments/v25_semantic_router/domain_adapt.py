"""Unsupervised domain LoRA over train and unlabeled test alarm language."""

from __future__ import annotations

import argparse
import faulthandler
import json
import signal
from pathlib import Path

from semantic_model import candidate_prompt, load_qwen, resolve_device
from v25_common import Bundle, read_json, seed_everything, write_json


def main():
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    import torch

    config = read_json(args.config)
    model_config = config["model"]
    seed_everything(config["seeds"][0])
    device = resolve_device(args.device)
    print(f"DAPT device={device}", flush=True)
    bundle = Bundle.load(args.data_root)
    print("DAPT bundle loaded", flush=True)
    tokenizer, model = load_qwen(args.model, model_config, trainable=True)
    print("DAPT model loaded on CPU", flush=True)
    model.to(device).train()
    print(f"DAPT model moved to {device}", flush=True)
    optimizer = torch.optim.AdamW(
        (value for value in model.parameters() if value.requires_grad),
        lr=model_config["learning_rate"],
        weight_decay=model_config["weight_decay"],
    )
    texts = [
        candidate_prompt(record)
        for split in ("train", "test")
        for order in bundle.records[split]
        for record in order["alarms"]
    ]
    print(f"DAPT examples={len(texts)} epochs={args.epochs}", flush=True)
    steps = 0
    for _ in range(args.epochs):
        for text in texts:
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=model_config["max_length"],
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded, labels=encoded["input_ids"])
            (output.loss / model_config["gradient_accumulation"]).backward()
            steps += 1
            if steps == 1 or steps % 50 == 0:
                print(
                    f"DAPT step={steps}/{len(texts) * args.epochs} loss={float(output.loss.detach().cpu()):.6f}",
                    flush=True,
                )
            if steps % model_config["gradient_accumulation"] == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    if steps % model_config["gradient_accumulation"]:
        optimizer.step()
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output / "adapter")
    tokenizer.save_pretrained(args.output / "adapter")
    write_json(
        args.output / "report.json",
        {
            "version": config["version"],
            "examples": len(texts),
            "epochs": args.epochs,
            "device": str(device),
            "labels_used": False,
            "test_usage": "unlabeled causal language modeling only",
        },
    )


if __name__ == "__main__":
    main()
