"""Full-scale worst-order forward/backward and memory smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_fold import exact_count_mask, make_model, resolve_device, seed_everything
from v24_data import GraphStore, batch_graphs
from v24_model import multitask_loss


def peak_memory(device):
    if device.type == "npu" and hasattr(torch, "npu"):
        return int(torch.npu.max_memory_allocated())
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated())
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config_probe.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed_everything(config["seeds"][0])
    device = resolve_device(args.device)
    train = GraphStore(args.data_root, "train")
    test = GraphStore(args.data_root, "test")
    target = round(config["target_test_predictions"] / len(test) * len(train))
    train.base_mask = exact_count_mask(train.v11, train.alarm_ptr, target, config["max_rootcauses"])
    node_counts = np.diff(train.node_ptr)
    edge_counts = np.diff(train.edge_ptr)
    alarm_counts = np.diff(train.alarm_ptr)
    indices = np.unique(
        [int(np.argmax(node_counts)), int(np.argmax(edge_counts)), int(np.argmax(alarm_counts))]
    )
    batch = batch_graphs(train, indices, device)
    model = make_model(config, train, device)
    positive_weight = torch.tensor(config["positive_weight"], dtype=torch.float32, device=device)
    output = model(batch)
    loss, parts = multitask_loss(
        output,
        batch,
        config["loss_weights"],
        positive_weight,
        config["hard_negatives_per_positive"],
    )
    loss.backward()
    finite = bool(torch.isfinite(loss).item()) and all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )
    report = {
        "version": config["version"],
        "device": str(device),
        "order_indices": indices.tolist(),
        "nodes": int(batch["node_type"].shape[0]),
        "edges": int(batch["edge_type"].shape[0]),
        "alarms": int(batch["v11"].shape[0]),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "loss": float(loss.detach().cpu()),
        "loss_parts": parts,
        "finite_gradients": finite,
        "peak_memory_bytes": peak_memory(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not finite:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
