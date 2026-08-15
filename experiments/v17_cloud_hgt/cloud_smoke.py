"""One-batch forward/backward test for the selected PyTorch NPU image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from train_fold import resolve_device, seed_everything
from v17_data import GraphStore, batch_graphs
from v17_model import ResidualRGT, pairwise_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    seed_everything(20260803)
    device = resolve_device(args.device)
    store = GraphStore(args.data_root, "train")
    batch = batch_graphs(store, np.arange(4), device)
    model = ResidualRGT(store.x.shape[1]).to(device)
    output = model(batch)
    bce = F.binary_cross_entropy_with_logits(output["logits"], batch["labels"])
    pair = pairwise_loss(output["logits"], batch["labels"], batch["v11"], batch["ptr"], 4)
    count = F.cross_entropy(output["count_logits"], batch["counts"])
    loss = 0.45 * bce + 0.4 * pair + 0.15 * count
    loss.backward()
    assert torch.isfinite(loss)
    report = {
        "ok": True,
        "torch": torch.__version__,
        "device": str(device),
        "nodes": int(batch["x"].shape[0]),
        "edges": int(batch["edge_src"].shape[0]),
        "loss": float(loss.detach().cpu()),
        "npu_available": bool(hasattr(torch, "npu") and torch.npu.is_available()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
