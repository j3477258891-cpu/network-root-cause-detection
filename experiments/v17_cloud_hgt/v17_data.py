"""Compact graph store and batching for V17."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from materialize_dataset import materialize_data_root


class GraphStore:
    def __init__(self, data_root: Path, split: str):
        self.data_root = materialize_data_root(Path(data_root))
        self.arrays = np.load(self.data_root / "rootcause_v17_graphs.npz")
        self.metadata = json.loads((self.data_root / "metadata.json").read_text(encoding="utf-8"))
        self.split = split
        self.x = self.arrays[f"{split}_x"]
        self.v11 = self.arrays[f"{split}_v11"]
        self.order_ptr = self.arrays[f"{split}_order_ptr"]
        self.edge_ptr = self.arrays[f"{split}_edge_ptr"]
        self.edge_src = self.arrays[f"{split}_edge_src"]
        self.edge_dst = self.arrays[f"{split}_edge_dst"]
        self.edge_type = self.arrays[f"{split}_edge_type"]
        self.labels = self.arrays["train_labels"] if split == "train" else None
        self.counts = self.arrays["train_counts"] if split == "train" else None
        self.folds = self.arrays["train_folds"] if split == "train" else None

    def __len__(self):
        return len(self.order_ptr) - 1

    def graph(self, order_index):
        node_start = int(self.order_ptr[order_index])
        node_stop = int(self.order_ptr[order_index + 1])
        edge_start = int(self.edge_ptr[order_index])
        edge_stop = int(self.edge_ptr[order_index + 1])
        output = {
            "order_index": int(order_index),
            "x": self.x[node_start:node_stop],
            "v11": self.v11[node_start:node_stop],
            "edge_src": self.edge_src[edge_start:edge_stop] - node_start,
            "edge_dst": self.edge_dst[edge_start:edge_stop] - node_start,
            "edge_type": self.edge_type[edge_start:edge_stop],
        }
        if self.labels is not None:
            output["labels"] = self.labels[node_start:node_stop]
            output["count"] = int(self.counts[order_index])
        return output


def batch_graphs(store, order_indices, device):
    import torch

    graphs = [store.graph(int(index)) for index in order_indices]
    xs, bases, sources, targets, relations = [], [], [], [], []
    labels, counts, ptr = [], [], [0]
    node_offset = 0
    for graph in graphs:
        xs.append(graph["x"])
        bases.append(graph["v11"])
        sources.append(graph["edge_src"] + node_offset)
        targets.append(graph["edge_dst"] + node_offset)
        relations.append(graph["edge_type"])
        if "labels" in graph:
            labels.append(graph["labels"])
            counts.append(graph["count"] - 1)
        node_offset += len(graph["x"])
        ptr.append(node_offset)
    output = {
        "x": torch.as_tensor(np.concatenate(xs), dtype=torch.float32, device=device),
        "v11": torch.as_tensor(np.concatenate(bases), dtype=torch.float32, device=device),
        "edge_src": torch.as_tensor(np.concatenate(sources), dtype=torch.long, device=device),
        "edge_dst": torch.as_tensor(np.concatenate(targets), dtype=torch.long, device=device),
        "edge_type": torch.as_tensor(np.concatenate(relations), dtype=torch.long, device=device),
        "ptr": torch.as_tensor(ptr, dtype=torch.long, device=device),
        "order_indices": np.asarray(order_indices, dtype=np.int64),
    }
    if labels:
        output["labels"] = torch.as_tensor(
            np.concatenate(labels), dtype=torch.float32, device=device
        )
        output["counts"] = torch.as_tensor(counts, dtype=torch.long, device=device)
    return output


def shuffled_batches(indices, batch_size, rng):
    indices = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]
