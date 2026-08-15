"""In-memory graph store and per-order batching for V24."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class GraphStore:
    def __init__(self, data_root: Path, split: str):
        self.data_root = Path(data_root)
        with np.load(self.data_root / "rootcause_v24_full_hetero.npz") as archive:
            self.arrays = {name: archive[name] for name in archive.files}
        self.metadata = json.loads(
            (self.data_root / "metadata.json").read_text(encoding="utf-8")
        )
        self.split = split
        for name in (
            "node_type", "node_numeric", "edge_src", "edge_dst", "edge_type",
            "alarm_node_index", "alarm_feature_index", "path_node_type",
            "path_edge_type", "path_length", "node_ptr", "edge_ptr", "alarm_ptr",
            "alarm_x", "v11",
        ):
            setattr(self, name, self.arrays[f"{split}_{name}"])
        self.labels = self.arrays["train_labels"] if split == "train" else None
        self.counts = self.arrays["train_counts"] if split == "train" else None
        self.folds = self.arrays["train_folds"] if split == "train" else None
        self.champion_mask = (
            self.arrays["test_champion_mask"].astype(bool) if split == "test" else None
        )
        self.base_mask = None

    def __len__(self):
        return len(self.node_ptr) - 1

    def graph(self, order_index: int) -> dict:
        ns, ne = int(self.node_ptr[order_index]), int(self.node_ptr[order_index + 1])
        es, ee = int(self.edge_ptr[order_index]), int(self.edge_ptr[order_index + 1])
        als, ale = int(self.alarm_ptr[order_index]), int(self.alarm_ptr[order_index + 1])
        output = {
            "order_index": order_index,
            "node_type": self.node_type[ns:ne],
            "node_numeric": self.node_numeric[ns:ne],
            "edge_src": self.edge_src[es:ee] - ns,
            "edge_dst": self.edge_dst[es:ee] - ns,
            "edge_type": self.edge_type[es:ee],
            "alarm_node_index": self.alarm_node_index[als:ale] - ns,
            "alarm_x": self.alarm_x[als:ale],
            "path_node_type": self.path_node_type[als:ale],
            "path_edge_type": self.path_edge_type[als:ale],
            "path_length": self.path_length[als:ale],
            "v11": self.v11[als:ale],
        }
        if self.labels is not None:
            output["labels"] = self.labels[als:ale]
            output["count"] = int(self.counts[order_index])
        if self.base_mask is not None:
            output["base_mask"] = self.base_mask[als:ale]
        return output


def batch_graphs(store: GraphStore, order_indices, device) -> dict:
    import torch

    graphs = [store.graph(int(index)) for index in order_indices]
    values = {
        name: []
        for name in (
            "node_type", "node_numeric", "edge_src", "edge_dst", "edge_type",
            "alarm_node_index", "alarm_x", "path_node_type", "path_edge_type",
            "path_length", "v11", "labels", "base_mask",
        )
    }
    node_ptr, alarm_ptr = [0], [0]
    node_offset = 0
    for graph in graphs:
        for name in ("node_type", "node_numeric", "edge_type", "alarm_x", "path_node_type", "path_edge_type", "path_length", "v11"):
            values[name].append(graph[name])
        values["edge_src"].append(graph["edge_src"] + node_offset)
        values["edge_dst"].append(graph["edge_dst"] + node_offset)
        values["alarm_node_index"].append(graph["alarm_node_index"] + node_offset)
        if "labels" in graph:
            values["labels"].append(graph["labels"])
        if "base_mask" in graph:
            values["base_mask"].append(graph["base_mask"])
        node_offset += len(graph["node_type"])
        node_ptr.append(node_offset)
        alarm_ptr.append(alarm_ptr[-1] + len(graph["alarm_node_index"]))

    def tensor(name, dtype):
        return torch.as_tensor(np.concatenate(values[name]), dtype=dtype, device=device)

    output = {
        "node_type": tensor("node_type", torch.long),
        "node_numeric": tensor("node_numeric", torch.float32),
        "edge_src": tensor("edge_src", torch.long),
        "edge_dst": tensor("edge_dst", torch.long),
        "edge_type": tensor("edge_type", torch.long),
        "alarm_node_index": tensor("alarm_node_index", torch.long),
        "alarm_x": tensor("alarm_x", torch.float32),
        "path_node_type": torch.as_tensor(np.concatenate(values["path_node_type"]), dtype=torch.long, device=device),
        "path_edge_type": torch.as_tensor(np.concatenate(values["path_edge_type"]), dtype=torch.long, device=device),
        "path_length": tensor("path_length", torch.long),
        "v11": tensor("v11", torch.float32),
        "node_ptr": torch.as_tensor(node_ptr, dtype=torch.long, device=device),
        "alarm_ptr": torch.as_tensor(alarm_ptr, dtype=torch.long, device=device),
        "order_indices": np.asarray(order_indices, dtype=np.int64),
    }
    if values["labels"]:
        output["labels"] = tensor("labels", torch.float32)
    if values["base_mask"]:
        output["base_mask"] = tensor("base_mask", torch.float32)
    return output


def shuffled_batches(indices, batch_size: int, rng):
    indices = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]

