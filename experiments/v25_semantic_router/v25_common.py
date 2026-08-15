"""Shared data contracts, decoding helpers, and validation for V25."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


IP_PATTERN = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
UUID_PATTERN = re.compile(
    r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
)
STATION_PATTERNS = (
    re.compile(r"(?i)(ManagedElement|NodeMe|eNodeB|gNodeB|DeviceId|OfficeId)\s*[=:]\s*([A-Za-z0-9_-]+)"),
    re.compile(r"(?i)(?:网元|基站)(?:ID|编号)?\s*[=:：]\s*([A-Za-z0-9_-]+)"),
)
LOCATION_KEYS = {
    "managedelement": "STATION",
    "nodeme": "STATION",
    "enodeb": "STATION",
    "gnodeb": "STATION",
    "deviceid": "DEVICE",
    "officeid": "STATION",
    "slot": "SLOT",
    "槽号": "SLOT",
    "port": "PORT",
    "端口": "PORT",
    "端口号": "PORT",
    "rack": "RACK",
    "柜号": "RACK",
    "框号": "FRAME",
    "replaceableunit": "UNIT",
    "board": "BOARD",
}


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def scalar(value) -> str:
    if isinstance(value, list):
        return " ".join(map(str, value))
    return str(value or "")


def clean_text(value, limit: int = 1200) -> str:
    text = scalar(value).replace("\x00", " ")
    text = UUID_PATTERN.sub("<RID>", text)
    text = IP_PATTERN.sub("<IP>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def station_ids(value) -> tuple[str, ...]:
    text = scalar(value)
    found = set()
    for pattern in STATION_PATTERNS:
        for match in pattern.finditer(text):
            found.add(match.group(match.lastindex or 1).upper())
    return tuple(sorted(found))


def normalized_location(value) -> str:
    """Keep location semantics while removing IP/RID and untyped free numbers."""
    text = clean_text(value, 1800)
    parts = re.split(r"[,;\n]", text)
    output = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        match = re.match(r"([^=:：]+)\s*[=:：]\s*(.*)", part)
        if not match:
            output.append(re.sub(r"\d+", "<NUM>", part))
            continue
        raw_key, raw_value = match.group(1).strip(), match.group(2).strip()
        key = LOCATION_KEYS.get(raw_key.lower(), LOCATION_KEYS.get(raw_key, raw_key.upper()))
        value_text = re.sub(r"\d+", "<NUM>", raw_value)
        output.append(f"{key}={value_text}")
    return ";".join(output)[:800]


def parse_addinfo(value) -> dict[str, str]:
    text = scalar(value)
    fields = {}
    for key, raw in re.findall(r"(?:^|;)([A-Za-z][A-Za-z0-9_-]*):([^;]*)", text):
        fields.setdefault(key.lower(), clean_text(raw, 200))
    return {
        "device_type": fields.get("devicetype", ""),
        "board_type": fields.get("boardtype", ""),
        "cause": fields.get("cause", ""),
        "radio": fields.get("radio", ""),
        "deployment": fields.get("deployment", ""),
    }


def time_summary(node: dict) -> str:
    values = node.get("timeLists", [])
    if not isinstance(values, list):
        values = []
    timeline = [int(bool(value)) for value in values[:6]]
    while len(timeline) < 6:
        timeline.append(0)
    return "".join(map(str, timeline))


def order_signature(alarms: list[dict]) -> tuple:
    titles = Counter(scalar(node.get("title")) for node in alarms)
    targets = sorted(
        scalar(node.get("title"))
        for node in alarms
        if node.get("label") == "TargetAlarm"
    )
    return tuple(sorted(titles.items())), tuple(targets), len(alarms)


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def connected_group_folds(orders: list[dict], fold_count: int = 5) -> tuple[np.ndarray, dict]:
    """Group orders sharing an exact template or a station identity."""
    union = UnionFind(len(orders))
    owners: dict[tuple, int] = {}
    for index, order in enumerate(orders):
        keys = [("signature", repr(order["signature"]))]
        keys.extend(("station", value) for value in order.get("station_ids", []))
        for key in keys:
            if key in owners:
                union.union(index, owners[key])
            else:
                owners[key] = index
    components = defaultdict(list)
    for index in range(len(orders)):
        components[union.find(index)].append(index)
    fold_sizes = [0] * fold_count
    folds = np.zeros(len(orders), dtype=np.int8)
    ranked = sorted(
        components.values(),
        key=lambda rows: (-len(rows), hashlib.sha256(repr(rows).encode()).hexdigest()),
    )
    for rows in ranked:
        fold = min(range(fold_count), key=lambda item: (fold_sizes[item], item))
        folds[rows] = fold
        fold_sizes[fold] += len(rows)
    return folds, {
        "component_count": len(components),
        "fold_sizes": fold_sizes,
        "largest_component": max(map(len, components.values()), default=0),
    }


def balanced_group_folds(keys: list[tuple], fold_count: int = 5) -> tuple[np.ndarray, dict]:
    groups = defaultdict(list)
    for index, key in enumerate(keys):
        groups[repr(key)].append(index)
    fold_sizes = [0] * fold_count
    folds = np.zeros(len(keys), dtype=np.int8)
    ranked = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), hashlib.sha256(item[0].encode()).hexdigest()),
    )
    for _, rows in ranked:
        fold = min(range(fold_count), key=lambda item: (fold_sizes[item], item))
        folds[rows] = fold
        fold_sizes[fold] += len(rows)
    return folds, {
        "group_count": len(groups),
        "fold_sizes": fold_sizes,
        "largest_group": max(map(len, groups.values()), default=0),
    }


@dataclass
class Bundle:
    root: Path
    arrays: dict[str, np.ndarray]
    metadata: dict
    records: dict[str, list[dict]]

    @classmethod
    def load(cls, root: Path) -> "Bundle":
        root = Path(root)
        with np.load(root / "v25_semantic_router.npz", allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        metadata = read_json(root / "metadata.json")
        with gzip.open(root / "semantic_records.json.gz", "rt", encoding="utf-8") as handle:
            records = json.load(handle)
        return cls(root, arrays, metadata, records)

    def ptr(self, split: str) -> np.ndarray:
        return self.arrays[f"{split}_alarm_ptr"]

    def rows_for_orders(self, split: str, order_indices) -> np.ndarray:
        ptr = self.ptr(split)
        chunks = [np.arange(ptr[i], ptr[i + 1]) for i in order_indices]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)


def exact_count_mask(scores, ptr, target_count: int, max_roots: int = 8) -> np.ndarray:
    selected = np.zeros(len(scores), dtype=bool)
    optional = []
    for start, stop in zip(ptr[:-1], ptr[1:]):
        start, stop = int(start), int(stop)
        ranked = np.argsort(-scores[start:stop], kind="stable")[:max_roots]
        if not len(ranked):
            raise ValueError((start, stop))
        selected[start + ranked[0]] = True
        optional.extend((start + ranked[1:]).tolist())
    remaining = target_count - int(selected.sum())
    if not 0 <= remaining <= len(optional):
        raise ValueError((target_count, int(selected.sum()), len(optional)))
    optional = np.asarray(optional, dtype=np.int64)
    optional = optional[np.argsort(-scores[optional], kind="stable")]
    selected[optional[:remaining]] = True
    return selected


def fixed_k_mask(scores, ptr, counts, max_roots: int = 8) -> np.ndarray:
    selected = np.zeros(len(scores), dtype=bool)
    for index, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:])):
        start, stop = int(start), int(stop)
        count = min(int(counts[index]), max_roots, stop - start)
        selected[start + np.argsort(-scores[start:stop], kind="stable")[:count]] = True
    return selected


def load_locks(path: Path) -> dict[str, set[tuple[str, str]]]:
    report = read_json(path)
    return {
        "in": {(item["order_id"], item["rid"]) for item in report["forced_in"]},
        "out": {(item["order_id"], item["rid"]) for item in report["forced_out"]},
    }


def bootstrap_lower(deltas, iterations: int = 20000, seed: int = 20260804) -> float:
    values = np.asarray(deltas)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).sum(axis=1)
    return float(np.quantile(samples, 0.025))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_submission(path: Path, test_records: list[dict], total: int = 1059) -> dict:
    expected = {order["order_id"]: order for order in test_records}
    seen, count = set(), 0
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["order_id", "output"]:
            raise ValueError(reader.fieldnames)
        for row in reader:
            order_id = row["order_id"]
            if order_id in seen or order_id not in expected:
                raise ValueError(f"invalid order: {order_id}")
            seen.add(order_id)
            roots = json.loads(row["output"])["rootcause"]
            if not 1 <= len(roots) <= 8:
                raise ValueError((order_id, len(roots)))
            nodes = {item["rid"]: item for item in expected[order_id]["alarms"]}
            root_rids = [item["@rid"] for item in roots]
            if len(set(root_rids)) != len(root_rids):
                raise ValueError(f"duplicate RID: {order_id}")
            for item in roots:
                source = nodes.get(item["@rid"])
                if source is None:
                    raise ValueError((order_id, item["@rid"]))
                for field in ("title", "location", "reason"):
                    if item.get(field, "") != source.get(field, ""):
                        raise ValueError((order_id, item["@rid"], field))
            count += len(roots)
    if seen != set(expected) or count != total:
        raise ValueError((len(seen), len(expected), count, total))
    return {"orders": len(seen), "predictions": count, "sha256": sha256(path)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if hasattr(torch, "npu"):
            torch.npu.manual_seed_all(seed)
    except ImportError:
        pass
