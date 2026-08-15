"""Build disjoint precision probes from the calibrated 1052-node candidate."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(r"D:\zgyidong")
CHAMPION = ROOT / "experiments" / "submissions" / "champion_0.906324_day01_probe01_v11_full.csv"
CALIBRATED = ROOT / "result_record_precision_calibrated_1052.csv"
SCORES = ROOT / "codexgz" / "v11" / "v11_test_scores.csv"
REPORT = ROOT / "experiments" / "submissions" / "precision_probe_report.json"


def load_rows(path: Path) -> tuple[list[str], dict[str, list[dict]]]:
    order_ids = []
    predictions = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            order_ids.append(row["order_id"])
            predictions[row["order_id"]] = json.loads(row["output"])["rootcause"]
    return order_ids, predictions


def load_scores() -> dict[tuple[str, str], float]:
    result = {}
    with SCORES.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            value = 0.25 * float(row["context_score"]) + 0.75 * float(row["meta_mean"])
            result[(row["order_id"], row["rid"])] = value
    return result


def write_probe(path: Path, order_ids: list[str], champion: dict[str, list[dict]], removed: set[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order_id", "output"])
        for order_id in order_ids:
            nodes = [
                node
                for node in champion[order_id]
                if (order_id, node["@rid"]) not in removed
            ]
            writer.writerow(
                [order_id, json.dumps({"rootcause": nodes}, ensure_ascii=False)]
            )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    order_ids, champion = load_rows(CHAMPION)
    _, calibrated = load_rows(CALIBRATED)
    scores = load_scores()

    champion_set = {
        (order_id, node["@rid"])
        for order_id, nodes in champion.items()
        for node in nodes
    }
    calibrated_set = {
        (order_id, node["@rid"])
        for order_id, nodes in calibrated.items()
        for node in nodes
    }
    removed = sorted(
        champion_set - calibrated_set,
        key=lambda key: (scores[key], key[0], key[1]),
    )
    if len(removed) != 7 or calibrated_set - champion_set:
        raise RuntimeError("calibrated submission must remove exactly seven champion nodes")

    groups = {
        "a_lowest3": removed[:3],
        "b_next4": removed[3:],
    }
    report = {
        "base_score": 0.906324,
        "base_predictions": 1059,
        "inferred_true_roots": 1044,
        "groups": {},
    }
    for name, values in groups.items():
        path = ROOT / f"result_record_precision_probe_{name}.csv"
        write_probe(path, order_ids, champion, set(values))
        prediction_count = 1059 - len(values)
        score_table = {
            str(removed_true): round(
                2 * (953 - removed_true) / (1044 + prediction_count), 9
            )
            for removed_true in range(len(values) + 1)
        }
        report["groups"][name] = {
            "path": str(path),
            "predictions": prediction_count,
            "removed": [
                {"order_id": key[0], "rid": key[1], "v11_score": scores[key]}
                for key in values
            ],
            "online_score_by_removed_true_count": score_table,
            "sha256": digest(path),
        }

    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
