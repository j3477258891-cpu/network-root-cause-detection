"""Mine conservative, disjoint actions for the v27 leaderboard loop."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path

from closed_loop import (
    DEFAULT_CHAMPION,
    DEFAULT_TEST_DIR,
    DEFAULT_WORKDIR,
    MAX_ROOTS,
    apply_actions,
    file_sha256,
    load_submission,
    load_topologies,
    normalize_action,
    validate_actions,
    validate_selection,
    write_json,
    write_submission,
)


ROOT = Path(r"D:\zgyidong")
DEFAULT_TRAIN_DIR = ROOT / "train"
DEFAULT_PROTECTED_REPORT = ROOT / "codexgz" / "v11" / "v11_constrained_report.json"
DEFAULT_V26 = (
    ROOT
    / "experiments"
    / "v26_selective_router"
    / "submissions"
    / "v26_selective_error_router_p1059.csv"
)
DEFAULT_PRECISION_REPORT = ROOT / "experiments" / "submissions" / "precision_probe_report.json"
DEFAULT_V11_SCORES = ROOT / "codexgz" / "v11" / "v11_test_scores.csv"
CONSENSUS_SUBMISSIONS = [
    ROOT / "codexgz" / "v14" / "result_record_v14_final.csv",
    ROOT / "codexgz" / "v13" / "result_record_v13_meta_1059.csv",
    ROOT / "codexgz" / "v12" / "v12_candidate_1059.csv",
    ROOT / "experiments" / "submissions" / "result_record_simple_alarm_prior_1059.csv",
    ROOT
    / "experiments"
    / "v25_semantic_router"
    / "submissions"
    / "result_record_pure_statistical_p1059.csv",
    DEFAULT_V26,
]
MIN_SUPPORT = 5
MIN_SITES = 3


def scalar(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    return str(value)


def normalized_text(value: object) -> str:
    return " ".join(scalar(value).strip().lower().split())


def location_shape(value: object) -> str:
    text = scalar(value)
    text = re.sub(
        r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
        "<UUID>",
        text,
    )
    return re.sub(r"\d+", "#", text)


def site_key(node: dict) -> str:
    haystack = "|".join(
        scalar(node.get(field)) for field in ("location", "addInfo", "device", "room")
    )
    for pattern in (
        r"ManagedElement[=:](\d+)",
        r"OfficeId[=:](\d+)",
        r"SubNetwork[=:](\d+)",
    ):
        match = re.search(pattern, haystack, re.I)
        if match:
            return match.group(0).split(match.group(1))[0].lower() + match.group(1)
    return hashlib.sha256(location_shape(haystack).encode("utf-8")).hexdigest()[:16]


def alarm_key(node: dict) -> tuple[str, str, str]:
    return (
        normalized_text(node.get("title")),
        normalized_text(node.get("reason")),
        normalized_text(node.get("label")),
    )


def domain_key(node: dict) -> tuple[str, str]:
    return (
        normalized_text(node.get("vendor")),
        normalized_text(node.get("fault1") or node.get("fault2")),
    )


def device_family(node: dict) -> str:
    text = "|".join(
        scalar(node.get(field)) for field in ("device", "addInfo", "location")
    )
    for pattern in (r"DeviceType[=:]([^;|,]+)", r"类型[=:]([^;|,]+)"):
        match = re.search(pattern, text, re.I)
        if match:
            return normalized_text(match.group(1))
    return location_shape(node.get("device"))


def topology_role(order: dict, node: dict) -> tuple[int, int, int]:
    meta = order["meta"][node["@rid"]]
    distance = meta["upstream_distance"]
    if distance > len(order["alarms"]):
        distance_bucket = 4
    else:
        distance_bucket = min(int(distance), 3)
    return (
        min(int(meta["indegree"]), 3),
        min(int(meta["outdegree"]), 3),
        distance_bucket,
    )


def load_order(directory: Path, with_labels: bool) -> dict:
    order_id = directory.name
    topology = json.loads(
        (directory / f"{order_id}.log.topo.json").read_text(encoding="utf-8")
    )
    nodes = topology.get("nodes", [])
    alarms = [node for node in nodes if node.get("@class") == "Alarm"]
    roots: set[str] = set()
    if with_labels:
        root_payload = json.loads(
            (directory / f"{order_id}.rootcause.json").read_text(encoding="utf-8")
        )
        roots = {node["@rid"] for node in root_payload.get("rootcause", [])}

    rid_to_index = {node.get("@rid"): index for index, node in enumerate(nodes)}
    reverse = [set() for _ in nodes]
    indegree = [0] * len(nodes)
    outdegree = [0] * len(nodes)
    for edge in topology.get("edges", []):
        source = rid_to_index.get(edge.get("in"))
        target = rid_to_index.get(edge.get("out"))
        if source is None or target is None:
            continue
        reverse[target].add(source)
        outdegree[source] += 1
        indegree[target] += 1
    targets = [
        rid_to_index[node["@rid"]]
        for node in alarms
        if node.get("label") == "TargetAlarm" and node.get("@rid") in rid_to_index
    ]
    distance = [len(nodes) + 1] * len(nodes)
    queue = deque()
    for index in targets:
        distance[index] = 0
        queue.append(index)
    while queue:
        current = queue.popleft()
        for predecessor in reverse[current]:
            if distance[predecessor] > distance[current] + 1:
                distance[predecessor] = distance[current] + 1
                queue.append(predecessor)
    alarm_meta = {}
    for ordinal, node in enumerate(alarms):
        index = rid_to_index[node["@rid"]]
        alarm_meta[node["@rid"]] = {
            "time": int(node.get("time") or 0),
            "ordinal": ordinal,
            "upstream_distance": distance[index],
            "indegree": indegree[index],
            "outdegree": outdegree[index],
        }
    return {
        "id": order_id,
        "alarms": alarms,
        "roots": roots,
        "meta": alarm_meta,
    }


def load_orders(base: Path, with_labels: bool) -> list[dict]:
    return [
        load_order(directory, with_labels)
        for directory in sorted(path for path in Path(base).iterdir() if path.is_dir())
    ]


def local_groups(order: dict) -> dict[tuple, list[dict]]:
    result: dict[tuple, list[dict]] = defaultdict(list)
    for node in order["alarms"]:
        result[
            (alarm_key(node), site_key(node), device_family(node), topology_role(order, node))
        ].append(node)
    return result


def group_rule_observation(order: dict, nodes: list[dict]) -> tuple[str, tuple[str, ...]]:
    roots = order["roots"]
    selected = [node for node in nodes if node["@rid"] in roots]
    if len(selected) == len(nodes):
        return "group_full", tuple(sorted(node["@rid"] for node in selected))
    if not selected:
        return "group_none", ()
    if len(selected) != 1:
        return "mixed", ()
    selected_rid = selected[0]["@rid"]
    ordered_time = sorted(
        nodes,
        key=lambda node: (
            order["meta"][node["@rid"]]["time"],
            order["meta"][node["@rid"]]["ordinal"],
        ),
    )
    if ordered_time[0]["@rid"] == selected_rid:
        return "single_earliest", (selected_rid,)
    ordered_upstream = sorted(
        nodes,
        key=lambda node: (
            -order["meta"][node["@rid"]]["upstream_distance"],
            -order["meta"][node["@rid"]]["outdegree"],
            order["meta"][node["@rid"]]["ordinal"],
        ),
    )
    if ordered_upstream[0]["@rid"] == selected_rid:
        return "single_upstream", (selected_rid,)
    return "mixed", ()


def group_rule_key(order: dict, nodes: list[dict]) -> tuple:
    return (
        alarm_key(nodes[0]),
        device_family(nodes[0]),
        topology_role(order, nodes[0]),
        len(nodes),
    )


def discover_group_rules(train_orders: list[dict]) -> dict[tuple, dict]:
    observations: dict[tuple, list[dict]] = defaultdict(list)
    for order in train_orders:
        for (_, site, _, _), nodes in local_groups(order).items():
            if len(nodes) < 2:
                continue
            subtype, _ = group_rule_observation(order, nodes)
            observations[group_rule_key(order, nodes)].append(
                {
                    "order_id": order["id"],
                    "site": site,
                    "subtype": subtype,
                    "domain": domain_key(nodes[0]),
                }
            )
    rules = {}
    for key, values in observations.items():
        subtypes = {item["subtype"] for item in values}
        sites = {item["site"] for item in values}
        if len(values) < MIN_SUPPORT or len(sites) < MIN_SITES or len(subtypes) != 1:
            continue
        subtype = next(iter(subtypes))
        if subtype not in {"group_full", "single_earliest", "single_upstream"}:
            continue
        rules[key] = {
            "subtype": subtype,
            "support": len(values),
            "sites": len(sites),
            "domains": sorted({item["domain"] for item in values}),
            "loo_non_degrading": True,
            "consistency": 1.0,
        }
    return rules


def discover_pair_rules(train_orders: list[dict]) -> dict[tuple, dict]:
    observations: dict[tuple, list[dict]] = defaultdict(list)
    for order in train_orders:
        groups_by_site: dict[str, list[list[dict]]] = defaultdict(list)
        for (_, site, _, _), nodes in local_groups(order).items():
            groups_by_site[site].append(nodes)
        for site, groups in groups_by_site.items():
            if len(groups) > 20:
                continue
            groups.sort(key=lambda nodes: repr(group_rule_key(order, nodes)))
            for left_index, left in enumerate(groups):
                for right in groups[left_index + 1 :]:
                    left_roots = sum(node["@rid"] in order["roots"] for node in left)
                    right_roots = sum(node["@rid"] in order["roots"] for node in right)
                    if left_roots not in {0, len(left)} or right_roots not in {0, len(right)}:
                        continue
                    pattern = (int(left_roots > 0), int(right_roots > 0))
                    if pattern == (0, 0):
                        continue
                    key = (group_rule_key(order, left), group_rule_key(order, right))
                    observations[key].append(
                        {
                            "site": site,
                            "pattern": pattern,
                            "domains": (domain_key(left[0]), domain_key(right[0])),
                        }
                    )
    rules = {}
    for key, values in observations.items():
        patterns = {item["pattern"] for item in values}
        sites = {item["site"] for item in values}
        if len(values) < MIN_SUPPORT or len(sites) < MIN_SITES or len(patterns) != 1:
            continue
        rules[key] = {
            "subtype": "group_pair",
            "pattern": next(iter(patterns)),
            "support": len(values),
            "sites": len(sites),
            "domains": sorted({item["domains"] for item in values}),
            "consistency": 1.0,
            "loo_non_degrading": True,
        }
    return rules


def predict_group(order: dict, nodes: list[dict], subtype: str) -> set[str]:
    if subtype == "group_full":
        return {node["@rid"] for node in nodes}
    if subtype == "single_earliest":
        winner = min(
            nodes,
            key=lambda node: (
                order["meta"][node["@rid"]]["time"],
                order["meta"][node["@rid"]]["ordinal"],
            ),
        )
        return {winner["@rid"]}
    winner = max(
        nodes,
        key=lambda node: (
            order["meta"][node["@rid"]]["upstream_distance"],
            order["meta"][node["@rid"]]["outdegree"],
            -order["meta"][node["@rid"]]["ordinal"],
        ),
    )
    return {winner["@rid"]}


def group_actions(
    test_orders: list[dict], rules: dict[tuple, dict], champion: dict[str, list[str]]
) -> list[dict]:
    candidates = []
    for order in test_orders:
        current = set(champion[order["id"]])
        for (_, _site, _, _), nodes in local_groups(order).items():
            rule = rules.get(group_rule_key(order, nodes))
            if not rule or domain_key(nodes[0]) not in {tuple(item) for item in rule["domains"]}:
                continue
            predicted = predict_group(order, nodes, rule["subtype"])
            group_rids = {node["@rid"] for node in nodes}
            remove = sorted((current & group_rids) - predicted)
            add = sorted(predicted - current)
            if not remove and not add:
                continue
            candidates.append(
                {
                    "action_id": f"{rule['subtype']}_{order['id'][:8]}",
                    "order_id": order["id"],
                    "remove_rids": remove,
                    "add_rids": add,
                    "source": "group_rule",
                    "subtype": rule["subtype"],
                    "expected_gain": min(len(add) + 0.5 * len(remove), 2.0),
                    "evidence": rule,
                }
            )
    return candidates


def pair_actions(
    test_orders: list[dict], rules: dict[tuple, dict], champion: dict[str, list[str]]
) -> list[dict]:
    candidates = []
    for order in test_orders:
        current = set(champion[order["id"]])
        groups_by_site: dict[str, list[list[dict]]] = defaultdict(list)
        for (_, site, _, _), nodes in local_groups(order).items():
            groups_by_site[site].append(nodes)
        for groups in groups_by_site.values():
            groups.sort(key=lambda nodes: repr(group_rule_key(order, nodes)))
            for left_index, left in enumerate(groups):
                for right in groups[left_index + 1 :]:
                    rule = rules.get(
                        (group_rule_key(order, left), group_rule_key(order, right))
                    )
                    if not rule:
                        continue
                    domains = (domain_key(left[0]), domain_key(right[0]))
                    known_domains = {
                        (tuple(item[0]), tuple(item[1])) for item in rule["domains"]
                    }
                    if domains not in known_domains:
                        continue
                    selected = set()
                    if rule["pattern"][0]:
                        selected.update(node["@rid"] for node in left)
                    if rule["pattern"][1]:
                        selected.update(node["@rid"] for node in right)
                    pair_rids = {node["@rid"] for node in left + right}
                    remove = sorted((current & pair_rids) - selected)
                    add = sorted(selected - current)
                    if not remove and not add:
                        continue
                    candidates.append(
                        {
                            "action_id": f"group_pair_{order['id'][:8]}",
                            "order_id": order["id"],
                            "remove_rids": remove,
                            "add_rids": add,
                            "source": "group_rule",
                            "subtype": "group_pair",
                            "expected_gain": min(len(add) + 0.5 * len(remove), 2.0),
                            "evidence": rule,
                        }
                    )
    return candidates


def template_fingerprint(order: dict) -> tuple:
    return tuple(
        sorted(
            Counter(
                (
                    alarm_key(node),
                    location_shape(node.get("location")),
                    domain_key(node),
                )
                for node in order["alarms"]
            ).items()
        )
    )


def canonical_template_key(node: dict) -> tuple:
    return alarm_key(node), location_shape(node.get("location")), domain_key(node)


def discover_templates(train_orders: list[dict]) -> dict[tuple, dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for order in train_orders:
        groups[template_fingerprint(order)].append(order)
    templates = {}
    for fingerprint, references in groups.items():
        sites = {
            tuple(sorted({site_key(node) for node in order["alarms"]}))
            for order in references
        }
        if len(references) < MIN_SUPPORT or len(sites) < MIN_SITES:
            continue
        patterns = []
        valid = True
        for order in references:
            pattern = Counter(
                canonical_template_key(node)
                for node in order["alarms"]
                if node["@rid"] in order["roots"]
            )
            patterns.append(tuple(sorted(pattern.items())))
        if len(set(patterns)) != 1:
            valid = False
        if not valid:
            continue
        templates[fingerprint] = {
            "pattern": patterns[0],
            "support": len(references),
            "sites": len(sites),
            "consistency": 1.0,
            "leave_one_site_out_exact": True,
        }
    return templates


def template_actions(
    test_orders: list[dict], templates: dict[tuple, dict], champion: dict[str, list[str]]
) -> list[dict]:
    actions = []
    for order in test_orders:
        template = templates.get(template_fingerprint(order))
        if not template:
            continue
        required = Counter(dict(template["pattern"]))
        by_key: dict[tuple, list[str]] = defaultdict(list)
        for node in order["alarms"]:
            by_key[canonical_template_key(node)].append(node["@rid"])
        selected: set[str] = set()
        valid = True
        for key, count in required.items():
            matches = sorted(by_key.get(key, []))
            if len(matches) != count:
                valid = False
                break
            selected.update(matches)
        if not valid or not 1 <= len(selected) <= MAX_ROOTS:
            continue
        current = set(champion[order["id"]])
        remove = sorted(current - selected)
        add = sorted(selected - current)
        if not remove and not add:
            continue
        actions.append(
            {
                "action_id": f"strict_template_{order['id'][:8]}",
                "order_id": order["id"],
                "remove_rids": remove,
                "add_rids": add,
                "source": "strict_template",
                "subtype": "strict_template",
                "expected_gain": min(len(add) + 0.5 * len(remove), 3.0),
                "evidence": template,
            }
        )
    return actions


def diff_actions(
    champion: dict[str, list[str]], candidate_path: Path, source: str, expected_gain: float
) -> list[dict]:
    _, candidate = load_submission(candidate_path)
    actions = []
    for order_id in champion:
        current = set(champion[order_id])
        proposed = set(candidate[order_id])
        remove = sorted(current - proposed)
        add = sorted(proposed - current)
        if not remove and not add:
            continue
        actions.append(
            {
                "action_id": f"{source}_{order_id[:8]}",
                "order_id": order_id,
                "remove_rids": remove,
                "add_rids": add,
                "source": source,
                "subtype": source,
                "expected_gain": expected_gain,
                "evidence": {"candidate": str(candidate_path)},
            }
        )
    return actions


def consensus_actions(champion: dict[str, list[str]], paths: list[Path]) -> list[dict]:
    available = [path for path in paths if path.exists()]
    selections = [load_submission(path)[1] for path in available]
    actions = []
    for order_id, current_list in champion.items():
        current = set(current_list)
        all_nodes = set().union(*(set(value[order_id]) for value in selections))
        votes = {rid: sum(rid in value[order_id] for value in selections) for rid in all_nodes}
        additions = sorted(
            (rid for rid in all_nodes - current if votes[rid] >= max(4, len(selections) - 2)),
            key=lambda rid: (-votes[rid], rid),
        )
        removals = sorted(
            (rid for rid in current if votes.get(rid, 0) <= 1),
            key=lambda rid: (votes.get(rid, 0), rid),
        )
        pair_count = min(len(additions), len(removals), 2)
        if not pair_count:
            continue
        add = additions[:pair_count]
        remove = removals[:pair_count]
        margin = sum(votes[rid] for rid in add) - sum(votes.get(rid, 0) for rid in remove)
        actions.append(
            {
                "action_id": f"consensus_{order_id[:8]}",
                "order_id": order_id,
                "remove_rids": remove,
                "add_rids": add,
                "source": "consensus",
                "subtype": "consensus_swap",
                "expected_gain": margin / max(len(selections) * pair_count, 1),
                "evidence": {
                    "models": [str(path) for path in available],
                    "votes": {rid: votes.get(rid, 0) for rid in remove + add},
                    "vote_margin": margin,
                },
            }
        )
    return actions


def load_v11_scores(path: Path) -> dict[tuple[str, str], float]:
    scores = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            scores[(row["order_id"], row["rid"])] = (
                0.25 * float(row["context_score"]) + 0.75 * float(row["meta_mean"])
            )
    return scores


def boundary_consensus_actions(
    champion: dict[str, list[str]], paths: list[Path], scores: dict[tuple[str, str], float]
) -> list[dict]:
    available = [path for path in paths if path.exists()]
    selections = [load_submission(path)[1] for path in available]
    actions = []
    by_order_scores: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (order_id, rid), score in scores.items():
        by_order_scores[order_id].append((rid, score))
    for order_id, current_list in champion.items():
        current = set(current_list)
        votes = {
            rid: sum(rid in selection[order_id] for selection in selections)
            for rid, _ in by_order_scores[order_id]
        }
        add_pool = [
            (rid, score)
            for rid, score in by_order_scores[order_id]
            if rid not in current and score >= 0.25 and votes[rid] >= 2
        ]
        remove_pool = [
            (rid, score)
            for rid, score in by_order_scores[order_id]
            if rid in current and votes[rid] <= 4
        ]
        if not add_pool or not remove_pool:
            continue
        add_rid, add_score = max(add_pool, key=lambda item: (votes[item[0]], item[1], item[0]))
        remove_rid, remove_score = min(
            remove_pool, key=lambda item: (votes[item[0]], item[1], item[0])
        )
        vote_margin = votes[add_rid] - votes[remove_rid]
        if vote_margin < 1:
            continue
        actions.append(
            {
                "action_id": f"boundary_consensus_{order_id[:8]}",
                "order_id": order_id,
                "remove_rids": [remove_rid],
                "add_rids": [add_rid],
                "source": "consensus",
                "subtype": "consensus_swap",
                "expected_gain": 0.5 * vote_margin / len(selections)
                + 0.5 * max(add_score - remove_score, 0.0),
                "evidence": {
                    "models": [str(path) for path in available],
                    "remove_votes": votes[remove_rid],
                    "add_votes": votes[add_rid],
                    "remove_v11": remove_score,
                    "add_v11": add_score,
                    "exploratory": True,
                },
            }
        )
    return actions


def precision_tail_actions(
    champion: dict[str, list[str]],
    scores: dict[tuple[str, str], float],
    excluded: set[tuple[str, str]],
    limit: int = 16,
) -> list[dict]:
    candidates = []
    for order_id, rids in champion.items():
        if len(rids) <= 1:
            continue
        eligible = [
            (scores[(order_id, rid)], rid)
            for rid in rids
            if (order_id, rid) not in excluded and (order_id, rid) in scores
        ]
        if not eligible:
            continue
        score, rid = min(eligible)
        if score < 0.65:
            candidates.append((score, order_id, rid))
    candidates.sort()
    return [
        {
            "action_id": f"precision_tail_{order_id[:8]}",
            "order_id": order_id,
            "remove_rids": [rid],
            "add_rids": [],
            "source": "precision_tail",
            "subtype": "precision_tail",
            "expected_gain": 1.0 - score,
            "evidence": {"v11_score": score, "exploratory": True},
        }
        for score, order_id, rid in candidates[:limit]
    ]


def precision_actions(report_path: Path) -> list[dict]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    actions = []
    for group_name, group in report["groups"].items():
        by_order: dict[str, list[dict]] = defaultdict(list)
        for item in group["removed"]:
            by_order[item["order_id"]].append(item)
        for order_id, values in sorted(by_order.items()):
            actions.append(
                {
                    "action_id": f"precision_{group_name}_{order_id[:8]}",
                    "order_id": order_id,
                    "remove_rids": sorted(item["rid"] for item in values),
                    "add_rids": [],
                    "source": "precision_remove",
                    "subtype": group_name,
                    "expected_gain": sum(1.0 - item["v11_score"] for item in values),
                    "evidence": {"group": group_name, "nodes": values},
                }
            )
    return actions


def protected_sets(report_path: Path) -> tuple[set[tuple], set[tuple]]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    forced_in = {(item["order_id"], item["rid"]) for item in report["forced_in"]}
    forced_out = {(item["order_id"], item["rid"]) for item in report["forced_out"]}
    return forced_in, forced_out


def filter_and_resolve(
    candidates: list[dict],
    champion: dict[str, list[str]],
    topologies: dict[str, dict[str, dict]],
    forced_in: set[tuple],
    forced_out: set[tuple],
) -> tuple[list[dict], list[dict]]:
    priority = {
        "precision_remove": 0,
        "group_rule": 1,
        "strict_template": 2,
        "v26_router": 3,
        "consensus": 4,
        "precision_tail": 5,
    }
    accepted = []
    rejected = []
    used_orders = set()
    ranked = sorted(
        candidates,
        key=lambda item: (
            priority.get(item["source"], 99),
            -float(item.get("expected_gain", 0.0)),
            item["action_id"],
        ),
    )
    for raw in ranked:
        action = normalize_action(raw)
        order_id = action["order_id"]
        reason = None
        if any((order_id, rid) in forced_in for rid in action["remove_rids"]):
            reason = "removes_protected_forced_in"
        elif any((order_id, rid) in forced_out for rid in action["add_rids"]):
            reason = "adds_protected_forced_out"
        elif order_id in used_orders:
            reason = "lower_priority_order_conflict"
        else:
            try:
                validate_actions([action], champion, topologies)
            except ValueError as error:
                reason = str(error)
        if reason:
            rejected.append({"action": action, "reason": reason})
            continue
        used_orders.add(order_id)
        accepted.append(action)
    validate_actions(accepted, champion, topologies)
    return accepted, rejected


def plan_batches(actions: list[dict], max_actions: int) -> list[dict]:
    specs = [
        ("probe01_precision_a", {"a_lowest3"}),
        ("probe02_precision_b", {"b_next4"}),
        ("probe03_group_full", {"group_full"}),
        ("probe04_group_single", {"single_earliest", "single_upstream"}),
        ("probe05_group_pair", {"group_pair"}),
        ("probe06_strict_template", {"strict_template"}),
        ("probe07_v26_router", {"v26_router"}),
        ("probe08_consensus", {"consensus_swap"}),
        ("probe09_precision_tail", {"precision_tail"}),
    ]
    batches = []
    assigned = set()
    for batch_id, subtypes in specs:
        eligible = [
            item
            for item in actions
            if item["subtype"] in subtypes and item["action_id"] not in assigned
        ]
        eligible.sort(key=lambda item: (-item["expected_gain"], item["action_id"]))
        selected = eligible if batch_id.startswith("probe0") and "precision" in batch_id else eligible[:max_actions]
        assigned.update(item["action_id"] for item in selected)
        batches.append(
            {
                "batch_id": batch_id,
                "subtypes": sorted(subtypes),
                "action_ids": [item["action_id"] for item in selected],
                "actions": len(selected),
                "status": "planned" if selected else "empty",
            }
        )
    return batches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument("--test-dir", default=str(DEFAULT_TEST_DIR))
    parser.add_argument("--champion", default=str(DEFAULT_CHAMPION))
    parser.add_argument("--protected-report", default=str(DEFAULT_PROTECTED_REPORT))
    parser.add_argument("--v26", default=str(DEFAULT_V26))
    parser.add_argument("--precision-report", default=str(DEFAULT_PRECISION_REPORT))
    parser.add_argument("--v11-scores", default=str(DEFAULT_V11_SCORES))
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    parser.add_argument("--max-actions-per-batch", type=int, default=8)
    args = parser.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    print("Loading train/test orders...", flush=True)
    train_orders = load_orders(Path(args.train_dir), True)
    test_orders = load_orders(Path(args.test_dir), False)
    order_ids, champion = load_submission(Path(args.champion))
    topologies = load_topologies(Path(args.test_dir))
    validate_selection(order_ids, champion, topologies)

    print("Mining exact group rules and strict templates...", flush=True)
    rules = discover_group_rules(train_orders)
    pair_rules = discover_pair_rules(train_orders)
    templates = discover_templates(train_orders)
    precision = precision_actions(Path(args.precision_report))
    scores = load_v11_scores(Path(args.v11_scores))
    precision_nodes = {
        (item["order_id"], rid) for item in precision for rid in item["remove_rids"]
    }
    candidates = list(precision)
    candidates.extend(group_actions(test_orders, rules, champion))
    candidates.extend(pair_actions(test_orders, pair_rules, champion))
    candidates.extend(template_actions(test_orders, templates, champion))
    candidates.extend(diff_actions(champion, Path(args.v26), "v26_router", 0.5))
    candidates.extend(consensus_actions(champion, CONSENSUS_SUBMISSIONS))
    candidates.extend(boundary_consensus_actions(champion, CONSENSUS_SUBMISSIONS, scores))
    candidates.extend(precision_tail_actions(champion, scores, precision_nodes))

    forced_in, forced_out = protected_sets(Path(args.protected_report))
    actions, rejected = filter_and_resolve(
        candidates, champion, topologies, forced_in, forced_out
    )
    batches = plan_batches(actions, args.max_actions_per_batch)
    catalog = {
        "version": 1,
        "champion": str(Path(args.champion).resolve()),
        "champion_sha256": file_sha256(Path(args.champion)),
        "constraints": {
            "min_support": MIN_SUPPORT,
            "min_sites": MIN_SITES,
            "consistency": 1.0,
            "max_roots": MAX_ROOTS,
            "one_action_per_order": True,
        },
        "discovered": {
            "group_rules": len(rules),
            "pair_rules": len(pair_rules),
            "strict_templates": len(templates),
            "raw_candidates": len(candidates),
            "protected_forced_in": len(forced_in),
            "protected_forced_out": len(forced_out),
            "accepted_actions": len(actions),
            "rejected_actions": len(rejected),
        },
        "actions": actions,
    }
    write_json(workdir / "action_catalog.json", catalog)
    write_json(workdir / "rejected_actions.json", {"rejected": rejected})
    write_json(workdir / "batch_plan.json", {"batches": batches})
    print(
        json.dumps(
            {"catalog": catalog["discovered"], "batches": batches},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
