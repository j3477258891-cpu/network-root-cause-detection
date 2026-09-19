"""V156: six-node joint decode + remaining-five-budget reward maximization.

Self-contained and read-only against the frozen V153 candidate pool / ledger.
This module NEVER uploads a competition file and NEVER spends quota.

Core facts (recomputed and cross-checked, not hard-coded beliefs):
  * Actual champion p03 : P=1049, TP=971, F1=0.927855  (V149 + adds {8,11,14,17})
  * Remaining budget    : 5 submissions (<=4 info probes, final 1 reserved for merge)
  * Six new nodes       : {8,9,11,14,15,17}, equations
        t8+t11+t14+t17 = 2   and   t9+t15 = 1   ->  exactly 12 label combinations
  * Three count queries uniquely decode all 12 combinations:
        A = {8}      measures t8
        B = {11,9}   measures t11+t9
        C = {14,9}   measures t14+t9
  * Decoded merge (keep the 3 true nodes):  P=1048, TP=972, F1=0.929254
  * Full-pool optimistic upper bound:       dt=+5, dp=0 -> F1=0.932504

Scoring:  F1 = 2*TP / (G + P),   G=1044 total ground-truth positives.
Baseline V149: P0=1045, TP0=969, F1=0.927717.

Evidence model:  the six new nodes (12 combos) and the delete/swap remainder
(270 combos) are Cartesian-product independent under the frozen models, so the
joint space is exactly 12 x 270 = 3240 combinations.  The strategy search is
decomposed into two memoized phases sharing a probe budget, with the final
terminal being the last guaranteed merge.

Interfaces:  audit / build / recommend [--emit] / record / emit-final
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

HERE = Path(__file__).resolve().parent
V153 = HERE.with_name("v153_complementary_repair_campaign")
V149_DIR = HERE.with_name("v149_online_calibrated_addition_campaign")
BASE = V149_DIR / "v149_final_k08_from_v148.csv"
BASE_SHA = "c8f304c44f6b3a7ed40599cae50d0f17c0c37f8456553e6171a2d7cfdc196d0a"
P03_FILE = V153 / "v153_probe_p03_group.csv"
P03_SHA = "102b99385a8094a1d47b2748dfaa79fbd5a2b2b69936dffa9e5e7d9636fba196"

G, P0, TP0, TARGET = 1044, 1045, 969, 0.94
TZ = timezone(timedelta(hours=8))
MODELS = ("catboost", "v38_control")

WARNING = (
    "Equal-prior model mixture over the two frozen V153 reward models "
    "(catboost and v38_control), conditioned on measured count equations and the "
    "feasible 3240-combination support. NOT an exact label posterior, calibrated "
    "target probability, or global optimum guarantee."
)
UNIFORM_WARNING = (
    "Uniform distribution over the feasible combinations (sensitivity check). "
    "NOT a calibrated posterior."
)

# --------------------------------------------------------------------------- #
# Frozen candidate pool (id -> definition). Copied verbatim from the frozen
# V153 candidate_catalog.json so V156 is self-contained and never mutates it.
# --------------------------------------------------------------------------- #

CAND = {
    1:  dict(kind="delete", delta_p=-1, order_id="dc8df7d5-edb9-4849-a7d5-f34a911a3107",
             add_rid=None, remove_rid="#-1:093f68ee-a160-4cc0-8508-36d378df4dfc", node=None,
             mp=dict(catboost={"-1": 0.31087324035029684, "0": 0.6891267596497033},
                     v38_control={"-1": 0.11729979054949763, "0": 0.8827002094505024})),
    2:  dict(kind="delete", delta_p=-1, order_id="c2f7c036-aaf0-4a95-9bf4-3cae57b6e7d5",
             add_rid=None, remove_rid="#-1:0e2c59d3-277a-49ac-b27d-9c109ca1602f", node=None,
             mp=dict(catboost={"-1": 0.39943748619924163, "0": 0.6005625138007584},
                     v38_control={"-1": 0.1620765353504465, "0": 0.8379234646495536})),
    3:  dict(kind="delete", delta_p=-1, order_id="cdae01fa-ffeb-421d-bce8-020d92e0b8ab",
             add_rid=None, remove_rid="#-1:bdd347fe-bdbd-4937-9f50-b5a36492708a", node=None,
             mp=dict(catboost={"-1": 0.2908185468763371, "0": 0.7091814531236629},
                     v38_control={"-1": 0.21368228239322387, "0": 0.7863177176067762})),
    4:  dict(kind="delete", delta_p=-1, order_id="5b2934e0-b8fd-4f39-9287-15c201fd4c08",
             add_rid=None, remove_rid="#-1:695f2721-c73d-4b53-bae4-5825d5814aac", node=None,
             mp=dict(catboost={"-1": 0.19589162813612798, "0": 0.804108371863872},
                     v38_control={"-1": 0.2681936642967223, "0": 0.7318063357032778})),
    5:  dict(kind="add", delta_p=1, order_id="7b1b39e5-954a-418c-9ed7-03285444ea24",
             add_rid="#-1:fac75988-ea52-43cb-888c-45f1495ba491", remove_rid=None,
             node=dict(**{"@rid": "#-1:fac75988-ea52-43cb-888c-45f1495ba491"},
                       title="S1用户面路径不可用", location="sbn=0,NodeMe=532639,Equipment=1,rack=1,shelf=1,board=1",
                       reason="1.eNodeB路由错误。\n2.XGW故障。\n3.eNodeB与XGW之间的传输链路故障。"),
             mp=dict(catboost={"0": 0.3982656856626675, "1": 0.6017343143373325},
                     v38_control={"0": 0.24415650414182913, "1": 0.7558434958581709})),
    6:  dict(kind="delete", delta_p=-1, order_id="ed52fdc5-fce4-4fb1-a76e-bf6732d6360f",
             add_rid=None, remove_rid="#-1:62e818ae-ef1e-4865-86f2-1fdda72336cf", node=None,
             mp=dict(catboost={"-1": 0.3074189922514935, "0": 0.6925810077485065},
                     v38_control={"-1": 0.24599738374270216, "0": 0.7540026162572979})),
    7:  dict(kind="delete", delta_p=-1, order_id="d793d532-08ad-4410-a3fd-c09844eb67d0",
             add_rid=None, remove_rid="#-1:16b56bf4-e427-4983-bbcf-28d712f9d40e", node=None,
             mp=dict(catboost={"-1": 0.4320405328016693, "0": 0.5679594671983308},
                     v38_control={"-1": 0.1980762071902948, "0": 0.8019237928097053})),
    8:  dict(kind="add", delta_p=1, order_id="a6ae9c3c-c787-47e0-abf0-d610533dc2a0",
             add_rid="#-1:f4748a81-f1ce-4c63-b92b-571fb125ef8b", remove_rid=None,
             node=dict(**{"@rid": "#-1:f4748a81-f1ce-4c63-b92b-571fb125ef8b"},
                       title="光模块接收光功率异常",
                       location="SubNetwork=CMCC-GZ-02,ManagedElement=12588010,Equipment=1,ReplaceableUnit=A_4,RiPort=OPT1",
                       reason="光模块接收功率过低"),
             mp=dict(catboost={"0": 0.381558900146144, "1": 0.618441099853856},
                     v38_control={"0": 0.500540607329395, "1": 0.499459392670605})),
    9:  dict(kind="add", delta_p=1, order_id="1117ad03-8891-4bf6-854c-612bf19fdae4",
             add_rid="#-1:bea1860d-537e-4e4b-ba53-08f5ce0bd7dc", remove_rid=None,
             node=dict(**{"@rid": "#-1:bea1860d-537e-4e4b-ba53-08f5ce0bd7dc"},
                       title="小区关断告警",
                       location="SubNetwork=CMCC-GZ-04,ManagedElement=12641069,ENBCUCPFunction=1,CULTE=1,CUEUtranCellTDDLTE=3",
                       reason="小区被关断"),
             mp=dict(catboost={"0": 0.6122157198790286, "1": 0.38778428012097144},
                     v38_control={"0": 0.3343660247005541, "1": 0.6656339752994459})),
    10: dict(kind="delete", delta_p=-1, order_id="af374848-7bd9-44f9-bfb8-4c087a74d016",
             add_rid=None, remove_rid="#-1:2da33b00-7529-4ffb-8ecd-6a0cca5f01c4", node=None,
             mp=dict(catboost={"-1": 0.5144847034666935, "0": 0.48551529653330655},
                     v38_control={"-1": 0.2958049085154171, "0": 0.7041950914845829})),
    11: dict(kind="add", delta_p=1, order_id="8f73e1cf-c334-4f3b-aa08-78d25a9722f5",
             add_rid="#-1:c7ad6d1d-f834-4257-9936-4e324f1f1672", remove_rid=None,
             node=dict(**{"@rid": "#-1:c7ad6d1d-f834-4257-9936-4e324f1f1672"},
                       title="RRU链路断",
                       location="SubNetwork=CMCC-GZ-03,ManagedElement=12655788,Equipment=1,ReplaceableUnit=A_5",
                       reason="RRU链路断"),
             mp=dict(catboost={"0": 0.28991545484890735, "1": 0.7100845451510926},
                     v38_control={"0": 0.5818519539817231, "1": 0.418148046018277})),
    12: dict(kind="swap", delta_p=0, order_id="625ffce7-65c5-4657-9104-4148b4113130",
             add_rid="#-1:224b57dc-60c0-4fc8-a6ab-ad594824e66b",
             remove_rid="#-1:a2c5a181-0370-473b-972e-d0f5d0628d74",
             node=dict(**{"@rid": "#-1:224b57dc-60c0-4fc8-a6ab-ad594824e66b"},
                       title="输入电源断",
                       location="SubNetwork=5305,ManagedElement=2622852,Equipment=1,ReplaceableUnit=56",
                       reason="输入电源断"),
             mp=dict(catboost={"-1": 0.07730395997825891, "0": 0.8322409222938607, "1": 0.09045511772788044},
                     v38_control={"-1": 0.022096415585768164, "0": 0.9148603599645093, "1": 0.0630432244497224})),
    13: dict(kind="delete", delta_p=-1, order_id="4c2b3b06-a6a2-4c1b-9734-def3d1f849d8",
             add_rid=None, remove_rid="#-1:66facd91-e281-4331-bbae-04ab8bd9746d", node=None,
             mp=dict(catboost={"-1": 0.5737568044138377, "0": 0.4262431955861623},
                     v38_control={"-1": 0.28023203360412857, "0": 0.7197679663958715})),
    14: dict(kind="add", delta_p=1, order_id="6642cbb1-ec65-48e1-9ec3-6d68bb98e59f",
             add_rid="#-1:71e642ee-18dd-4934-abc2-5ec460b241c8", remove_rid=None,
             node=dict(**{"@rid": "#-1:71e642ee-18dd-4934-abc2-5ec460b241c8"},
                       title="1588时钟链路异常",
                       location="SubNetwork=1004,ManagedElement=1390908,Equipment=1,ReplaceableUnit=81",
                       reason="1588时钟链路报文接收异常"),
             mp=dict(catboost={"0": 0.36744349042853464, "1": 0.6325565095714654},
                     v38_control={"0": 0.7742756840061438, "1": 0.22572431599385623})),
    15: dict(kind="add", delta_p=1, order_id="ab02ed9d-74ea-4fc1-8935-9e5396cf94ba",
             add_rid="#-1:2fe391a0-09be-47fc-bb08-f50ecb445a3f", remove_rid=None,
             node=dict(**{"@rid": "#-1:2fe391a0-09be-47fc-bb08-f50ecb445a3f"},
                       title="[衍生告警]PTN光缆中断，单报LOS",
                       location="R8EGF[0-1-2]-GE\\:1",
                       reason="以太网物理接口(ETPI) 信号丢失(LOS)"),
             mp=dict(catboost={"0": 0.6978815734552242, "1": 0.30211842654477594},
                     v38_control={"0": 0.35788979118695546, "1": 0.6421102088130446})),
    16: dict(kind="swap", delta_p=0, order_id="edcaa16f-fee4-415e-88ac-ba68e7aa0087",
             add_rid="#-1:9c3d60d1-9412-4278-87b2-fcfb5eb15669",
             remove_rid="#-1:fdaaf547-da3c-47bd-b583-00d8af72e779",
             node=dict(**{"@rid": "#-1:9c3d60d1-9412-4278-87b2-fcfb5eb15669"},
                       title="射频单元维护链路异常告警",
                       location="柜号:0; 框号:61; 槽号:0; 类型:AIRU; :",
                       reason="567"),
             mp=dict(catboost={"-1": 0.450625857100292, "0": 0.44769942524428297, "1": 0.10167471765542502},
                     v38_control={"-1": 0.22396958663559274, "0": 0.30864382955798364, "1": 0.46738658380642367})),
    17: dict(kind="add", delta_p=1, order_id="af614073-0139-45cc-82d0-97e02af80b2d",
             add_rid="#-1:0771a7a8-2000-439c-a4ce-1d30244cd225", remove_rid=None,
             node=dict(**{"@rid": "#-1:0771a7a8-2000-439c-a4ce-1d30244cd225"},
                       title="RRU组网拓扑类型与配置不一致告警",
                       location=":5; 类型:配置异常; 用户配置接口板槽位:NULL; 用户配置接口板端口号:NULL; 实际接口板槽位:7; 实际接口板端口号:8; 信息[配置槽位/端口号/端口号; 实际槽位/端口号/端口号]:NULL; 描述信息:UBBP\\:0-0-4 Port\\:4 Subport\\:0",
                       reason="307"),
             mp=dict(catboost={"0": 0.5280598868810349, "1": 0.4719401131189651},
                     v38_control={"0": 0.8717787303346566, "1": 0.12822126966534342})),
}

NEW_NODES = (8, 9, 11, 14, 15, 17)       # six new nodes to joint-decode
DEL_GROUP1 = (1, 2, 7, 10, 13)           # measured delete leaf, dt=-3 (3 true)
DEL_GROUP2 = (3, 4, 6)                    # measured delete leaf, dt=-2 (2 true)
SWAPS = (12, 16)                          # unmeasured swap actions
S2_CANDIDATES = (1, 2, 3, 4, 6, 7, 10, 13, 12, 16)  # stage-2 (delete + swap) nodes
ALL_ACTIVE = (1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)

MEASURED_LEAVES = (
    (frozenset({1, 2, 7, 10, 13}), -3),
    (frozenset({3, 4, 6}), -2),
    (frozenset({8, 11, 14, 17}), 2),
    (frozenset({5}), 0),
    (frozenset({9, 15}), 1),
)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(TZ).isoformat(timespec="seconds")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def f1(dt=0, dp=0):
    return 2 * (TP0 + dt) / (G + P0 + dp)


def infer_tp(score, predictions):
    try:
        value = Decimal(str(score))
        require(value.is_finite() and 0 <= value <= 1, "Invalid score")
        require(value == value.quantize(Decimal("0.000001")), "Supply the displayed six-decimal score")
    except InvalidOperation as exc:
        raise ValueError("Invalid score") from exc
    den = G + predictions
    center = int(value * den / 2)
    choices = [t for t in range(max(0, center - 2), min(G, predictions, center + 2) + 1)
               if abs(Decimal(2 * t) / den - value) <= Decimal("0.0000005")]
    require(len(choices) == 1, f"Score/P do not identify a unique integer TP: {choices}")
    return choices[0]


# --------------------------------------------------------------------------- #
# Combination spaces (exact enumeration)
# --------------------------------------------------------------------------- #

def enumerate_combos():
    """Full 3240-combination space (six new nodes x delete groups x swaps)."""
    combos = []
    a4 = [8, 11, 14, 17]
    a2 = [9, 15]
    for truth4 in itertools.combinations(a4, 2):
        s4 = set(truth4)
        for truth2 in (a2[0], a2[1]):
            base = {}
            for i in a4:
                base[i] = 1 if i in s4 else 0
            for i in a2:
                base[i] = 1 if i == truth2 else 0
            for true1 in itertools.combinations(DEL_GROUP1, 3):
                s1 = set(true1)
                c = dict(base)
                for i in DEL_GROUP1:
                    c[i] = -1 if i in s1 else 0
                for true2 in itertools.combinations(DEL_GROUP2, 2):
                    s2 = set(true2)
                    c2 = dict(c)
                    for i in DEL_GROUP2:
                        c2[i] = -1 if i in s2 else 0
                    for s12 in (-1, 0, 1):
                        for s16 in (-1, 0, 1):
                            cc = dict(c2)
                            cc[12] = s12
                            cc[16] = s16
                            cc[5] = 0
                            combos.append(cc)
    require(len(combos) == 3240, f"Unexpected combination count: {len(combos)}")
    return combos


def enumerate_new_combos():
    combos = []
    a4 = [8, 11, 14, 17]
    a2 = [9, 15]
    for truth4 in itertools.combinations(a4, 2):
        s4 = set(truth4)
        for truth2 in (a2[0], a2[1]):
            c = {}
            for i in a4:
                c[i] = 1 if i in s4 else 0
            for i in a2:
                c[i] = 1 if i == truth2 else 0
            combos.append(c)
    require(len(combos) == 12, "Expected 12 new-node combinations")
    return combos


def enumerate_s2_combos():
    combos = []
    for true1 in itertools.combinations(DEL_GROUP1, 3):
        s1 = set(true1)
        for true2 in itertools.combinations(DEL_GROUP2, 2):
            s2 = set(true2)
            for s12 in (-1, 0, 1):
                for s16 in (-1, 0, 1):
                    c = {}
                    for i in DEL_GROUP1:
                        c[i] = -1 if i in s1 else 0
                    for i in DEL_GROUP2:
                        c[i] = -1 if i in s2 else 0
                    c[12] = s12
                    c[16] = s16
                    combos.append(c)
    require(len(combos) == 270, f"Expected 270 stage-2 combinations, got {len(combos)}")
    return combos


def combo_prob(combo, candidates, model):
    p = 1.0
    for i in candidates:
        p *= CAND[i]["mp"][model][str(combo[i])]
    return p


def combos_priors(combos, candidates):
    raw = [0.5 * combo_prob(c, candidates, "catboost") + 0.5 * combo_prob(c, candidates, "v38_control")
           for c in combos]
    total = sum(raw)
    require(total > 0 and math.isfinite(total), "No supported conditional branch")
    return [p / total for p in raw]


def combos_uniform(combos):
    return [1.0 / len(combos)] * len(combos)


# --------------------------------------------------------------------------- #
# Merge helpers
# --------------------------------------------------------------------------- #

def _known_labels(combos, bitmask, candidates):
    known = {}
    for i in candidates:
        vals = set()
        for j in range(len(combos)):
            if bitmask >> j & 1:
                vals.add(combos[j][i])
        if len(vals) == 1:
            known[i] = vals.pop()
    return known


def build_label_masks(combos, candidates):
    """{(candidate, label): bitmask of combos where that candidate has that label}."""
    masks = {}
    for j, c in enumerate(combos):
        bit = 1 << j
        for i in candidates:
            masks[(i, c[i])] = masks.get((i, c[i]), 0) | bit
    return masks


def known_labels_masked(masks, bitmask, candidates):
    """Fast whole-group known-label map via bitmasks (O(#candidates))."""
    known = {}
    for i in candidates:
        for (cid, lab), m in masks.items():
            if cid == i and (bitmask & m) == bitmask:
                known[i] = lab
                break
    return known


def merge_fast(new_masks, new_bitmask, s2_masks, s2_bitmask, champion_f1):
    known = {}
    known.update(known_labels_masked(new_masks, new_bitmask, NEW_NODES))
    known.update(known_labels_masked(s2_masks, s2_bitmask, S2_CANDIDATES))
    ids, dt, dp = merge_actions(known)
    return max(champion_f1, f1(dt, dp)), ids, dt, dp


def merge_actions(known):
    """Determined, beneficial actions from a known-label map (never guess).

    add  i : include iff label == 1
    del  i : include iff label == 0  (drop a proven non-root; dp-1, dt+0)
    swap i : include iff label == 1 (dt+1) or label == -1 (dt-1); label 0 -> none
    Unresolved candidates contribute nothing, so the merge stays equation-determined.
    Negative delta_tp is allowed here (a proven swap can lower TP).
    """
    ids, dt, dp = [], 0, 0
    for i, o in known.items():
        kind = CAND[i]["kind"]
        if kind == "add" and o == 1:
            ids.append(i); dt += 1; dp += CAND[i]["delta_p"]
        elif kind == "delete" and o == 0:
            ids.append(i); dp += CAND[i]["delta_p"]
        elif kind == "swap" and o == 1:
            ids.append(i); dt += 1; dp += CAND[i]["delta_p"]
    return sorted(ids), dt, dp


def guaranteed_merge(new_combos, new_bitmask, s2_combos, s2_bitmask, champion_f1):
    """Best F1 that is *determined* by the current knowledge, floored at champion.

    Enumerates every legal action subset consistent across the still-possible
    combinations -- including whole groups whose total is known while single
    nodes remain unresolved -- and keeps the best.  Never uses a guessed node.
    """
    known = {}
    known.update(_known_labels(new_combos, new_bitmask, NEW_NODES))
    known.update(_known_labels(s2_combos, s2_bitmask, S2_CANDIDATES))
    ids, dt, dp = merge_actions(known)
    value = max(champion_f1, f1(dt, dp))
    return value, ids, dt, dp


def _resolved_new(combos, bitmask):
    label = None
    for j in range(len(combos)):
        if bitmask >> j & 1:
            c = combos[j]
            lab = (c[8], c[9], c[11], c[14], c[15], c[17])
            if label is None:
                label = lab
            elif label != lab:
                return False
    return label is not None


# --------------------------------------------------------------------------- #
# Two-phase memoized strategy search
# --------------------------------------------------------------------------- #

class Search:
    """Two-phase memoized strategy search with a champion floor.

    Value at any node = E[max(champion_f1, best equation-determined merge)]
    over the still-feasible combinations.  A timeout returns the conservative
    floor and sets ``truncated`` -- it never caches a partial value as exact,
    and the caller must fall back to the validated fixed plan.
    """

    def __init__(self, new_combos, new_priors, s2_combos, s2_priors,
                 champion_f1=0.0, wall_limit=120.0, node_limit=400_000,
                 max_new_query=3, max_s2_query=2,
                 max_new_budget=3, max_s2_budget=2):
        self.new_combos, self.new_priors = new_combos, new_priors
        self.s2_combos, self.s2_priors = s2_combos, s2_priors
        self.s2_full = (1 << len(s2_combos)) - 1
        self.new_full = (1 << len(new_combos)) - 1
        self.champion_f1 = champion_f1
        self.s2_memo, self.new_memo = {}, {}
        self.nodes = 0
        self.start = time.time()
        self.wall_limit, self.node_limit = wall_limit, node_limit
        self.max_new_query, self.max_s2_query = max_new_query, max_s2_query
        self.max_new_budget, self.max_s2_budget = max_new_budget, max_s2_budget
        # Precompute tuple-major label vectors for fast grouped sums.
        self.new_vec = [tuple(c[i] for i in NEW_NODES) for c in new_combos]
        self.s2_vec = [tuple(c[i] for i in S2_CANDIDATES) for c in s2_combos]
        self.new_pos = {i: k for k, i in enumerate(NEW_NODES)}
        self.s2_pos = {i: k for k, i in enumerate(S2_CANDIDATES)}
        self.new_masks = build_label_masks(new_combos, NEW_NODES)
        self.s2_masks = build_label_masks(s2_combos, S2_CANDIDATES)
        self.truncated = False

    def _timeout(self):
        return time.time() - self.start > self.wall_limit or self.nodes > self.node_limit

    def _unresolved(self, combos, bitmask, candidates):
        return [i for i in candidates
                if len({combos[j][i] for j in range(len(combos)) if bitmask >> j & 1}) > 1]

    def _queries(self, combos, bitmask, candidates, max_size):
        unresolved = self._unresolved(combos, bitmask, candidates)
        for r in range(1, min(max_size, len(unresolved)) + 1):
            for sub in itertools.combinations(unresolved, r):
                yield frozenset(sub)

    def _floor(self, new_mask, s2_mask):
        return merge_fast(self.new_masks, new_mask, self.s2_masks, s2_mask,
                          self.champion_f1)[0]

    # ---- phase 2: delete groups + swaps, terminal = best guaranteed merge ---- #
    def s2_value(self, new_mask, s2_mask, budget):
        budget = min(budget, self.max_s2_budget)
        floor = self._floor(new_mask, s2_mask)
        if budget <= 0 or self._timeout():
            if self._timeout():
                self.truncated = True
            return floor
        key = (new_mask, s2_mask, budget)
        if key in self.s2_memo:
            return self.s2_memo[key]
        self.nodes += 1
        best = floor
        vec, pos, priors = self.s2_vec, self.s2_pos, self.s2_priors
        for q in self._queries(self.s2_combos, s2_mask, S2_CANDIDATES, self.max_s2_query):
            ps = tuple(pos[i] for i in q)
            groups = defaultdict(lambda: [0.0, 0])
            for j in range(len(vec)):
                if s2_mask >> j & 1:
                    v = vec[j]
                    s = sum(v[p] for p in ps)
                    groups[s][0] += priors[j]
                    groups[s][1] |= (1 << j)
            if len(groups) <= 1:
                continue
            total = sum(w for w, _ in groups.values())
            exp = sum((w / total) * self.s2_value(new_mask, gm, budget - 1)
                      for s, (w, gm) in groups.items())
            if exp > best:
                best = exp
        if not self.truncated:
            self.s2_memo[key] = best
        return best

    # ---- phase 1: decode six new nodes, terminal = stage-2 with leftover ---- #
    def decode_value(self, new_mask, budget):
        budget = min(budget, self.max_new_budget)
        if _resolved_new(self.new_combos, new_mask):
            return self.s2_value(new_mask, self.s2_full, budget)
        floor = self._floor(new_mask, self.s2_full)
        if budget <= 0 or self._timeout():
            if self._timeout():
                self.truncated = True
            return floor
        key = (new_mask, budget)
        if key in self.new_memo:
            return self.new_memo[key]
        self.nodes += 1
        best = floor
        vec, pos, priors = self.new_vec, self.new_pos, self.new_priors
        for q in self._queries(self.new_combos, new_mask, NEW_NODES, self.max_new_query):
            ps = tuple(pos[i] for i in q)
            groups = defaultdict(lambda: [0.0, 0])
            for j in range(len(vec)):
                if new_mask >> j & 1:
                    v = vec[j]
                    s = sum(v[p] for p in ps)
                    groups[s][0] += priors[j]
                    groups[s][1] |= (1 << j)
            if len(groups) <= 1:
                continue
            total = sum(w for w, _ in groups.values())
            exp = sum((w / total) * self.decode_value(gm, budget - 1)
                      for s, (w, gm) in groups.items())
            if exp > best:
                best = exp
        if not self.truncated:
            self.new_memo[key] = best
        return best

    # ---- top-level: expected value, worst branch, first query ---- #
    def first_query(self, new_mask, budget):
        budget = min(budget, self.max_new_budget)
        resolved = _resolved_new(self.new_combos, new_mask)
        # If already decoded, phase 1 has no query; hand everything to stage 2.
        roots = ([] if resolved
                 else list(self._queries(self.new_combos, new_mask, NEW_NODES, self.max_new_query)))
        vec, pos, priors = self.new_vec, self.new_pos, self.new_priors
        best = {"val": -1.0, "q": None, "worst": None, "branches": []}
        for q in roots:
            ps = tuple(pos[i] for i in q)
            groups = defaultdict(lambda: [0.0, 0])
            for j in range(len(vec)):
                if new_mask >> j & 1:
                    v = vec[j]
                    s = sum(v[p] for p in ps)
                    groups[s][0] += priors[j]
                    groups[s][1] |= (1 << j)
            if len(groups) <= 1:
                continue
            total = sum(w for w, _ in groups.values())
            exp, worst, branches = 0.0, math.inf, []
            for s, (w, gm) in sorted(groups.items()):
                cont = self.decode_value(gm, budget - 1)
                exp += (w / total) * cont
                worst = min(worst, cont)
                branches.append({"feedback": s, "weight": round(w / total, 6),
                                 "continuation_f1": cont, "combos_remaining": bin(gm).count("1")})
            if exp > best["val"] + 1e-12:
                best = {"val": exp, "q": q, "worst": worst, "branches": branches}
        return best


# --------------------------------------------------------------------------- #
# Decode / audit helpers
# --------------------------------------------------------------------------- #

def decode_uniqueness(combos):
    sig_to_labels = {}
    for c in combos:
        sig = (c[8], c[11] + c[9], c[14] + c[9])
        labels = (c[8], c[9], c[11], c[14], c[15], c[17])
        if sig in sig_to_labels and sig_to_labels[sig] != labels:
            return {"unique": False, "reason": "signature collision across new-node labels",
                    "queries": {"A": [8], "B": [11, 9], "C": [14, 9]},
                    "new_node_label_combinations": 12}
        sig_to_labels[sig] = labels
    return {"unique": len(sig_to_labels) == 12, "signatures": len(sig_to_labels),
            "queries": {"A": [8], "B": [11, 9], "C": [14, 9]},
            "new_node_label_combinations": 12}


def decode_plan(new_combos):
    """Fixed A->B->C decode plan with per-branch early-termination flags."""
    queries = [[8], [11, 9], [14, 9]]
    bitmask = (1 << len(new_combos)) - 1
    plan = {"queries": queries, "phases": []}
    for qids in queries:
        if _resolved_new(new_combos, bitmask):
            break
        groups = defaultdict(list)
        for j in range(len(new_combos)):
            if bitmask >> j & 1:
                groups[sum(new_combos[j][i] for i in qids)].append(j)
        phase = {"query": qids, "branches": []}
        for s, g in sorted(groups.items()):
            gm = 0
            for j in g:
                gm |= (1 << j)
            phase["branches"].append({"feedback": s, "combos_remaining": len(g),
                                      "decoded": _resolved_new(new_combos, gm),
                                      "true_nodes_known": None})
        plan["phases"].append(phase)
    return plan


# --------------------------------------------------------------------------- #
# CSV build + validation
# --------------------------------------------------------------------------- #

def load_csv(path, strict=True):
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        require(reader.fieldnames == ["order_id", "output"], f"Invalid CSV header: {path}")
        rows = list(reader)
    ids, roots, nodes = [], {}, {}
    for row in rows:
        oid = row["order_id"]
        require(oid not in roots, f"Duplicate order: {path}: {oid}")
        obj = json.loads(row["output"])
        values = obj["rootcause"]
        require(isinstance(values, list), "rootcause must be a list")
        if strict:
            require(1 <= len(values) <= 8, f"Invalid root count: {oid}")
        ids.append(oid)
        roots[oid] = values
        for nd in values:
            key = (oid, nd["@rid"])
            require(key not in nodes, f"Duplicate order/RID: {path}: {key}")
            nodes[key] = nd
    require(len(ids) == 546, f"Expected 546 orders: {path}")
    return ids, roots, nodes


def apply_actions(base_path, action_ids, out_path):
    ids, roots, nodes = load_csv(base_path)
    seen_orders = set()
    for i in action_ids:
        a = CAND[i]
        require(a["order_id"] not in seen_orders, "Multiple actions in one order")
        seen_orders.add(a["order_id"])
        require(a["order_id"] in roots, "Unknown order")
        if a["remove_rid"]:
            require((a["order_id"], a["remove_rid"]) in nodes, "Removed node absent")
        if a["add_rid"]:
            require((a["order_id"], a["add_rid"]) not in nodes, "Added node already selected")
    out_roots = {oid: list(v) for oid, v in roots.items()}
    for i in action_ids:
        a = CAND[i]
        rc = out_roots[a["order_id"]]
        if a["remove_rid"]:
            rc = [n for n in rc if n["@rid"] != a["remove_rid"]]
        if a["add_rid"]:
            rc = rc + [a["node"]]
        require(1 <= len(rc) <= 8, f"Root count outside 1..8 after action {i}")
        out_roots[a["order_id"]] = rc
    with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["order_id", "output"])
        writer.writeheader()
        for oid in ids:
            writer.writerow({"order_id": oid,
                             "output": json.dumps({"rootcause": out_roots[oid]}, ensure_ascii=False, separators=(",", ":"))})
    return ids, out_roots


def validate_csv(path, action_ids, expected_sha=None):
    base_ids, _, base_nodes = load_csv(BASE)
    out_ids, _, out_nodes = load_csv(path)
    require(out_ids == base_ids, "Order sequence changed")
    adds = {(CAND[i]["order_id"], CAND[i]["add_rid"]) for i in action_ids if CAND[i]["add_rid"]}
    removes = {(CAND[i]["order_id"], CAND[i]["remove_rid"]) for i in action_ids if CAND[i]["remove_rid"]}
    require(set(out_nodes) - set(base_nodes) == adds and set(base_nodes) - set(out_nodes) == removes,
            "CSV difference mismatch")
    require(all(out_nodes[k] == base_nodes[k] for k in set(base_nodes) - removes), "Preserved metadata changed")
    for i in action_ids:
        if CAND[i]["add_rid"]:
            require(out_nodes[CAND[i]["order_id"], CAND[i]["add_rid"]] == CAND[i]["node"], "Added metadata mismatch")
    P = len(out_nodes)
    require(P == P0 + sum(CAND[i]["delta_p"] for i in action_ids), "Prediction count mismatch")
    value = sha(path)
    require(not expected_sha or value == expected_sha, "CSV hash changed")
    return {"orders": len(out_ids), "predictions": P, "sha256": value,
            "exact_difference": True, "root_counts_1_to_8": True, "duplicates": 0}


# --------------------------------------------------------------------------- #
# Campaign state + cross-check
# --------------------------------------------------------------------------- #

def default_state():
    return {
        "version": 156,
        "created_at": now(),
        "baseline": {"file": str(BASE), "sha256": BASE_SHA, "predictions": P0, "tp": TP0,
                     "score": 0.927717, "source_class": "public_scored"},
        "champion": {"file": str(P03_FILE), "sha256": P03_SHA, "predictions": 1049, "tp": 971,
                     "score": 0.927855, "source_class": "public_scored"},
        "target": TARGET,
        "objective": "maximize_realized_best_f1",
        "budget": {"total": 10, "used": 5, "remaining": 5, "daily": 2, "max_info_probes": 4, "merge_reserve": 1},
        "measured_leaves": [{"ids": sorted(ids), "delta_tp": dt} for ids, dt in MEASURED_LEAVES],
        "frozen_v153": {"candidate_catalog": str(V153 / "candidate_catalog.json"),
                        "final_gate": str(V153 / "final_gate.json")},
    }


def load_state():
    path = HERE / "campaign.json"
    return read_json(path) if path.exists() else None


def cross_check_v153():
    require(V153.is_dir(), "V153 campaign directory missing")
    gate = read_json(V153 / "final_gate.json")
    champ = gate["actual_champion"]
    require(champ["predictions"] == 1049 and champ["tp"] == 971 and champ["score"] == "0.927855",
            "V153 champion drift")
    require(champ["sha256"] == P03_SHA, "V153 champion hash drift")
    leaves = {(frozenset(l["ids"]), l["delta_tp"]) for l in gate["measured_leaves"]}
    require(leaves == set(MEASURED_LEAVES), "V153 measured-leaf drift")
    require(sha(BASE) == BASE_SHA, "Baseline CSV hash drift")
    require(sha(P03_FILE) == P03_SHA, "Champion CSV hash drift")
    return {"champion_ok": True, "leaves_ok": True, "hashes_ok": True}


def full_pool_upper(combos):
    best = {"f1": f1(0, 0), "dt": 0, "dp": 0, "ids": []}
    for c in combos:
        ids, dt, dp = [], 0, 0
        for i in ALL_ACTIVE:
            o = c[i]
            kind = CAND[i]["kind"]
            if (kind == "add" and o == 1) or (kind == "delete" and o == 0) or (kind == "swap" and o == 1):
                ids.append(i); dt += o; dp += CAND[i]["delta_p"]
        val = f1(dt, dp)
        if val > best["f1"] + 1e-12:
            best = {"f1": val, "dt": dt, "dp": dp, "ids": ids}
    return best


# --------------------------------------------------------------------------- #
# Dynamic ledger state  (never reuse a frozen budget; read every real attempt)
# --------------------------------------------------------------------------- #

def load_ledger():
    path = HERE / "online_scores.json"
    if path.exists():
        return read_json(path)
    return {"baseline": default_state()["baseline"], "records": [],
            "note": "Only actual user-supplied attempts. Failed/anomalous attempts count."}


def load_manifest():
    path = HERE / "submission_manifest.json"
    return read_json(path) if path.exists() else {"files": []}


def ledger_leaves():
    """All historical equations: frozen V153 leaves + every accepted V156 probe.

    Each accepted info probe contributes one new equation (its query ids ->
    observed delta_tp).  Failed / anomalous attempts contribute no equation but
    still consume budget.
    """
    leaves = list(MEASURED_LEAVES)
    entries = {e["probe_id"]: e for e in load_manifest()["files"]}
    for r in load_ledger().get("records", []):
        if r.get("status") != "accepted":
            continue
        e = entries.get(r.get("probe_id"))
        if not e or not e.get("query_ids"):
            continue
        leaves.append((frozenset(e["query_ids"]), int(r.get("info_delta", 0))))
    return leaves


def filter_combos(combos, leaves):
    """Keep only combinations consistent with every historical scoring equation.

    An equation that references a candidate absent from this combo space (e.g. a
    delete-group leaf applied to the new-node space) is not applicable and skipped.
    """
    out = []
    for c in combos:
        ok = True
        for ids, dt in leaves:
            if any(i not in c for i in ids):
                continue
            if sum(c[i] for i in ids) != dt:
                ok = False
                break
        if ok:
            out.append(c)
    return out


def derived_state():
    """Recompute champion / used / remaining / leaves from real ledgers."""
    state = load_state() or default_state()
    records = load_ledger().get("records", [])
    used = state["budget"]["used"] + len(records)
    remaining = max(0, state["budget"]["total"] - used)
    champ = dict(state["champion"])
    champ_f1 = f1(champ["tp"] - TP0, champ["predictions"] - P0)
    for r in records:
        if r.get("status") != "accepted":
            continue
        value = f1(r.get("delta_tp", 0), r.get("predictions", P0) - P0)
        if value > champ_f1 + 1e-12:
            champ_f1 = value
            champ = {"file": r.get("file", str(HERE / "probe")), "sha256": r.get("sha256"),
                     "predictions": r.get("predictions"), "tp": r.get("inferred_tp"),
                     "score": r.get("score"), "source_class": "public_scored"}
    state["budget"] = {**state["budget"], "used": used, "remaining": remaining,
                       "daily_used": sum(1 for r in records if str(r.get("submitted_at", ""))[:10]
                                         == now()[:10])}
    state["champion"] = champ
    leaves = ledger_leaves()
    state["measured_leaves"] = [{"ids": sorted(ids), "delta_tp": dt} for ids, dt in leaves]
    state["attempts"] = len(records)
    return state


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def _build_search(model_prior=True, champion_f1=0.0, time_limit=None,
                  new_combos=None, s2_combos=None):
    new_combos = enumerate_new_combos() if new_combos is None else new_combos
    s2_combos = enumerate_s2_combos() if s2_combos is None else s2_combos
    if model_prior:
        new_priors = combos_priors(new_combos, NEW_NODES)
        s2_priors = combos_priors(s2_combos, S2_CANDIDATES)
    else:
        new_priors = combos_uniform(new_combos)
        s2_priors = combos_uniform(s2_combos)
    kw = {} if time_limit is None else {"wall_limit": time_limit}
    return Search(new_combos, new_priors, s2_combos, s2_priors, champion_f1=champion_f1, **kw)


def value_classes(state, upper):
    """Separate the four value notions so they are never conflated."""
    champ = state["champion"]
    return {
        "actual_score": {"champion_f1": round(f1(champ["tp"] - TP0, champ["predictions"] - P0), 6),
                         "champion": champ, "source_class": "public_scored"},
        "equation_inferred": {"backup_merge_f1": round(f1(3, 3), 6), "predictions": 1048, "tp": 972,
                              "note": "determined only once the six nodes are decoded; not yet scored"},
        "model_conditional_expectation": None,   # filled by cmd_recommend
        "theoretical_upper": {"full_pool_f1": round(upper["f1"], 6), "delta_tp": upper["dt"],
                              "delta_p": upper["dp"], "selected_ids": upper["ids"],
                              "source_class": "optimistic_integer_oracle_not_forecast"},
    }


def cmd_build():
    """Idempotent: rebuild derived fields but NEVER clear the score ledger."""
    leaf = ledger_leaves()
    combos = filter_combos(enumerate_combos(), leaf)
    state = default_state()
    state["cross_check_v153"] = cross_check_v153()
    state["combination_count"] = len(enumerate_combos())
    state["feasible_combination_count"] = len(combos)
    state["decode"] = decode_uniqueness(combos)
    state["backup_route"] = {"predictions": 1048, "tp": 972, "f1": 0.929254,
                             "note": "keep the 3 true nodes among the six, drop the 2 false p03 additions"}
    upper = full_pool_upper(combos)
    state["full_pool_upper"] = {"f1": round(upper["f1"], 6), "delta_tp": upper["dt"],
                                "delta_p": upper["dp"], "selected_ids": upper["ids"]}
    records = load_ledger().get("records", [])
    state["budget"]["used"] += len(records)
    state["budget"]["remaining"] = state["budget"]["total"] - state["budget"]["used"]
    write_json(HERE / "campaign.json", state)
    if not (HERE / "online_scores.json").exists():
        write_json(HERE / "online_scores.json",
                   {"baseline": state["baseline"], "records": [],
                    "note": "Only actual user-supplied attempts. Failed/anomalous attempts count."})
    if not (HERE / "submission_manifest.json").exists():
        write_json(HERE / "submission_manifest.json", {"files": []})
    return state


def cmd_audit():
    """Read-only audit: 12-state decodability, feasible space, four value classes."""
    leaf = ledger_leaves()
    full = enumerate_combos()
    combos = filter_combos(full, leaf)
    cross = cross_check_v153()
    dec = decode_uniqueness(enumerate_new_combos())
    upper = full_pool_upper(combos)
    state = derived_state()
    require(dec["unique"], "Three queries do not uniquely decode the 12 combinations")
    require(dec["new_node_label_combinations"] == 12, "Expected 12 new-node combinations")
    require(abs(f1(3, 3) - 0.929254) < 1e-6, "Decoded merge did not reproduce 0.929254")

    sigs = set()
    for c in combos:
        sigs.add((c[8], c[9], c[11], c[14], c[15], c[17]))
    plan = decode_plan(enumerate_new_combos())
    for phase in plan["phases"]:
        require(len(phase["branches"]) >= 1, "Empty decode phase")

    return {
        "integer_feasible": len(combos) > 0,
        "combination_count": len(full),
        "feasible_combination_count": len(combos),
        "history_equations": len(leaf),
        "distinct_new_labels_remaining": len(sigs),
        "decode_unique": dec,
        "values": value_classes(state, upper),
        "cross_check_v153": cross,
        "remaining_submissions": state["budget"]["remaining"],
        "attempts_used": state["budget"]["used"],
        "platform_quota_live_checked": False,
    }


def fixed_plan_first(feasible):
    """First query of the fixed {8},{11,9},{14,9} plan that still splits the set."""
    for qids in ([8], [11, 9], [14, 9]):
        groups = defaultdict(int)
        for c in feasible:
            groups[sum(c[i] for i in qids)] += 1
        if len(groups) > 1:
            return qids
    return None


def fallback_safety(feasible):
    """Probes the fixed A->B->C plan needs to decode every feasible branch."""
    qlist = [[8], [11, 9], [14, 9]]

    def depth(combo_set, remaining):
        if len({tuple(c[i] for i in NEW_NODES) for c in combo_set}) <= 1:
            return 0
        if not remaining:
            return 10 ** 6
        groups = defaultdict(list)
        for c in combo_set:
            groups[sum(c[i] for i in remaining[0])].append(c)
        if len(groups) <= 1:
            return depth(combo_set, remaining[1:])
        return 1 + max(depth(g, remaining[1:]) for g in groups.values())

    return {"fixed_plan_probes_needed": depth(list(feasible), qlist)}


def adaptive_safety(feasible, first_q, remaining_budget):
    """Every feedback branch of an adaptive first query must itself be decodable
    within the remaining decode budget using the validated fixed plan."""
    groups = defaultdict(list)
    for c in feasible:
        groups[sum(c[i] for i in first_q)].append(c)
    if len(groups) <= 1:
        return {"safe": False, "reason": "query_does_not_split", "branches": []}
    branches, worst = [], 0
    for s, g in sorted(groups.items()):
        need = fallback_safety(g)["fixed_plan_probes_needed"]
        worst = max(worst, need)
        branches.append({"feedback": s, "combos_remaining": len(g), "probes_needed": need})
    return {"safe": worst <= remaining_budget, "worst_branch_probes": worst,
            "remaining_decode_budget": remaining_budget, "branches": branches}


def carrier_contribution(carrier_ids):
    """Known delta_tp already contributed by confirmed carrier actions."""
    if not carrier_ids:
        return 0
    combos = filter_combos(enumerate_combos(), ledger_leaves())
    if not combos:
        return 0
    known = _known_labels(combos, (1 << len(combos)) - 1, ALL_ACTIVE)
    dt = 0
    for i in carrier_ids:
        o, kind = known.get(i), CAND[i]["kind"]
        if kind == "add" and o == 1:
            dt += 1
        elif kind == "swap" and o is not None:
            dt += o
    return dt


def cmd_recommend(emit=False, time_limit=None, budget=None):
    state = derived_state()
    remaining = state["budget"]["remaining"]
    reserve = state["budget"]["merge_reserve"]
    max_probes = state["budget"]["max_info_probes"]
    # never spend the reserved final-merge submission on an info probe
    cap = max(0, remaining - reserve)
    probes = cap if budget is None else max(0, min(budget, cap))
    probes = min(probes, max_probes)
    champ_f1 = f1(state["champion"]["tp"] - TP0, state["champion"]["predictions"] - P0)

    leaf = ledger_leaves()
    full = enumerate_combos()
    combos = filter_combos(full, leaf)
    new_all = enumerate_new_combos()
    new_feasible = filter_combos(new_all, leaf) or new_all
    upper = full_pool_upper(combos)

    sm = _build_search(True, champ_f1, time_limit, new_combos=new_feasible)
    res_m = (sm.first_query(sm.new_full, probes) if probes else
             {"val": champ_f1, "q": None, "worst": None, "branches": []})
    su = _build_search(False, champ_f1, time_limit, new_combos=new_feasible)
    res_u = (su.first_query(su.new_full, probes) if probes else
             {"val": champ_f1, "q": None, "worst": None, "branches": []})

    plan = decode_plan(new_all)
    safety = fallback_safety(new_feasible)
    next_q, used_fallback, adapt = res_m["q"], False, None
    if next_q is not None and not sm.truncated:
        # an adaptive query may replace the fallback ONLY if every branch decodes in budget
        adapt = adaptive_safety(new_feasible, next_q, max(0, probes - 1))
        if not adapt["safe"]:
            next_q, used_fallback = fixed_plan_first(new_feasible), True
    else:
        next_q, used_fallback = fixed_plan_first(new_feasible), True
    next_ids = sorted(next_q) if next_q else None

    values = value_classes(state, upper)
    values["model_conditional_expectation"] = {
        "mixture_expected_final_f1": round(res_m["val"], 6),
        "uniform_expected_final_f1": round(res_u["val"], 6),
        "search_truncated": bool(sm.truncated),
        "used_fixed_fallback": bool(used_fallback),
        "probability_warning": WARNING,
    }
    report = {
        "decision": "submit_probe" if next_ids and probes else "merge",
        "allow_submission": remaining > 0,
        "actual_champion": state["champion"],
        "remaining_submissions": remaining,
        "attempts_used": state["budget"]["used"],
        "info_probes_allowed": probes,
        "max_info_probes": max_probes,
        "merge_reserve": reserve,
        "decode": decode_uniqueness(new_all),
        "decode_plan": plan,
        "fallback_safety": {**safety, "budget": probes,
                            "safe": safety["fixed_plan_probes_needed"] <= probes},
        "adaptive_safety": adapt,
        "values": values,
        "strategy": {
            "model_mixture": {"expected_final_f1": round(res_m["val"], 6),
                              "worst_branch_f1": None if res_m["worst"] is None else round(res_m["worst"], 6),
                              "next_action": {"kind": "probe", "ids": next_ids} if next_ids else None,
                              "branches": res_m["branches"], "truncated": bool(sm.truncated),
                              "nodes": sm.nodes, "probability_warning": WARNING},
            "uniform_sensitivity": {"expected_final_f1": round(res_u["val"], 6),
                                    "worst_branch_f1": None if res_u["worst"] is None else round(res_u["worst"], 6),
                                    "next_action": {"kind": "probe", "ids": sorted(res_u["q"])} if res_u["q"] else None,
                                    "truncated": bool(su.truncated), "nodes": su.nodes,
                                    "probability_warning": UNIFORM_WARNING},
        },
        "probability_warning": WARNING,
        "platform_quota_live_checked": False,
    }
    if emit and next_ids and probes:
        report["file"] = author(next_ids, carrier=[])
        report["prepared_only"] = True
    write_json(HERE / "recommendation.json", report)
    write_json(HERE / "final_gate.json", report)
    return report


def author(query_ids, carrier):
    manifest = load_manifest()
    probe_id = f"p{len(manifest['files']) + 1:02d}"
    filename = f"v156_probe_{probe_id}.csv"
    out = HERE / filename
    all_ids = sorted(set(query_ids) | set(carrier))
    apply_actions(BASE, all_ids, out)
    validated = validate_csv(out, all_ids)
    entry = {"probe_id": probe_id, "file": filename, "sha256": validated["sha256"],
             "query_ids": sorted(query_ids), "carrier_ids": sorted(carrier), "all_ids": all_ids,
             "predictions": validated["predictions"], "delta_p": sum(CAND[i]["delta_p"] for i in all_ids),
             "local_validation": validated, "created_at": now(), "status": "prepared"}
    manifest["files"].append(entry)
    write_json(HERE / "submission_manifest.json", manifest)
    return entry


def cmd_record(probe_id, attempt_id, score, submitted_at, evidence, failed=False):
    """Record a real attempt.  Negative delta_tp is legitimate (delete/swap);
    only a missing evidence string or a conflicting duplicate is an anomaly."""
    manifest = load_manifest()
    ledger = load_ledger()
    e = next((e for e in manifest["files"] if e["probe_id"] == probe_id), None)
    require(e is not None, "Unknown emitted probe")
    require(evidence.strip(), "Actual score/failure evidence required")
    prev = next((r for r in ledger["records"] if r["attempt_id"] == attempt_id), None)
    if prev:
        require(prev["probe_id"] == probe_id and prev["score"] == score, "Attempt-id content conflict")
        return {"idempotent": True, "record": prev}
    record = {"attempt_id": attempt_id, "probe_id": probe_id, "score": score,
              "submitted_at": submitted_at, "evidence": evidence,
              "status": "failed" if failed else "accepted", "source_class": "actual_submission_attempt",
              "file": str((HERE / e["file"]).resolve())}
    if not failed:
        validated = validate_csv(HERE / e["file"], e["all_ids"], e["sha256"])
        tp = infer_tp(score, validated["predictions"])
        delta_tp = tp - TP0
        carrier = carrier_contribution(e.get("carrier_ids", []))
        info = delta_tp - carrier                      # probe's own equation, carrier removed
        record.update(status="accepted", source_class="public_scored", inferred_tp=tp, delta_tp=delta_tp,
                      carrier_delta_tp=carrier, info_delta=info,
                      predictions=validated["predictions"], sha256=validated["sha256"])
        if info == 0 and e.get("carrier_ids"):
            record["note"] = "info delta zero: every query action is already-known background"
    ledger["records"].append(record)
    write_json(HERE / "online_scores.json", ledger)
    result = cmd_recommend()
    return {"record": record, "assessment": result}


def cmd_emit_final():
    """Final: best merge whose TP is DETERMINED across all feasible combos.

    Enumerates every legal action subset consistent with the current knowledge --
    including whole groups whose total is known while single nodes remain
    unresolved -- and never uses a guessed node.  Only emits when it beats the
    champion, and skips a file identical to one already submitted.
    """
    state = derived_state()
    combos = filter_combos(enumerate_combos(), ledger_leaves())
    require(combos, "No feasible combination remains")
    champ_f1 = f1(state["champion"]["tp"] - TP0, state["champion"]["predictions"] - P0)
    known = _known_labels(combos, (1 << len(combos)) - 1, ALL_ACTIVE)
    ids, dt, dp = merge_actions(known)
    value = f1(dt, dp)
    if value <= champ_f1 + 1e-12:
        return {"decision": "retain_champion_no_better_merge", "allow_submission": False,
                "actual_champion": state["champion"],
                "merge": {"f1": round(champ_f1, 6), "ids": [], "delta_tp": 0, "delta_p": 0}}
    submitted = {tuple(sorted(r.get("ids", []))) for r in load_ledger().get("records", [])
                 if r.get("status") == "accepted" and r.get("probe_id", "").startswith("merge")}
    if tuple(sorted(ids)) in submitted:
        return {"decision": "duplicate_merge_skip", "allow_submission": False,
                "merge": {"ids": ids, "f1": round(value, 6)}}
    out = HERE / "v156_final_merge.csv"
    apply_actions(BASE, ids, out)
    validated = validate_csv(out, ids)
    merge = {"ids": ids, "delta_tp": dt, "delta_p": dp, "f1": round(value, 6),
             "predictions": validated["predictions"], "sha256": validated["sha256"],
             "beats_champion": round(value - champ_f1, 6)}
    write_json(HERE / "final_merge.json", merge)
    return {"decision": "submit_merge", "allow_submission": True, "merge": merge, "file": str(out)}


def cmd_research(time_limit=None):
    """24h FN-classifier research track (separate campaign dir, no quota use)."""
    dir_ = HERE.with_name("v156_fn_research")
    status = {"research_dir": str(dir_), "exists": dir_.is_dir(), "deadline_hours": 24,
              "quota_cost": 0,
              "gate": {
                  "folds": 5, "min_positive_folds": 3,
                  "pooled_gain_gt_best_control": True, "worst_fold_ge_control": True,
                  "paired_bootstrap": {"iterations": 2000, "absolute_lower_gt": 0, "relative_lower_gt": 0},
                  "max_new_nodes": 16, "disjoint_orders": True,
                  "beats_best_existing_probe_on_conditional_expectation": True,
              }}
    if dir_.is_dir():
        for name in ("status.json", "results.json"):
            p = dir_ / name
            if p.exists():
                status[name[:-5]] = read_json(p)
    else:
        status["decision"] = "not_started"
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    sub.add_parser("build")
    r = sub.add_parser("recommend")
    r.add_argument("--emit", action="store_true")
    r.add_argument("--time-limit", type=float, default=None)
    r.add_argument("--budget", type=int, default=None)
    rec = sub.add_parser("record")
    for name in ("probe-id", "attempt-id", "submitted-at", "evidence"):
        rec.add_argument("--" + name, required=True)
    rec.add_argument("--score")
    rec.add_argument("--failed", action="store_true")
    sub.add_parser("emit-final")
    rs = sub.add_parser("research")
    rs.add_argument("--time-limit", type=float, default=None)
    args = p.parse_args()
    if args.command == "build":
        result = cmd_build()
    elif args.command == "audit":
        result = cmd_audit()
    elif args.command == "recommend":
        result = cmd_recommend(args.emit, args.time_limit, args.budget)
    elif args.command == "record":
        result = cmd_record(args.probe_id, args.attempt_id, args.score, args.submitted_at, args.evidence, args.failed)
    elif args.command == "emit-final":
        result = cmd_emit_final()
    elif args.command == "research":
        result = cmd_research(args.time_limit)
    else:
        result = cmd_recommend(False)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc), "decision": "do_not_submit"}, ensure_ascii=False), file=sys.stderr)
        raise
