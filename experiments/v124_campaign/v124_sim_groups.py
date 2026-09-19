"""Simulate the V124 group-decode campaign to estimate P(V>=26) and V distribution.

Per-action delta: +1 with p = nominal_p_net_gain, 0 with p_neutral, -1 otherwise.
Group net = sum of deltas. Decode: >0 keep whole group; =0 split-pool; -1 secondary
split-pool; <=-2 drop. Splits recover positive single candidates (approx: a 0-group
yields +1 with prob = max nominal in group, and the remaining pair is then 0/-1;
-1-group yields +1 with prob = max nominal).
"""
import json
import numpy as np

cat = json.load(open(r"D:/zgyidong/experiments/v120_swap_campaign/candidate_catalog.json", encoding="utf-8"))
by_id = {c["candidate_id"]: c for c in cat["candidates"]}

GROUPS = {
    1: ["v120_pair_008", "v120_pair_014", "v120_pair_044"],
    2: ["v120_pair_049", "v120_pair_084", "v120_pair_015"],
    3: ["v120_pair_011", "v120_pair_018", "v120_pair_045"],
    4: ["v120_pair_030", "v120_pair_054", "v120_pair_033"],
    5: ["v120_pair_075", "v120_pair_076", "v120_pair_017"],
    6: ["v120_pair_016", "v120_pair_052", "v120_pair_042"],
    7: ["v120_pair_083", "v120_pair_073", "v120_pair_046"],
    8: ["v120_pair_036", "v120_pair_004", "v120_pair_041"],
    9: ["v120_pair_013", "v120_pair_029", "v120_pair_056"],
    10: ["v120_pair_087", "v120_pair_077", "v120_pair_065"],
    11: ["v120_pair_026", "v120_pair_001", "v120_pair_021"],
    12: ["v120_pair_022", "v120_pair_055"],
}

# nominal probabilities per candidate
nom = {}
for g in GROUPS.values():
    for cid in g:
        nom[cid] = by_id[cid]["nominal_p_net_gain"]

names = [cid for g in GROUPS.values() for cid in g]
p_plus = np.asarray([nom[c] for c in names], dtype=np.float64)
P_NEUTRAL = 0.15
p_zero = np.full(len(names), P_NEUTRAL)
p_minus = np.clip(1.0 - p_plus - p_zero, 0.0, 1.0)

# group membership
g_of = []
for g in GROUPS.values():
    for cid in g:
        g_of.append(int(list(GROUPS.keys())[[i for i, gg in enumerate(GROUPS.values()) if cid in gg][0]] - 1))

rng = np.random.default_rng(20260825)
T = 200_000
deltas = np.zeros((T, len(names)), dtype=np.int8)
u = rng.random((T, len(names)))
deltas[u < p_plus] = 1
deltas[(u >= p_plus) & (u < p_plus + p_zero)] = 0
deltas[(u >= p_plus + p_zero)] = -1

V = np.zeros(T, dtype=np.float64)
for gi, g in enumerate(GROUPS.values()):
    idx = [i for i, gg in enumerate(g_of) if gg == gi]
    gsum = deltas[:, idx].sum(axis=1)
    best = p_plus[idx].max()
    # decode: >0 keep; =0 split (recover +1 w.p. ~best, else 0); -1 secondary split; <=-2 drop
    keep = gsum.copy()
    zero_mask = gsum == 0
    neg1_mask = gsum == -1
    keep[gsum <= -2] = 0
    keep[zero_mask] = rng.random(zero_mask.sum()) < best  # +1 if recovered
    keep[neg1_mask] = (rng.random(neg1_mask.sum()) < best)  # +1 if single recovered, else 0
    V += keep

p_ge26 = float(np.mean(V >= 26))
p_ge20 = float(np.mean(V >= 20))
p_ge12 = float(np.mean(V >= 12))
print("35-candidate group-decode simulation (p_plus = nominal, P0 = 0.15):")
print("  E[V]      = %.2f" % V.mean())
print("  P(V>=26)  = %.4f" % p_ge26)
print("  P(V>=20)  = %.4f" % p_ge20)
print("  P(V>=12)  = %.4f" % p_ge12)
print("  V 分位: p10=%.1f p25=%.1f p50=%.1f p75=%.1f p90=%.1f" % tuple(np.percentile(V, q) for q in (10, 25, 50, 75, 90)))

# 如果线上把 nominal 打 9 折（V41 教训）
p_plus2 = p_plus * 0.9
u2 = rng.random((T, len(names)))
d2 = np.zeros((T, len(names)), dtype=np.int8)
d2[u2 < p_plus2] = 1
d2[(u2 >= p_plus2) & (u2 < p_plus2 + p_zero)] = 0
d2[(u2 >= p_plus2 + p_zero)] = -1
V2 = np.zeros(T, dtype=np.float64)
for gi, g in enumerate(GROUPS.values()):
    idx = [i for i, gg in enumerate(g_of) if gg == gi]
    gsum = d2[:, idx].sum(axis=1)
    best = p_plus2[idx].max()
    keep = gsum.copy()
    zero_mask = gsum == 0
    neg1_mask = gsum == -1
    keep[gsum <= -2] = 0
    keep[zero_mask] = rng.random(zero_mask.sum()) < best
    keep[neg1_mask] = rng.random(neg1_mask.sum()) < best
    V2 += keep
print("nominal*0.9 (decay): E[V]=%.2f P(V>=26)=%.4f P(V>=12)=%.4f" % (V2.mean(), float(np.mean(V2 >= 26)), float(np.mean(V2 >= 12))))

# 每候选只看 p_plus>=0.6 的组（前3组）单独模拟
names_hi = [cid for g in list(GROUPS.values())[:3] for cid in g]
print("前3组 (9候选, nominal 0.61-0.71) 期望: %.2f" % (sum(nom[c] for c in names_hi) - 9 * (1 - P_NEUTRAL - np.mean([nom[c] for c in names_hi]))))
