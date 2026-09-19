"""Research-only nearest-order semantic transfer audit.

This module searches for same-order one-for-one replacements by transferring
root labels from *observable* nearest training orders.  It deliberately does
not use the query order's labels when constructing its neighbours or scores;
the train audit is leave-one-order-out.  It writes only a JSON research
report, never a submission CSV and never modifies a baseline.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
NPZ = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
BASE = ROOT / "experiments/v60_combined_checkpoint/highest_verified_combined.csv"
OUT = ROOT / "experiments/research_nearest_order_transfer"
MAX_ROOTS = 8


def s(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return "|".join(s(x) for x in v)
    return str(v)


def norm(v: Any) -> str:
    # The source export contains a mixture of normal and mojibake text.  We
    # preserve code points for exact matching but remove volatile numbers and
    # whitespace for the shape tiers.
    return re.sub(r"\s+", " ", re.sub(r"\d+", "<NUM>", s(v))).strip()


def raw(v: Any) -> str:
    return re.sub(r"\s+", " ", s(v)).strip()


def alarm_tier(a: dict[str, Any], tier: int) -> tuple[str, ...]:
    title = raw(a.get("title")); cause = raw(a.get("cause"))
    dev = raw(a.get("device_type")); board = raw(a.get("board_type"))
    radio = raw(a.get("radio")); deploy = raw(a.get("deployment"))
    reason = norm(a.get("reason")); timeline = raw(a.get("timeline"))
    loc = norm(a.get("location"))
    neigh = tuple(sorted(raw(x) for x in (a.get("neighbor_titles") or [])))
    if tier == 0:
        return (title, reason, cause, dev, board, radio, deploy, timeline, loc, *neigh)
    if tier == 1:
        return (title, reason, cause, dev, board, radio, deploy, timeline)
    if tier == 2:
        return (title, cause, dev, board, radio, deploy)
    if tier == 3:
        return (title, cause, dev)
    return (title,)


def multiset_jaccard(a: Counter, b: Counter) -> float:
    if not a and not b:
        return 1.0
    inter = sum((a & b).values())
    union = sum((a | b).values())
    return inter / union if union else 0.0


def set_jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if a | b else 0.0


def profile(o: dict[str, Any]) -> dict[str, Any]:
    alarms = o["alarms"]
    title = Counter(raw(a.get("title")) for a in alarms)
    # Attribute profiles exclude station/location and therefore are usable for
    # cross-station nearest-neighbour transfer.
    attr = Counter(alarm_tier(a, 2) for a in alarms)
    reason = Counter((raw(a.get("title")), norm(a.get("reason"))) for a in alarms)
    cause = Counter(raw(a.get("cause")) for a in alarms)
    boards = Counter(raw(a.get("board_type")) for a in alarms)
    timeline = Counter(raw(a.get("timeline")) for a in alarms)
    return {
        "n": len(alarms), "title": title, "attr": attr, "reason": reason,
        "cause": cause, "boards": boards, "timeline": timeline,
        "title_set": set(title), "attr_set": set(attr),
        "stations": set(raw(x) for x in (o.get("station_ids") or [])),
    }


def similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    # Weighted, station-independent similarity.  Exact alarm attributes have
    # more weight than broad title overlap; order size is a small stabilizer.
    tj = multiset_jaccard(a["title"], b["title"])
    aj = multiset_jaccard(a["attr"], b["attr"])
    rj = multiset_jaccard(a["reason"], b["reason"])
    cj = multiset_jaccard(a["cause"], b["cause"])
    bj = multiset_jaccard(a["boards"], b["boards"])
    size = 1.0 - min(abs(a["n"] - b["n"]) / max(a["n"], b["n"], 1), 1.0)
    return 0.36 * tj + 0.30 * aj + 0.12 * rj + 0.08 * cj + 0.06 * bj + 0.08 * size


def similarity_matrix(left: list[dict[str, Any]], right: list[dict[str, Any]],
                      same: bool = False) -> np.ndarray:
    """Precompute profile similarities once; repeated Python Counter work is
    prohibitively slow across the LOSO/configuration grid."""
    out = np.zeros((len(left), len(right)), dtype=np.float32)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            if same and i == j:
                out[i, j] = -1.0
            else:
                out[i, j] = similarity(a, b)
    return out


def nearest_matrix_row(scores: np.ndarray, qprof: dict[str, Any],
                       refs_profiles: list[dict[str, Any]], k: int,
                       min_score: float = 0.0) -> list[tuple[float, dict[str, Any]]]:
    """Fast nearest lookup with a station-disjoint preference."""
    order = np.argsort(-scores, kind="stable")
    vals = [(float(scores[j]), j) for j in order if float(scores[j]) >= min_score]
    dis = [(v, j) for v, j in vals
           if qprof["stations"].isdisjoint(refs_profiles[j]["stations"])]
    if len(dis) >= max(3, min(k, 5)):
        vals = dis
    return [(v, {"order": None, "profile": refs_profiles[j], "index": j}) for v, j in vals[:k]]


def matrix_neighbours(scores: np.ndarray, qi: int, query_profile: dict[str, Any],
                      orders: list[dict[str, Any]], profiles: list[dict[str, Any]],
                      k: int, min_score: float, exclude_self: bool = False) -> list[tuple[float, dict[str, Any]]]:
    row = np.asarray(scores[qi], dtype=np.float32).copy()
    if exclude_self:
        row[qi] = -1.0
    order = np.argsort(-row, kind="stable")
    vals = [(float(row[j]), int(j)) for j in order if float(row[j]) >= min_score]
    dis = [(v, j) for v, j in vals
           if query_profile["stations"].isdisjoint(profiles[j]["stations"])]
    if len(dis) >= max(3, min(k, 5)):
        vals = dis
    return [(v, {"order": orders[j], "profile": profiles[j], "index": j}) for v, j in vals[:k]]


def load_base() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with BASE.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[row["order_id"]] = {x["@rid"] for x in json.loads(row["output"])["rootcause"]}
    return out


def exact_count_mask(scores: np.ndarray, ptr: np.ndarray, target: int) -> np.ndarray:
    selected = np.zeros(len(scores), dtype=bool)
    optional: list[int] = []
    for st, en in zip(ptr[:-1], ptr[1:]):
        st, en = int(st), int(en)
        ranked = np.argsort(-scores[st:en], kind="stable")[:MAX_ROOTS]
        if len(ranked):
            selected[st + int(ranked[0])] = True
            optional.extend((st + ranked[1:]).tolist())
    optional = np.asarray(optional, dtype=np.int64)
    if optional.size:
        optional = optional[np.argsort(-scores[optional], kind="stable")]
    rem = int(target) - int(selected.sum())
    if rem < 0 or rem > len(optional):
        raise ValueError((target, int(selected.sum()), len(optional)))
    selected[optional[:rem]] = True
    return selected


def neighbour_list(query: dict[str, Any], qprof: dict[str, Any], refs: list[dict[str, Any]],
                   k: int, min_score: float = 0.0) -> list[tuple[float, dict[str, Any]]]:
    vals = []
    qsites = qprof["stations"]
    for r in refs:
        # Same station can leak near-identical repeated incidents.  Prefer
        # disjoint references; caller falls back when too few exist.
        rp = r["profile"]
        score = similarity(qprof, rp)
        if score >= min_score:
            vals.append((score, r))
    vals.sort(key=lambda x: (-x[0], x[1]["order"]["order_id"]))
    dis = [(v, r) for v, r in vals if qsites.isdisjoint(r["profile"]["stations"])]
    if len(dis) >= max(3, min(k, 5)):
        vals = dis
    return vals[:k]


def transfer_scores(query: dict[str, Any], neighbours: list[tuple[float, dict[str, Any]]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (score, support, exact_tier) for each query alarm."""
    # Build per-tier weighted maps. Each neighbour contributes at most one
    # vote per key (duplicate alarm keys are averaged within that order).
    maps: list[dict[tuple[str, ...], list[tuple[float, float]]]] = [defaultdict(list) for _ in range(5)]
    for sim, ref in neighbours:
        bytier = [defaultdict(list) for _ in range(5)]
        for a in ref["order"]["alarms"]:
            y = float(a.get("is_root") or 0)
            for t in range(5):
                bytier[t][alarm_tier(a, t)].append(y)
        for t in range(5):
            for key, ys in bytier[t].items():
                maps[t][key].append((sim, float(np.mean(ys))))
    scores=[]; supports=[]; tiers=[]
    for a in query["alarms"]:
        val = None; sup=0; chosen=4
        for t in range(5):
            vals = maps[t].get(alarm_tier(a,t), [])
            # Require at least two independent refs for a transfer.  The
            # score is similarity-weighted and shrinks towards 0.5.
            if len(vals) >= 2:
                w = np.asarray([max(v, 1e-6) ** 3 for v,_ in vals], dtype=float)
                yy = np.asarray([y for _,y in vals], dtype=float)
                val = float(np.dot(w, yy) / w.sum())
                sup = len(vals); chosen=t
                break
        if val is None:
            # No analogue: neutral score, zero support.
            val=0.5; sup=0; chosen=4
        scores.append(val); supports.append(sup); tiers.append(chosen)
    return np.asarray(scores), np.asarray(supports), np.asarray(tiers)


def evaluate_split(orders: list[dict[str, Any]], profiles: list[dict[str, Any]],
                   refs_idx: list[int], train_scores: np.ndarray, ptr: np.ndarray,
                   target: int, k: int, min_sim: float, thresholds: list[float]) -> dict[str, Any]:
    # Build the exact-count analogue of the known V11 baseline.
    base = exact_count_mask(train_scores, ptr, target)
    rows=[]; pair_rows=[]; changed_orders=0
    refs = [{"order": orders[i], "profile": profiles[i]} for i in refs_idx]
    for qi, q in enumerate(orders):
        qprof=profiles[qi]
        ns=neighbour_list(q,qprof,refs,k,min_sim)
        # remove self when refs_idx contains query; for LOSO it does not.
        ps,sup,tiers=transfer_scores(q,ns)
        st,en=int(ptr[qi]),int(ptr[qi+1]); n=en-st
        kk=int(base[st:en].sum())
        # Preserve baseline count and per-order cap.
        rank=np.argsort(-ps,kind='stable')[:kk]
        new=np.zeros(n,dtype=bool); new[rank]=True
        old=base[st:en]
        rem=np.flatnonzero(old & ~new); add=np.flatnonzero(new & ~old)
        if len(rem)!=len(add):
            continue
        if len(rem): changed_orders+=1
        truth=np.asarray([int(a.get('is_root') or 0) for a in q['alarms']],dtype=np.int8)
        delta=int(truth[add].sum()-truth[rem].sum())
        rows.append({"order_id":q['order_id'],"similarity":float(ns[0][0]) if ns else 0.0,
                     "neighbours":len(ns),"baseline_k":kk,"changed":len(rem),"delta":delta,
                     "tp_old":int(truth[old].sum()),"tp_new":int(truth[new].sum()),
                     "max_support":int(sup.max()) if len(sup) else 0,
                     "mean_support":float(sup.mean()) if len(sup) else 0.0})
        for rr,aa in zip(rem,add):
            pair_rows.append({"order_id":q['order_id'],"remove":q['alarms'][int(rr)]['rid'],"add":q['alarms'][int(aa)]['rid'],
                              "remove_score":float(ps[rr]),"add_score":float(ps[aa]),
                              "remove_support":int(sup[rr]),"add_support":int(sup[aa]),
                              "delta":int(truth[aa]-truth[rr]),"similarity":float(ns[0][0]) if ns else 0.0})
    out={"target":target,"k":k,"min_similarity":min_sim,"orders":len(rows),"changed_orders":changed_orders,
         "positive_orders":sum(r['delta']>0 for r in rows),"nonnegative_orders":sum(r['delta']>=0 for r in rows),
         "delta_sum":sum(r['delta'] for r in rows),"delta_mean":float(np.mean([r['delta'] for r in rows])) if rows else 0.0,
         "pair_count":len(pair_rows),"pair_positive_rate":float(np.mean([r['delta']>0 for r in pair_rows])) if pair_rows else 0.0,
         "pair_nonnegative_rate":float(np.mean([r['delta']>=0 for r in pair_rows])) if pair_rows else 0.0,
         "pairs":pair_rows,"rows":rows}
    for th in thresholds:
        chosen=[r for r in pair_rows if r['add_score']>=th and r['remove_score']<=1-th and r['add_support']>=2 and r['remove_support']>=2]
        out[f"threshold_{th:.2f}"]={"count":len(chosen),"positive":sum(r['delta']>0 for r in chosen),
                                    "positive_rate":float(np.mean([r['delta']>0 for r in chosen])) if chosen else 0.0,
                                    "mean_delta":float(np.mean([r['delta'] for r in chosen])) if chosen else 0.0}
    return out


def test_candidates(test: list[dict[str, Any]], test_profiles: list[dict[str, Any]], train: list[dict[str, Any]], train_profiles: list[dict[str, Any]], base: dict[str, set[str]], k: int, min_sim: float, threshold: float) -> list[dict[str, Any]]:
    refs=[{"order":o,"profile":p} for o,p in zip(train,train_profiles)]
    out=[]
    for q,p in zip(test,test_profiles):
        ns=neighbour_list(q,p,refs,k,min_sim)
        ps,sup,tiers=transfer_scores(q,ns)
        cur=base.get(q['order_id'],set()); n=len(q['alarms']); kk=len(cur)
        # Rank only with transfer scores, retaining the champion count.
        rank=np.argsort(-ps,kind='stable')[:kk]; new={q['alarms'][int(i)]['rid'] for i in rank}
        rem=sorted(cur-new); add=sorted(new-cur)
        if len(rem)!=len(add): continue
        for rr,aa in zip(rem,add):
            ri=next(i for i,a in enumerate(q['alarms']) if a['rid']==rr); ai=next(i for i,a in enumerate(q['alarms']) if a['rid']==aa)
            if ps[ai] < threshold or ps[ri] > 1-threshold or sup[ai]<2 or sup[ri]<2: continue
            out.append({"order_id":q['order_id'],"remove_rid":rr,"add_rid":aa,"remove_score":float(ps[ri]),"add_score":float(ps[ai]),
                        "remove_support":int(sup[ri]),"add_support":int(sup[ai]),"similarity":float(ns[0][0]) if ns else 0.0,
                        "neighbours":len(ns),"tier_add":int(tiers[ai]),"tier_remove":int(tiers[ri])})
    # One action per order for independence and no overlap.
    out.sort(key=lambda x:(-(x['add_score']-x['remove_score']),-x['similarity'],x['order_id']))
    return out


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    d=json.load(gzip.open(DATA,'rt',encoding='utf-8')); train=d['train']; test=d['test']
    with np.load(NPZ) as z: arr={k:z[k] for k in z.files}
    train_ptr=arr['train_alarm_ptr']; train_scores=arr['train_v11'];
    base=load_base()
    train_profiles=[profile(o) for o in train]; test_profiles=[profile(o) for o in test]
    # Leave-one-order-out for every query.  Thresholds and target counts are
    # reported separately to expose sensitivity to the proxy baseline.
    # Recompute every metric with explicit per-query exclusion.  The evaluator
    # below is intentionally local so it is impossible for the query order to
    # enter its own reference set.
    train_sim = similarity_matrix(train_profiles, train_profiles, same=True)
    test_sim = similarity_matrix(test_profiles, train_profiles, same=False)

    def eval_loso(target,k,ms):
        base_mask=exact_count_mask(train_scores,train_ptr,target); rows=[]; pairs=[]
        for qi,q in enumerate(train):
            ns=matrix_neighbours(train_sim,qi,train_profiles[qi],train,train_profiles,k,ms,exclude_self=True)
            ps,sup,tiers=transfer_scores(q,ns)
            st,en=int(train_ptr[qi]),int(train_ptr[qi+1]); kk=int(base_mask[st:en].sum()); old=base_mask[st:en]; rank=np.argsort(-ps,kind='stable')[:kk]; new=np.zeros(en-st,bool); new[rank]=True
            rem=np.flatnonzero(old&~new); add=np.flatnonzero(new&~old)
            if len(rem)!=len(add): continue
            truth=np.asarray([int(a.get('is_root') or 0) for a in q['alarms']],np.int8)
            rows.append(int(truth[add].sum()-truth[rem].sum()))
            for rr,aa in zip(rem,add): pairs.append({"delta":int(truth[aa]-truth[rr]),"add_score":float(ps[aa]),"remove_score":float(ps[rr]),"add_support":int(sup[aa]),"remove_support":int(sup[rr]),"order_id":q['order_id']})
        out={"target":target,"k":k,"min_similarity":ms,"orders":len(rows),"changed_orders":sum(x!=0 for x in rows),"positive_orders":sum(x>0 for x in rows),"nonnegative_orders":sum(x>=0 for x in rows),"delta_sum":sum(rows),"delta_mean":float(np.mean(rows)) if rows else 0.0,"pair_count":len(pairs),"pair_positive_rate":float(np.mean([x['delta']>0 for x in pairs])) if pairs else 0.0,"pair_nonnegative_rate":float(np.mean([x['delta']>=0 for x in pairs])) if pairs else 0.0}
        for th in (.65,.70,.75,.80):
            c=[x for x in pairs if x['add_score']>=th and x['remove_score']<=1-th and x['add_support']>=2 and x['remove_support']>=2]
            out[f"threshold_{th:.2f}"]={"count":len(c),"positive":sum(x['delta']>0 for x in c),"positive_rate":float(np.mean([x['delta']>0 for x in c])) if c else 0.0,"mean_delta":float(np.mean([x['delta'] for x in c])) if c else 0.0}
        return out
    # Six representative settings are enough to expose stability without
    # turning this research audit into a submission-time computation.
    configs=[eval_loso(t,k,ms) for t in (3041,3097,3169) for k in (10,20) for ms in (.15,)]
    # Test catalog at each threshold using the current champion count. Keep
    # only one candidate per order in the published report.
    catalogs={}
    for th in (.65,.70,.75,.80):
        cs=[]
        for qi,q in enumerate(test):
            ns=matrix_neighbours(test_sim,qi,test_profiles[qi],train,train_profiles,10,.15,exclude_self=False)
            ps,sup,tiers=transfer_scores(q,ns)
            cur=base.get(q['order_id'],set()); kk=len(cur)
            rank=np.argsort(-ps,kind='stable')[:kk]; new={q['alarms'][int(i)]['rid'] for i in rank}
            rem=sorted(cur-new); add=sorted(new-cur)
            if len(rem)!=len(add): continue
            for rr,aa in zip(rem,add):
                ri=next(i for i,a in enumerate(q['alarms']) if a['rid']==rr); ai=next(i for i,a in enumerate(q['alarms']) if a['rid']==aa)
                if ps[ai] < th or ps[ri] > 1-th or sup[ai]<2 or sup[ri]<2: continue
                cs.append({"order_id":q['order_id'],"remove_rid":rr,"add_rid":aa,"remove_score":float(ps[ri]),"add_score":float(ps[ai]),"remove_support":int(sup[ri]),"add_support":int(sup[ai]),"similarity":float(ns[0][0]) if ns else 0.0,"neighbours":len(ns),"tier_add":int(tiers[ai]),"tier_remove":int(tiers[ri])})
        seen=set(); uniq=[]
        for c in cs:
            if c['order_id'] in seen: continue
            seen.add(c['order_id']); uniq.append(c)
        catalogs[f"threshold_{th:.2f}"]={"count":len(uniq),"candidates":uniq[:300]}
    report={"method":"nearest_order_transfer_loso_v1","train_orders":len(train),"test_orders":len(test),"baseline_path":str(BASE),"baseline_p":sum(map(len,base.values())),"configs":configs,"test_catalogs":catalogs,"notes":["station/location excluded from similarity and alarm keys except optional shape", "all train scores are leave-one-order-out", "no submission file emitted"]}
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    summary={"train_orders":len(train),"test_orders":len(test),"baseline_p":sum(map(len,base.values())),"configs": [{k:v for k,v in c.items() if k not in ('pairs','rows')} for c in configs],"test_catalog_counts":{k:v['count'] for k,v in catalogs.items()}}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
