"""
Stage-1 gap quantification (F1/F2a/F2c) across 12 RE2-TT cases.
Read-only on both repos. Writes JSON to experiments/data/.
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "/home/jaredvel25/TraceRCA-CD")
from data.trainticket.download import simple_name
from trainticket_config import INVOLVED_SERVICES

INVOLVED_SET = set(INVOLVED_SERVICES)
CD_ROOT = Path("/home/jaredvel25/TraceRCA-CD")
FORK_ROOT = Path("/home/jaredvel25/RCAEval-Fork")
OUT_DIR = CD_ROOT / "experiments" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CASES = [
    # (full_name, fault_type, rep, gt_simple_name)
    ("ts-auth-service_cpu_1", "cpu", 1, "auth"),
    ("ts-auth-service_disk_1", "disk", 1, "auth"),
    ("ts-auth-service_socket_1", "socket", 1, "auth"),
    ("ts-order-service_cpu_1", "cpu", 1, "order"),
    ("ts-order-service_disk_1", "disk", 1, "order"),
    ("ts-order-service_socket_1", "socket", 1, "order"),
    ("ts-route-service_cpu_1", "cpu", 1, "route"),
    ("ts-route-service_disk_1", "disk", 1, "route"),
    ("ts-route-service_socket_1", "socket", 1, "route"),
    # OK-for-CD cases:
    ("ts-route-service_delay_1", "delay", 1, "route"),
    ("ts-train-service_delay_1", "delay", 1, "train"),
    ("ts-train-service_mem_1", "mem", 1, "train"),
]

def fork_case_dir(case):
    name, _, rep, _ = case
    base = "_".join(name.split("_")[:-1])  # ts-auth-service_cpu
    return FORK_ROOT / "data" / "re2tt" / base / str(rep)

def read_inject_us(case):
    fdir = fork_case_dir(case)
    return int((fdir / "inject_time.txt").read_text().strip()) * 1_000_000

def load_raw_csv(case):
    fdir = fork_case_dir(case)
    return pd.read_csv(
        fdir / "traces.csv",
        usecols=["traceID", "spanID", "serviceName", "methodName", "operationName",
                 "startTime", "duration", "statusCode", "parentSpanID"],
        dtype={"traceID": str, "spanID": str, "serviceName": str,
               "methodName": str, "operationName": str,
               "startTime": "int64", "duration": "int64",
               "statusCode": str, "parentSpanID": str},
        low_memory=False,
    )

def f1_drop(case):
    """Trace counts: raw post-inject traces vs. pkl test traces, plus serviceName mapping audit."""
    name, _, _, gt = case
    inject_us = read_inject_us(case)
    df = load_raw_csv(case)

    # Trace-level: a trace is post-inject if its earliest span's startTime >= inject_us
    min_per_trace = df.groupby("traceID")["startTime"].min()
    raw_post = int((min_per_trace >= inject_us).sum())
    raw_total = int(len(min_per_trace))
    raw_pre = raw_total - raw_post

    # PKL test traces
    pkl_path = CD_ROOT / "data" / "trainticket_re2tt" / "test" / f"{name}.pkl"
    with open(pkl_path, "rb") as f:
        pkl_traces = pickle.load(f)
    pkl_test_count = len(pkl_traces)

    # serviceName audit: which raw services have no INVOLVED mapping
    raw_services = sorted(df["serviceName"].dropna().unique().tolist())
    svc_to_simple = {s: simple_name(s) for s in raw_services}
    excluded = [(s, simple_name(s)) for s in raw_services if simple_name(s) not in INVOLVED_SET]
    included = [(s, simple_name(s)) for s in raw_services if simple_name(s) in INVOLVED_SET]
    gt_in_involved = gt in INVOLVED_SET

    # simple_name collision check
    from collections import Counter
    simple_counts = Counter([simple_name(s) for s in raw_services])
    collisions = {k: [s for s in raw_services if simple_name(s) == k] for k, c in simple_counts.items() if c > 1}

    return {
        "case": name,
        "fault_type": case[1],
        "gt_simple": gt,
        "gt_in_involved": gt_in_involved,
        "raw_total_traces": raw_total,
        "raw_pre_traces": raw_pre,
        "raw_post_traces": raw_post,
        "pkl_test_traces": pkl_test_count,
        "drop_pct": (raw_post - pkl_test_count) / max(raw_post, 1) * 100,
        "raw_service_count": len(raw_services),
        "excluded_services": excluded,
        "included_count": len(included),
        "simple_name_collisions": collisions,
    }

def load_useful_features(case_name):
    p = CD_ROOT / "output" / "trainticket_anomaly_detection.test" / f"{case_name}.useful_features.1"
    txt = p.read_text()
    # The file uses np.str_(...) literals -- create a shim.
    np_str_ = str
    return eval(txt, {"np": type("NS", (), {"str_": str})(), "np_str_": str})

def f2a_audit(case):
    """For GT service, list edges and which features survived."""
    name, _, _, gt = case
    uf = load_useful_features(name)
    # Normalize keys to plain str tuples
    norm = {(str(k[0]), str(k[1])): v for k, v in uf.items()}

    gt_edges = [k for k in norm if k[0] == gt or k[1] == gt]
    gt_edges_with_features = [k for k in gt_edges if norm[k]]
    gt_edges_no_features = [k for k in gt_edges if not norm[k]]

    return {
        "case": name,
        "gt": gt,
        "useful_features_total_keys": len(norm),
        "useful_features_empty": len(norm) == 0,
        "gt_edges_total_in_uf": len(gt_edges),
        "gt_edges_with_features": gt_edges_with_features,
        "gt_edges_no_features": gt_edges_no_features,
    }

def load_invo_result(case_name):
    p = CD_ROOT / "output" / "trainticket_anomaly_detection.test" / f"{case_name}.invo.result.pkl.1.1"
    with open(p, "rb") as f:
        df = pickle.load(f)
    return df

def f2c_symmetric_diff(case, sigma_threshold=3.0):
    """
    Compare:
      A = set of (source, target) edges with predict==1 in post-inject window (Stage 1)
      B = set of serviceName_methodName operations flagged by raw mu+sigma*std rule
          (raw per-span duration, pre-injection baseline)
    Return summarized counts and projection of B to (some_caller, callee) edge-space.
    """
    name, _, _, gt = case
    inject_us = read_inject_us(case)

    invo = load_invo_result(name)
    # Reset index — invo has source/target as both index AND column
    invo = invo.reset_index(drop=True)
    # Filter to anomalous (post-inject) only
    if "trace_label" in invo.columns:
        post = invo[invo["trace_label"] == 1]
    else:
        post = invo
    edge_predict = post.groupby(["source", "target"])["predict"].max()
    A_edges_flagged = set([k for k, v in edge_predict.items() if v == 1])
    A_edges_total = set(edge_predict.index)

    # Now raw 3σ per (serviceName_methodName) operation
    df = load_raw_csv(case)
    df["op"] = df["serviceName"].fillna("") + "_" + df["methodName"].fillna("")
    pre = df[df["startTime"] < inject_us]
    pos = df[df["startTime"] >= inject_us]
    stats = pre.groupby("op")["duration"].agg(["mean", "std"]).fillna(0.0)
    # Per-op flag counts in post window
    per_op = []
    for op, grp in pos.groupby("op"):
        if op not in stats.index:
            continue
        mu = stats.loc[op, "mean"]
        sd = stats.loc[op, "std"]
        if sd <= 0:
            continue
        flagged = int(((grp["duration"] - mu) > sigma_threshold * sd).sum())
        total = len(grp)
        if flagged > 0:
            per_op.append({"op": op, "flagged": flagged, "total": total,
                           "rate": flagged / total,
                           "service": op.split("_")[0],
                           "service_simple": simple_name(op.split("_")[0]) if op.split("_")[0] else ""})
    # Build B as set of services with ANY operation flagged
    services_flagged_raw = set()
    for r in per_op:
        if r["service_simple"]:
            services_flagged_raw.add(r["service_simple"])

    # B at edge level: take all (parent_service_simple, child_service_simple) edges
    # where the child has a flagged operation. For comparability with A, lift per-op flags
    # to caller→child edges using parent_span_id mapping.
    span_lookup = {row.spanID: row for row in df.itertuples(index=False)}
    flagged_op_set = {r["op"] for r in per_op}
    B_edges_flagged = set()
    # iterate post-inject spans; for each span whose op is flagged, find caller
    pos_idx = pos[pos["op"].isin(flagged_op_set)]
    for row in pos_idx.itertuples(index=False):
        parent_id = row.parentSpanID
        if not parent_id or parent_id not in span_lookup:
            continue
        parent = span_lookup[parent_id]
        src = simple_name(parent.serviceName) if pd.notna(parent.serviceName) else ""
        tgt = simple_name(row.serviceName) if pd.notna(row.serviceName) else ""
        if src and tgt and src != tgt and src in INVOLVED_SET and tgt in INVOLVED_SET:
            B_edges_flagged.add((src, tgt))

    # Sets
    only_A = A_edges_flagged - B_edges_flagged
    only_B = B_edges_flagged - A_edges_flagged
    both = A_edges_flagged & B_edges_flagged

    # GT-touching subsets
    gt_A = {e for e in A_edges_flagged if gt in e}
    gt_B = {e for e in B_edges_flagged if gt in e}

    return {
        "case": name,
        "fault_type": case[1],
        "gt": gt,
        "sigma_threshold": sigma_threshold,
        "A_edges_total_observed": len(A_edges_total),
        "A_edges_flagged_count": len(A_edges_flagged),
        "B_ops_flagged_count": len(per_op),
        "B_edges_flagged_count": len(B_edges_flagged),
        "only_A_count": len(only_A),
        "only_B_count": len(only_B),
        "both_count": len(both),
        "gt_in_A": len(gt_A),
        "gt_in_B": len(gt_B),
        "gt_A_edges": sorted(list(gt_A)),
        "gt_B_edges": sorted(list(gt_B)),
        "services_flagged_raw_count": len(services_flagged_raw),
        "gt_in_services_flagged_raw": gt in services_flagged_raw,
        "top_flagged_ops": sorted(per_op, key=lambda r: -r["flagged"])[:10],
    }

def main():
    f1_results = []
    f2a_results = []
    f2c_results = []

    for case in CASES:
        print(f"=== {case[0]} ===")
        try:
            r1 = f1_drop(case)
            f1_results.append(r1)
            print(f"  F1 raw_post={r1['raw_post_traces']} pkl={r1['pkl_test_traces']} drop={r1['drop_pct']:.1f}%")
        except Exception as e:
            print(f"  F1 ERROR: {e}")
            f1_results.append({"case": case[0], "error": str(e)})
        try:
            r2 = f2a_audit(case)
            f2a_results.append(r2)
            print(f"  F2a uf_keys={r2['useful_features_total_keys']} gt_edges_with_features={len(r2['gt_edges_with_features'])} gt_edges_no_features={len(r2['gt_edges_no_features'])}")
        except Exception as e:
            print(f"  F2a ERROR: {e}")
            f2a_results.append({"case": case[0], "error": str(e)})
        try:
            r3 = f2c_symmetric_diff(case, sigma_threshold=3.0)
            f2c_results.append(r3)
            print(f"  F2c A_flagged={r3['A_edges_flagged_count']} B_edges_flagged={r3['B_edges_flagged_count']} only_A={r3['only_A_count']} only_B={r3['only_B_count']} both={r3['both_count']} gt_in_A={r3['gt_in_A']} gt_in_B={r3['gt_in_B']}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  F2c ERROR: {e}")
            f2c_results.append({"case": case[0], "error": str(e)})

    with open(OUT_DIR / "f1_results.json", "w") as f:
        json.dump(f1_results, f, indent=2, default=str)
    with open(OUT_DIR / "f2a_results.json", "w") as f:
        json.dump(f2a_results, f, indent=2, default=str)
    with open(OUT_DIR / "f2c_results.json", "w") as f:
        json.dump(f2c_results, f, indent=2, default=str)

    print("\nDONE. Outputs in", OUT_DIR)

if __name__ == "__main__":
    main()
