"""
F3: Counterfactual rank using Fork's F1 formula on raw 3σ flags.
For each case:
  - baseline rank: read CD's existing rank pkl
  - counterfactual rank: replicate Fork's tracerca() on the same case data, then dedup-to-service
"""
import sys, json, pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "/home/jaredvel25/TraceRCA-CD")
from data.trainticket.download import simple_name

CD_ROOT = Path("/home/jaredvel25/TraceRCA-CD")
FORK_ROOT = Path("/home/jaredvel25/RCAEval-Fork")
OUT = CD_ROOT / "experiments" / "data"

CASES = [
    ("ts-auth-service_cpu_1", "cpu", "auth"),
    ("ts-auth-service_disk_1", "disk", "auth"),
    ("ts-auth-service_socket_1", "socket", "auth"),
    ("ts-order-service_cpu_1", "cpu", "order"),
    ("ts-order-service_disk_1", "disk", "order"),
    ("ts-order-service_socket_1", "socket", "order"),
    ("ts-route-service_cpu_1", "cpu", "route"),
    ("ts-route-service_disk_1", "disk", "route"),
    ("ts-route-service_socket_1", "socket", "route"),
    ("ts-route-service_delay_1", "delay", "route"),
    ("ts-train-service_delay_1", "delay", "train"),
    ("ts-train-service_mem_1", "mem", "train"),
]

def fork_case_dir(case):
    name, _, _ = case
    base = "_".join(name.split("_")[:-1])
    rep = name.split("_")[-1]
    return FORK_ROOT / "data" / "re2tt" / base / rep

def cd_baseline_rank(case_name, gt):
    p = CD_ROOT / "output" / "trainticket_root_cause_localization" / f"{case_name}.association_rule_mining.result.pkl.0.05.100"
    if not p.exists():
        return None, None
    with open(p, "rb") as f:
        r = pickle.load(f)
    ranks = r.get("Ours-noise=0", [])
    if gt in ranks:
        return ranks.index(gt) + 1, len(ranks)
    return None, len(ranks)

def fork_counterfactual_rank(case, gt):
    name, _, _ = case
    cdir = fork_case_dir(case)
    df = pd.read_csv(cdir / "traces.csv")
    inject_time = int((cdir / "inject_time.txt").read_text().strip()) * 1_000_000

    df["methodName"] = df["methodName"].fillna(df["operationName"])
    df["operation"] = df["serviceName"].astype(str) + "_" + df["methodName"].astype(str)

    normal_df = df[df["startTime"] + df["duration"] < inject_time]
    anomal_df = df[df["startTime"] + df["duration"] >= inject_time].copy()

    # 3σ on normal stats
    op_stats = normal_df.groupby("operation")["duration"].agg(["mean", "std"]).fillna(0.0)
    anomal_df["mean"] = anomal_df["operation"].map(op_stats["mean"]).astype(float)
    anomal_df["std"] = anomal_df["operation"].map(op_stats["std"]).astype(float)
    anomal_df["abnormal"] = (anomal_df["duration"] / 1_000) >= (anomal_df["mean"] / 1_000 + 3 * anomal_df["std"] / 1_000)
    # Note: divisions by 1000 cancel, but kept to mirror Fork

    operations = list(anomal_df["operation"].unique())
    abnormal_total = anomal_df["abnormal"].sum()
    if abnormal_total == 0:
        return None, 0
    ji = {}
    for op in operations:
        sub = anomal_df[anomal_df["operation"] == op]
        ab = sub["abnormal"].sum()
        s = ab / abnormal_total
        c = ab / max(len(sub), 1)
        ji[op] = (2 * s * c / (s + c)) if (s + c) > 0 else 0.0
    sorted_ops = sorted(ji.items(), key=lambda x: x[1], reverse=True)
    op_ranks = [op for op, v in sorted_ops if not np.isnan(v)]
    # Dedup to service first occurrence (using simple_name to match CD)
    seen, svc_ranks = set(), []
    for op in op_ranks:
        full_svc = op.split("_", 1)[0]
        svc = simple_name(full_svc) if full_svc else ""
        if svc and svc not in seen:
            seen.add(svc)
            svc_ranks.append(svc)
    if gt in svc_ranks:
        return svc_ranks.index(gt) + 1, len(svc_ranks)
    return None, len(svc_ranks)

def main():
    rows = []
    for case in CASES:
        name, ftype, gt = case
        bl_rank, bl_len = cd_baseline_rank(name, gt)
        cf_rank, cf_len = fork_counterfactual_rank(case, gt)
        delta = None
        if bl_rank is not None and cf_rank is not None:
            delta = bl_rank - cf_rank  # positive means CF improved
        rows.append({"case": name, "fault_type": ftype, "gt": gt,
                     "cd_baseline_rank": bl_rank, "cd_baseline_len": bl_len,
                     "counterfactual_rank": cf_rank, "counterfactual_len": cf_len,
                     "delta_rank": delta})
    with open(OUT / "f3_results.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    # Pretty print
    print(f"{'case':35s} {'ftype':6s} {'CD':>8s} {'CF':>8s} {'Δ':>5s}")
    for r in rows:
        cd = "EMPTY" if r["cd_baseline_rank"] is None else str(r["cd_baseline_rank"])
        cf = "MISS" if r["counterfactual_rank"] is None else str(r["counterfactual_rank"])
        d = "" if r["delta_rank"] is None else f"{r['delta_rank']:+d}"
        print(f"{r['case']:35s} {r['fault_type']:6s} {cd:>8s} {cf:>8s} {d:>5s}")
    # Aggregate by fault type
    print("\nAggregate by fault type (mean Δrank, only cases where both ranks are non-None):")
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        if r["delta_rank"] is not None:
            agg[r["fault_type"]].append(r["delta_rank"])
    for ftype, ds in sorted(agg.items()):
        print(f"  {ftype:8s} n={len(ds)}  mean Δ={np.mean(ds):+.2f}  values={ds}")

if __name__ == "__main__":
    main()
