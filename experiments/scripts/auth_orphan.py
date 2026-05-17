"""Probe auth's parent topology more carefully."""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, "/home/jaredvel25/TraceRCA-CD")
from data.trainticket.download import simple_name
from trainticket_config import INVOLVED_SERVICES

INVOLVED_SET = set(INVOLVED_SERVICES)
FORK_ROOT = Path("/home/jaredvel25/RCAEval-Fork")

for case in ["ts-auth-service_cpu/1", "ts-route-service_delay/1", "ts-train-service_mem/1"]:
    base, rep = case.split("/")
    df = pd.read_csv(FORK_ROOT / "data" / "re2tt" / base / rep / "traces.csv",
                     usecols=["traceID","spanID","serviceName","parentSpanID"], dtype=str)
    sid2svc = dict(zip(df["spanID"].fillna(""), df["serviceName"].fillna("")))
    print(f"\n=== {case} ===")
    print(f"  total spans: {len(df)}")
    print(f"  unique services: {df['serviceName'].nunique()}")

    # For each service, classify its span population by parent location
    for svc in [s for s in df["serviceName"].dropna().unique() if "auth" in s or "route" in s or "train" in s]:
        sub = df[df["serviceName"] == svc]
        n = len(sub)
        n_orphan = int(sub["parentSpanID"].fillna("").eq("").sum())
        # parent_present_in_dump
        parent_in_dump = sub["parentSpanID"].fillna("").map(lambda p: p != "" and p in sid2svc).sum()
        n_orphan_via_missing_parent = n - n_orphan - parent_in_dump
        # cross-service parents
        from collections import Counter
        parent_svcs = Counter()
        for p in sub["parentSpanID"].fillna(""):
            if p and p in sid2svc:
                parent_svcs[sid2svc[p]] += 1
        same_svc = parent_svcs.get(svc, 0)
        cross_svc_total = sum(v for k, v in parent_svcs.items() if k != svc)
        print(f"  {svc:30s} n={n:>5d}  no_parent={n_orphan:>5d}  parent_missing_from_dump={n_orphan_via_missing_parent:>5d}  same_svc_parent={same_svc:>5d}  cross_svc_parent={cross_svc_total:>5d}")
        for k, v in parent_svcs.most_common(5):
            print(f"      parent={k:30s} simple={simple_name(k):20s} in_involved={simple_name(k) in INVOLVED_SET!s:>5}  n={v}")
