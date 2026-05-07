"""
Other factors:
  - threshold sensitivity (replay Stage 1 with σ ∈ {0.5, 1, 2, 3} on cached stats over invo result)
  - simple_name collision check
  - forbidden_names probe (would gateway have been in top-K?)
  - predict propagation: of GT-containing test traces, what fraction has predict==1 on any invocation
  - auth caller analysis (why useful_features is empty for auth)
"""
import sys, json, pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "/home/jaredvel25/TraceRCA-CD")
from data.trainticket.download import simple_name
from trainticket_config import INVOLVED_SERVICES, FEATURE_NAMES

CD_ROOT = Path("/home/jaredvel25/TraceRCA-CD")
FORK_ROOT = Path("/home/jaredvel25/RCAEval-Fork")
OUT = CD_ROOT / "experiments" / "data"
INVOLVED_SET = set(INVOLVED_SERVICES)

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

class StubInstance:
    def __init__(self, *a, **kw): pass
    def __setstate__(self, state): self.state = state
    def __reduce__(self): return (StubInstance, ())

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("sklearn") or "random" in module or module == "imblearn":
            return StubInstance
        return super().find_class(module, name)

def load_cache():
    with open(CD_ROOT / "output" / "trainticket_anomaly_detection.models.0.0", "rb") as f:
        return SafeUnpickler(f).load()

def load_invo(name):
    with open(CD_ROOT / "output" / "trainticket_anomaly_detection.test" / f"{name}.invo.result.pkl.1.1", "rb") as f:
        return pickle.load(f).reset_index(drop=True)

def load_uf(name):
    txt = (CD_ROOT / "output" / "trainticket_anomaly_detection.test" / f"{name}.useful_features.1").read_text()
    return eval(txt, {"np": type("NS", (), {"str_": str})(), "np_str_": str})

def threshold_sweep():
    """For each case, replay Stage 1 logic at σ ∈ {0.5,1,2,3} using cache stats and useful_features whitelist."""
    cache = load_cache()
    rows = []
    for name, ftype, gt in CASES:
        invo = load_invo(name)
        post = invo[invo["trace_label"] == 1]
        uf_raw = load_uf(name)
        uf = {(str(k[0]), str(k[1])): v for k, v in uf_raw.items()}
        for sigma in (0.5, 1.0, 2.0, 3.0):
            n_edges_flagged = 0
            n_invos_flagged = 0
            gt_flagged_count = 0
            for (src, tgt), feats in uf.items():
                if not feats:
                    continue
                edge_post = post[(post["source"] == src) & (post["target"] == tgt)]
                if len(edge_post) == 0:
                    continue
                # Use cached mean+std
                flags = np.zeros(len(edge_post), dtype=bool)
                for feat in feats:
                    token = f"reference-{src}-{tgt}-{feat}-mean-variance"
                    if token in cache:
                        mu = cache[token]["mean"]
                        sd = cache[token]["std"]
                    else:
                        continue
                    vals = edge_post[feat].values
                    flag = np.abs(vals - mu) > sigma * sd
                    flags = flags | flag
                if flags.any():
                    n_edges_flagged += 1
                    n_invos_flagged += int(flags.sum())
                    if src == gt or tgt == gt:
                        gt_flagged_count += 1
            rows.append({"case": name, "fault_type": ftype, "gt": gt,
                         "sigma": sigma, "uf_keys": len(uf),
                         "edges_flagged": n_edges_flagged, "invos_flagged": n_invos_flagged,
                         "gt_edges_flagged": gt_flagged_count})
    return rows

def gateway_probe():
    """Would 'gateway' have been ranked by Stage 2/3 if it were not in forbidden_names?"""
    rows = []
    for name, ftype, gt in CASES:
        invo = load_invo(name)
        post = invo[invo["trace_label"] == 1]
        # Check if gateway is involved in any post-injection invocation, and how often predict==1 on those
        gw_invos = post[(post["source"] == "gateway") | (post["target"] == "gateway")]
        gw_total = len(gw_invos)
        gw_flagged = int((gw_invos["predict"] == 1).sum())
        rows.append({"case": name, "fault_type": ftype, "gt": gt,
                     "gateway_invos_total": gw_total, "gateway_invos_flagged": gw_flagged})
    return rows

def predict_propagation():
    """Of GT-containing post-inject traces in invo, what fraction has predict==1 on ANY invocation?"""
    rows = []
    for name, ftype, gt in CASES:
        invo = load_invo(name)
        post = invo[invo["trace_label"] == 1]
        gt_traces = set(post[(post["source"] == gt) | (post["target"] == gt)]["trace_id"].unique())
        all_traces = set(post["trace_id"].unique())
        # for each trace, predict.max
        per_trace = post.groupby("trace_id")["predict"].max()
        traces_abnormal = set(per_trace[per_trace == 1].index)
        gt_abn = gt_traces & traces_abnormal
        gt_total_count = len(gt_traces)
        rows.append({
            "case": name, "fault_type": ftype, "gt": gt,
            "post_traces_total": len(all_traces),
            "post_traces_abnormal": len(traces_abnormal),
            "gt_traces": gt_total_count,
            "gt_traces_abnormal": len(gt_abn),
            "gt_traces_abnormal_via_non_gt_edge": len(gt_abn) - sum(
                1 for tid in gt_abn
                if ((post["trace_id"] == tid) & ((post["source"] == gt) | (post["target"] == gt)) & (post["predict"] == 1)).any()
            ),
        })
    return rows

def auth_caller_analysis():
    """For auth cases: identify which raw services (full+simple names) call ts-auth-service."""
    rows = []
    for name, ftype, gt in CASES:
        if gt != "auth":
            continue
        base = "_".join(name.split("_")[:-1])
        rep = name.split("_")[-1]
        cdir = FORK_ROOT / "data" / "re2tt" / base / rep
        df = pd.read_csv(cdir / "traces.csv",
                         usecols=["traceID","spanID","serviceName","parentSpanID"],
                         dtype=str)
        sid2svc = dict(zip(df["spanID"], df["serviceName"]))
        auth_spans = df[df["serviceName"] == "ts-auth-service"]
        callers = []
        for psid in auth_spans["parentSpanID"].dropna():
            psvc = sid2svc.get(psid)
            if psvc:
                callers.append((psvc, simple_name(psvc), simple_name(psvc) in INVOLVED_SET))
        from collections import Counter
        ctr = Counter(callers)
        rows.append({"case": name, "auth_spans": len(auth_spans), "caller_counts": [
            {"caller_full": k[0], "caller_simple": k[1], "in_involved": k[2], "count": v}
            for k, v in ctr.most_common(10)
        ]})
    return rows

def main():
    out = {}
    out["threshold_sweep"] = threshold_sweep()
    out["gateway_probe"] = gateway_probe()
    out["predict_propagation"] = predict_propagation()
    out["auth_caller_analysis"] = auth_caller_analysis()
    with open(OUT / "other_factors.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    # Pretty print
    print("\n=== Threshold sweep (edges flagged & gt_edges flagged per case per σ) ===")
    print(f"{'case':30s} {'σ':>5s} {'uf':>4s} {'edges':>6s} {'gt':>4s} {'invos':>7s}")
    for r in out["threshold_sweep"]:
        print(f"{r['case']:30s} {r['sigma']:>5.1f} {r['uf_keys']:>4d} {r['edges_flagged']:>6d} {r['gt_edges_flagged']:>4d} {r['invos_flagged']:>7d}")
    print("\n=== Gateway probe ===")
    for r in out["gateway_probe"]:
        print(f"{r['case']:30s} gw_invos={r['gateway_invos_total']:>5d} gw_flagged={r['gateway_invos_flagged']:>5d}")
    print("\n=== Predict propagation ===")
    for r in out["predict_propagation"]:
        print(f"{r['case']:30s} traces={r['post_traces_total']:>5d} abnormal={r['post_traces_abnormal']:>5d} gt_traces={r['gt_traces']:>5d} gt_traces_abnormal={r['gt_traces_abnormal']:>5d} via_non_gt_edge={r['gt_traces_abnormal_via_non_gt_edge']:>5d}")
    print("\n=== Auth caller analysis ===")
    for r in out["auth_caller_analysis"]:
        print(f"\n{r['case']} (auth_spans={r['auth_spans']}):")
        for cc in r["caller_counts"]:
            print(f"  caller={cc['caller_full']:30s} simple={cc['caller_simple']:20s} in_involved={cc['in_involved']:>5}  n={cc['count']}")

if __name__ == "__main__":
    main()
