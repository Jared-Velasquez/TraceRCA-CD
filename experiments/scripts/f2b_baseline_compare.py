"""
F2b: Compare cached (historical) μ/σ against pre-injection-only μ/σ on the CURRENT case.
For each case, focus on GT-touching edges that are present in the useful_features whitelist.
"""
import sys, json, pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "/home/jaredvel25/TraceRCA-CD")
from data.trainticket.download import simple_name
from trainticket_config import INVOLVED_SERVICES

CD_ROOT = Path("/home/jaredvel25/TraceRCA-CD")
OUT = CD_ROOT / "experiments" / "data"
INVOLVED_SET = set(INVOLVED_SERVICES)
THRESHOLD = 1.0  # CD's actual threshold

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
    p = CD_ROOT / "output" / "trainticket_anomaly_detection.models.0.0"
    with open(p, "rb") as f:
        return SafeUnpickler(f).load()

def load_invo(case_name):
    p = CD_ROOT / "output" / "trainticket_anomaly_detection.test" / f"{case_name}.invo.result.pkl.1.1"
    with open(p, "rb") as f:
        df = pickle.load(f)
    return df.reset_index(drop=True)

def load_uf(case_name):
    p = CD_ROOT / "output" / "trainticket_anomaly_detection.test" / f"{case_name}.useful_features.1"
    return eval(p.read_text(), {"np": type("NS", (), {"str_": str})(), "np_str_": str})

def main():
    cache = load_cache()
    results = []
    for name, ftype, gt in CASES:
        invo = load_invo(name)
        uf_raw = load_uf(name)
        uf = {(str(k[0]), str(k[1])): v for k, v in uf_raw.items()}
        gt_edges = [k for k in uf if (k[0] == gt or k[1] == gt) and "latency" in uf[k]]

        # Compute pre/post-injection μ/σ on latency for these edges from invo
        # invo trace_label==0 is pre-window normal, ==1 is post-injection abnormal
        per_edge_stats = []
        for src, tgt in gt_edges:
            sub = invo[(invo["source"] == src) & (invo["target"] == tgt)]
            pre = sub[sub["trace_label"] == 0]["latency"].values
            post = sub[sub["trace_label"] == 1]["latency"].values
            cache_token = f"reference-{src}-{tgt}-latency-mean-variance"
            cache_entry = cache.get(cache_token, None)
            cmean = cache_entry["mean"] if cache_entry else None
            cstd = cache_entry["std"] if cache_entry else None
            pmean = float(pre.mean()) if len(pre) > 0 else None
            pstd = max(float(pre.std()), 0.1) if len(pre) > 1 else None
            postmean = float(post.mean()) if len(post) > 0 else None
            postmedian = float(np.median(post)) if len(post) > 0 else None

            # Flag rate using cached stats vs using pre-only stats
            def flag_rate(values, mu, sd):
                if mu is None or sd is None or len(values) == 0:
                    return None
                return float(np.mean(np.abs(values - mu) > THRESHOLD * sd))

            cflag = flag_rate(post, cmean, cstd)
            pflag = flag_rate(post, pmean, pstd)
            per_edge_stats.append({
                "edge": f"{src}->{tgt}",
                "cache_mean": cmean, "cache_std": cstd,
                "pre_mean": pmean, "pre_std": pstd,
                "post_mean": postmean, "post_median": postmedian,
                "n_pre": int(len(pre)), "n_post": int(len(post)),
                "flag_rate_cache_baseline": cflag,
                "flag_rate_preonly_baseline": pflag,
            })
        results.append({"case": name, "fault_type": ftype, "gt": gt, "gt_edges": per_edge_stats})

    with open(OUT / "f2b_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Pretty print
    for r in results:
        print(f"\n=== {r['case']} (gt={r['gt']}) ===")
        if not r["gt_edges"]:
            print("  No GT-touching edges in useful_features (Stage 1 cannot flag GT)")
            continue
        for e in r["gt_edges"]:
            def fmt(x, w=8):
                return ("%.0f" % x).rjust(w) if x is not None else "    None"
            def fmt3(x):
                return ("%.3f" % x) if x is not None else "  N/A"
            print(f"  {e['edge']:35s} cache(mu,std)=({fmt(e['cache_mean'])},{fmt(e['cache_std'])})  pre(mu,std)=({fmt(e['pre_mean'])},{fmt(e['pre_std'])})  post(med,mean)=({fmt(e['post_median'])},{fmt(e['post_mean'])})  flag_c={fmt3(e['flag_rate_cache_baseline'])} flag_p={fmt3(e['flag_rate_preonly_baseline'])} n_pre={e['n_pre']} n_post={e['n_post']}")

if __name__ == "__main__":
    main()
