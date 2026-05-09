"""
Phase 4 orchestrator: Stage 2/3 + HR@k collect for all 7 ablation cells.

For each cell × case, runs run_localization_association_rule_mining_20210516.py
(α=0, support=0.05, k=100) on the Stage 1 output produced by Phase 3, then
runs run_localization_collect.py per cell to produce eval.csv.

Output layout (under experiments/ablation/<cell>/):
    <case>/association_rule_mining.result.pkl.0.05.100   (per case)
    eval.csv                                              (cell-level HR@1, HR@3, MAR, MFR)
    eval_summary.csv                                      (whole-cell summary)

Usage:
    python3 experiments/scripts/run_localization_eval.py \
        [--cells baseline_0000 f2a_only ...] \
        [--alpha 0.0] [--support 0.05] [--k 100] \
        [--ablation-dir experiments/ablation]
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ABLATION_DIR = REPO_ROOT / 'experiments' / 'ablation'
LOCALIZATION_RUNNER = REPO_ROOT / 'run_localization_association_rule_mining_20210516.py'
COLLECT_RUNNER = REPO_ROOT / 'run_localization_collect.py'

ALL_CELLS = [
    'baseline_0000', 'f2a_only', 'f2b_only', 'perop_only',
    'f1b_only', 'f1b_perop', 'all_on',
]


def run(cmd, log=None):
    if log is not None:
        with open(log, 'a') as logf:
            logf.write(f"\n$ {' '.join(shlex.quote(c) for c in cmd)}\n")
            logf.flush()
            return subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return subprocess.call(cmd)


def stage23_for_cell(cell_dir: Path, alpha: float, support: float, k: int,
                     log_dir: Path) -> int:
    """Run Stage 2/3 for every case in a cell. Returns count of successful runs."""
    case_dirs = sorted(p for p in cell_dir.iterdir()
                       if p.is_dir() and p.name != 'logs'
                       and (p / 'invo.result.pkl').exists())
    n_ok = 0
    n_fail = 0
    for case_dir in case_dirs:
        case_id = case_dir.name
        invo_in = case_dir / 'invo.result.pkl'
        s23_out = case_dir / f'{case_id}.association_rule_mining.result.pkl.{support}.{k}'

        if s23_out.exists() and s23_out.stat().st_size > 0:
            n_ok += 1
            continue

        cmd = [
            sys.executable, str(LOCALIZATION_RUNNER),
            '-i', str(invo_in),
            '-o', str(s23_out),
            '--min-support-rate', str(support),
            '--caller-discount-alpha', str(alpha),
            '-q',
        ]
        rc = run(cmd, log=log_dir / f'{case_id}.s23.log')
        if rc != 0:
            n_fail += 1
            print(f'    [FAIL] {case_id}: rc={rc}', file=sys.stderr)
        else:
            n_ok += 1
    return n_ok, n_fail


def collect_for_cell(cell_dir: Path, support: float, k: int,
                     log_path: Path) -> Path | None:
    """Run run_localization_collect.py for a cell. Returns eval.csv path or None."""
    s23_files = sorted(cell_dir.rglob(
        f'*.association_rule_mining.result.pkl.{support}.{k}'))
    if not s23_files:
        print(f'  no Stage 2/3 outputs in {cell_dir}', file=sys.stderr)
        return None

    eval_csv = cell_dir / 'eval.csv'
    cmd = [sys.executable, str(COLLECT_RUNNER)]
    for f in s23_files:
        cmd += ['-i', str(f)]
    cmd += ['-o', str(eval_csv)]
    rc = run(cmd, log=log_path)
    return eval_csv if rc == 0 else None


def summarize_cell(eval_csv: Path) -> dict:
    """Parse eval.csv and compute mean HR@k and MAR/MFR for the 'Ours' method."""
    import csv
    from collections import defaultdict
    rows_by_method_metric = defaultdict(list)
    with open(eval_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            method = r.get('method', '').strip()
            metric = r.get('metric_name', '').strip()
            try:
                value = float(r.get('metric_value', '0'))
            except ValueError:
                continue
            rows_by_method_metric[(method, metric)].append(value)

    # Pick 'Ours-noise=0' if present, else the first 'Ours*' method
    methods = {m for m, _ in rows_by_method_metric.keys()}
    target = None
    for cand in ('Ours-noise=0', 'Ours'):
        if cand in methods:
            target = cand
            break
    if target is None:
        target = next((m for m in methods if m.startswith('Ours')), None)
    if target is None:
        return {}

    summary = {'method': target}
    metric_aliases = {
        'A@1': ('Top-1 Accuracy', 'A@1'),
        'A@2': ('Top-2 Accuracy', 'A@2'),
        'A@3': ('Top-3 Accuracy', 'A@3'),
        'MAR': ('MAR',),
        'MFR': ('MFR',),
    }
    for out_metric, candidates in metric_aliases.items():
        for cand in candidates:
            vals = rows_by_method_metric.get((target, cand), [])
            if vals:
                summary[out_metric] = sum(vals) / len(vals)
                summary[f'{out_metric}_n'] = len(vals)
                break
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', nargs='+', default=ALL_CELLS)
    ap.add_argument('--ablation-dir', default=str(DEFAULT_ABLATION_DIR))
    ap.add_argument('--alpha', type=float, default=0.0)
    ap.add_argument('--support', type=float, default=0.05)
    ap.add_argument('--k', type=int, default=100)
    args = ap.parse_args()

    ablation_root = Path(args.ablation_dir)
    if not ablation_root.exists():
        print(f'No ablation root at {ablation_root}', file=sys.stderr)
        sys.exit(2)

    t0 = time.time()
    summary_records = []

    for cell_name in args.cells:
        cell_dir = ablation_root / cell_name
        if not cell_dir.exists():
            print(f'[skip] no dir {cell_dir}', file=sys.stderr)
            continue
        log_dir = cell_dir / 'logs'
        log_dir.mkdir(exist_ok=True)

        print(f'\n[{cell_name}] Stage 2/3 (t+{time.time()-t0:.0f}s)')
        n_ok, n_fail = stage23_for_cell(cell_dir, args.alpha, args.support, args.k, log_dir)
        print(f'  S23 ok={n_ok} fail={n_fail}')

        print(f'[{cell_name}] collect (t+{time.time()-t0:.0f}s)')
        eval_csv = collect_for_cell(cell_dir, args.support, args.k,
                                    log_dir / 'collect.log')
        if eval_csv is None:
            print(f'  collect failed', file=sys.stderr)
            continue

        s = summarize_cell(eval_csv)
        s['cell'] = cell_name
        s['n_cases'] = n_ok
        summary_records.append(s)
        print(f"  HR@1={s.get('A@1', float('nan')):.3f} "
              f"HR@3={s.get('A@3', float('nan')):.3f} "
              f"MAR={s.get('MAR', float('nan')):.3f} "
              f"MFR={s.get('MFR', float('nan')):.3f}")

    # Write the cross-cell summary CSV
    summary_csv = ablation_root / 'eval_summary.csv'
    if summary_records:
        import csv
        cols = ['cell', 'method', 'n_cases', 'A@1', 'A@2', 'A@3', 'MAR', 'MFR']
        with open(summary_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            w.writeheader()
            for r in summary_records:
                w.writerow(r)
        print(f'\nSummary: {summary_csv}')
        # Print a markdown-ish table to stdout
        print(f'\n{"cell":<16} {"HR@1":>7} {"HR@3":>7} {"MAR":>7} {"MFR":>7} n')
        for r in summary_records:
            print(f"{r['cell']:<16} "
                  f"{r.get('A@1', float('nan')):>7.3f} "
                  f"{r.get('A@3', float('nan')):>7.3f} "
                  f"{r.get('MAR', float('nan')):>7.3f} "
                  f"{r.get('MFR', float('nan')):>7.3f} "
                  f"{r.get('n_cases', 0)}")

    print(f'\nPhase 4 complete in {int(time.time() - t0)}s')


if __name__ == '__main__':
    main()
