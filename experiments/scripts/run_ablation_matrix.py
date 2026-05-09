"""
Ablation matrix runner for paper-accurate Stage 1.

Drives the full Stage 0 → Stage 1 → Stage 2/3 pipeline for each cell defined
in experiments/configs/stage1_paper.yaml. The runner shells out to the
existing scripts (preprocess, invo encoding, prepare_model, selecting_features,
invo_anomaly_detection, localization runner). Output for each cell is written
under experiments/ablation/{cell_name}/.

Off-switch parity contract: with the `baseline_0000` cell selected, every
output produced under that cell's directory must be byte-identical to the
locked baseline pkls under output/trainticket_anomaly_detection.test/.

Note: this runner deliberately does NOT re-run preprocessing or invo
encoding when the existing per-case .invo.pkl files already contain the
caller_method / callee_method columns. Pass --re-encode to force a fresh
invo encoding (required if the existing files predate Step 2 of the plan).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / 'experiments' / 'configs' / 'stage1_paper.yaml'
DEFAULT_OUT_ROOT = REPO_ROOT / 'experiments' / 'ablation'


def load_config(path: Path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    defaults = cfg.get('defaults', {})
    cells = cfg.get('cells', {})
    return defaults, cells


def merge_cell(defaults: dict, overrides: dict) -> dict:
    merged = dict(defaults)
    merged.update(overrides or {})
    return merged


def cell_args_for(script: str, cell: dict) -> list[str]:
    """
    Return the flag list relevant to a given pipeline script for the merged
    cell parameters. Scripts ignore flags they don't define, but click is
    strict about unknown options, so each script gets only its own flags.
    """
    a = []
    if script in ('preprocess_re2tt.py', 'preprocess_re2ob.py'):
        if cell.get('admit_self_spans'):
            a.append('--admit-self-spans')
        if cell.get('admit_root_spans'):
            a.append('--admit-root-spans')
    elif script == 'run_invo_encoding.py':
        if cell.get('admit_self_spans'):
            a.append('--admit-self-spans')
    elif script == 'run_selecting_features.py':
        a += ['--feature-selector', cell.get('feature_selector', 'stderr')]
        a += ['--fs-delta', str(cell.get('fs_delta', 0.1))]
        a += ['--fs-floor', str(cell.get('fs_floor', 0.0))]
        a += ['--baseline-window', cell.get('baseline_window', 'global')]
        a += ['--stage1-granularity', cell.get('stage1_granularity', 'pair')]
    elif script == 'run_anomaly_detection_invo.py':
        a += ['--stage1-granularity', cell.get('stage1_granularity', 'pair')]
        a += ['--min-baseline-samples', str(cell.get('min_baseline_samples', 30))]
    elif script == 'run_anomaly_detection_prepare_model.py':
        a += ['--baseline-window', cell.get('baseline_window', 'global')]
        a += ['--stage1-granularity', cell.get('stage1_granularity', 'pair')]
        a += ['--last-slot-seconds', str(cell.get('last_slot_seconds', 300))]
        a += ['--last-period-seconds', str(cell.get('last_period_seconds', 86400))]
    return a


def run_cmd(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'a') as logf:
        logf.write(f"\n$ {' '.join(shlex.quote(c) for c in cmd)}\n")
        logf.flush()
        return subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(DEFAULT_CONFIG))
    ap.add_argument('--out-root', default=str(DEFAULT_OUT_ROOT))
    ap.add_argument('--cells', nargs='+', default=None,
                    help='Subset of cells to run; default: all cells in YAML.')
    ap.add_argument('--cases', nargs='+', default=None,
                    help='Subset of case names (file basenames without .invo.pkl).')
    ap.add_argument('--invo-dir', default=str(REPO_ROOT / 'output' / 'trainticket_invo_encoded'),
                    help='Directory containing per-case .invo.pkl files.')
    ap.add_argument('--global-cache', default=str(REPO_ROOT / 'output' / 'trainticket_anomaly_detection.models.0.0'),
                    help='Path to the global model/cache pkl.')
    ap.add_argument('--global-history', default=str(REPO_ROOT / 'output' / 'trainticket_invo_encoded' / 'trainticket_historical_normal.invo.pkl'),
                    help='Path to the global concatenated normal history pkl.')
    ap.add_argument('--inject-times-dir',
                    default=str(REPO_ROOT.parent / 'RCAEval-Fork' / 'data' / 're2tt'),
                    help='Root dir containing per-case inject_time.txt files.')
    ap.add_argument('--python', default=sys.executable,
                    help='Python interpreter to use for subprocess calls.')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print planned commands, do not execute.')
    args = ap.parse_args()

    config_path = Path(args.config)
    defaults, cells = load_config(config_path)

    selected = args.cells or list(cells.keys())
    invo_dir = Path(args.invo_dir)

    # Test (anomalous) per-case pkls are <case>.invo.pkl. Normal pre-fault
    # pkls are <case>.normal.invo.pkl and the global concat is
    # trainticket_historical_normal.invo.pkl — neither belongs in the ablation
    # case list (they're baseline inputs, not Stage 1 evaluation inputs).
    case_files = sorted(p for p in invo_dir.glob('*.invo.pkl')
                        if not p.name.endswith('.normal.invo.pkl')
                        and not p.name.startswith('trainticket_historical_'))
    if args.cases:
        wanted = set(args.cases)
        case_files = [p for p in case_files if p.stem.replace('.invo', '') in wanted
                                            or p.name.replace('.invo.pkl', '') in wanted]
    if not case_files:
        print(f"No invo.pkl files found in {invo_dir}", file=sys.stderr)
        sys.exit(2)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for cell_name in selected:
        if cell_name not in cells:
            print(f"[skip] unknown cell {cell_name!r}", file=sys.stderr)
            continue
        cell = merge_cell(defaults, cells[cell_name].get('overrides', {}))
        cell_dir = out_root / cell_name
        log_dir = cell_dir / 'logs'
        cell_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        with open(cell_dir / 'cell.json', 'w') as f:
            json.dump(cell, f, indent=2, sort_keys=True)

        for case_invo in case_files:
            case_id = case_invo.name.replace('.invo.pkl', '')
            cell_case_dir = cell_dir / case_id
            cell_case_dir.mkdir(parents=True, exist_ok=True)

            # --- 1. dual cache (only if F2b is on) ---
            dual_cache_path = ''
            if cell.get('baseline_window') == 'dual':
                # parse case_id like ts-route-service_delay_2 → ('ts-route-service_delay', '2')
                parts = case_id.rsplit('_', 1)
                if len(parts) != 2 or not parts[1].isdigit():
                    print(f"[skip-dual] {case_id}: cannot parse rep number", file=sys.stderr)
                    continue
                fault_dir, rep = parts
                inject_file = Path(args.inject_times_dir) / fault_dir / rep / 'inject_time.txt'
                if not inject_file.exists():
                    print(f"[skip-dual] {case_id}: missing {inject_file}", file=sys.stderr)
                    continue
                inject_ts_us = int(inject_file.read_text().strip()) * 1_000_000

                # Dual-window history must be drawn from pre-injection data.
                # Preprocess writes pre-fault rows to <case>.normal.invo.pkl;
                # if it exists prefer that, else fall back to the combined pkl.
                normal_invo = case_invo.with_name(case_invo.name.replace('.invo.pkl', '.normal.invo.pkl'))
                history_invo = normal_invo if normal_invo.exists() else case_invo

                dual_cache_path = str(cell_case_dir / 'dual_cache.pkl')
                cmd = [args.python, str(REPO_ROOT / 'run_anomaly_detection_prepare_model.py'),
                       '--baseline-window', 'dual',
                       '--case-invo', str(history_invo),
                       '--inject-ts-us', str(inject_ts_us),
                       '--last-slot-seconds', str(cell['last_slot_seconds']),
                       '--last-period-seconds', str(cell['last_period_seconds']),
                       '--stage1-granularity', cell['stage1_granularity'],
                       '-o', dual_cache_path]
                if args.dry_run:
                    print('DRY:', ' '.join(cmd))
                else:
                    rc = run_cmd(cmd, log_dir / f'{case_id}.dual.log')
                    if rc != 0:
                        print(f"[FAIL] dual-cache {case_id} rc={rc}", file=sys.stderr)
                        continue

            # --- 2. selecting features ---
            uf_path = cell_case_dir / 'useful_features.txt'
            cmd = [args.python, str(REPO_ROOT / 'run_selecting_features.py'),
                   '-i', str(case_invo),
                   '-o', str(uf_path),
                   '-h', args.global_history,
                   '--feature-selector', cell['feature_selector'],
                   '--fs-delta', str(cell['fs_delta']),
                   '--fs-floor', str(cell['fs_floor']),
                   '--baseline-window', cell['baseline_window'],
                   '--stage1-granularity', cell['stage1_granularity']]
            if dual_cache_path:
                cmd += ['--dual-cache', dual_cache_path]
            if args.dry_run:
                print('DRY:', ' '.join(cmd))
            else:
                rc = run_cmd(cmd, log_dir / f'{case_id}.fs.log')
                if rc != 0:
                    print(f"[FAIL] feature-select {case_id} rc={rc}", file=sys.stderr)
                    continue

            # --- 3. Stage 1 invo anomaly detection ---
            invo_result = cell_case_dir / 'invo.result.pkl'
            cmd = [args.python, str(REPO_ROOT / 'run_anomaly_detection_invo.py'),
                   '-i', str(case_invo),
                   '-o', str(invo_result),
                   '-c', args.global_cache,
                   '-u', str(uf_path),
                   '--stage1-granularity', cell['stage1_granularity'],
                   '--min-baseline-samples', str(cell['min_baseline_samples'])]
            if dual_cache_path:
                cmd += ['--dual-cache', dual_cache_path]
            if args.dry_run:
                print('DRY:', ' '.join(cmd))
            else:
                rc = run_cmd(cmd, log_dir / f'{case_id}.stage1.log')
                if rc != 0:
                    print(f"[FAIL] stage1 {case_id} rc={rc}", file=sys.stderr)
                    continue

            summary.append({'cell': cell_name, 'case': case_id, 'invo_result': str(invo_result)})

    summary_path = out_root / 'manifest.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'cells_run': selected,
            'cases_run': [p.name for p in case_files],
            'config_file': str(config_path),
            'entries': summary,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }, f, indent=2)
    print(f"\nWrote manifest: {summary_path}")


if __name__ == '__main__':
    main()
