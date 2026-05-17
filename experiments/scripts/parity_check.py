"""
Off-switch parity check for paper-accurate Stage 1.

Re-runs run_selecting_features.py and run_anomaly_detection_invo.py with all
new flags at their NEUTRAL DEFAULTS (= legacy baseline) and diffs the outputs
against the existing baseline pkls under
output/trainticket_anomaly_detection.test/.

Acceptance:
    - useful_features.1 file matches existing useful_features.1 byte-for-byte.
    - invo.result.pkl.1.1 matches existing pkl on the 'predict' column for
      every (source, target) row.

Usage:
    python experiments/scripts/parity_check.py
"""
from __future__ import annotations

import os
import pickle
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INVO_DIR = REPO_ROOT / 'output' / 'trainticket_invo_encoded'
TEST_DIR = REPO_ROOT / 'output' / 'trainticket_anomaly_detection.test'
GLOBAL_HISTORY = INVO_DIR / 'trainticket_historical_normal.invo.pkl'
GLOBAL_CACHE = REPO_ROOT / 'output' / 'trainticket_anomaly_detection.models.0.0'


def run(cmd, capture=False):
    print('$', ' '.join(shlex.quote(c) for c in cmd))
    if capture:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    return subprocess.check_call(cmd)


def diff_useful_features(new_path: Path, ref_path: Path) -> tuple[bool, str]:
    if not ref_path.exists():
        return False, f"ref missing: {ref_path}"
    a = new_path.read_text()
    b = ref_path.read_text()
    return (a == b), f"len_a={len(a)}, len_b={len(b)}"


def diff_invo_result(new_path: Path, ref_path: Path) -> tuple[bool, str]:
    if not ref_path.exists():
        return False, f"ref missing: {ref_path}"
    with open(new_path, 'rb') as f:
        a = pickle.load(f)
    with open(ref_path, 'rb') as f:
        b = pickle.load(f)
    if 'predict' not in a.columns or 'predict' not in b.columns:
        return False, "predict column missing"
    if len(a) != len(b):
        return False, f"row_count_a={len(a)} row_count_b={len(b)}"
    a_pred = a['predict'].fillna(-1).values
    b_pred = b['predict'].fillna(-1).values
    if (a_pred == b_pred).all():
        return True, f"rows={len(a)}"
    n_diff = int((a_pred != b_pred).sum())
    return False, f"rows={len(a)}, n_diff={n_diff}"


def main():
    invo_pkls = sorted(INVO_DIR.glob('*.invo.pkl'))
    invo_pkls = [p for p in invo_pkls if 'historical' not in p.name and '.normal.' not in p.name]
    if not invo_pkls:
        print(f"No per-case invo.pkl in {INVO_DIR}", file=sys.stderr)
        sys.exit(2)

    print(f"Found {len(invo_pkls)} case(s).")
    fs_results = []
    invo_results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for invo in invo_pkls:
            case_id = invo.name.replace('.invo.pkl', '')

            # 1. useful features (defaults = stderr selector, global window, pair)
            uf_new = tmp / f'{case_id}.useful_features.1'
            run([sys.executable, str(REPO_ROOT / 'run_selecting_features.py'),
                 '-i', str(invo),
                 '-o', str(uf_new),
                 '-h', str(GLOBAL_HISTORY),
                 '--fisher', '1'])
            uf_ref = TEST_DIR / f'{case_id}.useful_features.1'
            ok, msg = diff_useful_features(uf_new, uf_ref)
            fs_results.append((case_id, ok, msg))

            # 2. invo result (defaults = pair granularity, no dual cache)
            invo_new = tmp / f'{case_id}.invo.result.pkl.1.1'
            run([sys.executable, str(REPO_ROOT / 'run_anomaly_detection_invo.py'),
                 '-i', str(invo),
                 '-o', str(invo_new),
                 '-h', str(GLOBAL_HISTORY),
                 '-u', str(uf_ref),  # use the locked baseline useful_features
                 '-c', str(GLOBAL_CACHE),
                 '--threshold', '1'])
            invo_ref = TEST_DIR / f'{case_id}.invo.result.pkl.1.1'
            ok, msg = diff_invo_result(invo_new, invo_ref)
            invo_results.append((case_id, ok, msg))

    print('\n=== feature selection parity ===')
    for case_id, ok, msg in fs_results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {case_id}  {msg}")

    print('\n=== invo Stage 1 parity ===')
    for case_id, ok, msg in invo_results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {case_id}  {msg}")

    fs_pass = sum(1 for _, ok, _ in fs_results if ok)
    invo_pass = sum(1 for _, ok, _ in invo_results if ok)
    print(f"\nfs:   {fs_pass}/{len(fs_results)} pass")
    print(f"invo: {invo_pass}/{len(invo_results)} pass")
    if fs_pass != len(fs_results) or invo_pass != len(invo_results):
        sys.exit(1)


if __name__ == '__main__':
    main()
