"""
Feature-selection-only parity check (Step 4 verification).

Runs run_selecting_features.py with default flags (= legacy stderr_criteria
+ global window + pair granularity) across every case and diffs the output
file against output/trainticket_anomaly_detection.test/{case}.useful_features.1.

This isolates Step 4 from the run_anomaly_detection_invo.py path (which on
this machine is blocked by an unrelated numpy/sklearn pickle version
mismatch in the existing global cache).
"""
from __future__ import annotations

import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
INVO_DIR = REPO_ROOT / 'output' / 'trainticket_invo_encoded'
TEST_DIR = REPO_ROOT / 'output' / 'trainticket_anomaly_detection.test'
GLOBAL_HISTORY = INVO_DIR / 'trainticket_historical_normal.invo.pkl'


def parse_useful_features(text):
    """
    Parse a useful_features dict from its text repr. The legacy reference
    files were emitted under NumPy 2.x where keys repr as np.str_('foo'),
    while local NumPy 1.x emits plain 'foo' — equivalent but lexically
    different. Normalize by converting all keys to plain Python str.
    """
    d = eval(text, {'np': np, '__builtins__': {}})
    return {tuple(str(k_part) for k_part in key): list(val) for key, val in d.items()}


def main():
    invo_pkls = [p for p in sorted(INVO_DIR.glob('*.invo.pkl'))
                 if 'historical' not in p.name and '.normal.' not in p.name]
    if not invo_pkls:
        print(f"No per-case invo.pkl in {INVO_DIR}", file=sys.stderr)
        sys.exit(2)

    print(f"Found {len(invo_pkls)} case(s).")
    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for invo in invo_pkls:
            case_id = invo.name.replace('.invo.pkl', '')
            uf_new = tmp / f'{case_id}.useful_features.1'
            cmd = [sys.executable, str(REPO_ROOT / 'run_selecting_features.py'),
                   '-i', str(invo),
                   '-o', str(uf_new),
                   '-h', str(GLOBAL_HISTORY),
                   '--fisher', '1']
            try:
                subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                results.append((case_id, False, f'subprocess rc={e.returncode}'))
                continue
            uf_ref = TEST_DIR / f'{case_id}.useful_features.1'
            if not uf_ref.exists():
                results.append((case_id, False, 'ref missing'))
                continue
            try:
                a = parse_useful_features(uf_new.read_text())
                b = parse_useful_features(uf_ref.read_text())
            except Exception as e:
                results.append((case_id, False, f'parse error: {e}'))
                continue
            results.append((case_id, a == b, f'keys={len(a)}/{len(b)}'))

    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n_pass}/{len(results)} pass")
    for case_id, ok, msg in results:
        if not ok:
            print(f"  [FAIL] {case_id}  {msg}")
    if n_pass != len(results):
        sys.exit(1)


if __name__ == '__main__':
    main()
