import importlib
import pickle

import click
import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.under_sampling import RandomUnderSampler
from trainticket_config import FEATURE_NAMES


def extract_data(path):
    x = np.load(path)
    return x['data'], x['labels'], x['masks'], x['trace_ids']


def _baseline_token(key_tuple, feature: str, granularity: str) -> str:
    """Cache token convention used by Stage 1 for baseline mean/std lookups."""
    if granularity == 'operation':
        src, tgt, method = key_tuple
        return f"reference-{src}-{tgt}-{method}-{feature}-mean-variance"
    src, tgt = key_tuple
    return f"reference-{src}-{tgt}-{feature}-mean-variance"


def extract_dual_window_history(invo_pkl_path: str, inject_ts_us: int,
                                slot_seconds: int, period_seconds: int):
    """
    Return (history_df, slot_df, period_df) — invocations whose start_timestamp
    falls in [inject - slot, inject) or [inject - period - slot, inject - period).
    """
    with open(invo_pkl_path, 'rb') as f:
        df = pickle.load(f)
    last_slot_start = inject_ts_us - slot_seconds * 1_000_000
    last_period_end = inject_ts_us - period_seconds * 1_000_000
    last_period_start = last_period_end - slot_seconds * 1_000_000

    in_last_slot = (df['start_timestamp'] >= last_slot_start) & \
                   (df['start_timestamp'] < inject_ts_us)
    in_last_period = (df['start_timestamp'] >= last_period_start) & \
                     (df['start_timestamp'] < last_period_end)

    slot_df = df[in_last_slot]
    period_df = df[in_last_period]
    history = df[in_last_slot | in_last_period]
    return history, slot_df, period_df


def compute_baseline_cache(history_df, granularity: str, feature_names):
    """Compute per-key mean/std/n for each feature over a history dataframe."""
    if granularity == 'operation':
        keys = ['source', 'target', 'callee_method']
    else:
        keys = ['source', 'target']

    if len(history_df) == 0:
        return {}, []

    indexed = history_df.set_index(keys=keys, drop=False).sort_index()
    indices = np.unique(indexed.index.values)

    cache = {}
    for key_tuple in indices:
        if granularity == 'pair' and not isinstance(key_tuple, tuple):
            key_tuple = tuple(key_tuple) if hasattr(key_tuple, '__len__') else key_tuple
        n = len(np.atleast_1d(indexed.loc[key_tuple, 'source'].values
                              if hasattr(indexed.loc[key_tuple, 'source'], 'values')
                              else [indexed.loc[key_tuple, 'source']]))
        for feature in feature_names:
            vals = indexed.loc[key_tuple, feature]
            arr = np.atleast_1d(vals.values if hasattr(vals, 'values') else vals)
            token = _baseline_token(key_tuple, feature, granularity)
            cache[token] = {
                'mean': float(np.mean(arr)),
                'std': float(np.maximum(np.std(arr), 0.1)),
                'n': int(arr.shape[0]),
            }
    return cache, list(indices)


def build_dual_cache(case_invo_path: str, output_path: str,
                    inject_ts_us: int, slot_seconds: int, period_seconds: int,
                    granularity: str, feature_names):
    """
    Build a per-case dual-window cache file. Output schema:
        {
            'baseline': {token: {mean, std, n}, ...},
            'history':  pd.DataFrame,    # the union window, raw rows
            'slot':     pd.DataFrame,    # last-slot window only
            'period':   pd.DataFrame,    # last-period window only
            'granularity': 'pair' | 'operation',
            'feature_names': [...],
            'window':   {'inject_ts_us', 'slot_seconds', 'period_seconds'},
        }
    Stage 1 reads only `baseline`. run_selecting_features reads `history` for
    the paper feature selector.
    """
    history, slot_df, period_df = extract_dual_window_history(
        case_invo_path, inject_ts_us, slot_seconds, period_seconds,
    )
    baseline, _ = compute_baseline_cache(history, granularity, feature_names)

    if len(slot_df) == 0 and len(period_df) > 0:
        logger.warning(f"{case_invo_path}: dual window using period-only "
                       f"({len(period_df)} rows; slot empty)")
    if len(period_df) == 0 and len(slot_df) > 0:
        logger.warning(f"{case_invo_path}: dual window using slot-only "
                       f"({len(slot_df)} rows; period empty — short trace dump?)")

    out = {
        'baseline': baseline,
        'history': history,
        'slot': slot_df,
        'period': period_df,
        'granularity': granularity,
        'feature_names': list(feature_names),
        'window': {
            'inject_ts_us': inject_ts_us,
            'slot_seconds': slot_seconds,
            'period_seconds': period_seconds,
        },
    }
    with open(output_path, 'wb+') as f:
        pickle.dump(out, f)


@click.command('train-anoamly-detection-model')
@click.option('-i', '--invo-history', default='historical_data.pkl', type=str)
@click.option('-t', '--trace-history', default='historical_data.pkl', type=str)
@click.option('-o', '--output-file', type=str)
@click.option('--baseline-window', type=click.Choice(['global', 'dual']), default='global',
              help='F2b: global=legacy concatenated history; dual=per-case dual-window.')
@click.option('--case-invo', default='', type=str,
              help='[dual only] Path to a single case .invo.pkl produced by run_invo_encoding.')
@click.option('--inject-ts-us', default=0, type=int,
              help='[dual only] Fault injection timestamp in microseconds.')
@click.option('--last-slot-seconds', default=300, type=int,
              help='[dual only] Length of last-slot window (seconds).')
@click.option('--last-period-seconds', default=86400, type=int,
              help='[dual only] Period offset (seconds, e.g. 86400 = 1 day).')
@click.option('--stage1-granularity', type=click.Choice(['pair', 'operation']), default='pair',
              help='[dual only] Cache key granularity: (src,tgt) or (src,tgt,callee_method).')
@click.option('--dataset', type=click.Choice(['tt', 'ob']), default='tt',
              help='[dual only] Dataset config to load FEATURE_NAMES from.')
def main(trace_history, invo_history, output_file,
         baseline_window, case_invo, inject_ts_us, last_slot_seconds,
         last_period_seconds, stage1_granularity, dataset):
    if baseline_window == 'dual':
        if not case_invo or inject_ts_us <= 0:
            raise click.UsageError(
                '--baseline-window=dual requires --case-invo and --inject-ts-us'
            )
        cfg = importlib.import_module(
            'trainticket_config' if dataset == 'tt' else 'onlineboutique_config'
        )
        build_dual_cache(
            case_invo, output_file, inject_ts_us, last_slot_seconds,
            last_period_seconds, stage1_granularity, cfg.FEATURE_NAMES,
        )
        return

    # ----- legacy global path (off-switch baseline) -----
    his_data, his_labels, his_masks, his_trace_ids = extract_data(trace_history)
    # rus = RandomUnderSampler(random_state=0)
    # X_resampled, y_resampled = rus.fit_resample(his_data, his_labels)

    result = {}
    for algorithm in ['RF-Trace', 'MLP-Trace']:
        if algorithm == 'RF-Trace':
            model = RandomForestClassifier(n_estimators=100, n_jobs=10, verbose=0)
        elif algorithm == 'MLP-Trace':
            model = MLPClassifier(batch_size=256, early_stopping=True, verbose=0, learning_rate_init=1e-4,
                                  max_iter=100, hidden_layer_sizes=(100, 100))
        elif algorithm == 'KNN-Trace':
            model = KNeighborsClassifier()
        else:
            raise RuntimeError()
        model.fit(his_data, his_labels)
        result[algorithm] = model

    with open(invo_history, 'rb') as f:
        invo_history = pickle.load(f)
    if stage1_granularity == 'operation':
        idx_keys = ['source', 'target', 'callee_method']
    else:
        idx_keys = ['source', 'target']
    invo_history = invo_history.set_index(keys=idx_keys, drop=False).sort_index()
    indices = np.unique(invo_history.index.values)
    for key in indices:
        if stage1_granularity == 'operation' and not isinstance(key, tuple):
            key = tuple(key)
        reference = invo_history.loc[key, FEATURE_NAMES].values
        if stage1_granularity == 'operation':
            src, tgt, method = key
            token = f"IF-{src}-{tgt}-{method}"
        else:
            src, tgt = key
            token = f"IF-{src}-{tgt}"
        model = IsolationForest(contamination=0.01, n_jobs=10)
        model.fit(reference)
        result[token] = model

    for key in indices:
        if stage1_granularity == 'operation' and not isinstance(key, tuple):
            key = tuple(key)
        if stage1_granularity == 'operation':
            src, tgt, method = key
            kstr = f"{src}-{tgt}-{method}"
        else:
            src, tgt = key
            kstr = f"{src}-{tgt}"
        for feature in FEATURE_NAMES:
            reference = invo_history.loc[key, feature].values
            token = f"reference-{kstr}-{feature}-mean-variance"
            result[token] = {
                'mean': np.mean(reference[:]),
                'std': np.maximum(np.std(reference[:]), 0.1)
            }

    with open(output_file, 'wb+') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    main()
