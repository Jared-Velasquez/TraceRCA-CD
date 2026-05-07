import pickle
import time
from pathlib import Path

import click
import numpy as np
from sklearn.ensemble import IsolationForest
from loguru import logger
from trainticket_config import FEATURE_NAMES
from diskcache import Cache

DEBUG = True

threshold = 1.0
MIN_BASELINE_SAMPLES = 30


def _baseline_lookup(token, dual_cache, global_cache, min_samples):
    """
    Look up a baseline (mean, std) by token. Order:
        dual_cache (if entry has n >= min_samples) → global_cache → None
    """
    if dual_cache:
        entry = dual_cache.get('baseline', {}).get(token)
        if entry is not None and entry.get('n', 0) >= min_samples:
            return entry['mean'], entry['std']
    if token in global_cache:
        return global_cache[token]['mean'], global_cache[token]['std']
    return None


def anomaly_detection_isolation_forest(df, result_column, history, cache):
    indices = np.unique(df.index.values)
    for source, target in indices:
        empirical = df.loc[(source, target), FEATURE_NAMES].values
        # reference = history.loc[(source, target), FEATURE_NAMES].values
        token = f"IF-{source}-{target}"
        if token not in cache:
            df.loc[(source, target), result_column] = 0
            continue
        model = cache[token]
        predict = model.predict(empirical)
        df.loc[(source, target), result_column] = predict
    return df


def anomaly_detection_3sigma_without_useful_features(df, result_column, history, cache,
                                                    dual_cache=None,
                                                    min_baseline_samples=MIN_BASELINE_SAMPLES):
    indices = np.unique(df.index.values)
    useful_feature = {key: FEATURE_NAMES for key in indices}
    return anomaly_detection_3sigma(df, result_column, None, useful_feature, cache=cache,
                                   dual_cache=dual_cache,
                                   min_baseline_samples=min_baseline_samples)


def anomaly_detection_3sigma(df, result_column, history, useful_feature, cache,
                             dual_cache=None,
                             min_baseline_samples=MIN_BASELINE_SAMPLES):
    indices = np.unique(df.index.values)
    for source, target in indices:
        if (source, target) not in useful_feature:  # all features are not useful
            df.loc[(source, target), result_column] = 0
            continue
        features = useful_feature[(source, target)]
        empirical = df.loc[(source, target), features].values
        mean, std = [], []
        for idx, feature in enumerate(features):
            token = f"reference-{source}-{target}-{feature}-mean-variance"
            looked_up = _baseline_lookup(token, dual_cache, cache, min_baseline_samples)
            if looked_up is not None:
                mean.append(looked_up[0])
                std.append(looked_up[1])
            else:
                mean.append(np.mean(empirical, axis=0)[idx])
                std.append(np.maximum(np.std(empirical, axis=0)[idx], 0.1))
        mean = np.asarray(mean)
        std = np.asarray(std)
        predict = np.zeros(empirical.shape)
        for idx, feature in enumerate(features):
            predict[:, idx] = np.abs(empirical[:, idx] - mean[idx]) > threshold * std[idx]
        predict = np.max(predict, axis=1)

        df.loc[(source, target), result_column] = predict
    return df


@click.command('invocation anomaly detection')
@click.option('-i', '--input', 'input_file', default="*.pkl", type=str)
@click.option('-o', '--output', 'output_file', default='.', type=str)
@click.option('-h', '--history', default='historical_data.pkl', type=str)
@click.option('-u', '--useful-feature', "useful_feature", default='.', type=str)
@click.option('-c', '--cache', 'cache_file', default='.', type=str)
@click.option('-t', '--threshold', 'main_threshold', default=1, type=float)
@click.option('--dual-cache', 'dual_cache_file', default='', type=str,
              help='F2b: per-case dual-window cache file (optional). '
                   'If provided, baselines are looked up there first; falls back to --cache.')
@click.option('--min-baseline-samples', default=MIN_BASELINE_SAMPLES, type=int,
              help='[dual only] Min n in dual cache to use it; else fall back to global.')
def invo_anomaly_detection_main(input_file, output_file, history, useful_feature, cache_file,
                                main_threshold, dual_cache_file, min_baseline_samples):
    global threshold
    threshold = main_threshold

    history = None
    with open(useful_feature, 'r') as f:
        useful_feature = eval("".join(f.readlines()))
    with open(cache_file, 'rb+') as f:
        cache = pickle.load(f)

    dual_cache = None
    if dual_cache_file:
        with open(dual_cache_file, 'rb') as f:
            dual_cache = pickle.load(f)

    input_file = Path(input_file)

    with open(input_file, 'rb') as f:
        df = pickle.load(f)
    df = df.set_index(keys=['source', 'target'], drop=False).sort_index()
    tic = time.time()
    df = anomaly_detection_3sigma(df, 'Ours-predict', None, useful_feature, cache=cache,
                                  dual_cache=dual_cache,
                                  min_baseline_samples=min_baseline_samples)
    toc = time.time()
    print("algo:", "ours", "time:", toc - tic, 'invos:', len(df))

    if len(df) == 0:
        for col in ('Ours-predict', 'NoSelection-predict', 'IF-predict', 'predict'):
            df[col] = np.nan
        with open(output_file, 'wb+') as f:
            pickle.dump(df, f)
        return

    df = anomaly_detection_3sigma_without_useful_features(
        df, 'NoSelection-predict', None, cache=cache,
        dual_cache=dual_cache,
        min_baseline_samples=min_baseline_samples,
    )

    # tic = time.time()
    df = anomaly_detection_isolation_forest(df, 'IF-predict', None, cache=cache)
    # toc = time.time()
    # print("algo:", "IF", "time:", toc - tic, 'invos:', len(df))

    df['predict'] = df['Ours-predict']
    with open(output_file, 'wb+') as f:
        pickle.dump(df, f)


if __name__ == '__main__':
    invo_anomaly_detection_main()

