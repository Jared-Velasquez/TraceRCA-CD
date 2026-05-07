import pickle
from collections import defaultdict
from itertools import product
from pathlib import Path

import click
import numpy as np
import seaborn as sns
from loguru import logger
from tqdm import tqdm
from pprint import pprint
from trainticket_config import FEATURE_NAMES

DEBUG = False  # very slow


def distribution_criteria(empirical, reference, threshold):
    empirical, reference = np.array(empirical), np.array(reference)
    historical_mean, historical_std = np.mean(reference), np.std(reference)
    ref_ratio = sum(np.abs(reference - historical_mean) > 3 * historical_std) / reference.shape[0]
    emp_ratio = sum(np.abs(empirical - historical_mean) > 3 * historical_std) / empirical.shape[0]
    return (emp_ratio - ref_ratio) > threshold * ref_ratio


def fisher_criteria(empirical, reference, side='two-sided'):
    if side == 'two-sided':
        diff_mean = (np.abs(np.mean(empirical) - np.mean(reference)) ** 2)
    elif side == 'less':
        diff_mean = np.maximum(np.mean(empirical) - np.mean(reference), 0) ** 2
    elif side == 'greater':
        diff_mean = np.maximum(np.mean(reference) - np.mean(empirical), 0) ** 2
    else:
        raise RuntimeError(f'invalid side: {side}')
    variance = np.maximum(np.var(empirical) + np.var(reference), 0.1)
    return diff_mean / variance


def stderr_criteria(empirical, reference, threshold):
    empirical, reference = np.array(empirical), np.array(reference)
    historical_mean, historical_std = np.mean(reference), np.std(reference)
    historical_std = np.maximum(historical_std, historical_mean * 0.01 + 0.01)
    ref_ratio = np.mean(np.abs(reference - historical_mean)) / historical_std
    emp_ratio = np.mean(np.abs(empirical - historical_mean)) / historical_std
    return (emp_ratio - ref_ratio) > threshold * ref_ratio + 1.0


def paper_criteria(empirical, reference, delta_fs, fs_floor=0.0):
    """
    F2a: paper Eq. (IWQoS 2021, p.3): alpha_after - alpha_before > delta_fs * alpha_before.

    alpha = mean(|v - mu_ref| / sigma_ref) over the relevant invocation set.
    fs_floor (optional, non-paper) requires alpha_after > fs_floor; default 0.0 = paper-faithful.
    """
    ref = np.asarray(reference, dtype=float)
    emp = np.asarray(empirical, dtype=float)
    if ref.size == 0 or emp.size == 0:
        return False
    mu = float(np.mean(ref))
    sd = float(np.maximum(np.std(ref), 0.1))
    alpha_before = float(np.mean(np.abs(ref - mu) / sd))
    alpha_after = float(np.mean(np.abs(emp - mu) / sd))
    if alpha_after <= fs_floor:
        return False
    return (alpha_after - alpha_before) > (delta_fs * alpha_before)


@click.command('invocation feature selection')
@click.option('-i', '--input', 'input_file', default="*.pkl", type=str)
@click.option('-o', '--output', 'output_file', default='.', type=str)
@click.option('-h', '--history', default='historical_data.pkl', type=str)
@click.option("-f", "--fisher", "fisher_threshold", default=1, type=float)
@click.option('--feature-selector', type=click.Choice(['stderr', 'paper']), default='stderr',
              help='F2a: stderr (legacy) or paper (alpha_after - alpha_before > delta_fs * alpha_before).')
@click.option('--fs-delta', default=0.1, type=float,
              help='[paper only] delta_fs threshold (paper default 0.1).')
@click.option('--fs-floor', default=0.0, type=float,
              help='[paper only] non-paper guard: require alpha_after > fs_floor (default 0.0).')
@click.option('--baseline-window', type=click.Choice(['global', 'dual']), default='global',
              help='F2b: global=use --history pkl; dual=use --dual-cache pkl history field.')
@click.option('--dual-cache', 'dual_cache_file', default='', type=str,
              help='[dual only] Per-case dual-window cache pkl produced by prepare_model dual mode.')
@click.option('--stage1-granularity', type=click.Choice(['pair', 'operation']), default='pair',
              help='Group by (source,target) [legacy] or (source,target,callee_method) [per-op].')
def selecting_feature_main(input_file: str, output_file: str, history: str, fisher_threshold,
                           feature_selector, fs_delta, fs_floor,
                           baseline_window, dual_cache_file, stage1_granularity):
    input_file = Path(input_file)
    output_file = Path(output_file)

    if baseline_window == 'dual':
        if not dual_cache_file:
            raise click.UsageError('--baseline-window=dual requires --dual-cache')
        with open(dual_cache_file, 'rb') as f:
            dual_cache = pickle.load(f)
        history = dual_cache['history']
    else:
        with open(history, 'rb') as f:
            history = pickle.load(f)

    with open(str(input_file), 'rb') as f:
        df = pickle.load(f)
    if stage1_granularity == 'operation':
        index_keys = ['source', 'target', 'callee_method']
    else:
        index_keys = ['source', 'target']
    df = df.set_index(keys=index_keys, drop=True).sort_index()
    history = history.set_index(keys=index_keys, drop=True).sort_index()
    indices = np.intersect1d(np.unique(df.index.values), np.unique(history.index.values))
    useful_features_dict = defaultdict(list)
    if DEBUG:
        plot_dir = output_file.parent / 'selecting_feature.debug'
        plot_dir.mkdir(exist_ok=True)
    for key, feature in tqdm(product(indices, FEATURE_NAMES)):
        # Normalize key shape so dict lookup uses canonical tuples.
        if stage1_granularity == 'operation' and not isinstance(key, tuple):
            key = tuple(key)
        empirical = np.sort(df.loc[key, feature].values)
        reference = np.sort(history.loc[key, feature].values)
        p_value = -1
        if feature_selector == 'paper':
            fisher = paper_criteria(empirical, reference, fs_delta, fs_floor)
        else:
            fisher = stderr_criteria(empirical, reference, fisher_threshold)
        if fisher:
            useful_features_dict[key].append(feature)
        try:
            if DEBUG:
                import matplotlib.pyplot as plt
                from matplotlib.figure import Figure
                fig = Figure(figsize=(4, 3))
                sns.distplot(empirical, label='Empirical')
                sns.distplot(reference, label='Reference')
                plt.xlabel(feature)
                plt.ylabel('PDF')
                plt.legend()
                key_label = "_".join(map(str, key)) if isinstance(key, tuple) else str(key)
                plt.title(f"{key_label}, ks={p_value:.2f}, fisher={fisher:.2f}")
                plt.savefig(
                    plot_dir / f"{input_file.name.split('.')[0]}_{key_label}_{feature}.pdf",
                    bbox_inches='tight', pad_inches=0
                )
        except:
            pass
            # logger.debug(f"{input_file.name} {source} {target} {feature} {fisher}")
            # useful_features_dict[(source, target)].append(feature)
    # logger.debug(f"{input_file.name} {dict(useful_features_dict)}")
    print(f"{input_file.name} {dict(useful_features_dict)}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w+') as f:
        print(dict(useful_features_dict), file=f)


if __name__ == '__main__':
    selecting_feature_main()
