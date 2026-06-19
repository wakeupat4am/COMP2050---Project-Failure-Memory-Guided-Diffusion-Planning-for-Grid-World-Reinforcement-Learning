from __future__ import annotations

from collections.abc import Iterable
from math import sqrt

import numpy as np
import pandas as pd


RAW_METRIC_COLUMNS = {
    "success": "Success Rate",
    "collision": "Collision Rate",
    "total_return": "Average Return",
    "path_length": "Path Length",
    "optimality_gap": "Optimality Gap",
    "repeated_failure_rate": "Repeated Failure Rate",
    "inference_time_per_action": "Inference Time",
}

_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.16,
    14: 2.145,
    15: 2.131,
    16: 2.12,
    17: 2.11,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.08,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.06,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _t_critical_95(df: int) -> float:
    if df <= 0:
        return 0.0
    if df in _T_CRITICAL_95:
        return _T_CRITICAL_95[df]
    if df > 30:
        return 1.96
    lower_dfs = [candidate for candidate in _T_CRITICAL_95 if candidate < df]
    return _T_CRITICAL_95[max(lower_dfs)]


def _ci_halfwidth(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    std = float(np.std(values, ddof=1))
    return _t_critical_95(len(values) - 1) * std / sqrt(len(values))


def aggregate_seed_metrics(
    raw_df: pd.DataFrame,
    group_cols: Iterable[str],
    seed_col: str = "Seed",
) -> pd.DataFrame:
    group_cols = list(group_cols)
    records = []
    grouped = raw_df.groupby(group_cols + [seed_col], dropna=False, sort=True)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols + [seed_col], keys))
        row["Episodes"] = int(len(group))
        for raw_col, label in RAW_METRIC_COLUMNS.items():
            row[label] = float(group[raw_col].mean(skipna=True))
        records.append(row)
    return pd.DataFrame(records)


def summarize_seed_metrics(
    per_seed_df: pd.DataFrame,
    group_cols: Iterable[str],
) -> pd.DataFrame:
    group_cols = list(group_cols)
    records = []
    grouped = per_seed_df.groupby(group_cols, dropna=False, sort=True)
    metric_labels = list(RAW_METRIC_COLUMNS.values())
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["Seed Count"] = int(group["Seed"].nunique()) if "Seed" in group.columns else int(len(group))
        row["Episodes per Seed"] = float(group["Episodes"].mean()) if "Episodes" in group.columns else np.nan
        for label in metric_labels:
            values = group[label].to_numpy(dtype=float)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ci_halfwidth = _ci_halfwidth(values)
            row[f"{label} Mean"] = mean
            row[f"{label} Std"] = std
            row[f"{label} CI Low"] = mean - ci_halfwidth
            row[f"{label} CI High"] = mean + ci_halfwidth
            row[f"{label} CI Halfwidth"] = ci_halfwidth
        records.append(row)
    return pd.DataFrame(records)


def aggregate_raw_to_seed_summary(
    raw_df: pd.DataFrame,
    group_cols: Iterable[str],
    seed_col: str = "Seed",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed_df = aggregate_seed_metrics(raw_df=raw_df, group_cols=group_cols, seed_col=seed_col)
    summary_df = summarize_seed_metrics(per_seed_df=per_seed_df, group_cols=group_cols)
    return per_seed_df, summary_df


def format_mean_std(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} +- {std:.{digits}f}"


def format_mean_ci(mean: float, ci_low: float, ci_high: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} [{ci_low:.{digits}f}, {ci_high:.{digits}f}]"
