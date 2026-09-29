from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def sorted_columns(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {col: np.sort(df[col].to_numpy(dtype=np.float64)) for col in df.columns}


def column_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        x = df[col].to_numpy(dtype=np.float64)
        values, counts = np.unique(x, return_counts=True)
        mode_numpy = float(values[np.argmax(counts)])
        mode_pd = df[col].mode(dropna=True)
        rows.append({
            "column": col,
            "mean_numpy": np.mean(x),
            "mean_pandas": df[col].mean(),
            "sample_variance_numpy": np.var(x, ddof=1),
            "sample_variance_pandas": df[col].var(ddof=1),
            "mode_numpy": mode_numpy,
            "mode_pandas": float(mode_pd.iloc[0]) if not mode_pd.empty else np.nan,
            "mode_scipy_check": float(stats.mode(x, keepdims=False).mode),
            "mode_count": int(np.max(counts)),
            "median_numpy": np.median(x),
            "median_pandas": df[col].median(),
        })
    return pd.DataFrame(rows).set_index("column")


def confidence_intervals(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        x = df[col].to_numpy(dtype=np.float64)
        n = len(x)
        mean = np.mean(x)
        s = np.std(x, ddof=1)
        s2 = np.var(x, ddof=1)
        tcrit = stats.t.ppf(1 - alpha/2, df=n-1)
        chi_low = stats.chi2.ppf(alpha/2, df=n-1)
        chi_high = stats.chi2.ppf(1-alpha/2, df=n-1)
        rows.append({
            "column": col,
            "mean": mean,
            "mean_ci_low": mean - tcrit*s/np.sqrt(n),
            "mean_ci_high": mean + tcrit*s/np.sqrt(n),
            "sample_variance": s2,
            "var_ci_low": (n-1)*s2/chi_high,
            "var_ci_high": (n-1)*s2/chi_low,
            "alpha": alpha,
        })
    return pd.DataFrame(rows).set_index("column")


def covariance_correlation(df: pd.DataFrame):
    x = df.to_numpy(dtype=np.float64)
    cols = df.columns
    return (
        pd.DataFrame(np.cov(x, rowvar=False, ddof=1), index=cols, columns=cols),
        pd.DataFrame(np.corrcoef(x, rowvar=False), index=cols, columns=cols),
        df.cov(), df.corr(),
    )


def correlation_significance(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for a, b in combinations(df.columns, 2):
        r, p = stats.pearsonr(df[a].to_numpy(float), df[b].to_numpy(float))
        rows.append({"col1": a, "col2": b, "r": r, "p_value": p, "significant": bool(p < alpha)})
    return pd.DataFrame(rows)


def distribution_tests(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Совместимая с новыми версиями SciPy реализация задания 27."""
    cols = list(df.columns)
    if len(cols) < 2:
        raise ValueError("Нужно минимум 2 столбца")

    x1 = df[cols[0]].to_numpy(dtype=np.float64)
    loc = float(np.min(x1))
    scale = float(np.max(x1) - np.min(x1))
    if scale == 0:
        uniform_statistic, uniform_pvalue = np.nan, 0.0
    else:
        res = stats.kstest(x1, lambda value: stats.uniform.cdf(value, loc=loc, scale=scale))
        uniform_statistic, uniform_pvalue = float(res.statistic), float(res.pvalue)

    x2 = df[cols[1]].to_numpy(dtype=np.float64)
    mu = float(np.mean(x2))
    sigma = float(np.std(x2, ddof=1))
    if sigma == 0:
        normal_ks_statistic, normal_ks_pvalue = np.nan, 0.0
    else:
        res = stats.kstest(x2, lambda value: stats.norm.cdf(value, loc=mu, scale=sigma))
        normal_ks_statistic, normal_ks_pvalue = float(res.statistic), float(res.pvalue)

    shapiro = stats.shapiro(x2)
    return {
        "alpha": alpha,
        "uniform_first_column": {
            "column": cols[0], "statistic": uniform_statistic,
            "p_value": uniform_pvalue, "reject_H0": bool(uniform_pvalue < alpha),
        },
        "normal_second_column": {
            "column": cols[1], "ks_statistic": normal_ks_statistic,
            "ks_p_value": normal_ks_pvalue,
            "shapiro_statistic": float(shapiro.statistic),
            "shapiro_p_value": float(shapiro.pvalue),
            "reject_H0_by_shapiro": bool(shapiro.pvalue < alpha),
        },
    }


def vector_products(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in combinations(df.columns, 2):
        x, y = df[a].to_numpy(float), df[b].to_numpy(float)
        rows.append({"col1": a, "col2": b, "dot": float(np.dot(x, y)), "cross_first_3": np.cross(x[:3], y[:3]).tolist()})
    return pd.DataFrame(rows)


def norms(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({col: {"L1": np.linalg.norm(df[col].to_numpy(float), 1), "L2": np.linalg.norm(df[col].to_numpy(float), 2)} for col in df.columns}).T
