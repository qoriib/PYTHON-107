import io
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def load_csv(uploaded_file: io.BytesIO) -> pd.DataFrame:
    """Read a CSV file-like object into a DataFrame with common encodings."""
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin-1")


def detect_likert_columns(df: pd.DataFrame) -> list[str]:
    """Return columns whose stripped name starts with 'Saya'."""
    return [col for col in df.columns if col.strip().startswith("Saya")]


def suggest_xy_split(likert_cols: list[str]) -> tuple[list[str], list[str]]:
    """Heuristic: items before the one containing 'mampu menahan diri' are X, the rest are Y."""
    x_cols, y_cols = [], []
    split_index = None
    for idx, col in enumerate(likert_cols):
        if "mampu menahan diri" in col.lower():
            split_index = idx
            break
    if split_index is None:
        return likert_cols, []
    x_cols = likert_cols[:split_index]
    y_cols = likert_cols[split_index:]
    return x_cols, y_cols


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert selected columns to numeric Likert responses."""
    converted = df.copy()
    for col in cols:
        converted[col] = pd.to_numeric(converted[col], errors="coerce")
    return converted


def descriptive_stats(series: pd.Series) -> dict:
    """Compute key descriptive stats for a numeric series."""
    clean = series.dropna()
    mode_vals = clean.mode()
    mode = mode_vals.iloc[0] if not mode_vals.empty else np.nan
    return {
        "Mean": clean.mean(),
        "Median": clean.median(),
        "Mode": mode,
        "Min": clean.min(),
        "Max": clean.max(),
        "Std Dev": clean.std(ddof=1) if len(clean) > 1 else np.nan,
        "N": len(clean),
    }


def correlation_block(x: pd.Series, y: pd.Series) -> dict:
    """Calculate Pearson and Spearman correlations."""
    mask = ~(x.isna() | y.isna())
    x_clean = x[mask]
    y_clean = y[mask]
    pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean) if len(x_clean) > 1 else (np.nan, np.nan)
    spearman_r, spearman_p = stats.spearmanr(x_clean, y_clean) if len(x_clean) > 1 else (np.nan, np.nan)
    return {
        "Pearson r": pearson_r,
        "Pearson p-value": pearson_p,
        "Spearman rho": spearman_r,
        "Spearman p-value": spearman_p,
        "N pairs": len(x_clean),
    }


def correlation_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Approximate Fisher z confidence interval for correlation."""
    if n <= 3 or math.isnan(r):
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1 / math.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lower = np.tanh(z - z_crit * se)
    upper = np.tanh(z + z_crit * se)
    return (lower, upper)


def strength_label(value: float) -> str:
    """Return qualitative strength label for a correlation coefficient."""
    if math.isnan(value):
        return "N/A"
    abs_val = abs(value)
    if abs_val < 0.2:
        return "Very weak"
    if abs_val < 0.4:
        return "Weak"
    if abs_val < 0.6:
        return "Moderate"
    if abs_val < 0.8:
        return "Strong"
    return "Very strong"


def make_hist(data: pd.Series, title: str):
    fig, ax = plt.subplots()
    ax.hist(data.dropna(), bins=10, color="#2f6ee2", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Skor")
    ax.set_ylabel("Frekuensi")
    return fig


def make_scatter(x: pd.Series, y: pd.Series, title: str):
    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=0.7, color="#e26f46")
    ax.set_title(title)
    ax.set_xlabel("X_total")
    ax.set_ylabel("Y_total")
    return fig


def download_with_totals(df: pd.DataFrame, x_total: pd.Series | None, y_total: pd.Series | None):
    enriched = df.copy()
    if x_total is not None:
        enriched["X_total"] = x_total
    if y_total is not None:
        enriched["Y_total"] = y_total
    return enriched.to_csv(index=False).encode("utf-8")
