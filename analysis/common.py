from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl


ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "csv"
DATA_DIR = ROOT / "data-0515"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

MODELS = [
    "Claude Opus 4.6",
    "Claude Sonnet 4",
    "DeepSeek V3.2",
    "DeepSeek R1",
]
NEWER_MODELS = [MODELS[0], MODELS[2]]
OLDER_MODELS = [MODELS[1], MODELS[3]]
FAMILY_PAIRS = {
    "Claude": (MODELS[0], MODELS[1]),
    "DeepSeek": (MODELS[2], MODELS[3]),
}
MODEL_DATA_DIRS = {
    MODELS[0]: DATA_DIR / "claude_out_default",
    MODELS[1]: DATA_DIR / "sonnet4_out_default",
    MODELS[2]: DATA_DIR / "ds_out_default",
    MODELS[3]: DATA_DIR / "r1_out_default",
}

SEED = 20260713
N_BOOT = 20_000
N_PERM = 100_000

BODY_FONT_SIZE = 10
SINGLE_COLUMN_WIDTH = 3.5
DOUBLE_COLUMN_WIDTH = 7.16


def configure_plot_style() -> None:
    """Match figure text to the 10 pt IEEEtran body text at final size."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": BODY_FONT_SIZE,
        "axes.titlesize": BODY_FONT_SIZE,
        "axes.labelsize": BODY_FONT_SIZE,
        "xtick.labelsize": BODY_FONT_SIZE,
        "ytick.labelsize": BODY_FONT_SIZE,
        "legend.fontsize": BODY_FONT_SIZE,
        "figure.titlesize": BODY_FONT_SIZE,
    })


def ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "codeql": pd.read_csv(CSV_DIR / "codeql_per_instance.csv"),
        "codescene": pd.read_csv(CSV_DIR / "codescene_per_instance.csv", na_values=["nan"]),
        "dynamic": pd.read_csv(CSV_DIR / "dynamic_perf.csv"),
        "dynamic_runs": pd.read_csv(CSV_DIR / "dynamic_perf_runs.csv"),
        "patch": pd.read_csv(CSV_DIR / "patch_metrics.csv"),
        "rules": pd.read_csv(CSV_DIR / "rule_catalog.csv"),
    }


def resolved_sets(codeql: pd.DataFrame) -> dict[str, set[str]]:
    return {
        model: set(codeql.loc[codeql["model"].eq(model), "instance_id"])
        for model in MODELS
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def paired_wilcoxon_permutation(
    newer: np.ndarray,
    older: np.ndarray,
    seed: int = SEED,
    n_perm: int = N_PERM,
) -> dict[str, float | int]:
    """Two-sided signed-rank randomization test with deterministic Monte Carlo."""
    diff = np.asarray(newer, dtype=float) - np.asarray(older, dtype=float)
    diff = diff[np.isfinite(diff)]
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return {"p_value": 1.0, "rank_biserial": 0.0, "n_nonzero": 0}

    ranks = average_ranks(np.abs(nonzero))
    observed = abs(float(np.sum(np.sign(nonzero) * ranks)))
    denominator = float(np.sum(ranks))
    effect = float(np.sum(np.sign(nonzero) * ranks) / denominator)

    if len(nonzero) <= 20:
        masks = np.arange(1 << len(nonzero), dtype=np.uint64)[:, None]
        bits = ((masks >> np.arange(len(nonzero), dtype=np.uint64)) & 1).astype(float)
        signs = 2.0 * bits - 1.0
        stats = np.abs(signs @ ranks)
        p_value = float(np.mean(stats >= observed - 1e-12))
    else:
        rng = np.random.default_rng(seed)
        extreme = 0
        done = 0
        batch = 5_000
        while done < n_perm:
            size = min(batch, n_perm - done)
            signs = rng.choice((-1.0, 1.0), size=(size, len(nonzero)))
            stats = np.abs(signs @ ranks)
            extreme += int(np.sum(stats >= observed - 1e-12))
            done += size
        p_value = (extreme + 1.0) / (n_perm + 1.0)

    return {
        "p_value": p_value,
        "rank_biserial": effect,
        "n_nonzero": int(len(nonzero)),
    }


def paired_bootstrap_ci(
    newer: np.ndarray,
    older: np.ndarray,
    statistic: str = "median",
    seed: int = SEED,
    n_boot: int = N_BOOT,
) -> tuple[float, float, float]:
    diff = np.asarray(newer, dtype=float) - np.asarray(older, dtype=float)
    diff = diff[np.isfinite(diff)]
    if statistic == "median":
        point = float(np.median(diff))
        reducer = np.median
    elif statistic == "mean":
        point = float(np.mean(diff))
        reducer = np.mean
    else:
        raise ValueError(f"Unsupported statistic: {statistic}")
    if len(diff) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    for start in range(0, n_boot, 1_000):
        size = min(1_000, n_boot - start)
        indices = rng.integers(0, len(diff), size=(size, len(diff)))
        values[start : start + size] = reducer(diff[indices], axis=1)
    low, high = np.quantile(values, [0.025, 0.975])
    return point, float(low), float(high)


def paired_summary(newer: pd.Series, older: pd.Series, seed_offset: int = 0) -> dict[str, float | int]:
    frame = pd.concat([newer.rename("newer"), older.rename("older")], axis=1).dropna()
    new = frame["newer"].to_numpy(dtype=float)
    old = frame["older"].to_numpy(dtype=float)
    test = paired_wilcoxon_permutation(new, old, seed=SEED + seed_offset)
    point, low, high = paired_bootstrap_ci(new, old, seed=SEED + seed_offset)
    return {
        "n": int(len(frame)),
        "newer_median": float(np.median(new)),
        "older_median": float(np.median(old)),
        "median_paired_difference": point,
        "ci_2.5": low,
        "ci_97.5": high,
        "wins_newer_lower": int(np.sum(new < old)),
        "ties": int(np.sum(new == old)),
        "losses_newer_higher": int(np.sum(new > old)),
        **test,
    }


def holm_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    n = len(values)
    for rank, idx in enumerate(order):
        candidate = (n - rank) * values[idx]
        running = max(running, candidate)
        adjusted[idx] = min(1.0, running)
    return pd.Series(adjusted, index=p_values.index)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean(x[:, None] > y[None, :]) - np.mean(x[:, None] < y[None, :]))


def permutation_median_test(
    x: np.ndarray,
    y: np.ndarray,
    seed: int = SEED,
    n_perm: int = N_PERM,
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    observed = abs(float(np.median(x) - np.median(y)))
    pooled = np.concatenate([x, y])
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(pooled)
        stat = abs(float(np.median(shuffled[: len(x)]) - np.median(shuffled[len(x) :])))
        extreme += stat >= observed - 1e-12
    return float((extreme + 1) / (n_perm + 1))


def mcnemar_exact(newer_present: np.ndarray, older_present: np.ndarray) -> tuple[int, int, float]:
    new = np.asarray(newer_present, dtype=bool)
    old = np.asarray(older_present, dtype=bool)
    newer_only = int(np.sum(new & ~old))
    older_only = int(np.sum(~new & old))
    n = newer_only + older_only
    if n == 0:
        return newer_only, older_only, 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(newer_only, older_only) + 1)) / (2**n)
    return newer_only, older_only, min(1.0, 2.0 * tail)


def bootstrap_rate_difference(
    newer_present: np.ndarray,
    older_present: np.ndarray,
    seed: int = SEED,
    n_boot: int = N_BOOT,
) -> tuple[float, float, float]:
    new = np.asarray(newer_present, dtype=float)
    old = np.asarray(older_present, dtype=float)
    diff = new - old
    point = float(np.mean(diff))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = np.mean(diff[indices], axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return point, float(low), float(high)


def repository_name(instance_id: str) -> str:
    return instance_id.split("__", 1)[0]


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def write_summary(path: Path, title: str, lines: list[str]) -> None:
    path.write_text("\n".join([title, "=" * len(title), *lines, ""]) + "\n")
