from __future__ import annotations

from collections import Counter
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    DOUBLE_COLUMN_WIDTH,
    FAMILY_PAIRS,
    FIGURES_DIR,
    MODEL_DATA_DIRS,
    MODELS,
    RESULTS_DIR,
    bootstrap_rate_difference,
    ensure_output_dirs,
    holm_adjust,
    load_inputs,
    mcnemar_exact,
    read_json,
    resolved_sets,
    write_summary,
    configure_plot_style,
)


def codeql_patterns(model: str, instance_ids: set[str], catalog: dict[str, str]) -> pd.DataFrame:
    rows = []
    root = MODEL_DATA_DIRS[model] / "codeql_analysis"
    for instance_id in sorted(instance_ids):
        path = root / f"{instance_id}.diff.json"
        if not path.exists():
            continue
        data = read_json(path)
        counts = Counter(item.get("ruleId", "<unknown>") for item in data.get("added_results", []))
        for rule_id, count in counts.items():
            rows.append({
                "model": model,
                "instance_id": instance_id,
                "pattern": rule_id,
                "category": catalog.get(rule_id, "other"),
                "count": count,
                "present": 1,
            })
    return pd.DataFrame(rows)


def codescene_patterns(model: str, instance_ids: set[str]) -> pd.DataFrame:
    rows = []
    root = MODEL_DATA_DIRS[model] / "codescene_analysis"
    for instance_id in sorted(instance_ids):
        path = root / f"{instance_id}.diff.json"
        if not path.exists():
            continue
        counts = Counter()
        for file_result in read_json(path):
            for finding in file_result.get("findings", []):
                category = finding.get("category", "<unknown>")
                degraded = finding.get("change-type") == "introduced"
                degraded = degraded or any(
                    detail.get("change-type") == "degraded"
                    for detail in finding.get("change-details", [])
                )
                if degraded:
                    counts[category] += 1
        for category, count in counts.items():
            rows.append({
                "model": model,
                "instance_id": instance_id,
                "pattern": category,
                "category": category,
                "count": count,
                "present": 1,
            })
    return pd.DataFrame(rows)


def paired_pattern_stats(
    long_df: pd.DataFrame,
    family: str,
    newer: str,
    older: str,
    ids: set[str],
    tool: str,
) -> pd.DataFrame:
    patterns = sorted(long_df["pattern"].unique()) if not long_df.empty else []
    rows = []
    for index, pattern in enumerate(patterns):
        subset = long_df[long_df["pattern"].eq(pattern)]
        presence = subset.pivot_table(
            index="instance_id", columns="model", values="present", aggfunc="max", fill_value=0
        ).reindex(index=sorted(ids), columns=[newer, older], fill_value=0)
        new = presence[newer].to_numpy(dtype=int)
        old = presence[older].to_numpy(dtype=int)
        newer_only, older_only, p_value = mcnemar_exact(new, old)
        diff, low, high = bootstrap_rate_difference(new, old, seed=20260713 + index)
        category_values = subset.loc[subset["pattern"].eq(pattern), "category"]
        rows.append({
            "tool": tool,
            "family": family,
            "pattern": pattern,
            "category": category_values.iloc[0] if len(category_values) else "",
            "n_instances": len(ids),
            "newer_patch_prevalence": float(np.mean(new)),
            "older_patch_prevalence": float(np.mean(old)),
            "rate_difference": diff,
            "ci_2.5": low,
            "ci_97.5": high,
            "newer_only_discordant": newer_only,
            "older_only_discordant": older_only,
            "mcnemar_p": p_value,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["holm_p_within_tool_family"] = holm_adjust(result["mcnemar_p"])
    return result


def all_four_prevalence(long_df: pd.DataFrame, ids: set[str], tool: str) -> pd.DataFrame:
    rows = []
    for pattern in sorted(long_df["pattern"].unique()):
        subset = long_df[long_df["pattern"].eq(pattern)]
        for model in MODELS:
            present_ids = set(subset.loc[subset["model"].eq(model), "instance_id"])
            rows.append({
                "tool": tool,
                "pattern": pattern,
                "model": model,
                "n_instances": len(ids),
                "patch_prevalence": len(present_ids & ids) / len(ids),
            })
    return pd.DataFrame(rows)


def main() -> None:
    configure_plot_style()
    ensure_output_dirs()
    inputs = load_inputs()
    sets = resolved_sets(inputs["codeql"])
    catalog = dict(zip(inputs["rules"]["ruleId"], inputs["rules"]["category"]))
    all_four = set.intersection(*(sets[model] for model in MODELS))

    codeql_frames = []
    codescene_frames = []
    for model in MODELS:
        codeql_frames.append(codeql_patterns(model, sets[model], catalog))
        codescene_frames.append(codescene_patterns(model, sets[model]))
    codeql_long = pd.concat(codeql_frames, ignore_index=True)
    codescene_long = pd.concat(codescene_frames, ignore_index=True)
    codeql_long.to_csv(RESULTS_DIR / "codeql_pattern_presence_long.csv", index=False)
    codescene_long.to_csv(RESULTS_DIR / "codescene_pattern_presence_long.csv", index=False)

    stats_frames = []
    for family, (newer, older) in FAMILY_PAIRS.items():
        ids = sets[newer] & sets[older]
        stats_frames.append(paired_pattern_stats(codeql_long, family, newer, older, ids, "CodeQL"))
        stats_frames.append(paired_pattern_stats(codescene_long, family, newer, older, ids, "CodeScene"))
    stats = pd.concat(stats_frames, ignore_index=True)
    stats.to_csv(RESULTS_DIR / "matched_issue_pattern_statistics.csv", index=False)

    all_four_table = pd.concat([
        all_four_prevalence(codeql_long, all_four, "CodeQL"),
        all_four_prevalence(codescene_long, all_four, "CodeScene"),
    ], ignore_index=True)
    all_four_table.to_csv(RESULTS_DIR / "all_four_issue_pattern_prevalence.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_WIDTH, 5.4), constrained_layout=True)
    for row_index, tool in enumerate(["CodeQL", "CodeScene"]):
        for col_index, family in enumerate(FAMILY_PAIRS):
            ax = axes[row_index, col_index]
            subset = stats[(stats["tool"].eq(tool)) & (stats["family"].eq(family))].copy()
            subset = subset.reindex(subset["rate_difference"].abs().sort_values(ascending=False).index).head(8)
            subset = subset.sort_values("rate_difference")
            colors = np.where(subset["rate_difference"] >= 0, "#b45f4d", "#2f6b8a")
            labels = subset["pattern"].str.replace("py/", "", regex=False).map(
                lambda value: "\n".join(textwrap.wrap(value, width=27, break_long_words=False))
            )
            ax.barh(labels, subset["rate_difference"], color=colors)
            ax.axvline(0, color="#333333", linewidth=0.8)
            ax.set_title(f"{family}: {tool}")
            ax.set_xlabel("Prevalence difference")
            ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGURES_DIR / "fig_matched_issue_patterns.png", dpi=300)
    plt.close(fig)

    lines = [
        "The unit is patch-level presence, not the raw number of findings.",
        "CodeScene presence includes introduced findings or degraded details.",
        "McNemar exact tests are Holm-corrected within each tool and family.",
    ]
    for tool in ["CodeQL", "CodeScene"]:
        for family in FAMILY_PAIRS:
            top = stats[(stats["tool"].eq(tool)) & (stats["family"].eq(family))].copy()
            top = top.reindex(top["rate_difference"].abs().sort_values(ascending=False).index).head(3)
            for _, row in top.iterrows():
                lines.append(
                    f"{family} | {tool} | {row['pattern']} | rate diff={row['rate_difference']:.3f} | "
                    f"p_adj={row['holm_p_within_tool_family']:.4g}"
                )
    write_summary(RESULTS_DIR / "04_matched_patterns_summary.txt", "Matched issue-pattern analysis", lines)
    print(stats.sort_values(["tool", "family", "holm_p_within_tool_family"]).head(30).to_string(index=False))


if __name__ == "__main__":
    main()
