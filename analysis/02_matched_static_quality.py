from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    DOUBLE_COLUMN_WIDTH,
    FAMILY_PAIRS,
    FIGURES_DIR,
    MODELS,
    RESULTS_DIR,
    ensure_output_dirs,
    holm_adjust,
    load_inputs,
    paired_summary,
    resolved_sets,
    write_summary,
    configure_plot_style,
)


METRICS = {
    "CodeQL net change": ("codeql", "net"),
    "CodeScene Code Health delta": ("codescene", "score_delta_min"),
}


def matched_vectors(df: pd.DataFrame, metric: str, ids: set[str], newer: str, older: str):
    wide = (
        df[df["instance_id"].isin(ids) & df["model"].isin([newer, older])]
        .pivot(index="instance_id", columns="model", values=metric)
        .reindex(columns=[newer, older])
    )
    return wide[newer], wide[older], wide


def main() -> None:
    configure_plot_style()
    ensure_output_dirs()
    inputs = load_inputs()
    inputs["codescene"]["score_delta_min"] = inputs["codescene"]["score_delta_min"].fillna(0.0)
    sets = resolved_sets(inputs["codeql"])
    all_four = set.intersection(*(sets[model] for model in MODELS))

    rows = []
    paired_rows = []
    differences: dict[tuple[str, str], np.ndarray] = {}
    offset = 0
    for comparison, pair in [*FAMILY_PAIRS.items(), ("All four", ("newer average", "older average"))]:
        for metric_name, (source, metric) in METRICS.items():
            df = inputs[source]
            if comparison == "All four":
                wide = df[df["instance_id"].isin(all_four) & df["model"].isin(MODELS)].pivot(
                    index="instance_id", columns="model", values=metric
                )
                newer = wide[[MODELS[0], MODELS[2]]].mean(axis=1)
                older = wide[[MODELS[1], MODELS[3]]].mean(axis=1)
            else:
                new_model, old_model = pair
                ids = sets[new_model] & sets[old_model]
                newer, older, wide = matched_vectors(df, metric, ids, new_model, old_model)
            result = paired_summary(newer, older, seed_offset=offset)
            row = {"comparison": comparison, "metric": metric_name, **result}
            rows.append(row)
            aligned = pd.concat([newer.rename("newer"), older.rename("older")], axis=1).dropna()
            differences[(comparison, metric_name)] = (aligned["newer"] - aligned["older"]).to_numpy()
            for instance_id, values in aligned.iterrows():
                paired_rows.append({
                    "comparison": comparison,
                    "metric": metric_name,
                    "instance_id": instance_id,
                    "newer": values["newer"],
                    "older": values["older"],
                    "difference_newer_minus_older": values["newer"] - values["older"],
                })
            offset += 1

    results = pd.DataFrame(rows)
    results["holm_p_across_static_tests"] = holm_adjust(results["p_value"])
    results.to_csv(RESULTS_DIR / "matched_static_quality.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(RESULTS_DIR / "matched_static_quality_pairs.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(DOUBLE_COLUMN_WIDTH, 5.8), sharex=False)
    comparisons = ["Claude", "DeepSeek", "All four"]
    colors = ["#2f6b8a", "#4f8a5b", "#b45f4d"]
    for ax, metric_name in zip(axes, METRICS):
        data = [differences[(comparison, metric_name)] for comparison in comparisons]
        parts = ax.violinplot(data, positions=np.arange(3), showmedians=True, showextrema=False)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.55)
        ax.axhline(0, color="#333333", linewidth=0.9)
        ax.set_xticks(np.arange(3), comparisons)
        ax.set_ylabel("Newer - older")
        ax.set_title(metric_name)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_matched_static_quality.png", dpi=300)
    plt.close(fig)

    lines = [
        "CodeScene missing deltas are treated as neutral (0), matching the manuscript definition.",
        "All-four comparisons average the two newer and two older values within each instance before testing.",
    ]
    for row in rows:
        lines.append(
            f"{row['comparison']} | {row['metric']} | n={row['n']} | "
            f"median diff={row['median_paired_difference']:.4g} | "
            f"p={row['p_value']:.4g} | r_rb={row['rank_biserial']:.4g}"
        )
    write_summary(RESULTS_DIR / "02_matched_static_summary.txt", "Matched static-quality analysis", lines)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
