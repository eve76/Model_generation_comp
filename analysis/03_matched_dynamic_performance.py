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
    paired_bootstrap_ci,
    paired_summary,
    resolved_sets,
    write_summary,
    configure_plot_style,
)


METRICS = {
    "CPU time (s)": "cpu_mean",
    "Peak memory (MiB)": "memory_mean",
}


def main() -> None:
    configure_plot_style()
    ensure_output_dirs()
    inputs = load_inputs()
    dynamic = inputs["dynamic"]
    sets = resolved_sets(inputs["codeql"])
    all_four = set.intersection(*(sets[model] for model in MODELS))
    rows = []
    pair_rows = []
    differences: dict[tuple[str, str], np.ndarray] = {}
    offset = 20

    for comparison, pair in [*FAMILY_PAIRS.items(), ("All four", ("newer average", "older average"))]:
        for metric_name, metric in METRICS.items():
            if comparison == "All four":
                wide = dynamic[dynamic["instance_id"].isin(all_four) & dynamic["model"].isin(MODELS)].pivot(
                    index="instance_id", columns="model", values=metric
                )
                newer = wide[[MODELS[0], MODELS[2]]].mean(axis=1)
                older = wide[[MODELS[1], MODELS[3]]].mean(axis=1)
            else:
                new_model, old_model = pair
                ids = sets[new_model] & sets[old_model]
                wide = dynamic[dynamic["instance_id"].isin(ids) & dynamic["model"].isin(pair)].pivot(
                    index="instance_id", columns="model", values=metric
                )
                newer, older = wide[new_model], wide[old_model]

            aligned = pd.concat([newer.rename("newer"), older.rename("older")], axis=1).dropna()
            result = paired_summary(aligned["newer"], aligned["older"], seed_offset=offset)
            log_new = np.log(aligned["newer"])
            log_old = np.log(aligned["older"])
            log_point, log_low, log_high = paired_bootstrap_ci(
                log_new.to_numpy(), log_old.to_numpy(), seed=20260713 + offset
            )
            result.update({
                "median_ratio_newer_over_older": float(np.exp(log_point)),
                "ratio_ci_2.5": float(np.exp(log_low)),
                "ratio_ci_97.5": float(np.exp(log_high)),
            })
            rows.append({"comparison": comparison, "metric": metric_name, **result})
            differences[(comparison, metric_name)] = (aligned["newer"] - aligned["older"]).to_numpy()
            for instance_id, values in aligned.iterrows():
                pair_rows.append({
                    "comparison": comparison,
                    "metric": metric_name,
                    "instance_id": instance_id,
                    "newer": values["newer"],
                    "older": values["older"],
                    "difference_newer_minus_older": values["newer"] - values["older"],
                    "ratio_newer_over_older": values["newer"] / values["older"],
                })
            offset += 1

    results = pd.DataFrame(rows)
    results["holm_p_across_dynamic_tests"] = holm_adjust(results["p_value"])
    results.to_csv(RESULTS_DIR / "matched_dynamic_performance.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(RESULTS_DIR / "matched_dynamic_pairs.csv", index=False)

    variability = dynamic[dynamic["model"].isin(MODELS)].copy()
    variability["cpu_cv"] = variability["cpu_std"] / variability["cpu_mean"]
    variability["memory_cv"] = variability["memory_std"] / variability["memory_mean"]
    variability_summary = variability.groupby("model", as_index=False).agg(
        n=("instance_id", "size"),
        median_cpu_cv=("cpu_cv", "median"),
        p90_cpu_cv=("cpu_cv", lambda x: x.quantile(0.9)),
        median_memory_cv=("memory_cv", "median"),
        p90_memory_cv=("memory_cv", lambda x: x.quantile(0.9)),
    )
    variability_summary.to_csv(RESULTS_DIR / "dynamic_run_variability.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH, 3.25))
    comparisons = ["Claude", "DeepSeek", "All four"]
    colors = ["#2f6b8a", "#4f8a5b", "#b45f4d"]
    for ax, metric_name in zip(axes, METRICS):
        data = [differences[(comparison, metric_name)] for comparison in comparisons]
        box = ax.boxplot(data, labels=comparisons, patch_artist=True, showfliers=False)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax.axhline(0, color="#333333", linewidth=0.9)
        ax.set_ylabel("Newer - older")
        ax.set_title(metric_name, loc="center")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_matched_dynamic_performance.png", dpi=300)
    plt.close(fig)

    lines = [
        "Each instance value is the mean over five harness runs.",
        "All comparisons are paired on identical resolved instances.",
    ]
    for row in rows:
        lines.append(
            f"{row['comparison']} | {row['metric']} | n={row['n']} | "
            f"median diff={row['median_paired_difference']:.4g} | "
            f"median ratio={row['median_ratio_newer_over_older']:.4g} | "
            f"p={row['p_value']:.4g} | r_rb={row['rank_biserial']:.4g}"
        )
    write_summary(RESULTS_DIR / "03_matched_dynamic_summary.txt", "Matched dynamic-performance analysis", lines)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
