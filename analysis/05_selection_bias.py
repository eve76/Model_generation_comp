from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    DOUBLE_COLUMN_WIDTH,
    FAMILY_PAIRS,
    FIGURES_DIR,
    RESULTS_DIR,
    cliffs_delta,
    ensure_output_dirs,
    holm_adjust,
    load_inputs,
    permutation_median_test,
    repository_name,
    resolved_sets,
    write_summary,
    configure_plot_style,
)


FEATURES = {
    "Gold CPU time (s)": "gold_cpu",
    "Gold peak memory (MiB)": "gold_memory",
    "Gold patch churn": "gold_churn",
    "Gold files touched": "gold_files",
}


def main() -> None:
    configure_plot_style()
    ensure_output_dirs()
    inputs = load_inputs()
    sets = resolved_sets(inputs["codeql"])

    gold_dynamic = inputs["dynamic"][inputs["dynamic"]["model"].eq("Gold (developer)")][
        ["instance_id", "cpu_mean", "memory_mean"]
    ].rename(columns={"cpu_mean": "gold_cpu", "memory_mean": "gold_memory"})
    gold_patch = inputs["patch"][inputs["patch"]["model"].eq("Gold (developer)")].copy()
    gold_patch["gold_churn"] = gold_patch["patch_lines_added"] + gold_patch["patch_lines_removed"]
    gold_patch = gold_patch[["instance_id", "gold_churn", "patch_files_touched"]].rename(
        columns={"patch_files_touched": "gold_files"}
    )

    membership_rows = []
    for family, (newer, older) in FAMILY_PAIRS.items():
        universe = sets[newer] | sets[older]
        for instance_id in sorted(universe):
            if instance_id in sets[newer] and instance_id in sets[older]:
                group = "common"
            elif instance_id in sets[newer]:
                group = "newer_only"
            else:
                group = "older_only"
            membership_rows.append({
                "family": family,
                "instance_id": instance_id,
                "selection_group": group,
                "repository": repository_name(instance_id),
            })
    membership = pd.DataFrame(membership_rows).merge(gold_dynamic, on="instance_id", how="left").merge(
        gold_patch, on="instance_id", how="left"
    )
    membership.to_csv(RESULTS_DIR / "selection_group_instance_features.csv", index=False)

    descriptive = membership.groupby(["family", "selection_group"], as_index=False).agg(
        n_instances=("instance_id", "size"),
        gold_cpu_n=("gold_cpu", "count"),
        gold_cpu_median=("gold_cpu", "median"),
        gold_memory_median=("gold_memory", "median"),
        gold_churn_median=("gold_churn", "median"),
        gold_files_median=("gold_files", "median"),
    )
    descriptive.to_csv(RESULTS_DIR / "selection_group_descriptive.csv", index=False)

    repository_table = membership.groupby(
        ["family", "selection_group", "repository"], as_index=False
    ).size().rename(columns={"size": "n_instances"})
    repository_table.to_csv(RESULTS_DIR / "selection_group_repository_distribution.csv", index=False)

    tests = []
    offset = 100
    for family in FAMILY_PAIRS:
        family_df = membership[membership["family"].eq(family)]
        common = family_df[family_df["selection_group"].eq("common")]
        for comparison_group in ["newer_only", "older_only"]:
            other = family_df[family_df["selection_group"].eq(comparison_group)]
            if other.empty:
                continue
            for feature_name, feature in FEATURES.items():
                x = common[feature].dropna().to_numpy()
                y = other[feature].dropna().to_numpy()
                if len(x) < 2 or len(y) < 2:
                    continue
                tests.append({
                    "family": family,
                    "comparison": f"common vs {comparison_group}",
                    "feature": feature_name,
                    "n_common": len(x),
                    "n_other": len(y),
                    "common_median": float(np.median(x)),
                    "other_median": float(np.median(y)),
                    "cliffs_delta_common_vs_other": cliffs_delta(x, y),
                    "permutation_p": permutation_median_test(x, y, seed=20260713 + offset, n_perm=20_000),
                })
                offset += 1
    tests = pd.DataFrame(tests)
    tests["holm_p"] = holm_adjust(tests["permutation_p"])
    tests.to_csv(RESULTS_DIR / "selection_bias_tests.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH, 3.8))
    plot_features = [("gold_cpu", "Gold CPU time (s)"), ("gold_churn", "Gold patch churn")]
    positions = []
    labels = []
    for ax, (feature, label) in zip(axes, plot_features):
        data = []
        positions.clear()
        labels.clear()
        pos = 1
        for family in FAMILY_PAIRS:
            for group in ["common", "newer_only", "older_only"]:
                values = membership.loc[
                    membership["family"].eq(family) & membership["selection_group"].eq(group), feature
                ].dropna().to_numpy()
                if len(values):
                    data.append(values)
                    positions.append(pos)
                    labels.append(f"{family}\n{group.replace('_', ' ')}")
                    pos += 1
            pos += 0.5
        box = ax.boxplot(data, positions=positions, labels=labels, patch_artist=True, showfliers=False)
        for patch in box["boxes"]:
            patch.set_facecolor("#4f8a5b")
            patch.set_alpha(0.55)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", labelrotation=30)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_selection_bias.png", dpi=300)
    plt.close(fig)

    lines = [
        "Groups are defined separately within each model family.",
        "Gold-patch resource and size measures are used as task-level proxies.",
        "Permutation tests compare medians; Holm correction covers all reported feature tests.",
    ]
    for _, row in tests.iterrows():
        lines.append(
            f"{row['family']} | {row['comparison']} | {row['feature']} | "
            f"medians={row['common_median']:.3g}/{row['other_median']:.3g} | "
            f"delta={row['cliffs_delta_common_vs_other']:.3f} | p_adj={row['holm_p']:.4g}"
        )
    write_summary(RESULTS_DIR / "05_selection_bias_summary.txt", "Resolved-set selection analysis", lines)
    print(descriptive.to_string(index=False))
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
