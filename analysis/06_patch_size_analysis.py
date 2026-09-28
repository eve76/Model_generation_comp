from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    DOUBLE_COLUMN_WIDTH,
    FAMILY_PAIRS,
    FIGURES_DIR,
    RESULTS_DIR,
    ensure_output_dirs,
    holm_adjust,
    load_inputs,
    paired_summary,
    resolved_sets,
    write_summary,
    configure_plot_style,
)


PATCH_METRICS = {
    "Changed lines": "patch_churn",
    "Files touched": "patch_files_touched",
    "Patch hunks": "patch_hunks",
}


def spearman_permutation(x: pd.Series, y: pd.Series, seed: int, n_perm: int = 20_000):
    frame = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    xr = frame["x"].rank(method="average").to_numpy()
    yr = frame["y"].rank(method="average").to_numpy()
    observed = float(np.corrcoef(xr, yr)[0, 1]) if len(frame) > 2 else np.nan
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        permuted = rng.permutation(yr)
        value = float(np.corrcoef(xr, permuted)[0, 1])
        extreme += abs(value) >= abs(observed) - 1e-12
    return len(frame), observed, (extreme + 1) / (n_perm + 1)


def main() -> None:
    configure_plot_style()
    ensure_output_dirs()
    inputs = load_inputs()
    sets = resolved_sets(inputs["codeql"])
    patch = inputs["patch"].copy()
    patch["patch_churn"] = patch["patch_lines_added"] + patch["patch_lines_removed"]

    pair_results = []
    pair_values = []
    offset = 200
    for family, (newer, older) in FAMILY_PAIRS.items():
        ids = sets[newer] & sets[older]
        for metric_name, metric in PATCH_METRICS.items():
            wide = patch[patch["instance_id"].isin(ids) & patch["model"].isin([newer, older])].pivot(
                index="instance_id", columns="model", values=metric
            )
            result = paired_summary(wide[newer], wide[older], seed_offset=offset)
            pair_results.append({"family": family, "metric": metric_name, **result})
            aligned = wide[[newer, older]].dropna()
            for instance_id, values in aligned.iterrows():
                pair_values.append({
                    "family": family,
                    "metric": metric_name,
                    "instance_id": instance_id,
                    "newer": values[newer],
                    "older": values[older],
                    "difference_newer_minus_older": values[newer] - values[older],
                })
            offset += 1
    pair_results = pd.DataFrame(pair_results)
    pair_results["holm_p"] = holm_adjust(pair_results["p_value"])
    pair_results.to_csv(RESULTS_DIR / "matched_patch_size.csv", index=False)
    pd.DataFrame(pair_values).to_csv(RESULTS_DIR / "matched_patch_size_pairs.csv", index=False)

    merged = (
        inputs["codeql"]
        .merge(inputs["codescene"][["model", "instance_id", "n_introduced_findings", "n_degraded_details"]],
               on=["model", "instance_id"], how="left")
        .merge(patch[["model", "instance_id", "patch_churn", "patch_files_touched"]],
               on=["model", "instance_id"], how="left")
    )
    merged["codescene_degradations"] = (
        merged["n_introduced_findings"].fillna(0) + merged["n_degraded_details"].fillna(0)
    )
    merged["codeql_added_per_100_lines"] = 100 * merged["added"] / merged["patch_churn"].clip(lower=1)
    merged["codescene_degradations_per_100_lines"] = (
        100 * merged["codescene_degradations"] / merged["patch_churn"].clip(lower=1)
    )
    merged.to_csv(RESULTS_DIR / "patch_size_quality_merged.csv", index=False)

    associations = []
    targets = {
        "CodeQL added findings": "added",
        "CodeQL net change": "net",
        "CodeScene degradations": "codescene_degradations",
    }
    offset = 300
    for model, model_df in merged.groupby("model"):
        for target_name, target in targets.items():
            n, rho, p_value = spearman_permutation(
                model_df["patch_churn"], model_df[target], seed=20260713 + offset
            )
            associations.append({
                "model": model,
                "association": f"patch churn vs {target_name}",
                "n": n,
                "spearman_rho": rho,
                "permutation_p": p_value,
            })
            offset += 1
    associations = pd.DataFrame(associations)
    associations["holm_p"] = holm_adjust(associations["permutation_p"])
    associations.to_csv(RESULTS_DIR / "patch_size_quality_associations.csv", index=False)

    normalized = merged.groupby("model", as_index=False).agg(
        n=("instance_id", "size"),
        median_patch_churn=("patch_churn", "median"),
        median_codeql_added_per_100_lines=("codeql_added_per_100_lines", "median"),
        median_codescene_degradations_per_100_lines=("codescene_degradations_per_100_lines", "median"),
    )
    normalized.to_csv(RESULTS_DIR / "patch_size_normalized_descriptive.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH, 3.6))
    for ax, (target, label) in zip(
        axes,
        [("added", "CodeQL added findings"), ("codescene_degradations", "CodeScene degradations")],
    ):
        for model, model_df in merged.groupby("model"):
            ax.scatter(model_df["patch_churn"], model_df[target], s=12, alpha=0.45, label=model)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel("Changed lines")
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_patch_size_associations.png", dpi=300)
    plt.close(fig)

    lines = [
        "Patch churn is lines added plus lines removed.",
        "Matched patch-size tests use the same family-specific common resolved sets as the quality analysis.",
        "Associations use deterministic Spearman permutation tests and are exploratory.",
    ]
    for _, row in pair_results.iterrows():
        lines.append(
            f"{row['family']} | {row['metric']} | n={row['n']} | median diff={row['median_paired_difference']:.3g} | "
            f"p_adj={row['holm_p']:.4g}"
        )
    for _, row in associations[associations["holm_p"] < 0.05].iterrows():
        lines.append(
            f"Association | {row['model']} | {row['association']} | rho={row['spearman_rho']:.3f} | "
            f"p_adj={row['holm_p']:.4g}"
        )
    write_summary(RESULTS_DIR / "06_patch_size_summary.txt", "Patch-size analysis", lines)
    print(pair_results.to_string(index=False))
    print(associations.to_string(index=False))


if __name__ == "__main__":
    main()
