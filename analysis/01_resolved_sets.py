from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from common import (
    SINGLE_COLUMN_WIDTH,
    FAMILY_PAIRS,
    FIGURES_DIR,
    MODELS,
    RESULTS_DIR,
    ensure_output_dirs,
    load_inputs,
    resolved_sets,
    configure_plot_style,
    write_summary,
)


def main() -> None:
    configure_plot_style()
    ensure_output_dirs()
    codeql = load_inputs()["codeql"]
    sets = resolved_sets(codeql)
    all_ids = sorted(set().union(*sets.values()))

    membership = pd.DataFrame({"instance_id": all_ids})
    for model in MODELS:
        membership[model] = membership["instance_id"].isin(sets[model]).astype(int)
    membership["n_models_resolved"] = membership[MODELS].sum(axis=1)
    membership.to_csv(RESULTS_DIR / "resolved_set_membership.csv", index=False)

    rows = [
        {"set": model, "n_instances": len(sets[model])}
        for model in MODELS
    ]
    for family, (newer, older) in FAMILY_PAIRS.items():
        rows.append({"set": f"{family} common", "n_instances": len(sets[newer] & sets[older])})
        rows.append({"set": f"{family} newer only", "n_instances": len(sets[newer] - sets[older])})
        rows.append({"set": f"{family} older only", "n_instances": len(sets[older] - sets[newer])})
    all_four = set.intersection(*(sets[model] for model in MODELS))
    rows.append({"set": "All four common", "n_instances": len(all_four)})
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "resolved_overlap_summary.csv", index=False)

    fig_rows = summary[summary["set"].isin([
        *MODELS,
        "Claude common",
        "DeepSeek common",
        "All four common",
    ])]
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.55))
    colors = ["#2f6b8a", "#8a8f98", "#2f6b8a", "#8a8f98", "#4f8a5b", "#4f8a5b", "#b45f4d"]
    bars = ax.barh(fig_rows["set"], fig_rows["n_instances"], color=colors)
    ax.bar_label(bars, padding=3)
    ax.set_xlabel("Resolved instances")
    ax.set_xlim(0, max(fig_rows["n_instances"]) * 1.16)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_resolved_overlap.png", dpi=300)
    plt.close(fig)

    write_summary(
        RESULTS_DIR / "01_resolved_sets_summary.txt",
        "Resolved-set analysis",
        [
            *(f"{row['set']}: {row['n_instances']}" for row in rows),
            "Inputs: csv/codeql_per_instance.csv",
            "Membership output: resolved_set_membership.csv",
        ],
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
