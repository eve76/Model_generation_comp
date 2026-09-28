from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from common import (
    DOUBLE_COLUMN_WIDTH,
    FIGURES_DIR,
    configure_plot_style,
    ensure_output_dirs,
    load_inputs,
)


MODEL_ORDER = [
    "Claude Opus 4.6",
    "Claude Sonnet 4",
    "DeepSeek V3.2",
    "DeepSeek R1",
    "Gold (developer)",
]
DISPLAY_NAMES = {
    "Claude Opus 4.6": "Claude\nOpus 4.6",
    "Claude Sonnet 4": "Claude\nSonnet 4",
    "DeepSeek V3.2": "DeepSeek\nV3.2",
    "DeepSeek R1": "DeepSeek\nR1",
    "Gold (developer)": "Gold",
}
COLORS = ["#d7301f", "#bdbdbd", "#1a9850"]


def codeql_direction(net: float) -> str:
    if net > 0:
        return "Regress"
    if net < 0:
        return "Improve"
    return "Neutral"


def codescene_direction(row: pd.Series) -> str:
    delta = row["score_delta_min"]
    if row["is_empty_diff"] == 1 or pd.isna(delta) or delta == 0:
        return "Neutral"
    return "Regress" if delta < 0 else "Improve"


def direction_share(frame: pd.DataFrame) -> pd.DataFrame:
    shares = frame.groupby("model")["direction"].value_counts(normalize=True).unstack(fill_value=0)
    return shares.reindex(MODEL_ORDER).fillna(0).reindex(
        columns=["Regress", "Neutral", "Improve"], fill_value=0
    ) * 100


def main() -> None:
    configure_plot_style()
    ensure_output_dirs()
    inputs = load_inputs()

    codeql = inputs["codeql"].copy()
    codeql["direction"] = codeql["net"].map(codeql_direction)
    codescene = inputs["codescene"].copy()
    codescene["direction"] = codescene.apply(codescene_direction, axis=1)

    ql_share = direction_share(codeql)
    cs_share = direction_share(codescene)

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH, 3.0), sharey=True)
    for ax, share, title in (
        (axes[0], ql_share, "(a) CodeQL"),
        (axes[1], cs_share, "(b) CodeScene"),
    ):
        plotted = share.rename(index=DISPLAY_NAMES)
        plotted.plot(kind="bar", stacked=True, ax=ax, color=COLORS, width=0.72, legend=False)
        ax.set_ylim(0, 108)
        ax.set_xlabel("")
        ax.set_title(title, loc="center")
        ax.tick_params(axis="x", rotation=0)
        for index, model in enumerate(share.index):
            regress, _, improve = share.loc[model]
            if regress > 0.5:
                ax.text(index, regress / 2, f"{regress:.1f}%", ha="center", va="center")
            ax.text(index, 101.5, f"{improve:.1f}%", ha="center", va="bottom")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Resolved patches (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in COLORS]
    fig.legend(
        handles,
        ["Regress", "Neutral", "Improve"],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(FIGURES_DIR / "fig_overall_direction_share.png", dpi=300)
    plt.close(fig)

    print("CodeQL direction share (%):")
    print(ql_share.round(1).to_string())
    print("\nCodeScene direction share (%):")
    print(cs_share.round(1).to_string())


if __name__ == "__main__":
    main()
