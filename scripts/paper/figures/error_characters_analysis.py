from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.paper.utils import PROJECT_ROOT, FIGURES_OUTPUT_DIR, load_json


def load_error_rates(json_path, char_list):
    data = load_json(json_path, {})
    total = data.get("total_predictions", 1)
    errors = data.get("frequent_error", {}).get("manchu", {})
    return [errors.get(c, 0) / total for c, _ in char_list]


def create_heatmap():
    char_pairs = [
        ("ᠠ", "A"),
        ("ᡝ", "E"),
        ("ᡳ", "I"),
        ("ᠣ", "O"),
        ("ᡠ", "U"),
        ("ᡡ", "Ū"),
        ("ᠨ", "N"),
        ("ᠩ", "NG"),
        ("ᡴ", "K"),
        ("ᡤ", "G"),
        ("ᡥ", "H"),
        ("ᠪ", "B"),
        ("ᡦ", "P"),
        ("ᠰ", "S"),
        ("ᡧ", "Š"),
        ("ᡨ", "T"),
        ("ᡩ", "D"),
        ("ᠯ", "L"),
        ("ᠮ", "M"),
        ("ᠴ", "C"),
        ("ᠵ", "J"),
        ("ᠶ", "Y"),
        ("ᡵ", "R"),
        ("ᡶ", "F"),
        ("ᠸ", "W"),
        ("ᡯ", "Z"),
    ]

    models = [
        ("llama-32-11b", "LLaMA-3.2-11B"),
        ("qwen-25-7b", "Qwen-2.5-7B"),
        ("qwen-25-3b", "Qwen-2.5-3B"),
    ]

    metrics_dir = PROJECT_ROOT / "results" / "metrics"

    val_rates = [
        load_error_rates(metrics_dir / model / "best_checkpoint" / "validation.json", char_pairs)
        for model, _ in models
    ]
    test_rates = [
        load_error_rates(metrics_dir / model / "best_checkpoint" / "test.json", char_pairs)
        for model, _ in models
    ]
    rates = np.array(val_rates + test_rates).T

    order = np.argsort(rates.sum(1))[::-1]
    char_pairs = [char_pairs[i] for i in order]
    rates = rates[order]

    val_data = rates[:, : len(models)]
    test_data = rates[:, len(models) :]

    z_max = rates.max()

    vspace = 0.01
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=vspace)

    fig.add_trace(
        go.Heatmap(
            z=val_data.T,
            x=[c for c, _ in char_pairs],
            y=[m for _, m in models],
            colorscale="Viridis_r",
            zmin=0,
            zmax=z_max,
            colorbar=dict(
                title=dict(text="Error Rate", side="right"),
                tick0=0,
                dtick=0.1,
                tickformat=".1f",
                len=1.0,
                y=0.5,
                yanchor="middle",
                ticks="outside",
                ticklen=5,
                tickwidth=1.5,
            ),
            hovertemplate="Char: %{x}<br>Model: %{y}<br>Error: %{z:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=test_data.T,
            x=[c for c, _ in char_pairs],
            y=[m for _, m in models],
            colorscale="Viridis_r",
            zmin=0,
            zmax=z_max,
            showscale=False,
            hovertemplate="Char: %{x}<br>Model: %{y}<br>Error: %{z:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        width=1000,
        height=500,
        margin=dict(l=125, r=80, t=90, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(
            text="Character Error Rates across Models",
            x=0.5,
            y=1.0,
            xanchor="center",
            yanchor="top",
            font=dict(
                size=22,
                family="Times New Roman, Times, serif",
                color="black",
                weight="bold",
            ),
        ),
        font=dict(family="Times New Roman, Times, serif", color="black"),
    )

    fig.update_xaxes(
        tickfont=dict(size=18, family="Noto Sans Mongolian, Arial Unicode MS"),
        ticks="outside",
        ticklen=5,
        tickwidth=1.5,
        showline=True,
        linewidth=1.5,
        linecolor="black",
        showgrid=False,
        row=2,
        col=1,
    )

    fig.update_xaxes(
        ticks="outside",
        ticklen=5,
        tickwidth=1.5,
        showline=True,
        linewidth=1.5,
        linecolor="black",
        side="top",
        showticklabels=False,
        showgrid=False,
        row=1,
        col=1,
    )

    fig.update_yaxes(
        tickfont=dict(size=14, family="Times New Roman, Times, serif"),
        ticks="outside",
        ticklen=5,
        tickwidth=1.5,
        showline=True,
        linewidth=1.5,
        linecolor="black",
    )

    for i, (_, roman) in enumerate(char_pairs):
        fig.add_annotation(
            x=i,
            y=1.035,
            text=roman,
            showarrow=False,
            xref="x1",
            yref="paper",
            xanchor="center",
            yanchor="bottom",
            font=dict(size=16, family="Times New Roman, Times, serif"),
        )

    h = (1 - vspace) / 2
    val_center = 1 - h / 2
    test_center = h / 2

    for y_pos, label in [(val_center, "Validation"), (test_center, "Test")]:
        fig.add_annotation(
            x=-0.15,
            y=y_pos,
            xref="paper",
            yref="paper",
            text=label,
            textangle=-90,
            showarrow=False,
            font=dict(size=18, family="Times New Roman, Times, serif"),
            xanchor="center",
            yanchor="middle",
        )

    boundary_y = h + vspace / 2  # midpoint of the gap
    fig.add_shape(
        type="line",
        x0=0,
        x1=1.0,
        y0=boundary_y,
        y1=boundary_y,
        xref="paper",
        yref="paper",
        line=dict(color="black", width=2),
    )

    FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_OUTPUT_DIR / "08_error_characters_heatmap"

    try:
        fig.write_image(f"{out}.png", width=1000, height=500, scale=2)
        fig.write_image(f"{out}.pdf", width=1000, height=500)
    except Exception as e:
        print("Static export failed – install kaleido:", e)

    print("Saved:", f"{out}.png and {out}.pdf")
    return fig


def main():
    create_heatmap()


if __name__ == "__main__":
    main()
