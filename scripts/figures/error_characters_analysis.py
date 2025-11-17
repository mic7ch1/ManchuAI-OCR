from pathlib import Path
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

# Add project root to sys.path
root = Path(__file__).resolve().parents[2]
sys.path.append(str(root))


def load_error_rates(json_path, char_list):
    with open(json_path) as f:
        data = json.load(f)
    total = data.get("total_predictions", 1)
    errors = data.get("frequent_error", {}).get("manchu", {})
    return [errors.get(c, 0) / total for c, _ in char_list]


def create_heatmap():
    # 1 ──────────── DATA
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

    val_models = [
        ("llama-32-11b_validation", "LLaMA-3.2-11B"),
        ("qwen-25-7b_validation", "Qwen-2.5-7B"),
        ("qwen-25-3b_validation", "Qwen-2.5-3B"),
    ]

    test_models = [
        ("llama-32-11b_test", "LLaMA-3.2-11B"),
        ("qwen-25-7b_test", "Qwen-2.5-7B"),
        ("qwen-25-3b_test", "Qwen-2.5-3B"),
    ]

    metrics_dir = root / "results" / "metrics"

    # full matrix (chars × models)
    rates = np.array(
        [
            load_error_rates(metrics_dir / f"{key}.json", char_pairs)
            for key, _ in val_models + test_models
        ]
    ).T

    # sort characters by overall error (high → low)
    order = np.argsort(rates.sum(1))[::-1]
    char_pairs = [char_pairs[i] for i in order]
    rates = rates[order]

    val_data = rates[:, : len(val_models)]
    test_data = rates[:, len(val_models) :]

    # global colour scale maximum
    z_max = rates.max()

    # 2 ──────────── FIGURE WITH TWO SUBPLOTS
    vspace = 0.01
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=vspace)

    # trace 1: validation
    fig.add_trace(
        go.Heatmap(
            z=val_data.T,
            x=[c for c, _ in char_pairs],
            y=[m for _, m in val_models],
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

    # trace 2: test
    fig.add_trace(
        go.Heatmap(
            z=test_data.T,
            x=[c for c, _ in char_pairs],
            y=[m for _, m in test_models],
            colorscale="Viridis_r",
            zmin=0,
            zmax=z_max,
            showscale=False,
            hovertemplate="Char: %{x}<br>Model: %{y}<br>Error: %{z:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # 3 ──────────── LAYOUT POLISH
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

    # Bottom (row 2) x-axis ticks
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

    # Top (row 1) x-axis ticks (no labels)
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

    # roman transliteration row above the top subplot
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

    # rotated dataset labels between heatmap and colorbar
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

    # divider line between the two heatmaps (black)
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

    # 4 ──────────── EXPORT
    out_dir = root / "results" / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "error_characters_heatmap"

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
