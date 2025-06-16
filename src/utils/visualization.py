import sys
from PIL import Image
import matplotlib
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))


def setup_fonts(font_dir="fonts"):
    mongolian_font_path = project_root / "fonts" / "NotoSansMongolian-Regular.ttf"
    font_files = fm.findSystemFonts(fontpaths=[font_dir])
    for font_file in font_files:
        fm.fontManager.addfont(font_file)

    matplotlib.rcParams["font.family"] = [
        "DejaVu Sans",
        "Noto Sans",
        "Noto Sans Mongolian",
        "sans-serif",
    ]

    mongolian_font_prop = FontProperties(
        fname=mongolian_font_path,
        family=["Noto Sans Mongolian", "DejaVu Sans", "Noto Sans", "sans-serif"],
    )

    return mongolian_font_prop


def generate_image(
    image_data,
    original_manchu,
    predicted_manchu,
    original_roman,
    predicted_roman,
    inference_time,
    mongolian_font_prop,
    output_dir,
):
    # Check for required fields - allow empty roman for CRNN models
    if not image_data or not original_manchu or not predicted_manchu:
        return

    # Detect CRNN model (only predicts Manchu, roman is empty)
    is_crnn_model = not original_roman and not predicted_roman

    is_correct = original_manchu.strip() == predicted_manchu.strip()

    # Calculate length mismatch - ignore roman for CRNN models
    if is_crnn_model:
        length_mismatch = len(original_manchu) != len(predicted_manchu)
    else:
        length_mismatch = len(original_manchu) != len(predicted_manchu) or len(
            original_roman
        ) != len(predicted_roman)

    inference_time_s = inference_time / 1000.0

    fig = plt.figure(figsize=(12, 8))
    gs = plt.GridSpec(2, 1, height_ratios=[2, 3], hspace=0.15)

    ax_image = plt.subplot(gs[0])
    render_image_section(ax_image, image_data)

    ax_text = plt.subplot(gs[1])
    render_text_section(
        ax_text,
        original_manchu,
        predicted_manchu,
        original_roman,
        predicted_roman,
        is_correct,
        length_mismatch,
        inference_time_s,
        mongolian_font_prop,
        is_crnn_model,
    )

    if hasattr(image_data, "filename"):
        filename = Path(image_data.filename).name
    elif isinstance(image_data, (str, Path)):
        filename = Path(image_data).name
    else:
        filename = "generated_image"

    plt.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_image_section(ax, image_data):
    ax.axis("off")

    if isinstance(image_data, Image.Image):
        ax.imshow(np.array(image_data))
    elif isinstance(image_data, np.ndarray):
        ax.imshow(image_data)
    else:
        image_path = Path(image_data)
        img = Image.open(image_path)
        ax.imshow(np.array(img))


def render_text_section(
    ax,
    gt_manchu,
    pred_manchu,
    gt_roman,
    pred_roman,
    is_correct,
    length_mismatch,
    inference_time_s,
    mongolian_font_prop,
    is_crnn_model,
):
    ax.axis("off")

    rect = plt.Rectangle(
        (0.02, 0.02),
        0.96,
        0.96,
        fill=True,
        color="#f8f8f8",
        alpha=0.9,
        transform=ax.transAxes,
    )
    ax.add_patch(rect)

    correct_mark = "✓" if is_correct else "✗"
    correct_color = "green" if is_correct else "red"
    ax.text(
        0.5,
        0.92,
        f"Match Status: {correct_mark}",
        fontsize=14,
        fontweight="bold",
        ha="center",
        color=correct_color,
        bbox=dict(facecolor="white", alpha=0.8, pad=4, boxstyle="round,pad=0.3"),
    )

    if length_mismatch:
        if is_crnn_model:
            message = f"Length Mismatch!\nGround Truth: {len(gt_manchu)} chars (Manchu)\nPredicted: {len(pred_manchu)} chars (Manchu)"
        else:
            message = f"Length Mismatch!\nGround Truth: {len(gt_manchu)} chars (Manchu), {len(gt_roman)} chars (Roman)\nPredicted: {len(pred_manchu)} chars (Manchu), {len(pred_roman)} chars (Roman)"

        ax.text(
            0.5,
            0.18,
            message,
            fontsize=11,
            ha="center",
            va="center",
            color="red",
            fontweight="bold",
            bbox=dict(facecolor="#ffe6e6", alpha=0.8, pad=8, boxstyle="round,pad=0.4"),
        )

    ax.axvline(x=0.5, ymin=0.1, ymax=0.75, color="lightgray", linewidth=1)

    render_column(
        ax,
        "Ground Truth",
        gt_manchu,
        gt_roman,
        0.05,
        mongolian_font_prop,
        "#e8f4f8",
        length_mismatch,
        None,
        is_crnn_model,
    )

    comparison_data = None if length_mismatch else (gt_manchu, gt_roman)
    render_column(
        ax,
        "Predicted",
        pred_manchu,
        pred_roman,
        0.52,
        mongolian_font_prop,
        "#f0f8e8" if is_correct else "#f8e8e8",
        length_mismatch,
        comparison_data,
        is_crnn_model,
    )

    ax.text(
        0.95,
        0.05,
        f"Inference: {inference_time_s:.2f}s",
        fontsize=9,
        ha="right",
        va="bottom",
        color="#666666",
        family="monospace",
        alpha=0.8,
    )


def render_column(
    ax,
    title,
    manchu_text,
    roman_text,
    x_pos,
    mongolian_font_prop,
    background_color,
    length_mismatch,
    comparison_data,
    is_crnn_model,
):
    if background_color:
        rect = plt.Rectangle(
            (x_pos - 0.02, 0.15),
            0.46,
            0.6,
            fill=True,
            color=background_color,
            alpha=0.3,
            transform=ax.transAxes,
            zorder=0,
        )
        ax.add_patch(rect)

    y_pos = 0.72
    is_predicted = "Predicted" in title

    ax.text(x_pos, y_pos, title, fontsize=14, fontweight="bold", color="#333333")
    y_pos -= 0.12

    ax.text(x_pos, y_pos, "Manchu:", fontsize=11, fontweight="bold", color="#666666")
    y_pos -= 0.08

    if length_mismatch:
        ax.text(
            x_pos + 0.02,
            y_pos,
            manchu_text,
            fontsize=13,
            ha="left",
            fontproperties=mongolian_font_prop,
            color="red",
            wrap=True,
        )
    elif comparison_data and is_predicted:
        render_highlighted_text(
            ax,
            x_pos + 0.02,
            y_pos,
            manchu_text,
            comparison_data[0],
            mongolian_font_prop,
            13,
        )
    else:
        ax.text(
            x_pos + 0.02,
            y_pos,
            manchu_text,
            fontsize=13,
            ha="left",
            fontproperties=mongolian_font_prop,
            color="#000000",
            wrap=True,
        )

    y_pos -= 0.12

    # Only display Roman text for non-CRNN models
    if not is_crnn_model:
        ax.text(x_pos, y_pos, "Roman:", fontsize=11, fontweight="bold", color="#666666")
        y_pos -= 0.08

        if length_mismatch:
            ax.text(
                x_pos + 0.02,
                y_pos,
                roman_text,
                fontsize=12,
                ha="left",
                color="red",
                wrap=True,
            )
        elif comparison_data and is_predicted:
            render_highlighted_text(
                ax, x_pos + 0.02, y_pos, roman_text, comparison_data[1], None, 12
            )
        else:
            ax.text(
                x_pos + 0.02,
                y_pos,
                roman_text,
                fontsize=12,
                ha="left",
                color="#000000",
                wrap=True,
            )


def render_highlighted_text(
    ax, x_pos, y_pos, predicted_text, ground_truth_text, font_prop, fontsize
):
    min_length = min(len(predicted_text), len(ground_truth_text))
    current_x = x_pos

    i = 0
    while i < len(predicted_text):
        start_i = i
        is_correct = i < min_length and predicted_text[i] == ground_truth_text[i]

        while i < len(predicted_text) and (
            (i >= min_length and not is_correct)
            or (
                i < min_length
                and (predicted_text[i] == ground_truth_text[i]) == is_correct
            )
        ):
            i += 1

        segment_text = predicted_text[start_i:i]
        color = "black" if is_correct and start_i < min_length else "red"
        weight = "normal" if color == "black" else "bold"

        text_obj = ax.text(
            current_x,
            y_pos,
            segment_text,
            fontsize=fontsize,
            ha="left",
            fontproperties=font_prop,
            color=color,
            fontweight=weight,
        )

        if i < len(predicted_text):
            bbox = text_obj.get_window_extent(renderer=ax.figure.canvas.get_renderer())
            bbox_axes = bbox.transformed(ax.transAxes.inverted())
            current_x += bbox_axes.width
