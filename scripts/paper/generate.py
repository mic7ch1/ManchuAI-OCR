from pathlib import Path
import sys
import argparse

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.paper.utils import (
    FIGURES_OUTPUT_DIR,
    TABLES_OUTPUT_DIR,
    load_metrics_data,
)

from scripts.paper.figures.performance_comparison_chart import (
    create_performance_comparison_chart,
)
from scripts.paper.figures.checkpoint_trends import eval_checkpoints_by_wa
from scripts.paper.figures.comparison_vlm_vs_crnn import (
    create_vlm_vs_crnn_chart,
    compute_metrics_data,
)
from scripts.paper.figures.error_characters_analysis import create_heatmap
from scripts.paper.figures.training_performance import plot_training_performance

from scripts.paper.tables.performance_table_manchu import main as generate_performance_table_manchu
from scripts.paper.tables.performance_table_roman import main as generate_performance_table_roman
from scripts.paper.tables.error_table import main as generate_error_table
from scripts.paper.tables.hyperparameters_table import main as generate_hyperparameters_table
from scripts.paper.tables.software_environment_table import main as generate_software_environment_table
from scripts.paper.tables.crnn_architecture_table import main as generate_crnn_architecture_table
from scripts.paper.tables.crnn_training_table import main as generate_crnn_training_table
from src.evaluation.utils import print_header


def run_all_outputs():
    """Generate all figures and tables."""
    FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_header("ManchuAI-OCR Output Generator")
    print(f"Figures directory: {FIGURES_OUTPUT_DIR}")
    print(f"Tables directory: {TABLES_OUTPUT_DIR}\n")

    print("Loading shared data...")
    try:
        metrics_data = load_metrics_data("results/metrics")
        print(f"  Loaded metrics for {len(metrics_data)} models")
    except Exception as e:
        print(f"  Failed to load metrics data: {e}")
        metrics_data = {}

    results = {"success": [], "failed": [], "skipped": []}

    print_header("FIGURES")

    print("\n[1/5] Performance Comparison Chart (VLM)")
    try:
        create_performance_comparison_chart(metrics_data, FIGURES_OUTPUT_DIR)
        results["success"].append("Performance comparison chart")
        print("  Saved performance_comparison.png/pdf")
    except Exception as e:
        results["failed"].append(("Performance comparison chart", str(e)))
        print(f"  Failed: {e}")

    print("\n[2/5] Checkpoint Evaluation Trends")
    try:
        eval_checkpoints_by_wa("results/metrics", FIGURES_OUTPUT_DIR)
        results["success"].append("Checkpoint trends")
        print("  Saved 05a_checkpoint_trends.png/pdf")
    except Exception as e:
        results["failed"].append(("Checkpoint trends", str(e)))
        print(f"  Failed: {e}")

    print("\n[3/5] VLM vs CRNN Comparison")
    try:
        models = ["llama-32-11b", "crnn-base-3m"]
        splits = ["validation", "test"]
        vlm_crnn_metrics = compute_metrics_data(models, splits)
        create_vlm_vs_crnn_chart(vlm_crnn_metrics, FIGURES_OUTPUT_DIR)
        results["success"].append("VLM vs CRNN comparison")
        print("  Saved 06_comparison_vlm_vs_crnn.png/pdf")
    except Exception as e:
        results["failed"].append(("VLM vs CRNN comparison", str(e)))
        print(f"  Failed: {e}")

    print("\n[4/5] Character Error Rate Heatmap")
    try:
        create_heatmap()
        results["success"].append("Error characters heatmap")
        print("  Saved 07_error_characters_heatmap.png/pdf")
    except Exception as e:
        results["failed"].append(("Error characters heatmap", str(e)))
        print(f"  Failed: {e}")
        if "kaleido" in str(e).lower():
            print("  Tip: Install kaleido for Plotly export: pip install kaleido")

    print("\n[5/5] Training Stability (Gradient Norms)")
    try:
        plot_training_performance("models/VLM", FIGURES_OUTPUT_DIR)
        results["success"].append("Training stability")
        print("  Saved 05b_training_performance.png/pdf")
    except Exception as e:
        results["failed"].append(("Training stability", str(e)))
        print(f"  Failed: {e}")

    print_header("TABLES")

    print("\n[1/4] Hyperparameters Table")
    try:
        generate_hyperparameters_table()
        results["success"].append("Hyperparameters table")
        print("  Saved 01_hyperparameters_table.tex")
    except Exception as e:
        results["failed"].append(("Hyperparameters table", str(e)))
        print(f"  Failed: {e}")

    print("\n[2/7] Manchu Performance Comparison Table")
    try:
        generate_performance_table_manchu()
        results["success"].append("Manchu performance comparison table")
        print("  Saved 02_performance_comparison_manchu.tex")
    except Exception as e:
        results["failed"].append(("Manchu performance comparison table", str(e)))
        print(f"  Failed: {e}")

    print("\n[3/7] Roman Performance Comparison Table")
    try:
        generate_performance_table_roman()
        results["success"].append("Roman performance comparison table")
        print("  Saved 03_performance_comparison_roman.tex")
    except Exception as e:
        results["failed"].append(("Roman performance comparison table", str(e)))
        print(f"  Failed: {e}")

    print("\n[4/7] Error Analysis Table")
    try:
        generate_error_table()
        results["success"].append("Error analysis table")
        print("  Saved 04_error_analysis.tex")
    except Exception as e:
        results["failed"].append(("Error analysis table", str(e)))
        print(f"  Failed: {e}")

    print("\n[5/8] Software Environment Table")
    try:
        generate_software_environment_table()
        results["success"].append("Software environment table")
        print("  Saved 06_software_environment.tex")
    except Exception as e:
        results["failed"].append(("Software environment table", str(e)))
        print(f"  Failed: {e}")

    print("\n[6/8] Inference Settings Table")
    try:
        from scripts.paper.tables.inference_settings_table import main as generate_inference_settings_table
        generate_inference_settings_table()
        results["success"].append("Inference settings table")
        print("  Saved 07_inference_settings.tex")
    except Exception as e:
        results["failed"].append(("Inference settings table", str(e)))
        print(f"  Failed: {e}")

    print("\n[7/8] CRNN Architecture Table")
    try:
        generate_crnn_architecture_table()
        results["success"].append("CRNN architecture table")
        print("  Saved 08_crnn_architecture.tex")
    except Exception as e:
        results["failed"].append(("CRNN architecture table", str(e)))
        print(f"  Failed: {e}")

    print("\n[8/8] CRNN Training Table")
    try:
        generate_crnn_training_table()
        results["success"].append("CRNN training table")
        print("  Saved 09_crnn_training.tex")
    except Exception as e:
        results["failed"].append(("CRNN training table", str(e)))
        print(f"  Failed: {e}")

    print_header("Summary")
    print(f"Successful: {len(results['success'])}")
    for item in results["success"]:
        print(f"  - {item}")

    if results["skipped"]:
        print(f"\nSkipped: {len(results['skipped'])}")
        for item in results["skipped"]:
            print(f"  - {item}")

    if results["failed"]:
        print(f"\nFailed: {len(results['failed'])}")
        for item, error in results["failed"]:
            print(f"  - {item}: {error}")

    print_header(
        f"Figures saved to: {FIGURES_OUTPUT_DIR.absolute()}\nTables saved to: {TABLES_OUTPUT_DIR.absolute()}"
    )

    return len(results["failed"]) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate all figures and tables for ManchuAI-OCR paper",
    )
    args = parser.parse_args()

    success = run_all_outputs()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
