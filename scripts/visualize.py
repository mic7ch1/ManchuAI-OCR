#!/usr/bin/env python3
"""
Visualization orchestrator for ManchuAI-OCR.

This script runs all visualization scripts from scripts/figures/ directory.
Each visualization can be run individually or all together.
"""

from pathlib import Path
import sys
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import visualization functions from scripts/figures
from scripts.figures.utils import load_metrics_data, load_detailed_results
from scripts.figures.accuracy_bar_plot import create_accuracy_bar_plot
from scripts.figures.cer_bar_plot import create_cer_bar_plot
from scripts.figures.f1_bar_plot import create_f1_bar_plot
from scripts.figures.inference_time_comparison import create_inference_time_comparison
from scripts.figures.performance_comparison_chart import (
    create_performance_comparison_chart,
)
from scripts.figures.checkpoint_trends import eval_checkpoints_by_wa
from scripts.figures.word_length_analysis import (
    analyze_word_length_performance_by_model,
)
from scripts.figures.comparison_vlm_vs_crnn import (
    create_vlm_vs_crnn_chart,
    compute_metrics_data,
)
from scripts.figures.error_characters_analysis import create_heatmap
from scripts.figures.model_training import plot_model_training
from scripts.figures.training_performance import plot_training_performance


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def run_all_visualizations(output_dir="results/paper/figures", metrics_dir="results/metrics"):
    """Run all visualization scripts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(metrics_dir)

    print_section("ManchuAI-OCR Visualization Suite")
    print(f"Output directory: {output_path}")
    print(f"Metrics directory: {metrics_path}\n")

    # Load shared data once for efficiency
    print("Loading shared data...")
    try:
        metrics_data = load_metrics_data(metrics_dir)
        print(f"  ✓ Loaded metrics for {len(metrics_data)} models")
    except Exception as e:
        print(f"  ✗ Failed to load metrics data: {e}")
        metrics_data = {}

    try:
        detailed_results = load_detailed_results("results/test")
        print(f"  ✓ Loaded detailed test results for {len(detailed_results)} models")
    except Exception as e:
        print(f"  ✗ Failed to load detailed results: {e}")
        detailed_results = {}

    # Track results
    results = {"success": [], "failed": [], "skipped": []}

    # 1. Accuracy comparison
    print_section("1/11: Accuracy Bar Plot")
    try:
        create_accuracy_bar_plot(metrics_data, output_path)
        results["success"].append("Accuracy comparison")
        print("  ✓ Saved accuracy_comparison.png/pdf")
    except Exception as e:
        results["failed"].append(("Accuracy comparison", str(e)))
        print(f"  ✗ Failed: {e}")

    # 2. CER comparison
    print_section("2/11: Character Error Rate Comparison")
    try:
        create_cer_bar_plot(metrics_data, output_path)
        results["success"].append("CER comparison")
        print("  ✓ Saved cer_comparison.png/pdf")
    except Exception as e:
        results["failed"].append(("CER comparison", str(e)))
        print(f"  ✗ Failed: {e}")

    # 3. F1 score comparison
    print_section("3/11: F1 Score Comparison")
    try:
        create_f1_bar_plot(metrics_data, output_path)
        results["success"].append("F1 score comparison")
        print("  ✓ Saved f1_comparison.png/pdf")
    except Exception as e:
        results["failed"].append(("F1 score comparison", str(e)))
        print(f"  ✗ Failed: {e}")

    # 4. Inference time comparison
    print_section("4/11: Inference Time Comparison")
    try:
        create_inference_time_comparison(metrics_data, output_path)
        results["success"].append("Inference time comparison")
        print("  ✓ Saved inference_time_comparison.png/pdf")
    except Exception as e:
        results["failed"].append(("Inference time comparison", str(e)))
        print(f"  ✗ Failed: {e}")

    # 5. Performance comparison chart
    print_section("5/11: Performance Comparison Chart (VLM)")
    try:
        create_performance_comparison_chart(metrics_data, output_path)
        results["success"].append("Performance comparison chart")
        print("  ✓ Saved performance_comparison.png/pdf")
    except Exception as e:
        results["failed"].append(("Performance comparison chart", str(e)))
        print(f"  ✗ Failed: {e}")

    # 6. Checkpoint evaluation trends
    print_section("6/11: Checkpoint Evaluation Trends")
    try:
        eval_checkpoints_by_wa(metrics_path, output_path)
        results["success"].append("Checkpoint trends")
        print("  ✓ Saved checkpoint_trends.png/pdf")
    except Exception as e:
        results["failed"].append(("Checkpoint trends", str(e)))
        print(f"  ✗ Failed: {e}")

    # 7. Word length analysis
    print_section("7/11: Word Length Performance Analysis")
    try:
        if detailed_results:
            analyze_word_length_performance_by_model(detailed_results, output_path)
            results["success"].append("Word length analysis")
            print("  ✓ Saved word_length_analysis_by_model.png/pdf")
        else:
            results["skipped"].append("Word length analysis (no detailed results)")
            print("  ⊘ Skipped: No detailed results available")
    except Exception as e:
        results["failed"].append(("Word length analysis", str(e)))
        print(f"  ✗ Failed: {e}")

    # 8. VLM vs CRNN comparison
    print_section("8/11: VLM vs CRNN Comparison")
    try:
        models = ["llama-32-11b", "crnn-base-3m"]
        splits = ["validation", "test"]
        vlm_crnn_metrics = compute_metrics_data(models, splits)
        create_vlm_vs_crnn_chart(vlm_crnn_metrics, output_path)
        results["success"].append("VLM vs CRNN comparison")
        print("  ✓ Saved comparison_vlm_vs_crnn.png/pdf")
    except Exception as e:
        results["failed"].append(("VLM vs CRNN comparison", str(e)))
        print(f"  ✗ Failed: {e}")

    # 9. Error characters heatmap (uses Plotly)
    print_section("9/11: Character Error Rate Heatmap")
    try:
        create_heatmap()
        results["success"].append("Error characters heatmap")
        print("  ✓ Saved error_characters_heatmap.png/pdf")
    except Exception as e:
        results["failed"].append(("Error characters heatmap", str(e)))
        print(f"  ✗ Failed: {e}")
        if "kaleido" in str(e).lower():
            print("  ℹ  Tip: Install kaleido for Plotly export: pip install kaleido")

    # 10. Model training visualization
    print_section("10/11: Model Training Curves")
    try:
        plot_model_training(metrics_path, output_path)
        results["success"].append("Model training curves")
        print("  ✓ Saved model_training.png/pdf")
    except Exception as e:
        results["failed"].append(("Model training curves", str(e)))
        print(f"  ✗ Failed: {e}")

    # 11. Training performance/stability
    print_section("11/11: Training Stability (Gradient Norms)")
    try:
        plot_training_performance(metrics_path, output_path)
        results["success"].append("Training stability")
        print("  ✓ Saved training_performance.png/pdf")
    except Exception as e:
        results["failed"].append(("Training stability", str(e)))
        print(f"  ✗ Failed: {e}")

    # Summary
    print_section("Summary")
    print(f"✓ Successful: {len(results['success'])}")
    for item in results["success"]:
        print(f"    - {item}")

    if results["skipped"]:
        print(f"\n⊘ Skipped: {len(results['skipped'])}")
        for item in results["skipped"]:
            print(f"    - {item}")

    if results["failed"]:
        print(f"\n✗ Failed: {len(results['failed'])}")
        for item, error in results["failed"]:
            print(f"    - {item}: {error}")

    print(f"\n{'=' * 60}")
    print(f"All figures saved to: {output_path.absolute()}")
    print('=' * 60)

    return len(results["failed"]) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate all visualizations for ManchuAI-OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/visualize.py                    # Run all visualizations
  python scripts/visualize.py --output ./figs    # Custom output directory
  python scripts/visualize.py --metrics ./data   # Custom metrics directory
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        default="results/paper/figures",
        help="Output directory for figures (default: results/paper/figures)",
    )

    parser.add_argument(
        "--metrics",
        "-m",
        default="results/metrics",
        help="Metrics directory (default: results/metrics)",
    )

    args = parser.parse_args()

    success = run_all_visualizations(
        output_dir=args.output, metrics_dir=args.metrics
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
