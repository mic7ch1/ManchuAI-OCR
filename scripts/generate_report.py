import datetime
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


from src.utils.config import ConfigLoader
from src.utils.visualization import setup_fonts, generate_image


def main(target_models=None):
    config_loader = ConfigLoader()
    models_config_list = config_loader.get_config("models")

    if target_models:
        models_config_list = [
            m for m in models_config_list if m["name"] in target_models
        ]

    evaluation_output = project_root / "results"

    for model_config in models_config_list:
        model_name = model_config["name"]

        if not model_config:
            print(f"Warning: Model {model_name} not found in models.yaml. Skipping.")
            continue

        model_test_examples_output = (
            project_root / "results" / "test_examples" / model_name
        )

        test_examples_output = model_test_examples_output / "samples"
        test_summary_output = model_test_examples_output
        test_examples_output.mkdir(parents=True, exist_ok=True)

        try:
            test_file = str(evaluation_output / "test" / model_name) + ".json"
            metrics_file = (
                str(evaluation_output / "metrics" / model_name) + "_test.json"
            )

            with open(test_file, "r") as f:
                test_results = json.load(f)

            with open(metrics_file, "r") as f:
                test_metrics = json.load(f)

            generate_test_examples(test_results, test_examples_output)
            generate_test_summary(test_metrics, test_summary_output)
            print(
                f"Finished generating test examples and summary for {model_name}. Find the results in {test_summary_output}."
            )
        except FileNotFoundError as e:
            print(f"Warning: File not found for {model_name}: {e}. Skipping.")
            continue
        except Exception as e:
            print(f"Error processing {model_name}: {e}. Skipping.")
            continue


def generate_test_examples(test_results, test_examples_output):
    mongolian_font_prop = setup_fonts()  # Setup once, use for all images

    for result in test_results:
        generate_image(
            result["image_path"],
            result["manchu_gt"],
            result["manchu_pred"],
            result["roman_gt"],
            result["roman_pred"],
            result["inference_time"],
            mongolian_font_prop,
            test_examples_output,
        )


def generate_test_summary(test_metrics, test_summary_output):
    title = "Manchu OCR Test Results"
    report = f"""{title}
{'=' * len(title)}

Total samples evaluated: {test_metrics['total_predictions']}

Manchu Word Accuracy: {test_metrics['manchu_word_accuracy']:.2f}%
Manchu CER: {test_metrics['manchu_cer']:.4f}
Manchu F1 Score: {test_metrics['manchu_f1_score']:.4f}

Roman Word Accuracy: {test_metrics['roman_word_accuracy']:.2f}%
Roman CER: {test_metrics['roman_cer']:.4f}
Roman F1 Score: {test_metrics['roman_f1_score']:.4f}

Inference Time: {test_metrics['inference_time']:.2f}ms

Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    output_path = str(test_summary_output) + "/report.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main(target_models=["qwen-25-3b", "qwen-25-7b", "llama-32-11b", "crnn-base-3m"])
