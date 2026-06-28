"""
Generate LaTeX table for software environment specifications.
"""

from pathlib import Path
import platform
import sys

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from scripts.paper.utils import TABLES_OUTPUT_DIR, write_latex_table


def get_software_versions():
    """Get software versions from the current environment."""
    versions = {}

    try:
        import distro

        versions["Operating System"] = f"{distro.name()} {distro.version()}"
    except ImportError:
        versions["Operating System"] = platform.platform()

    try:
        import torch

        if torch.cuda.is_available():
            versions["CUDA"] = torch.version.cuda
        else:
            versions["CUDA"] = "N/A"
    except ImportError:
        versions["CUDA"] = "N/A"

    versions["Python"] = f"{sys.version_info.major}.{sys.version_info.minor}"

    try:
        import torch

        versions["PyTorch"] = ".".join(torch.__version__.split(".")[:2])
    except ImportError:
        versions["PyTorch"] = "N/A"

    try:
        import transformers

        versions["Transformers"] = ".".join(transformers.__version__.split(".")[:2])
    except ImportError:
        versions["Transformers"] = "N/A"

    try:
        import datasets

        versions["Datasets"] = ".".join(datasets.__version__.split(".")[:2])
    except ImportError:
        versions["Datasets"] = "N/A"

    try:
        import unsloth

        versions["Unsloth"] = unsloth.__version__
    except ImportError:
        versions["Unsloth"] = "N/A"

    return versions


def create_software_environment_table(output_dir):
    """Generate LaTeX table for software environment specifications."""
    versions = get_software_versions()

    rows = []
    for component, version in versions.items():
        rows.append(f"{component} & {version} \\\\")

    rows_str = "\n".join(rows)

    latex_content = rf"""\begin{{table}}[H]
\TBL{{\caption{{Software environment specifications.}}\label{{tab:software_specs}}}}{{\centering
\begin{{tabular}}{{l@{{\hspace{{1em}}}}l}}
\toprule
\textbf{{Component}} & \textbf{{Version}} \\
\midrule
{rows_str}
\botrule%
\end{{tabular}}
}}
\end{{table}}
"""

    write_latex_table(
        output_dir,
        "06_software_environment.tex",
        latex_content,
        "software environment table",
    )


def main():
    TABLES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    create_software_environment_table(TABLES_OUTPUT_DIR)


if __name__ == "__main__":
    main()
