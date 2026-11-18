
"""Generate model comparison graphs for the PathSnap report."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def load_results(path, required_columns):
    """Load and validate actual experiment results."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist yet. "
            "Run the model evaluations first."
        )

    df = pd.read_csv(path)

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    if df["Model"].duplicated().any():
        raise ValueError(f"{path}: duplicate model names")

    return df.set_index("Model")


def save_bar(values, title, ylabel, path, ymax=None):
    """Save a single model comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(values.index, values.values)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Machine Learning Model")

    if ymax is not None:
        ax.set_ylim(0, ymax)

    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

    print(f"Saved: {path}")


def save_grouped_bar(df, columns, labels, title, ylabel,
                     path, ymax=None):
    """Compare multiple performance metrics across models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    positions = list(range(len(df)))
    width = 0.35

    for i, (column, label) in enumerate(zip(columns, labels)):
        offsets = [
            x + (i - (len(columns) - 1) / 2) * width
            for x in positions
        ]

        ax.bar(
            offsets,
            df[column],
            width=width,
            label=label,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(df.index, rotation=25)
    ax.set_title(title)
    ax.set_xlabel("Machine Learning Model")
    ax.set_ylabel(ylabel)

    if ymax is not None:
        ax.set_ylim(0, ymax)

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create graphs from real PathSnap model results."
    )

    parser.add_argument(
        "--cv",
        default="results/cv_model_comparison.csv",
    )
    parser.add_argument(
        "--test",
        default="results/student_test_model_comparison.csv",
    )
    parser.add_argument(
        "--out",
        default="results/plots",
    )

    args = parser.parse_args()

    cv = load_results(
        Path(args.cv),
        ["Model", "CV_Macro_F1", "CV_LogLoss"],
    )

    test = load_results(
        Path(args.test),
        ["Model", "Macro_F1", "Top1_Accuracy", "Log_Loss"],
    )

    if set(cv.index) != set(test.index):
        raise ValueError(
            "Cross-validation and student-test model names differ."
        )

    test = test.reindex(cv.index)

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    # Graph 1: Cross-validation Macro F1
    save_bar(
        cv["CV_Macro_F1"],
        "Cross-Validation Macro F1 Comparison",
        "Macro F1 Score",
        output / "cv_macro_f1.png",
        ymax=1,
    )

    # Graph 2: Student-testing model performance
    save_grouped_bar(
        test,
        ["Top1_Accuracy", "Macro_F1"],
        ["Top-1 Accuracy", "Macro F1"],
        "Student-Test Model Performance",
        "Score",
        output / "student_test_comparison.png",
        ymax=1,
    )

    # Graph 3: Log Loss comparison
    losses = pd.DataFrame({
        "CV_LogLoss": cv["CV_LogLoss"],
        "Test_LogLoss": test["Log_Loss"],
    })

    save_grouped_bar(
        losses,
        ["CV_LogLoss", "Test_LogLoss"],
        ["Cross-validation", "Student testing"],
        "Model Log Loss Comparison",
        "Log Loss",
        output / "log_loss_comparison.png",
    )

    # Consolidated results table
    combined = cv.join(test)
    combined.to_csv(
        output / "consolidated_model_comparison.csv"
    )

    print("\nAll comparison graphs generated successfully!")


if __name__ == "__main__":
    main()
