
"""Evaluate five PathSnap models on the 10 student-testing trips."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from train_models import get_models


def validate_data(data, features, name):
    """Check the expected candidate dataset format."""
    required = {
        "trip_id", "observation_id", "arc_id", "label"
    } | set(features)

    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{name}: Missing columns: {missing}")

    if data.empty:
        raise ValueError(f"{name}: Dataset is empty")

    if data[list(required)].isna().any().any():
        raise ValueError(f"{name}: Missing required values")

    if not data["label"].isin([0, 1]).all():
        raise ValueError(f"{name}: Labels must be 0 or 1")

    if not all(
        pd.api.types.is_numeric_dtype(data[col])
        for col in features
    ):
        raise ValueError(f"{name}: Features must be numeric")

    if data.duplicated(
        ["trip_id", "observation_id", "arc_id"]
    ).any():
        raise ValueError(f"{name}: Duplicate candidates found")


def calculate_top1(data, probabilities):
    """Select the highest-scoring road candidate per observation."""
    temp = data[
        ["trip_id", "observation_id", "label"]
    ].copy().reset_index(drop=True)

    temp["score"] = probabilities

    # One positive candidate is expected when reference alignment
    # identifies a true candidate within the generated set.
    positives = temp.groupby(
        ["trip_id", "observation_id"]
    )["label"].sum()

    if (positives > 1).any():
        raise ValueError(
            "Multiple positive candidates for an observation. "
            "Check Person B's labelling method."
        )

    if (positives == 0).any():
        print(
            f"Warning: {(positives == 0).sum()} observations "
            "have no positive candidate. Confirm whether "
            "these are genuine candidate misses."
        )

    selected = temp.loc[
        temp.groupby(
            ["trip_id", "observation_id"]
        )["score"].idxmax()
    ]

    return float((selected["label"] == 1).mean())


def evaluate(train_data, test_data, features):
    """Train on training trips and evaluate on student-test trips."""
    x_train = train_data[features]
    y_train = train_data["label"].astype(int)

    x_test = test_data[features]
    y_test = test_data["label"].astype(int)

    results = []

    for name, model in get_models().items():
        print(f"\nTraining {name}...")

        start = time.perf_counter()
        model.fit(x_train, y_train)
        training_time = time.perf_counter() - start

        start = time.perf_counter()
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        inference_time = time.perf_counter() - start

        result = {
            "Model": name,
            "Accuracy": accuracy_score(
                y_test, predictions
            ),
            "Macro_F1": f1_score(
                y_test, predictions,
                average="macro", zero_division=0
            ),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "Top1_Accuracy": calculate_top1(
                test_data, probabilities
            ),
            "Log_Loss": log_loss(
                y_test, probabilities, labels=[0, 1]
            ),
            "Training_Time_s": training_time,
            "Inference_Time_s": inference_time,
        }

        results.append(result)

        print(
            f"Macro F1: {result['Macro_F1']:.4f} | "
            f"Top-1: {result['Top1_Accuracy']:.4f}"
        )

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument(
        "--features", nargs="+", required=True
    )

    args = parser.parse_args()

    train_data = pd.read_csv(args.train)
    test_data = pd.read_csv(args.test)

    validate_data(train_data, args.features, "Training")
    validate_data(test_data, args.features, "Student testing")

    if set(train_data["trip_id"]) & set(test_data["trip_id"]):
        raise ValueError(
            "Training and student-testing trip IDs overlap!"
        )

    if train_data["label"].nunique() != 2:
        raise ValueError("Training requires both label classes")

    results = evaluate(
        train_data, test_data, args.features
    )

    output = Path("results")
    output.mkdir(exist_ok=True)

    results.to_csv(
        output / "student_test_model_comparison.csv",
        index=False
    )

    print("\nFinal student-test comparison:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
