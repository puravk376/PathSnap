"""Train and evaluate PathSnap models using trip-aware cross-validation."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, log_loss

from train_models import get_models


def top1_accuracy(df, probabilities):
    """Check whether the highest-scoring candidate is correct."""
    temp = df[["trip_id", "observation_id", "label"]].copy()
    temp["score"] = probabilities

    selected = temp.loc[
        temp.groupby(["trip_id", "observation_id"])["score"].idxmax()
    ]

    return selected["label"].eq(1).mean()


def evaluate_models(data, features):
    """Evaluate all models without mixing trips across folds."""

    x = data[features]
    y = data["label"].astype(int)
    groups = data["trip_id"]

    cv = GroupKFold(n_splits=5)
    results = []

    for name, model in get_models().items():
        print(f"\nTraining {name}...")

        scores = []
        top1_scores = []
        losses = []

        for train_idx, val_idx in cv.split(x, y, groups):
            x_train = x.iloc[train_idx]
            x_val = x.iloc[val_idx]

            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            model.fit(x_train, y_train)

            predictions = model.predict(x_val)
            probabilities = model.predict_proba(x_val)[:, 1]

            scores.append(
                f1_score(y_val, predictions, average="macro")
            )

            losses.append(
                log_loss(y_val, probabilities, labels=[0, 1])
            )

            top1_scores.append(
                top1_accuracy(
                    data.iloc[val_idx], probabilities
                )
            )

        results.append({
            "Model": name,
            "CV_Macro_F1": np.mean(scores),
            "CV_Top1_Accuracy": np.mean(top1_scores),
            "CV_LogLoss": np.mean(losses),
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Candidate CSV from Person B")
    parser.add_argument(
        "--features", nargs="+", required=True,
        help="Numerical feature columns approved for training"
    )
    args = parser.parse_args()

    data = pd.read_csv(args.dataset)

    required = {"trip_id", "observation_id", "label"}
    required.update(args.features)

    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if data["trip_id"].nunique() < 5:
        raise ValueError("At least 5 distinct training trips required")

    results = evaluate_models(data, args.features)

    Path("results").mkdir(exist_ok=True)
    results.to_csv("results/cv_model_comparison.csv", index=False)

    print("\nModel Comparison:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()