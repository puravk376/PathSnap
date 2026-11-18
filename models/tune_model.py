
"""Bayesian hyperparameter optimisation for PathSnap using Optuna."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold


RANDOM_STATE = 42


def cross_validation_score(data, features, params):
    """Evaluate LightGBM with trip-aware cross-validation."""

    X = data[features]
    y = data["label"].astype(int)
    groups = data["trip_id"]

    cv = GroupKFold(n_splits=5)
    scores = []

    for train_idx, val_idx in cv.split(X, y, groups):
        model = LGBMClassifier(
            **params,
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=-1,
        )

        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        predictions = model.predict(X.iloc[val_idx])

        score = f1_score(
            y.iloc[val_idx],
            predictions,
            average="macro",
            labels=[0, 1],
            zero_division=0,
        )

        scores.append(score)

    return float(np.mean(scores))


def objective(trial, data, features):
    """Define the hyperparameter search space."""

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", 50, 300
        ),
        "max_depth": trial.suggest_int(
            "max_depth", 3, 12
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.2, log=True
        ),
        "num_leaves": trial.suggest_int(
            "num_leaves", 15, 63
        ),
        "min_child_samples": trial.suggest_int(
            "min_child_samples", 10, 50
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 0.001, 10.0, log=True
        ),
    }

    return cross_validation_score(data, features, params)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset",
        help="Training candidate CSV from Person B",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="Numerical training features",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of Optuna trials",
    )

    args = parser.parse_args()

    data = pd.read_csv(args.dataset)

    required = {"trip_id", "label", *args.features}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if data["trip_id"].nunique() < 5:
        raise ValueError("At least five training trips required")

    if data["label"].isna().any():
        raise ValueError("Missing training labels")

    if not set(data["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary: 0 or 1")

    if data["label"].nunique() < 2:
        raise ValueError("Training requires both classes")

    if data[args.features].isna().any().any():
        raise ValueError("Missing feature values must be handled")

    if not all(
        pd.api.types.is_numeric_dtype(data[col])
        for col in args.features
    ):
        raise ValueError("All selected features must be numeric")

    output_dir = Path("results/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    study.optimize(
        lambda trial: objective(trial, data, args.features),
        n_trials=args.trials,
    )

    default_params = {
        "n_estimators": 100,
    }

    baseline_score = cross_validation_score(
        data, args.features, default_params
    )

    print("\nUntuned CV Macro F1:", baseline_score)
    print("Tuned CV Macro F1:", study.best_value)
    print("Best parameters:", study.best_params)

    study.trials_dataframe().to_csv(
        output_dir / "lightgbm_optuna_trials.csv",
        index=False,
    )

    comparison = {
        "untuned_cv_macro_f1": baseline_score,
        "tuned_cv_macro_f1": study.best_value,
        "best_parameters": study.best_params,
        "number_of_trials": args.trials,
    }

    with open(
        output_dir / "lightgbm_tuning_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(comparison, file, indent=4)

    final_model = LGBMClassifier(
        **study.best_params,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1,
    )

    final_model.fit(data[args.features], data["label"])

    joblib.dump(
        {"model": final_model, "features": args.features},
        output_dir / "lightgbm_tuned.joblib",
    )

    print("\nTuning completed successfully!")


if __name__ == "__main__":
    main()
