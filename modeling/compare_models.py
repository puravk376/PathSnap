"""
PathSnap - Model experimentation and comparison (6 models, 6 algorithm families)

Task: binary classification of (GPS point, candidate arc) rows -> is this the reference arc?
The predicted probability is later used as the HMM emission score.

Protocol
  * Training data: reliable rows of the 80 training trips, sub-sampled by GPS point
    (--sample-frac, default 0.2) so all six models train in reasonable laptop time.
  * Cross-validation: 5-fold GroupKFold grouped by trip_id (no trip in both train and validation).
  * Student-test: each model is refit on the whole sample and evaluated on ALL 10 student_test trips.

Outputs (results/models/):
  cv_folds.csv               per-fold CV metrics for every model
  model_comparison.csv       the consolidated comparison table for the report
  cv_macro_f1.png, student_test_top1_f1.png, log_loss.png, inference_time.png
  confusion_<model>.png      student-test confusion matrices
  loss_curve_<model>.png     only for models with epoch-wise histories (MLP, HistGradientBoosting)

Usage:
  python modeling/compare_models.py results/features_train.parquet results/features_student_test.parquet data/student_test
  optional: --sample-frac 0.2
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (SEED, load_train, load_table, sample_observations, top1_accuracy, top1_choice,  # noqa: E402
                    route_continuity, candidate_miss_rate, feature_list)

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (f1_score, log_loss, accuracy_score, precision_score, recall_score,
                             confusion_matrix, ConfusionMatrixDisplay)

OUT = os.path.join("results", "models")


def get_models():
    """name -> (algorithm family, key configuration text, estimator)"""
    return {
        "Logistic Regression": ("Linear", "StandardScaler, L2, C=1.0",
                                make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))),
        "Gaussian Naive Bayes": ("Probabilistic", "default var_smoothing=1e-9",
                                 GaussianNB()),
        "Decision Tree": ("Tree", "max_depth=12, min_samples_leaf=50",
                          DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, random_state=SEED)),
        "Random Forest": ("Bagging ensemble", "100 trees, min_samples_leaf=5, max_samples=0.5",
                          RandomForestClassifier(n_estimators=100, min_samples_leaf=5, max_samples=0.5,
                                                 n_jobs=-1, random_state=SEED)),
        "HistGradientBoosting": ("Boosting", "lr=0.1, max_iter=300, early stopping on log loss",
                                 HistGradientBoostingClassifier(learning_rate=0.1, max_iter=300,
                                                                early_stopping=True, scoring="loss",
                                                                validation_fraction=0.1, random_state=SEED)),
        "MLP": ("Neural network", "StandardScaler, hidden=(64,32), adam, early stopping",
                make_pipeline(StandardScaler(),
                              MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512,
                                            max_iter=60, early_stopping=True, n_iter_no_change=5,
                                            random_state=SEED))),
    }


def clf_metrics(y, proba):
    pred = (proba >= 0.5).astype(int)
    return dict(
        accuracy=accuracy_score(y, pred),
        precision_macro=precision_score(y, pred, average="macro", zero_division=0),
        recall_macro=recall_score(y, pred, average="macro", zero_division=0),
        f1_macro=f1_score(y, pred, average="macro"),
        log_loss=log_loss(y, np.clip(proba, 1e-7, 1 - 1e-7), labels=[0, 1]),
    )


def bar(df, col, title, ylabel, path, err=None, fmt="{:.3f}"):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(df["model"], df[col], yerr=df[err] if err else None, capsize=4, color="#1565c0")
    for b, v in zip(bars, df[col]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=9)
    ax.set(title=title, ylabel=ylabel)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train"); ap.add_argument("test"); ap.add_argument("test_dir")
    ap.add_argument("--sample-frac", type=float, default=0.2)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    feats = feature_list()
    train = sample_observations(load_train(args.train), args.sample_frac)
    test = load_table(args.test)
    X, y, groups = train[feats], train["label"].to_numpy(), train["trip_id"].to_numpy()
    Xt, yt = test[feats], test["label"].to_numpy()
    print(f"train sample: {len(train)} rows / {train.trip_id.nunique()} trips | "
          f"student_test: {len(test)} rows / {test.trip_id.nunique()} trips | features: {len(feats)}")
    print(f"student_test candidate miss rate: {candidate_miss_rate(test)*100:.2f}%\n")

    fold_rows, summary = [], []
    for name, (family, config, model) in get_models().items():
        # ---------------- group-aware cross-validation
        for k, (tr_i, va_i) in enumerate(GroupKFold(n_splits=args.folds).split(X, y, groups), 1):
            model.fit(X.iloc[tr_i], y[tr_i])
            p = model.predict_proba(X.iloc[va_i])[:, 1]
            m = clf_metrics(y[va_i], p)
            m.update(model=name, fold=k, top1=top1_accuracy(train.iloc[va_i], p))
            fold_rows.append(m)
        f = pd.DataFrame([r for r in fold_rows if r["model"] == name])

        # ---------------- refit on full sample, evaluate on student_test
        t0 = time.perf_counter(); model.fit(X, y); train_s = time.perf_counter() - t0
        t0 = time.perf_counter(); pt = model.predict_proba(Xt)[:, 1]; infer_s = time.perf_counter() - t0
        tm = clf_metrics(yt, pt)
        choice = top1_choice(test, pt)
        row = dict(model=name, family=family, config=config,
                   cv_f1_macro=f.f1_macro.mean(), cv_f1_macro_std=f.f1_macro.std(),
                   cv_top1=f.top1.mean(), cv_log_loss=f.log_loss.mean(),
                   test_top1=float(choice.label.mean()),
                   test_f1_macro=tm["f1_macro"], test_precision_macro=tm["precision_macro"],
                   test_recall_macro=tm["recall_macro"], test_accuracy=tm["accuracy"],
                   test_log_loss=tm["log_loss"],
                   test_route_continuity=route_continuity(choice, args.test_dir),
                   train_time_s=train_s, inference_time_s=infer_s,
                   inference_ms_per_1k_rows=infer_s / len(test) * 1e6)
        summary.append(row)
        print(f"{name:22s} CV F1 {row['cv_f1_macro']:.3f}±{row['cv_f1_macro_std']:.3f} | "
              f"test Top-1 {row['test_top1']:.3f} F1 {row['test_f1_macro']:.3f} "
              f"logloss {row['test_log_loss']:.3f} cont {row['test_route_continuity']:.3f} | "
              f"train {train_s:.1f}s infer {infer_s:.2f}s", flush=True)

        # confusion matrix
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay(confusion_matrix(yt, (pt >= 0.5).astype(int)),
                               display_labels=["wrong arc", "true arc"]).plot(ax=ax, colorbar=False, values_format="d")
        ax.set_title(f"{name}\nstudent_test (threshold 0.5)")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"confusion_{name.replace(' ', '_')}.png"), dpi=150); plt.close(fig)

        # epoch-wise histories (only models that really have them)
        est = model[-1] if hasattr(model, "steps") else model
        if isinstance(est, HistGradientBoostingClassifier):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(-est.train_score_, label="training log loss")
            ax.plot(-est.validation_score_, label="validation log loss (10% held out)")
            ax.set(title=f"{name}: loss per boosting iteration", xlabel="iteration", ylabel="log loss"); ax.legend()
            fig.tight_layout(); fig.savefig(os.path.join(OUT, f"loss_curve_{name}.png"), dpi=150); plt.close(fig)
        if isinstance(est, MLPClassifier):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(est.loss_curve_, label="training loss (log loss)", color="#1565c0")
            ax.set(xlabel="epoch", ylabel="training loss", title=f"{name}: training history")
            ax2 = ax.twinx(); ax2.plot(est.validation_scores_, label="validation accuracy", color="#ef6c00")
            ax2.set_ylabel("validation accuracy (10% held out)")
            fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.85))
            fig.tight_layout(); fig.savefig(os.path.join(OUT, "loss_curve_MLP.png"), dpi=150); plt.close(fig)

    pd.DataFrame(fold_rows).to_csv(os.path.join(OUT, "cv_folds.csv"), index=False)
    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)
    with open(os.path.join(OUT, "run_config.json"), "w") as fh:
        json.dump(dict(sample_frac=args.sample_frac, folds=args.folds, seed=SEED, features=feats,
                       train_rows=len(train), test_rows=len(test)), fh, indent=2)

    bar(s, "cv_f1_macro", "Cross-validation Macro F1 (5-fold GroupKFold, train)", "Macro F1",
        os.path.join(OUT, "cv_macro_f1.png"), err="cv_f1_macro_std")
    fig, ax = plt.subplots(figsize=(9, 4.5)); w = 0.38; xs = np.arange(len(s))
    ax.bar(xs - w / 2, s.test_top1, w, label="Top-1 accuracy", color="#1565c0")
    ax.bar(xs + w / 2, s.test_f1_macro, w, label="Macro F1", color="#ef6c00")
    ax.set_xticks(xs); ax.set_xticklabels(s.model, rotation=20, ha="right"); ax.set_ylim(0, 1)
    ax.set(title="Student-test Top-1 accuracy and Macro F1"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "student_test_top1_f1.png"), dpi=150); plt.close(fig)
    bar(s, "test_log_loss", "Student-test log loss (lower is better)", "log loss", os.path.join(OUT, "log_loss.png"))
    bar(s, "inference_ms_per_1k_rows", "Inference time on student_test", "ms per 1,000 candidate rows",
        os.path.join(OUT, "inference_time.png"), fmt="{:.1f}")

    pd.set_option("display.width", 200)
    print("\n", s[["model", "cv_f1_macro", "test_top1", "test_f1_macro", "test_log_loss",
                   "test_route_continuity", "inference_time_s"]].round(4).to_string(index=False))
    print(f"\nsaved to {OUT}/")


if __name__ == "__main__":
    main()
