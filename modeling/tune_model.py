"""
PathSnap - Bayesian hyperparameter optimisation of the selected model (HistGradientBoosting)

Search : Optuna TPE sampler (Bayesian optimisation), fixed seed.
Objective: mean observation-level Top-1 accuracy over GroupKFold (grouped by trip) on the
           80 training trips. Top-1 is the primary metric because the route is built from the
           best candidate per GPS point; Macro F1 and log loss are also logged for every trial.
Data   : same reliable, point-level sample of the training trips as compare_models.py.
           student_test is NOT used inside the search; evaluator_test is never used.

After the search:
  * untuned (default config from compare_models.py) vs tuned are compared with identical
    data and metrics (5-fold CV Macro F1, student-test Top-1, Macro F1, log loss, inference time)
  * the tuned model is refit on ALL reliable training rows and exported to artifacts/

Outputs:
  results/tuning/optuna_trials.csv, best_params.json, tuned_vs_untuned.csv,
  optimisation_history.png, param_importance.png (if available)
  artifacts/model.joblib, artifacts/model_config.json

Usage:
  python modeling/tune_model.py results/features_train.parquet results/features_student_test.parquet data/student_test
  optional: --trials 25 --sample-frac 0.2
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import joblib
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SEED, load_train, load_table, sample_observations, top1_accuracy, top1_choice, route_continuity, feature_list  # noqa: E402
from compare_models import clf_metrics  # noqa: E402

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier

OUT = os.path.join("results", "tuning")
ART = "artifacts"
UNTUNED = dict(learning_rate=0.1, max_iter=300, early_stopping=True, scoring="loss",
               validation_fraction=0.1, random_state=SEED)


def make_model(params):
    return HistGradientBoostingClassifier(**{**UNTUNED, **params})


def cv_eval(params, X, y, groups, df, folds):
    tops, f1s, lls = [], [], []
    for tr_i, va_i in GroupKFold(n_splits=folds).split(X, y, groups):
        m = make_model(params).fit(X.iloc[tr_i], y[tr_i])
        p = m.predict_proba(X.iloc[va_i])[:, 1]
        cm = clf_metrics(y[va_i], p)
        tops.append(top1_accuracy(df.iloc[va_i], p)); f1s.append(cm["f1_macro"]); lls.append(cm["log_loss"])
    return float(np.mean(tops)), float(np.mean(f1s)), float(np.std(f1s)), float(np.mean(lls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train"); ap.add_argument("test"); ap.add_argument("test_dir")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--sample-frac", type=float, default=0.2)
    ap.add_argument("--search-folds", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(ART, exist_ok=True)

    feats = feature_list()
    full = load_train(args.train)
    train = sample_observations(full, args.sample_frac)
    test = load_table(args.test)
    X, y, groups = train[feats], train["label"].to_numpy(), train["trip_id"].to_numpy()

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            max_iter=trial.suggest_int("max_iter", 100, 600, step=50),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 127),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 300, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
            max_features=trial.suggest_float("max_features", 0.4, 1.0),
        )
        top1, f1, f1_std, ll = cv_eval(params, X, y, groups, train, args.search_folds)
        trial.set_user_attr("cv_f1_macro", f1); trial.set_user_attr("cv_log_loss", ll)
        return top1

    print(f"Bayesian search: {args.trials} trials, {args.search_folds}-fold GroupKFold, {len(train)} rows")
    t0 = time.perf_counter()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                study_name="pathsnap_hgb")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=args.trials,
                   callbacks=[lambda s, t: print(f"  trial {t.number:2d}: CV Top-1 {t.value:.4f} "
                                                 f"(best {s.best_value:.4f})", flush=True)])
    search_s = time.perf_counter() - t0
    trials = study.trials_dataframe()
    trials.to_csv(os.path.join(OUT, "optuna_trials.csv"), index=False)
    best = study.best_params
    with open(os.path.join(OUT, "best_params.json"), "w") as f:
        json.dump(dict(best_params=best, best_cv_top1=study.best_value, n_trials=args.trials,
                       search_time_s=search_s, objective="mean GroupKFold Top-1 accuracy"), f, indent=2)
    print(f"best CV Top-1 {study.best_value:.4f} with {best} ({search_s/60:.1f} min)")

    fig, ax = plt.subplots(figsize=(7, 4))
    vals = trials["value"].to_numpy()
    ax.plot(vals, "o", color="#90caf9", label="trial"); ax.plot(np.maximum.accumulate(vals), color="#1565c0", label="best so far")
    ax.set(title="Optuna TPE optimisation history", xlabel="trial", ylabel="CV Top-1 accuracy"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "optimisation_history.png"), dpi=150); plt.close(fig)
    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(7, 4)); ax.barh(list(imp)[::-1], list(imp.values())[::-1], color="#1565c0")
        ax.set(title="Hyperparameter importance (fANOVA)", xlabel="importance")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "param_importance.png"), dpi=150); plt.close(fig)
    except Exception as e:  # importance needs enough completed trials
        print("param importance skipped:", e)

    # ---------------- untuned vs tuned, same data + metrics
    rows = []
    for name, params in [("Untuned HistGradientBoosting", {}), ("Tuned HistGradientBoosting", best)]:
        top1_cv, f1_cv, f1_std, ll_cv = cv_eval(params, X, y, groups, train, 5)
        m = make_model(params).fit(X, y)
        t0 = time.perf_counter(); p = m.predict_proba(test[feats])[:, 1]; inf = time.perf_counter() - t0
        tm = clf_metrics(test["label"].to_numpy(), p)
        ch = top1_choice(test, p)
        rows.append(dict(model=name, cv_f1_macro=f1_cv, cv_f1_macro_std=f1_std, cv_top1=top1_cv, cv_log_loss=ll_cv,
                         test_top1=float(ch.label.mean()), test_f1_macro=tm["f1_macro"], test_log_loss=tm["log_loss"],
                         test_route_continuity=route_continuity(ch, args.test_dir), inference_time_s=inf))
    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(OUT, "tuned_vs_untuned.csv"), index=False)
    print("\n", comp.round(4).to_string(index=False))

    # ---------------- final model: refit tuned config on ALL reliable training rows
    t0 = time.perf_counter()
    final = make_model(best).fit(full[feats], full["label"].to_numpy())
    fit_s = time.perf_counter() - t0
    joblib.dump(final, os.path.join(ART, "model.joblib"))
    with open(os.path.join(ART, "model_config.json"), "w") as f:
        json.dump(dict(model="HistGradientBoostingClassifier", params={**UNTUNED, **best},
                       features=feats, candidate_radius_m=50.0, trained_on="80 training trips, reliable rows",
                       train_rows=len(full), fit_time_s=fit_s), f, indent=2)
    print(f"\nfinal model refit on {len(full)} rows in {fit_s:.1f}s -> {ART}/model.joblib")


if __name__ == "__main__":
    main()
