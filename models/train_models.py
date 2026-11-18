"""Training pipeline for PathSnap candidate classification."""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier


RANDOM_STATE = 42


def get_models():
    """Return the five models for initial experimentation."""

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),

        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5),
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=100,
            verbosity=-1,
            random_state=RANDOM_STATE,
        ),
    }

    return models


if __name__ == "__main__":
    models = get_models()

    print("PathSnap ML Models:")
    for i, name in enumerate(models, start=1):
        print(f"{i}. {name}")