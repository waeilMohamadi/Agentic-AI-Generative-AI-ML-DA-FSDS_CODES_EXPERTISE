import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from xgboost import XGBClassifier

from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn
import mlflow.xgboost


# =========================
# 1) Create imbalanced data
# =========================
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=2,
    n_redundant=8,
    weights=[0.9, 0.1],
    flip_y=0,
    random_state=42
)

print("Overall class distribution:", np.unique(y, return_counts=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print("Train class distribution:", np.unique(y_train, return_counts=True))
print("Test class distribution :", np.unique(y_test, return_counts=True))


# =========================
# 2) Define models
# =========================
models = [
    (
        "Logistic Regression",
        LogisticRegression(C=1, solver="liblinear", random_state=42)
    ),
    (
        "Random Forest",
        RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)
    ),
    (
        "XGBClassifier",
        XGBClassifier(
            eval_metric="logloss",
            random_state=42
        )
    ),
    (
        "XGBClassifier + SMOTETomek",
        Pipeline(steps=[
            ("smt", SMOTETomek(random_state=42)),
            ("xgb", XGBClassifier(
                eval_metric="logloss",
                random_state=42
            ))
        ])
    )
]


# =========================
# 3) Setup MLflow
# =========================
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("imbalanced_classification_v2")


# =========================
# 4) Train + Evaluate + Log
# =========================
for model_name, model in models:
    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Print to console
    print(f"\n===== {model_name} =====")
    print(classification_report(y_test, y_pred))

    # Log to MLflow
    with mlflow.start_run(run_name=model_name):
        # log basic info
        mlflow.log_param("model_name", model_name)

        # log metrics
        mlflow.log_metric("accuracy", report["accuracy"])
        mlflow.log_metric("recall_class_0", report["0"]["recall"])
        mlflow.log_metric("recall_class_1", report["1"]["recall"])
        mlflow.log_metric("f1_macro", report["macro avg"]["f1-score"])

        # log params (minimal & clear)
        if model_name == "Logistic Regression":
            mlflow.log_params({"C": 1, "solver": "liblinear"})
            mlflow.sklearn.log_model(model, "model")

        elif model_name == "Random Forest":
            mlflow.log_params({"n_estimators": 30, "max_depth": 3})
            mlflow.sklearn.log_model(model, "model")

        elif model_name == "XGBClassifier":
            mlflow.log_params({"eval_metric": "logloss"})
            # ✅ FIX: log Booster to avoid `_estimator_type` error
            mlflow.xgboost.log_model(model.get_booster(), "model")

        else:
            # Pipeline with SMOTE -> must be logged with sklearn flavor
            mlflow.log_param("resampling", "SMOTETomek")
            mlflow.log_param("eval_metric", "logloss")
            mlflow.sklearn.log_model(model, "model")

print("\n✅ Done. Open MLflow UI: http://localhost:5000")
