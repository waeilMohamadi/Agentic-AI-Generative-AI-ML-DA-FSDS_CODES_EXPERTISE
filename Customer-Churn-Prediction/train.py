"""Train and evaluate a churn model without leaking test data into preprocessing."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL = ROOT / "models" / "churn_pipeline.joblib"
REPORT = ROOT / "metrics.json"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    return frame


def make_pipeline(features: list[str], estimator) -> Pipeline:
    categorical = [column for column in features if column not in NUMERIC]
    preprocess = ColumnTransformer(
        [
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), NUMERIC),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("classifier", estimator)])


def train() -> dict:
    frame = prepare_frame(pd.read_csv(DATA))
    X = frame.drop(columns=["customerID", "Churn"])
    y = frame["Churn"].map({"No": 0, "Yes": 1})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=15, learning_rate=0.08, random_state=42
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    for name, estimator in candidates.items():
        pipeline = make_pipeline(list(X.columns), estimator)
        values = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="average_precision")
        scores[name] = {"mean_average_precision": round(float(values.mean()), 4),
                        "std_average_precision": round(float(values.std()), 4)}
    selected = max(scores, key=lambda name: scores[name]["mean_average_precision"])
    model = make_pipeline(list(X.columns), candidates[selected])
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    report = {
        "dataset_rows": len(frame), "train_rows": len(X_train), "test_rows": len(X_test),
        "selection_metric": "5-fold train-only average precision",
        "cross_validation": scores, "selected_model": selected,
        "held_out_test": {
            "average_precision": round(float(average_precision_score(y_test, probabilities)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
            "classification_threshold": 0.5,
            "classification_report": classification_report(y_test, predictions, output_dict=True),
        },
    }
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "features": list(X.columns)}, MODEL)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
