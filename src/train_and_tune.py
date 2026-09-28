"""Reproduce the StreamLoop churn-model tuning study.

The holdout set is created before learned preprocessing. Model search and
candidate selection use only the training partition.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DATA_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
RANDOM_STATE = 42
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def _download_verified_dataset() -> io.BytesIO:
    """Download the public sample data and reject unexpected content."""
    request = Request(DATA_URL, headers={"User-Agent": "streamloop-portfolio/0.1"})
    with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed HTTPS URL
        content = response.read()

    digest = hashlib.sha256(content).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError(
            "Dataset checksum mismatch. The upstream file may have changed; "
            "review it before updating DATA_SHA256."
        )
    return io.BytesIO(content)


def load_and_clean(
    source: str | Path | io.StringIO | io.BytesIO | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load data and perform non-learned, row-independent cleaning.

    Passing ``None`` downloads the known public sample and verifies its checksum.
    A local path or in-memory stream is accepted to support tests and offline use.
    """
    df = pd.read_csv(_download_verified_dataset() if source is None else source)
    required_columns = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["Churn"])
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    # The pseudonymous identifier is not a useful feature and could enable
    # memorisation or accidental disclosure.
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Blank TotalCharges values become missing and are imputed inside the pipeline.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    mapped_target = df["Churn"].map({"Yes": 1, "No": 0})
    if mapped_target.isna().any():
        unexpected = sorted(df.loc[mapped_target.isna(), "Churn"].astype(str).unique())
        raise ValueError(f"Unexpected Churn labels: {unexpected}")

    y = mapped_target.astype(int)
    X = df.drop(columns=["Churn"])
    return X, y


def build_pipeline() -> Pipeline:
    """Build preprocessing plus a default random-forest classifier."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            ),
        ]
    )


def evaluate(model, X_test, y_test) -> dict:
    """Return holdout metrics, including positive-class precision and recall."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_churn": float(precision_score(y_test, y_pred, pos_label=1)),
        "recall_churn": float(recall_score(y_test, y_pred, pos_label=1)),
        "f1_churn": float(f1_score(y_test, y_pred, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["No", "Yes"]
        ),
    }


def cv_stability_table(search) -> pd.DataFrame:
    """Sort candidates by mean cross-validation recall and variability."""
    results = pd.DataFrame(search.cv_results_)
    cols = [
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
        "params",
    ]
    table = results[cols].copy()
    table["cv_lower"] = table["mean_test_score"] - table["std_test_score"]
    table = table.sort_values(
        ["mean_test_score", "std_test_score"], ascending=[False, True]
    )
    return table


def pick_stable_candidate(search) -> dict:
    """Choose the lowest-variance top-five candidate within 0.01 of the best."""
    table = cv_stability_table(search).head(5).reset_index(drop=True)
    best_mean = table.loc[0, "mean_test_score"]
    candidates = table[table["mean_test_score"] >= best_mean - 0.01]
    chosen = candidates.sort_values("std_test_score", ascending=True).iloc[0]
    return {
        "params": chosen["params"],
        "mean_test_score": float(chosen["mean_test_score"]),
        "std_test_score": float(chosen["std_test_score"]),
        "rank_test_score": int(chosen["rank_test_score"]),
        "rationale": (
            "Among the leading mean CV-recall candidates, choose the lowest "
            "fold-to-fold standard deviation when mean recall is within 0.01 "
            "of the best score."
        ),
        "top5": table.to_dict(orient="records"),
    }


def main() -> None:
    print("=== 1. Load and minimally clean data ===")
    X, y = load_and_clean()
    print(f"Rows: {len(X)} | Features: {X.shape[1]} | Churn rate: {y.mean():.3f}")

    print("\n=== 2. Train/test split before learned preprocessing ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    print("\n=== 3. Baseline (default hyperparameters) ===")
    baseline = build_pipeline()
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluate(baseline, X_test, y_test)
    print(
        f"Baseline recall_churn={baseline_metrics['recall_churn']:.4f} "
        f"precision={baseline_metrics['precision_churn']:.4f} "
        f"f1={baseline_metrics['f1_churn']:.4f} "
        f"accuracy={baseline_metrics['accuracy']:.4f}"
    )

    # Business scoring: maximise recall for the positive (churn) class.
    scoring = "recall"

    print("\n=== 4. RandomizedSearchCV (broad search) ===")
    random_pipe = build_pipeline()
    random_param_distributions = {
        "classifier__n_estimators": [100, 200, 300, 400],
        "classifier__max_depth": [None, 5, 10, 15, 20, 30],
        "classifier__min_samples_split": [2, 5, 10, 20],
        "classifier__min_samples_leaf": [1, 2, 4, 8],
        "classifier__max_features": ["sqrt", "log2", None],
        "classifier__class_weight": [None, "balanced", "balanced_subsample"],
        "classifier__bootstrap": [True, False],
    }
    random_search = RandomizedSearchCV(
        estimator=random_pipe,
        param_distributions=random_param_distributions,
        n_iter=25,
        scoring=scoring,
        cv=5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=1,
    )
    # Search sees training data only; the holdout set is not passed here.
    random_search.fit(X_train, y_train)
    print(f"Best randomized-search CV recall: {random_search.best_score_:.4f}")
    print(f"Best randomized-search parameters: {random_search.best_params_}")

    print("\n=== 5. GridSearchCV (local refinement) ===")
    bp = random_search.best_params_

    def neighbors_int(value, options, pad=1):
        if value is None:
            return [None, 10, 15, 20]
        ordered = sorted(options)
        idx = ordered.index(value) if value in ordered else 0
        lo = max(0, idx - pad)
        hi = min(len(ordered), idx + pad + 1)
        return ordered[lo:hi]

    n_est = bp["classifier__n_estimators"]
    depth = bp["classifier__max_depth"]
    mss = bp["classifier__min_samples_split"]
    msl = bp["classifier__min_samples_leaf"]

    grid_param_grid = {
        "classifier__n_estimators": sorted(
            set(neighbors_int(n_est, [100, 200, 300, 400]) + [n_est])
        ),
        "classifier__max_depth": (
            [None, 10, 15, 20]
            if depth is None
            else sorted(set(neighbors_int(depth, [5, 10, 15, 20, 30]) + [depth]))
        ),
        "classifier__min_samples_split": sorted(
            set(neighbors_int(mss, [2, 5, 10, 20]) + [mss])
        ),
        "classifier__min_samples_leaf": sorted(
            set(neighbors_int(msl, [1, 2, 4, 8]) + [msl])
        ),
        "classifier__max_features": [bp["classifier__max_features"]],
        "classifier__class_weight": [bp["classifier__class_weight"]],
        "classifier__bootstrap": [bp["classifier__bootstrap"]],
    }

    grid_pipe = build_pipeline()
    grid_search = GridSearchCV(
        estimator=grid_pipe,
        param_grid=grid_param_grid,
        scoring=scoring,
        cv=5,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    # Grid search also sees training data only.
    grid_search.fit(X_train, y_train)
    print(f"Best grid-search CV recall: {grid_search.best_score_:.4f}")
    print(f"Best grid-search parameters: {grid_search.best_params_}")

    print("\n=== 6. Inspect variation across CV folds ===")
    selection = pick_stable_candidate(grid_search)
    print(
        f"Selected candidate: mean={selection['mean_test_score']:.4f} "
        f"std={selection['std_test_score']:.4f} rank={selection['rank_test_score']}"
    )
    print(f"Selected parameters: {selection['params']}")

    # GridSearchCV has already refitted best_estimator_. If the stability rule
    # selects another configuration, fit that configuration on all training data.
    if selection["params"] == grid_search.best_params_:
        final_model = grid_search.best_estimator_
        used_best_estimator = True
    else:
        final_model = build_pipeline()
        final_model.set_params(**selection["params"])
        final_model.fit(X_train, y_train)
        used_best_estimator = False

    print("\n=== 7. Final holdout evaluation ===")
    tuned_metrics = evaluate(final_model, X_test, y_test)
    print(
        f"Tuned recall_churn={tuned_metrics['recall_churn']:.4f} "
        f"precision={tuned_metrics['precision_churn']:.4f} "
        f"f1={tuned_metrics['f1_churn']:.4f} "
        f"accuracy={tuned_metrics['accuracy']:.4f}"
    )

    summary = {
        "scoring": scoring,
        "scoring_justification": (
            "The scenario assumes a missed churner (false negative) costs more "
            "than an unnecessary retention contact (false positive), so search "
            "optimises recall for the positive churn class."
        ),
        "baseline_metrics": baseline_metrics,
        "random_search": {
            "best_score_cv": float(random_search.best_score_),
            "best_params": random_search.best_params_,
            "n_iter": 25,
            "cv": 5,
        },
        "grid_search": {
            "best_score_cv": float(grid_search.best_score_),
            "best_params": grid_search.best_params_,
            "param_grid": grid_param_grid,
            "cv": 5,
        },
        "final_selection": selection,
        "used_best_estimator": used_best_estimator,
        "tuned_metrics": tuned_metrics,
        "delta_recall": float(
            tuned_metrics["recall_churn"] - baseline_metrics["recall_churn"]
        ),
        "delta_precision": float(
            tuned_metrics["precision_churn"] - baseline_metrics["precision_churn"]
        ),
        "delta_f1": float(tuned_metrics["f1_churn"] - baseline_metrics["f1_churn"]),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "random_state": RANDOM_STATE,
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS / "metrics_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()
