"""
StreamLoop – Ajuste sistemático del modelo de cancelación (churn).

Flujo:
1. Carga + limpieza mínima
2. Train/test split (antes de cualquier otra operación)
3. Pipeline (preprocesado + clasificador) → baseline
4. RandomizedSearchCV (exploración amplia)
5. GridSearchCV (refinado local)
6. Selección final por media y estabilidad en CV
7. Evaluación única en test del modelo ajustado
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
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

warnings.filterwarnings("ignore")

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
RANDOM_STATE = 42
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def load_and_clean(url: str = DATA_URL) -> tuple[pd.DataFrame, pd.Series]:
    """Carga el CSV y aplica limpieza mínima (sin imputar ni escalar)."""
    df = pd.read_csv(url)

    # ID no es predictivo
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # TotalCharges llega como object con strings vacíos
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Target binario: Yes=1 (churn), No=0
    y = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    X = df.drop(columns=["Churn"])
    return X, y


def build_pipeline() -> Pipeline:
    """Pipeline completo: preprocesado + RandomForest (defaults)."""
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    categorical_features = [
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
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
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
    """Métricas alineadas al negocio (recall de churn es prioritaria)."""
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
    """Tabla de candidatos ordenados por media CV y con std (estabilidad)."""
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
    """
    Elige entre los top-5 por mean_test_score el de menor std_test_score.
    Preferimos estabilidad si la media no cae más de 0.01 respecto al mejor.
    """
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
            "Entre los mejores por recall medio en CV, se eligió la configuración "
            "con menor desviación estándar entre folds (más estable), siempre que "
            "la media no baje más de 0.01 respecto al mejor promedio."
        ),
        "top5": table.to_dict(orient="records"),
    }


def main() -> None:
    print("=== 1. Carga y limpieza mínima ===")
    X, y = load_and_clean()
    print(f"Filas: {len(X)} | Features: {X.shape[1]} | Churn rate: {y.mean():.3f}")

    print("\n=== 2. Train/test split (antes de cualquier otra operación) ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    print("\n=== 3. Baseline (hiperparámetros por defecto) ===")
    baseline = build_pipeline()
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluate(baseline, X_test, y_test)
    print(
        f"Baseline recall_churn={baseline_metrics['recall_churn']:.4f} "
        f"precision={baseline_metrics['precision_churn']:.4f} "
        f"f1={baseline_metrics['f1_churn']:.4f} "
        f"accuracy={baseline_metrics['accuracy']:.4f}"
    )

    # Scoring de negocio: maximizar recall de la clase Churn=Yes
    scoring = "recall"

    print("\n=== 4. RandomizedSearchCV (exploración amplia) ===")
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
    # IMPORTANTE: solo sobre train; el test no se toca aquí
    random_search.fit(X_train, y_train)
    print(f"Mejor recall CV (random): {random_search.best_score_:.4f}")
    print(f"Mejores params (random): {random_search.best_params_}")

    print("\n=== 5. GridSearchCV (refinado alrededor de la zona prometedora) ===")
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
    # IMPORTANTE: solo sobre train
    grid_search.fit(X_train, y_train)
    print(f"Mejor recall CV (grid): {grid_search.best_score_:.4f}")
    print(f"Mejores params (grid): {grid_search.best_params_}")

    print("\n=== 6. Análisis de estabilidad en cv_results_ ===")
    selection = pick_stable_candidate(grid_search)
    print(
        f"Candidato estable: mean={selection['mean_test_score']:.4f} "
        f"std={selection['std_test_score']:.4f} rank={selection['rank_test_score']}"
    )
    print(f"Params elegidos: {selection['params']}")

    # best_estimator_ de GridSearchCV ya está reentrenado (refit=True).
    # Si el candidato estable coincide con best_params_, usamos best_estimator_.
    # Si no, reentrenamos UNA vez con esos params sobre todo el train
    # (no es reentrenar best_estimator_ a mano tras el search; es otra config).
    if selection["params"] == grid_search.best_params_:
        final_model = grid_search.best_estimator_
        used_best_estimator = True
    else:
        final_model = build_pipeline()
        final_model.set_params(**selection["params"])
        final_model.fit(X_train, y_train)
        used_best_estimator = False

    print("\n=== 7. Evaluación final en test (única vez) ===")
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
            "StreamLoop pierde más dinero por un churn no detectado (FN) que por "
            "ofrecer retención innecesaria (FP). Por eso scoring='recall' sobre la "
            "clase positiva (Churn=Yes)."
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

    # Serializar params (pueden contener None)
    out_path = ARTIFACTS / "metrics_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResumen guardado en {out_path}")


if __name__ == "__main__":
    main()
