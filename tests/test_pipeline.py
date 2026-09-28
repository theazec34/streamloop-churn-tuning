from __future__ import annotations

import io
from types import SimpleNamespace

import pandas as pd
import pytest

from src.train_and_tune import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_pipeline,
    load_and_clean,
    pick_stable_candidate,
)


def sample_frame() -> pd.DataFrame:
    """Return a tiny frame with the same feature contract as the source data."""
    rows = 4
    data: dict[str, list[object]] = {
        "customerID": ["A", "B", "C", "D"],
        "tenure": [1, 2, 3, 4],
        "MonthlyCharges": [20.0, 40.0, 60.0, 80.0],
        "TotalCharges": [1.0, None, 5.0, 7.0],
        "SeniorCitizen": [0, 0, 1, 1],
        "Churn": ["No", "Yes", "No", "Yes"],
    }
    for feature in CATEGORICAL_FEATURES:
        data[feature] = ["No", "Yes", "No", "Yes"]
    assert all(len(values) == rows for values in data.values())
    return pd.DataFrame(data)


def test_load_and_clean_removes_identifier_and_maps_target() -> None:
    raw = sample_frame()
    raw["TotalCharges"] = raw["TotalCharges"].astype("object")
    raw.loc[1, "TotalCharges"] = " "
    stream = io.StringIO()
    raw.to_csv(stream, index=False)
    stream.seek(0)

    features, target = load_and_clean(stream)

    assert "customerID" not in features
    assert "Churn" not in features
    assert target.tolist() == [0, 1, 0, 1]
    assert pd.api.types.is_numeric_dtype(features["TotalCharges"])
    assert pd.isna(features.loc[1, "TotalCharges"])


def test_load_and_clean_rejects_unknown_target_label() -> None:
    raw = sample_frame()
    raw.loc[0, "Churn"] = "Maybe"
    stream = io.StringIO()
    raw.to_csv(stream, index=False)
    stream.seek(0)

    with pytest.raises(ValueError, match="Unexpected Churn labels"):
        load_and_clean(stream)


def test_pipeline_preprocessing_is_train_only_and_handles_new_category() -> None:
    raw = sample_frame()
    features = raw.drop(columns=["customerID", "Churn"])
    target = raw["Churn"].map({"Yes": 1, "No": 0})
    pipeline = build_pipeline().fit(features.iloc[:3], target.iloc[:3])

    numeric_imputer = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["num"]
        .named_steps["imputer"]
    )
    total_charges_index = NUMERIC_FEATURES.index("TotalCharges")
    assert numeric_imputer.statistics_[total_charges_index] == pytest.approx(3.0)

    unseen = features.iloc[[3]].copy()
    unseen.loc[:, "PaymentMethod"] = "A category absent from training"
    prediction = pipeline.predict(unseen)
    assert prediction.shape == (1,)


def test_stability_rule_prefers_lower_variance_within_recall_tolerance() -> None:
    search = SimpleNamespace(
        cv_results_={
            "mean_test_score": [0.81, 0.805, 0.70],
            "std_test_score": [0.03, 0.01, 0.001],
            "rank_test_score": [1, 2, 3],
            "params": [
                {"choice": "best-mean"},
                {"choice": "stable"},
                {"choice": "low"},
            ],
        }
    )

    selected = pick_stable_candidate(search)

    assert selected["params"] == {"choice": "stable"}
    assert selected["mean_test_score"] == pytest.approx(0.805)
