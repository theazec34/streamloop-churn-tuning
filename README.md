# StreamLoop churn model tuning

This junior machine-learning portfolio project studies a fictional retention
problem using IBM's public Telco Customer Churn sample. It asks a narrow
question: can systematic random-forest tuning detect more customers labelled
as churners than the default model?

This is an analysis, not a deployed product. “StreamLoop” is a scenario name;
the repository contains no StreamLoop customer data and makes no claim that the
model creates business value in production.

## Business framing

The scenario assumes that failing to identify a customer who will churn (a
false negative) costs more than contacting a customer who would have stayed (a
false positive). Hyperparameter search therefore maximises **recall for
`Churn=Yes`**.

Recall answers “of the customers who churned, how many did the model find?”
Precision answers “of the customers flagged, how many actually churned?”
Optimising recall is not free: the tuned model finds 111 more churners in the
holdout data, but also creates 186 more false alarms. A real decision would
need contact and churn costs, team capacity, and threshold analysis; those
inputs are not available here.

## Method and leakage prevention

1. Remove the pseudonymous ID, parse `TotalCharges`, and encode the target.
2. Make one stratified 80/20 train/test split with `random_state=42`.
3. Keep median imputation, scaling, and one-hot encoding inside a
   scikit-learn `Pipeline`.
4. Fit a default random-forest baseline.
5. Run `RandomizedSearchCV`, then a smaller `GridSearchCV`, on training data
   only. Each CV fold fits its own preprocessing steps.
6. Among the five leading grid candidates within 0.01 recall of the best,
   choose the one with the lowest fold-to-fold standard deviation.
7. Compare the baseline and selected model on the untouched holdout partition.

Putting learned preprocessing inside the pipeline prevents medians, category
levels, and scaling statistics from being learned from validation or holdout
rows. The target and `customerID` are never model features. The test metrics
were inspected for the baseline and final model, but not used by either search.

## Reproduced results

The checked-in output is [`artifacts/metrics_summary.json`](artifacts/metrics_summary.json).

| Holdout metric | Default model | Tuned model | Change |
|---|---:|---:|---:|
| Churn recall | 0.481 | **0.778** | +0.297 |
| Churn precision | **0.623** | 0.497 | -0.126 |
| Churn F1 | 0.543 | **0.606** | +0.063 |
| Accuracy | **0.785** | 0.732 | -0.053 |
| ROC-AUC | 0.820 | **0.837** | +0.017 |
| False negatives | 194 | **83** | -111 |
| False positives | **109** | 295 | +186 |

These figures describe one holdout split of this sample only. See
[`tuning_report.md`](tuning_report.md) for the search results and exact
parameters.

## Reproduce the analysis

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"

pytest
ruff check .
python -m src.train_and_tune
```

The script downloads the same public CSV used by the original analysis and
checks its SHA-256 digest before reading it. It then overwrites
`artifacts/metrics_summary.json`. For the notebook:

```bash
python -m pip install -e ".[notebook]"
jupyter notebook explore.ipynb
```

Dependency ranges are declared in `pyproject.toml`; no lock file is provided,
so exact transitive package versions can still affect future reproduction.

## Privacy and security notes

- No raw data, credentials, model pickle, or direct personal identifiers are
  committed.
- The public sample includes customer-level demographic, service, and billing
  fields. `customerID` is pseudonymous rather than anonymous, so it is dropped
  before display and modelling.
- `gender`, `SeniorCitizen`, `Partner`, and `Dependents` can be sensitive or
  proxy attributes. A real project would require a lawful basis, access
  controls, retention rules, fairness review, and monitoring.
- The fixed HTTPS source and checksum make silent upstream changes fail closed.
  Local input remains the operator's responsibility.

## Limitations

- This is a small public teaching dataset with no evidence that it represents
  a streaming service or current customers.
- One random holdout split is weaker than repeated or temporal validation.
- The stability rule and 0.01 tolerance are a judgement call, not proof of
  production stability.
- The default 0.5 decision threshold was not tuned against actual costs.
- With `bootstrap=False`, scikit-learn effectively treats
  `class_weight="balanced_subsample"` like full-data balanced weights; that
  search dimension is therefore partly redundant.
- No calibration, fairness analysis, drift analysis, causal uplift estimate,
  or retention-experiment outcome is included.
- Scaling is unnecessary for random forests but is retained to preserve the
  original analysis and its reproduced metrics.
