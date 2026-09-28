# StreamLoop tuning report

## Question

Under the fictional StreamLoop scenario, a missed churner is assumed to cost
more than an unnecessary retention contact. The model search therefore
optimises positive-class **recall**, not accuracy. No monetary cost data was
provided, so this is an explicit modelling assumption rather than a measured
business fact.

## Data and split

| Item | Value |
|----------|-------|
| Source | IBM Telco Customer Churn public sample |
| Rows | 7,043 |
| Features | 19 after removing `customerID` |
| Churn rate | about 26.5% |
| Train / test | 5,634 / 1,409 (80/20, stratified, `random_state=42`) |

Only row-independent parsing happens before the split: remove `customerID`,
convert `TotalCharges` to numeric, and map `Churn` to `{Yes: 1, No: 0}`.
Median imputation, scaling, and one-hot encoding remain inside a
`ColumnTransformer` and `Pipeline`. This means each CV fold learns those values
from its training fold only; validation and holdout rows cannot leak into them.

The holdout set is not passed to either search. It is inspected for the
baseline and the final selected model.

## Search metric

```text
scoring = "recall"  # positive class = churn
```

Recall measures the share of actual churners detected. Precision measures the
share of flagged customers who actually churn. Accuracy alone can obscure poor
churn recall because `No` is the majority class.

## Default baseline

The baseline is
`Pipeline(preprocessor, RandomForestClassifier(random_state=42))` with the
classifier defaults.

| Metric | Value |
|---------|-------|
| Recall (Churn) | **0.481** |
| Precision (Churn) | 0.623 |
| F1 (Churn) | 0.543 |
| Accuracy | 0.785 |
| ROC-AUC | 0.820 |
| Confusion matrix `[[TN, FP], [FN, TP]]` | `[[926, 109], [194, 180]]` |

The 0.785 accuracy masks 194 missed churners and a churn recall of 0.481.

## Stage 1: `RandomizedSearchCV`

- Broad search over tree count, depth, split and leaf sizes, feature sampling,
  class weighting, and bootstrap behaviour
- `n_iter=25`, `cv=5`, `n_jobs=-1`, `refit=True`, `scoring="recall"`
- Training partition only

| Result | Value |
|-----------|-------|
| Best mean CV recall | **0.773** |
| Selected region | `class_weight='balanced_subsample'`, moderate tree depth |

Best random-search parameters:

```python
{
  "classifier__n_estimators": 200,
  "classifier__min_samples_split": 10,
  "classifier__min_samples_leaf": 8,
  "classifier__max_features": "log2",
  "classifier__max_depth": 10,
  "classifier__class_weight": "balanced_subsample",
  "classifier__bootstrap": False,
}
```

## Stage 2: `GridSearchCV`

The grid refines neighbouring integer values around the random-search result
while holding `max_features`, `class_weight`, and `bootstrap` fixed.

| Result | Value |
|-----------|-------|
| Best mean CV recall (rank 1) | **0.804** (std about 0.017) |
| Stability-rule choice (rank 2) | **0.801** (std about 0.013) |

### Leading `cv_results_` candidates

| Rank | Mean CV recall | Std | Note |
|------|----------------|-----|------------|
| 1 | 0.8040 | 0.0167 | Highest mean |
| 2 | 0.8013 | **0.0131** | Similar mean, lower variation |
| 2 | 0.8013 | 0.0159 | Similar mean, higher variation |

## Final candidate

The predeclared code rule considers the five leading candidates within 0.01
recall of the best, then chooses the lowest standard deviation. It selects rank
2 (mean 0.8013, std 0.0131) instead of rank 1 (0.8040, std 0.0167).

This is a reasonable preference for this exercise, but lower variation across
five folds does not establish production stability. Final parameters:

```python
{
  "classifier__bootstrap": False,
  "classifier__class_weight": "balanced_subsample",
  "classifier__max_depth": 5,
  "classifier__max_features": "log2",
  "classifier__min_samples_leaf": 4,
  "classifier__min_samples_split": 10,
  "classifier__n_estimators": 100,
}
```

`GridSearchCV` uses `refit=True`. Because the rule selects a different
configuration from `best_estimator_`, that selected configuration is fitted
once on all training rows.

## Holdout comparison

| Metric | Baseline | Tuned | Change |
|---------|----------|----------|---|
| Recall (Churn) | 0.481 | **0.778** | **+0.297** |
| Precision (Churn) | 0.623 | 0.497 | −0.126 |
| F1 (Churn) | 0.543 | **0.606** | **+0.063** |
| Accuracy | 0.785 | 0.732 | −0.053 |
| ROC-AUC | 0.820 | **0.837** | **+0.017** |
| False negatives | 194 | **83** | **-111** |
| False positives | 109 | 295 | +186 |

Tuned confusion matrix: `[[740, 295], [83, 291]]`.

## Interpretation

The tuned model raises churn recall by about 29.7 percentage points and reduces
false negatives by 111. It also lowers precision by about 12.6 points and adds
186 false positives. That trade-off matches the stated assumption, but calling
it “acceptable” would require costs and operational capacity that this project
does not have.

## Limitations

- Results come from one random holdout split of a public teaching dataset.
- Random rather than temporal splitting may not reflect future-customer drift.
- Search uses a limited parameter budget and the default 0.5 threshold.
- No confidence interval, calibration, fairness, uplift, or retention experiment
  is reported.
- Demographic and relationship fields can be sensitive or act as proxies.
- The holdout comparison supports this retrospective study, not a deployment
  claim.
