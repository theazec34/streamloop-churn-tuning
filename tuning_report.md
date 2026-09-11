# Informe de ajuste de hiperparámetros — StreamLoop

## Contexto

Clasificador binario de churn para StreamLoop. El coste de un **falso negativo** (cliente que cancela sin ser detectado) es mayor que el de un **falso positivo** (oferta de retención innecesaria). Por eso la búsqueda optimiza **`recall` de la clase `Churn=Yes`**, no accuracy.

## Datos y partición

| Concepto | Valor |
|----------|-------|
| Fuente | Telco Customer Churn (IBM) |
| Filas | 7 043 |
| Features | 19 (tras quitar `customerID`) |
| Tasa de churn | ~26.5 % |
| Train / Test | 5 634 / 1 409 (80/20, estratificado, `random_state=42`) |

Limpieza mínima fuera del pipeline: eliminar `customerID`, convertir `TotalCharges` a numérico (`errors="coerce"`) y mapear `Churn` a `{Yes:1, No:0}`.

Imputación, escalado y one-hot encoding viven **dentro** del `Pipeline` vía `ColumnTransformer`.

## Métrica de scoring

```text
scoring = "recall"   # clase positiva = Churn
```

**Justificación:** maximizar la detección de cancelaciones reduce el coste de negocio dominante (FN). Accuracy por defecto de scikit-learn habría favorecido la clase mayoritaria (`No`) y ocultado un recall de churn bajo.

## Baseline (defaults)

`Pipeline(preprocessor, RandomForestClassifier(random_state=42))` sin tuning. Evaluación en test **una sola vez**:

| Métrica | Valor |
|---------|-------|
| Recall (Churn) | **0.481** |
| Precision (Churn) | 0.623 |
| F1 (Churn) | 0.543 |
| Accuracy | 0.785 |
| ROC-AUC | 0.820 |
| Matriz de confusión `[[TN, FP], [FN, TP]]` | `[[926, 109], [194, 180]]` |

El accuracy parece razonable, pero **casi la mitad de los churners escapan** (194 FN). Eso confirma que accuracy no es la métrica adecuada.

## Búsqueda — fase 1: `RandomizedSearchCV`

- Espacio amplio: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `class_weight`, `bootstrap`
- `n_iter=25`, `cv=5`, `n_jobs=-1`, `refit=True`, `scoring="recall"`
- **Solo sobre train** (el test no participa)

| Resultado | Valor |
|-----------|-------|
| Mejor recall CV | **0.773** |
| Hallazgo clave | `class_weight='balanced_subsample'` y árboles no demasiado profundos |

Mejores hiperparámetros (random):

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

## Búsqueda — fase 2: `GridSearchCV`

Se acotó la malla alrededor de la zona encontrada por el random search (vecinos de `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`; se fijaron `max_features`, `class_weight` y `bootstrap`).

| Resultado | Valor |
|-----------|-------|
| Mejor recall CV (rank 1) | **0.804** (std ≈ 0.017) |
| Candidato estable elegido (rank 2) | **0.801** (std ≈ 0.013) |

### Inspección de `cv_results_` (top candidatos)

| Rank | mean recall CV | std | Comentario |
|------|----------------|-----|------------|
| 1 | 0.8040 | 0.0167 | Mejor media |
| 2 | 0.8013 | **0.0131** | Media casi igual, **más estable** |
| 2 | 0.8013 | 0.0159 | Similar media, más varianza |

## Modelo final elegido

Se eligió el candidato **rank 2** (media 0.8013, std 0.0131) frente al rank 1 (0.8040 / 0.0167):

- La diferencia de media es **&lt; 0.01**
- La desviación entre folds es menor → comportamiento más predecible en producción
- Hiperparámetros finales:

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

`GridSearchCV` se ejecutó con `refit=True`. El estimador final usa la configuración estable seleccionada tras revisar `cv_results_` (no se reentrenó a mano el `best_estimator_` del rank 1; se ajustó la configuración elegida una vez sobre todo el train).

## Evaluación final en test (única vez)

| Métrica | Baseline | Ajustado | Δ |
|---------|----------|----------|---|
| Recall (Churn) | 0.481 | **0.778** | **+0.297** |
| Precision (Churn) | 0.623 | 0.497 | −0.126 |
| F1 (Churn) | 0.543 | **0.606** | **+0.063** |
| Accuracy | 0.785 | 0.732 | −0.053 |
| ROC-AUC | 0.820 | **0.837** | **+0.017** |
| FN (churn no detectado) | 194 | **83** | **−111** |
| FP (retención innecesaria) | 109 | 295 | +186 |

Matriz del modelo ajustado: `[[740, 295], [83, 291]]`.

## Trade-offs

1. **Recall ↑ / Precision ↓:** esperado y deseable bajo la lógica de StreamLoop: preferimos contactar de más a clientes en riesgo que dejar escapar cancelaciones.
2. **Accuracy ↓:** engañosa aquí; la clase mayoritaria (`No`) es más fácil de acertar. Un accuracy más bajo con muchos menos FN es un mejor resultado de negocio.
3. **Estabilidad vs máximo CV:** se sacrificaron ~0.3 pp de recall CV a cambio de menor varianza entre folds.
4. **Coste computacional:** random (amplio, barato) → grid (estrecho, preciso). No se gastó presupuesto en una grid enorme desde el inicio.

## Conclusión

El proceso sistemático (baseline → random → grid → revisión de estabilidad) eleva el recall de churn de **0.48 a 0.78** y reduce los falsos negativos de **194 a 83** en el test set, con un trade-off de precisión aceptable para la prioridad de negocio de StreamLoop.
