# StreamLoop – Ajustando el Modelo de Cancelación

Proyecto de ajuste sistemático de hiperparámetros para un clasificador de churn (cancelación) de la plataforma de streaming **StreamLoop**.

## Objetivo

Pasar de un clasificador con hiperparámetros por defecto a una configuración elegida de forma sistemática:

1. **Baseline** con `Pipeline` (preprocesado + clasificador) y defaults.
2. **RandomizedSearchCV** para explorar un espacio amplio de forma barata.
3. **GridSearchCV** para refinar la zona prometedora.
4. **Selección final** mirando media y varianza entre folds (`cv_results_`), no solo el mejor promedio.
5. Evaluación en test **exactamente dos veces**: baseline y modelo final.

## Prioridad de negocio

> StreamLoop pierde mucho más dinero por un cliente que cancela sin ser detectado que por uno al que se le ofrece retención sin necesidad.

Por eso el `scoring` de la búsqueda es **`recall`** de la clase positiva (`Churn=Yes`), priorizando minimizar falsos negativos.

## Dataset

Telco Customer Churn (IBM), cargado desde:

`https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`

- Target: `Churn` (`Yes` / `No`)
- Features: atributos de cuenta, servicios y facturación

## Estructura del repo

```
.
├── explore.ipynb          # Notebook principal (baseline + búsqueda + evaluación)
├── tuning_report.md       # Comparación baseline vs ajustado y justificación
├── src/train_and_tune.py  # Script reproducible del mismo flujo
├── requirements.txt
└── learn.json
```

## Cómo ejecutar

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Opción A: notebook
jupyter notebook explore.ipynb

# Opción B: script
python src/train_and_tune.py
```

El script guarda un resumen en `artifacts/metrics_summary.json`.

## Resultados resumidos

| Modelo | Recall (Churn) | Precision (Churn) | F1 | Accuracy |
|--------|----------------|-------------------|----|----------|
| Baseline (defaults) | 0.481 | 0.623 | 0.543 | 0.785 |
| Ajustado (selección estable) | **0.778** | 0.497 | 0.606 | 0.732 |

El recall de churn sube ~30 puntos porcentuales. La precisión baja (más falsos positivos), un trade-off alineado con la prioridad de negocio.

Detalle y justificación en [`tuning_report.md`](tuning_report.md).

## Requisitos de evaluación cubiertos

- Preprocesado **dentro** del `Pipeline`
- Baseline con defaults antes del tuning
- `RandomizedSearchCV` + `GridSearchCV` con CV; el test no participa en la búsqueda
- `scoring` justificado por el negocio (`recall`)
- Inspección de `cv_results_` (media y std entre folds)
- Comparación baseline vs ajustado en el mismo set de métricas
