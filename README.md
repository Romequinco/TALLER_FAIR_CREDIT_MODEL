# Taller B4-T1 — Diseño de Redes Confiables (Justicia e Incertidumbre)

Diseño, entrenamiento y auditoría de un modelo de clasificación neuronal para la
**concesión de créditos** que sea simultáneamente **preciso**, **justo** (Fair Learning)
y **honesto** respecto a su nivel de confianza (Incertidumbre).

El modelo se entrena sobre el dataset
[**Home Credit Default Risk**](https://www.kaggle.com/competitions/home-credit-default-risk/overview),
centrado en perfiles con poco historial crediticio.

- **Variable objetivo:** `TARGET` — clasificación binaria (1: dificultades de pago, 0: pagó a tiempo).
- **Variable sensible:** `CODE_GENDER` — el modelo **no debe discriminar** en base al género.
- **Variables de entrada:** ingresos, anualidades y puntuaciones de fuentes externas
  (`EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`). Estas últimas tienen valores ausentes
  imputados, clave para el estudio de la incertidumbre.

## Grupo

- Gonzalo de Ramón
- Alonso Díaz
- Oscar Romero

## Las 4 tareas

1. **Arquitectura customizada** (`src/custom_layers.py`)
   Capa customizada que calcula internamente el *Ratio de Endeudamiento* combinando
   variables financieras de entrada y aplica una saturación / restricción matemática
   sobre dicho ratio antes de pasarlo a las capas densas.

2. **Aprendizaje justo — FAIR loss** (`src/fair_loss.py`)
   Función de coste customizada que combina el error de clasificación con una
   penalización por dependencia estadística entre la variable predicha y la sensible
   (`CODE_GENDER`).

3. **AutoML — Keras Tuner** (`src/tuning.py`)
   Búsqueda de la topología óptima de la red con Keras Tuner, explorando el trade-off
   entre precisión y justicia sobre la **frontera de Pareto** (AUC vs dependencia FAIR).

4. **Incertidumbre** (`src/uncertainty.py`)
   Modificación del modelo para que la predicción sobre test devuelva tanto la clase
   predicha como la varianza / incertidumbre de la misma.

## Estado del proyecto (2026-06-23)

Trabajo hecho y lo que falta. El registro completo de decisiones de diseño está en
[`docs/DECISIONES.md`](docs/DECISIONES.md).

- ✅ **Andamiaje del repo y teoría.** Documentación por tarea en
  [`docs/teoria/`](docs/teoria/) (seis documentos: uno por cada una de las 4 tareas +
  apoyo transversal de dependencia y contexto de redes confiables).
- ✅ **Dataset verificado.** Home Credit oficial (307.511 × 122), documentado en
  [`data/README.md`](data/README.md). Solo en local; **no se versiona** (`.gitignore`).
- ✅ **EDA avanzado** ([`notebooks/01_EDA.ipynb`](notebooks/01_EDA.ipynb)), orientado a
  las 4 tareas.
- ✅ **Preprocesado completo** ([`notebooks/02_preprocesado.ipynb`](notebooks/02_preprocesado.ipynb)):
  split train/val/test **sin fuga**, flags de ausencia, centinela `DAYS_EMPLOYED`,
  winsorización, `log1p`, `DAYS_BIRTH` (edad) y contrato de salida `(X, y, s)` por corte.
  Las decisiones de preprocesado **D-P.1 a D-P.7 están Confirmadas**, con la **D-P.2
  cerrada por experimento** (gana la imputación por **mediana**; ver *Anexo D-P.2* del notebook).
- ✅ **Registro de decisiones** en [`docs/DECISIONES.md`](docs/DECISIONES.md).
- ✅ **Modelo base implementado** ([`notebooks/03_modelo_base.ipynb`](notebooks/03_modelo_base.ipynb)):
  MLP de referencia **sin FAIR ni capa custom** (`λ = 0`) — `Dense(64) → Dropout(0.3) → Dense(32) →
  Dropout(0.3) → sigmoide`, compilado con BCE + Adam + métrica AUC, `class_weight` balanced para el
  desbalance, parada por `val_auc` (`restore_best_weights`) y auditoría de equidad por group gap M−F.
  Es la línea base del "base vs mejor FAIR" (E5) y monta el dropout que reusan Keras Tuner (06) y
  MC-Dropout (07). Sus decisiones de diseño **D-MB.1 a D-MB.5 están Confirmadas**.
- ✅ **Tarea 1 — Capa custom implementada** ([`notebooks/04_tarea1_capa_custom.ipynb`](notebooks/04_tarea1_capa_custom.ipynb),
  [`src/custom_layers.py`](src/custom_layers.py)): capa `DebtRatioSaturatingLayer` (DTI + saturación `x^p`
  entrenable) integrada en el MLP. AUC ≈ 0,745 y mejora del group gap respecto al base.
- ✅ **Tarea 2 — FAIR loss implementada** ([`notebooks/05_tarea2_fair_loss.ipynb`](notebooks/05_tarea2_fair_loss.ipynb),
  [`src/fair_loss.py`](src/fair_loss.py)): pérdida customizada `L = BCE(ŷ, TARGET) + λ·D(ŷ, S)` con **tres
  medidas de dependencia** entre la predicción y `CODE_GENDER` — `corr²` (principal), **HSIC** y **MMD** — todas
  diferenciables sobre la probabilidad sigmoide. El género viaja empaquetado en `y_true = [TARGET, S]` (patrón
  `D-2.5`). Alimenta el barrido de `λ` y la tabla E5 "base vs mejor FAIR". Decisiones `D-2.x` **Confirmadas**.
- ✅ **Tarea 3 — Keras Tuner implementada** ([`notebooks/06_tarea3_keras_tuner.ipynb`](notebooks/06_tarea3_keras_tuner.ipynb),
  [`src/tuning.py`](src/tuning.py)): búsqueda de topología con Keras Tuner (**RandomSearch ≈ Hyperband**; empate
  técnico → se elige **RandomSearch**) sobre la capa custom DTI de la Tarea 1. **Backbone elegido:** 1 capa,
  64 unidades, `dropout 0.3`, `relu`, `lr ≈ 0.0072`. Sobre ese backbone se construye la **frontera de Pareto**
  Precisión (AUC) vs Dependencia FAIR **comparando las 3 medidas** (`corr²`/HSIC/MMD) con la FAIR loss real de la
  Tarea 2; el compromiso se elige por **presupuesto de AUC**: **`corr²`, `λ*=5`**.
  **Resultado (test):** **−72 % de group gap** (5.615 → 1.568 pp) por solo **−0.59 pp de AUC** (0.7404 → 0.7345);
  *equalized-odds* también mejora (ΔTPR 8.28 → 1.82, ΔFPR 11.08 → 3.35). Modelo de compromiso persistido para la
  Tarea 4 (MC-Dropout) en [`data/models/06_modelo_compromiso.{weights.h5,json}`](data/models/). Artefactos en
  `results/figures/` y `results/tables/` con prefijo **`06_tuner__`**. Decisiones `D-3.1`–`D-3.4` **Confirmadas**.
- 🟡 **Tarea 4 — Incertidumbre pendiente** ([`notebooks/07_tarea4_incertidumbre.ipynb`](notebooks/07_tarea4_incertidumbre.ipynb),
  [`src/uncertainty.py`](src/uncertainty.py)): hereda el modelo de compromiso del 06 (`data/models/06_modelo_compromiso.*`,
  con dropout 0.3) para MC-Dropout; decisiones `D-4.x` en **Propuesta/Abierta**.

## Estructura del repositorio

```
taller-b4-t1-fairness/
├── README.md              # Este fichero
├── .gitignore
├── requirements.txt
├── data/                  # Dataset (CSV solo en local, no versionado — ver data/README.md)
│   ├── README.md          # Cómo obtener el dataset desde Kaggle
│   ├── processed/         # (ignorado) train/val/test.parquet + preprocessor.joblib + metadata.json
│   └── models/            # Modelos persistidos (06_modelo_compromiso.{weights.h5,json}) para la Tarea 4
├── docs/                  # Documentación y material de estudio
│   ├── INDEX.md           # Índice maestro de fuentes y teoría
│   ├── DECISIONES.md      # Registro único de decisiones de diseño del grupo
│   ├── CONVENCIONES_MODELADO.md # Convenciones comunes de los notebooks 03–07
│   ├── _fuentes/          # Material original sin procesar (ignorado por git)
│   │   ├── clases-master/ # Apuntes y notebooks de las clases del máster
│   │   └── profe-lectures/# Lecturas/slides del profesor
│   └── teoria/            # Documentos de teoría elaborados (uno por tarea + apoyo)
├── notebooks/             # Notebooks de exploración y experimentación
│   ├── 01_EDA.ipynb       # EDA avanzado orientado a las 4 tareas
│   ├── 02_preprocesado.ipynb     # Pipeline de preprocesado sin fuga (D-P.1 a D-P.7)
│   ├── 03_modelo_base.ipynb      # Modelo base sin FAIR (referencia E5) — ✅ implementado
│   ├── 04_tarea1_capa_custom.ipynb   # Tarea 1 — capa custom del ratio — ✅ implementado
│   ├── 05_tarea2_fair_loss.ipynb     # Tarea 2 — FAIR loss 3 medidas (tabla E5) — ✅ implementado
│   ├── 06_tarea3_keras_tuner.ipynb   # Tarea 3 — Keras Tuner + Pareto E2 — ✅ implementado
│   └── 07_tarea4_incertidumbre.ipynb # Tarea 4 — clase + varianza (E3) — 🟡 esqueleto
├── src/                   # Código fuente del proyecto
│   ├── custom_layers.py   # Tarea 1 — capa del ratio de endeudamiento — ✅
│   ├── fair_loss.py       # Tarea 2 — FAIR loss con 3 medidas (corr²/HSIC/MMD) — ✅
│   ├── tuning.py          # Tarea 3 — búsqueda de topología (Keras Tuner) — ✅
│   └── uncertainty.py     # Tarea 4 — predicción con clase + varianza — 🟡
├── results/               # Salidas reproducibles del código
│   ├── figures/           # Gráficas (Pareto 3 medidas, equalized-odds, curvas de loss)
│   └── tables/            # Tablas (Pareto, barridos por semilla, base vs. mejor FAIR)
└── report/                # Documento de presentación / entregable PDF
```

## Cómo ejecutar

### 1. Entorno y dependencias

```bash
python -m venv .venv
# Linux/macOS:  source .venv/bin/activate
# Windows:      .venv\Scripts\activate
pip install -r requirements.txt
```

**Dependencias clave:** **Keras 3 + TensorFlow** (backend único del grupo, todo en `float32`),
**Keras Tuner** (búsqueda de topología de la Tarea 3, requiere `tensorboard` en tiempo de
búsqueda), además de `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `pyarrow` y
`jupyter`. Lista completa en [`requirements.txt`](requirements.txt).

### 2. Dataset

El CSV de Home Credit **no se versiona** (solo en local). Sigue las instrucciones de
[`data/README.md`](data/README.md) para descargarlo de Kaggle y colocarlo donde lo espera el
preprocesado.

### 3. Notebooks (ejecutar en orden 01 → 07)

Cada notebook consume las salidas del anterior; respeta el orden:

| Orden | Notebook | Produce |
| ----- | -------- | ------- |
| 01 | `01_EDA.ipynb` | Diagnóstico del dataset |
| 02 | `02_preprocesado.ipynb` | `data/processed/` — `train/val/test.parquet`, `preprocessor.joblib`, contrato `(X, y, s)` |
| 03 | `03_modelo_base.ipynb` | Modelo base sin FAIR (referencia E5) |
| 04 | `04_tarea1_capa_custom.ipynb` | Capa custom DTI integrada en el MLP |
| 05 | `05_tarea2_fair_loss.ipynb` | FAIR loss (3 medidas) + tabla E5 |
| 06 | `06_tarea3_keras_tuner.ipynb` | Keras Tuner + frontera de Pareto; persiste `data/models/06_modelo_compromiso.*` y artefactos `results/.../06_tuner__*` |
| 07 | `07_tarea4_incertidumbre.ipynb` | (🟡 pendiente) MC-Dropout sobre el modelo de compromiso del 06 |

```bash
jupyter lab    # abrir y ejecutar los notebooks de notebooks/ en orden
```
