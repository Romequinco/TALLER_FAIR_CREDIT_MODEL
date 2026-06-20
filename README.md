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
   entre precisión y justicia.

4. **Incertidumbre** (`src/uncertainty.py`)
   Modificación del modelo para que la predicción sobre test devuelva tanto la clase
   predicha como la varianza / incertidumbre de la misma.

## Estado del proyecto (2026-06-20)

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
- 🟡 **Tareas 1-4 — esqueletos montados, pendiente implementar.** Los cuatro notebooks restantes
  (`04_tarea1_capa_custom`, `05_tarea2_fair_loss`, `06_tarea3_keras_tuner`, `07_tarea4_incertidumbre`)
  están creados con estructura, imports y **carga de datos `(X, y, s)` verificada** (ejecutan limpio),
  siguiendo [`docs/CONVENCIONES_MODELADO.md`](docs/CONVENCIONES_MODELADO.md). **Falta la lógica de
  modelos** (Tarea 1 capa custom, Tarea 2 FAIR loss, Tarea 3 Keras Tuner, Tarea 4 incertidumbre): los
  módulos de `src/` siguen siendo esqueletos y sus decisiones de diseño (`D-1.x`–`D-4.x`) siguen en
  estado **Propuesta/Abierta**, a validar con el grupo antes de codificar.

## Estructura del repositorio

```
taller-b4-t1-fairness/
├── README.md              # Este fichero
├── .gitignore
├── requirements.txt
├── data/                  # Dataset (CSV solo en local, no versionado — ver data/README.md)
│   ├── README.md          # Cómo obtener el dataset desde Kaggle
│   └── processed/         # (ignorado) train/val/test.parquet + preprocessor.joblib + metadata.json
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
│   ├── 04_tarea1_capa_custom.ipynb   # Tarea 1 — capa custom del ratio — esqueleto
│   ├── 05_tarea2_fair_loss.ipynb     # Tarea 2 — FAIR loss (tabla E5) — esqueleto
│   ├── 06_tarea3_keras_tuner.ipynb   # Tarea 3 — Keras Tuner (Pareto E2) — esqueleto
│   └── 07_tarea4_incertidumbre.ipynb # Tarea 4 — clase + varianza (E3) — esqueleto
├── src/                   # Código fuente del proyecto (módulos aún esqueletos)
│   ├── custom_layers.py   # Tarea 1 — capa del ratio de endeudamiento
│   ├── fair_loss.py       # Tarea 2 — loss con penalización de dependencia
│   ├── tuning.py          # Tarea 3 — búsqueda de topología (Keras Tuner)
│   └── uncertainty.py     # Tarea 4 — predicción con clase + varianza
├── results/               # Salidas reproducibles del código
│   ├── figures/           # Gráficas (Pareto, distribución de incertidumbre, curvas de loss)
│   └── tables/            # Tablas (comparativa modelo base vs. mejor modelo FAIR)
└── report/                # Documento de presentación / entregable PDF
```

## Cómo ejecutar

> _Pendiente._ (Aquí se documentará la instalación de dependencias, la descarga del
> dataset, y los comandos para reproducir el entrenamiento, la optimización y la
> generación de todas las gráficas y tablas del informe.)

```bash
# pip install -r requirements.txt
# ...
```
