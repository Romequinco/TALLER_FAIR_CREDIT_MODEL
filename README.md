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

- _[Nombre integrante 1]_
- _[Nombre integrante 2]_
- _[Nombre integrante 3]_

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

## Estructura del repositorio

```
taller-b4-t1-fairness/
├── README.md              # Este fichero
├── .gitignore
├── requirements.txt
├── docs/                  # Documentación y material de estudio
│   ├── INDEX.md           # Índice maestro de fuentes y teoría
│   ├── _fuentes/          # Material original sin procesar
│   │   ├── clases-master/ # Apuntes y notebooks de las clases del máster
│   │   └── profe-lectures/# Lecturas/slides del profesor
│   └── teoria/            # Documentos de teoría elaborados
├── notebooks/             # Notebooks de exploración y experimentación
├── src/                   # Código fuente del proyecto
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
