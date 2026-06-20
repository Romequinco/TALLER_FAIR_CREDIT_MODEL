# Índice maestro — Taller B4-T1

Índice de navegación del material de estudio y la documentación del taller.

## Inventario de fuentes

Catálogo completo del material original en [`_fuentes/INVENTARIO.md`](_fuentes/INVENTARIO.md)
(93 ficheros: 20 míos en `clases-master/`, 73 del profe en `profe-lectures/`).
Texto plano extraído de PDFs y notebooks pesados en `_fuentes/_extracted/` (mirrors
derivados para lectura; la fuente de verdad sigue siendo el fichero original).

## Documentos de teoría

Síntesis por **tarea del taller** (no por tema del profe), en [`teoria/`](teoria/).

| Documento | Cubre | Estado |
| --------- | ----- | ------ |
| [00-fundamentos-dependencia.md](teoria/00-fundamentos-dependencia.md) | Apoyo transversal: medidas de dependencia (Pearson, HSIC, CKA, MI) y gaussianización que sustentan la FAIR loss | ✅ |
| [01-capa-custom.md](teoria/01-capa-custom.md) | **Tarea 1** — capa custom del Ratio de Endeudamiento con saturación | ✅ |
| [02-fair-loss.md](teoria/02-fair-loss.md) | **Tarea 2** — FAIR loss (error de clasificación + penalización de dependencia con la variable sensible) | ✅ |
| [03-keras-tuner.md](teoria/03-keras-tuner.md) | **Tarea 3** — AutoML / búsqueda de topología con Keras Tuner | ✅ |
| [04-incertidumbre.md](teoria/04-incertidumbre.md) | **Tarea 4** — predicción con clase + varianza (puente profe→MC-Dropout/ensembles) | ✅ |
| [05-contexto-confiabilidad.md](teoria/05-contexto-confiabilidad.md) | Contexto tangencial: robustez, adversarial, regularización ("redes confiables") | ✅ |

## Mapa taller ↔ teoría

| Tarea del taller | Teoría de soporte |
| ---------------- | ----------------- |
| 1 — Capa customizada (ratio de endeudamiento) | [01-capa-custom.md](teoria/01-capa-custom.md) |
| 2 — FAIR loss (dependencia estadística) | [02-fair-loss.md](teoria/02-fair-loss.md) + apoyo en [00-fundamentos-dependencia.md](teoria/00-fundamentos-dependencia.md) |
| 3 — Keras Tuner (topología) | [03-keras-tuner.md](teoria/03-keras-tuner.md) |
| 4 — Incertidumbre (clase + varianza) | [04-incertidumbre.md](teoria/04-incertidumbre.md) |
| — Contexto "redes confiables" (no es tarea) | [05-contexto-confiabilidad.md](teoria/05-contexto-confiabilidad.md) |

## Notebooks (flujo de trabajo)

Pipeline correlativo en [`notebooks/`](../notebooks/): del diagnóstico al modelado de las 4 tareas.
El contrato de datos `(X, y, s)` y las decisiones de preprocesado (`D-P.*`) quedan cerrados en
`02_preprocesado` y se **consumen** (no se rediscuten) en los notebooks de modelado.

| Notebook | Cubre | Estado |
| -------- | ----- | ------ |
| `01_EDA.ipynb` | EDA avanzado orientado a las 4 tareas | ✅ completo |
| `02_preprocesado.ipynb` | Pipeline sin fuga, contrato `(X, y, s)`, decisiones `D-P.1`–`D-P.7` | ✅ completo |
| `03_modelo_base.ipynb` | Modelo base sin FAIR (referencia "base vs mejor FAIR", E5; decisiones `D-MB.1`–`D-MB.5`) | ✅ completo |
| `04_tarea1_capa_custom.ipynb` | **Tarea 1** — capa custom del ratio de endeudamiento (`D-1.x`) | 🟡 esqueleto |
| `05_tarea2_fair_loss.ipynb` | **Tarea 2** — FAIR loss; produce la tabla E5 (`D-2.x`) | 🟡 esqueleto |
| `06_tarea3_keras_tuner.ipynb` | **Tarea 3** — AutoML / Keras Tuner; produce la Pareto E2 (`D-3.x`) | 🟡 esqueleto |
| `07_tarea4_incertidumbre.ipynb` | **Tarea 4** — clase + varianza; produce E3 (`D-4.x`) | 🟡 esqueleto |

> 🟡 **esqueleto** = estructura, imports y carga de datos montados y verificados (ejecutan
> limpio); falta implementar la lógica de modelos. Las convenciones comunes que todos respetan
> (carga `(X, y, s)`, rutas de `results/`, formato del bloque de decisiones, paleta y mapa de
> entregables E1–E5) están en [`CONVENCIONES_MODELADO.md`](CONVENCIONES_MODELADO.md).
>
> Dependencias clave: **03** (base) alimenta la tabla E5 de **05** y la comparación de **06**/**07**;
> el **dropout** del espacio de búsqueda de **06** (`D-3.2`) es la palanca que **07** reutiliza para
> MC-Dropout (`D-4.1`).

## Decisiones de diseño

Registro único de decisiones del grupo (qué se elige en cada hueco abierto y por qué),
consolidando los huecos de `teoria/` y los hallazgos del EDA: [`DECISIONES.md`](DECISIONES.md).
