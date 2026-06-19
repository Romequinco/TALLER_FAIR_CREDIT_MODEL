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
