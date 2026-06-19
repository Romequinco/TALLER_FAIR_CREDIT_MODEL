"""Tarea 2 — Aprendizaje justo (FAIR loss).

Función de coste customizada para el entrenamiento justo del clasificador.

La loss combina dos términos:
  1. El error de clasificación habitual (p. ej. entropía cruzada binaria sobre TARGET).
  2. Una penalización por dependencia estadística entre la variable predicha y la
     variable sensible (CODE_GENDER), ponderada por un coeficiente de fairness que
     controla el trade-off precisión <-> justicia.

Aquí vivirán la definición de la métrica de dependencia escogida y la función/clase
de pérdida combinada. Sin implementación todavía.
"""

# import tensorflow as tf
# from tensorflow import keras
