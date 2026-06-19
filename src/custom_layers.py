"""Tarea 1 — Arquitectura customizada.

Capa(s) de Keras a medida para el modelo de concesión de créditos.

La capa principal calcula internamente el "Ratio de Endeudamiento" combinando
variables financieras de entrada (p. ej. anualidad sobre ingresos) y aplica una
saturación / restricción matemática sobre ese ratio antes de propagarlo a las
capas densas posteriores. El objetivo es incorporar conocimiento del dominio
(restricción física/matemática) directamente en la arquitectura de la red.

Aquí solo vivirá la definición de la(s) capa(s) customizada(s); el ensamblaje del
modelo completo se hará en otro módulo / notebook.
"""

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
