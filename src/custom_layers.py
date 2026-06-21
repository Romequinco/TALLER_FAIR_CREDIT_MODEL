"""Tarea 1 — Arquitectura customizada.

Capa(s) de Keras a medida para el modelo de concesión de créditos.

La capa principal calcula internamente el "Ratio de Endeudamiento" combinando
variables financieras de entrada (p. ej. anualidad sobre ingresos) y aplica una
saturación / restricción matemática sobre ese ratio antes de propagarlo a las
capas densas posteriores.
"""

import keras

EPS = 1e-7
P_MIN, P_MAX = 0.1, 3.0
# Tras log1p + StandardScaler (NB 02), |income| puede ser ~0: epsilon mayor evita DTI explosivo
DTI_DENOM_EPS = 0.25
DTI_CLIP = 5.0


def _signed_power(x, p):
    """
    Potencia entrenable segura para cualquier x real: sign(x) * |x|^p.

    Necesaria porque income/annuity llegan estandarizados (media 0, ~50% negativos).
    x^p con p no entero y x<0 es NaN en los reales; con p=1 coincide con la identidad.
    """
    return keras.ops.sign(x) * keras.ops.power(keras.ops.abs(x) + EPS, p)


@keras.saving.register_keras_serializable(package="CustomLayers")
class DebtRatioSaturatingLayer(keras.layers.Layer):
    """
    Capa custom DTI + saturación sign(x)*|x|^p (D-1.1 a D-1.6).
    Usa solo keras.ops (backend-agnóstico).
    """

    def __init__(self, income_idx=0, annuity_idx=2, n_features=13, **kwargs):
        super().__init__(**kwargs)
        self.income_idx = int(income_idx)
        self.annuity_idx = int(annuity_idx)
        self.n_features = int(n_features)

    def build(self, input_shape):
        # D-1.3: p init 1.0 => identidad al arrancar
        self.p = self.add_weight(
            name="saturate_exponent",
            shape=(1,),
            initializer=keras.initializers.Constant(1.0),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        p_clipped = keras.ops.clip(self.p, P_MIN, P_MAX)

        income = inputs[:, self.income_idx : self.income_idx + 1]
        annuity = inputs[:, self.annuity_idx : self.annuity_idx + 1]

        # D-1.2 / D-1.4: saturación sobre columnas financieras (signed-power)
        income_sat = _signed_power(income, p_clipped)
        annuity_sat = _signed_power(annuity, p_clipped)

        # D-1.1 / D-1.5: DTI estable sobre proxy ya escalado (|income| + eps, clip)
        dti = keras.ops.divide(annuity, keras.ops.abs(income) + DTI_DENOM_EPS)
        dti = keras.ops.clip(dti, -DTI_CLIP, DTI_CLIP)
        dti_sat = _signed_power(dti, p_clipped)

        # D-1.6: sustituir cols saturadas + concatenar ratio
        cols = []
        for i in range(self.n_features):
            if i == self.income_idx:
                cols.append(income_sat)
            elif i == self.annuity_idx:
                cols.append(annuity_sat)
            else:
                cols.append(inputs[:, i : i + 1])
        inputs_modified = keras.ops.concatenate(cols, axis=-1)
        return keras.ops.concatenate([inputs_modified, dti_sat], axis=-1)

    def compute_output_shape(self, input_shape):
        if input_shape[-1] is not None:
            return (input_shape[0], input_shape[-1] + 1)
        return (input_shape[0], None)

    def get_config(self):
        config = super().get_config()
        config.update({
            "income_idx": self.income_idx,
            "annuity_idx": self.annuity_idx,
            "n_features": self.n_features,
        })
        return config


RatioEndeudamientoLayer = DebtRatioSaturatingLayer
