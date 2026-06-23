"""Tarea 4 — Incertidumbre: clase + varianza.

Inferencia que cuantifica la incertidumbre de las predicciones del clasificador de
crédito. Dos señales, ambas del material del profe (docs/teoria/04-incertidumbre.md):

  * MC-Dropout (D-4.1): T pasadas con el dropout ACTIVO en inferencia
    (``model(X, training=True)``). La media de las T probabilidades fija la CLASE
    (umbral tau) y la VARIANZA entre pasadas es la incertidumbre epistémica.
  * Segundo modelo del error (entrega base, §3.2): un modelo que predice
    ``error = |p - y|`` a partir de las mismas X (incertidumbre heteroscedástica).

El dropout que reutiliza MC-Dropout es el MISMO que fija el tuner del NB06 (cruce
D-3.2 <-> D-4.1); aquí no se introduce un dropout nuevo.
"""

from __future__ import annotations

import numpy as np
import keras


# --------------------------------------------------------------------------- #
# 1. MC-Dropout: T pasadas estocásticas -> media + varianza de la probabilidad
# --------------------------------------------------------------------------- #
def mc_dropout_predict(model, X, T=50, tau=0.5, seed=42):
    """T pasadas con dropout activo -> (p_bar, Var[p], clase)  (D-4.1).

    ``model(X, training=True)`` deja el dropout encendido en inferencia: cada
    pasada es una sub-red distinta (un "comité" dentro de un modelo). Devuelve un
    dict con la matriz de pasadas y los estadísticos por fila.

    p_bar = (1/T) Σ p_t        -> probabilidad consenso; clase = 1[p_bar > tau]
    var   = (1/T) Σ (p_t-p_bar)^2  -> incertidumbre epistémica (lo que pide el Taller)
    """
    X = np.asarray(X, "float32")
    if seed is not None:
        keras.utils.set_random_seed(int(seed))
    preds = np.empty((int(T), len(X)), dtype="float32")
    for t in range(int(T)):
        preds[t] = np.asarray(model(X, training=True)).ravel()
    p_bar = preds.mean(axis=0)
    var = preds.var(axis=0)
    return {
        "p_bar": p_bar,
        "var": var,
        "std": np.sqrt(var),
        "clase": (p_bar > tau).astype(int),
        "preds": preds,
    }


def estabilidad_T(preds):
    """Estabilidad de la varianza media según crece T (para fijar D-4.2).

    Con la matriz ``preds`` (T, n) ya calculada, devuelve, para cada t, la media de
    la varianza por fila usando solo las primeras t pasadas -> se ve a partir de qué
    T se estabiliza el estimador (no hace falta reentrenar)."""
    T = preds.shape[0]
    ts = np.unique(np.linspace(2, T, min(T - 1, 25)).astype(int))
    var_media = [preds[:t].var(axis=0).mean() for t in ts]
    return np.array(ts), np.array(var_media)


# --------------------------------------------------------------------------- #
# 2. Incertidumbre aleatoria (Bernoulli) y descomposición (D-4.5, extensión)
# --------------------------------------------------------------------------- #
def aleatoric_bernoulli(p_bar):
    """Incertidumbre ALEATORIA de un clasificador binario: Var Bernoulli p(1-p).

    Es la duda "del mundo" (irreducible): p≈0.5 es máximamente incierta aunque el
    modelo sea estable. Coexiste con la epistémica Var[p] (dispersión entre pasadas)
    y se correlaciona con ella, pero no es lo mismo (04-incertidumbre.md §2.5)."""
    p = np.asarray(p_bar, "float64")
    return p * (1.0 - p)


# --------------------------------------------------------------------------- #
# 3. Calibración de la incertidumbre (¿más varianza <-> más error real?)
# --------------------------------------------------------------------------- #
def calibracion_por_cuantil(var, y, p_bar, tau=0.5, n_bins=5):
    """¿La varianza está bien calibrada? Tasa de error real por cuantil de varianza.

    Agrupa test en ``n_bins`` cuantiles de Var[p] y mide la tasa de error
    (clase != y) en cada uno. Si la varianza es informativa, el error sube con ella.
    Devuelve un dict de arrays paralelos."""
    var = np.asarray(var); y = np.asarray(y).astype(int)
    yhat = (np.asarray(p_bar) > tau).astype(int)
    err = (yhat != y).astype(float)
    # bordes por cuantil (únicos para evitar bins vacíos)
    bordes = np.unique(np.quantile(var, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(var, bordes[1:-1]), 0, len(bordes) - 2)
    out = {"bin": [], "var_media": [], "tasa_error": [], "n": []}
    for b in range(len(bordes) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        out["bin"].append(b)
        out["var_media"].append(float(var[m].mean()))
        out["tasa_error"].append(float(err[m].mean()))
        out["n"].append(int(m.sum()))
    return {k: np.array(v) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# 4. Segundo modelo del error (entrega base fiel al profe, §3.2)
# --------------------------------------------------------------------------- #
def build_error_model(n_features=13, lr=1e-3):
    """MLP pequeño que predice ``error = |p - y|`` a partir de X (regresión >=0).

    Salida con ``softplus`` para forzar error >= 0; pérdida MSE. Es la 'forma más
    simple' que pide entregar el profe (heteroscedástica), complementaria a la
    Var[p] epistémica de MC-Dropout."""
    model = keras.Sequential([
        keras.Input(shape=(n_features,), name="entrada_error"),
        keras.layers.Dense(32, activation="relu", name="err_densa_1"),
        keras.layers.Dense(16, activation="relu", name="err_densa_2"),
        keras.layers.Dense(1, activation="softplus", name="err_salida"),
    ], name="modelo_error")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse",
                  metrics=["mae"])
    return model
