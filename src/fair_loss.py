"""Tarea 2 — Aprendizaje justo (FAIR loss).

Función de coste customizada para el entrenamiento justo del clasificador de
crédito. La pérdida combina dos términos:

    L = BCE(y_hat, TARGET)  +  lambda * D(y_hat, S)

donde:
  * BCE  = entropía cruzada binaria sobre TARGET (1 = impaga, 0 = paga).
  * D    = medida de dependencia estadística entre la predicción y la variable
           sensible S = CODE_GENDER (M = 1 / F = 0).
  * lambda = peso del trade-off precisión <-> justicia. lambda = 0 recupera el
           modelo base; lambda alto fuerza independencia a costa de precisión.

Decisiones del grupo aplicadas aquí (ver docs/DECISIONES.md):
  * D-2.1 / D-2.3 : se PENALIZA con correlacion^2 (principal) o HSIC (extra),
                    pero la JUSTICIA se REPORTA con el group gap, mas legible
                    para S binaria.
  * Operamos sobre la PROBABILIDAD y_hat (salida sigmoide), no sobre el logit:
                    es coherente con lo que se reporta (group gap), esta acotada
                    en [0,1] y es la cantidad interpretable.
  * Backend unico del grupo: TensorFlow. Todo en float32 y con keras.ops para
                    que la penalizacion sea diferenciable.

Patron de paso de S a la loss (D-2.5):
  Keras exige la firma (y_true, y_pred). Empaquetamos el genero JUNTO al target
  en y_true como una matriz de 2 columnas  y_true = [[TARGET, S], ...]  y lo
  desempaquetamos dentro de la loss. Asi S viaja por batches alineado con y_hat
  sin ser nunca una entrada de prediccion del modelo.
"""

from __future__ import annotations

import keras
from keras import ops

# Epsilon de estabilidad (evita /0 en desviaciones tipicas y logs).
_EPS = 1e-7


# ---------------------------------------------------------------------------
# 0. Utilidad: desempaquetar [TARGET, S] de y_true
# ---------------------------------------------------------------------------
def split_target_sensible(y_true):
    """Separa y_true = [[TARGET, S], ...] en (y, s), ambos con forma (batch, 1).

    Si y_true llega con una sola columna (entrenamiento SIN fairness), s se
    devuelve como None para que la loss se reduzca a BCE puro.
    """
    y_true = ops.cast(y_true, "float32")
    if y_true.shape[-1] is not None and y_true.shape[-1] >= 2:
        y = y_true[:, 0:1]
        s = y_true[:, 1:2]
        return y, s
    return y_true, None


# ---------------------------------------------------------------------------
# 1. Medidas de dependencia D(y_hat, S)  (diferenciables, sobre el batch)
# ---------------------------------------------------------------------------
def dependence_corr2(y_pred, s):
    """Correlacion de Pearson AL CUADRADO entre la prediccion y el genero.

    Para S binaria, r^2 captura el desplazamiento de medias entre grupos, que
    es esencialmente toda la dependencia posible. Barata, estable y diferenciable
    (solo usa momentos del batch). Es la penalizacion PRINCIPAL del grupo.

    Devuelve un escalar en [0, 1]: 0 = sin dependencia lineal, 1 = dependencia
    lineal total.
    """
    y_pred = ops.cast(y_pred, "float32")
    s = ops.cast(s, "float32")

    yp = ops.reshape(y_pred, (-1,))
    ss = ops.reshape(s, (-1,))

    yp_c = yp - ops.mean(yp)
    ss_c = ss - ops.mean(ss)

    cov = ops.mean(yp_c * ss_c)
    std_yp = ops.sqrt(ops.mean(yp_c * yp_c) + _EPS)
    std_ss = ops.sqrt(ops.mean(ss_c * ss_c) + _EPS)

    r = cov / (std_yp * std_ss)
    return r * r


def _rbf_kernel(z, sigma):
    """Matriz kernel RBF (Gram) n x n de un vector columna z."""
    z = ops.reshape(z, (-1, 1))
    # Distancias al cuadrado por pares: ||zi - zj||^2
    sq = ops.sum(z * z, axis=1, keepdims=True)
    d2 = sq - 2.0 * ops.matmul(z, ops.transpose(z)) + ops.transpose(sq)
    d2 = ops.maximum(d2, 0.0)
    return ops.exp(-d2 / (2.0 * sigma * sigma + _EPS))


def dependence_hsic(y_pred, s, sigma_y=0.1, sigma_s=0.5):
    """HSIC (Hilbert-Schmidt Independence Criterion) con kernels RBF.

    Captura dependencia de CUALQUIER forma (lineal y no lineal); con kernel
    caracteristico, HSIC = 0 solo bajo independencia. Mas potente que corr^2
    pero mas caro (matrices n x n) y mas ruidoso en batches pequenos.

    Se ofrece como EXTRA / comparacion. Para S binaria su ventaja sobre corr^2
    es marginal (el desplazamiento de medias ya lo capta la correlacion).

    Nota: los anchos de banda sigma se fijan por defecto; para un ajuste fino se
    usaria la heuristica de la mediana de las distancias por pares.
    """
    y_pred = ops.cast(y_pred, "float32")
    s = ops.cast(s, "float32")

    n = ops.shape(y_pred)[0]
    n_f = ops.cast(n, "float32")

    Ky = _rbf_kernel(y_pred, sigma_y)
    Ks = _rbf_kernel(s, sigma_s)

    # Matriz de centrado H = I - (1/n) 1 1^T
    H = ops.eye(n) - ops.ones((n, n)) / n_f

    KyH = ops.matmul(Ky, H)
    KsH = ops.matmul(Ks, H)
    hsic = ops.trace(ops.matmul(KyH, KsH)) / ((n_f - 1.0) ** 2 + _EPS)
    return hsic


def dependence_mmd(y_pred, s, sigma=0.15):
    """MMD^2 entre las distribuciones del score de los dos grupos (S=1 vs S=0).

    Para S BINARIA es la medida natural y la mas a medida del problema: en vez de
    correlacionar y_hat con S (corr^2) o medir dependencia generica (HSIC), compara
    DIRECTAMENTE  P(y_hat | S=1)  contra  P(y_hat | S=0)  con un test kernel de dos
    muestras (Maximum Mean Discrepancy). MMD^2 = 0 sii ambas distribuciones de score
    coinciden, asi que penalizarla empuja a que hombres y mujeres reciban la MISMA
    distribucion de puntuaciones (paridad estadistica), no solo la misma media.

    Ventaja sobre HSIC para S binaria: HSIC se diluye con el desbalance de grupos
    (apenas registra ~0.02), mientras que MMD apunta justo a la diferencia de
    distribuciones entre los dos grupos. Es diferenciable (solo matmuls sobre el
    kernel RBF de las predicciones) y, escrita con pesos de pertenencia, no necesita
    enmascarar por grupo (compatible con el grafo de Keras/TF).

    Formulacion con pesos (w1 = S, w0 = 1-S) sobre el kernel K de las predicciones:
        MMD^2 = w1^T K w1 / n1^2  +  w0^T K w0 / n0^2  -  2 * w1^T K w0 / (n1 n0).
    """
    y_pred = ops.cast(y_pred, "float32")
    s = ops.reshape(ops.cast(s, "float32"), (-1, 1))

    K = _rbf_kernel(y_pred, sigma)          # n x n sobre las PREDICCIONES
    w1 = s                                   # pertenencia al grupo S=1 (columna)
    w0 = 1.0 - s                             # pertenencia al grupo S=0
    n1 = ops.sum(w1) + _EPS
    n0 = ops.sum(w0) + _EPS

    k11 = ops.sum(ops.matmul(K, w1) * w1) / (n1 * n1)
    k00 = ops.sum(ops.matmul(K, w0) * w0) / (n0 * n0)
    k10 = ops.sum(ops.matmul(K, w0) * w1) / (n1 * n0)

    mmd2 = k11 + k00 - 2.0 * k10
    return ops.maximum(mmd2, 0.0)


# Registro de medidas disponibles, para seleccionar por nombre desde el notebook.
DEPENDENCE_MEASURES = {
    "corr2": dependence_corr2,   # lineal: solo desplazamiento de medias
    "hsic": dependence_hsic,     # kernel generico: cualquier dependencia
    "mmd": dependence_mmd,       # dos-muestras: a medida para S binaria
}


# ---------------------------------------------------------------------------
# 2. La FAIR loss combinada (factory -> funcion compatible con Keras)
# ---------------------------------------------------------------------------
def make_fair_loss(lam=0.0, measure="corr2", from_logits=False):
    """Construye la loss  L = BCE(y, y_hat) + lam * D(y_hat, S).

    Parametros
    ----------
    lam : float
        Peso del termino de fairness. lam = 0 -> BCE puro (modelo base).
    measure : str
        Clave en DEPENDENCE_MEASURES: "corr2" (principal) o "hsic" (extra).
    from_logits : bool
        False (por defecto) -> y_pred es la PROBABILIDAD (salida sigmoide).

    Uso
    ---
        model.compile(optimizer="adam", loss=make_fair_loss(lam=2.0))
        # y_train_fair = np.column_stack([y_train, s_train])  -> shape (n, 2)
        model.fit(X_train, y_train_fair, ...)

    La funcion devuelta tiene firma (y_true, y_pred) y desempaqueta S de y_true.
    """
    dep_fn = DEPENDENCE_MEASURES[measure]

    def fair_loss(y_true, y_pred):
        y, s = split_target_sensible(y_true)
        y_pred = ops.cast(y_pred, "float32")

        bce = keras.losses.binary_crossentropy(y, y_pred, from_logits=from_logits)
        bce = ops.mean(bce)

        if s is None or lam == 0.0:
            return bce

        dep = dep_fn(y_pred, s)
        return bce + lam * dep

    fair_loss.__name__ = f"fair_loss_{measure}_lam{lam}"
    return fair_loss


class FairAUC(keras.metrics.AUC):
    """AUC que desempaqueta y_true = [TARGET, S] y mide solo sobre TARGET.

    Necesaria porque, al empaquetar el genero en y_true (D-2.5), las metricas
    estandar de Keras reciben 2 columnas y fallan. Esta toma solo la columna 0.
    """

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.cast(y_true, "float32")
        if y_true.shape[-1] is not None and y_true.shape[-1] >= 2:
            y_true = y_true[:, 0:1]
        return super().update_state(y_true, y_pred, sample_weight)


# ---------------------------------------------------------------------------
# 3. Metricas de JUSTICIA para reportar (NumPy, fuera del grafo de entrenamiento)
# ---------------------------------------------------------------------------
def group_gap(y_pred, s):
    """Group gap = mean(y_hat | S=1) - mean(y_hat | S=0).

    Para CODE_GENDER (M=1 / F=0): diferencia de score medio de impago predicho
    entre hombres y mujeres. Es la lectura PRINCIPAL de (in)justicia del grupo.
    Cercano a 0 = predicciones equilibradas entre grupos.

    Trabaja con arrays NumPy (no es parte de la loss): se usa al evaluar en test.
    """
    import numpy as np

    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()

    m1 = s == 1
    m0 = s == 0
    if m1.sum() == 0 or m0.sum() == 0:
        return float("nan")
    return float(y_pred[m1].mean() - y_pred[m0].mean())


def corr2_np(y_pred, s):
    """Correlacion^2 en NumPy, para reportar la misma medida que se penaliza."""
    import numpy as np

    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()
    if y_pred.std() < 1e-12 or s.std() < 1e-12:
        return 0.0
    r = np.corrcoef(y_pred, s)[0, 1]
    return float(r * r)
