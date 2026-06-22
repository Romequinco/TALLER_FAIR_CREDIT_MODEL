"""Tarea 3 — AutoML / Keras Tuner.

Búsqueda de la topología óptima de la red (nº de capas, unidades, dropout,
learning rate, activación) con Keras Tuner, y construcción de la **frontera de
Pareto Precisión (AUC) vs Dependencia FAIR (group gap)** barriendo el
coeficiente de fairness ``lambda_fair`` en un bucle externo (D-3.3).

Decisiones implementadas (ver docs/DECISIONES.md):
  - D-3.1  estrategia de búsqueda: RandomSearch / Hyperband (se comparan y elige).
  - D-3.2  espacio: nº capas, unidades/capa, dropout + tasa (SIEMPRE), lr (log),
           activación. ``lambda_fair`` es eje externo, no hiperparámetro del tuner.
  - D-3.3  pares (AUC, group gap) por bucle externo sobre lambda.
  - D-3.4  objetivo del tuner = ``val_auc`` (maximizar).
  - D-2.3  equidad = group gap  Δ = mean(p|M) − mean(p|F).
  - D-2.4  precisión = AUC-ROC.
  - D-2.5  ``s`` (género) entra empaquetado en ``y_true = [y, s]``.

Desacople de la Tarea 2: si ``src/fair_loss.py`` ya define ``fair_loss`` se usa
esa (HSIC del compañero); si no, se usa un **fallback local** corr²(p, s) — el
caso sencillo de D-2.1 — para que el notebook 6 produzca el Pareto igualmente.
"""

from __future__ import annotations

import json
import numpy as np
import keras
from sklearn.metrics import roc_auc_score, accuracy_score

try:
    import keras_tuner as kt
    _HAS_KT = True
except Exception:  # pragma: no cover
    kt = None
    _HAS_KT = False

from src.custom_layers import DebtRatioSaturatingLayer

EPS = 1e-7


# --------------------------------------------------------------------------- #
# 1. Dependencia FAIR (fallback) y resolución de la loss
# --------------------------------------------------------------------------- #
def corr2_dependency(p, s):
    """Dependencia diferenciable corr²(p, s) (D-2.1, caso sencillo).

    Para ``s`` binaria (CODE_GENDER) corr² equivale, salvo escala, a un
    desplazamiento de medias entre grupos → penaliza justo el group gap (D-2.3).
    Vale 0 bajo independencia y es derivable con ``keras.ops``.
    """
    p = keras.ops.reshape(p, (-1,))
    s = keras.ops.cast(keras.ops.reshape(s, (-1,)), p.dtype)
    pm = p - keras.ops.mean(p)
    sm = s - keras.ops.mean(s)
    cov = keras.ops.mean(pm * sm)
    var_p = keras.ops.mean(pm * pm)
    var_s = keras.ops.mean(sm * sm)
    corr = cov / (keras.ops.sqrt(var_p * var_s) + EPS)
    return corr * corr


def make_fair_loss(lambda_fair, w0=1.0, w1=1.0):
    """Fallback local: BCE ponderada (balance de clases, D-MB.3) + λ·corr²(p, s).

    ``y_true`` llega empaquetado como ``[y, s]`` (D-2.5); ``y_pred`` es la
    probabilidad P(impago) (D-2.6). El término de fairness es un estadístico de
    batch, por eso la loss devuelve un escalar y el balance de clases se aplica
    DENTRO (no vía ``class_weight``, incompatible con el ``y_true`` empaquetado).
    """
    lambda_fair = float(lambda_fair)

    def loss(y_true, y_pred):
        y = keras.ops.cast(y_true[:, 0], "float32")
        s = y_true[:, 1]
        p = keras.ops.clip(keras.ops.reshape(y_pred, (-1,)), EPS, 1.0 - EPS)
        w = y * w1 + (1.0 - y) * w0
        bce = -(y * keras.ops.log(p) + (1.0 - y) * keras.ops.log(1.0 - p))
        bce = keras.ops.mean(w * bce)
        if lambda_fair == 0.0:
            return bce
        return bce + lambda_fair * corr2_dependency(p, s)

    return loss


def resolve_fair_loss(lambda_fair, w0=1.0, w1=1.0):
    """Devuelve (loss_callable, fuente). Prioriza la fair_loss del compañero B.

    Solo cae al fallback si ``fair_loss`` NO existe todavía (ImportError/AttributeError).
    Si existe pero falla en ejecución, el error sale a la luz durante el entrenamiento
    (no se enmascara). Nota: el fallback aplica balance de clases (w0/w1) dentro de la
    BCE; la HSIC externa se invoca SIN w0/w1 -> al enchufarla, unificar el tratamiento
    del desbalance para que la escala de λ sea comparable (ver H3 de la revisión)."""
    try:
        from src.fair_loss import fair_loss as _ext  # type: ignore
    except (ImportError, AttributeError):
        return make_fair_loss(lambda_fair, w0, w1), "fallback:corr2"
    return (lambda yt, yp: _ext(yt, yp, float(lambda_fair))), "src.fair_loss"


def fair_loss_source():
    """Diagnóstico: ¿qué dependencia se usa ('src.fair_loss' o fallback:corr2)?"""
    return resolve_fair_loss(1.0)[1]


# --------------------------------------------------------------------------- #
# 2. Métrica AUC que desempaqueta y_true = [y, s]
# --------------------------------------------------------------------------- #
class SlicedAUC(keras.metrics.AUC):
    """AUC que ignora la columna ``s`` del ``y_true`` empaquetado (usa y_true[:,0])."""

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true[:, 0:1], y_pred, sample_weight)


# --------------------------------------------------------------------------- #
# 3. build_model(hp) — espacio de búsqueda (D-3.2)
# --------------------------------------------------------------------------- #
def make_build_model(lambda_fair, n_features=13, include_custom_layer=True,
                     class_weights=(1.0, 1.0), units_max=128):
    """Fábrica de ``build_model(hp)`` con ``lambda_fair`` fijo (eje externo, D-3.3).

    ``units_max`` amplía el rango de unidades para que el tuner pueda alcanzar la
    capacidad del modelo base 03 (64→32) y no quede dominado por él (ver crítica
    'el afinado no bate al base')."""
    w0, w1 = float(class_weights[0]), float(class_weights[1])

    def build_model(hp):
        n_layers = hp.Int("n_layers", 1, 3)
        activation = hp.Choice("activation", ["relu", "tanh"])
        dropout_rate = hp.Float("dropout_rate", 0.1, 0.5, step=0.1)  # dropout SIEMPRE (D-3.2↔D-4.1)
        lr = hp.Float("lr", 1e-4, 1e-2, sampling="log")

        inputs = keras.Input(shape=(n_features,), name="entrada")
        x = inputs
        if include_custom_layer:
            x = DebtRatioSaturatingLayer(
                income_idx=0, annuity_idx=2, n_features=n_features,
                name="capa_dti_saturacion",
            )(x)
        for i in range(n_layers):
            units = hp.Int(f"units_{i}", 16, units_max, step=16)
            x = keras.layers.Dense(units, activation=activation, name=f"densa_{i}")(x)
            x = keras.layers.Dropout(dropout_rate, name=f"dropout_{i}")(x)
        out = keras.layers.Dense(1, activation="sigmoid", name="salida")(x)

        model = keras.Model(inputs, out, name=f"mlp_tuner_lam{lambda_fair}")
        loss, _ = resolve_fair_loss(lambda_fair, w0, w1)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=loss,
            metrics=[SlicedAUC(name="auc")],
        )
        return model

    return build_model


# --------------------------------------------------------------------------- #
# 4. Utilidades de datos, evaluación y Pareto
# --------------------------------------------------------------------------- #
def pack_ys(y, s):
    """Empaqueta y_true = [y, s] como float32 (D-2.5)."""
    return np.column_stack([np.asarray(y, "float32"), np.asarray(s, "float32")])


def class_weights_balanced(y):
    """class_weight 'balanced' desde train (D-MB.3) -> (w0, w1)."""
    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight("balanced", classes=np.array([0, 1]),
                              y=np.asarray(y).astype(int))
    return float(cw[0]), float(cw[1])


def evaluate(model, X, y, s):
    """Devuelve métricas (AUC D-2.4, group gap D-2.3, accuracy) y las probas."""
    p = model.predict(np.asarray(X, "float32"), verbose=0).ravel()
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(int)
    auc = roc_auc_score(y, p)
    gap_pp = (p[s == 1].mean() - p[s == 0].mean()) * 100.0
    acc = accuracy_score(y, (p >= 0.5).astype(int))
    return {"auc": float(auc), "gap_pp": float(gap_pp), "accuracy": float(acc)}, p


def pareto_front(points):
    """Frente no dominado: maximizar AUC, minimizar |group gap|.

    ``points`` = lista de dicts con 'auc' y 'gap_pp'. Devuelve los no dominados
    ordenados por |gap_pp| (de más justo a menos)."""
    nd = []
    for i, a in enumerate(points):
        dominated = False
        for j, b in enumerate(points):
            if i == j:
                continue
            mejor_o_igual = b["auc"] >= a["auc"] and abs(b["gap_pp"]) <= abs(a["gap_pp"])
            estricto = b["auc"] > a["auc"] or abs(b["gap_pp"]) < abs(a["gap_pp"])
            if mejor_o_igual and estricto:
                dominated = True
                break
        if not dominated:
            nd.append(a)
    return sorted(nd, key=lambda d: abs(d["gap_pp"]))


# --------------------------------------------------------------------------- #
# 5. Tuner + barrido de lambda (D-3.1, D-3.3)
# --------------------------------------------------------------------------- #
def build_tuner(build_model_fn, strategy="hyperband", directory="kt_dir",
                project_name="tune", max_epochs=15, max_trials=8, factor=3,
                seed=42, overwrite=True):
    """Crea el tuner (objective = val_auc max, D-3.4). strategy in {hyperband, random}."""
    if not _HAS_KT:
        raise ImportError("keras-tuner no está instalado: pip install keras-tuner")
    objective = kt.Objective("val_auc", direction="max")
    common = dict(objective=objective, directory=directory,
                  project_name=project_name, overwrite=overwrite, seed=seed)
    if strategy == "hyperband":
        return kt.Hyperband(build_model_fn, max_epochs=max_epochs, factor=factor, **common)
    if strategy == "random":
        return kt.RandomSearch(build_model_fn, max_trials=max_trials, **common)
    raise ValueError(f"strategy desconocida: {strategy}")


def _search_arrays(X_train, y_train, s_train, X_val, y_val, s_val, subsample=None, seed=42):
    Xtr = np.asarray(X_train, "float32")
    ytr_pack = pack_ys(y_train, s_train)
    if subsample is not None and subsample < len(Xtr):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(Xtr), size=subsample, replace=False)
        Xtr, ytr_pack = Xtr[idx], ytr_pack[idx]
    Xvl = np.asarray(X_val, "float32")
    yvl_pack = pack_ys(y_val, s_val)
    return Xtr, ytr_pack, Xvl, yvl_pack


def tune_one_lambda(lam, data, strategy="hyperband", n_features=13,
                    include_custom_layer=True, class_weights=(1.0, 1.0),
                    max_epochs=15, max_trials=8, factor=3, final_epochs=40,
                    search_subsample=None, batch_size=512, directory="kt_dir",
                    seed=42, units_max=128, test=None, verbose=0):
    """Busca topología para un ``lam`` fijo, reentrena el mejor y evalúa en val.

    ``data`` = (X_train, y_train, s_train, X_val, y_val, s_val).
    ``test`` = (X_test, y_test, s_test) opcional -> añade métricas de test al registro.
    Devuelve (registro_dict, best_model, history)."""
    X_train, y_train, s_train, X_val, y_val, s_val = data
    Xtr, ytr_pack, Xvl, yvl_pack = _search_arrays(
        X_train, y_train, s_train, X_val, y_val, s_val, search_subsample, seed)

    build_fn = make_build_model(lam, n_features, include_custom_layer, class_weights, units_max)
    tag = f"{strategy}_lam{str(lam).replace('.', 'p')}"
    tuner = build_tuner(build_fn, strategy=strategy, directory=directory,
                        project_name=tag, max_epochs=max_epochs,
                        max_trials=max_trials, factor=factor, seed=seed)

    stop = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                         patience=6, restore_best_weights=True)
    search_kwargs = dict(validation_data=(Xvl, yvl_pack), batch_size=batch_size,
                         callbacks=[stop], verbose=verbose)
    if strategy != "hyperband":   # Hyperband fija las épocas por bracket; no pasarlas
        search_kwargs["epochs"] = max_epochs
    tuner.search(Xtr, ytr_pack, **search_kwargs)

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Reentreno del mejor en TODO el train (los trials de Hyperband ven pocas épocas)
    best_model = build_fn(best_hp)
    final_stop = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                               patience=10, restore_best_weights=True)
    history = best_model.fit(
        np.asarray(X_train, "float32"), pack_ys(y_train, s_train),
        validation_data=(Xvl, yvl_pack),
        epochs=final_epochs, batch_size=batch_size,
        callbacks=[final_stop], verbose=verbose)

    metrics_val, _ = evaluate(best_model, X_val, y_val, s_val)
    registro = {
        "lambda": float(lam),
        "auc": metrics_val["auc"],
        "gap_pp": metrics_val["gap_pp"],
        "accuracy": metrics_val["accuracy"],
        "n_layers": best_hp.get("n_layers"),
        "units_0": best_hp.get("units_0"),
        "dropout_rate": best_hp.get("dropout_rate"),
        "lr": best_hp.get("lr"),
        "activation": best_hp.get("activation"),
        "hp_values": json.dumps(best_hp.values),   # topología completa reconstruible (todas las units_i)
        "fair_source": fair_loss_source(),
    }
    if test is not None:
        metrics_test, _ = evaluate(best_model, *test)
        registro["auc_test"] = metrics_test["auc"]
        registro["gap_pp_test"] = metrics_test["gap_pp"]
    return registro, best_model, history


def run_lambda_sweep(data, lambdas, strategy="hyperband", **kwargs):
    """Barre ``lambdas`` (D-3.3). Devuelve (lista_registros, dict_modelos, dict_histories)."""
    registros, modelos, histories = [], {}, {}
    for lam in lambdas:
        reg, model, hist = tune_one_lambda(lam, data, strategy=strategy, **kwargs)
        registros.append(reg)
        modelos[lam] = model
        histories[lam] = hist
    return registros, modelos, histories


# --------------------------------------------------------------------------- #
# 6. Barrido LIMPIO: topología FIJA + multi-semilla (aísla λ, da barras de error)
# --------------------------------------------------------------------------- #
def build_fixed_model(hp_values, lambda_fair, n_features=13,
                      include_custom_layer=True, class_weights=(1.0, 1.0)):
    """Construye una topología FIJA (de un dict de hiperparámetros) para un λ dado.

    A diferencia de ``make_build_model`` (que deja que el tuner elija), aquí la
    arquitectura es la MISMA para todos los λ -> el único cambio es λ, así la
    frontera mide el efecto del fairness sin confundirlo con la topología."""
    w0, w1 = float(class_weights[0]), float(class_weights[1])
    n_layers = int(hp_values["n_layers"])
    activation = hp_values["activation"]
    dropout_rate = float(hp_values["dropout_rate"])
    lr = float(hp_values["lr"])

    inputs = keras.Input(shape=(n_features,), name="entrada")
    x = inputs
    if include_custom_layer:
        x = DebtRatioSaturatingLayer(income_idx=0, annuity_idx=2,
                                     n_features=n_features, name="capa_dti_saturacion")(x)
    for i in range(n_layers):
        units = int(hp_values.get(f"units_{i}", hp_values["units_0"]))
        x = keras.layers.Dense(units, activation=activation, name=f"densa_{i}")(x)
        x = keras.layers.Dropout(dropout_rate, name=f"dropout_{i}")(x)
    out = keras.layers.Dense(1, activation="sigmoid", name="salida")(x)
    model = keras.Model(inputs, out)
    loss, _ = resolve_fair_loss(lambda_fair, w0, w1)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss=loss, metrics=[SlicedAUC(name="auc")])
    return model


def sweep_lambda_fixed(data, lambdas, hp_values, seeds=(42, 7, 123),
                       n_features=13, include_custom_layer=True, class_weights=(1.0, 1.0),
                       epochs=40, batch_size=512, test=None, verbose=0):
    """Barre λ con TOPOLOGÍA FIJA (``hp_values``) y varias ``seeds``.

    Para cada λ entrena una vez por semilla y agrega media ± std de (AUC, group gap)
    en validación -> barras de error que muestran si las diferencias entre λ superan
    el ruido de entrenamiento. ``test`` opcional añade media de métricas en test.
    Devuelve lista de dicts (una fila por λ)."""
    X_train, y_train, s_train, X_val, y_val, s_val = data
    Xtr = np.asarray(X_train, "float32"); ytr = pack_ys(y_train, s_train)
    Xvl = np.asarray(X_val, "float32"); yvl = pack_ys(y_val, s_val)

    filas = []
    for lam in lambdas:
        aucs, gaps, aucs_t, gaps_t = [], [], [], []
        for sd in seeds:
            keras.utils.set_random_seed(int(sd))
            model = build_fixed_model(hp_values, lam, n_features,
                                      include_custom_layer, class_weights)
            stop = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                                 patience=8, restore_best_weights=True)
            model.fit(Xtr, ytr, validation_data=(Xvl, yvl), epochs=epochs,
                      batch_size=batch_size, callbacks=[stop], verbose=verbose)
            mv, _ = evaluate(model, X_val, y_val, s_val)
            aucs.append(mv["auc"]); gaps.append(mv["gap_pp"])
            if test is not None:
                mt, _ = evaluate(model, *test)
                aucs_t.append(mt["auc"]); gaps_t.append(mt["gap_pp"])
        fila = {
            "lambda": float(lam),
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "gap_mean": float(np.mean(gaps)), "gap_std": float(np.std(gaps)),
            "n_seeds": len(seeds),
        }
        if test is not None:
            fila["auc_test_mean"] = float(np.mean(aucs_t))
            fila["auc_test_std"] = float(np.std(aucs_t))
            fila["gap_test_mean"] = float(np.mean(gaps_t))
            fila["gap_test_std"] = float(np.std(gaps_t))
        filas.append(fila)
    return filas
