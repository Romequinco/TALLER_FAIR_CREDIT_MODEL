# Convenciones comunes de los notebooks de modelado (03–07)

> **Para qué sirve este documento.** Los cinco notebooks de modelado
> (`03_modelo_base`, `04_tarea1_capa_custom`, `05_tarea2_fair_loss`,
> `06_tarea3_keras_tuner`, `07_tarea4_incertidumbre`) deben llegar **alineados**:
> misma carga de datos, mismas rutas de salida, mismo estilo, mismo formato del
> bloque de decisiones. Este fichero es la **fuente de verdad de convenciones**;
> cada notebook lo respeta al pie de la letra. No implementa lógica de modelos.
>
> Correlativos a `notebooks/01_EDA.ipynb` y `notebooks/02_preprocesado.ipynb`.
> El contrato de datos `(X, y, s)` y las decisiones `D-P.*` ya están **cerrados**
> en el preprocesado; aquí se **consumen**, no se rediscuten.

---

## (a) Carga de datos estándar

Todos los notebooks cargan `data/processed/` con **este mismo snippet** (verificado
y funcional sobre los parquets presentes). Lee `metadata.json` para no hardcodear la
lista de features, y materializa el contrato `(X, y, s)` por split.

```python
import json
from pathlib import Path
import pandas as pd

# --- Rutas y metadatos (fuente de verdad: metadata.json del preprocesado) ---
PROC_DIR   = Path("../data/processed")                       # relativo a notebooks/
META       = json.loads((PROC_DIR / "metadata.json").read_text(encoding="utf-8"))
FEATURES_X = META["columns"]["features_X"]   # 13 features, en orden
SENSIBLE   = META["columns"]["sensible"]     # "CODE_GENDER"  (s)
TARGET     = META["columns"]["target"]       # "TARGET"       (y)

def cargar_split(nombre):
    """Devuelve (X, y, s) para 'train' | 'val' | 'test'.
    X = DataFrame solo con las 13 features (SIN genero).
    y = Series TARGET (1=impaga, 0=paga).  s = Series CODE_GENDER (M=1/F=0).
    """
    df = pd.read_parquet(PROC_DIR / f"{nombre}.parquet")
    X = df[FEATURES_X]          # input del modelo: el genero NUNCA entra aqui
    y = df[TARGET]
    s = df[SENSIBLE]
    assert SENSIBLE not in X.columns, "FUGA: el genero esta dentro de X"
    return X, y, s

# Materializar los tres cortes
X_train, y_train, s_train = cargar_split("train")
X_val,   y_val,   s_val   = cargar_split("val")
X_test,  y_test,  s_test  = cargar_split("test")

# Resumen de control
print(f"{'split':<7}{'X (filas, cols)':>20}{'y':>12}{'s':>12}{'tasa_impago':>14}")
for n, (X, y, s) in {"train": (X_train, y_train, s_train),
                     "val":   (X_val,   y_val,   s_val),
                     "test":  (X_test,  y_test,  s_test)}.items():
    print(f"{n:<7}{str(tuple(X.shape)):>20}{str(tuple(y.shape)):>12}"
          f"{str(tuple(s.shape)):>12}{y.mean():>14.4%}")
```

**Salida real (ejecutada sobre los parquets actuales):**

```
split       X (filas, cols)           y           s   tasa_impago
train          (215254, 13)   (215254,)   (215254,)       8.0728%
val             (46126, 13)    (46126,)    (46126,)       8.0735%
test            (46127, 13)    (46127,)    (46127,)       8.0734%
```

**Reglas de uso del contrato `(X, y, s)`:**

- `X` tiene **exactamente 13 columnas** y en este orden:
  `AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3,
  DAYS_EMPLOYED, DAYS_BIRTH, EXT_SOURCE_1_missing, EXT_SOURCE_2_missing,
  EXT_SOURCE_3_missing, N_EXT_MISSING, DAYS_EMPLOYED_ANOM`.
- **El género `s` (`CODE_GENDER`) NUNCA es input de predicción.** Solo se usa en:
  (1) la **FAIR loss de la Tarea 2** (notebook 05) como término de penalización de
  dependencia, y (2) la **auditoría de equidad** (group gap M−F, tasas por grupo) en
  cualquier notebook que reporte justicia. La línea base de equidad a batir, fijada por
  el EDA, es **group gap +3,14 pp** (M 10,14 % vs F 7,00 % de impago real).
- Para Keras, convertir a tensores con `X_*.to_numpy(dtype="float32")`,
  `y_*.to_numpy(dtype="float32")`, `s_*.to_numpy(dtype="float32")` en el punto de uso.
- Las features ya vienen **winsorizadas + `log1p` + escaladas** (D-P.3/D-P.5) y los nulos
  **imputados por mediana** con flags `*_missing` conservados (D-P.2). **No** se vuelve a
  escalar ni imputar en los notebooks de modelado.

---

## (b) Figuras y tablas: dónde y con qué nombre

- **Figuras** → `results/figures/`  (formato `.png`)
- **Tablas**  → `results/tables/`   (formato `.csv`)

**Nomenclatura** (prefijo por número de notebook + etiqueta de tarea + descripción):

```
results/figures/<NN>_<tarea>__<descripcion>.png
results/tables/<NN>_<tarea>__<descripcion>.csv
```

donde `<NN>` ∈ {03,04,05,06,07} y `<tarea>` es una etiqueta corta estable
(`base`, `custom`, `fair`, `tuner`, `incert`). Separador `__` (doble guión bajo) entre
bloque y descripción; descripción en `snake_case` sin acentos.

**Ejemplos coherentes con los entregables obligatorios:**

| Notebook | Figuras | Tablas |
|---|---|---|
| **03 base** | `03_base__curva_loss.png`, `03_base__roc_test.png` | `03_base__metricas_test.csv` |
| **04 custom** | `04_custom__curva_loss.png`, `04_custom__exponentes_p_aprendidos.png` | `04_custom__metricas_test.csv` |
| **05 fair** | `05_fair__curva_loss.png`, `05_fair__group_gap_vs_lambda.png` | `05_fair__base_vs_mejor_fair.csv`, `05_fair__barrido_lambda.csv` |
| **06 tuner** | `06_tuner__pareto_auc_vs_gap.png` (E2 oficial), `06_tuner__pareto_dependencia.png` (E2 literal), `06_tuner__curva_loss_mejor.png`, `06_tuner__curvas_loss_lambda.png` (E4) | `06_tuner__pareto_puntos.csv`, `06_tuner__trials.csv`, `06_tuner__base_vs_mejor_fair.csv` (E5) |
| **07 incert** | `07_incert__varianza_buen_vs_mal_pagador.png`, `07_incert__varianza_vs_n_ext_missing.png` | `07_incert__incertidumbre_test.csv` |

> **Inventario completo del NB06** (Tarea 3, terminada e iterada — la tabla de arriba
> solo muestra los artefactos cabecera). El notebook produce ahora:
>
> *Figuras* (`results/figures/`):
> - `06_tuner__pareto_auc_vs_gap.png` — **E2 oficial**: Pareto del tuner (topología
>   variable, medida `corr²`), eje Y = AUC, eje X = `|group gap|`.
> - `06_tuner__pareto_dependencia.png` — **E2 literal**: eje X = `D(ŷ, género)`, la
>   dependencia que de verdad penaliza la loss (lo que el enunciado pide minimizar).
> - `06_tuner__pareto_3medidas.png` — comparación de las **3 medidas** (`corr²`/`HSIC`/`MMD`)
>   sobre backbone fijo, con la **estrella del compromiso** elegido.
> - `06_tuner__pareto_limpia_semillas.png` — Pareto limpia `corr²` con **barras de error
>   multi-semilla** (topología fija, aísla el efecto de λ del ruido de entrenamiento).
> - `06_tuner__equalized_odds_base_vs_fair.png` — barras `ΔTPR`/`ΔFPR` base vs FAIR.
> - `06_tuner__curva_loss_mejor.png` — **E4**: convergencia del modelo de compromiso.
> - `06_tuner__curvas_loss_lambda.png` — **E4**: panel de convergencia de todos los λ del barrido.
>
> *Tablas* (`results/tables/`):
> - `06_tuner__pareto_puntos.csv` — puntos del frente.
> - `06_tuner__barrido_3medidas_seeds.csv` — barrido por medida y λ con `dep_mean`/`dep_std`,
>   `gap`, `auc`, `ΔTPR`/`ΔFPR`.
> - `06_tuner__pareto_limpio_seeds.csv` — barrido `corr²` topología fija multi-semilla.
> - `06_tuner__trials.csv` — `hp_values` completos, `measure`, estrategia de cada trial.
> - `06_tuner__base_vs_mejor_fair.csv` — **E5**: base 03 + base tuner (λ=0) + mejor FAIR,
>   con group gap y equalized-odds.
>
> *Modelo (handoff a la Tarea 4)* (`data/models/`):
> - `06_modelo_compromiso.weights.h5` + `.json` (backbone, `lambda_star`, `measure_star`,
>   `fair_source`). **Compromiso elegido: medida `corr²`, λ\* = 5.**

> Nota: la tabla **"base vs mejor FAIR en test"** (entregable obligatorio) la produce el
> notebook **05** (`05_fair__base_vs_mejor_fair.csv`), porque necesita ambos números a la
> vez; el modelo base lo aporta el notebook 03 (ver mapa en la sección (e)).

**Snippet mínimo de guardado** (usar siempre estas rutas y parámetros):

```python
from pathlib import Path
FIG_DIR = Path("../results/figures"); FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR = Path("../results/tables");  TAB_DIR.mkdir(parents=True, exist_ok=True)

# Figura
fig.savefig(FIG_DIR / "06_tuner__pareto_auc_vs_gap.png", bbox_inches="tight", dpi=150)

# Tabla
df_resultados.to_csv(TAB_DIR / "05_fair__base_vs_mejor_fair.csv", index=False)
```

---

## (c) Bloque "Decisiones a tomar antes de empezar"

Cada notebook **abre** (celda markdown nº 0, justo bajo el título) con una tabla que
replica el estilo de la celda 0 de `02_preprocesado.ipynb`. Copia **solo las fichas de su
tarea** desde `docs/DECISIONES.md`, con su **ESTADO REAL**. Esas decisiones **ya se tomaron
y están registradas en `docs/DECISIONES.md`**: tras ejecutar las cuatro tareas, la inmensa
mayoría quedó **Confirmada** (o **Resuelta**, como D-2.1), y solo **D-2.7** sigue **Abierta**
(σ del kernel HSIC/MMD, que no afecta al compromiso final con `corr²`). El código ya está
escrito y ejecutado; el bloque (c) **documenta** el estado en que cerró cada ficha, no abre
una validación previa.

**Plantilla exacta:**

```markdown
## Decisiones de esta tarea (estado real)

> Fichas de `docs/DECISIONES.md` para esta tarea. **Estado real** copiado tal cual.
> Las decisiones ya se tomaron y se ejecutaron; casi todas quedan **Confirmada**
> (o **Resuelta**). Si alguna sigue **Abierta**, se indica la limitación declarada.

| Decisión | Opciones | Estado |
|---|---|---|
| **D-x.1** · <título corto> | <opción A> / <opción B> / ... | Confirmada |
| **D-x.2** · <título corto> | <opción A> / <opción B> / ... | Abierta |
```

**Ejemplo de fila real (Tarea 1, ficha D-1.1):**

```markdown
| **D-1.1** · Columnas del ratio de endeudamiento | (a) AMT_ANNUITY/AMT_INCOME_TOTAL (DTI) / (b) AMT_CREDIT/AMT_INCOME_TOTAL / (c) varios ratios | Confirmada |
```

Estados válidos (idénticos a `DECISIONES.md`): **Propuesta**, **Confirmada**,
**Resuelta**, **Revisar**, **Abierta**. Citar el código de ficha (`D-1.1`, `D-2.3`, …) sin renombrar.

---

## (d) Estilo / paleta heredados y celda de Setup

**Paleta semántica** (heredada del EDA / preprocesado, NO cambiar):

| Constante | Hex | Significado |
|---|---|---|
| `COLOR_PAGA`   | `#2c7fb8` | azul — paga (TARGET=0, "Buen pagador") |
| `COLOR_IMPAGA` | `#d7301f` | rojo — impaga (TARGET=1, "Mal pagador") |
| `COLOR_ACENTO` | `#41ab5d` | verde — neutro / acento |

**Convenciones de redacción:** todo el contenido (markdown y comentarios) en **español**;
markdown con encabezados claros, negritas para conceptos, y referencias a fichas
`D-x.y` y documentos `docs/teoria/` cuando se justifique una decisión.

**Celda de Setup común** (segunda celda de cada notebook, tras el bloque de decisiones):

```python
# === Setup comun (notebooks de modelado 03-07) ===
import os
os.environ["KERAS_BACKEND"] = "tensorflow"   # backend unico para todo el grupo

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reproducibilidad
RNG = 42
np.random.seed(RNG)
import random; random.seed(RNG)
try:
    import keras
    keras.utils.set_random_seed(RNG)
except Exception:
    pass

# Estilo heredado del EDA / preprocesado
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
COLOR_PAGA   = "#2c7fb8"   # TARGET=0  (paga)
COLOR_IMPAGA = "#d7301f"   # TARGET=1  (impaga)
COLOR_ACENTO = "#41ab5d"   # neutro

# Rutas estandar
PROC_DIR = Path("../data/processed")
FIG_DIR  = Path("../results/figures"); FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR  = Path("../results/tables");  TAB_DIR.mkdir(parents=True, exist_ok=True)
```

> El `import` de Keras y el módulo de `src/` correspondiente (ver (e)) se hacen en este
> Setup o en la celda inmediatamente posterior, según notebook. La semilla **RNG=42** es
> la misma del preprocesado (`metadata.json → seed: 42`).

---

## (e) Mapa de entregables y dependencias

**Entregables obligatorios del enunciado** (`Taller_B4_T1.pdf` §5):

- **E1** · Código de entrenamiento, optimización y evaluación de **cada** modelo (genera
  todas las gráficas y tablas).
- **E2** · Scatter / **Curva de Pareto** Precisión (eje Y) vs **Dependencia FAIR** (eje X)
  para distintos valores de fairness (de Keras Tuner).
- **E3** · Gráfico de **distribución de la incertidumbre** (varianza de las predicciones)
  "Buen pagador" vs "Mal pagador".
- **E4** · **Curvas de loss** que muestren convergencia, **para cada entrenamiento final**.
- **E5** · **Tabla "modelo Base (sin FAIR) vs mejor modelo FAIR"** en test, **remarcando el
  valor del mejor modelo en test**.
- *(Presentación)* explicación de la restricción matemática de la capa custom + métrica de
  dependencia de la loss; análisis de la Pareto; reflexión incertidumbre↔`EXT_SOURCE`.

**Asignación notebook → entregable → módulo `src/` → dependencias:**

| NB | Tarea | Entregable(s) que produce | Módulo `src/` | Depende de |
|---|---|---|---|---|
| **03** `modelo_base` | — (referencia) | **E5 (parte base)**, E4 (loss base), E1 | — (sin módulo propio; utilidades comunes: build MLP, métricas) | preprocesado |
| **04** `tarea1_capa_custom` | Tarea 1 | E1, E4; restricción matemática de la capa (presentación) | `custom_layers.py` | 03 (arquitectura/baseline) |
| **05** `tarea2_fair_loss` | Tarea 2 | **E5 (tabla base vs mejor FAIR, completa)**, E4, group gap; métrica de dependencia (presentación) | `fair_loss.py` | **03 (base como referencia)**, 04 (capa, opcional) |
| **06** `tarea3_keras_tuner` | Tarea 3 | **E2 (Pareto AUC vs gap + variante dependencia)**, E4 (loss del compromiso y panel por λ), **E5 (su propia tabla base vs mejor FAIR)**, E1 | `tuning.py` | 04+05 (capa custom + **FAIR loss REAL de la T2** en el `build_model`) |
| **07** `tarea4_incertidumbre` | Tarea 4 | **E3 (varianza buen vs mal pagador)**, varianza↔`EXT_SOURCE`, E1 | `uncertainty.py` | **06 (topología + dropout + modelo de compromiso `data/models/06_modelo_compromiso.*`)** |

**Dependencias clave a respetar:**

1. **03 (base) → 05/06/07.** El **modelo base sin FAIR** (03) es la **referencia** que
   consume la tabla comparativa **"base vs mejor FAIR"** (E5, en el notebook 05). 03 fija el
   AUC/accuracy de partida sobre el que se mide el sacrificio de la justicia. 06 y 07 también
   lo usan como punto de comparación de precisión.

2. **Cruce D-4.1 (MC-Dropout, Tarea 4) ↔ D-3.2 (dropout en el espacio de búsqueda,
   Tarea 3) — VIGENTE.** El **dropout es la palanca compartida**: el tuner (06) incluye
   `dropout` y su `rate` en el espacio de búsqueda **sí o sí** (D-3.2, ver `make_build_model`
   en `tuning.py`) **porque** la Tarea 4 (07) lo reutiliza para **MC-Dropout** (`training=True`
   en inferencia, T pasadas → `Var[p]`, D-4.1). Es decir, 07 **hereda la topología con dropout**
   que 06 deja fijada y **carga `data/models/06_modelo_compromiso.weights.h5` + `.json`**
   (backbone + `lambda_star`/`measure_star`/`fair_source`) como punto de partida del MC-Dropout;
   no introduce un dropout nuevo desacoplado.

3. **El NB06 consume la FAIR loss REAL de la Tarea 2 (no el fallback).** `tuning.py` toma la
   medida de dependencia (`corr²`/`HSIC`/`MMD`) directamente del registro
   `src.fair_loss.DEPENDENCE_MEASURES` (Tarea 2 ya entregada y verificada) vía `resolve_fair_loss`,
   y la combina con su BCE ponderada por clases. El **fallback local `corr2_dependency`** queda
   solo como red de seguridad si `src/fair_loss.py` no estuviera disponible; en condiciones
   normales `fair_source = "src.fair_loss:<measure>"`, no `"fallback:corr2"`. El **compromiso
   elegido** para el handoff es **medida `corr²`, λ\* = 5**.

4. **Mapa `src/` ↔ tarea:** `custom_layers.py`↔04, `fair_loss.py`↔05, `tuning.py`↔06,
   `uncertainty.py`↔07. El **03 base no tiene módulo propio**: define un MLP estándar y
   utilidades comunes (construcción del modelo, cálculo de AUC/accuracy, group gap) que las
   demás tareas reutilizan.

**¿Algún entregable obligatorio se queda sin notebook?** No. Cobertura completa:

- E1 → todos (03–07).  E2 → 06.  E3 → 07.  E5 → 03 (base) + 05 (tabla oficial) + 06
  (su propia `06_tuner__base_vs_mejor_fair.csv`: base 03 + base tuner λ=0 + mejor FAIR).
- **E4 (curvas de loss "para cada entrenamiento final")** es **transversal**: lo deben
  producir **todos** los notebooks que entrenen un modelo final (03, 04, 05, 06 —`curva_loss_mejor`
  del compromiso + `curvas_loss_lambda` con el panel de todo el barrido—, 07). **Aviso para las
  siguientes tandas:** no olvidar guardar la curva de loss en cada notebook; es el entregable más
  fácil de dejarse huérfano por ser repetitivo.
- Los tres puntos de la **presentación** quedan cubiertos por 04 (restricción matemática),
  05 (métrica de dependencia), 06 (análisis Pareto) y 07 (reflexión incertidumbre).

---

## Anexo · Estados reales de las fichas de decisión (de `DECISIONES.md`)

Para que cada notebook copie su bloque (c) **sin error**. Estados verificados contra
`docs/DECISIONES.md` y su "Resumen de estados".

**Tarea 1 — Capa custom (NB 04)** — 6 fichas, **todas Confirmada** (NB04, 2026-06-23):

| Ficha | Título | Estado |
|---|---|---|
| D-1.1 | Columnas del ratio de endeudamiento (resuelto: DTI `AMT_ANNUITY/(AMT_INCOME_TOTAL+ε)`) | **Confirmada** |
| D-1.2 | Qué saturación aplicar (exponente entrenable `x^p`, aprendido `p≈0,87`) | **Confirmada** |
| D-1.3 | Rango e inicialización de la saturación (`p∈[0.1,3]`, init 1) | **Confirmada** |
| D-1.4 | Posición de la capa custom (primera capa sobre inputs financieros) | **Confirmada** |
| D-1.5 | Divisiones por cero / nulos (epsilon en denominador) | **Confirmada** |
| D-1.6 | Salida de la capa (concatenar ratio → 14 features) | **Confirmada** |

**Tarea 2 — FAIR loss (NB 05)** — 7 fichas, **5 Confirmada + 1 Resuelta (D-2.1) + 1 Abierta (D-2.7)**:

| Ficha | Título | Estado |
|---|---|---|
| D-2.1 | Medida de dependencia en la penalización (elegida **`corr²`**; HSIC/MMD comparadas) | **Resuelta** |
| D-2.2 | Forma de combinar ajuste+penalización (`BCE + λ·D`) | **Confirmada** |
| D-2.3 | Métrica de equidad a reportar (**group gap** + tasas) | **Confirmada** |
| D-2.4 | Métrica de precisión de la Pareto (**AUC-ROC**) | **Confirmada** |
| D-2.5 | Cómo pasar `S` a la loss (concatenar `[y, S]` en `y_true`) | **Confirmada** |
| D-2.6 | Dependencia sobre probabilidad o logit (probabilidad `ŷ`) | **Confirmada** |
| D-2.7 | Batch size y σ del kernel | **Abierta** (σ por defecto; heurística de la mediana pendiente) |

**Tarea 3 — Keras Tuner (NB 06)** — 5 fichas (se añadió D-3.5), **todas Confirmada**:

| Ficha | Título | Estado |
|---|---|---|
| D-3.1 | Estrategia de búsqueda (**RandomSearch**; empate técnico con Hyperband) | **Confirmada** |
| D-3.2 | Hiperparámetros del espacio (incluye **dropout sí o sí** → Tarea 4) | **Confirmada** |
| D-3.3 | Extraer pares (precisión, dependencia) (**bucle externo sobre λ** + barrido limpio multi-semilla) | **Confirmada** |
| D-3.4 | Métrica objetivo del tuner (**`val_auc`**, fairness como eje externo) | **Confirmada** |
| D-3.5 | Integración T2↔T3 (tuner usa la **FAIR loss real** de `src.fair_loss`, no el fallback) | **Confirmada** |

**Tarea 4 — Incertidumbre (NB 07)** — 5 fichas, **todas Confirmada** (NB07, 2026-06-23):

| Ficha | Título | Estado |
|---|---|---|
| D-4.1 | Método para la varianza (**MC-Dropout** T=100 + 2º modelo del error) | **Confirmada** |
| D-4.2 | Número de pasadas / miembros T (**T = 100**, varianza estabilizada antes) | **Confirmada** |
| D-4.3 | Medir calidad de `EXT_SOURCE` (`N_EXT_MISSING` 0–3 + flags `*_missing`) | **Confirmada** |
| D-4.4 | Umbral τ de clasificación (**τ por coste = 0,66**; revisable con matriz de coste real) | **Confirmada** |
| D-4.5 | Descomposición aleatoria/epistémica + calibración (extensión implementada) | **Confirmada** |

**Preprocesado (D-P.1 a D-P.7): las 7 Confirmada** (no se rediscuten; el modelado solo
las consume vía el contrato `(X, y, s)`). D-P.2 cerrada por experimento (gana mediana,
AUC val = 0,728034).

**Modelo base (D-MB.1 a D-MB.5): las 5 Confirmada** (Grupo, 2026-06-20).

> Total `DECISIONES.md`: **35 fichas → 0 Propuesta · 33 Confirmada · 1 Resuelta (D-2.1) ·
> 0 Revisar · 1 Abierta (D-2.7)**.
