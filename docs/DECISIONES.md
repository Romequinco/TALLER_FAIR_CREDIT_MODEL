# Registro de decisiones de diseño — Taller B4-T1

Este documento es el **log único de decisiones del grupo**. Recoge cada hueco abierto
del taller (los marcados en `docs/teoria/*.md` como "Huecos / decisiones pendientes" y los
que abre el EDA de `notebooks/01_EDA.ipynb`) y fija qué se elige y por qué.

**Cómo se usa:**

- Cada decisión es una ficha con **Opciones**, una **Propuesta razonada** (apoyada siempre en
  una referencia concreta: documento de teoría o hallazgo del EDA) y un **Estado**.
- Estados posibles:
  - **Propuesta** — recomendación por defecto, **pendiente de validar por los tres**.
  - **Confirmada** — el grupo la ha discutido y aceptado (rellenar *Decidido por / fecha*).
  - **Revisar** — se aceptó pero hay que reabrirla si cambian los datos/resultados.
  - **Abierta** — no hay evidencia suficiente para recomendar; se listan opciones sin elegir.
- El campo *Decidido por / fecha* va **vacío**: lo rellena el grupo al confirmar.

> Convención: las referencias `[EDA]` apuntan a `notebooks/01_EDA.ipynb`; las `[Txx]` a
> `docs/teoria/`. Las cifras citadas salen de la ejecución real del EDA sobre
> `data/application_train.csv` (307.511 filas).

---

## Tarea 1 — Capa custom (ratio de endeudamiento)

Fuente de huecos: [`docs/teoria/01-capa-custom.md`](teoria/01-capa-custom.md) §5.

### D-1.1 · Columnas que forman el ratio de endeudamiento
- **Decisión:** qué variables exactas van en numerador/denominador del ratio.
- **Opciones:** (a) `AMT_ANNUITY / AMT_INCOME_TOTAL` (carga de la cuota sobre el ingreso, "DTI");
  (b) `AMT_CREDIT / AMT_INCOME_TOTAL` (apalancamiento total); (c) varios ratios a la vez;
  (d) "todas con todas" (descartada por explosión combinatoria, [T1] §2.1).
- **Propuesta:** **DTI = `AMT_ANNUITY / AMT_INCOME_TOTAL`** como ratio principal, opcionalmente
  acompañado de `AMT_CREDIT/AMT_INCOME_TOTAL`. Porque el [EDA] muestra que el DTI tiene una
  relación **no monótona real con el impago** (joroba por deciles: 7,14 % → 8,85 % en D8 → 8,19 % en D10),
  señal que la red puede explotar, y es una variable limpia (mediana 0,16; p99 0,48). Es además la
  correspondencia que sugiere [T1] §4 ("lo que paga / lo que cobra").
- **Estado:** Propuesta
- **Decidido por / fecha:** _(pendiente)_

### D-1.2 · Qué saturación aplicar
- **Decisión:** qué operación de saturación/restricción usa la capa.
- **Opciones:** (a) **exponente entrenable acotado** `x^p` (lo construido en clase); (b) sigmoide;
  (c) `clip` directo; (d) normalización acotada.
- **Propuesta:** **exponente entrenable `x^p`** sobre las variables financieras sesgadas, porque es
  la saturación que el profesor desarrolla explícitamente ([T1] §2.2) y porque el [EDA] confirma su
  motivo de dominio: el ingreso tiene **skew ≈ 392** (cola larguísima), justo el caso en que un
  exponente `p<1` "uniformiza" la distribución y no malgasta capacidad en la cola.
- **Estado:** Propuesta
- **Decidido por / fecha:** _(pendiente)_

### D-1.3 · Valores de la saturación (rango e inicialización)
- **Decisión:** rango del exponente y valor inicial.
- **Opciones:** rango `[0.1, 3]` con init `1` (valores de clase); otros rangos a ajustar.
- **Propuesta:** **`p ∈ [0.1, 3]`, init `1`** (clip sobre el parámetro), tal como en [T1] §2.2.
  Init a 1 hace que la capa empiece como identidad y el clip evita inestabilidad numérica. El profesor
  los da "por ejemplo", así que quedan como punto de partida ajustable → marcar **Revisar** tras
  los primeros entrenamientos.
- **Estado:** Propuesta
- **Decidido por / fecha:** _(pendiente)_

### D-1.4 · Posición de la capa custom
- **Decisión:** dónde se inserta (antes de las densas, interna, o ambas).
- **Opciones:** (a) saturación sobre inputs crudos al principio; (b) capa interna tras una densa;
  (c) probar ambas.
- **Propuesta:** **saturación sobre los inputs financieros crudos al principio** (donde el sesgo es
  real) **y** dejar abierto probar una variante interna. [T1] §4 avisa de que el exponente como primera
  capa **sobre un ratio ya acotado** aporta poco, pero sobre inputs sesgados (ingreso/anualidad,
  skew 392 en [EDA]) sí uniformiza. El ratio de D-1.1 se calcula en la misma capa custom antes de las densas.
- **Estado:** Propuesta
- **Decidido por / fecha:** _(pendiente)_

### D-1.5 · Tratamiento de divisiones por cero / nulos en el ratio
- **Decisión:** cómo evitar `inf`/`NaN` al dividir.
- **Opciones:** (a) epsilon en el denominador; (b) recorte previo; (c) imputar antes.
- **Propuesta:** **epsilon en el denominador** (`annuity / (income + ε)`) dentro de la capa, en
  `keras.ops`. El [EDA] detecta **12 nulos en `AMT_ANNUITY`** y el ingreso no tiene ceros pero la
  protección es barata; la imputación de nulos se resuelve antes en preprocesado (ver D-P.2).
- **Estado:** Propuesta
- **Decidido por / fecha:** _(pendiente)_

### D-1.6 · Salida de la capa de ratio
- **Decisión:** la capa devuelve solo el/los ratio(s) o las features originales **+** los ratios.
- **Opciones:** (a) solo ratios; (b) concatenar ratios a las dimensiones originales.
- **Propuesta:** **concatenar** el/los ratio(s) a las features originales (no sustituirlas), para no
  perder información que las densas sí usan. Se descarta la versión "todas con todas" por explosión
  combinatoria ([T1] §2.1, §5).
- **Estado:** Propuesta
- **Decidido por / fecha:** _(pendiente)_

---

## Tarea 2 — FAIR loss (penalizar dependencia predicción↔género)

Fuente de huecos: [`docs/teoria/02-fair-loss.md`](teoria/02-fair-loss.md) §5 y
[`00-fundamentos-dependencia.md`](teoria/00-fundamentos-dependencia.md).

> **Premisa validada por el EDA:** borrar `CODE_GENDER` **no** elimina el sesgo. El [EDA] mide que
> `EXT_SOURCE_1` correlaciona **−0,31 con el género** (F 0,546 / M 0,407) y que el group gap bruto
> M−F (+3,14 pp) **casi desaparece (+0,02 pp) al controlar por `EXT_SOURCE_1`**: el género se filtra
> por la variable más predictiva. Esto hace la FAIR loss **necesaria, no opcional**.

### D-2.1 · Qué medida de dependencia usar en la penalización
- **Decisión:** qué `D(ŷ, S)` entra en la loss.
- **Opciones:** Pearson / Spearman / HSIC / distance correlation (dCor) / información mutua (MI).
- **Propuesta:** **HSIC** (o correlación² como base sencilla) **durante el entrenamiento**, porque es
  diferenciable, capta dependencia no lineal y vale 0 solo bajo independencia ([T2] §2). Apoyo del [EDA]:
  Pearson es **ciego a la no-linealidad** (DTI Pearson +0,014 vs **dCor 0,052**, ≈4×), así que una
  medida lineal pura se quedaría corta. **Matiz [T2] §4:** como `CODE_GENDER` es **binaria**, la
  dependencia relevante es esencialmente un desplazamiento de medias, y CKA "apenas registra" (~0,02);
  por eso la penalización con correlación/HSIC es suficiente **y la métrica que se *reporta* es el group
  gap** (ver D-2.3). MI se descarta como término de loss (cara, ávida de datos, no diferenciable directa).
- **Estado:** Propuesta
- **Decidido por / fecha:** _(pendiente)_

### D-2.2 · Forma de combinar ajuste + penalización
- **Decisión:** estructura de la loss total.
- **Opciones:** `L = BCE(ŷ, TARGET) + λ·D(ŷ, S)` (Lagrangiano); alternativas escalarizadas.
- **Propuesta:** **`L = BCE(ŷ, TARGET) + λ·D(ŷ, CODE_GENDER)`**, con `λ` barrido para trazar la frontera
  de Pareto ([T2] §2). `fit` = entropía cruzada binaria sobre `TARGET`; `CODE_GENDER` solo se usa en
  entrenamiento, nunca como entrada de predicción.
- **Estado:** Propuesta
- **Decidido por / fecha:** _(pendiente)_

### D-2.3 · Métrica de equidad a reportar (S binaria)
- **Decisión:** qué número entregamos como "justicia".
- **Opciones:** (a) **group gap** `Δ = mean(ŷ|M) − mean(ŷ|F)`; (b) tasas por grupo (aprobación /
  falso-rechazo); (c) CKA/HSIC residual.
- **Propuesta:** **group gap** como métrica principal **+** tasas por grupo como apoyo. El [EDA] fija la
  **línea base a batir: +3,14 pp** (M 10,14 % vs F 7,00 % de impago real). [T2] §4 recomienda
  explícitamente el group gap para `S` binaria. **Aviso de auditoría:** no medir la equidad solo con
  `EXT_SOURCE_2`, que el [EDA] muestra **neutra al género** (F 0,516 / M 0,511) y daría falsa sensación
  de justicia; el canal real es `EXT_SOURCE_1`.
- **Estado:** Propuesta
- **Decidido por / fecha:** _(pendiente)_

### D-2.4 · Métrica de precisión para la curva de Pareto
- **Decisión:** qué va en el eje "precisión" de la frontera.
- **Opciones:** AUC-ROC / accuracy / KS.
- **Propuesta:** **AUC-ROC**, porque es la métrica oficial de Home Credit y es robusta al fuerte
  desbalance que mide el [EDA] (**8,07 % de impagos, ratio 11,4:1**), donde la accuracy engaña
  (un trivial "siempre paga" acierta el 91,93 %).
- **Estado:** Propuesta
- **Decidido por / fecha:** _(pendiente)_

### D-2.5 · Cómo pasar `S` (género) a la loss
- **Decisión:** mecanismo para que la loss vea `CODE_GENDER` por batch.
- **Opciones:** (a) argumento `q` de `FairModelWrapper` (lib `keras-fairkl`); (b) concatenar `[y, S]`
  en `y_true` y desempaquetar dentro de la loss.
- **Propuesta:** **concatenar `[y, S]` en `y_true`** y desempaquetar en la loss, para no depender de
  una librería externa y controlar el cálculo (patrón clásico de Keras, [T2] §3). Si se prefiere rapidez,
  `keras-fairkl` es válido. Decisión de implementación de bajo riesgo.
- **Estado:** Propuesta
- **Decidido por / fecha:** _(pendiente)_

### D-2.6 · Definir la dependencia sobre la probabilidad o el logit
- **Decisión:** `D` opera sobre `P(TARGET=1)` o sobre el score pre-sigmoide.
- **Opciones:** (a) probabilidad `ŷ = P(TARGET=1)`; (b) logit.
- **Propuesta:** sobre la **probabilidad `ŷ`**, que es la magnitud interpretable y la que define el
  group gap reportado (D-2.3). [T2] §5 lo deja abierto; se elige la probabilidad por coherencia con la métrica.
- **Estado:** Propuesta
- **Decidido por / fecha:** _(pendiente)_

### D-2.7 · Tamaño de batch y ancho de banda σ del kernel
- **Decisión:** cómo estimar la dependencia en mini-batch de forma estable.
- **Opciones:** batch grande vs pequeño; σ del RBF por heurística de la mediana global vs recalculado por batch.
- **Propuesta:** **σ por heurística de la mediana** ([T2] §2) y batch **suficientemente grande** para
  que el estimador de dependencia no sea ruidoso (el ejemplo de Sharpe del profe opera sobre el batch
  completo, [T2] §3). **No hay evidencia en el EDA** sobre el tamaño óptimo concreto → fijar por prueba
  y marcar **Revisar**.
- **Estado:** Abierta (proponer valor tras pruebas)
- **Decidido por / fecha:** _(pendiente)_

---

## Tarea 3 — AutoML / Keras Tuner

Fuente de huecos: [`docs/teoria/03-keras-tuner.md`](teoria/03-keras-tuner.md) §5.
Implementación: [`src/tuning.py`](../src/tuning.py) + [`notebooks/06_tarea3_keras_tuner.ipynb`](../notebooks/06_tarea3_keras_tuner.ipynb).

> **Resultado de la ejecución (2026-06-22).** Frontera de Pareto **AUC vs |group gap|** barriendo
> `λ ∈ {0, 0.5, 1, 2, 5}`. Para evitar confundir el efecto del fairness con el de la topología, además
> del barrido del tuner se hace un **barrido LIMPIO con topología fija (backbone de mayor AUC) y 3
> semillas** (media ± std). **Compromiso elegido: λ\*=0.5** sobre el backbone `1 capa · 80 u · dropout 0.3
> · lr 6.7e-3 · relu`. **Precio de la justicia en test:** `λ=0` AUC 0.7407 / gap +5.73 pp → `λ*=0.5`
> AUC 0.7407 / gap +3.77 pp, es decir **ΔAUC ≈ 0 (dentro de 1σ) reduciendo el gap ~34 %**. La
> dependencia usada es el **fallback `corr²`** (la HSIC de D-2.1 aún no entregada en `src/fair_loss.py`);
> el diseño es **enchufable** y las figuras se marcan *preliminar*. Modelo persistido para el NB07 en
> `data/models/06_modelo_compromiso.*` (cruce D-3.2 ↔ D-4.1). Entregables: `results/figures/06_tuner__pareto_auc_vs_gap.png`
> (tuner), `06_tuner__pareto_limpia_semillas.png` (limpia con barras de error), `06_tuner__curva_loss_mejor.png`;
> tablas `06_tuner__trials.csv`, `06_tuner__pareto_limpio_seeds.csv`.

### D-3.1 · Estrategia de búsqueda
- **Decisión:** qué tuner usar.
- **Opciones:** RandomSearch / Hyperband / BayesianOptimization.
- **Propuesta:** **Hyperband** (o RandomSearch como base), porque la propia charla del profe avisa de
  que en NAS lo sofisticado "no funciona mucho mejor que random" ([T3] §2), y Hyperband exprime mejor el
  presupuesto descartando configuraciones malas pronto. Bayesiana queda como alternativa si sobra tiempo.
- **Resultado (2026-06-22):** se compararon **Hyperband vs RandomSearch** en λ=1.0 → **empate técnico**
  (Δval_auc ≈ 0.0000 < 0.001). Se usa **Hyperband** por la propuesta por defecto, dejando constancia de que
  RandomSearch sería igual de válido y ~2× más barato (la teoría lo respalda).
- **Estado:** **Confirmada** *(implementación; pendiente ratificar en grupo)*
- **Decidido por / fecha:** Oscar / 2026-06-22

### D-3.2 · Hiperparámetros del espacio de búsqueda
- **Decisión:** qué se busca.
- **Opciones:** nº capas, unidades/capa, dropout (+rate), learning rate, activación, `λ_fair`.
- **Propuesta:** incluir **nº de capas, unidades por capa, dropout y su tasa, learning rate (log),
  activación y `λ_fair`** ([T3] §4). El **dropout entra sí o sí** porque es la palanca de MC-Dropout de la
  Tarea 4 (D-4.1); `λ_fair` se trata como el eje que se barre para la frontera (D-3.3).
- **Resultado (2026-06-22):** espacio implementado = `n_layers∈[1,3]`, `units_i∈[16,128] step 16`,
  `dropout_rate∈[0.1,0.5] step 0.1` (**siempre**, el backbone elegido tiene 0.3 → apto MC-Dropout),
  `lr∈[1e-4,1e-2] log`, `activation∈{relu,tanh}`. `λ_fair` **NO** entra en el tuner: es eje externo (D-3.3/D-3.4).
- **Estado:** **Confirmada** *(implementación; pendiente ratificar en grupo)*
- **Decidido por / fecha:** Oscar / 2026-06-22

### D-3.3 · Cómo extraer los pares (precisión, dependencia FAIR) por trial
- **Decisión:** cómo se materializa la curva de Pareto, ya que el tuner optimiza un solo escalar.
- **Opciones:** (a) callback que registra dos métricas por trial; (b) **bucle externo sobre `λ`**
  reentrenando y guardando `(AUC, group gap/D)`; (c) tuner multiobjetivo (Optuna).
- **Propuesta:** **bucle externo sobre `λ`** registrando `(AUC, group gap)` por valor, más Keras Tuner
  para la topología dentro de cada `λ`. La frontera de Pareto **no sale "gratis"** del `objective` escalar
  ([T3] §4, §5); hay que recoger las dos métricas a mano. El eje X es la dependencia FAIR (D-2.3), el eje Y
  la precisión (D-2.4).
- **Resultado (2026-06-22):** implementado el bucle externo sobre `λ`. **Mejora sobre la propuesta:** como
  el tuner elige una topología distinta por λ (confunde fairness con arquitectura) y una sola semilla cae en
  el ruido, se añade un **barrido limpio con topología fija + multi-semilla** (media±std) que es la frontera
  de referencia; la del tuner queda como evidencia del AutoML. Todos los λ se evalúan también en **test**.
- **Estado:** **Confirmada** *(implementación; pendiente ratificar en grupo)*
- **Decidido por / fecha:** Oscar / 2026-06-22

### D-3.4 · Métrica objetivo del tuner
- **Decisión:** qué se pasa a `objective`.
- **Opciones:** `val_loss` / `val_auc` / métrica custom que combine precisión y fairness.
- **Propuesta:** **`val_auc`** (maximizar) como objetivo de la topología, **tratando el fairness como
  eje externo** (D-3.3) y no como objetivo del tuner. Evita mezclar dos objetivos en un escalar, que el
  material no resuelve ([T3] §5 hueco 1-2).
- **Resultado (2026-06-22):** implementado `objective = kt.Objective("val_auc", "max")`, con una métrica
  `SlicedAUC` que desempaqueta `y_true=[y,s]` (D-2.5) para medir AUC solo contra `y`. Confirmado.
- **Estado:** **Confirmada** *(implementación; pendiente ratificar en grupo)*
- **Decidido por / fecha:** Oscar / 2026-06-22

---

## Tarea 4 — Incertidumbre (clase + varianza)

Fuente de huecos: [`docs/teoria/04-incertidumbre.md`](teoria/04-incertidumbre.md) §5.

### D-4.1 · Método para producir la varianza
- **Decisión:** cómo se obtiene la incertidumbre sobre test.
- **Opciones:** (a) **segundo modelo que predice el error** (lo mínimo que pide el profe, [T4] §3.2);
  (b) **MC-Dropout** (T pasadas con dropout activo, [T4] §3.3); (c) **deep ensemble** (T redes).
- **Propuesta:** **MC-Dropout como método principal** + **segundo modelo del error como entrega base
  fiel al profe**. MC-Dropout es barato (una sola red), da incertidumbre **epistémica** (`Var[p]`) y
  conecta directo con el análisis de `EXT_SOURCE` ([T4] §4.2): la pregunta del taller (¿más duda donde
  faltan fuentes?) es epistémica. El dropout ya está en el espacio de búsqueda (D-3.2).
- **Estado:** Propuesta
- **Decidido por / fecha:** _(pendiente)_

### D-4.2 · Número de pasadas / miembros T
- **Decisión:** cuántas pasadas de MC-Dropout (o miembros de ensemble).
- **Opciones:** el material usa 15 (ensemble) y 100 (MC-Dropout en el lab); no fija un óptimo.
- **Propuesta:** **T = 50–100**, comprobando que la varianza se estabiliza ([T4] §5 hueco 2). Valor
  concreto a fijar por prueba → **Revisar**. Sin evidencia del EDA para un T exacto.
- **Estado:** Abierta (fijar T tras comprobar estabilidad)
- **Decidido por / fecha:** _(pendiente)_

### D-4.3 · Cómo medir la "calidad de EXT_SOURCE" por solicitante
- **Decisión:** qué indicador resume la información externa disponible de cada fila.
- **Opciones:** (a) **`N_EXT_MISSING`** (nº de fuentes ausentes, 0–3); (b) flag binario "alguna imputada";
  (c) magnitud del valor imputado.
- **Propuesta:** **`N_EXT_MISSING` (0–3) + los flags `*_missing` por fuente**. Fuerte respaldo del [EDA]:
  (1) las tres `EXT_SOURCE` **no faltan juntas** (correlación de sus flags de ausencia ≈ 0: _1↔_3 = 0,036,
  _1↔_2 = 0,011), así que el conteo es un **gradiente real** y no redundante; (2) la **ausencia predice
  impago** de forma casi monótona (0→7,3 % · 1→8,2 % · 2→9,9 %). Es la materia prima ideal para cruzar
  con `Var[p]`.
- **Estado:** Propuesta
- **Decidido por / fecha:** _(pendiente)_

### D-4.4 · Umbral τ de clasificación
- **Decisión:** dónde se corta `P(impago)` para asignar clase.
- **Opciones:** 0,5 fijo / umbral ajustado por coste de falsos negativos / umbral por percentil.
- **Propuesta:** **no fijar 0,5 por defecto.** Con el desbalance del [EDA] (**11,4:1**) y un coste de
  falso negativo (impago no detectado) alto, el umbral óptimo será < 0,5; es una **decisión de política**
  ([T4] §5 hueco 4), a fijar con la matriz de coste del grupo. Se listan opciones sin imponer una.
- **Estado:** Abierta (decisión de política del grupo)
- **Decidido por / fecha:** _(pendiente)_

### D-4.5 · Descomposición aleatoria vs epistémica y calibración (extensiones)
- **Decisión:** si se separa incertidumbre aleatoria/epistémica y si se comprueba calibración.
- **Opciones:** hacerlo / dejarlo como mejora.
- **Propuesta:** **opcional, no obligatorio.** El enunciado pide clase + varianza, no la descomposición
  formal ([T4] §5 huecos 5-6). El [EDA] anticipa que tiene sentido mirarlo: la banda IC95 % de la tasa de
  impago **se ensancha donde hay menos datos**, y la clase minoritaria (impago) debería concentrar más
  varianza en el gráfico "buen vs mal pagador". Dejar como extensión si hay tiempo.
- **Estado:** Propuesta (como extensión)
- **Decidido por / fecha:** _(pendiente)_

---

## Preprocesado (transversal a las 4 tareas)

Decisiones derivadas sobre todo del [EDA]; sin sección propia en teoría, pero condicionan todas las tareas.

### D-P.1 · Centinela 365243 en `DAYS_EMPLOYED`
- **Decisión:** qué hacer con el valor "no aplica" de Home Credit.
- **Opciones:** (a) **mapear a `NaN` + crear flag** `DAYS_EMPLOYED_ANOM`; (b) dejarlo; (c) borrar filas.
- **Propuesta:** **mapear 365243 → `NaN` y crear un flag binario.** El [EDA] confirma que afecta a
  **55.374 filas (18,01 %)**, son casi todos **pensionistas** y ese grupo **impaga menos (5,40 % vs
  8,66 %)**: el flag lleva señal y dejar el centinela crudo (≈1.000 años) destrozaría cualquier escala/log.
- **Estado:** **Confirmada**
- **Decidido por / fecha:** Grupo / 2026-06-19

### D-P.2 · Imputación de `EXT_SOURCE` y flags de ausencia
- **Decisión:** cómo rellenar los nulos de las fuentes externas.
- **Opciones:** mediana / modelo (p. ej. regresor) / KNN; con o sin flags.
- **Propuesta:** **crear los flags `*_missing` ANTES de imputar** (ya hechos en el EDA) y luego imputar
  (mediana como base, modelo si mejora), **conservando los flags como features**. El [EDA] justifica los
  flags: nulos **56,4 % / 19,8 % / 0,2 %** y la ausencia predice impago (D-4.3). La imputación se ajusta
  **solo en train** (ver D-P.6).
- **Decisión del grupo (2026-06-19):** los flags `*_missing` se crean **siempre antes de imputar** (fijo).
  La imputación arranca con **mediana por defecto**, pero el **pipeline queda parametrizado** para enchufar
  una imputación alternativa (KNN o regresor) sin reescribir el código. La elección final (mediana vs
  alternativa) **se decidirá midiendo AUC en validación**, con cada imputador **ajustado solo en train**.
- **Cierre experimental (2026-06-20):** se comparó **mediana vs KNN vs IterativeImputer** con un **proxy
  neutro (`LogisticRegression`)** midiendo **AUC-ROC en validación**, con cada imputador **ajustado solo en
  train** (anti-fuga, D-P.6). Resultado: **gana la mediana** con **AUC val = 0,728034**. El `IterativeImputer`
  (BayesianRidge) queda en **empate técnico** (ΔAUC = **−0,000168**, dentro del ruido) y `KNNImputer(k=5)` es
  **claramente peor** (AUC val = 0,724801, ΔAUC = **−0,0032**) y además **~300× más caro** (≈ 389 s vs ≈ 1 s).
  Ninguna alternativa supera a la mediana de forma clara y material → **se mantiene `SimpleImputer(strategy="median")`
  como imputación definitiva** por ser la más simple, barata y determinista sin pérdida de AUC. Los flags
  `*_missing` se siguen creando **siempre antes de imputar** (parte ya fija). Detalle en
  [`notebooks/02_preprocesado.ipynb`](../notebooks/02_preprocesado.ipynb), "Anexo · Experimento D-P.2".
- **Estado:** **Confirmada**
- **Decidido por / fecha:** Grupo / 2026-06-20

### D-P.3 · Variables a transformar con log
- **Decisión:** qué columnas se log-transforman antes de modelar.
- **Opciones:** ninguna / solo ingreso / ingreso + anualidad + crédito.
- **Propuesta:** **log para `AMT_INCOME_TOTAL`, `AMT_CREDIT` y `AMT_ANNUITY`** (las tres de cola larga),
  porque el [EDA] mide skews de **391,6 / 1,2 / 1,6**. El ingreso es el caso extremo. (Complementa, no
  sustituye, la saturación `x^p` de la Tarea 1, que la red aprende.)
- **Nota:** se usa **`log1p`** (no `log`) para tolerar ceros con seguridad.
- **Estado:** **Confirmada**
- **Decidido por / fecha:** Grupo / 2026-06-19

### D-P.4 · Categoría `XNA` de `CODE_GENDER`
- **Decisión:** qué hacer con los registros sin género.
- **Opciones:** (a) **descartar las filas**; (b) imputar al género mayoritario; (c) tercera categoría.
- **Propuesta:** **descartar las 4 filas `XNA`.** El [EDA] mide que son **4 de 307.511 (0,0013 %)**:
  irrelevantes en volumen y necesarios de quitar para tener un `CODE_GENDER` **binario limpio**, que es
  lo que asume la FAIR loss de la Tarea 2.
- **Estado:** **Confirmada**
- **Decidido por / fecha:** Grupo / 2026-06-19

### D-P.5 · Outlier de ingreso (117.000.000)
- **Decisión:** cómo tratar los ingresos extremos.
- **Opciones:** (a) winsorizar (recorte a un percentil); (b) recortar solo el caso de 117M; (c) confiar en
  log + saturación.
- **Propuesta:** **winsorizar la cola alta** (p. ej. a p99,9 ≈ 900k) o al menos recortar el caso de 117M.
  El [EDA] lo cuantifica: **117M es un caso único = 130× el p99,9**, y solo 3 filas superan 10M. El log
  (D-P.3) lo amortigua, pero el caso de 117M conviene tratarlo explícitamente.
- **Decisión del grupo (2026-06-19):** **winsorizar la cola alta de `AMT_INCOME_TOTAL` a p99,9**, con el
  **percentil calculado solo en train** y aplicado a validación y test (anti-fuga, ver D-P.6).
- **Estado:** **Confirmada**
- **Decidido por / fecha:** Grupo / 2026-06-19

### D-P.6 · Separar train/val/test antes de imputar y escalar (anti-fuga)
- **Decisión:** orden del pipeline para no filtrar información de validación/test.
- **Opciones:** (a) **split primero, ajustar imputador/escalador solo en train**; (b) imputar/escalar
  sobre todo y luego dividir (incorrecto).
- **Propuesta:** **split estratificado por `TARGET` PRIMERO**, y ajustar imputación (D-P.2), escalado y
  winsorización (D-P.5) **solo con estadísticos de train**, aplicándolos después a test. Es regla estándar
  anti-fuga y es **especialmente crítico aquí** porque la imputación de `EXT_SOURCE` (56 % de nulos en una
  fuente, [EDA]) usa estadísticos que no deben ver el test. La estratificación protege el 8,07 % de
  positivos. Coherente con [T2] §1 (el género solo se usa en entrenamiento).
- **Decisión del grupo (2026-06-19):** split estratificado por `TARGET` en **tres cortes: train /
  validación / test**. **Todos** los parámetros (imputación, winsorización, escalado) se aprenden **solo
  del train** y se aplican a validación y test. La **validación** existe para que Keras Tuner (Tarea 3,
  D-3.x) elija la topología **sin tocar el test**, que queda reservado para la evaluación final imparcial.
- **Estado:** **Confirmada**
- **Decidido por / fecha:** Grupo / 2026-06-19

### D-P.7 · Inclusión de `DAYS_BIRTH` (edad) — alineación con el esqueleto del profe
- **Decisión:** añadir `DAYS_BIRTH` al preprocesado y con qué transformación.
- **Opciones:** (a) `abs(DAYS_BIRTH)/365` (edad en años, como el profe); (b) `log1p`; (c) no incluirla.
- **Propuesta:** **`abs(DAYS_BIRTH)/365` (edad en años positivos)**, igual que `load_home_credit_data`
  del esqueleto del profe (`docs/_fuentes/clases-master/Lectura_datos_Taller_B4_T1.ipynb`). Se descarta
  `log1p`: el [EDA] confirma que la edad está **acotada (20,5–69,1 años) y es casi simétrica (skew 0,12)**,
  no tiene la cola larga que justifica el log en las financieras (skew 392); loguearla distorsionaría una
  variable ya bien condicionada y rompería su interpretabilidad. Es un **predictor legítimo del impago**
  (corr **−0,078** con `TARGET`, por encima de las financieras crudas). Se incorpora como continua: pasa por
  imputación (sin nulos → no-op) y **escalado ajustado solo en train** (anti-fuga, coherente con D-P.6).
  **No** se winsoriza ni se loguea.
- **Decisión del grupo (2026-06-19):** incluida para ser fieles a la plantilla del profe (variable
  predictiva legítima del impago); transformación `abs/365`. Nuestra `DAYS_EMPLOYED` + flag de centinela
  (D-P.1) **se mantiene además**: la inclusión de la edad es **aditiva**, no la sustituye.
- **Estado:** **Confirmada**
- **Decidido por / fecha:** Grupo / 2026-06-19

---

## Modelo base (NB 03)

Decisiones que fija el notebook [`03_modelo_base.ipynb`](../notebooks/03_modelo_base.ipynb): el
modelo de **referencia sin FAIR ni capa custom** (`λ = 0`) que sirve de línea base del
"base vs mejor FAIR" (entregable E5). Consume el contrato `(X, y, s)` y las decisiones de
preprocesado de `02_preprocesado` (D-P.*) **sin rediscutirlas**. Artefactos que produce:
`results/figures/03_base__curva_loss.png` (E4), `results/figures/03_base__roc_test.png` (opcional)
y `results/tables/03_base__metricas_test.csv` (parte base de E5).

### D-MB.1 · Arquitectura del MLP base
- **Decisión:** qué red es el modelo de referencia, sin capa custom (Tarea 1) ni FAIR loss (Tarea 2).
- **Opciones:** (a) modelo lineal / regresión logística como base mínima; (b) **MLP pequeño**
  `Dense(64) → Dense(32)` con dropout intermedio; (c) red más profunda/ancha.
- **Propuesta:** **MLP `Sequential`: `Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dropout(0.3) →
  Dense(1, sigmoid)`**, sin capa custom y con **`λ_fair = 0`**. Es la **referencia ingenua mínima pero
  realista** (no un lineal de juguete) sobre la que medir lo que aportan las Tareas 1-4. El **dropout 0,3
  se monta FIJO** aunque aquí no se optimice: deja la **cadena 06→07** lista — el tuner de la Tarea 3
  (D-3.2) buscará la *tasa* de dropout y la Tarea 4 (D-4.1) reutiliza ese mismo dropout como motor de
  **MC-Dropout**. Sin él, el base no sería comparable arquitectónicamente con los modelos posteriores.
- **Decisión del grupo (2026-06-20):** arquitectura `64→32` ReLU con `Dropout(0.3)` entre densas y salida
  sigmoide, fijada como base común; el dropout queda montado de serie por la dependencia D-3.2 ↔ D-4.1.
- **Estado:** **Confirmada**
- **Decidido por / fecha:** Grupo / 2026-06-20

### D-MB.2 · Compilación (loss, optimizador, métrica)
- **Decisión:** con qué se compila el modelo base.
- **Opciones:** loss `binary_crossentropy` vs focal; optimizador Adam vs SGD; métrica de seguimiento
  AUC vs accuracy.
- **Propuesta:** **`loss = binary_crossentropy`, `optimizer = Adam`, métrica `AUC`**. La **AUC** se elige
  como métrica de seguimiento por su robustez al fuerte desbalance que mide el [EDA] (**8,07 % de impagos,
  ratio 11,4:1**), donde la accuracy engaña (un trivial "siempre paga" acierta el 91,93 %). Es coherente
  con **D-2.4**, que ya fija AUC-ROC como eje de precisión de la curva de Pareto: el base y las tareas se
  miden con la misma vara.
- **Decisión del grupo (2026-06-20):** BCE + Adam + métrica AUC como compilación estándar del base,
  alineada con D-2.4.
- **Estado:** **Confirmada**
- **Decidido por / fecha:** Grupo / 2026-06-20

### D-MB.3 · Tratamiento del desbalance en el base
- **Decisión:** cómo compensar el 8,07 % de impago al entrenar el modelo base.
- **Opciones:** (a) **`class_weight` balanced**; (b) oversampling / SMOTE; (c) undersampling; (d) no hacer nada.
- **Propuesta:** **`class_weight` "balanced" calculado desde la proporción de train** (clase 1 ≈ **6,19**,
  clase 0 ≈ **0,544**). No se hace oversampling ni undersampling: el reponderado **no altera la distribución
  real** de los datos ni infla el tamaño del train, es determinista y barato, y basta para que la red no
  colapse en la clase mayoritaria con el desbalance que mide el [EDA] (**ratio 11,4:1**). Los pesos se derivan
  **solo de la proporción de train** (coherente con el anti-fuga de D-P.6).
- **Decisión del grupo (2026-06-20):** reponderado por `class_weight` balanced desde train; se descarta el
  resampling para el base por mantener la distribución original.
- **Estado:** **Confirmada**
- **Decidido por / fecha:** Grupo / 2026-06-20

### D-MB.4 · Entrenamiento, parada y umbral
- **Decisión:** protocolo de entrenamiento, parada y corte de clasificación del base.
- **Opciones:** parada por **nº fijo de épocas** vs **EarlyStopping**; gestión del *learning rate* fijo vs
  **`ReduceLROnPlateau`**; conservar los pesos de la última época vs los del **mejor `val_auc`**
  (`ModelCheckpoint`); monitorizar `val_loss` vs `val_auc`; umbral 0,5 fijo vs ajustado por coste.
- **Propuesta:** **validar sobre el split `val` (D-P.6)**, entrenar un **nº fijo de épocas (150)** —sin
  EarlyStopping, dejando que el `fit` complete las 150— y gobernar la convergencia con
  **`ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=8, min_lr=1e-6)`** (baja el LR cuando el AUC se
  estanca) más **`ModelCheckpoint(monitor="val_auc", save_best_only=True)`**, recargando al final los pesos del
  **mejor `val_auc`** (no los de la época 150) para no arrastrar un eventual sobreajuste del tramo final.
  Umbral **0,5 provisional** para las métricas de test; se reporta **AUC-ROC (principal)** y
  **accuracy (secundaria)**. El umbral 0,5 es deliberadamente **provisional**: el ajuste por coste de falso
  negativo es una **decisión de política** que se difiere a **D-4.4** (Tarea 4), no al base. Usar el split `val`
  (y no el test) para monitorizar la parada/LR deja el test reservado a la evaluación final imparcial, como
  fija D-P.6.
- **Decisión del grupo (2026-06-20):** 150 épocas fijas (sin EarlyStopping) + `ReduceLROnPlateau` sobre
  `val_auc` + `ModelCheckpoint` que conserva los mejores pesos por `val_auc` (recargados al terminar);
  evaluación en test con umbral 0,5 provisional; el umbral por coste se decide en D-4.4.
- **Estado:** **Confirmada**
- **Decidido por / fecha:** Grupo / 2026-06-20

### D-MB.5 · Auditoría de equidad del modelo base
- **Decisión:** si el base —que no lleva FAIR— audita su propia equidad y con qué métrica.
- **Opciones:** (a) no auditar (es "solo" la base); (b) **reportar group gap M−F + tasas por grupo**;
  (c) métricas de dependencia (HSIC/CKA) residuales.
- **Propuesta:** **reportar el group gap `Δ = mean(ŷ|M) − mean(ŷ|F)` + las tasas por grupo** ya en el base,
  contrastándolo con la **línea base del [EDA] (+3,14 pp**, M 10,14 % vs F 7,00 % de impago real). El **género
  NUNCA es input**: el gap mide el sesgo que se filtra por las variables predictivas (el [EDA] muestra que se
  cuela vía `EXT_SOURCE_1`). Esta cifra es la **cota a batir por la FAIR loss** y da sentido al "base vs mejor
  FAIR" (E5); es la misma métrica de equidad que fija **D-2.3**.
- **Decisión del grupo (2026-06-20):** auditar el base con group gap M−F + tasas por grupo (métrica de D-2.3),
  sin usar el género como entrada, para fijar la referencia de equidad.
- **Estado:** **Confirmada**
- **Decidido por / fecha:** Grupo / 2026-06-20

---

## Resumen de estados

| Sección | Fichas | Propuesta | Confirmada | Revisar | Abierta |
| --- | --- | --- | --- | --- | --- |
| Tarea 1 — Capa custom | 6 | 6 | 0 | 0 | 0 |
| Tarea 2 — FAIR loss | 7 | 6 | 0 | 0 | 1 (D-2.7) |
| Tarea 3 — Keras Tuner | 4 | 0 | 4 | 0 | 0 |
| Tarea 4 — Incertidumbre | 5 | 3 | 0 | 0 | 2 (D-4.2, D-4.4) |
| Preprocesado | 7 | 0 | 7 | 0 | 0 |
| Modelo base (NB 03) | 5 | 0 | 5 | 0 | 0 |
| **Total** | **34** | **15** | **16** | **0** | **3** |

> **Tarea 3 confirmada (implementación, 2026-06-22):** las 4 fichas D-3.1–D-3.4 quedan implementadas y
> validadas por ejecución (ver el bloque de resultado al inicio de la sección). Marcadas **Confirmada
> (implementación)** a falta de ratificación formal del grupo. La dependencia FAIR usada es el fallback
> `corr²` porque la HSIC de la Tarea 2 (D-2.1) aún no está entregada; al enchufarla, re-ejecutar el NB06.

**Preprocesado validado por el grupo:** las **7** fichas (D-P.1 a D-P.7) están **Confirmadas**; ya no queda
ninguna en **Revisar**. D-P.2 quedó **Confirmada (2026-06-20)** tras el experimento de AUC en validación, que
dio ganadora a la **mediana** (AUC val = 0,728034); el resto (D-P.1, D-P.3, D-P.4, D-P.5, D-P.6, D-P.7) se
confirmaron el 2026-06-19.

**Modelo base confirmado (2026-06-20):** las **5** fichas (D-MB.1 a D-MB.5) están **Confirmadas**: fijan la
arquitectura `64→32` ReLU con `Dropout(0.3)` (sin capa custom ni FAIR), la compilación BCE + Adam + AUC, el
`class_weight` balanced, la parada por `val_auc` con umbral 0,5 provisional y la auditoría de equidad por group
gap. Registran **decisiones de diseño**, no resultados numéricos (AUC/gap reales se rellenarán tras ejecutar el
notebook). Las decisiones de las Tareas 1-4 siguen pendientes de validación.
