# Tarea 3 — AutoML / Keras Tuner

> Documento teórico del Taller B4-T1 (máster MIAX). Clasificador neuronal de
> crédito (Home Credit Default Risk). Esta tarea cubre la búsqueda automática
> de la topología de la red con **Keras Tuner**, y cómo del barrido sale la
> curva de Pareto Precisión vs Dependencia FAIR.

---

## 1. Intuición / resumen

Cuando montamos una red neuronal a mano tomamos un montón de decisiones de
diseño que no son los pesos que el entrenamiento aprende, sino los
**hiperparámetros**: cuántas capas, cuántas neuronas por capa, qué función de
activación, si poner *dropout* o no, qué *learning rate*… Probar esas
combinaciones "a boleo" es lento y poco sistemático. La idea del **AutoML** es
precisamente *automatizar todos los pasos del pipeline de ML* en lugar de
ajustarlos manualmente (`clases-master/Automatic_ML_2026.pdf`, pág. 4-5).

El pipeline completo de ML incluye preparación de datos, preprocesado,
selección/extracción de características, selección de modelo, **selección de
hiperparámetros**, entrenamiento, métricas y empaquetado del *pipeline*. AutoML
aspira a automatizar cada uno de esos pasos
(`clases-master/Automatic_ML_2026.pdf`, pág. 4-5).

En esta tarea nos centramos en una porción concreta de ese AutoML: la
**búsqueda de la topología/hiperparámetros de una red Keras** mediante
**Keras Tuner**. El profesor lo resume de forma muy directa: definimos el
modelo con una variable de hiperparámetros, y esa variable "puede elegir entre"
las opciones que le demos; el tuner se encarga de probarlas y devolverte el
resumen de cada *trial* (`clases-master/26 06 13.pdf` *(transcripción de
clase)*). Su recomendación explícita: si hay que elegir una sola herramienta
con la que jugar, **Keras Tuner**, porque es "el más estándar y tiene la
mayoría de las opciones que tienen todos los demás"
(`clases-master/26 06 13.pdf`).

Históricamente el AutoML ha pasado por varias olas: algoritmos evolutivos
(1989-2008), optimización bayesiana —típicamente con procesos gaussianos—
(2013-2016) y aprendizaje por refuerzo (2017-2020), llegando este último a
gastar 800 GPUs durante 3-4 semanas para alcanzar el estado del arte
(`clases-master/Automatic_ML_2026.pdf`, pág. 6).

---

## 2. Formulación matemática

### Espacio de hiperparámetros

Sea un modelo con un vector de hiperparámetros
$\theta = (\theta_1, \theta_2, \dots, \theta_k)$ donde cada $\theta_i$ vive en
su propio dominio $\Theta_i$ (un conjunto discreto de opciones, un rango entero,
un rango continuo, un booleano…). El **espacio de búsqueda** es el producto
$\Theta = \Theta_1 \times \Theta_2 \times \dots \times \Theta_k$.

En el material aparecen estos tipos de hiperparámetro (vía la API de Keras
Tuner, `clases-master/Keras_Tuner_Basico.ipynb`):

- **Discreto / categórico** — `hp.Choice('units', [8, 16, 32])`,
  `hp.Choice("activation", ["relu", "tanh"])`.
- **Entero en rango** — `hp.Int("units", min_value=32, max_value=512, step=32)`.
- **Booleano** — `hp.Boolean("dropout")` (poner o no una capa).
- **Continuo (con muestreo)** —
  `hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")`,
  donde `sampling="log"` indica que el *learning rate* se explora en escala
  logarítmica.

### Objetivo a optimizar

La búsqueda elige el conjunto de hiperparámetros que optimiza una métrica
medida en validación:

$$
\theta^\star = \arg\min_{\theta \in \Theta} \; \mathcal{L}_{\text{val}}(\theta)
$$

En los ejemplos el `objective` es `'val_loss'` (se minimiza); también podría
fijarse `val_accuracy` (se maximiza). En Keras Tuner el objetivo se pasa
directamente al tuner, p. ej. `objective='val_loss'`
(`clases-master/Keras_Tuner_Basico.ipynb`). En Optuna, la función `objective`
*devuelve* el valor a optimizar —en el ejemplo
`val_loss = history.history['val_loss'][-1]`— y el estudio se crea con
`direction='minimize'` (`clases-master/Optuna_basico.ipynb`).

> Importante: el material optimiza **un único objetivo escalar**. La
> optimización simultánea de dos objetivos (precisión y fairness) no está
> resuelta en las fuentes y se trata en §4 y §5.

### Estrategias de búsqueda

El material distingue varias estrategias (en general para NAS,
`clases-master/Automatic_ML_2026.pdf`, pág. 25; y como tuners concretos de
Keras Tuner, pág. 31):

- **Aleatoria (Random Search)** — se muestrean combinaciones al azar del
  espacio $\Theta$. En Keras Tuner: `keras_tuner.RandomSearch(...)` con
  `max_trials` (nº de combinaciones a probar)
  (`clases-master/Keras_Tuner_Basico.ipynb`).
- **Hyperband** — "probar con pocas iteraciones"
  (`clases-master/Automatic_ML_2026.pdf`, pág. 31): asigna pocos recursos a
  muchas configuraciones y va concentrando recursos en las prometedoras. En
  Keras Tuner: `keras_tuner.Hyperband(..., factor=3)`
  (`clases-master/Keras_Tuner_Basico.ipynb`).
- **Optimización bayesiana** — modeliza la relación hiperparámetros→objetivo
  (típicamente con procesos gaussianos) para elegir de forma "inteligente" la
  siguiente combinación a probar
  (`clases-master/Automatic_ML_2026.pdf`, págs. 6, 25). En Keras Tuner:
  `keras_tuner.BayesianOptimization(..., max_trials=3)`
  (`clases-master/Keras_Tuner_Basico.ipynb`).
- Otras citadas para NAS pero no usadas en el taller: Deep RL, random con
  *hill-climbing*, algoritmos evolutivos / *network morphism*, y búsqueda
  **multiobjetivo** (memoria, tiempo, tamaño del modelo)
  (`clases-master/Automatic_ML_2026.pdf`, pág. 25). Esta última pista
  multiobjetivo es justo la que conecta con la curva de Pareto de §4.

Nota del profesor: aunque herramientas como AutoKeras presumen de búsquedas
"más inteligentes" (bayesiana con distancias entre arquitecturas), "al final
acaban funcionando de una forma muy parecida" a la búsqueda estándar de Keras
Tuner (`clases-master/26 06 13.pdf`). Y la conclusión escéptica de la
charla: en NAS, métodos sofisticados "no funcionan mucho mejor que random
[Real et al. 2018]" (`clases-master/Automatic_ML_2026.pdf`, pág. 41).

---

## 3. Implementación

### Patrón Keras Tuner

El patrón canónico tiene tres piezas: (1) una función `build_model(hp)` que
construye y compila el modelo usando el objeto `hp` para declarar los
hiperparámetros; (2) un *tuner* que define la estrategia de búsqueda y el
objetivo; (3) la llamada `tuner.search(...)` que entrena y evalúa cada *trial*
(`clases-master/Keras_Tuner_Basico.ipynb`).

```python
import keras_tuner
import keras

# (1) Función que construye el modelo en función de los hiperparámetros
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten())

    # nº de neuronas: elegir entre un conjunto discreto...
    model.add(keras.layers.Dense(
        hp.Choice('units', [8, 16, 32]),
        # ...o un rango entero con paso:
        # units=hp.Int("units", min_value=32, max_value=512, step=32),
        activation=hp.Choice("activation", ["relu", "tanh"])))

    # añadir (o no) una capa de dropout
    if hp.Boolean("dropout"):
        model.add(keras.layers.Dropout(rate=0.25))

    model.add(keras.layers.Dense(1, activation='relu'))

    # learning rate continuo, muestreado en escala log
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",                 # en el ejemplo de regresión
        # loss="categorical_crossentropy", metrics=["accuracy"],  # para clasificación
    )
    return model

# (2) Estrategia de búsqueda + objetivo
tuner = keras_tuner.RandomSearch(
    build_model,
    objective='val_loss',
    overwrite=True,
    max_trials=3)

# Alternativas de tuner (misma build_model):
# tuner = keras_tuner.Hyperband(build_model, objective='val_loss', overwrite=True, factor=3)
# tuner = keras_tuner.BayesianOptimization(build_model, objective='val_loss', overwrite=True, max_trials=3)

# Inspeccionar el espacio de búsqueda
tuner.search_space_summary()

# (3) Ejecutar la búsqueda (entrena cada trial)
tuner.search(XX_tr, YY_tr, epochs=2, validation_split=0.1)

# Recuperar resultados y el/los mejores modelos
tuner.results_summary()
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()
```

Fuente del patrón y de cada fragmento:
`clases-master/Keras_Tuner_Basico.ipynb`. (Instalación:
`pip install keras-tuner --upgrade`.)

Claves a recordar:

- Cambiar de **estrategia** es cambiar solo la clase del tuner
  (`RandomSearch` / `Hyperband` / `BayesianOptimization`); `build_model` no
  cambia.
- `max_trials` controla cuántas combinaciones se prueban en Random/Bayesian;
  `factor` controla el descarte progresivo en Hyperband.
- `overwrite=True` evita reutilizar resultados de una búsqueda anterior.
- `get_best_models(num_models=k)` devuelve los `k` mejores modelos ya
  entrenados; `results_summary()` lista los *trials*.

### Comparación breve con Optuna

Optuna implementa la misma idea con otra ergonomía: en vez de un objeto `hp`
dentro de un `build_model`, se escribe una función `objective(trial)` que (a)
declara los hiperparámetros con `trial.suggest_*`, (b) construye y entrena el
modelo, y (c) **devuelve** la métrica a optimizar
(`clases-master/Optuna_basico.ipynb`):

```python
import optuna, keras

def objective(trial):
    units = trial.suggest_categorical('units', [8, 16, 32])
    activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
    use_dropout = trial.suggest_categorical('dropout', [True, False])
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    model = keras.Sequential()
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=units, activation=activation))
    if use_dropout:
        model.add(keras.layers.Dropout(rate=0.25))
    model.add(keras.layers.Dense(1, activation='relu'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')

    history = model.fit(XX_tr, YY_tr, epochs=2, validation_split=0.1, verbose=1)
    return history.history['val_loss'][-1]   # Optuna minimiza esto

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=3)

print("Mejor val_loss:", study.best_value)
print("Mejores hiperparámetros:", study.best_params)
```

Correspondencia conceptual (la lógica es la misma, cambia la sintaxis):

| Concepto              | Keras Tuner                          | Optuna                                  |
|-----------------------|--------------------------------------|-----------------------------------------|
| Declarar HP           | `hp.Choice/Int/Float/Boolean`        | `trial.suggest_categorical/int/float`   |
| Definir objetivo      | `objective='val_loss'` (en el tuner) | `return val_loss` + `direction='minimize'` |
| Lanzar la búsqueda    | `tuner.search(...)`                   | `study.optimize(objective, n_trials=...)` |
| Nº de pruebas         | `max_trials`                          | `n_trials`                              |
| Mejores resultados    | `tuner.get_best_models()`             | `study.best_params` (reconstruir modelo)|

Una diferencia práctica: Optuna entrega los **mejores parámetros**
(`study.best_params`) y hay que **reconstruir** el modelo a mano para verlo;
Keras Tuner ya te devuelve el modelo entrenado
(`clases-master/Optuna_basico.ipynb`, `clases-master/Keras_Tuner_Basico.ipynb`).

Optuna se presenta en la charla con la tríada **Objetivo (objective) /
Parámetros a probar (trial) / Entrenar (optimize)**
(`clases-master/Automatic_ML_2026.pdf`, pág. 29-30). El profesor lo describe
como herramienta muy útil para no estar cambiando hiperparámetros a mano
(`clases-master/26 06 13.pdf`).

---

## 4. Conexión con el Taller B4-T1

En el taller la red es un **clasificador de impago** (Home Credit Default
Risk). La Tarea 3 pide usar Keras Tuner para encontrar la **topología óptima**
y, sobre el barrido de la búsqueda, construir una **curva/scatter de Pareto
Precisión (eje Y) vs Dependencia FAIR (eje X)**.

### Qué hiperparámetros tiene sentido buscar aquí

Trasladando directamente la API vista en
`clases-master/Keras_Tuner_Basico.ipynb` al clasificador:

- **Nº de capas ocultas** — añadir o no capas (`hp.Boolean(...)`) o iterar un
  `hp.Int("n_layers", ...)` para apilar varias densas.
- **Nº de unidades por capa** — `hp.Int("units", min_value=..., max_value=...,
  step=...)` o `hp.Choice('units', [...])`.
- **Dropout** — `hp.Boolean("dropout")` y, si se quiere, su tasa con
  `hp.Float("rate", ...)`.
- **Learning rate** — `hp.Float("lr", 1e-4, 1e-2, sampling="log")`.
- **Función de activación** — `hp.Choice("activation", ["relu", "tanh"])`.
- **Coeficiente $\lambda$ de fairness** — el peso del término de penalización
  FAIR en la función de coste, declarado como un hiperparámetro más, p. ej.
  `hp.Float("lambda_fair", ...)` o `hp.Choice("lambda_fair", [...])`.

> El nombre y el rol del $\lambda$ de fairness vienen del propio enunciado del
> taller (compensación precisión vs dependencia FAIR); **la API para declararlo
> como hiperparámetro es la de Keras Tuner** vista en las fuentes. La *pérdida*
> que combina precisión y fairness ($\text{BCE} + \lambda\cdot D(\hat y, S)$) es
> la de la Tarea 2 (`docs/teoria/02-fair-loss.md`).

Para clasificación, el `compile` cambiaría respecto al ejemplo de regresión:
una salida `Dense(1, activation='sigmoid')` con
`loss="binary_crossentropy"` y `metrics=["accuracy"]` (extrapolación del patrón
de `clases-master/Keras_Tuner_Basico.ipynb`, que para clasificación ya muestra
`categorical_crossentropy` + `accuracy`).

### Cómo del barrido sale la curva de Pareto

La idea es tratar el coeficiente $\lambda$ de fairness como el hiperparámetro
que **barremos**, y para cada valor explorado registrar dos métricas en vez de
una sola:

- $y$ = **Precisión** del clasificador (calidad predictiva).
- $x$ = **Dependencia FAIR** (cuánto depende la predicción del atributo
  protegido; menor = más justo).

Cada *trial* del tuner (cada combinación de hiperparámetros, en particular cada
$\lambda$) produce un punto $(x, y)$. Al dibujar todos los puntos del barrido
se obtiene el **scatter Precisión vs Dependencia FAIR**, y el subconjunto de
puntos no dominados —aquellos en los que no se puede mejorar la precisión sin
empeorar la dependencia FAIR, ni viceversa— forma el **frente de Pareto**. Así,
el mismo barrido de hiperparámetros que Keras Tuner ya hace sirve para
materializar el compromiso precisión-fairness.

Esto enlaza con la idea de **búsqueda multiobjetivo** que la charla menciona
para NAS ("memoria, tiempo, tamaño del modelo…",
`clases-master/Automatic_ML_2026.pdf`, pág. 25): aquí los dos objetivos son
precisión y fairness.

> Matiz de implementación: el `objective` de Keras Tuner optimiza **un escalar**
> (p. ej. `val_loss`). El frente de Pareto NO sale "gratis" de elegir el mejor
> modelo; sale de **recoger las dos métricas de todos los trials** y graficarlas.
> Cómo extraer esos pares por trial (callbacks, `results_summary`, o ejecutar la
> búsqueda en bucle sobre $\lambda$) es una decisión de diseño.

### Dos fronteras de Pareto distintas (no confundirlas)

La implementación final del taller produce **dos** scatters de Pareto que miden
cosas distintas y conviene no mezclar:

1. **Pareto "oficial" obtenida por Keras Tuner** — la del barrido del tuner, con
   **topología variable**: cada trial cambia capas/unidades/dropout/activación/
   $\lambda$, y se registran (Precisión, Dependencia FAIR) por trial. Responde a
   "¿qué arquitectura barata domina a las demás?".
2. **Pareto comparada de las 3 medidas** — sobre un **backbone fijo**
   (la mejor topología que dio el tuner), se barre $\lambda$ por separado para
   cada una de las tres medidas de dependencia de la Tarea 2 (corr², HSIC, MMD),
   con **varias semillas** para promediar el ruido de inicialización. Responde a
   "fijada la arquitectura, ¿qué medida y qué $\lambda$ compran más justicia por
   euro de precisión?".

Además, los dos **ejes** de "dependencia" no son intercambiables:

- el **eje de dependencia $D$ literal** es el valor de la medida penalizada
  (corr²/HSIC/MMD del propio entrenamiento), que vive en unidades distintas
  según la medida;
- el **eje *group gap*** $\;\Delta = \overline{\hat y}_{S=1}-\overline{\hat y}_{S=0}\;$
  (en puntos porcentuales) es un **proxy interpretable y común** de
  (in)justicia que permite comparar las tres medidas en la misma regla, ya que
  $D$ no es comparable entre medidas. La elección del compromiso se hace mirando
  el group gap (y las tasas por grupo), no el $D$ crudo.

### Resultado real del taller

Ejecutado el plan anterior (NB06), los hechos:

- **Estrategia de búsqueda**: Hyperband y RandomSearch quedaron en **empate
  técnico** (misma calidad de mejor trial dentro del ruido), así que se eligió
  **RandomSearch** por simplicidad y reproducibilidad —coherente con el aviso de
  la charla de que lo sofisticado no suele batir a random.
- **Backbone ganador del tuner**: **1 capa oculta, 64 unidades, dropout 0.3,
  activación `relu`** (salida sigmoide + BCE).
- **Comparación de las 3 medidas sobre ese backbone**: por **presupuesto de AUC
  en validación** (aceptar como mucho una pequeña caída de AUC y, dentro de ese
  presupuesto, quedarse con la que más baja el group gap), el compromiso
  $(\text{medida}^\star, \lambda^\star)$ elegido fue **corr² con $\lambda^\star = 5$**.
- **Efecto en test del modelo elegido**: el group gap cae un **−72 %**
  (de **5.615 pp** a **1.568 pp**) a cambio de solo **−0.59 pp de AUC**.
- Además del group gap se reporta el criterio de **equalized odds**
  (las diferencias por grupo $\Delta\text{TPR}$ y $\Delta\text{FPR}$), para no
  leer la justicia por una única métrica de medias.

> Lectura: con `CODE_GENDER` binaria, la diferencia entre grupos es esencialmente
> un desplazamiento de medias, así que la penalización lineal (corr²) compra más
> justicia por unidad de AUC que HSIC (que se diluye con el desbalance) o MMD
> (más expresiva pero sin ventaja neta aquí). El detalle teórico de las tres
> medidas está en `docs/teoria/02-fair-loss.md`.

---

## 5. Decisiones tomadas (lo que las fuentes no zanjaban)

Las fuentes de clase no cubrían varios puntos; así se resolvieron en la
implementación final:

1. **Optimización simultánea de dos objetivos.** El material optimiza un único
   escalar (`val_loss`/`val_accuracy`). Se resolvió **barriendo $\lambda$** y
   tratando la fairness como **eje** (no como objetivo del tuner): el tuner
   busca topología por un escalar y, sobre el backbone ganador, se barre
   $\lambda$ para materializar la frontera. No se usó un tuner multiobjetivo.

2. **Métrica objetivo del tuner y métrica de precisión de la frontera.** El
   tuner optimiza una métrica predictiva estándar; la calidad de la frontera y
   la elección del compromiso se leen con **AUC en validación** (no $R^2$, que
   era de los ejemplos de regresión).

3. **Definición operativa de "Dependencia FAIR".** Resuelta en la Tarea 2: la
   penalización es una de las tres medidas (corr²/HSIC/MMD) sobre `CODE_GENDER`,
   y como lectura común y comparable se usa el **group gap** (y las tasas por
   grupo). Detalle en `docs/teoria/02-fair-loss.md`.

4. **Cómo entra $\lambda$ en la pérdida.** `lambda_fair` se declara con la API de
   Keras Tuner, y el término que multiplica es la FAIR loss de la Tarea 2
   ($\text{BCE} + \lambda\cdot D(\hat y, S)$). La elección
   $(\text{medida}^\star,\lambda^\star)$ se hizo por presupuesto de AUC:
   **corr² con $\lambda^\star=5$**.

5. **Extracción de los pares (Precisión, Dependencia) por trial.** Se registran
   las dos métricas por configuración mediante un **bucle externo sobre
   $\lambda$** (con varias semillas en la comparación de medidas), además de
   `results_summary()` para el barrido de topología del tuner.

6. **Estrategia de búsqueda y presupuesto.** Hyperband y RandomSearch quedaron
   en **empate técnico**, y se eligió **RandomSearch** por simplicidad —en línea
   con el aviso de que lo sofisticado no suele batir a random
   (`clases-master/Automatic_ML_2026.pdf`, pág. 41). Backbone resultante:
   1 capa / 64u / dropout 0.3 / `relu`.

7. **AutoKeras / AutoML "de caja" como alternativa.** La charla describe
   AutoKeras (`StructuredDataClassifier` encajaría con datos tabulares de
   crédito, `clases-master/Automatic_ML_2026.pdf`, pág. 37), pero el taller
   pide explícitamente Keras Tuner para controlar la topología; usar AutoKeras
   daría menos control sobre el barrido de $\lambda$ y la curva de Pareto.

---

### Fuentes citadas

- `clases-master/Keras_Tuner_Basico.ipynb` — patrón Keras Tuner (FUENTE
  PRINCIPAL).
- `clases-master/Optuna_basico.ipynb` — alternativa con Optuna.
- `clases-master/Automatic_ML_2026.pdf` — charla "Machine Learning automático"
  (Valero Laparra): qué es AutoML, opciones, NAS, estrategias, AutoKeras.
- `clases-master/26 06 13.pdf` *(transcripción de clase)* — comentarios
  del profesor sobre Keras Tuner como opción estándar y recomendada.
