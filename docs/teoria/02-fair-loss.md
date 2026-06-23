# Tarea 2 — La función de coste FAIR (penalizar la dependencia predicción↔género)

> Documento de teoría para el Taller B4-T1. Cubre la **Tarea 2**: diseñar una
> función de coste a medida que combine el error de clasificación con una
> **penalización por dependencia estadística** entre la predicción del modelo y
> la variable sensible (`CODE_GENDER`).

---

## 1. Intuición / resumen

### El problema en una frase

Queremos un clasificador de crédito que prediga bien el impago (`TARGET`) pero
cuyas decisiones **no dependan del género** (`CODE_GENDER`). Dicho de forma
operativa: si cambiásemos únicamente el género de un solicitante y dejásemos
todo lo demás igual, la puntuación no debería moverse. Esto es la criterio de
**paridad estadística**:

$$
\hat y \;\perp\; S
\qquad\Longleftrightarrow\qquad
\operatorname{Dep}(\hat y, S) = 0,
$$

donde $\hat y$ es la predicción y $S$ la variable sensible. El lado izquierdo es
una exigencia moral/regulatoria; el derecho es una **cantidad medible**. Esa
equivalencia es la idea central de todo el bloque: *una vez que "justo" es una
dependencia, se puede auditar con un número y entrenar contra él con una
penalización* (`profe-lectures/topics/fairness/01-what-is-fair.qmd`).

### Por qué borrar la variable sensible NO basta (proxies)

El arreglo ingenuo —"si el modelo no debe usar el género, borremos la columna
del género"— **falla**, y es justamente el primer arreglo que se le ocurre a
todo el mundo. Falla porque otras variables son **proxies** del género: el tipo
de ocupación, las horas trabajadas, ciertos sectores. La mecánica es una línea
de álgebra de regresión: si el modelo ajusta $\hat y = X\beta$, entonces

$$
\operatorname{Cov}(\hat y, S) = \beta^\top \operatorname{Cov}(X, S),
$$

y borrar la columna de $S$ solo pone a cero **una** entrada de
$\operatorname{Cov}(X, S)$. Toda variable restante que covaríe con $S$ mantiene
su canal abierto; y si $S$ realmente ayuda a predecir el objetivo, el
optimizador **reclutará activamente** esos canales porque llevan señal que le
estamos pagando por encontrar. Borrar la columna no elimina la dependencia: la
**reenruta** (`profe-lectures/topics/fairness/02-bias-and-proxies.qmd`).

> Consecuencia clave para el taller: la neutralidad hay que **medirla contra
> $S$** y optimizarla, lo que significa que $S$ debe estar disponible en tiempo
> de **entrenamiento**, aunque nunca sea una **entrada** del modelo en
> producción. La auditoría necesita $S$ para testar el enlace; el modelo en sí
> no lo toca al predecir.

Hay un segundo motivo por el que el chequeo estándar es ciego: suele usar la
**correlación de Pearson** entre puntuación y género, y Pearson solo ve líneas
rectas. Una puntuación puede **curvarse** con $S$ (p. ej. depender de $S^2$) y
seguir dando correlación ≈ 0. El número dice "neutral", la dependencia está ahí
(`profe-lectures/topics/fairness/01-what-is-fair.qmd`, figura del "hook";
mismo fenómeno que el par convexo del capítulo de dependencia,
`profe-lectures/topics/dependence/01-correlation.qmd`).

### La idea: penalizar la dependencia, no esconder la variable

En lugar de exigir neutralidad perfecta (que suele costar precisión), la
**ponemos en el precio**. El objetivo de todo el bloque cabe en una línea:

$$
\text{loss} \;=\;
\underbrace{\text{fit}(\hat y, y)}_{\text{predecir bien}}
\;+\; \mu \cdot
\underbrace{\operatorname{Dep}(\hat y, S)}_{\text{ciego ante } S}.
$$

`fit` es el error de predicción habitual; $\operatorname{Dep}$ es **cualquier
medida de dependencia** aplicada entre las predicciones y $S$; y $\mu$ (a veces
$\lambda$) regula el cambio entre precisión y neutralidad. En palabras del
profesor: *"crearnos una función de coste donde voy a minimizar el MSE más voy a
penalizar que haya correlación entre la $\hat y$ y la $S$… quiero que la
correlación sea pequeña, porque es la variable sensible"*
(`clases-master/26 06 18 MIAX 14.pdf`, ~01:27–01:28).

> **Resumen ELI5.** El aprendizaje justo no *esconde* la variable sensible: le
> añade "depender de $S$" a la factura, de modo que el optimizador decide por su
> cuenta dejar de usarla.

---

## 2. Formulación matemática

### El coste total y el papel de λ (μ)

$$
\mathcal{L}(\theta) \;=\;
\mathbb{E}\big[\,\ell(\hat y_\theta, y)\,\big]
\;+\;
\lambda \cdot \operatorname{D}(\hat y_\theta, S),
$$

donde $\ell$ es la pérdida de clasificación (en el taller, **entropía cruzada
binaria** sobre `TARGET`), $\operatorname{D}$ es una medida de dependencia entre
la predicción y la variable sensible, y $\lambda \ge 0$ el peso del
*trade-off*.

Este término no es un truco arbitrario: es el **Lagrangiano** del problema
restringido "minimiza el error sujeto a $\operatorname{D}(\hat y, S) = 0$".
$\lambda$ es el multiplicador de Lagrange. Barriéndolo de $0$ a $\infty$ se
traza la **frontera de Pareto** entre precisión y neutralidad —exactamente la
curva que pide el taller. No hay un único "modelo justo": hay una **curva**, y
$\lambda$ elige el punto de operación
(`profe-lectures/topics/fairness/01-what-is-fair.qmd`, callout del Lagrangiano).

- $\lambda = 0$: modelo puramente preciso, libre de cabalgar sobre la señal de
  $S$ (peor caso de justicia).
- $\lambda$ grande: predicciones cada vez más independientes de $S$, a costa de
  la precisión que descansaba sobre $S$.
- La curva tiene un **codo**: a la derecha del codo la neutralidad es casi
  gratis; a la izquierda, cada caída adicional de la dependencia cuesta
  precisión real (`profe-lectures/topics/fairness/03-fair-kernel-learning.qmd`,
  fig. del *frontier* / elbow).

### Las medidas de dependencia $\operatorname{D}$ candidatas

El material presenta una **escalera** de medidas, cada una capaz de cancelar una
*forma* distinta de dependencia. De más barata-y-ciega a más flexible-y-cara
(`profe-lectures/topics/fairness/index.qmd`, "ladder of removal";
`profe-lectures/topics/dependence/index.qmd`).

#### (a) Correlación de Pearson — solo líneas rectas

$$
r = \frac{\operatorname{cov}(\hat y, S)}{\sigma_{\hat y}\,\sigma_S}\in[-1,1].
$$

Barata y universal, pero solo ve co-movimiento **lineal**. Penalizar $r^2$
("neutralidad lineal") elimina la parte recta de la dependencia y **deja el
agujero curvo**: si $\hat y$ depende de $S$ por una función par
($\hat y = S^2$), entonces $\operatorname{Cov}(S^2,S)=\mathbb{E}[S^3]=0$ para $S$
simétrica, y el residuo parece "no correlacionado" siendo plenamente dependiente
(`profe-lectures/topics/fairness/03-fair-kernel-learning.qmd`, rung 2 y su
callout). Es la medida que el profesor usa en el ejemplo en vivo del taller, y la
más sencilla de implementar.

> Matiz importante para el taller (caso `CODE_GENDER`): cuando $S$ es **binaria**
> (hombre/mujer, 0/1), las medidas kernel apenas registran el desplazamiento de
> medias y se quedan cerca de 0.02; en ese caso conviene reportar el
> **group gap** (diferencia de puntuación media entre grupos) como lectura más
> nítida. La correlación/diferencia de medias gana aquí; CKA brilla cuando $S$ es
> multivariante o continua
> (`profe-lectures/topics/fairness/labs/lab-jobB-neutralize-credit.qmd`, callout
> de la sección 2).

#### (b) HSIC — dependencia sin densidad (núcleo)

La idea kernel: incrustar los datos en un espacio de características más rico y
testar la dependencia **allí**. Con kernels característicos (p. ej. RBF/Gaussiano)
el HSIC es cero *exactamente* bajo independencia, y nunca estima una densidad:

$$
\operatorname{HSIC}(\hat y, S)
  = \frac{1}{(n-1)^2}\operatorname{tr}(K_{\hat y}\,H\,K_{S}\,H),
$$

donde $K_{\hat y}, K_{S}$ son las matrices kernel (Gram) de $\hat y$ y $S$, y
$H = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top$ es la matriz de centrado. El
kernel RBF es

$$
k(z_i, z_j) = \exp\!\Big(\!-\frac{\lVert z_i - z_j\rVert^2}{2\sigma^2}\Big)\in(0,1],
$$

con el ancho de banda $\sigma$ fijado normalmente por la **heurística de la
mediana**, $\sigma = \sqrt{\operatorname{median}(d^2)/2}$
(`profe-lectures/topics/dependence/02-kernels-hsic.qmd`). HSIC capta *cualquier*
forma de dependencia (lineal y curva) y es **diferenciable** respecto de los
parámetros del modelo, lo que lo hace usable como penalización.

#### (b') MMD — comparar las dos distribuciones de score (a medida para $S$ binaria)

HSIC pregunta "¿hay *algún* tipo de dependencia entre $\hat y$ y $S$?". Cuando
$S$ es **binaria** esa pregunta es más fina de lo necesario y, peor, se **diluye
con el desbalance** de grupos (con `CODE_GENDER` HSIC apenas registra ~0.02). La
pregunta natural es más concreta: *¿reciben los dos grupos la misma distribución
de puntuaciones?* Eso es exactamente un **test kernel de dos muestras**, la
**Maximum Mean Discrepancy (MMD)**: en vez de correlacionar $\hat y$ con $S$,
compara **las distribuciones completas** del score por grupo,
$P(\hat y \mid S=1)$ frente a $P(\hat y \mid S=0)$ —no solo sus medias.

Incrustando cada distribución por su media en el RKHS de un kernel
característico (RBF sobre las predicciones), la MMD es la distancia entre esas
dos medias incrustadas. Escrita con **pesos de pertenencia** $w_1 = S$ y
$w_0 = 1-S$ sobre el kernel $K$ de las predicciones (con $n_1=\sum w_1$,
$n_0=\sum w_0$):

$$
\operatorname{MMD}^2 \;=\;
\frac{w_1^\top K\,w_1}{n_1^{\,2}}
\;+\;
\frac{w_0^\top K\,w_0}{n_0^{\,2}}
\;-\;
2\,\frac{w_1^\top K\,w_0}{n_1\,n_0}.
$$

Los dos primeros términos miden la auto-similitud dentro de cada grupo y el
tercero la similitud cruzada; cuando ambas distribuciones coinciden, los tres se
cancelan. Propiedades clave para usarla como penalización:

- **$\operatorname{MMD}^2 = 0$ si y solo si** las dos distribuciones de score
  coinciden (con kernel característico), no solo sus medias —es por tanto una
  exigencia de **paridad estadística** plena, no de mera igualdad de medias.
- **No se diluye con el desbalance**: apunta justo a la diferencia entre los dos
  grupos, así que para $S$ binaria es más nítida que HSIC (que reparte su masa
  sobre "cualquier" dependencia y casi no se mueve).
- Es **diferenciable** (solo `matmul`s sobre el kernel RBF de las predicciones)
  y, gracias a los pesos $w_1,w_0$, no necesita enmascarar por grupo, así que
  encaja en el grafo de Keras/TF como término de coste.

Así, en la escalera: corr² ve solo el desplazamiento de **medias** (lineal),
HSIC ve **cualquier** dependencia genérica, y MMD —entre ambas para el caso
binario— compara las **distribuciones enteras** de los dos grupos, que es
precisamente lo que "no discriminar por género" exige
(`src/fair_loss.py`, `dependence_mmd`).

#### (c) CKA — HSIC normalizado (la elección práctica)

El HSIC en crudo lleva las unidades del kernel al cuadrado: su escala depende del
kernel, el ancho de banda y el tamaño de muestra, así que un valor aislado es
ilegible. Se normaliza igual que Pearson normaliza la covarianza:

$$
\operatorname{CKA}(\hat y, S)
  = \frac{\operatorname{HSIC}(\hat y, S)}
         {\sqrt{\operatorname{HSIC}(\hat y,\hat y)\,\operatorname{HSIC}(S,S)}}
  \;\in\;[0,1].
$$

Ahora el número *habla*: CKA = 0 ⇔ independencia (con kernel característico),
CKA = 1 ⇔ máxima dependencia, y —crucial para el taller— **$\lambda$ significa
lo mismo en todos los datasets** porque penalizar "CKA = 0.3" tiene un
significado fijo, mientras que "HSIC = 0.003" no significa nada en aislado
(`profe-lectures/topics/dependence/02-kernels-hsic.qmd`;
`profe-lectures/topics/fairness/03-fair-kernel-learning.qmd`, rung 3).

> Regla práctica del material: usa **HSIC** dentro de optimizadores y tests
> (las constantes se cancelan); usa **CKA** cuando el número deba *leerse*
> (paneles, *scorecards*, penalizaciones con peso interpretable). El kernel
> lineal recupera los clásicos: con $k(z_i,z_j)=z_iz_j$, CKA colapsa a la
> correlación de Pearson al cuadrado (`cka_linear`).

#### (d) Información mutua — dependencia en bits

La vía "a través de la densidad". La información mutua mide cuántos bits se
malgastan al suponer independencia, es decir la divergencia KL entre el conjunto
real y el producto de marginales:

$$
I(\hat y; S) = D_{\mathrm{KL}}\!\big(p(\hat y, S)\,\|\,p(\hat y)\,p(S)\big)
            = H(\hat y) + H(S) - H(\hat y, S) \;\ge\; 0,
$$

cero *solo* bajo verdadera independencia; ve dependencia no lineal y no monótona;
y reporta unidades reales (bits o nats). Sus contras: necesita estimar una
densidad, es ávida de datos y el valor depende del estimador
(`profe-lectures/topics/dependence/03-mutual-information.qmd`). Como penalización,
la "vía de la información" reconecta con Gaussianization y, en la práctica, solo
la variante **condicional** (Gaussianizar $\hat y$ *dado* $S$) produce un modelo
genuinamente justo; la Gaussianización marginal o conjunta **no** basta
(`profe-lectures/topics/fairness/04-tie-back.qmd`).

#### Resumen de qué cancela cada rung

| Rung | $\operatorname{D}$ | Elimina | Deja |
|------|--------------------|---------|------|
| 1 · Borrar la columna | — | nada (los proxies filtran) | todo |
| 2 · Neutralidad lineal | Pearson / `cka_linear` | la recta | la curva |
| 3 · Penalización CKA | `cka_rbf` (HSIC normalizado) | recta **y** curva | ~nada |
| 4 · Información | MI / total correlation | cualquier forma (vía densidad) | ~nada |
| 5 · Adversarial | sonda aprendida | lo que una red pueda hallar | depende |

(`profe-lectures/topics/fairness/index.qmd`, "ladder of removal";
`profe-lectures/topics/fairness/04-tie-back.qmd`, fig. "what each rung can
cancel".)

---

## 3. Implementación: custom loss en Keras/TF

### El patrón de función de coste a medida

El material del profesor enseña primero el **patrón general** de una loss
customizada en Keras (`clases-master/Keras_custom_loss_clase.ipynb`): una función
`(y_true, y_pred) -> escalar` construida con `keras.ops`, que se pasa al
`compile`. Ejemplo del notebook (loss asimétrica que penaliza 10× los errores en
retornos negativos):

```python
import keras

lambda_val = 10.0

def custom_loss(y_true, y_pred):
    squared_error = keras.ops.square(y_pred - y_true)
    loss_part_positive = keras.ops.where(y_true >= 0, squared_error, 0.0)
    loss_part_negative = keras.ops.where(y_true < 0,
                                         lambda_val * squared_error, 0.0)
    return keras.ops.mean(loss_part_positive + loss_part_negative)

model.compile(loss=custom_loss, optimizer='SGD', metrics=['mse'])
```

Dos lecciones del notebook: (1) toda la aritmética debe ir con `keras.ops` (no
NumPy) para que sea **diferenciable** y corra en el backend; (2) un modelo
entrenado *con* la loss customizada gana en esa métrica frente a uno entrenado
con MSE (sección "SIN CUSTOM LOSS"). El mismo notebook contiene
`Keras_custom_loss_clase_sharpe.ipynb`
(`clases-master/Keras_custom_loss_clase_sharpe.ipynb`), donde la loss es **el
ratio de Sharpe negativo** —un ejemplo de pérdida financiera a medida que opera
sobre el *batch completo* (media y desviación de los retornos de cartera):

```python
def sharpe_ratio_loss(y_true, y_pred):
    portfolio_returns = keras.ops.sum(y_pred * y_true, axis=-1)
    mean = keras.ops.mean(portfolio_returns)
    std  = keras.ops.std(portfolio_returns) + keras.backend.epsilon()
    return -(mean / std)          # maximizar Sharpe = minimizar su negativo
```

Esto es directamente relevante: igual que el Sharpe necesita estadísticos del
batch (media, std), **la penalización de dependencia necesita estadísticos del
batch** (matrices kernel sobre $\hat y$ y $S$ del batch).

### Añadir el término de penalización

El esqueleto de la loss FAIR combina el ajuste y la dependencia:

```python
def fair_loss(y_true, y_pred, S, lam):
    fit = keras.losses.binary_crossentropy(y_true, y_pred)   # clasificación
    dep = dependence(y_pred, S)         # p.ej. cka_rbf(y_pred, S) o corr²
    return keras.ops.mean(fit) + lam * dep
```

El material muestra que esto se hace **end to end** con la librería del curso
`keras-fairkl` (`FairModelWrapper`), que envuelve *cualquier* modelo Keras y
añade $\mu\cdot\operatorname{CKA}(\hat y, S)$ a la task-loss que pongas en
`compile`. El detalle clave: la variable sensible entra en `fit` como argumento
`q` (solo en entrenamiento), nunca como entrada de predicción
(`profe-lectures/topics/fairness/03-fair-kernel-learning.qmd`, sección
"keras-fairkl"; `profe-lectures/topics/fairness/labs/lab-tool2-cka-penalty.qmd`):

```python
import os; os.environ["KERAS_BACKEND"] = "jax"
import keras
from keras import layers
from fairkl import FairModelWrapper

X_in = np.column_stack([x, S])     # la red VE S; la penalización la vigila

mlp = keras.Sequential([
    layers.Dense(32, activation="tanh"),
    layers.Dense(32, activation="tanh"),
    layers.Dense(1),
])
model = FairModelWrapper(mlp, mu=4.0)              # loss = task_loss + mu·CKA(ŷ,S)
model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")
model.fit(X_in, y[:, None], q=S[:, None],          # S entra como q (training-only)
          epochs=400, batch_size=len(x), verbose=0)
```

> Nota: en el lab se sugiere intercambiar `CKALoss` por `HSICLoss`
> (`FairModelWrapper(mlp, mu=..., fairness_loss=fairkl.HSICLoss())`). El mismo
> $\mu$ se comporta distinto porque CKA está acotado en $[0,1]$ y HSIC no
> (`profe-lectures/topics/fairness/labs/lab-tool2-cka-penalty.qmd`, sec. 4).

El lab `lab-tool2-cka-penalty.qmd` también muestra el patrón de **barrido de
$\mu$** para trazar la frontera (la rutina docente `fair_fit` es un *stand-in* de
forma cerrada; la versión de producción es la red entrenada):

```python
sweep = [0.0, 0.5, 1.0, 2.0, 5.0, 8.0, 20.0]
for mu in sweep:
    p   = fair_fit(x, y, S, mu=mu)
    acc = max(1 - np.mean((p - y)**2) / np.var(y), 0.0)
    dep = cka_rbf(p, S)
    # (acc, dep) es un punto de la curva de Pareto
```

### Nota práctica: la loss necesita acceso a S en el batch

La penalización se evalúa entre $\hat y$ y $S$ **del mismo batch**, así que el
entrenamiento debe entregar $S$ a la loss. Las dos vías que aparecen en el
material:

1. **Pasar $S$ como argumento `q` en `fit`** (lo que hace `keras-fairkl`):
   training-only, nunca entrada de predicción.
2. **Concatenar $S$ a `y_true`** (truco clásico de Keras): empaquetar
   `[y, S]` como objetivo y desempaquetarlo dentro de la loss
   (`y_real, S = y_true[:,0], y_true[:,1]`), de modo que la firma estándar
   `(y_true, y_pred)` siga valiendo.

En ambos casos $S$ debe poder mezclarse por batches de forma coherente con
$\hat y$, y la métrica de dependencia debe ser diferenciable (de ahí
HSIC/CKA con `keras.ops`, o $r^2$ con momentos del batch).

---

## 4. Conexión con el Taller B4-T1

### Mapeo de variables

- **$S$ = `CODE_GENDER`**: variable sensible **binaria** (1 = un género, 0 = el
  otro). El objetivo es un modelo "que no discrimine por género"
  (`clases-master/26 06 18 MIAX 14.pdf`, ~01:14).
- **$\hat y$ = $P(\text{TARGET}=1)$**: la probabilidad de impago que produce el
  clasificador neuronal (salida sigmoide).
- **`fit`** = entropía cruzada binaria contra `TARGET` (1 = impago, 0 = pagó).
- **Entradas legítimas**: ingresos, anualidades, `EXT_SOURCE_1/2/3`. `CODE_GENDER`
  **no** es entrada de predicción, pero sí está disponible en entrenamiento para
  alimentar la penalización.

$$
\mathcal{L} \;=\;
\underbrace{\text{BCE}\big(\hat y,\ \text{TARGET}\big)}_{\text{clasificación}}
\;+\;
\lambda\cdot\underbrace{\operatorname{D}\big(\hat y,\ \texttt{CODE\_GENDER}\big)}_{\text{penalización FAIR}}.
$$

### Qué medida de dependencia elegir y por qué

Hay una tensión que el propio material resuelve:

- El **camino más directo y fiel al taller** (lo que el profesor hace en vivo) es
  penalizar la **correlación** entre $\hat y$ y $S$: *"penalizar que haya
  correlación entre la $\hat y$ y la $S$… que la correlación sea pequeña"*
  (`clases-master/26 06 18 MIAX 14.pdf`). Es simple, diferenciable con momentos
  del batch, y suficiente porque `CODE_GENDER` es binaria (la dependencia
  relevante es esencialmente un desplazamiento de medias entre grupos).
- La **opción "héroe"** del material es la **penalización CKA** (`cka_rbf`), que
  además captura dependencia no lineal, está acotada en $[0,1]$ (así $\lambda$ es
  interpretable y comparable) y es diferenciable
  (`profe-lectures/topics/fairness/03-fair-kernel-learning.qmd`, rung 3).
- **Aviso del propio material para $S$ binaria**: con una sola columna sensible
  binaria, CKA "apenas registra" el desplazamiento de medias (se queda ~0.02), y
  conviene reportar el **group gap**
  $\;\Delta = \overline{\hat y}_{S=1} - \overline{\hat y}_{S=0}\;$ como lectura
  principal de (in)justicia
  (`profe-lectures/topics/fairness/labs/lab-jobB-neutralize-credit.qmd`).

> Recomendación operativa coherente con las fuentes: **penalizar con
> correlación/HSIC durante el entrenamiento** y **reportar el group gap (y/o las
> tasas por grupo: aprobación y falso-rechazo)** como entregable de justicia.
> El caso `lab-jobB-neutralize-credit.qmd` es prácticamente el mismo problema
> (modelo que descarta el atributo protegido, demuestra la fuga vía proxies, y
> neutraliza el resultado) y usa exactamente este esquema.

> **Lo que pasó en la práctica (NB06).** Se compararon en pie de igualdad las
> **tres** medidas —corr², HSIC y MMD— barriendo $\lambda$ con cada una y
> eligiendo el compromiso por **presupuesto de AUC en validación**. Para
> `CODE_GENDER` (binaria) ganó **corr²**: con el mismo presupuesto de pérdida de
> precisión, la penalización lineal sobre la diferencia de medias es la que más
> reduce el group gap, mientras que HSIC se diluye con el desbalance y la mayor
> expresividad de MMD no compensa su coste para una sensible binaria. La MMD
> sigue siendo la medida conceptualmente *a medida* del caso binario (compara
> las distribuciones enteras), pero aquí la diferencia entre grupos es
> esencialmente un desplazamiento de medias que corr² ya captura.

### Cómo barrer λ para la curva de Pareto

El entregable de la Tarea 2 es la **frontera Precisión vs Dependencia FAIR**:

1. Para cada $\lambda \in \{0,\ 0.5,\ 1,\ 2,\ 5,\ 8,\ 20,\dots\}$, entrenar el
   clasificador con $\mathcal{L} = \text{BCE} + \lambda\cdot\operatorname{D}$.
2. Para cada modelo, registrar **(a)** una métrica de precisión (AUC / accuracy
   sobre `TARGET`) y **(b)** la dependencia residual ($\operatorname{D}(\hat y,S)$
   y/o group gap).
3. Dibujar precisión vs dependencia: cada $\lambda$ es un punto de la curva. El
   **codo** marca dónde la neutralidad deja de ser "casi gratis" y empieza a
   costar precisión real; elegir el punto de operación es una decisión de
   política, no de matemáticas
   (`profe-lectures/topics/fairness/03-fair-kernel-learning.qmd` y
   `lab-tool2-cka-penalty.qmd`, fig. del *frontier*).

El criterio práctico del lab: *"elige el $\lambda$ más pequeño que empuje la
dependencia por debajo de tu umbral; todo lo que esté a su izquierda compra
neutralidad que quizá no necesites a un coste de precisión que sí pagas"*.

---

## 5. Huecos / decisiones pendientes

Lo que el material **no zanja** para el taller y hay que decidir al implementar:

1. **Qué medida concreta usar en la penalización.** Las fuentes ofrecen un menú
   (correlación, HSIC, CKA, MI). Para `CODE_GENDER` binaria, correlación / group
   gap es lo más natural y lo que hace el profesor; CKA es el "héroe" general
   pero "apenas registra" con $S$ binaria. **Decisión abierta**: penalizar con
   correlación²/HSIC pero *reportar* group gap, o forzar CKA igualmente. El
   material recomienda lo primero, sin imponerlo.

2. **Cómo estimar la dependencia en mini-batch de forma diferenciable.** El
   HSIC/CKA construye matrices $n\times n$; en batches grandes esto es costoso y
   el estimador es ruidoso en batches pequeños. No hay en las fuentes una receta
   de tamaño de batch ni de estimador *minibatch* de CKA para clasificación
   binaria; `keras-fairkl` lo encapsula pero no se detalla el estimador interno.
   El ancho de banda $\sigma$ del RBF se fija por mediana, pero **recalcularlo
   por batch vs fijarlo global** queda a criterio.

3. **Cómo pasar $S$ a la loss en Keras.** Dos patrones documentados (argumento
   `q` de `FairModelWrapper`, o concatenar `[y, S]` en `y_true`), pero la
   elección concreta de implementación para el taller no está fijada. Si no se
   usa `keras-fairkl`, hay que cablear el desempaquetado a mano.

4. **`fit` exacta y arquitectura.** El taller es **clasificación** (BCE), pero
   los ejemplos del profesor de custom loss son de **regresión** (MSE, Sharpe).
   Hay que adaptar el patrón a salida sigmoide + BCE; no hay un notebook del
   profe que combine BCE + penalización de género (es justo lo que la Tarea 2
   pide construir).

5. **Definir la dependencia sobre la probabilidad $\hat y$ o sobre el logit.**
   Las fuentes hablan de "predicciones" $\hat y$ genéricas; para un clasificador
   conviene decidir si la penalización opera sobre $P(\text{TARGET}=1)$ o sobre
   el score pre-sigmoide. No está especificado.

6. **Métrica de precisión para la frontera.** El material usa $R^2$ (regresión);
   para el taller hay que elegir AUC, accuracy, o KS sobre `TARGET`. Decisión
   abierta.

7. **Umbral de "justo" y punto de operación.** Dónde pararse en la curva de
   Pareto es explícitamente una decisión de política/ética, no matemática
   (`profe-lectures/topics/fairness/03-fair-kernel-learning.qmd`). El taller
   tendrá que fijar un criterio (p. ej. group gap por debajo de X) para reportar
   un modelo concreto.
</content>
</invoke>
