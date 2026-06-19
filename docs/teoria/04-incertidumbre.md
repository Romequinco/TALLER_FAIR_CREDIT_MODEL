# Tarea 4 — Incertidumbre: devolver clase + varianza

> **Cómo leer este documento.** El material del profesor enfoca la incertidumbre
> hacia **finanzas** (VaR, bandas predictivas, propagación de ruido, Monte-Carlo
> gaussiano) y hacia **regresión** (predecir media y ruido de un valor continuo).
> El Taller B4-T1 pide algo distinto: un **clasificador binario** de crédito que,
> sobre `test`, devuelva la **clase** predicha *y* una **varianza / incertidumbre**.
> Por eso, a lo largo del texto:
> - **[Del material]** marca lo que está literalmente en los `.qmd` del profe o en
>   la transcripción de clase.
> - **[Propuesta razonada]** marca el puente que construyo hacia el clasificador,
>   coherente con el material pero no escrito explícitamente en él.

---

## 1. Intuición / resumen

### 1.1 La incertidumbre *es* probabilidad

**[Del material]** La tesis central de todo el capítulo es literal:

> "Uncertainty *is* probability. This chapter treats inputs, parameters, and
> predictions as **distributions**, not point values."
> — `profe-lectures/topics/uncertainty/index.qmd`

Una predicción no es un número, es una distribución. Reportar solo el punto
(`ŷ`) tira exactamente lo que importa al riesgo: *cuánto* podrías equivocarte y
*en qué dirección* (`01-uncertainty-is-probability.qmd`, "A Point Estimate Hides
the Tail"). El capítulo incluso da un número para "cuánta" incertidumbre hay: la
**entropía** `H(X) = E[log 1/p(X)]` (la sorpresa media).

### 1.2 Dos preguntas, dos actos

**[Del material]** El material organiza la incertidumbre en dos preguntas
(`index.qmd`, `01-uncertainty-is-probability.qmd`):

- **Acto 1 — ¿Qué puede hacer el mundo?** Generar escenarios del *conjunto*
  (joint) de una cartera y precificar la cola (VaR). Es la "escalera" de
  generadores: independiente → gaussiano → Student-t → RBIG
  (`03-noise-and-propagation.qmd`).
- **Acto 2 — ¿Cuán equivocada está mi predicción?** Una banda alrededor de
  `ŷ = f(x)` (`05-predictive-bands.qmd`).

El Taller vive en el **Acto 2**: no nos preguntamos por la cola de una cartera,
sino por la confianza de una predicción concreta.

### 1.3 Dos tipos de incertidumbre (aleatoria vs epistémica)

**[Del material]** El profe separa explícitamente dos fuentes
(`05-predictive-bands.qmd`):

$$
\text{varianza predictiva} =
\underbrace{\sigma^2_{\text{aleatoria}}}_{\text{ruido del mundo}} +
\underbrace{\sigma^2_{\text{epistémica}}}_{\text{duda del modelo}}
$$

- **Aleatoria** — *¿cuánto ruido hay aquí?* Irreducible: más datos no la
  encogen. La respuesta es **predecirla**.
- **Epistémica** — *¿el modelo ha visto datos aquí?* Curable: encoge al añadir
  datos. La respuesta es **admitirla** (un comité de redes confiesa lo que una
  sola no puede).

El test que las separa, literal del material: **¿más datos la reducirían?**
Sí → epistémica. No → aleatoria.

> **Por qué importa para el Taller [Propuesta razonada].** La pregunta de
> análisis del Taller (¿hay más incertidumbre cuando `EXT_SOURCE_1/2/3` están
> ausentes/imputadas?) es, en el vocabulario del profe, sobre todo una pregunta
> de incertidumbre **epistémica**: en esos perfiles el modelo dispone de peor
> información, así que *debería* estar menos seguro. Esta distinción
> aleatoria/epistémica es del material; aplicarla a `EXT_SOURCE` es el puente.

### 1.4 Un modelo honesto

**[Del material]** "Claim only what your evidence justifies"
(`01-uncertainty-is-probability.qmd`). Un modelo bien calibrado a veces debe
volverse *menos* seguro. En clase Valero lo aterriza al caso de crédito:

> "para esos casos de mucha incertidumbre, a lo mejor que lo revisara un humano
> [...] con que saltara una alarma."
> — `clases-master/26 06 18 MIAX 14.pdf`

Es decir, la utilidad práctica de devolver la varianza es: las predicciones muy
inciertas se derivan a revisión humana en vez de automatizarse.

---

## 2. Formulación matemática

### 2.1 Propagación de ruido (lo del profe)

**[Del material]** `02-probability-under-transformation.qmd`. Empujar una
distribución por una función la **reforma** según su derivada (regla de cambio
de variable):

$$
p_Y(y) = \frac{p_X(x)}{\lvert f'(x)\rvert}.
$$

Para ruido pequeño, esto colapsa a la fórmula de laboratorio:

$$
\sigma_y \approx \lvert f'(x)\rvert \, \sigma_x .
$$

El error de salida escala con la **pendiente** del sistema. Cuando la función se
curva o el ruido es ancho, la fórmula lineal ya no basta y solo queda
**muestrear**: ese es el puente a Monte-Carlo.

### 2.2 Monte-Carlo gaussiano (lo del profe)

**[Del material]** `03-noise-and-propagation.qmd` (Rung 2) y
`labs/lab-tool2-gaussian-mc.qmd`. Se ajusta media y covarianza y se muestrea una
normal multivariante:

$$
x = \mu + L\,z, \qquad z \sim \mathcal N(0, I), \qquad L L^\top = \Sigma .
$$

El factor de Cholesky `L` "colorea" ruido blanco con la covarianza ajustada.
Captura el co-movimiento **lineal**, pero sus colas son finas (subestima el VaR).
La idea reutilizable para el Taller no es el VaR en sí, sino el **patrón**:
*generar muchas realizaciones y leer la dispersión de la salida.*

### 2.3 Bandas predictivas: predecir el ruido (lo del profe)

**[Del material]** `05-predictive-bands.qmd` (Tool A) y `04-regression-recipe.qmd`.
Para **regresión**, en vez de una salida (la media) se dan **dos** salidas por
entrada — `μ(x)` y `σ(x)` — y se entrenan con la **log-verosimilitud gaussiana
negativa (NLL)**:

$$
L(\theta) = \frac{1}{n}\sum_i \left[
\frac{\big(y_i - \mu(x_i)\big)^2}{2\,\sigma(x_i)^2} + \log \sigma(x_i)
\right].
$$

Lectura de incentivos: el error grande se *perdona* donde la red declara `σ(x)`
grande, pero declarar `σ` grande en todas partes se *penaliza* con `log σ`. "La
pérdida **es** el modelo de ruido, entrenado." Esta es la versión continua de
"predecir la incertidumbre".

### 2.4 Ensembles y MC-Dropout: la duda del modelo (lo del profe)

**[Del material]** `05-predictive-bands.qmd` (Tool B). La incertidumbre
epistémica es invisible para el modelo que la tiene ("a network cannot grade its
own homework"), así que se pregunta a un **comité**:

1. Reentrenar la **misma** red sobre datos remuestreados, con init aleatoria
   distinta → **deep ensemble**.
2. Donde los miembros **coinciden**, los datos fijaron el modelo.
3. Donde **divergen**, el modelo está adivinando.

Con `T` miembros que predicen `f_t(x)`:

$$
\mu_{\text{epi}}(x) = \frac{1}{T}\sum_{t=1}^{T} f_t(x), \qquad
\sigma^2_{\text{epi}}(x) = \frac{1}{T}\sum_{t=1}^{T}\big(f_t(x) - \mu_{\text{epi}}(x)\big)^2 .
$$

Y el material cita explícitamente las alternativas más baratas
(`05-predictive-bands.qmd`, callout "The Committee's Cheaper Cousins"; y
`labs/lab-tool3-bands.qmd`, sección "Your Turn"):

> "**MC dropout** — train one network with dropout layers and *leave dropout on
> at prediction time*; each stochastic forward pass is a slightly different
> sub-network, a committee hiding inside one model."

La duda total combina ambas fuentes en cuadratura:

$$
\sigma^2_{\text{total}}(x) = \sigma^2_{\text{aleatoria}}(x) + \sigma^2_{\text{epistémica}}(x).
$$

### 2.5 Puente: media y varianza de la *probabilidad* en clasificación

**[Propuesta razonada]** El Taller es un clasificador binario Keras (salida una
probabilidad `p = P(impago)` vía sigmoide). El material no da la fórmula para
"clase + varianza" en clasificación, pero se obtiene trasladando 2.4 de la salida
de regresión a la **probabilidad predicha**.

Hagamos `T` pasadas estocásticas (MC-Dropout con `dropout` activo en inferencia,
o `T` miembros de un ensemble). Cada pasada da una probabilidad `p_t(x)`:

$$
\bar p(x) = \frac{1}{T}\sum_{t=1}^{T} p_t(x)
\quad\Longrightarrow\quad
\hat y(x) = \mathbf{1}\big[\bar p(x) > \tau\big] \;\;(\text{la CLASE}),
$$
$$
\operatorname{Var}[p(x)] = \frac{1}{T}\sum_{t=1}^{T}\big(p_t(x) - \bar p(x)\big)^2
\quad (\text{la VARIANZA / incertidumbre}).
$$

- La **media** de las `T` probabilidades fija la clase (con umbral `τ`, normalmente
  0.5 salvo ajuste por coste/desbalanceo).
- La **varianza** de esas `T` probabilidades es la incertidumbre epistémica que
  pide el Taller.

> Matiz importante **[Propuesta razonada]**: en clasificación binaria coexisten
> dos "incertidumbres" que conviene no confundir:
> 1. La **confianza propia** de una única pasada: `p` cercana a 0.5 ya indica
>    duda (incertidumbre aleatoria; su varianza Bernoulli es `p(1−p)`).
> 2. La **dispersión entre pasadas** `Var[p]` (epistémica): cuánto baila la
>    propia `p` al perturbar el modelo.
>
> Esta distinción la apunta Valero en clase para el caso de crédito: una `p`
> intermedia "ya es un indicio de que no lo tiene muy claro [...] pero yo no le
> estoy pidiendo que me dé su varianza [...] van a estar relacionados, pero no
> tiene por qué ser exacta" (`clases-master/26 06 18 MIAX 14.pdf`). Es decir,
> reconoce que `p≈0.5` y la varianza están **correlacionadas pero no son lo
> mismo**.

---

## 3. Implementación (patrón, no el código del Taller)

### 3.1 El patrón del profe: Monte-Carlo gaussiano

**[Del material]** `lab-tool2-gaussian-mc.qmd`: ajustar `μ, Σ`, sacar `L`
(Cholesky), muestrear `z ~ N(0,I)`, devolver `μ + L z`, y leer la dispersión de
la salida. El núcleo conceptual ("muchas realizaciones → estadística de la
salida") es el mismo que usaremos en clasificación, cambiando el muestreo
gaussiano por pasadas estocásticas de la red.

### 3.2 El método LITERAL del Taller según Valero: un segundo modelo del error

**[Del material]** En la clase del taller (`clases-master/26 06 18 MIAX 14.pdf`)
Valero implementa "la forma más tonta / la más simple" de medir la incertidumbre
a la salida, y es **la que pide entregar**. El patrón:

1. Entrenar el modelo principal (clasificador) → predicciones sobre train/val.
2. Construir una variable `error = |predicción − realidad|` en train/val.
3. Entrenar un **segundo modelo** que, a partir de las mismas `X` de entrada,
   prediga ese `error` (= una estimación de "cuán lejos estaré de lo real").
4. Sobre `test`, el primer modelo da la **clase** y el segundo da la
   **incertidumbre** estimada.

> "voy a generarme una variable extra [...] le voy a llamar error y ahora voy a
> entrenar otro modelo [...] que, en lugar de predecir el valor de salida,
> prediga el error." — `clases-master/26 06 18 MIAX 14.pdf`

Variante también mostrada por Valero (la "segunda opción"): **añadir la
predicción del primer modelo como entrada extra** del segundo (`X` extendidas =
`X` + `p_predicha`), con la idea de que una `p≈0.5` es informativa de error alto.
En clase observó que aporta poco, porque esa `p` ya se deriva de las mismas `X`
(`clases-master/26 06 18 MIAX 14.pdf`).

> Nota **[Propuesta razonada]**: este "segundo modelo del error" es esencialmente
> el **Tool A** del profe (`05-predictive-bands.qmd`) pero desacoplado en dos
> redes en lugar de dos cabezas con pérdida NLL; produce una incertidumbre de
> tipo **aleatorio/heteroscedástico** (predice la magnitud del error esperado),
> no la dispersión epistémica entre pasadas.

### 3.3 El puente a Keras: MC-Dropout / ensemble para clase + varianza

**[Propuesta razonada]** (anclado en el callout de MC-dropout del material). Patrón
para el clasificador del Taller:

```text
# Opción A — MC-Dropout (una sola red, T pasadas)
#   - La red lleva capas Dropout entre densas.
#   - En INFERENCIA se deja el dropout ACTIVO: model(x, training=True).
preds = [model(X_test, training=True) for _ in range(T)]   # T arrays de probabilidades
p_bar = mean(preds, axis=0)          # media -> probabilidad consenso
var   = var(preds,  axis=0)          # varianza -> incertidumbre (lo que pide el Taller)
clase = (p_bar > tau).astype(int)    # clase predicha

# Opción B — Deep ensemble (T redes con init/datos distintos)
#   - Entrenar T modelos; cada uno predice p_t(X_test); misma agregación.
```

Puntos de diseño:
- `training=True` en inferencia es la clave de MC-Dropout: cada pasada apaga
  neuronas distintas → sub-red distinta → comité dentro de un modelo.
- Devolver `(clase, var)` cumple literalmente el enunciado ("la predicción sobre
  test devuelve tanto la CLASE como la VARIANZA").
- Es compatible con el método de 3.2: se pueden entregar ambas señales (la del
  segundo modelo y la `Var[p]` de MC-Dropout) y compararlas.

---

## 4. Conexión con el Taller B4-T1

### 4.1 Devolver clase + varianza sobre `test`

**[Propuesta razonada]** Tres rutas, todas coherentes con el material; basta una,
pero conviene saber qué tipo de incertidumbre da cada una:

| Ruta | Origen | Qué incertidumbre mide | Esfuerzo |
|------|--------|------------------------|----------|
| Segundo modelo del error (3.2) | **[Del material]** (Valero, clase) | aleatoria/heteroscedástica | bajo (lo que pide entregar) |
| MC-Dropout, T pasadas (3.3 A) | [Propuesta] (callout del material) | epistémica (`Var[p]`) | medio (añadir Dropout) |
| Deep ensemble, T modelos (3.3 B) | **[Del material]** Tool B | epistémica | alto (T entrenamientos) |

Lo mínimo y fiel al profe es la primera; el puente recomendado para enriquecer
el análisis de `EXT_SOURCE` es MC-Dropout, porque la pregunta es epistémica.

### 4.2 Relacionar la varianza con la calidad de `EXT_SOURCE`

**[Propuesta razonada]** `EXT_SOURCE_1/2/3` son fuentes externas de scoring; cuando
faltan y se imputan, el modelo trabaja con peor información. Hipótesis,
en lenguaje del profe: esos perfiles caen en regiones donde "el modelo ha visto
menos" → mayor incertidumbre **epistémica** → mayor `Var[p]`. Pasos sugeridos:

1. Definir un indicador de "calidad de fuentes externas" por fila. Opciones:
   número de `EXT_SOURCE_k` ausentes (0–3), o un flag binario "alguna imputada".
   *(Decisión pendiente, ver §5.)*
2. Calcular la incertidumbre por fila en `test` (la varianza de 4.1).
3. Comparar la distribución de la varianza entre perfiles con fuentes completas
   vs imputadas (boxplot/histograma, o correlación nº-ausentes ↔ varianza).

Esto operacionaliza la idea del material de que la banda debe ser **ancha donde
el modelo es ciego** (`05-predictive-bands.qmd`).

### 4.3 El gráfico Buen vs Mal pagador

**[Propuesta razonada]** El Taller pide un gráfico de **distribución de la
varianza** comparando "Buen pagador" vs "Mal pagador". Forma natural: dos
histogramas (o KDE/violín) superpuestos de la varianza por clase. Referentes en
el material para el estilo "distribución/cobertura": los histogramas de pérdida y
las tiras de calibración de `05-predictive-bands.qmd` y `lab-tool2-gaussian-mc.qmd`.

> Interpretación esperada **[Propuesta razonada]**: si una clase (típicamente la
> minoritaria, "Mal pagador") concentra más varianza, el modelo está
> sistemáticamente menos seguro en ella — justo el tipo de hallazgo que, según
> Valero, justifica derivar esos casos a revisión humana / disparar una alarma
> (`clases-master/26 06 18 MIAX 14.pdf`). Conviene cruzar este gráfico con el de
> §4.2: ¿la mayor varianza coincide con perfiles de `EXT_SOURCE` imputado?

---

## 5. Huecos / decisiones pendientes

Lo que el material **no** resuelve y hay que decidir:

1. **Qué método usar como "varianza" oficial.** El material literal del profe
   ofrece (a) el segundo modelo del error (`clases-master/26 06 18 MIAX 14.pdf`,
   lo mínimo exigido) y (b) ensembles/MC-Dropout (`05-predictive-bands.qmd`). La
   `Var[p]` por muestreo (3.3) es **[Propuesta razonada]**. *Decisión:* elegir
   una como principal; sugiero MC-Dropout por el análisis epistémico de
   `EXT_SOURCE`, manteniendo el segundo-modelo como entrega base.
2. **Número de pasadas / miembros `T`.** El material usa 15 miembros de ensemble
   (`05-predictive-bands.qmd`) y "100 veces" para MC-Dropout en el ejercicio
   propuesto del lab (`lab-tool3-bands.qmd`, "Your Turn"), pero **no fija** un T
   óptimo para clasificación. *Decisión pendiente* (p. ej. T=50–100, comprobando
   estabilidad de la varianza).
3. **Cómo medir "calidad de `EXT_SOURCE`".** No aparece en el material (es
   específico del dataset Home Credit). Hay que decidir: conteo de ausentes
   (0–3), flag binario, o usar la magnitud del valor imputado. Además, si la
   imputación borra la señal de "faltante", conviene **conservar flags de
   missing** antes de imputar para poder hacer el análisis.
4. **Umbral `τ` de clasificación.** El material de regresión no aplica; en un
   problema de crédito desbalanceado el umbral no tiene por qué ser 0.5
   (Valero menciona que la salida es "cercana a una probabilidad", no 0/1 puro,
   `clases-master/26 06 18 MIAX 14.pdf`). *Decisión pendiente* según coste de
   falsos negativos.
5. **Aleatoria vs epistémica en clasificación.** El material define la
   descomposición en cuadratura para regresión (`05-predictive-bands.qmd`); su
   equivalente exacto en clasificación (p. ej. incertidumbre predictiva total vs
   esperada, descomposición por entropía mutua) **no** está en el material y
   queda como posible extensión **[Propuesta razonada]**, no obligatoria.
6. **Calibración.** El profe insiste en que la cobertura media es una promesa
   débil y que hay que comprobar *cuándo* falla (`05-predictive-bands.qmd`,
   "Calibration"). Para el Taller, comprobar si la varianza está *bien calibrada*
   (alta varianza ↔ más errores reales) es deseable pero no lo pide el enunciado;
   queda como mejora.
