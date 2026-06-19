# Fundamentos de dependencia y gaussianización

> **Documento de apoyo (marco conceptual breve).** Recoge solo los fundamentos
> de medición de dependencia estadística y de gaussianización que sustentan la
> penalización FAIR de la Tarea 2. El desarrollo a fondo de la FAIR loss está en
> [`docs/teoria/02-fair-loss.md`](02-fair-loss.md).

---

## 1. Intuición / resumen

**Dependencia** es la respuesta a una pregunta simple: *¿saber el valor de una
variable me dice algo sobre la otra?* La correlación de Pearson responde solo a
una versión recortada de esa pregunta —*¿se mueven juntas en línea recta?*— y por
eso tiene puntos ciegos: una relación fuerte pero curva o en forma de V puede dar
Pearson ≈ 0 aunque una variable sea casi una función determinista de la otra
(`profe-lectures/topics/dependence/01-correlation.qmd`). En finanzas esto es el
fallo motivador: una cartera con correlaciones bajas por todas partes que aun así
cae toda a la vez ante un shock (`profe-lectures/topics/dependence/index.qmd`).

La fuente organiza todas las medidas en **dos caminos**
(`profe-lectures/topics/dependence/index.qmd`):

1. **Sin densidad** — comparar los datos directamente:
   `medida = comparar(transformar(x), transformar(y))`. Cubre Pearson, Spearman,
   correlación de distancias y los métodos de kernel (HSIC/CKA).
2. **A través de la densidad** — leer la dependencia de `p(x,y)` como
   *información* (información mutua, correlación total). Requiere estimar una
   densidad: o se *asume* su forma (paramétrico) o se *aprende* (gaussianización,
   flows).

**Gaussianizar** es aprender un mapa **invertible** que convierte datos
enredados y dependientes en una gaussiana estándar, donde cada coordenada es
independiente (`profe-lectures/topics/gaussianization/index.qmd`). Sirve para dos
cosas: hacia delante *mide y elimina* dependencia (la distorsión que hace falta
para gaussianizar **es** la dependencia); hacia atrás *genera* muestras. Para
nosotros importa la dirección hacia delante: hace estimable la información mutua
en dimensiones altas, donde un histograma es inviable.

---

## 2. Medidas clave (fórmula esencial)

### Camino 1 — sin densidad

**Pearson `r`** — co-movimiento lineal estandarizado
(`profe-lectures/topics/dependence/01-correlation.qmd`):

$$ r = \frac{\operatorname{cov}(X,Y)}{\sigma_X\,\sigma_Y} \in [-1,1]. $$

Solo ve líneas rectas; además confunde *fuerza* del vínculo con *estrechez* del
ruido alrededor.

**Spearman `ρ`** — el mismo cálculo sobre los rangos:
`Spearman = corr(rank(x), rank(y))`. Captura cualquier relación *monótona*, pero
falla con una U/V.

**Correlación de distancias** `dCor` — mismo molde con una transformación más
rica (distancias por pares doblemente centradas):

$$ \operatorname{dCor}(X,Y)=\frac{\operatorname{dCov}(X,Y)}{\sqrt{\operatorname{dVar}(X)\,\operatorname{dVar}(Y)}}\in[0,1],\qquad \operatorname{dCor}=0 \iff X\perp Y. $$

Cero *exactamente* bajo independencia: la propiedad que la separa de Pearson y
Spearman.

**HSIC / CKA** — kernels (`profe-lectures/topics/dependence/02-kernels-hsic.qmd`).
Un kernel es "una distancia leída como cercanía" (RBF gaussiano). HSIC compara las
matrices de similitud centradas:

$$ \operatorname{HSIC}(X,Y)=\frac{1}{(n-1)^2}\operatorname{tr}(K_X H K_Y H), $$

con $H = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top$ la matriz de centrado. Con un
kernel característico (RBF), HSIC = 0 *solo* bajo independencia, y **nunca estima
una densidad**. Como su valor bruto no es interpretable (depende del kernel, el
ancho de banda y `n`), se normaliza a **CKA**:

$$ \operatorname{CKA}(X,Y)=\frac{\operatorname{HSIC}(X,Y)}{\sqrt{\operatorname{HSIC}(X,X)\,\operatorname{HSIC}(Y,Y)}}\in[0,1]. $$

Regla práctica de la fuente: usar **HSIC** dentro de optimizadores y tests (las
constantes se cancelan, es diferenciable); usar **CKA** cuando el número deba
*leerse* (dashboards, pesos de penalización interpretables). El kernel lineal
recupera $r^2$ como caso particular.

### Camino 2 — a través de la densidad

**Información mutua** `I(X;Y)`
(`profe-lectures/topics/dependence/03-mutual-information.qmd`). Tres lecturas del
mismo número:

$$ I(X;Y)=\underbrace{D_{\mathrm{KL}}\!\big(p(x,y)\,\|\,p(x)p(y)\big)}_{\text{KL desde la independencia}}=\underbrace{H(X)+H(Y)-H(X,Y)}_{\text{entropía compartida}}\ge 0. $$

Mide en **bits/nats** la distancia entre el conjunto real y la referencia de
independencia $p(x)p(y)$. Es cero *solo* bajo independencia real y ve dependencia
no lineal y no monótona. Generaliza a un vector como **correlación total**
$\mathrm{TC}(\mathbf{x}) = \sum_i H(x_i) - H(\mathbf{x})$.

**Linfoot R** — reescala la MI al rango $[0,1]$ tipo correlación:
$R = \sqrt{1 - e^{-2I}}$. Para una gaussiana bivariante devuelve exactamente
$|\rho|$ —"la correlación es la sombra que proyecta la MI cuando el mundo es
gaussiano"—; cualquier exceso de $R$ sobre $r$ es dependencia no gaussiana
(colas, curvatura).

### Gaussianización como vía de estimación

El obstáculo del Camino 2 es estimar $p(\mathbf{x})$: en 1-D es fácil, pero un
histograma necesita ~$10^d$ muestras (maldición de la dimensionalidad)
(`profe-lectures/topics/dependence/05-gaussianization.qmd`). La solución es
*aprender un mapa invertible* que vuelva los datos gaussianos. La regla base es el
**cambio de variables** (`profe-lectures/topics/gaussianization/01-change-of-variables.qmd`):

$$ p_X(x)=p_Z\big(T(x)\big)\,\Big|\det \tfrac{\partial T}{\partial x}\Big|. $$

El movimiento 1-D es `T(x) = Φ⁻¹(F̂_X(x))` (uniformizar con la CDF, luego salir por
la gaussiana), monótono e invertible. Pero reformar marginales no toca la
dependencia *entre* ejes; por eso **RBIG**
(`profe-lectures/topics/gaussianization/02-rbig.qmd`) alterna
*gaussianizar marginales* + *rotar*, iterado, hasta una gaussiana esférica. Como
cada paso es invertible, la dependencia eliminada se contabiliza vía el Jacobiano
y **es** la correlación total
(`profe-lectures/topics/gaussianization/03-mutual-information.qmd`). Punto clave:
solo la gaussianización **conjunta** elimina dependencia; gaussianizar cada
marginal por separado no.

---

## 3. Por qué sustenta la FAIR loss

La Tarea 2 penaliza la dependencia estadística entre la predicción $\hat{y}$ y la
variable sensible $S = $ `CODE_GENDER`. Esa penalización necesita un término
cuantitativo `D(ŷ, S)` que sea **diferenciable** y que valga cero *exactamente*
cuando predicción y atributo sensible son independientes. Las medidas de este
documento son justo las candidatas:

$$ \mathcal{L} = \text{loss de ajuste} + \mu \cdot D(\hat{y}, S). $$

La propia fuente lo señala: HSIC es diferenciable respecto a los parámetros, así
que "medir una dependencia y prohibirla son la misma idea con el signo cambiado"
(`profe-lectures/topics/dependence/02-kernels-hsic.qmd`). Es exactamente la
penalización de equidad
$\mathcal{L} = \text{fit loss} + \mu \cdot \operatorname{HSIC}(\hat{y}, S)$, con
CKA preferible cuando $\mu$ deba ser interpretable. La información mutua vía
gaussianización ofrece la alternativa basada en densidad
(`profe-lectures/topics/gaussianization/03-mutual-information.qmd`), conectada con
fair learning según las propias notas
(`profe-lectures/topics/fairness/03-fair-kernel-learning.qmd`, citada de pasada en
las fuentes).

**El desarrollo de qué `D` se elige, cómo se implementa y cómo se calibra `μ`
está en [`docs/teoria/02-fair-loss.md`](02-fair-loss.md).** Este documento solo
aporta el marco de medición.

---

## 4. Huecos

- **No se desarrolla** la elección concreta de `D(ŷ, S)` para la Tarea 2 ni su
  implementación: corresponde a `02-fair-loss.md`. Las fuentes solo dan la forma
  general de la penalización.
- La página de fairness (`profe-lectures/topics/fairness/03-fair-kernel-learning.qmd`)
  se cita en las notas de gaussianización pero **no se leyó directamente** aquí;
  su contenido específico queda fuera de este marco.
- Detalles que se omiten por ser soporte: correlación cruzada multivariante
  (coeficiente RV), ventanas/bloques temporales, divergencia KL en profundidad y
  causalidad (`04-windows-and-blocks.qmd`, `06-causality.qmd`).
- No se cubre cómo se obtiene $\hat{y}$ ni la naturaleza de `CODE_GENDER` en el
  dataset del taller (binaria/categórica), lo que puede condicionar qué medida es
  más apropiada.
