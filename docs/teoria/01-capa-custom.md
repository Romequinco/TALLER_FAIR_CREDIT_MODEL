# Tarea 1 — Capa custom: Ratio de Endeudamiento con saturación

> Documento de contexto interno (teoría). No es el entregable del taller.
> Cada afirmación cita su fuente con la ruta ORIGINAL. Los ficheros en
> `docs/_fuentes/_extracted/*.txt` son solo el mirror de texto que se ha leído.

---

## 1. Intuición / resumen

Una **capa custom** en Keras es una capa que tú defines (subclase de
`keras.layers.Layer`) cuando necesitas una operación que las capas estándar
(densas, atención…) no hacen. El propio profesor lo resume así: *"podemos pensar
en aplicar una capa cuando quiero hacer operaciones que no he visto en las otras
capas. Una depuración clásica en datos es dividir, y no hemos visto ninguna capa
que haya hecho ninguna división"* (`clases-master/26 06 13.pdf`,
mirror `pdf_transcript_26-06-13.txt`, l. 6082-6090).

La idea de fondo del taller es **meter conocimiento de dominio dentro de la
arquitectura** en vez de esperar que la red lo aprenda sola. Dos motivos:

- **Operaciones que la red no tiene de fábrica.** Las densas solo combinan
  linealmente y aplican activaciones; no dividen. Si sabes que una característica
  útil es un *ratio* (una división entre variables financieras), puedes
  calcularla dentro de una capa y dársela hecha a la red
  (`clases-master/26 06 13.pdf`, l. 6096-6107). En teoría una red
  con suficientes neuronas y capas podría aproximar la división, pero "esos
  suficientes pueden ser [muchos]" — si tú ya sabes que el ratio importa, lo
  metes y te ahorras capacidad (misma fuente, l. 6096-6104).

- **Diseñar a medida funciona.** Como motivación general de "customizar"
  componentes (capas y costes) en lugar de usar lo genérico, la charla de
  métricas perceptuales bio-inspiradas muestra que un modelo diseñado con
  conocimiento del problema bate a métricas genéricas tipo MSE/SSIM con muchísimos
  menos parámetros (Bio model: 0.91 de correlación con 36 K parámetros frente a
  LPIPS 0.81 con 24.7 M) (`clases-master/Custom.pdf`, pág. 15). Esa charla cierra
  precisamente con una sección "CUSTOMIZING AI – FINANCES"
  (`clases-master/Custom.pdf`, pág. 27).

En el taller se sugiere explícitamente como ejemplo de capa la del **ratio de
endeudamiento**: *"una capa customizada que calcule el ratio de endeudamiento,
que haga cosas de división entre las variables financieras de entrada, para
sacar cuál es el porcentaje de endeudamiento de esa persona; depende de lo que
cobre y de lo que deba"* (`clases-master/26 06 18 MIAX 14.pdf`,
mirror `pdf_transcript_26-06-18.txt`, l. 2118-2128).

A esto se le añade una **saturación**: una operación no lineal que **comprime los
valores grandes y deja casi igual los pequeños**, para no gastar la misma
capacidad de la red en la cola de pocos clientes con valores enormes que en la
masa de clientes con valores pequeños (`clases-master/26 06 18 MIAX 14.pdf`,
l. 2329-2404).

---

## 2. Formulación matemática

### 2.1 El ratio (división entre variables financieras)

El material describe el ratio de forma conceptual, no con una fórmula cerrada.
Dos formulaciones aparecen en las fuentes:

- **Ratio de endeudamiento** = relación entre lo que la persona *debe/paga* y lo
  que *cobra* (`clases-master/26 06 18 MIAX 14.pdf`, l. 2124-2128). En notación,
  una forma natural sería:

  $$r = \frac{\text{deuda/pago periódico}}{\text{ingresos}}$$

  > ⚠️ La fórmula concreta (qué columnas exactas van en numerador y denominador)
  > **no** está en el material; es una propuesta razonable. Ver Huecos.

- **Ratio genérico ingreso/normalizador**, ejemplificado como ingreso dividido
  por el PIB o por la población del país: *"coja los ingresos y los divida por el
  PIB del país… eso va a ser el ratio de más o menos cuánto cobra una persona
  respecto a dónde vive"* (`clases-master/26 06 13.pdf`,
  l. 6042-6054).

También se planteó (y se **descartó**) una variante de "todos los ratios contra
todos": generar, además de las $d$ dimensiones de entrada, las divisiones de cada
dimensión entre las demás. Se descartó porque el número de combinaciones explota
($\sim d!$ / del orden de $10^{28}$ para 27 variables) y es inmanejable
(`clases-master/26 06 18 MIAX 14.pdf`, l. 2235-2296).

### 2.2 La saturación (vía exponente entrenable)

La saturación que se implementa en clase es **elevar cada dimensión a un
exponente** $p$ entrenable, uno por característica:

$$y_i = x_i^{\,p_i}$$

con el exponente **inicializado a 1** (para que al principio la capa "no haga
nada" y sea la identidad) y **restringido a un intervalo** para que el
entrenamiento no se desestabilice (`clases-master/26 06 18 MIAX 14.pdf`,
l. 2458-2486). El rango usado en clase fue:

$$p_i \in [0.1,\; 3]$$

(`clases-master/26 06 18 MIAX 14.pdf`, l. 2464-2540). La restricción se aplica
con un *clip* sobre el parámetro (misma fuente, l. 2646-2648).

Interpretación del exponente (`clases-master/26 06 18 MIAX 14.pdf`,
l. 2749-2757):

- $p < 1$ → **función saturante**: comprime los valores grandes (los acerca entre
  sí) y deja relativamente más rango a los pequeños.
- $p > 1$ → función **exponencial/parabólica**: el efecto contrario (expande).
- $p = 1$ → identidad (no toca los datos).

Motivo de dominio: las variables financieras (p. ej. ingresos anuales) están muy
sesgadas — mucha gente con ingreso bajo y muy poca con ingreso muy alto. Un
exponente $<1$ transforma esa distribución hacia algo más **uniforme**, dedicando
"recursos" de la red a la zona densa de clientes y no a la cola
(`clases-master/26 06 18 MIAX 14.pdf`, l. 2333-2404).

> Nota: el enunciado del taller habla de "saturación / restricción matemática".
> En el material la saturación concreta que se construye es el **exponente
> entrenable acotado**. Otras saturaciones clásicas (sigmoide, `clip`,
> normalización acotada) son coherentes con la idea pero **no** se desarrollan
> como tales en las fuentes (ver Huecos). El concepto de "Restrict max / Restrict
> min" aparece, pero en el contexto de displays de imagen, no de finanzas
> (`clases-master/Custom.pdf`, págs. 23-25).

---

## 3. Implementación (patrón de capa custom en Keras)

El patrón mínimo de una capa custom tiene **3 métodos imprescindibles**
(`__init__`, `build`, `call`) más uno opcional (`compute_output_shape`)
(`clases-master/Custom.pdf`, pág. 32; reiterado en
`clases-master/26 06 18 MIAX 14.pdf`, l. 2503-2510).

Patrón canónico de las diapositivas (`clases-master/Custom.pdf`, pág. 32):

```python
class Mi_capa(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # input_shape[-1] es la dimensión de la característica de entrada
        input_dim = input_shape[-1]
        # Creamos la matriz de pesos W
        self.w = self.add_weight(shape=(input_dim, self.units))
        super().build(input_shape)          # llamar al final

    def call(self, inputs):
        return keras.ops.matmul(inputs, self.w)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
```

Ejemplo equivalente del notebook de referencia, con pesos entrenables
(`kernel`, `bias`) creados en `build` y la operación en `call`
(`clases-master/Keras_custom_layer.ipynb`, mirror `nb_Keras_custom_layer.txt`,
celdas 3 y 7):

```python
from keras.layers import Layer

class MyLayer(Layer):
    def __init__(self, output_dim, activ, **kwargs):
        self.output_dim = output_dim
        self.activ = activ
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim,),
                                    initializer='uniform',
                                    trainable=True)
        super(MyLayer, self).build(input_shape)

    def call(self, x):
        if self.activ == 'sigmoid':
            return keras.ops.nn.sigmoid(keras.ops.matmul(x, self.kernel) + self.bias)
        else:
            return keras.ops.nn.relu(keras.ops.matmul(x, self.kernel) + self.bias)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
```

Y se inserta como una capa más en el modelo
(`clases-master/Keras_custom_layer.ipynb`, celda 5):

```python
model = Sequential()
model.add(Dense(64, input_shape=(100,)))
model.add(MyLayer(output_dim=10, activ='sigmoid'))
model.add(Dense(64, activation='relu'))
...
```

Detalles de implementación que recalca el profesor para la capa de saturación:

- **Usar solo `keras.ops`** (p. ej. `keras.ops.power`, `keras.ops.clip`) en lugar
  de funciones específicas de un backend, para que el código sirva igual con
  TensorFlow, PyTorch o JAX por debajo (`clases-master/26 06 18 MIAX 14.pdf`,
  l. 2448-2455; `clases-master/Custom.pdf`, pág. 28 — "En Keras usar: keras.ops").
- Para la capa de exponentes, la **salida tiene el mismo tamaño que la entrada**
  (no hay reducción), así que `compute_output_shape` devuelve la forma de entrada
  (`clases-master/26 06 18 MIAX 14.pdf`, l. 2507-2510).
- Los **parámetros entrenables** son los exponentes: un vector de longitud igual
  al nº de dimensiones de entrada, inicializado a unos, con clip a $[0.1, 3]$
  (`clases-master/26 06 18 MIAX 14.pdf`, l. 2530-2544).
- En `call`, la operación es elevar las entradas a esos exponentes (power /
  elevado) (`clases-master/26 06 18 MIAX 14.pdf`, l. 2543-2547).

> No reimplementar aquí la capa del taller: estos bloques son solo el **patrón**.

---

## 4. Conexión con el Taller B4-T1 (Home Credit)

Cómo aterriza todo lo anterior en el clasificador de concesión de crédito:

- **Qué variables forman el ratio.** El ratio de endeudamiento relaciona "lo que
  cobra" con "lo que debe/paga" (`clases-master/26 06 18 MIAX 14.pdf`,
  l. 2124-2128). En el dataset Home Credit, los candidatos naturales son la
  **anualidad del préstamo** (`AMT_ANNUITY`) y los **ingresos**
  (`AMT_INCOME_TOTAL`) — p. ej. `AMT_ANNUITY / AMT_INCOME_TOTAL` como carga de la
  cuota sobre el ingreso.
  > ⚠️ Esta correspondencia de columnas concretas es **propuesta nuestra**: el
  > material no nombra las columnas exactas (el profesor dice incluso "no recuerdo
  > exactamente cuáles son los valores del taller",
  > `clases-master/26 06 13.pdf`, l. 6030-6033). Ver Huecos.

- **Por qué una capa y no calcularlo a mano.** El ratio es una **división**, que
  las densas no hacen; meterlo como capa le da a la red una característica que de
  otro modo le costaría aproximar (`clases-master/26 06 13.pdf`,
  l. 6061-6107).

- **Dónde colocar la saturación.** La saturación (exponente $<1$) tiene sentido
  sobre **variables financieras muy sesgadas** (ingresos, anualidades) para
  uniformizar su distribución antes de las densas
  (`clases-master/26 06 18 MIAX 14.pdf`, l. 2333-2404). En clase se probó la capa
  de exponentes **al principio** (sobre los datos de entrada) y se sugirió probar
  también **después** de una densa (`clases-master/26 06 18 MIAX 14.pdf`,
  l. 2576-2602, 2589-2592).

- **Matiz sobre la posición.** El profesor advierte que el exponente como
  **primera** capa "no tiene diferencia" si va sobre el ratio ya calculado, y que
  cobra más sentido como **capa interna**, después de que la red haya hecho
  operaciones (`clases-master/26 06 18 MIAX 14.pdf`, l. 2423-2442). Es decir: la
  saturación sobre inputs crudos sesgados sí ayuda (uniformizar), pero sobre una
  cantidad ya acotada puede aportar poco. Hay que decidir con criterio.

- **Encaje en la arquitectura.** Esquema coherente con el material: entrada →
  (capa custom que calcula ratio(s) y/o aplica saturación con exponente acotado)
  → capas `Dense` → salida de clasificación. La capa custom va **antes** de las
  densas, como en el patrón `model.add(...)` del notebook
  (`clases-master/Keras_custom_layer.ipynb`, celda 5).

- **Estabilidad numérica.** Acotar el exponente (clip $[0.1, 3]$) evita que los
  datos "se vayan" / "descuadren" durante el entrenamiento
  (`clases-master/26 06 18 MIAX 14.pdf`, l. 2464-2476). Cuidado además con
  **divisiones por cero / valores ausentes**: las EXT_SOURCE y otras vienen
  imputadas; el material no trata este punto (ver Huecos).

---

## 5. Huecos / decisiones pendientes

Lo que el material **no** fija y hay que decidir con criterio propio:

1. **Fórmula exacta del ratio de endeudamiento.** Las fuentes dan la idea
   ("lo que cobra / lo que debe", ingreso/PIB) pero **no** las columnas concretas
   de Home Credit ni la expresión cerrada. La asignación
   `AMT_ANNUITY / AMT_INCOME_TOTAL` (u otra) es propuesta, no material de clase
   (`clases-master/26 06 18 MIAX 14.pdf`, l. 2124-2128;
   `clases-master/26 06 13.pdf`, l. 6030-6062).

2. **Qué saturación usar.** En clase la saturación se implementa como **exponente
   entrenable acotado** ($x^p$, $p\in[0.1,3]$, init 1). El enunciado menciona
   "saturación/restricción" en general; **sigmoide, `clip` directo o
   normalización acotada** son alternativas plausibles pero **no** desarrolladas
   en las fuentes. Decidir cuál se usa.

3. **Valores de corte concretos.** El rango $[0.1, 3]$ y la inicialización a 1
   vienen de clase, pero el profesor los pone "por ejemplo" / sobre la marcha
   (`clases-master/26 06 18 MIAX 14.pdf`, l. 2464-2470); pueden ajustarse.

4. **Posición óptima de la capa.** Antes vs. después de las densas no queda
   resuelto: el profesor lo deja como algo a **probar** y matiza que el exponente
   como primera capa puede no aportar (`clases-master/26 06 18 MIAX 14.pdf`,
   l. 2423-2442, 2589-2592).

5. **Manejo de divisiones por cero / nulos** al calcular el ratio (epsilon en el
   denominador, recorte de outliers). No tratado en las fuentes, pese a que el
   dataset tiene valores imputados.

6. **Salida de la capa de ratio.** Si se devuelven las dimensiones originales
   *más* los ratios, o solo los ratios. Se planteó la versión "todas con todas" y
   se descartó por explosión combinatoria, pero no se cierra qué subconjunto de
   ratios conservar (`clases-master/26 06 18 MIAX 14.pdf`, l. 2229-2296).

7. **Transcripciones ruidosas.** Las transcripciones (`26 06 18`, `26 06 13`)
   son subtítulos automáticos con errores; las citas se han verificado por
   contexto, pero conviene contrastar con el notebook/PDF originales antes de
   fijar nada en el entregable. Las transcripciones `26 06 12` y `26 06 13` hablan
   sobre todo de **Sharpe ratio** (otra tarea/coste), no del ratio de
   endeudamiento; la única parte útil para esta tarea es el pasaje de la capa de
   división en `26 06 13` (l. 6020-6107).
