# 05 — Contexto: robustez y confiabilidad de redes

> **Aviso.** Este documento es **contexto tangencial**, no una de las 4 tareas del taller "Diseño de Redes Confiables (Justicia e Incertidumbre)". Recoge material de clase sobre **robustez**, **regularización** y **ejemplos adversariales** que aparece en las fuentes pero que **no corresponde** a ninguna tarea (capa custom, FAIR loss, Keras Tuner, incertidumbre). Se conserva porque conecta con el lema general de "redes confiables".

---

## 1. Qué es este material y por qué lo guardamos

El hilo conductor del taller es construir redes *confiables*. La justicia (fairness) y la incertidumbre son dos dimensiones de esa confiabilidad, y son las que abordan las 4 tareas. Pero existe una tercera dimensión clásica que el material de clase también toca: la **robustez**, entendida como la capacidad de una red de no degradarse ante perturbaciones de la entrada (ruido, transformaciones, ataques deliberados).

Las técnicas que aparecen en estas fuentes son:

- **Regularización** en sentido amplio: restricciones sobre arquitectura, entrenamiento, función de coste y datos para evitar el sobreajuste y forzar comportamientos esperados (`clases-master/2026_Robustez_de_redes.pdf`).
- **Ejemplos adversariales** y el método **FGSM** (Fast Gradient Sign Method): perturbaciones mínimas, calculadas con el gradiente de la pérdida respecto a la *entrada*, que engañan a un modelo ya entrenado (`clases-master/Adversarial_examples.ipynb`).
- **Data augmentation**: ampliar el conjunto de entrenamiento con copias ruidosas/transformadas para mejorar la generalización (`clases-master/Copia de CNN_SM_1D_3ch_data_augmentation.ipynb`).

Conviene tenerlo recogido porque, conceptualmente, robustez y regularización son parte del mismo mapa de "qué hace que una red sea fiable". Pero —insistimos— **no es una tarea del taller**.

---

## 2. Resumen por fuente

### `clases-master/Adversarial_examples.ipynb`
Tutorial centrado en el **FGSM** aplicado a imágenes, sobre un `MobileNetV2` preentrenado en ImageNet. La idea clave: un ejemplo adversarial se construye con `adv_x = x + ε·sign(∇ₓ J(θ, x, y))`, es decir, moviendo la imagen en la dirección que *maximiza* la pérdida, tomando el gradiente respecto a la entrada (no a los pesos, que quedan fijos). El notebook muestra cómo, al subir `ε`, es cada vez más fácil engañar a la red a costa de que la perturbación se vuelva visible. Incluye además una variante con optimizador Adam que parte de **ruido puro** y lo "esculpe" hasta que el modelo lo clasifica como una clase objetivo. Aporta la intuición central de la vulnerabilidad adversarial.

### `clases-master/CNN_SM_1D_3ch_adversarial_attack.ipynb`
Traslada el ataque adversarial al dominio de **series temporales financieras** (precios de apertura de GOOGL, AA, IAE). Construye una CNN 1D de 3 canales (ventanas de 14 días para predecir el día 15) con una `Conv1D` + `Dense`, y luego genera perturbaciones con el gradiente de la predicción respecto a la entrada. Compara el error del modelo sobre datos limpios frente a datos perturbados (`model.evaluate`), e incluso reentrena un segundo modelo sobre los datos atacados (esbozo de *entrenamiento adversarial* como defensa). Aporta que los ataques no son exclusivos de visión: también afectan a predicción financiera.

### `clases-master/Copia de CNN_SM_1D_3ch_data_augmentation.ipynb`
Misma CNN 1D de 3 canales y mismos datos financieros, pero aquí la técnica es **data augmentation por inyección de ruido**: replica el conjunto de entrenamiento ~10 veces añadiendo ruido gaussiano pequeño (`0.01·randn`) a cada copia. Entrena un modelo base y otro sobre los datos aumentados, y compara errores. Aporta el otro lado de la robustez: en vez de atacar, se *fortalece* la red exponiéndola a variaciones controladas de la entrada.

### `clases-master/2026_Robustez_de_redes.pdf`
PDF de **diapositivas con muy poco texto extraíble** (ver Huecos). De lo poco legible se obtiene un índice conceptual: **tipos de regularización** organizados en cuatro ejes —Modelo/Arquitectura (tamaño, pesos), Entrenamiento (Dropout, Batch Normalization, Transfer learning), Función de coste (forzar información a priori, penalizar lo inesperado, desviaciones respecto a un modelo físico, *fair learning*) y Datos—, además de secciones sobre **Ruido**, **Transformaciones naturales**, **Data Augmentation** y **Adversarial Examples** (con enlaces externos: FGSM de TensorFlow, ataques de audio, one-pixel attack, RL). Aporta el marco general que ordena las otras tres fuentes.

---

## 3. Relación con el taller y huecos

**Por qué NO es ninguna de las 4 tareas.** Las tareas del taller son: (1) capa custom, (2) FAIR loss, (3) Keras Tuner y (4) incertidumbre. Este material no implementa ninguna de ellas: no define capas personalizadas para las tareas, no construye la función de pérdida de justicia del taller, no usa búsqueda de hiperparámetros y no cuantifica incertidumbre. Trata una dimensión distinta (robustez ante perturbaciones/ataques) con ejemplos propios (ImágenesNet, series financieras).

**Puentes débiles (conexiones, no equivalencias):**
- *Robustez como tercera dimensión de confiabilidad.* Bajo el lema "redes confiables", la robustez ante ruido y ataques convive con la justicia y la incertidumbre; es contexto que enriquece el "para qué" del taller.
- *Regularización vía función de coste ↔ FAIR loss.* La diapositiva de robustez lista explícitamente, dentro de "Función de coste", la opción de penalizar lo no deseado y menciona "fair learning". Eso conecta de forma natural con la tarea de FAIR loss: ambas modifican el objetivo de entrenamiento para imponer una propiedad deseada. Es el puente más sólido entre este material y el taller.
- *Regularización ↔ capa custom.* La regularización por arquitectura/entrenamiento es el telón de fondo conceptual de por qué se diseñan capas a medida, pero la relación es genérica.

**Huecos y limitaciones:**
- **`clases-master/2026_Robustez_de_redes.pdf` casi no tiene texto extraíble (~1 KB).** Son diapositivas con títulos sueltos e imágenes; el contenido real (gráficos, ejemplos visuales, explicaciones orales) **no está disponible** en el texto. Lo recogido en la sección 2 es solo el esqueleto de títulos y el índice de regularización; cualquier desarrollo más fino sería especulación.
- Ninguna fuente formaliza una **defensa adversarial completa** ni evalúa robustez de forma sistemática; el reentrenamiento sobre datos atacados aparece solo esbozado.
- No hay material que ligue explícitamente estas técnicas con incertidumbre cuantificada; la conexión robustez↔incertidumbre queda como intuición, no documentada en las fuentes.
