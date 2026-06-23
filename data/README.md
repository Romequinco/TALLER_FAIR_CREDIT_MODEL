# `data/` — Dataset del taller (Home Credit Default Risk)

Esta carpeta contiene el dataset del taller **solo en local**. El CSV **no se versiona**
(pesa ~158 MB, es de Kaggle y está cubierto por `.gitignore`). Solo se versionan este
`README.md` y el `.gitkeep`.

## Qué dataset es

**Home Credit Default Risk** — predicción de impago de crédito en perfiles con poco
historial crediticio.

- Competición / datos en Kaggle: <https://www.kaggle.com/competitions/home-credit-default-risk/overview>
- Fichero que usamos: **`application_train.csv`**.

## Cómo obtenerlo (cada miembro del grupo)

1. Descarga `application_train.csv` desde Kaggle (link de arriba; requiere cuenta y
   aceptar las reglas de la competición).
2. Colócalo **directamente** en esta carpeta:

   ```
   data/application_train.csv
   ```

3. **No lo subas al repo.** Ya está ignorado por `.gitignore` (`data/*` + `*.csv`);
   no fuerces su inclusión. Cada persona lo descarga por su cuenta.

> Estructura esperada en local (lo marcado con `(ignorado)` no se versiona):
>
> ```
> data/
> ├── README.md                 # versionado
> ├── .gitkeep                  # versionado
> └── application_train.csv     # (ignorado) — lo pones tú
> ```

## Columnas que usa el taller

Según nuestra documentación (`docs/teoria/`) y el enunciado:

| Rol | Columna(s) | Notas |
| --- | --- | --- |
| **Objetivo** | `TARGET` | Binaria: `1` = dificultades de pago, `0` = pagó a tiempo. |
| **Sensible** | `CODE_GENDER` | Género del solicitante. El modelo **no debe discriminar** por esta variable (Tarea 2, FAIR loss). |
| **Entradas** | Ingresos (p. ej. `AMT_INCOME_TOTAL`), anualidades (p. ej. `AMT_ANNUITY`) | Base del "Ratio de Endeudamiento" de la Tarea 1. |
| **Entradas** | `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` | Puntuaciones de fuentes externas; tienen **valores ausentes imputados**, clave para el estudio de incertidumbre (Tarea 4). |

> Los nombres exactos de las columnas de ingresos/anualidad (`AMT_INCOME_TOTAL`,
> `AMT_ANNUITY`) son del esquema estándar de Home Credit; confírmalos al cargar el CSV.

## Notebook de lectura / preprocesado

El notebook esqueleto que da el profesor para la lectura inicial está en:

```
docs/_fuentes/clases-master/Lectura_datos_Taller_B4_T1.ipynb
```

⚠️ Esa carpeta (`docs/_fuentes/`) está **ignorada por git** (material de clase, no se
versiona). La implementación ya se hizo: a partir de ese esqueleto se construyó el pipeline
versionado en `notebooks/` (01–07), que es desde donde se trabaja.

## ¿Descarga automatizada desde Kaggle?

Revisado `docs/_fuentes/profe-lectures/scripts/download_credit_data.py`: **no es
reutilizable** para nuestro caso. Pese al nombre, ese script descarga un dataset
distinto — el **US Census ACS Income** (vía el paquete `folktables`, tarea ACSIncome;
objetivo `income_high`, atributo protegido `sex`), que es el dataset de *fairness* del
curso del profesor, **no** el Home Credit de Kaggle. No sirve para bajar
`application_train.csv`.

Para automatizar la descarga de Home Credit habría que usar la **Kaggle API**
(`pip install kaggle`, credenciales en `~/.kaggle/kaggle.json`), p. ej.:

```bash
kaggle competitions download -c home-credit-default-risk -f application_train.csv -p data/
```

(Requiere haber aceptado las reglas de la competición en la web de Kaggle.) Esta receta
es orientativa; la vía manual de arriba es suficiente.
