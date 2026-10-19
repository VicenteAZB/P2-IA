# Detección de fraude — INFO1185, Problema 2

Comparación de kNN, árbol de decisión, SVM y Random Forest propio, con y sin SMOTE.
El criterio principal es maximizar la sensibilidad del fraude; en empate, su precisión.

## Preparación

Ejecución comprobada con **Python 3.14.8** en Linux. Versiones principales:
NumPy 2.5.4, pandas 3.0.6, SciPy 1.18.1, scikit-learn 1.9.1,
imbalanced-learn 0.14.2, Matplotlib 3.11.2 y joblib 1.6.0.
Las dependencias fijadas están en `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Descargar [Credit Card Fraud Detection de Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
y colocar **`creditcard.csv` junto a los archivos Python**. No se incluye el dataset.

## Ejecución

Desde la raíz del proyecto, con el entorno activado:

```bash
python exploracion.py
python busqueda.py
python evaluacion.py
```

La evaluación requiere `resultados_busqueda/busqueda.joblib`, generado por la búsqueda.
La ejecución completa de referencia tardó unos **16 minutos**; depende del equipo.
La búsqueda usa varios núcleos y puede consumir bastante memoria.

| Archivo | Función |
|---|---|
| `datos.py` | Carga, limpieza, submuestra y partición común. |
| `exploracion.py` | Conteos y gráficos descriptivos. |
| `random_forest_propio.py` | Bootstrap, entrenamiento de árboles y votación. |
| `pipelines.py` | Escalado, filtro KS, SMOTE opcional y clasificador. |
| `busqueda.py` | Grid Search con cinco folds estratificados. |
| `evaluacion.py` | Métricas de validación y prueba, tablas y figuras. |

## Reproducibilidad y metodología

- **Semilla general: `5040`**, utilizada en la submuestra, partición, CV, SMOTE y bootstrap de evaluación.
  El bosque deriva de ella una semilla por árbol.
- Se eliminan duplicados exactos y se toman **20.000 filas estratificadas**.
- Partición fija y estratificada: **70 % entrenamiento / 30 % prueba**.
  `Class` es el objetivo, no una variable de entrada; fraude (`1`) es la clase positiva.
- Dentro de cada fold se ajustan escalado, selección KS (`p < 0,05`) y SMOTE, si corresponde.
  Validación y prueba no se balancean. No se aplica un nuevo PCA.
- Los mejores hiperparámetros de cada una de las ocho condiciones se eligen solo por CV.
  Después se reajusta cada configuración con todo el entrenamiento y se evalúa en prueba.
- Si cambia la muestra, semilla, pipeline o grilla, volver a ejecutar búsqueda y evaluación;
  no reutilizar un archivo de búsqueda anterior.

## Resultados generados

- `resultados_exploracion/`: 12 figuras de exploración.
- `resultados_busqueda/`: resumen CSV y resultados serializados de búsqueda.
- `resultados_evaluacion/`: `validacion_cv.csv`, `prueba.csv`,
  `caracteristicas_seleccionadas.csv` y nueve figuras.

Los resultados se presentan por separado:
- **Validación cruzada:** media y desviación estándar de cada métrica entre los cinco folds.
- **Prueba:** cada métrica se calcula sobre el conjunto reservado, después de reajustar
  el modelo con todo el entrenamiento. Como complemento, se incluye una desviación
  estimada mediante 1.000 remuestreos bootstrap de las predicciones, sin reentrenar.

La muestra de referencia contiene 23 fraudes en entrenamiento y 10 en prueba:
un fraude en prueba equivale a 10 puntos porcentuales de sensibilidad. Los resultados
deben interpretarse con esa limitación, sin hacer nuevos ajustes a partir de prueba.
