"""
busqueda.py — Búsqueda de hiperparámetros con validación cruzada estratificada (5 folds).

Usa solo el conjunto de ENTRENAMIENTO. Criterio: maximizar la sensibilidad media de la clase
fraude (= minimizar la tasa media de falsos negativos); en empate, mayor precisión de fraude.

Las mismas particiones de validación se usan para todos los modelos y condiciones
(StratifiedKFold con semilla fija).

Scaler, selector KS y SMOTE se cachean en una carpeta temporal compartida por los 8 pipelines,
de modo que se ajustan una sola vez por fold y no una vez por combinación de hiperparámetros.

Ejecutar:
    python busqueda.py

Guarda en resultados_busqueda/: busqueda.joblib (lo lee la evaluación final) y
busqueda_resumen.csv.
"""
import shutil
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from datos import SEMILLA, cargar_particion
from pipelines import ALFA_KS, MODELOS, clave, etiqueta, make_pipeline

CARPETA = Path(__file__).parent / "resultados_busqueda"
CARPETA.mkdir(exist_ok=True)

N_FOLDS = 5

# Sensibilidad de fraude (criterio principal) y precisión de fraude (desempate)
SCORING = {"recall": "recall",
           "precision": make_scorer(precision_score, zero_division=0)}

# Profundidades compartidas por el árbol y el RF propio, para que sean comparables
PROFUNDIDADES = [3, 5, 8, 12, None]

# Espacios de búsqueda acotados (prefijo clf__ = nombre del paso en el Pipeline).
# Mismo espacio con y sin SMOTE, para que la diferencia entre condiciones venga solo de SMOTE.
GRILLAS = {
    "knn": {"clf__n_neighbors": [1, 3, 5, 11, 21], "clf__weights": ["uniform", "distance"]},
    "arbol": {"clf__max_depth": PROFUNDIDADES},
    "svm": {"clf__C": [0.1, 1, 10, 100], "clf__gamma": [0.01, 0.1]},
    "rf_propio": {"clf__profundidad_maxima": PROFUNDIDADES,
                  "clf__max_features": ["sqrt", 0.3, 0.5]},
}


def crear_cv():
    """Mismas particiones para todo: misma semilla -> mismos folds."""
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA)


def _mejor_indice(cv_results):
    """Mayor sensibilidad media; en empate, mayor precisión media (el redondeo evita ruido float)."""
    r = np.round(cv_results["mean_test_recall"], 10)
    p = np.round(cv_results["mean_test_precision"], 10)
    return int(np.lexsort((-p, -r))[0])          # la última clave (-r) es la principal


def buscar(pipe, grilla, X, y, n_jobs):
    """Grid Search con CV; reajusta con todo el entrenamiento la mejor combinación."""
    gs = GridSearchCV(pipe, grilla, scoring=SCORING, refit=_mejor_indice, cv=crear_cv(),
                      n_jobs=n_jobs, error_score="raise")
    gs.fit(X, y)
    res = pd.DataFrame(gs.cv_results_)
    fila = res.loc[gs.best_index_]
    soporte = gs.best_estimator_.named_steps["selector"].get_support()
    return {
        "params": gs.best_params_,
        "recall_media": float(fila["mean_test_recall"]),
        "recall_std": float(fila["std_test_recall"]),
        "precision_media": float(fila["mean_test_precision"]),
        "precision_std": float(fila["std_test_precision"]),
        "variables": list(X.columns[soporte]),
        "n_combinaciones": len(res),
        "cv_results": res[[c for c in res.columns if c.startswith(("param_", "mean_test", "std_test"))]],
    }


def main():
    X_ent, _, y_ent, _ = cargar_particion()      # la prueba no se usa en este módulo
    configuraciones = [(m, s) for m in MODELOS for s in (False, True)]
    print("BÚSQUEDA DE HIPERPARÁMETROS (validación cruzada estratificada)")
    print(f"Datos de entrenamiento : {X_ent.shape[0]} filas, {X_ent.shape[1]} variables, "
          f"{int(y_ent.sum())} fraudes")
    print(f"Folds                  : {N_FOLDS}")
    print("Criterio de selección  : mayor sensibilidad media de fraude; "
          "desempate por precisión de fraude")
    print(f"Selección de variables : filtro KS (p < {ALFA_KS}), ajustado dentro de cada fold")

    cache = tempfile.mkdtemp(prefix="cache_busqueda_")
    resultados = {"_meta": {"n_folds": N_FOLDS, "alfa_ks": ALFA_KS}}
    try:
        for i, (modelo, smote) in enumerate(configuraciones, start=1):
            nombre = etiqueta(modelo, smote)
            n_jobs = 1 if modelo == "rf_propio" else -1   # el RF propio ya paraleliza por dentro
            print(f"\n{'=' * 70}\n[{i}/{len(configuraciones)}] {nombre}")
            t0 = time.time()
            pipe = make_pipeline(modelo, smote, memory=cache)
            res = buscar(pipe, GRILLAS[modelo], X_ent, y_ent, n_jobs)
            res["etiqueta"] = nombre
            resultados[clave(modelo, smote)] = res
            print(f"  Combinaciones evaluadas : {res['n_combinaciones']}")
            print(f"  Mejores hiperparámetros : {res['params']}")
            print(f"  Sensibilidad (CV)       : {res['recall_media']:.3f} ± {res['recall_std']:.3f}")
            print(f"  Precisión (CV)          : {res['precision_media']:.3f} ± {res['precision_std']:.3f}")
            print(f"  Variables elegidas KS   : {len(res['variables'])} -> {', '.join(res['variables'])}")
            print(f"  Tiempo                  : {time.time() - t0:.0f} s")
            joblib.dump(resultados, CARPETA / "busqueda.joblib")
    finally:
        shutil.rmtree(cache, ignore_errors=True)

    filas = []
    for k, r in resultados.items():
        if k == "_meta":
            continue
        filas.append({
            "configuracion": r["etiqueta"],
            "sensibilidad_media": r["recall_media"], "sensibilidad_std": r["recall_std"],
            "precision_media": r["precision_media"], "precision_std": r["precision_std"],
            "n_variables": len(r["variables"]),
            "hiperparametros": r["params"],
            "variables": ", ".join(r["variables"]),
        })
    resumen = pd.DataFrame(filas)
    resumen.to_csv(CARPETA / "busqueda_resumen.csv", index=False)

    print(f"\n{'=' * 70}\nRESUMEN FINAL (validación cruzada, solo entrenamiento)")
    print(resumen[["configuracion", "sensibilidad_media", "sensibilidad_std",
                   "precision_media", "precision_std", "n_variables"]]
          .round(3).to_string(index=False))
    print("\nHiperparámetros seleccionados:")
    for fila in filas:
        print(f"  {fila['configuracion']:22s} {fila['hiperparametros']}")
    print("\nArchivos guardados: resultados_busqueda/busqueda.joblib y "
          "resultados_busqueda/busqueda_resumen.csv")


if __name__ == "__main__":
    main()