"""
evaluacion.py — Evaluación final de las 8 configuraciones (4 modelos, sin SMOTE y con SMOTE).

1. VALIDACIÓN: con los mejores hiperparámetros de busqueda.py, calcula las 6 métricas en los
   5 folds (media ± desv. estándar) y la matriz de confusión sumada de los 5 folds.
   Solo usa el conjunto de entrenamiento.
2. PRUEBA: reentrena cada configuración con TODO el entrenamiento y la evalúa una vez en el
   conjunto de prueba. La desviación estándar se obtiene por bootstrap del conjunto de prueba
   (se remuestrean las predicciones; el modelo no se reajusta).
3. Registra las variables conservadas por el filtro KS (el filtro no depende del clasificador,
   por lo que se informa una sola vez), su estabilidad entre folds y los conteos de clase
   antes/después de SMOTE.

El conjunto de prueba no se usa para elegir nada. Validación y prueba se guardan por separado.

Ejecutar:
    python evaluacion.py        (requiere haber corrido busqueda.py)

Guarda en resultados_evaluacion/: validacion_cv.csv, prueba.csv, caracteristicas_seleccionadas.csv
y las figuras 01 a 09.
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import (confusion_matrix, f1_score, make_scorer, precision_score,
                             recall_score)
from sklearn.model_selection import cross_validate

from busqueda import N_FOLDS, crear_cv
from datos import SEMILLA, cargar_particion
from pipelines import MODELOS, NOMBRES, SelectorKS, clave, etiqueta, make_pipeline

RUTA_BUSQUEDA = Path(__file__).parent / "resultados_busqueda" / "busqueda.joblib"
CARPETA = Path(__file__).parent / "resultados_evaluacion"
CARPETA.mkdir(exist_ok=True)

VERDE = "#2e9e4f"
ROJO = "#d64545"
COLOR_SIN = "#2E75B6"
COLOR_SMOTE = "#E67E22"
COLOR_PRUEBA = "#C0392B"
COLOR_NEUTRO = "#8FA9C4"
COLOR_DESCARTADA = "#D0D0D0"

N_BOOTSTRAP = 1000
METRICAS = ["sensibilidad", "especificidad", "precision", "f1", "exactitud", "fnr"]

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})


def _celda(y_real, y_pred, i):
    """Celda i de la matriz de confusión aplanada (0=TN, 1=FP, 2=FN, 3=TP)."""
    return confusion_matrix(y_real, y_pred, labels=[0, 1]).ravel()[i]


SCORING_CV = {
    "sensibilidad": "recall",
    "especificidad": make_scorer(recall_score, pos_label=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "exactitud": "accuracy",
    "tn": make_scorer(_celda, i=0),
    "fp": make_scorer(_celda, i=1),
    "fn": make_scorer(_celda, i=2),
    "tp": make_scorer(_celda, i=3),
}


# ----------------------------------------------------------------- utilidades
def f(valor, d=3):
    """Número con coma decimal."""
    return f"{valor:.{d}f}".replace(".", ",")


def guardar(fig, nombre):
    fig.savefig(CARPETA / nombre, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   -> figura guardada: resultados_evaluacion/{nombre}")


def metricas_de_conteos(tp, fn, fp, tn):
    """Las 6 métricas a partir de la matriz de confusión (clase positiva = fraude)."""
    def div(a, b):
        return a / b if b > 0 else 0.0
    sens = div(tp, tp + fn)
    prec = div(tp, tp + fp)
    return {"sensibilidad": sens, "especificidad": div(tn, tn + fp), "precision": prec,
            "f1": div(2 * prec * sens, prec + sens),
            "exactitud": div(tp + tn, tp + tn + fp + fn), "fnr": 1 - sens}


def bootstrap_prueba(y_real, y_pred):
    """Desviación estándar de las métricas al remuestrear el conjunto de prueba."""
    rng = np.random.default_rng(SEMILLA)
    y_real, y_pred = np.asarray(y_real), np.asarray(y_pred)
    filas = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y_real), len(y_real))
        yr, yp = y_real[idx], y_pred[idx]
        tp = int(((yr == 1) & (yp == 1)).sum())
        fn = int(((yr == 1) & (yp == 0)).sum())
        fp = int(((yr == 0) & (yp == 1)).sum())
        tn = int(((yr == 0) & (yp == 0)).sum())
        filas.append(metricas_de_conteos(tp, fn, fp, tn))
    return pd.DataFrame(filas).std(ddof=1)


def conteos_smote(pipe, X, y):
    """Ejemplos por clase (legítimos/fraudes) antes y después de SMOTE, con el pipeline ya ajustado."""
    Xt = pipe[:2].transform(X)                       # scaler + selector
    _, y_nuevo = pipe.named_steps["sampler"].fit_resample(Xt, y)
    antes, despues = np.bincount(y), np.bincount(y_nuevo)
    return f"{antes[0]}/{antes[1]} -> {despues[0]}/{despues[1]}"


def folds_que_conservan(X, y):
    """En cuántos folds de CV se conserva cada variable (el filtro se ajusta solo con el
    entrenamiento de cada fold)."""
    cuenta = np.zeros(X.shape[1], dtype=int)
    for idx_ent, _ in crear_cv().split(X, y):
        cuenta += SelectorKS().fit(X.iloc[idx_ent], y.iloc[idx_ent]).get_support()
    return cuenta


def tabla_formateada(df, media, columnas_extra=()):
    """Tabla 'valor ± desv.' por métrica. media(m) devuelve la columna del valor de la métrica m."""
    tabla = {"configuración": df["configuracion"]}
    for m in METRICAS:
        tabla[m] = [f"{f(a)} ± {f(b)}" for a, b in zip(media(m), df[f"{m}_std"])]
    for c in columnas_extra:
        tabla[c] = df[c]
    return pd.DataFrame(tabla)


def colores_config(nombres):
    return [COLOR_SMOTE if "SMOTE" in n else COLOR_SIN for n in nombres]


# -------------------------------------------------------------------- figuras
def figura_matrices(matrices, titulo, nombre):
    """Matrices de confusión con el fraude (clase positiva) primero: [[TP, FN], [FP, TN]].
    Una columna por modelo, arriba sin SMOTE y abajo con SMOTE."""

    # Color por casilla: filas = real, columnas = predicho
    colores = [[VERDE, ROJO],   # Fraude real:    Fraude pred. (TP) / Legítimo pred. (FN)
               [ROJO, VERDE]]   # Legítimo real:  Fraude pred. (FP) / Legítimo pred. (TN)

    fig, ejes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, (config, m) in zip(ejes.T.ravel(), matrices.items()):
        m = m[::-1, ::-1]

        # Fondo blanco, solo para fijar los límites del eje
        ax.imshow(np.zeros(m.shape), cmap="gray_r", vmin=0, vmax=1)
        ax.grid(False)

        # Cada casilla con su color
        for i in range(2):
            for j in range(2):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor=colores[i][j],
                                       edgecolor="white", linewidth=2, zorder=1))

        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, f"{int(v):,}".replace(",", "."), ha="center", va="center",
                    color="white", fontsize=11, fontweight="bold", zorder=3)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Fraude", "Legítimo"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Fraude", "Legítimo"])
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_title(config)

    fig.suptitle(titulo, fontweight="bold")
    fig.tight_layout()
    guardar(fig, nombre)

def figura_metricas(configs, valores, titulo, nombre):
    """Mapa de calor configuración x métrica. La tasa de falsos negativos se colorea invertida
    (menor = mejor) para que el color más intenso signifique siempre mejor desempeño."""
    valores = np.asarray(valores)
    color = valores.copy()
    color[:, METRICAS.index("fnr")] = 1 - color[:, METRICAS.index("fnr")]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.imshow(color, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(valores):
        ax.text(j, i, f(v), ha="center", va="center",
                color="white" if color[i, j] > 0.6 else "black", fontweight="bold")
    ax.set_xticks(range(len(METRICAS)))
    ax.set_xticklabels(["Sensibilidad", "Especificidad", "Precisión", "F1", "Exactitud",
                        "Tasa de falsos\nnegativos"])
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs)
    ax.set_title(titulo)
    fig.tight_layout()
    guardar(fig, nombre)


def figura_sensibilidad(validacion, prueba):
    """Sensibilidad en validación cruzada y en prueba, lado a lado, con su desviación estándar."""
    configs = list(validacion["configuracion"])
    x = np.arange(len(configs))
    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - 0.2, validacion["sensibilidad_media"], 0.4, color=COLOR_NEUTRO,
                yerr=validacion["sensibilidad_std"], capsize=3,
                label="Validación cruzada (± desv. entre folds)")
    b2 = ax.bar(x + 0.2, prueba["sensibilidad"], 0.4, color=COLOR_PRUEBA,
                yerr=prueba["sensibilidad_std"], capsize=3, label="Prueba (± desv. bootstrap)")
    for barras in (b1, b2):
        ax.bar_label(barras, labels=[f(v, 2) for v in barras.datavalues], padding=14, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right")
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Sensibilidad (fraudes detectados)")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.set_title("Sensibilidad de las 8 configuraciones: validación cruzada vs prueba")
    fig.tight_layout()
    guardar(fig, "05_sensibilidad_validacion_vs_prueba.png")


def figura_fn_fp(prueba):
    """Fraudes no detectados (FN) y falsas alarmas (FP) en prueba."""
    configs = list(prueba["configuracion"])
    colores = colores_config(configs)
    y = np.arange(len(configs))
    fig, ejes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, col, titulo in [(ejes[0], "FN", "Fraudes no detectados (falsos negativos)"),
                            (ejes[1], "FP", "Falsas alarmas (falsos positivos)")]:
        barras = ax.barh(y, prueba[col], color=colores)
        ax.bar_label(barras, padding=3)
        ax.set_xlim(0, prueba[col].max() * 1.15 + 1)
        ax.set_title(titulo)
    ejes[0].set_yticks(y)
    ejes[0].set_yticklabels(configs)
    ejes[0].invert_yaxis()
    ejes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLOR_SIN),
                            plt.Rectangle((0, 0), 1, 1, color=COLOR_SMOTE)],
                   labels=["Sin SMOTE", "Con SMOTE"], frameon=False, loc="lower right")
    fig.suptitle("Fraudes no detectados vs falsas alarmas en prueba", fontweight="bold")
    fig.tight_layout()
    guardar(fig, "06_fraudes_perdidos_vs_falsas_alarmas.png")


def figura_efecto_smote(prueba):
    """Sensibilidad y precisión en prueba de cada modelo sin y con SMOTE."""
    x = np.arange(len(MODELOS))
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col, titulo in [(ejes[0], "sensibilidad", "Sensibilidad (fraudes detectados)"),
                            (ejes[1], "precision", "Precisión (alarmas que eran fraude)")]:
        sin = [prueba.loc[prueba["configuracion"] == etiqueta(m, False), col].iloc[0] for m in MODELOS]
        con = [prueba.loc[prueba["configuracion"] == etiqueta(m, True), col].iloc[0] for m in MODELOS]
        b1 = ax.bar(x - 0.2, sin, 0.4, color=COLOR_SIN, label="Sin SMOTE")
        b2 = ax.bar(x + 0.2, con, 0.4, color=COLOR_SMOTE, label="Con SMOTE")
        for barras in (b1, b2):
            ax.bar_label(barras, labels=[f(v, 2) for v in barras.datavalues], padding=2, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([NOMBRES[m] for m in MODELOS])
        ax.set_ylim(0, 1.15)
        ax.set_title(titulo)
    ejes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Efecto de SMOTE en prueba", fontweight="bold")
    fig.tight_layout()
    guardar(fig, "07_efecto_smote.png")


def figura_sensibilidad_folds(sens_folds):
    """Sensibilidad de cada fold de validación: muestra cuánto varía con tan pocos fraudes."""
    configs = list(sens_folds)
    valores = np.array([list(v) + [np.mean(v)] for v in sens_folds.values()])
    n_folds = valores.shape[1] - 1
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.imshow(valores, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(valores):
        ax.text(j, i, f(v, 2), ha="center", va="center", fontsize=10,
                color="white" if v > 0.6 else "black",
                fontweight="bold" if j == n_folds else "normal")
    ax.axvline(n_folds - 0.5, color="white", linewidth=4)
    ax.set_xticks(range(n_folds + 1))
    ax.set_xticklabels([f"Fold {i + 1}" for i in range(n_folds)] + ["Media"])
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs)
    ax.set_title("Sensibilidad en cada fold de validación\n"
                 "(con pocos fraudes por fold, un solo fraude mueve la sensibilidad varios puntos)")
    fig.tight_layout()
    guardar(fig, "08_sensibilidad_por_fold.png")


def figura_variables(caracteristicas):
    """Estadístico KS de cada variable (ajuste final). Sobre cada barra: en cuántos folds de CV
    la variable fue conservada."""
    c = caracteristicas.sort_values("ks", ascending=False)
    x = np.arange(len(c))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x, c["ks"], color=[COLOR_NEUTRO if v else COLOR_DESCARTADA for v in c["conservada"]])
    for xi, (ks, folds) in enumerate(zip(c["ks"], c["folds_conservada"])):
        ax.text(xi, ks + 0.01, f"{folds}/{N_FOLDS}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(c["variable"], rotation=90, fontsize=9)
    ax.set_ylim(0, c["ks"].max() * 1.12)
    ax.set_ylabel("Estadístico KS (entrenamiento completo)")
    ax.legend(handles=[Patch(color=COLOR_NEUTRO, label="Conservada por el filtro KS"),
                       Patch(color=COLOR_DESCARTADA, label="Descartada")],
              frameon=False, loc="upper right")
    ax.set_title("Variables conservadas por el filtro KS\n"
                 "(sobre cada barra: folds de validación cruzada en que fue conservada)")
    fig.tight_layout()
    guardar(fig, "09_variables_seleccionadas.png")


# ----------------------------------------------------------------------- main
def main():
    resultados = joblib.load(RUTA_BUSQUEDA)
    X_ent, X_pru, y_ent, y_pru = cargar_particion()
    print("EVALUACIÓN FINAL")
    print(f"Entrenamiento: {X_ent.shape[0]} filas ({int(y_ent.sum())} fraudes) | "
          f"Prueba: {X_pru.shape[0]} filas ({int(y_pru.sum())} fraudes)\n")

    val_filas, test_filas, selectores = [], [], []
    matrices_val, matrices_pru, sens_folds = {}, {}, {}
    for modelo in MODELOS:
        for smote in (False, True):
            res = resultados[clave(modelo, smote)]
            pipe = make_pipeline(modelo, smote).set_params(**res["params"])
            nombre = etiqueta(modelo, smote)
            print(f"{'=' * 70}\n{nombre} | hiperparámetros: {res['params']}")

            # Validación: solo entrenamiento, mismos folds que la búsqueda
            cv = cross_validate(pipe, X_ent, y_ent, cv=crear_cv(), scoring=SCORING_CV,
                                n_jobs=1 if modelo == "rf_propio" else -1)
            fila = {"configuracion": nombre}
            for m in ("sensibilidad", "especificidad", "precision", "f1", "exactitud"):
                fila[f"{m}_media"] = cv[f"test_{m}"].mean()
                fila[f"{m}_std"] = cv[f"test_{m}"].std()
            fila["fnr_media"] = 1 - fila["sensibilidad_media"]
            fila["fnr_std"] = fila["sensibilidad_std"]
            tn_v, fp_v, fn_v, tp_v = (int(cv[f"test_{c}"].sum()) for c in ("tn", "fp", "fn", "tp"))
            fila.update({"TP": tp_v, "FN": fn_v, "FP": fp_v, "TN": tn_v})
            val_filas.append(fila)
            matrices_val[nombre] = np.array([[tn_v, fp_v], [fn_v, tp_v]])
            sens_folds[nombre] = cv["test_sensibilidad"]

            # Prueba: reentrenar con todo el entrenamiento y evaluar una vez
            pipe.fit(X_ent, y_ent)
            pred = pipe.predict(X_pru)
            tn, fp, fn, tp = confusion_matrix(y_pru, pred, labels=[0, 1]).ravel()
            punto = metricas_de_conteos(tp, fn, fp, tn)
            desv = bootstrap_prueba(y_pru, pred)
            test_filas.append({"configuracion": nombre, **punto,
                               **{f"{m}_std": desv[m] for m in METRICAS},
                               "TP": tp, "FN": fn, "FP": fp, "TN": tn})
            matrices_pru[nombre] = np.array([[tn, fp], [fn, tp]])

            selector = pipe.named_steps["selector"]
            selectores.append(selector)

            if smote:
                print(f"  SMOTE (legítimos/fraudes): {conteos_smote(pipe, X_ent, y_ent)}")
            print(f"  Variables KS             : {int(selector.get_support().sum())}")
            print(f"  Validación (CV)          : sensibilidad = {f(fila['sensibilidad_media'])} "
                  f"± {f(fila['sensibilidad_std'])}")
            print(f"  Prueba                   : sensibilidad = {f(punto['sensibilidad'])} | "
                  f"FN = {fn} | FP = {fp}")

    validacion = pd.DataFrame(val_filas)
    prueba = pd.DataFrame(test_filas)
    validacion.to_csv(CARPETA / "validacion_cv.csv", index=False)
    prueba.to_csv(CARPETA / "prueba.csv", index=False)

    # Variables del filtro KS: se informan una sola vez (no dependen del clasificador)
    mascaras = [s.get_support() for s in selectores]
    coinciden = all(np.array_equal(mascaras[0], m) for m in mascaras)
    caracteristicas = pd.DataFrame({
        "variable": list(X_ent.columns),
        "ks": selectores[0].ks_,
        "conservada": mascaras[0],
        "folds_conservada": folds_que_conservan(X_ent, y_ent),
    }).sort_values("ks", ascending=False)
    caracteristicas.to_csv(CARPETA / "caracteristicas_seleccionadas.csv", index=False)

    print(f"\n{'=' * 70}\nRESULTADOS EN VALIDACIÓN CRUZADA (media ± desv. estándar entre los 5 folds)")
    print(tabla_formateada(validacion, lambda m: validacion[f"{m}_media"]).to_string(index=False))
    print("\nRESULTADOS EN PRUEBA (valor ± desv. estándar por bootstrap del conjunto de prueba)")
    print(tabla_formateada(prueba, lambda m: prueba[m], ("FN", "FP")).to_string(index=False))

    conservadas = caracteristicas.loc[caracteristicas["conservada"], "variable"]
    print("\nVARIABLES CONSERVADAS POR EL FILTRO KS (ajuste final con todo el entrenamiento)")
    print(f"   {len(conservadas)} de {len(caracteristicas)}: {', '.join(conservadas)}")
    print(f"   Mismas variables en las 8 configuraciones: {'sí' if coinciden else 'NO'}")
    print(caracteristicas.round(3).to_string(index=False))

    print("\nFIGURAS")
    configs = list(validacion["configuracion"])
    figura_matrices(matrices_val, "Matrices de confusión en validación cruzada (suma de los 5 folds)",
                    "01_matrices_confusion_validacion.png")
    figura_matrices(matrices_pru, "Matrices de confusión en prueba",
                    "02_matrices_confusion_prueba.png")
    figura_metricas(configs, validacion[[f"{m}_media" for m in METRICAS]].to_numpy(),
                    "Métricas en validación cruzada (media de los 5 folds)",
                    "03_metricas_validacion.png")
    figura_metricas(configs, prueba[METRICAS].to_numpy(), "Métricas en prueba",
                    "04_metricas_prueba.png")
    figura_sensibilidad(validacion, prueba)
    figura_fn_fp(prueba)
    figura_efecto_smote(prueba)
    figura_sensibilidad_folds(sens_folds)
    figura_variables(caracteristicas)


if __name__ == "__main__":
    main()