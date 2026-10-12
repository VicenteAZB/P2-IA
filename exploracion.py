"""
exploracion.py
Análisis exploratorio. Es el ÚNICO módulo que muestra cifras y guarda figuras; datos.py solo carga.

Reglas de uso de los datos:
  - Las variables predictoras solo se analizan en ENTRENAMIENTO.
  - De PRUEBA solo se usan los conteos de clase (verificar estratificación y exactitud trivial).
  - Los análisis de relación con el objetivo son DESCRIPTIVOS; la selección real de
    características se hace dentro de la validación cruzada.

Ejecutar:
    python exploracion.py            # guarda las figuras en resultados/
    python exploracion.py --mostrar  # además las abre en ventanas
"""
import sys
from pathlib import Path

import matplotlib
MOSTRAR = "--mostrar" in sys.argv
if not MOSTRAR:
    matplotlib.use("Agg")                  # guardar figuras sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy.stats import ks_2samp
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold

from datos import SEMILLA, cargar_etapas, particionar

CARPETA = Path(__file__).parent / "resultados"
CARPETA.mkdir(exist_ok=True)

COLOR_LEGITIMO = "#2E75B6"
COLOR_FRAUDE = "#C0392B"
COLOR_PCA = "#8FA9C4"
COLOR_ORIGINAL = "#E67E22"
MONTOS_MARCAS = [0, 1, 10, 100, 1000, 10000, 100000]
N_FOLDS = 5                                # debe coincidir con la validación cruzada final

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})


# ----------------------------------------------------------------- utilidades
def formato(valor, decimales=0):
    """Número con formato español: punto para miles y coma para decimales."""
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def guardar(fig, nombre):
    fig.savefig(CARPETA / nombre, dpi=200, bbox_inches="tight")
    print(f"   -> figura guardada: resultados/{nombre}\n")
    if MOSTRAR:
        plt.show()
    plt.close(fig)


def figura_tabla(celdas, columnas, titulo, nombre, ancho=8, destacar_filas=()):
    """Figura-tabla reutilizable. destacar_filas: índices de fila (sin encabezado) en negrita."""
    fig, ax = plt.subplots(figsize=(ancho, 0.5 + 0.38 * (len(celdas) + 1)))
    ax.axis("off")
    t = ax.table(cellText=celdas, colLabels=columnas, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.7)
    for (fila, _), celda in t.get_celld().items():
        if fila == 0:
            celda.set_facecolor("#2F3E54")
            celda.set_text_props(color="white", fontweight="bold")
        elif (fila - 1) in destacar_filas:
            celda.set_text_props(fontweight="bold")
    ax.set_title(titulo)
    guardar(fig, nombre)


def separar_por_clase(serie, y):
    """Devuelve (valores de legítimos, valores de fraudes) de una variable."""
    return serie[y == 0], serie[y == 1]


def estadisticas_por_clase(legitimos, fraudes):
    """Resumen con asimetría y valores extremos (regla del rango intercuartílico)."""
    columnas = {}
    for etiqueta, v in [("Legítimos", legitimos), ("Fraudes", fraudes)]:
        q1, q3 = v.quantile([0.25, 0.75])
        limite = q3 + 1.5 * (q3 - q1)
        columnas[etiqueta] = {
            "n": len(v), "mediana": v.median(), "media": v.mean(),
            "percentil 99": v.quantile(0.99), "máximo": v.max(),
            "asimetría": v.skew(), "% valores extremos": 100 * (v > limite).mean(),
        }
    return pd.DataFrame(columnas).round(2)


def marcas_de_monto(ax, maximo, eje="x"):
    """Escala log en el eje elegido, rotulada con los montos reales."""
    montos = [m for m in MONTOS_MARCAS if m <= maximo]
    posiciones = np.log1p(montos)
    rotulos = [formato(m) for m in montos]
    if eje == "x":
        ax.set_xticks(posiciones)
        ax.set_xticklabels(rotulos)
    else:
        ax.set_yticks(posiciones)
        ax.set_yticklabels(rotulos)


def calcular_ks(X, y):
    """Estadístico KS por variable (0 = distribuciones idénticas, 1 = totalmente separadas)."""
    return pd.Series({c: ks_2samp(X.loc[y == 0, c], X.loc[y == 1, c]).statistic
                      for c in X.columns})


# ------------------------------------------------------- 1. Calidad de datos
def figura_calidad_datos(crudo, limpio, submuestra, y_ent, y_pru):
    n_crudo, n_limpio = len(crudo), len(limpio)
    duplicados = n_crudo - n_limpio
    fraudes_crudo = int(crudo["Class"].sum())
    fraudes_limpio = int(limpio["Class"].sum())
    fraudes_dup = fraudes_crudo - fraudes_limpio
    faltantes = int(crudo.isna().sum().sum())
    n_sub, fraudes_sub = len(submuestra), int(submuestra["Class"].sum())

    print("1. CALIDAD DE LOS DATOS Y LIMPIEZA")
    print(f"   Registros: {formato(n_crudo)} | Características predictoras: {crudo.shape[1] - 1}")
    print(f"   Valores faltantes: {formato(faltantes)}")
    print(f"   Duplicados exactos: {formato(duplicados)} (de ellos, {fraudes_dup} eran fraudes)")
    print(f"   Fraudes: {formato(fraudes_crudo)} -> {formato(fraudes_limpio)} tras limpiar")
    print(f"   Submuestra: {formato(n_sub)} filas, {fraudes_sub} fraudes "
          f"({formato(100 * fraudes_sub / n_sub, 3)} %)")

    filas = [
        ["Registros originales", formato(n_crudo)],
        ["Características predictoras", str(crudo.shape[1] - 1)],
        ["Valores faltantes", formato(faltantes)],
        ["Duplicados exactos eliminados", f"{formato(duplicados)}  ({fraudes_dup} eran fraudes)"],
        ["Registros tras limpiar", formato(n_limpio)],
        ["Fraudes originales → tras limpiar", f"{formato(fraudes_crudo)} → {formato(fraudes_limpio)}"],
        ["Submuestra estratificada", f"{formato(n_sub)}  ({fraudes_sub} fraudes)"],
        ["Entrenamiento (70 %)", f"{formato(len(y_ent))}  ({int(y_ent.sum())} fraudes)"],
        ["Prueba (30 %)", f"{formato(len(y_pru))}  ({int(y_pru.sum())} fraudes)"],
    ]
    figura_tabla(filas, ["Indicador", "Valor"],
                 "Calidad de los datos y decisiones de limpieza", "01_calidad_datos.png",
                 ancho=8.5, destacar_filas=(3, 4))


# ------------------------------------------------------------------ 2. Clases
def figura_distribucion_clases(y):
    conteo = y.value_counts().sort_index()
    n_leg, n_fra = int(conteo[0]), int(conteo[1])
    total = n_leg + n_fra
    print("2. DISTRIBUCIÓN DE CLASES (entrenamiento)")
    print(f"   Legítimos: {formato(n_leg)} | Fraudes: {formato(n_fra)}")
    print(f"   Proporción de fraude: {formato(100 * n_fra / total, 3)} %")
    print(f"   Razón de desbalance: 1 fraude por cada {formato(n_leg / n_fra)} legítimos")

    fig, ejes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, escala in zip(ejes, ["linear", "log"]):
        valores = [n_leg, n_fra]
        barras = ax.bar(["Legítimos", "Fraudes"], valores, color=[COLOR_LEGITIMO, COLOR_FRAUDE])
        ax.set_yscale(escala)
        for barra, valor in zip(barras, valores):
            ax.annotate(f"{formato(valor)}\n({formato(100 * valor / total, 2)} %)",
                        (barra.get_x() + barra.get_width() / 2, valor), ha="center",
                        va="bottom", xytext=(0, 3), textcoords="offset points")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: formato(v)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel("Cantidad de transacciones")
    ejes[0].set_ylim(0, n_leg * 1.2)
    ejes[0].set_title("Escala lineal")
    ejes[1].set_ylim(1, n_leg * 6)
    ejes[1].set_title("Escala logarítmica (cada marca = ×10)")
    fig.suptitle("Distribución de clases en el conjunto de entrenamiento", fontweight="bold")
    fig.tight_layout()
    guardar(fig, "02_distribucion_clases.png")


# --------------------------------------------------------------- 3. Partición
def figura_particion(y_ent, y_pru):
    filas = []
    for nombre, y in [("Entrenamiento", y_ent), ("Prueba", y_pru)]:
        filas.append([nombre, int((y == 0).sum()), int((y == 1).sum())])
    filas.append(["Total", filas[0][1] + filas[1][1], filas[0][2] + filas[1][2]])
    tabla = pd.DataFrame(filas, columns=["Conjunto", "Legítimos", "Fraudes"])
    tabla["Total"] = tabla["Legítimos"] + tabla["Fraudes"]
    tabla["% de fraude"] = 100 * tabla["Fraudes"] / tabla["Total"]

    print("3. PARTICIÓN ENTRENAMIENTO / PRUEBA")
    print(tabla.round(3).to_string(index=False))
    dif = abs(tabla.loc[0, "% de fraude"] - tabla.loc[1, "% de fraude"])
    print(f"   Diferencia de proporción de fraude entre entrenamiento y prueba: "
          f"{formato(dif, 4)} puntos porcentuales")

    celdas = [[f, formato(l), formato(fr), formato(t), formato(p, 3) + " %"]
              for f, l, fr, t, p in tabla.itertuples(index=False)]
    figura_tabla(celdas, ["Conjunto", "Legítimos", "Fraudes", "Total", "% de fraude"],
                 "La partición estratificada conserva la proporción de fraude",
                 "03_particion_estratificada.png", destacar_filas=(2,))


# -------------------------------------------- 4. Fraudes disponibles por fold
def figura_fraudes_por_fold(y_ent, y_pru):
    """Cuántos fraudes quedan realmente en cada partición de la validación cruzada."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA)
    y_arr = y_ent.to_numpy()
    filas = []
    for i, (idx_ent, idx_val) in enumerate(skf.split(np.zeros(len(y_arr)), y_arr), start=1):
        filas.append((i, int(y_arr[idx_ent].sum()), int(y_arr[idx_val].sum())))
    folds = pd.DataFrame(filas, columns=["fold", "fraudes_entrenar", "fraudes_validar"])

    print("4. FRAUDES DISPONIBLES POR PARTICIÓN DE VALIDACIÓN CRUZADA")
    print(folds.to_string(index=False))
    min_val = int(folds["fraudes_validar"].min())
    print(f"   Con ~{min_val} fraudes en validación, cada fraude vale "
          f"{formato(100 / min_val, 1)} puntos de sensibilidad -> desviaciones estándar grandes.")
    print(f"   En prueba hay {int(y_pru.sum())} fraudes: "
          f"cada uno vale {formato(100 / int(y_pru.sum()), 1)} puntos de sensibilidad.")

    x = np.arange(N_FOLDS)
    fig, ax = plt.subplots(figsize=(9, 4))
    b1 = ax.bar(x - 0.2, folds["fraudes_entrenar"], 0.4, color=COLOR_LEGITIMO, label="train")
    b2 = ax.bar(x + 0.2, folds["fraudes_validar"], 0.4, color=COLOR_FRAUDE, label="test")
    for barras in (b1, b2):
        ax.bar_label(barras, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in folds["fold"]])
    ax.set_ylabel("Cantidad de fraudes")
    ax.set_ylim(0, folds["fraudes_entrenar"].max() * 1.2)
    ax.legend(frameon=False)
    ax.set_title(f"Pocos fraudes por partición: la sensibilidad en validación será ruidosa\n"
                 f"(validación cruzada estratificada de {N_FOLDS} particiones)")
    fig.tight_layout()
    guardar(fig, "04_fraudes_por_fold.png")


# ------------------------------------------------ 5. Por qué la exactitud engaña
def figura_exactitud_enganosa(y_ent, y_pru):
    """Clasificador trivial 'siempre legítimo'. Solo usa conteos de clase (no variables)."""
    print("5. CLASIFICADOR TRIVIAL: SIEMPRE PREDICE 'LEGÍTIMO'")
    resultados = {}
    for nombre, y in [("Entrenamiento", y_ent), ("Prueba", y_pru)]:
        exactitud = 100 * (y == 0).mean()
        sensibilidad = 0.0                       # no detecta ningún fraude
        resultados[nombre] = (exactitud, sensibilidad)
        print(f"   {nombre}: exactitud = {formato(exactitud, 3)} % | sensibilidad = "
              f"{formato(sensibilidad, 1)} % | fraudes no detectados = {int(y.sum())} de {int(y.sum())}")

    x = np.arange(len(resultados))
    fig, ax = plt.subplots(figsize=(8, 4))
    b1 = ax.bar(x - 0.2, [v[0] for v in resultados.values()], 0.4, color=COLOR_LEGITIMO, label="Exactitud")
    b2 = ax.bar(x + 0.2, [v[1] for v in resultados.values()], 0.4, color=COLOR_FRAUDE, label="Sensibilidad")
    ax.bar_label(b1, labels=[formato(v[0], 2) + " %" for v in resultados.values()], padding=2)
    ax.bar_label(b2, labels=[formato(v[1], 0) + " %" for v in resultados.values()], padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(list(resultados))
    ax.set_ylim(0, 115)
    ax.set_ylabel("%")
    ax.legend(frameon=False, loc="center right")
    ax.set_title("La exactitud engaña: predecir siempre «legítimo» parece excelente\n"
                 "pero no detecta ni un solo fraude")
    fig.tight_layout()
    guardar(fig, "05_exactitud_enganosa.png")


# ------------------------------------------------------- 6. Escala de variables
def figura_escala_variables(X):
    desviaciones = X.std()
    print("6. ESCALA DE LAS VARIABLES (desviación estándar)")
    print(f"   Mayor: {desviaciones.idxmax()} = {formato(desviaciones.max(), 2)} | "
          f"menor: {desviaciones.idxmin()} = {formato(desviaciones.min(), 2)} | "
          f"cociente: {formato(desviaciones.max() / desviaciones.min())} veces")
    print("   Time:", formato(desviaciones["Time"], 1), "| Amount:", formato(desviaciones["Amount"], 1))

    colores = [COLOR_ORIGINAL if c in ("Time", "Amount") else COLOR_PCA for c in desviaciones.index]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(desviaciones.index, desviaciones.values, color=colores)
    ax.set_yscale("log")
    ax.set_ylim(top=desviaciones.max() * 20)        # espacio libre arriba para la leyenda
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: formato(v)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("Desviación estándar (escala log)")
    ax.tick_params(axis="x", rotation=90)
    for nombre in ("Time", "Amount"):
        ax.annotate(formato(desviaciones[nombre], 0),
                    (list(desviaciones.index).index(nombre), desviaciones[nombre]),
                    ha="center", va="bottom", xytext=(0, 3), textcoords="offset points",
                    fontweight="bold")
    ax.legend(handles=[Patch(color=COLOR_PCA, label="V1–V28 (componentes PCA)"),
                       Patch(color=COLOR_ORIGINAL, label="Variables originales (Time, Amount)")],
              loc="upper center", frameon=False)
    ax.set_title("Las variables están en escalas muy distintas: kNN y SVM requieren escalado")
    fig.tight_layout()
    guardar(fig, "06_escala_variables.png")


# ----------------------------------------------------------------- 7. Amount
def figura_amount(X, y):
    leg, fra = separar_por_clase(X["Amount"], y)
    print("7. AMOUNT POR CLASE")
    print(estadisticas_por_clase(leg, fra).to_string())
    print("   (% valores extremos = sobre Q3 + 1,5·IQR de su propia clase)")

    l_leg, l_fra = np.log1p(leg), np.log1p(fra)
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 2]})

    bins = np.linspace(0, np.log1p(X["Amount"].max()), 41)
    for valores, color, etiqueta in [(l_leg, COLOR_LEGITIMO, f"Legítimos (n = {formato(len(leg))})"),
                                     (l_fra, COLOR_FRAUDE, f"Fraudes (n = {formato(len(fra))})")]:
        ejes[0].hist(valores, bins=bins, weights=np.full(len(valores), 100 / len(valores)),
                     alpha=0.55, color=color, label=etiqueta)
    marcas_de_monto(ejes[0], X["Amount"].max(), "x")
    ejes[0].set_xlabel("Monto de la transacción (escala logarítmica)")
    ejes[0].set_ylabel("% de las transacciones de la clase")
    ejes[0].set_title("Distribución del monto")
    ejes[0].legend(frameon=False)

    caja = ejes[1].boxplot([l_leg, l_fra], patch_artist=True, widths=0.55,
                           flierprops={"marker": "o", "markersize": 2, "alpha": 0.4},
                           medianprops={"color": "black"})
    for parte, color in zip(caja["boxes"], [COLOR_LEGITIMO, COLOR_FRAUDE]):
        parte.set_facecolor(color)
        parte.set_alpha(0.6)
    ejes[1].set_xticks([1, 2])
    ejes[1].set_xticklabels(["Legítimos", "Fraudes"])
    marcas_de_monto(ejes[1], X["Amount"].max(), "y")
    ejes[1].set_ylabel("Monto (escala logarítmica)")
    ejes[1].set_title("Mediana, cuartiles y valores extremos")
    fig.suptitle(f"Monto de las transacciones por clase (asimetría global: "
                 f"{formato(X['Amount'].skew(), 1)})", fontweight="bold")
    fig.tight_layout()
    guardar(fig, "07_amount_por_clase.png")


# ------------------------------------------------------------------ 8. Time
def figura_time(X, y):
    horas = X["Time"] / 3600
    leg, fra = separar_por_clase(horas, y)
    print("8. TIME POR CLASE (en horas desde la primera transacción)")
    print(estadisticas_por_clase(leg, fra).to_string())

    horas_max = int(np.ceil(horas.max() / 2) * 2)
    bins = np.arange(0, horas_max + 2, 2)
    fig, ax = plt.subplots(figsize=(10, 4))
    for valores, color, etiqueta in [(leg, COLOR_LEGITIMO, f"Legítimos (n = {formato(len(leg))})"),
                                     (fra, COLOR_FRAUDE, f"Fraudes (n = {formato(len(fra))})")]:
        ax.hist(valores, bins=bins, weights=np.full(len(valores), 100 / len(valores)),
                alpha=0.55, color=color, label=etiqueta)
    if horas_max > 24:
        ax.axvline(24, color="gray", linestyle="--", linewidth=1)
        ax.text(24.3, ax.get_ylim()[1] * 0.95, "día 2", color="gray", va="top")
    ax.set_xlabel("Horas transcurridas desde la primera transacción (tramos de 2 h)")
    ax.set_ylabel("% de las transacciones de la clase")
    ax.set_title("¿Cuándo ocurren las transacciones?  Time por clase")
    ax.legend(frameon=False)
    fig.tight_layout()
    guardar(fig, "08_time_por_clase.png")


# ----------------------------------------------- 9. Variables V más llamativas
def figura_variables_llamativas(X, y, ks, cantidad=6):
    """Cajas de las V cuyas distribuciones más difieren entre clases (estadístico KS)."""
    ks_v = ks[[c for c in X.columns if c.startswith("V")]].sort_values(ascending=False)
    print("9. VARIABLES V CON MAYOR DIFERENCIA ENTRE CLASES (estadístico KS)")
    print(ks_v.head(10).round(3).to_string())

    fig, ejes = plt.subplots(2, 3, figsize=(11, 6.5))
    n_leg, n_fra = int((y == 0).sum()), int((y == 1).sum())
    for ax, variable in zip(ejes.ravel(), ks_v.index[:cantidad]):
        leg, fra = separar_por_clase(X[variable], y)
        caja = ax.boxplot([leg, fra], patch_artist=True, widths=0.55, showfliers=False,
                          medianprops={"color": "black"})
        for parte, color in zip(caja["boxes"], [COLOR_LEGITIMO, COLOR_FRAUDE]):
            parte.set_facecolor(color)
            parte.set_alpha(0.6)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"Legítimos\n(n = {formato(n_leg)})", f"Fraudes\n(n = {formato(n_fra)})"])
        ax.set_title(f"{variable}   (KS = {formato(ks_v[variable], 2)})")
    fig.suptitle("Variables V cuya distribución más difiere entre fraudes y legítimos\n"
                 "(no se muestran los valores atípicos)", fontweight="bold")
    fig.tight_layout()
    guardar(fig, "09_variables_llamativas.png")
    return list(ks_v.index)


# ------------------------------------- 10. Relación de cada variable con la clase
def figura_relacion_objetivo(X, y, ks, top=10):
    """Compara cuatro formas de medir la relación variable-objetivo.

    - Pearson (con Class binaria = correlación punto-biserial): diferencias LINEALES de
      promedio; sensible a valores extremos.
    - Spearman: igual idea pero con rangos; robusta a extremos (puede perder señal que
      viene justamente de los extremos).
    - KS: compara las distribuciones completas de ambas clases (no paramétrico).
    - Información mutua: cualquier dependencia, también no lineal (muy ruidosa con pocos fraudes).
    Si las medidas coinciden en las mejores variables, la conclusión es robusta.
    Es solo descriptivo (entrenamiento); la selección real va dentro de la validación cruzada.
    """
    from itertools import combinations

    tabla = pd.DataFrame({
        "Pearson r": X.apply(lambda c: c.corr(y)),
        "Spearman ρ": X.apply(lambda c: c.corr(y, method="spearman")),
        "KS": ks,
        "Info. mutua": pd.Series(mutual_info_classif(X, y, random_state=SEMILLA), index=X.columns),
    })
    tabla["|Pearson|"] = tabla["Pearson r"].abs()
    tabla["|Spearman|"] = tabla["Spearman ρ"].abs()

    print("10. RELACIÓN DE CADA VARIABLE CON LA CLASE (entrenamiento)")
    print(tabla.sort_values("KS", ascending=False).head(10)
          [["Pearson r", "Spearman ρ", "KS", "Info. mutua"]].round(3).to_string())
    print("   Pearson con Class binaria = correlación punto-biserial. Con ~0,17 % de fraude, los")
    print("   valores absolutos quedan comprimidos: se usan para ORDENAR, no para interpretar.")

    medidas = [("|Pearson|", "Pearson (lineal)"),
               ("|Spearman|", "Spearman (rangos)"),
               ("KS", "KS (distribuciones)"),
               ("Info. mutua", "Información mutua (no lineal)")]
    tops = {col: set(tabla[col].sort_values(ascending=False).head(top).index) for col, _ in medidas}
    comunes = set.intersection(*tops.values())
    print(f"   Variables en el top {top} de las CUATRO medidas: {sorted(comunes)}")
    for (a, _), (b, _) in combinations(medidas, 2):
        print(f"   Coincidencia {a} - {b}: {len(tops[a] & tops[b])}/{top}")
    for nombre in ("Time", "Amount"):
        pos = int(tabla["KS"].rank(ascending=False)[nombre])
        print(f"   {nombre}: KS = {formato(tabla.loc[nombre, 'KS'], 3)} "
              f"(puesto {pos} de {len(tabla)} según KS)")

    fig, ejes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (col, titulo) in zip(ejes.ravel(), medidas):
        serie = tabla[col].sort_values(ascending=False).head(top)
        colores = [COLOR_FRAUDE if v in comunes else COLOR_PCA for v in serie.index]
        ax.barh(serie.index[::-1], serie.values[::-1], color=colores[::-1])
        ax.set_title(titulo)
        ax.set_xlabel(col)
    ejes[1, 1].legend(handles=[Patch(color=COLOR_FRAUDE, label=f"En el top {top} de las cuatro"),
                               Patch(color=COLOR_PCA, label="Solo en algunas medidas")],
                      loc="lower right", frameon=False)
    fig.suptitle(f"Las {top} variables más relacionadas con el fraude según cuatro medidas",
                 fontweight="bold")
    fig.tight_layout()
    guardar(fig, "10_relacion_con_objetivo.png")
    return tabla

# ------------------------------------------------ 11. Separabilidad visual
def figura_dispersion(X, y, variables_top):
    a, b = variables_top[:2]
    print(f"11. SEPARABILIDAD VISUAL: {a} vs {b} (las dos V con mayor KS)")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(X.loc[y == 0, a], X.loc[y == 0, b], s=6, alpha=0.25, color=COLOR_LEGITIMO,
               label=f"Legítimos (n = {formato(int((y == 0).sum()))})")
    ax.scatter(X.loc[y == 1, a], X.loc[y == 1, b], s=22, alpha=0.9, color=COLOR_FRAUDE,
               edgecolor="white", linewidth=0.4, label=f"Fraudes (n = {formato(int(y.sum()))})")
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.legend(frameon=False)
    ax.set_title(f"¿Se separan las clases? {a} vs {b}")
    fig.tight_layout()
    guardar(fig, "11_dispersion_top2.png")


# --------------------------------------- 12. Correlación entre predictoras
def figura_correlacion_predictoras(X):
    corr = X.corr(method="pearson")
    vs = [c for c in X.columns if c.startswith("V")]
    entre_v = np.array(corr.loc[vs, vs])
    pares = np.abs(entre_v[np.triu_indices(len(vs), k=1)])
    print("12. CORRELACIÓN ENTRE PREDICTORAS (Pearson)")
    print(f"   Máxima |correlación| entre dos variables V: {formato(pares.max(), 3)}")
    print(f"   % de pares de V con |correlación| > 0,3: {formato(100 * (pares > 0.3).mean(), 1)} %")
    print("   (Las V salen de PCA: son linealmente no correlacionadas por construcción; eso no")
    print("    descarta dependencias no lineales.)")
    for nombre in ("Time", "Amount"):
        top = corr.loc[nombre, vs].abs().sort_values(ascending=False).head(3).round(3)
        print(f"   {nombre} se relaciona más con: " + ", ".join(f"{k} ({formato(v, 3)})" for k, v in top.items()))

    fig, ax = plt.subplots(figsize=(9, 7.5))
    imagen = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(imagen, ax=ax, shrink=0.8, label="Correlación de Pearson")
    ax.set_title("Correlación entre variables predictoras (entrenamiento)")
    fig.tight_layout()
    guardar(fig, "12_correlacion_predictoras.png")


# ----------------------------------------- 13. Colas pesadas y valores atípicos
def figura_colas(X):
    asimetria = X.skew().abs().sort_values(ascending=False)
    n_pesadas = int((asimetria > 1).sum())
    print("13. ASIMETRÍA Y COLAS PESADAS")
    print(f"   Variables con |asimetría| > 1: {n_pesadas} de {len(asimetria)}")
    print("   Más asimétricas: " + ", ".join(f"{k} ({formato(v, 1)})" for k, v in asimetria.head(5).items()))
    print("   Decisión: NO se recortan valores atípicos; en fraude los extremos suelen ser la señal.")

    colores = [COLOR_ORIGINAL if c in ("Time", "Amount") else COLOR_PCA for c in asimetria.index]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(asimetria.index, asimetria.values, color=colores)
    ax.axhline(1, color="gray", linestyle="--", linewidth=1)
    ax.text(len(asimetria) - 0.5, 1, " |asimetría| = 1", color="gray", va="bottom", ha="right")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: formato(v, 1)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel("|Asimetría| (escala log)")
    ax.set_title(f"{n_pesadas} de {len(asimetria)} variables tienen colas marcadas; no se recortan los atípicos")
    fig.tight_layout()
    guardar(fig, "13_asimetria_colas.png")


# ----------------------------------------------------------------------- main
def main():
    crudo, limpio, submuestra = cargar_etapas()
    X_ent, _, y_ent, y_pru = particionar(submuestra)   # de la prueba solo se usa y_pru (conteos)
    print(f"Entrenamiento: {formato(X_ent.shape[0])} filas, {X_ent.shape[1]} variables\n")

    figura_calidad_datos(crudo, limpio, submuestra, y_ent, y_pru)
    figura_distribucion_clases(y_ent)
    figura_particion(y_ent, y_pru)
    figura_fraudes_por_fold(y_ent, y_pru)
    figura_exactitud_enganosa(y_ent, y_pru)
    figura_escala_variables(X_ent)
    figura_amount(X_ent, y_ent)
    figura_time(X_ent, y_ent)
    ks = calcular_ks(X_ent, y_ent)
    variables_top = figura_variables_llamativas(X_ent, y_ent, ks)
    figura_relacion_objetivo(X_ent, y_ent, ks)
    figura_dispersion(X_ent, y_ent, variables_top)
    figura_correlacion_predictoras(X_ent)
    figura_colas(X_ent)


if __name__ == "__main__":
    main()