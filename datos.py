"""
datos.py
Módulo 1: carga, limpieza, submuestra estratificada y partición entrenamiento/prueba.

Este módulo SOLO prepara los datos. No imprime ni grafica nada: toda la exploración
(cifras y figuras) vive en exploracion.py.

Uso desde otros módulos:
    from datos import cargar_particion
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = cargar_particion()

Uso desde exploracion.py (necesita ver cada etapa para documentar la limpieza):
    from datos import cargar_etapas, particionar
    crudo, limpio, submuestra = cargar_etapas()
    X_ent, X_pru, y_ent, y_pru = particionar(submuestra)
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# --- Parámetros globales del proyecto ---
SEMILLA = 5040
N_SUBMUESTRA = 50_000
PROPORCION_PRUEBA = 0.30
RUTA_CSV = Path(__file__).parent / "creditcard.csv"  # el CSV debe estar junto a este archivo


def cargar_crudo():
    """Lee el CSV original sin modificarlo."""
    return pd.read_csv(RUTA_CSV)


def limpiar(datos):
    """Elimina duplicados exactos. Se hace ANTES de particionar para que una misma
    fila no pueda quedar a la vez en entrenamiento y en prueba."""
    return datos.drop_duplicates().reset_index(drop=True)


def tomar_submuestra(datos):
    """Submuestra estratificada de N_SUBMUESTRA filas (conserva la proporción de fraude)."""
    submuestra, _ = train_test_split(
        datos, train_size=N_SUBMUESTRA, stratify=datos["Class"], random_state=SEMILLA
    )
    return submuestra


def cargar_etapas():
    """Devuelve (crudo, limpio, submuestra): los datos en cada etapa del preprocesamiento."""
    crudo = cargar_crudo()
    limpio = limpiar(crudo)
    submuestra = tomar_submuestra(limpio)
    return crudo, limpio, submuestra


def particionar(submuestra):
    """Separa X / y (Class no entra como predictora) y hace la partición 70/30 estratificada."""
    X = submuestra.drop(columns="Class")
    y = submuestra["Class"]
    return train_test_split(
        X, y, test_size=PROPORCION_PRUEBA, stratify=y, random_state=SEMILLA
    )


def cargar_particion():
    """Devuelve X_entrenamiento, X_prueba, y_entrenamiento, y_prueba.

    Con la semilla fija, la partición es idéntica en cada ejecución.
    """
    _, _, submuestra = cargar_etapas()
    return particionar(submuestra)


if __name__ == "__main__":
    # Comprobación mínima de que la carga funciona (la exploración está en exploracion.py)
    X_ent, X_pru, y_ent, y_pru = cargar_particion()
    print(f"Entrenamiento: {X_ent.shape} | Prueba: {X_pru.shape}")