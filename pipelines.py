"""
pipelines.py — Factory de Pipelines (solo definición, sin ajustar).

Cada modelo tiene dos versiones: sin compensación del desbalance y con SMOTE.
Se arman 8 pipelines: 4 modelos x {sin SMOTE, con SMOTE}.

Orden: scaler -> selector (KS) -> sampler -> clf.
Todo se ajusta dentro del Pipeline, solo con el train de cada fold, así no hay fuga.
Se usa imblearn.Pipeline porque el sampler solo debe tocar el train, nunca la validación.

Decisiones de diseño:
- Escalado: kNN y SVM (usan distancias) siempre. También cualquier pipeline con SMOTE,
  porque SMOTE busca vecinos por distancia euclidiana y sin escalar Time (σ≈47.000)
  dominaría la vecindad. Para los árboles el escalado no cambia el resultado.
- Selección de características: filtro estadístico con el test de Kolmogorov-Smirnov. Se
  conservan las variables cuya distribución difiere entre fraude y legítimo con p < ALFA_KS.
  KS no pertenece a ninguno de los clasificadores, solo usa el orden de los valores (no
  depende de la escala) y por eso elige lo mismo en las 8 configuraciones.
- Árboles: criterio de entropía (ganancia de información, el criterio de ID3).
- La compensación es SMOTE en los cuatro modelos (una sola estrategia para todos).
"""
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import ks_2samp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from datos import SEMILLA
from random_forest_propio import RandomForestPropio

MODELOS = ("knn", "arbol", "svm", "rf_propio")
NOMBRES = {"knn": "kNN", "arbol": "Árbol", "svm": "SVM", "rf_propio": "RF propio"}

ALFA_KS = 0.05


class SelectorKS(BaseEstimator, TransformerMixin):
    """Conserva las variables cuya distribución difiere entre clases (test KS, p < alfa).

    Si ninguna supera el umbral, se conserva la de mayor estadístico KS.
    """

    def __init__(self, alfa=ALFA_KS):
        self.alfa = alfa

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        pruebas = [ks_2samp(X[y == 0, j], X[y == 1, j]) for j in range(X.shape[1])]
        self.ks_ = np.array([p.statistic for p in pruebas])
        self.mascara_ = np.array([p.pvalue for p in pruebas]) < self.alfa
        if not self.mascara_.any():
            self.mascara_[np.argmax(self.ks_)] = True
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.mascara_]

    def get_support(self, indices=False):
        """Máscara booleana (o índices) de las variables conservadas."""
        return np.flatnonzero(self.mascara_) if indices else self.mascara_


def clave(modelo, smote):
    """Identificador de la configuración: 'arbol' o 'arbol_smote'."""
    return f"{modelo}_smote" if smote else modelo


def etiqueta(modelo, smote):
    """Nombre legible de la configuración: 'Árbol' o 'Árbol con SMOTE'."""
    return f"{NOMBRES[modelo]} con SMOTE" if smote else NOMBRES[modelo]


def _crear_scaler(modelo, smote):
    """StandardScaler en kNN, SVM y siempre que haya SMOTE; en el resto, passthrough."""
    if modelo in ("knn", "svm") or smote:
        return StandardScaler()
    return "passthrough"


def _crear_sampler(smote):
    """Sin SMOTE -> passthrough; con SMOTE -> SMOTE."""
    if not smote:
        return "passthrough"
    return SMOTE(random_state=SEMILLA, k_neighbors=5)


def _crear_clasificador(modelo):
    """Clasificador con valores por defecto; el ajuste fino se hace en busqueda.py."""
    if modelo == "knn":
        return KNeighborsClassifier()
    if modelo == "arbol":
        return DecisionTreeClassifier(criterion="entropy", random_state=SEMILLA)
    if modelo == "svm":
        return SVC(random_state=SEMILLA)
    if modelo == "rf_propio":
        return RandomForestPropio(semilla=SEMILLA)
    raise ValueError(f"modelo debe ser {MODELOS}, llegó: {modelo!r}")


def make_pipeline(modelo, smote, memory=None):
    """Pipeline sin ajustar. smote=False -> sin compensación; smote=True -> con SMOTE.

    memory: carpeta de caché para scaler, selector y sampler. Así el selector KS se ajusta
    una sola vez por fold y se reutiliza en todas las combinaciones de hiperparámetros.
    """
    if modelo not in MODELOS:
        raise ValueError(f"modelo debe ser {MODELOS}, llegó: {modelo!r}")
    return Pipeline([
        ("scaler", _crear_scaler(modelo, smote)),
        ("selector", SelectorKS()),
        ("sampler", _crear_sampler(smote)),
        ("clf", _crear_clasificador(modelo)),
    ], memory=memory)