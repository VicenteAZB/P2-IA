"""
random_forest_propio.py
Random Forest de implementación propia (bootstrap + bagging + votación).

Usa DecisionTreeClassifier solo como árbol base (criterio de entropía, como ID3). NO se usa
RandomForestClassifier ni BaggingClassifier: el ensamble (muestras bootstrap, entrenamiento
de cada árbol y votación) se construye aquí.
"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_array, check_X_y


def _entrenar_un_arbol(X, y, semilla_arbol, profundidad_maxima, max_features,
                       min_muestras_division):
    """Entrena UN árbol independiente con su propia muestra bootstrap."""
    generador = np.random.default_rng(semilla_arbol)

    # Muestra bootstrap: n índices con reemplazo (mismo tamaño que el conjunto de entrenamiento)
    n = X.shape[0]
    indices = generador.integers(0, n, size=n)

    arbol = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=profundidad_maxima,
        max_features=max_features,          # subconjunto aleatorio de variables EN CADA DIVISIÓN
        min_samples_split=min_muestras_division,
        random_state=semilla_arbol,         # semilla distinta por árbol
    )
    arbol.fit(X[indices], y[indices])
    return arbol


class RandomForestPropio(ClassifierMixin, BaseEstimator):
    """Random Forest propio.

    Parámetros configurables
    ------------------------
    n_arboles : número de árboles del bosque (usar un número impar).
    profundidad_maxima : profundidad máxima de cada árbol (None = sin límite).
    max_features : variables consideradas en cada división ("sqrt", "log2", int, float).
    semilla : semilla general; de ella se derivan las semillas de cada árbol.
    min_muestras_division : mínimo de muestras para dividir un nodo.
    n_jobs : procesos en paralelo para entrenar los árboles (-1 = todos los núcleos).

    Regla de desempate: si hay empate en la votación (solo posible con un número par
    de árboles), se elige la clase de mayor índice, es decir, el fraude (1). Así se
    respeta la prioridad del proyecto: minimizar los falsos negativos.
    """

    def __init__(self, n_arboles=101, profundidad_maxima=None, max_features="sqrt",
                 semilla=5040, min_muestras_division=2, n_jobs=-1):
        self.n_arboles = n_arboles
        self.profundidad_maxima = profundidad_maxima
        self.max_features = max_features
        self.semilla = semilla
        self.min_muestras_division = min_muestras_division
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Entrena el bosque: un árbol por cada muestra bootstrap."""
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        # Semillas distintas por árbol, derivadas de la semilla general (reproducible)
        generador = np.random.default_rng(self.semilla)
        semillas_arboles = generador.integers(0, 2**31 - 1, size=self.n_arboles)

        # Los árboles son independientes entre sí, por eso se entrenan en paralelo
        self.arboles_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_entrenar_un_arbol)(
                X, y, int(s), self.profundidad_maxima, self.max_features,
                self.min_muestras_division,
            )
            for s in semillas_arboles
        )
        return self

    def _contar_votos(self, X):
        """Matriz (n_muestras x n_clases) con los votos que recibe cada clase."""
        X = check_array(X)
        votos = np.zeros((X.shape[0], len(self.classes_)), dtype=int)
        for arbol in self.arboles_:
            predicciones = arbol.predict(X)
            columna = np.searchsorted(self.classes_, predicciones)  # etiqueta -> columna
            votos[np.arange(X.shape[0]), columna] += 1
        return votos

    def predict(self, X):
        """Votación mayoritaria. Empate -> clase de mayor índice (fraude)."""
        votos = self._contar_votos(X)
        # argmax sobre las columnas invertidas hace que el empate gane la última clase
        columna = len(self.classes_) - 1 - np.argmax(votos[:, ::-1], axis=1)
        return self.classes_[columna]

    def predict_proba(self, X):
        """Fracción de árboles que votó por cada clase."""
        return self._contar_votos(X) / len(self.arboles_)