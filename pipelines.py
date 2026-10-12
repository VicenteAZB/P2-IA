"""
pipelines.py — Factory de Pipelines (solo definición, sin ajustar).

Dos tipos para comparar: "ros" (RandomOverSampler) y "smote" (SMOTE).
Cada tipo trae 8 pipelines: 4 modelos x {sin, con} compensación.

Orden: scaler -> selector -> sampler -> clf.
Todo se ajusta dentro del Pipeline (solo con el train de cada fold),
así no hay fuga. Se usa imblearn.Pipeline porque el sampler solo
debe tocar el train, no la validación.

Escalado solo en kNN/SVM (usan distancias). En Árbol/RF es passthrough.
CON usa solo sampler, sin class_weight (evita doble compensación).
"""
from functools import partial

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from datos import SEMILLA
from random_forest_propio import RandomForestPropio

# --- Constantes ---
MODELOS = ("knn", "arbol", "svm", "rf_propio")
TIPOS_SAMPLER = ("ros", "smote")
SELECTORES = ("kbest", "from_model")
K_DEFECTO = 15  # variables de SelectKBest (se buscará en el Grid futuro)

# mutual_info con semilla fija (reproducible)
_mi_con_semilla = partial(mutual_info_classif, random_state=SEMILLA)


def _crear_scaler(modelo):
    """StandardScaler solo en kNN/SVM; resto passthrough."""
    if modelo in ("knn", "svm"):
        return StandardScaler()
    return "passthrough"


def _crear_selector(selector="kbest", k=K_DEFECTO):
    """SelectKBest(mutual_info) o SelectFromModel con árbol base."""
    if selector == "kbest":
        return SelectKBest(score_func=_mi_con_semilla, k=k)
    if selector == "from_model":
        base = DecisionTreeClassifier(random_state=SEMILLA)
        return SelectFromModel(estimator=base)
    raise ValueError(f"selector debe ser {SELECTORES}, llegó: {selector!r}")


def _crear_sampler(compensar, sampler_tipo="ros"):
    """SIN -> passthrough; CON -> ROS o SMOTE."""
    if not compensar:
        return "passthrough"
    if sampler_tipo == "ros":
        return RandomOverSampler(random_state=SEMILLA)
    if sampler_tipo == "smote":
        # ~48 fraudes por fold de CV, k=5 es seguro
        return SMOTE(random_state=SEMILLA, k_neighbors=5)
    raise ValueError(f"sampler_tipo debe ser {TIPOS_SAMPLER}, llegó: {sampler_tipo!r}")


def _crear_clasificador(modelo):
    """Clasificador con defaults (el ajuste fino va en el Grid futuro)."""
    if modelo == "knn":
        return KNeighborsClassifier()
    if modelo == "arbol":
        return DecisionTreeClassifier(random_state=SEMILLA)
    if modelo == "svm":
        return SVC(random_state=SEMILLA)
    if modelo == "rf_propio":
        # impar: la votación nunca empata
        return RandomForestPropio(semilla=SEMILLA)
    raise ValueError(f"modelo debe ser {MODELOS}, llegó: {modelo!r}")


def make_pipeline(modelo, compensar, sampler_tipo="ros", selector="kbest", k=K_DEFECTO):
    """Arma un Pipeline sin ajustar. compensar=False -> condición SIN."""
    if modelo not in MODELOS:
        raise ValueError(f"modelo debe ser {MODELOS}, llegó: {modelo!r}")
    return Pipeline([
        ("scaler", _crear_scaler(modelo)),
        ("selector", _crear_selector(selector, k)),
        ("sampler", _crear_sampler(compensar, sampler_tipo)),
        ("clf", _crear_clasificador(modelo)),
    ])


def get_pipelines(sampler_tipo="ros", selector="kbest", k=K_DEFECTO):
    """Los 8 pipelines del tipo pedido, sin ajustar. Claves: {modelo}_{sin,con}."""
    if sampler_tipo not in TIPOS_SAMPLER:
        raise ValueError(f"sampler_tipo debe ser {TIPOS_SAMPLER}, llegó: {sampler_tipo!r}")
    pipes = {}
    for modelo in MODELOS:
        pipes[f"{modelo}_sin"] = make_pipeline(
            modelo, compensar=False, sampler_tipo=sampler_tipo, selector=selector, k=k
        )
        pipes[f"{modelo}_con"] = make_pipeline(
            modelo, compensar=True, sampler_tipo=sampler_tipo, selector=selector, k=k
        )
    return pipes


# 2 tipos listos para comparar y elegir uno.
PIPELINES_ROS = get_pipelines("ros")
PIPELINES_SMOTE = get_pipelines("smote")


def describir_pipelines(pipes):
    """Una línea por pipeline: nombre -> pasos activos (sin ajustar)."""
    lineas = []
    for nombre, pipe in pipes.items():
        pasos = []
        for paso, obj in pipe.steps:
            pasos.append(paso if isinstance(obj, str) else f"{paso}:{type(obj).__name__}")
        lineas.append(f"{nombre:15s} | {' -> '.join(pasos)}")
    return "\n".join(lineas)


if __name__ == "__main__":
    # Muestra la estructura sin ajustar (sin fuga posible).
    print(f"SEMILLA={SEMILLA} | K_DEFECTO={K_DEFECTO}\n")
    print("--- TIPO A: ROS ---")
    print(describir_pipelines(PIPELINES_ROS))
    print("\n--- TIPO B: SMOTE ---")
    print(describir_pipelines(PIPELINES_SMOTE))
