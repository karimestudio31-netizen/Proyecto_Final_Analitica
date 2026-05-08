import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Configuración del Dataset
COLS = ['ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity',
        'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS',
        'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc',
        'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD',
        'Meth', 'Mush', 'Nicotine', 'Semer', 'VSA']

FEATURE_COLS = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity',
                'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']

# Las 4 sustancias escogidas
TARGETS = {
    'Cannabis':  'Cannabis',
    'Nicotina':  'Nicotine',
    'Extasis':   'Ecstasy',
    'Cocaina':   'Coke',
}


def cargar_datos():
    return pd.read_csv('drug_consumption.data', header=None, names=COLS)


def obtener_features(dataset):
    return dataset[FEATURE_COLS].values


def crear_variable_objetivo(dataset, target_col):
    """Binariza: CL0/CL1 → 0 (No usuario), CL2-CL6 → 1 (Usuario)."""
    return dataset[target_col].apply(lambda x: 0 if x in ['CL0', 'CL1'] else 1).values


# Entrenamiento
print("  Entrenamiento: Drug Personality Predictor")

dataset = cargar_datos()
X = obtener_features(dataset)

for nombre, col in TARGETS.items():
    print(f"  Sustancia: {nombre}  (columna: {col})")

    y = crear_variable_objetivo(dataset, col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # ── Red Neuronal con búsqueda de hiperparámetros ────────
    print("  [MLP] Buscando mejores hiperparámetros...")
    gs = RandomizedSearchCV(
        MLPClassifier(max_iter=15000, random_state=42),
        {
            "hidden_layer_sizes": [(8,), (16,), (32,), (64,), (32, 16)],
            "activation": ["relu", "tanh"],
            "alpha": [0.001, 0.01, 0.1],
        },
        n_iter=12, cv=3, n_jobs=-1, scoring='accuracy', random_state=42
    )
    gs.fit(X_tr, y_train)
    best_mlp = gs.best_estimator_
    pred_mlp = best_mlp.predict(X_te)
    acc_mlp = accuracy_score(y_test, pred_mlp)
    print(f"  [MLP] Mejor config: {gs.best_params_}")
    print(f"  [MLP] Accuracy: {acc_mlp:.4f}")

    # ── Regresión Logística ─────────────────────────────────
    print("  [LR]  Entrenando Regresión Logística...")
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_tr, y_train)
    pred_lr = logreg.predict(X_te)
    acc_lr = accuracy_score(y_test, pred_lr)
    print(f"  [LR]  Accuracy: {acc_lr:.4f}")

    # ── Guardar artefactos ─────────────────────────────────
    joblib.dump(scaler,   f"scaler_{nombre.lower()}.pkl")
    joblib.dump(best_mlp, f"mlp_{nombre.lower()}.pkl")
    joblib.dump(logreg,   f"logreg_{nombre.lower()}.pkl")
    print(f"  Artefactos guardados para {nombre}.")

print("\n✅ ¡Entrenamiento completado para las 4 sustancias!")
