# Train_Model.py

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(input_path: str):
    # Cargar los datos
    df = pd.read_csv(input_path)

    # Separar características (X) y objetivo (y)
    X = df.drop(columns=["ausencia"])
    y = df["ausencia"]

    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Definir el modelo base
    model = RandomForestClassifier(random_state=42)

    # Definir las combinaciones de parámetros a probar
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"]
    }

    print("🔍 Iniciando búsqueda de hiperparámetros... esto puede tardar unos minutos")

    # Búsqueda con validación cruzada
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,               # 3 divisiones de validación
        scoring="accuracy", # métrica de evaluación
        n_jobs=-1,          # usar todos los núcleos disponibles
        verbose=2
    )

    # Entrenar todas las combinaciones
    grid.fit(X_train, y_train)

    # Obtener el mejor modelo
    best_model = grid.best_estimator_

    # Evaluar en el conjunto de prueba
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n✅ Mejor combinación de parámetros: {grid.best_params_}")
    print(f"🎯 Precisión en test: {acc:.4f}")

    # Guardar el mejor modelo
    with open("models/random_forest.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print("💾 Mejor modelo guardado en models/random_forest.pkl")

if __name__ == "__main__":
    train_model("data/processed/empleados_features.csv")
