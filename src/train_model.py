import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

def train_model(input_path: str):
    # Cargar los datos
    df = pd.read_csv(input_path)

    # Separar características (X) y objetivo (y)
    X = df.drop(columns=["ausencia"])
    y = df["ausencia"]
    
    # ✅ Verificar que tenemos 3 clases
    print("\n📊 Distribución de clases en el dataset:")
    print(y.value_counts().sort_index())
    print("\n📈 Porcentajes:")
    print(y.value_counts(normalize=True).sort_index() * 100)
    
    clases_unicas = y.unique()
    print(f"\n✅ Clases detectadas: {sorted(clases_unicas)}")
    
    if len(clases_unicas) < 3:
        print("⚠️  ADVERTENCIA: Solo se detectaron", len(clases_unicas), "clases.")
        print("   Deberías tener 3 clases: 0 (Presente), 1 (Ausente), 2 (Tardanza)")
        print("   Verifica que preprocess.py esté clasificando correctamente.")
    
    # Dividir en train/test estratificado (mantiene proporción de clases)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📦 Conjunto de entrenamiento: {len(X_train)} registros")
    print(f"📦 Conjunto de prueba: {len(X_test)} registros")

    # Definir el modelo base
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    # ✅ Parámetros optimizados para clasificación multiclase
    param_grid = {
        "n_estimators": [200, 300, 500],
        "max_depth": [15, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"],
        "max_features": ["sqrt", "log2", None]
    }

    print("\n🔍 Iniciando búsqueda de hiperparámetros...")
    print("   Esto puede tardar varios minutos...")

    # Búsqueda con validación cruzada estratificada
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,  # 5 divisiones
        scoring="f1_weighted",  # F1 ponderado para clases desbalanceadas
        n_jobs=-1,
        verbose=2
    )

    # Entrenar todas las combinaciones
    grid.fit(X_train, y_train)

    # Obtener el mejor modelo
    best_model = grid.best_estimator_

    # Evaluar en el conjunto de prueba
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n✅ Mejor combinación de parámetros:")
    for param, value in grid.best_params_.items():
        print(f"   {param}: {value}")
    
    print(f"\n🎯 Precisión en test: {acc:.4f}")
    
    # ✅ Reporte detallado de clasificación
    print("\n📊 Reporte de Clasificación:")
    print("=" * 60)
    class_names = ['Presente', 'Ausente', 'Tardanza']
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # ✅ Matriz de confusión
    print("\n📊 Matriz de Confusión:")
    print("=" * 60)
    cm = confusion_matrix(y_test, y_pred)
    print("                Predicho")
    print("                Presente  Ausente  Tardanza")
    for i, row in enumerate(cm):
        print(f"Real {class_names[i]:10s} {row[0]:8d} {row[1]:8d} {row[2]:8d}")
    
    # ✅ Importancia de características
    print("\n📊 Top 10 características más importantes:")
    print("=" * 60)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']:20s}: {row['importance']:.4f}")

    # Guardar el mejor modelo
    with open("models/random_forest.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print("\n💾 Mejor modelo guardado en models/random_forest.pkl")
    
    # ✅ Verificar que el modelo predice las 3 clases
    clases_predichas = np.unique(y_pred)
    print(f"\n✅ Clases predichas en test: {sorted(clases_predichas)}")
    
    if len(clases_predichas) < 3:
        print("⚠️  ADVERTENCIA: El modelo solo está prediciendo", len(clases_predichas), "clases")
        print("   Posibles causas:")
        print("   - Datos muy desbalanceados")
        print("   - Features no suficientemente discriminativas")
        print("   - Necesitas más datos de la clase minoritaria")

if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    train_model("data/processed/empleados_features.csv")