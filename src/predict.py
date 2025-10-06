# Predict.py

import pandas as pd
import pickle

def predict_absences(input_path: str):
    # Cargar modelo
    with open("models/random_forest.pkl", "rb") as f:
        model = pickle.load(f)

    # Cargar datos
    df = pd.read_csv(input_path)

    # ✅ Eliminar la columna objetivo si existe
    if "ausencia" in df.columns:
        df = df.drop(columns=["ausencia"])

    # Hacer predicciones
    predictions = model.predict(df)

    # Guardar resultados
    output = pd.DataFrame(predictions, columns=["prediccion"])
    output.to_csv("data/processed/predicciones.csv", index=False)
    print("✅ Predicciones guardadas en data/processed/predicciones.csv")

if __name__ == "__main__":
    predict_absences("data/processed/empleados_features.csv")
