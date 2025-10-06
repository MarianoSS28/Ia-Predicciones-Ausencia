import pandas as pd
import numpy as np

def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalizar nombres de columnas
    df.columns = [c.strip().lower() for c in df.columns]

    # Convertir fecha a datetime
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce', dayfirst=True)

    # Crear columnas derivadas
    df['dia_semana'] = df['fecha'].dt.dayofweek  # 0=lunes, 6=domingo

    # Convertir horas a datetime.time y calcular diferencias (en minutos)
    def time_diff(row):
        try:
            entrada_teo = pd.to_datetime(row['hora_entrada_teorica']).time()
            entrada_real = pd.to_datetime(row['hora_entrada_real']).time()
            diff = (
                pd.to_datetime(str(entrada_real)) - pd.to_datetime(str(entrada_teo))
            ).total_seconds() / 60
            return diff
        except:
            return np.nan

    df['tardanza_min'] = df.apply(time_diff, axis=1)

    # ✅ NUEVO: Convertir columna "ausencia" a clasificación multiclase
    # 0 = Presente
    # 1 = Ausente
    # 2 = Tardanza
    
    df['ausencia'] = df['ausencia'].fillna('Presente')
    df['ausencia'] = df['ausencia'].astype(str).str.strip().str.lower()
    
    def clasificar_ausencia(valor):
        if 'ausente' in valor:
            return 1  # Ausente
        elif 'tardanza' in valor or 'tarde' in valor:
            return 2  # Tardanza
        else:
            return 0  # Presente
    
    df['ausencia'] = df['ausencia'].apply(clasificar_ausencia)
    
    # ✅ Verificar la distribución
    print("\n📊 Distribución de clases:")
    print(df['ausencia'].value_counts().sort_index())
    print("\n📈 Porcentajes:")
    print(df['ausencia'].value_counts(normalize=True).sort_index() * 100)
    
    # Rellenar nulos numéricos con 0
    df = df.fillna(0)

    # Quitar columnas no útiles
    df = df.drop(columns=['nombre_empleado'], errors='ignore')

    return df

if __name__ == "__main__":
    data = load_and_clean_data("data/raw/fichajes.csv")
    
    print("\n✅ Primeras filas del dataset procesado:")
    print(data.head(10))
    
    print("\n✅ Columnas finales:")
    print(data.columns.tolist())
    
    data.to_csv("data/processed/empleados_clean.csv", index=False)
    print("\n💾 Archivo guardado: data/processed/empleados_clean.csv")