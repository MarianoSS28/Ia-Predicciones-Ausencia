#features.py

import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Asegurar que 'fecha' sea datetime
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce', dayfirst=True)

    # Crear features temporales
    df['mes'] = df['fecha'].dt.month
    df['anio'] = df['fecha'].dt.year
    df['dia_mes'] = df['fecha'].dt.day
    df['semana_anio'] = df['fecha'].dt.isocalendar().week
    
    # Features de días
    df['es_viernes'] = (df['dia_semana'] == 4).astype(int)
    df['es_lunes'] = (df['dia_semana'] == 0).astype(int)
    df['es_fin_semana'] = (df['dia_semana'] >= 5).astype(int)
    
    # Features de tardanzas
    df['tarde'] = (df['tardanza_min'] > 15).astype(int)
    df['muy_tarde'] = (df['tardanza_min'] > 30).astype(int)
    
    # ✅ ELIMINAR TODAS LAS COLUMNAS NO NUMÉRICAS
    columnas_a_eliminar = [
        'fecha',
        'hora_entrada_teorica',
        'hora_entrada_real',
        'hora_salida_teorica',
        'hora_salida_real',
        'nombre_empleado'  # ← Por si acaso quedó
    ]
    
    df = df.drop(columns=columnas_a_eliminar, errors='ignore')
    
    # ✅ VERIFICAR QUE SOLO QUEDEN COLUMNAS NUMÉRICAS
    print("\n🔍 Columnas finales en el dataset:")
    print(df.columns.tolist())
    print(f"\n📊 Tipos de datos:")
    print(df.dtypes)
    
    # ✅ CONVERTIR TODO A NUMÉRICO
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"⚠️  ADVERTENCIA: La columna '{col}' es de tipo texto, intentando convertir...")
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Rellenar NaN con 0
    df = df.fillna(0)
    
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/empleados_clean.csv")
    df = build_features(df)
    
    # ✅ VALIDACIÓN FINAL
    print("\n✅ Validación final:")
    print(f"   - Total de columnas: {len(df.columns)}")
    print(f"   - Total de filas: {len(df)}")
    print(f"   - Columnas no numéricas: {df.select_dtypes(include=['object']).columns.tolist()}")
    
    if len(df.select_dtypes(include=['object']).columns) > 0:
        print("\n❌ ERROR: Todavía hay columnas de texto!")
        print(df.select_dtypes(include=['object']).head())
    else:
        print("\n✅ PERFECTO: Solo columnas numéricas")
    
    df.to_csv("data/processed/empleados_features.csv", index=False)
    print("\n✅ Archivo generado: data/processed/empleados_features.csv")