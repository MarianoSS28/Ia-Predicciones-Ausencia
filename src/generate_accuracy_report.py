import pandas as pd
import pickle
from datetime import datetime

def generate_accuracy_report(features_path: str, original_csv_path: str = "data/raw/fichajes.csv"):
    """
    Genera un reporte HTML comparando predicciones vs datos reales por fecha
    """
    # Cargar modelo
    with open("models/random_forest.pkl", "rb") as f:
        model = pickle.load(f)

    # Cargar datos originales con la columna de ausencia real
    df_original = pd.read_csv(original_csv_path)
    df_original.columns = [c.strip().lower() for c in df_original.columns]
    df_original['fecha'] = pd.to_datetime(df_original['fecha'], errors='coerce', dayfirst=True)
    
    # Cargar features procesados
    df_features = pd.read_csv(features_path)
    
    # Obtener ausencias reales del dataset original
    # Convertir ausencias a clasificación multiclase
    df_original['ausencia_real'] = df_original['ausencia'].fillna('Presente')
    df_original['ausencia_real'] = df_original['ausencia_real'].astype(str).str.strip().str.lower()
    
    def clasificar_ausencia(valor):
        if 'ausente' in valor:
            return 1  # Ausente
        elif 'tardanza' in valor or 'tarde' in valor:
            return 2  # Tardanza
        else:
            return 0  # Presente
    
    df_original['ausencia_real'] = df_original['ausencia_real'].apply(clasificar_ausencia)
    
    # Hacer predicciones
    X = df_features.drop(columns=["ausencia"], errors='ignore')
    predicciones = model.predict(X)
    
    # Combinar predicciones con datos originales
    df_comparacion = pd.DataFrame({
        'fecha': df_original['fecha'],
        'ausencia_real': df_original['ausencia_real'],
        'ausencia_predicha': predicciones
    })
    
    # Calcular métricas por fecha
    metricas_por_fecha = df_comparacion.groupby('fecha').apply(lambda x: pd.Series({
        'predicciones_acertadas': (x['ausencia_real'] == x['ausencia_predicha']).sum(),
        'total_ausencias_reales': (x['ausencia_real'] == 1).sum(),
        'total_registros': len(x)
    })).reset_index()
    
    # Calcular tasa de predicción acertada (TPA)
    metricas_por_fecha['tpa'] = (metricas_por_fecha['predicciones_acertadas'] / 
                                   metricas_por_fecha['total_registros'] * 100)
    
    # Formatear fecha
    metricas_por_fecha['fecha_str'] = metricas_por_fecha['fecha'].dt.strftime('%d/%m/%Y')
    
    # Ordenar por fecha
    metricas_por_fecha = metricas_por_fecha.sort_values('fecha')
    
    # Calcular métricas globales
    total_predicciones = metricas_por_fecha['total_registros'].sum()
    total_acertadas = metricas_por_fecha['predicciones_acertadas'].sum()
    tpa_global = (total_acertadas / total_predicciones * 100) if total_predicciones > 0 else 0
    total_ausencias_reales = metricas_por_fecha['total_ausencias_reales'].sum()
    
    # Generar HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Reporte de Precisión de Predicciones</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                min-height: 100vh;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .header h1 {{
                font-size: 36px;
                margin-bottom: 10px;
            }}
            .header p {{
                opacity: 0.9;
                font-size: 14px;
            }}
            .summary {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 20px;
                padding: 30px;
                background: #f8f9fa;
            }}
            .card {{
                background: white;
                padding: 25px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                text-align: center;
                transition: transform 0.3s;
            }}
            .card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 12px rgba(0,0,0,0.15);
            }}
            .card h3 {{
                font-size: 13px;
                color: #7f8c8d;
                margin-bottom: 15px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .card .value {{
                font-size: 42px;
                font-weight: bold;
                margin-bottom: 10px;
            }}
            .card .percentage {{
                font-size: 16px;
                color: #95a5a6;
            }}
            .content {{
                padding: 30px;
            }}
            h2 {{
                color: #2c3e50;
                margin: 30px 0 20px 0;
                padding-bottom: 10px;
                border-bottom: 3px solid #3498db;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border-radius: 8px;
                overflow: hidden;
            }}
            th {{
                background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
                color: white;
                padding: 15px;
                text-align: left;
                font-weight: 600;
                text-transform: uppercase;
                font-size: 12px;
                letter-spacing: 1px;
            }}
            td {{
                padding: 12px 15px;
                border-bottom: 1px solid #ecf0f1;
            }}
            tr:hover {{
                background-color: #f8f9fa;
            }}
            tr:last-child td {{
                border-bottom: none;
            }}
            .high-accuracy {{
                background-color: #d4edda !important;
            }}
            .medium-accuracy {{
                background-color: #fff3cd !important;
            }}
            .low-accuracy {{
                background-color: #f8d7da !important;
            }}
            .accuracy-bar {{
                height: 8px;
                background: #ecf0f1;
                border-radius: 4px;
                overflow: hidden;
                margin-top: 5px;
            }}
            .accuracy-fill {{
                height: 100%;
                transition: width 0.3s;
            }}
            .accuracy-fill-green {{
                background: linear-gradient(90deg, #2ecc71, #27ae60);
            }}
            .accuracy-fill-yellow {{
                background: linear-gradient(90deg, #f39c12, #e67e22);
            }}
            .accuracy-fill-red {{
                background: linear-gradient(90deg, #e74c3c, #c0392b);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 Reporte de Precisión de Predicciones</h1>
                <p>Comparación de Predicciones vs Datos Reales por Fecha</p>
                <p>Generado: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
            </div>

            <div class="summary">
                <div class="card">
                    <h3>Precisión Global</h3>
                    <div class="value" style="color: #3498db;">{tpa_global:.1f}%</div>
                    <div class="percentage">{total_acertadas:,} de {total_predicciones:,}</div>
                </div>
                <div class="card">
                    <h3>Total Predicciones</h3>
                    <div class="value" style="color: #34495e;">{total_predicciones:,}</div>
                </div>
                <div class="card">
                    <h3>Predicciones Acertadas</h3>
                    <div class="value" style="color: #27ae60;">{total_acertadas:,}</div>
                </div>
                <div class="card">
                    <h3>Total Ausencias Reales</h3>
                    <div class="value" style="color: #e74c3c;">{total_ausencias_reales:,}</div>
                </div>
            </div>

            <div class="content">
                <h2>📊 Precisión de Predicciones por Fecha</h2>
                <p style="color: #7f8c8d; margin-bottom: 15px; font-size: 14px;">
                    <strong>PA = Predicciones Acertadas</strong> | <strong>TA = Total Ausencias Reales</strong> | <strong>TPA = Tasa de Predicción Acertada</strong>
                </p>
                <table>
                    <thead>
                        <tr>
                            <th>Fecha</th>
                            <th>Predicciones Acertadas (PA)</th>
                            <th>Total de Ausencias Reales (TA)</th>
                            <th>Tasa de Predicción Acertada (TPA)</th>
                        </tr>
                    </thead>
                    <tbody>
    """

    for _, row in metricas_por_fecha.iterrows():
        tpa = row['tpa']
        
        # Determinar clase de fila y color de barra
        if tpa >= 90:
            row_class = "high-accuracy"
            bar_color = "accuracy-fill-green"
        elif tpa >= 75:
            row_class = "high-accuracy"
            bar_color = "accuracy-fill-green"
        elif tpa >= 60:
            row_class = "medium-accuracy"
            bar_color = "accuracy-fill-yellow"
        else:
            row_class = "low-accuracy"
            bar_color = "accuracy-fill-red"
        
        html_content += f"""
                        <tr class="{row_class}">
                            <td><strong>{row['fecha_str']}</strong></td>
                            <td style="text-align: center;"><strong style="color: #27ae60; font-size: 16px;">{int(row['predicciones_acertadas'])}</strong></td>
                            <td style="text-align: center;"><strong style="color: #e74c3c; font-size: 16px;">{int(row['total_ausencias_reales'])}</strong></td>
                            <td>
                                <strong style="font-size: 18px;">{tpa:.1f}%</strong>
                                <div class="accuracy-bar">
                                    <div class="accuracy-fill {bar_color}" style="width: {min(tpa, 100)}%"></div>
                                </div>
                            </td>
                        </tr>
        """

    html_content += f"""
                    </tbody>
                    <tfoot style="background: #f8f9fa; font-weight: bold;">
                        <tr>
                            <td style="padding: 15px;"><strong>TOTAL</strong></td>
                            <td style="text-align: center;"><strong style="color: #27ae60; font-size: 16px;">{total_acertadas:,}</strong></td>
                            <td style="text-align: center;"><strong style="color: #e74c3c; font-size: 16px;">{total_ausencias_reales:,}</strong></td>
                            <td>
                                <strong style="font-size: 18px;">Promedio: {tpa_global:.1f}%</strong>
                            </td>
                        </tr>
                    </tfoot>
                </table>
            </div>
        </div>
    </body>
    </html>
    """

    # Guardar archivos
    import os
    os.makedirs("reports", exist_ok=True)
    
    with open("reports/reporte_precision.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    metricas_por_fecha.to_csv("data/processed/metricas_precision_por_fecha.csv", index=False)

    print("\n✅ Reporte de precisión generado:")
    print("   📄 HTML: reports/reporte_precision.html")
    print("   📊 CSV:  data/processed/metricas_precision_por_fecha.csv")
    print(f"\n🎯 Precisión Global del Modelo: {tpa_global:.2f}%")
    print(f"   ✅ Predicciones Acertadas: {total_acertadas:,}")
    print(f"   📊 Total Predicciones: {total_predicciones:,}")
    print(f"   🔴 Total Ausencias Reales: {total_ausencias_reales:,}")
    print("\n💡 Abre el archivo HTML en tu navegador para ver el reporte visual completo")

if __name__ == "__main__":
    import os
    os.makedirs("reports", exist_ok=True)
    generate_accuracy_report("data/processed/empleados_features.csv", "data/raw/fichajes.csv")