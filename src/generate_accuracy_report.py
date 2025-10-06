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
    
    # ✅ MÉTRICAS MEJORADAS - Más claras y específicas
    metricas_por_fecha = df_comparacion.groupby('fecha').apply(lambda x: pd.Series({
        # Métricas generales
        'total_registros': len(x),
        'predicciones_correctas': (x['ausencia_real'] == x['ausencia_predicha']).sum(),
        
        # Métricas por clase
        'total_presentes_reales': (x['ausencia_real'] == 0).sum(),
        'total_ausentes_reales': (x['ausencia_real'] == 1).sum(),
        'total_tardanzas_reales': (x['ausencia_real'] == 2).sum(),
        
        # Predicciones acertadas por clase
        'presentes_acertados': ((x['ausencia_real'] == 0) & (x['ausencia_predicha'] == 0)).sum(),
        'ausentes_acertados': ((x['ausencia_real'] == 1) & (x['ausencia_predicha'] == 1)).sum(),
        'tardanzas_acertadas': ((x['ausencia_real'] == 2) & (x['ausencia_predicha'] == 2)).sum(),
        
        # Errores
        'falsos_positivos_ausencia': ((x['ausencia_real'] != 1) & (x['ausencia_predicha'] == 1)).sum(),
        'falsos_negativos_ausencia': ((x['ausencia_real'] == 1) & (x['ausencia_predicha'] != 1)).sum(),
    })).reset_index()
    
    # Calcular tasas de acierto
    metricas_por_fecha['tasa_global'] = (metricas_por_fecha['predicciones_correctas'] / 
                                          metricas_por_fecha['total_registros'] * 100)
    
    metricas_por_fecha['tasa_ausencias'] = (
        metricas_por_fecha['ausentes_acertados'] / 
        metricas_por_fecha['total_ausentes_reales'].replace(0, 1) * 100
    ).fillna(0)
    
    # Formatear fecha
    metricas_por_fecha['fecha_str'] = metricas_por_fecha['fecha'].dt.strftime('%d/%m/%Y')
    
    # Ordenar por fecha
    metricas_por_fecha = metricas_por_fecha.sort_values('fecha')
    
    # Calcular métricas globales
    total_registros = metricas_por_fecha['total_registros'].sum()
    total_correctas = metricas_por_fecha['predicciones_correctas'].sum()
    tasa_global = (total_correctas / total_registros * 100) if total_registros > 0 else 0
    
    total_ausentes_reales = metricas_por_fecha['total_ausentes_reales'].sum()
    ausentes_acertados = metricas_por_fecha['ausentes_acertados'].sum()
    tasa_ausencias_global = (ausentes_acertados / total_ausentes_reales * 100) if total_ausentes_reales > 0 else 0
    
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
                max-width: 1400px;
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
                font-size: 11px;
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
            .info-box {{
                background: #e8f4f8;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin: 20px 0;
                border-radius: 5px;
            }}
            .info-box strong {{
                color: #2c3e50;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 Reporte de Precisión de Predicciones</h1>
                <p>Análisis Detallado de Predicciones vs Datos Reales</p>
                <p>Generado: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
            </div>

            <div class="summary">
                <div class="card">
                    <h3>Precisión Global</h3>
                    <div class="value" style="color: #3498db;">{tasa_global:.1f}%</div>
                    <div class="percentage">{total_correctas:,} / {total_registros:,}</div>
                </div>
                <div class="card">
                    <h3>Total Registros</h3>
                    <div class="value" style="color: #34495e;">{total_registros:,}</div>
                </div>
                <div class="card">
                    <h3>Predicciones Correctas</h3>
                    <div class="value" style="color: #27ae60;">{total_correctas:,}</div>
                </div>
                <div class="card">
                    <h3>🔴 Precisión en Ausencias</h3>
                    <div class="value" style="color: #e74c3c;">{tasa_ausencias_global:.1f}%</div>
                    <div class="percentage">{ausentes_acertados:,} / {total_ausentes_reales:,}</div>
                </div>
            </div>

            <div class="content">
                <div class="info-box">
                    <p><strong>📊 Cómo leer este reporte:</strong></p>
                    <ul style="margin-left: 20px; margin-top: 10px; line-height: 1.8;">
                        <li><strong>Precisión Global:</strong> Porcentaje de TODAS las predicciones correctas (presentes, ausentes y tardanzas)</li>
                        <li><strong>Ausencias Detectadas:</strong> De todas las ausencias reales, cuántas detectó el modelo</li>
                        <li><strong>Tasa Global = Predicciones Correctas / Total Registros</strong></li>
                    </ul>
                </div>

                <h2>📊 Precisión de Predicciones por Fecha</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Fecha</th>
                            <th>Total Registros</th>
                            <th>✅ Correctas</th>
                            <th>🔴 Ausencias Reales</th>
                            <th>🎯 Ausencias Detectadas</th>
                            <th>Precisión Global</th>
                            <th>Precisión en Ausencias</th>
                        </tr>
                    </thead>
                    <tbody>
    """

    for _, row in metricas_por_fecha.iterrows():
        tasa = row['tasa_global']
        tasa_aus = row['tasa_ausencias']
        
        # Determinar clase de fila
        if tasa >= 90:
            row_class = "high-accuracy"
            bar_color = "accuracy-fill-green"
        elif tasa >= 75:
            row_class = "high-accuracy"
            bar_color = "accuracy-fill-green"
        elif tasa >= 60:
            row_class = "medium-accuracy"
            bar_color = "accuracy-fill-yellow"
        else:
            row_class = "low-accuracy"
            bar_color = "accuracy-fill-red"
        
        ausentes_reales = int(row['total_ausentes_reales'])
        ausentes_detectados = int(row['ausentes_acertados'])
        
        html_content += f"""
                        <tr class="{row_class}">
                            <td><strong>{row['fecha_str']}</strong></td>
                            <td style="text-align: center;">{int(row['total_registros'])}</td>
                            <td style="text-align: center;"><strong style="color: #27ae60;">{int(row['predicciones_correctas'])}</strong></td>
                            <td style="text-align: center;"><strong style="color: #e74c3c;">{ausentes_reales}</strong></td>
                            <td style="text-align: center;">
                                <strong style="color: #3498db;">{ausentes_detectados}</strong> / {ausentes_reales}
                            </td>
                            <td>
                                <strong style="font-size: 16px;">{tasa:.1f}%</strong>
                                <div class="accuracy-bar">
                                    <div class="accuracy-fill {bar_color}" style="width: {min(tasa, 100)}%"></div>
                                </div>
                            </td>
                            <td>
                                <strong style="font-size: 16px; color: {'#27ae60' if tasa_aus >= 80 else '#f39c12' if tasa_aus >= 60 else '#e74c3c'};">{tasa_aus:.1f}%</strong>
                                <div class="accuracy-bar">
                                    <div class="accuracy-fill {'accuracy-fill-green' if tasa_aus >= 80 else 'accuracy-fill-yellow' if tasa_aus >= 60 else 'accuracy-fill-red'}" style="width: {min(tasa_aus, 100)}%"></div>
                                </div>
                            </td>
                        </tr>
        """

    html_content += f"""
                    </tbody>
                    <tfoot style="background: #f8f9fa; font-weight: bold;">
                        <tr>
                            <td style="padding: 15px;"><strong>TOTAL</strong></td>
                            <td style="text-align: center;"><strong>{total_registros:,}</strong></td>
                            <td style="text-align: center;"><strong style="color: #27ae60;">{total_correctas:,}</strong></td>
                            <td style="text-align: center;"><strong style="color: #e74c3c;">{total_ausentes_reales:,}</strong></td>
                            <td style="text-align: center;"><strong style="color: #3498db;">{ausentes_acertados:,}</strong></td>
                            <td><strong>{tasa_global:.1f}%</strong></td>
                            <td><strong>{tasa_ausencias_global:.1f}%</strong></td>
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
    print(f"\n🎯 Métricas Globales:")
    print(f"   ✅ Precisión Global: {tasa_global:.2f}% ({total_correctas:,}/{total_registros:,})")
    print(f"   🔴 Precisión en Ausencias: {tasa_ausencias_global:.2f}% ({ausentes_acertados:,}/{total_ausentes_reales:,})")
    print("\n💡 Abre el archivo HTML en tu navegador para ver el reporte visual completo")

if __name__ == "__main__":
    import os
    os.makedirs("reports", exist_ok=True)
    generate_accuracy_report("data/processed/empleados_features.csv", "data/raw/fichajes.csv")