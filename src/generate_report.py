import pandas as pd
import pickle
from datetime import datetime

def generate_html_report(input_path: str, original_csv_path: str = "data/raw/fichajes.csv"):
    print("📊 Iniciando generación de reporte...")
    
    # Cargar modelo y datos
    print("   Cargando modelo...")
    with open("models/random_forest.pkl", "rb") as f:
        model = pickle.load(f)

    # ✅ Cargar datos original con nombres
    print("   Cargando datos originales...")
    df_original = pd.read_csv(original_csv_path)
    df_original.columns = [c.strip().lower() for c in df_original.columns]
    
    # Convertir fecha correctamente
    df_original['fecha'] = pd.to_datetime(df_original['fecha'], errors='coerce', dayfirst=True)
    
    if df_original['fecha'].isna().any():
        print(f"⚠️  Advertencia: {df_original['fecha'].isna().sum()} fechas inválidas encontradas")
        df_original['fecha'] = df_original['fecha'].fillna(pd.Timestamp.now())
    
    # Cargar features procesados
    print("   Cargando features...")
    df = pd.read_csv(input_path)

    if "ausencia" in df.columns:
        X = df.drop(columns=["ausencia"])
    else:
        X = df

    # Predicciones
    print("   Realizando predicciones...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Mapeo de días
    dias_map = {0: 'Lun', 1: 'Mar', 2: 'Mié', 3: 'Jue', 4: 'Vie', 5: 'Sáb', 6: 'Dom'}

    # Crear DataFrame con resultados
    print("   Creando DataFrame de resultados...")
    reporte = pd.DataFrame({
        'empleado_id': df_original['empleado_id'].fillna('Sin ID').astype(str),
        'nombre_empleado': df_original['nombre_empleado'].fillna('Sin nombre'),
        'fecha': df_original['fecha'],
        'fecha_str': df_original['fecha'].dt.strftime('%d/%m/%Y'),
        'dia_semana': df['dia_semana'].fillna(0).astype(int).map(dias_map),
        'mes': df['mes'].fillna(0).astype(int),
        'anio': df_original['fecha'].dt.year,
        'tardanza_min': df['tardanza_min'].fillna(0),
        'prediccion': predictions,
        'prob_presente': probabilities[:, 0],
        'prob_tardanza': probabilities[:, 1]  # ✅ Solo 2 clases
    })

    # ✅ CALCULAR PROBABILIDAD MENSUAL POR EMPLEADO
    print("   Calculando probabilidades mensuales...")
    reporte_mensual = reporte.groupby(['empleado_id', 'nombre_empleado', 'mes', 'anio']).agg({
        'prob_tardanza': 'mean',
        'prob_presente': 'mean',
        'prediccion': ['count', lambda x: (x == 1).sum()]
    }).reset_index()
    
    reporte_mensual.columns = ['empleado_id', 'nombre_empleado', 'mes', 'anio', 
                                'prob_tardanza_promedio', 'prob_asistencia_promedio', 
                                'total_dias', 'dias_tardanza_predichos']
    reporte_mensual['prob_tardanza_promedio'] = reporte_mensual['prob_tardanza_promedio'] * 100
    reporte_mensual['prob_asistencia_promedio'] = reporte_mensual['prob_asistencia_promedio'] * 100
    reporte_mensual = reporte_mensual.sort_values('prob_tardanza_promedio', ascending=False)

    # Mapeo de meses
    meses_map = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
                 7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
    reporte_mensual['mes_nombre'] = reporte_mensual['mes'].map(meses_map)

    # Estadísticas
    total = len(reporte)
    presentes = (predictions == 0).sum()
    tardanzas = (predictions == 1).sum()

    print("   Generando HTML...")
    
    # ✅ OPTIMIZACIÓN: Usar list comprehension en lugar de concatenación de strings
    html_parts = []
    
    html_parts.append(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Reporte de Predicción de Asistencia</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
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
            .header h1 {{ font-size: 36px; margin-bottom: 10px; }}
            .header p {{ opacity: 0.9; font-size: 14px; }}
            .summary {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
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
                font-size: 14px;
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
            .card .percentage {{ font-size: 16px; color: #95a5a6; }}
            .content {{ padding: 30px; }}
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
            td {{ padding: 12px 15px; border-bottom: 1px solid #ecf0f1; }}
            tr:hover {{ background-color: #f8f9fa; }}
            tr:last-child td {{ border-bottom: none; }}
            .badge {{
                padding: 6px 12px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 11px;
                text-transform: uppercase;
                display: inline-block;
            }}
            .badge-presente {{ background: #d4edda; color: #155724; }}
            .badge-tardanza {{ background: #fff3cd; color: #856404; }}
            .high-risk {{ background-color: #fff5f5 !important; }}
            .prob-bar {{
                height: 8px;
                background: #ecf0f1;
                border-radius: 4px;
                overflow: hidden;
                margin-top: 5px;
            }}
            .prob-fill {{ height: 100%; transition: width 0.3s; }}
            .prob-fill-yellow {{ background: linear-gradient(90deg, #f39c12, #e67e22); }}
            .prob-fill-green {{ background: linear-gradient(90deg, #2ecc71, #27ae60); }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Reporte de Predicción de Asistencia</h1>
                <p>Generado: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
            </div>

            <div class="summary">
                <div class="card">
                    <h3>Total Registros</h3>
                    <div class="value" style="color: #3498db;">{total:,}</div>
                </div>
                <div class="card">
                    <h3>🟢 Presentes</h3>
                    <div class="value" style="color: #27ae60;">{presentes:,}</div>
                    <div class="percentage">{presentes/total*100:.1f}%</div>
                </div>
                <div class="card">
                    <h3>🟡 Tardanzas</h3>
                    <div class="value" style="color: #f39c12;">{tardanzas:,}</div>
                    <div class="percentage">{tardanzas/total*100:.1f}%</div>
                </div>
            </div>

            <div class="content">
                <h2>📅 Probabilidad Mensual por Empleado (Top 50)</h2>
                <p style="color: #7f8c8d; margin-bottom: 15px; font-size: 14px;">
                    📊 Top 50 empleados con mayor probabilidad de tardanza mensual
                </p>
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Nombre Empleado</th>
                            <th>Mes</th>
                            <th>Año</th>
                            <th>🟢 Prob. Asistencia</th>
                            <th>🟡 Prob. Tardanza</th>
                            <th>Días Laborales</th>
                            <th>Tardanzas Predichas</th>
                            <th>Nivel de Riesgo</th>
                        </tr>
                    </thead>
                    <tbody>
    """)

    # ✅ OPTIMIZACIÓN: Solo mostrar top 50 empleados
    for _, row in reporte_mensual.head(50).iterrows():
        prob_tardanza_pct = row['prob_tardanza_promedio']
        prob_asistencia_pct = row['prob_asistencia_promedio']
        
        if prob_tardanza_pct >= 60:
            nivel_riesgo = "🔥 ALTO"
            row_class = "high-risk"
        elif prob_tardanza_pct >= 40:
            nivel_riesgo = "⚠️ MEDIO"
            row_class = ""
        else:
            nivel_riesgo = "✅ BAJO"
            row_class = ""
        
        empleado_id_corto = str(row['empleado_id'])[:8] + "..." if len(str(row['empleado_id'])) > 8 else str(row['empleado_id'])
        
        html_parts.append(f"""
                        <tr class="{row_class}">
                            <td><strong>{empleado_id_corto}</strong></td>
                            <td><strong>{row['nombre_empleado']}</strong></td>
                            <td>{row['mes_nombre']}</td>
                            <td>{int(row['anio'])}</td>
                            <td>
                                <strong style="color: #27ae60; font-size: 16px;">{prob_asistencia_pct:.1f}%</strong>
                                <div class="prob-bar">
                                    <div class="prob-fill prob-fill-green" style="width: {min(prob_asistencia_pct, 100)}%"></div>
                                </div>
                            </td>
                            <td>
                                <strong style="color: #f39c12; font-size: 16px;">{prob_tardanza_pct:.1f}%</strong>
                                <div class="prob-bar">
                                    <div class="prob-fill prob-fill-yellow" style="width: {min(prob_tardanza_pct, 100)}%"></div>
                                </div>
                            </td>
                            <td><strong>{int(row['total_dias'])}</strong> días</td>
                            <td><strong style="color: #f39c12;">{int(row['dias_tardanza_predichos'])}</strong> días</td>
                            <td><strong>{nivel_riesgo}</strong></td>
                        </tr>
        """)

    html_parts.append("""
                    </tbody>
                </table>

                <h2>🟡 Empleados con Mayor Riesgo de Tardanza (Top 30 Días)</h2>
    """)

    # ✅ OPTIMIZACIÓN: Solo top 30 tardanzas
    tardanzas_pred = reporte[reporte['prob_tardanza'] > 0.5].sort_values('prob_tardanza', ascending=False).head(30)
    
    if len(tardanzas_pred) > 0:
        html_parts.append("""
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Nombre</th>
                            <th>Fecha</th>
                            <th>Día</th>
                            <th>Predicción</th>
                            <th>Probabilidad Tardanza</th>
                            <th>Minutos de Tardanza</th>
                        </tr>
                    </thead>
                    <tbody>
        """)
        
        for _, row in tardanzas_pred.iterrows():
            pred = int(row['prediccion'])
            badge_class = 'badge-presente' if pred == 0 else 'badge-tardanza'
            badge_text = 'PRESENTE' if pred == 0 else 'TARDANZA'
            prob_pct = row['prob_tardanza'] * 100
            tardanza = row['tardanza_min']
            empleado_id_corto = str(row['empleado_id'])[:8] + "..." if len(str(row['empleado_id'])) > 8 else str(row['empleado_id'])
            
            html_parts.append(f"""
                            <tr>
                                <td><strong>{empleado_id_corto}</strong></td>
                                <td><strong>{row['nombre_empleado']}</strong></td>
                                <td>{row['fecha_str']}</td>
                                <td>{row['dia_semana']}</td>
                                <td><span class="badge {badge_class}">{badge_text}</span></td>
                                <td>
                                    <strong>{prob_pct:.1f}%</strong>
                                    <div class="prob-bar">
                                        <div class="prob-fill prob-fill-yellow" style="width: {prob_pct}%"></div>
                                    </div>
                                </td>
                                <td><strong>{tardanza:.0f}</strong> min</td>
                            </tr>
            """)
        
        html_parts.append("""
                    </tbody>
                </table>
        """)
    else:
        html_parts.append("<p>✅ No hay tardanzas de alto riesgo (>50%)</p>")

    html_parts.append("""
            </div>
        </div>
    </body>
    </html>
    """)

    # ✅ OPTIMIZACIÓN: Join una sola vez al final
    html_content = "".join(html_parts)

    # Guardar archivos
    import os
    os.makedirs("reports", exist_ok=True)
    
    print("   Guardando archivo HTML...")
    with open("reports/reporte_ausencias.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print("   Guardando CSVs...")
    reporte.to_csv("data/processed/predicciones_detalladas.csv", index=False)
    reporte_mensual.to_csv("data/processed/probabilidad_mensual_empleados.csv", index=False)

    print("\n✅ Reportes generados:")
    print("   📄 HTML: reports/reporte_ausencias.html")
    print("   📊 CSV Detallado:  data/processed/predicciones_detalladas.csv")
    print("   📊 CSV Mensual:    data/processed/probabilidad_mensual_empleados.csv")
    print("\n💡 Abre el archivo HTML en tu navegador para ver el reporte visual")

if __name__ == "__main__":
    import os
    os.makedirs("reports", exist_ok=True)
    generate_html_report("data/processed/empleados_features.csv", "data/raw/fichajes.csv")