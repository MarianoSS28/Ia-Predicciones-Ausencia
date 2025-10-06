import pandas as pd
import pickle
from datetime import datetime

def generate_html_report(input_path: str, original_csv_path: str = "data/raw/fichajes.csv"):
    # Cargar modelo y datos
    with open("models/random_forest.pkl", "rb") as f:
        model = pickle.load(f)

    # ✅ Cargar datos original con nombres
    df_original = pd.read_csv(original_csv_path)
    df_original.columns = [c.strip().lower() for c in df_original.columns]
    
    # Convertir fecha correctamente
    df_original['fecha'] = pd.to_datetime(df_original['fecha'], errors='coerce', dayfirst=True)
    
    # Si hay fechas inválidas, usar fecha actual como placeholder
    if df_original['fecha'].isna().any():
        print(f"⚠️  Advertencia: {df_original['fecha'].isna().sum()} fechas inválidas encontradas")
        df_original['fecha'] = df_original['fecha'].fillna(pd.Timestamp.now())
    
    # Cargar features procesados
    df = pd.read_csv(input_path)

    if "ausencia" in df.columns:
        X = df.drop(columns=["ausencia"])
    else:
        X = df

    # Predicciones
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Mapeo de días
    dias_map = {0: 'Lun', 1: 'Mar', 2: 'Mié', 3: 'Jue', 4: 'Vie', 5: 'Sáb', 6: 'Dom'}

    # Crear DataFrame con resultados
    reporte = pd.DataFrame({
        'empleado_id': df_original['empleado_id'].fillna(0).astype(int),
        'nombre_empleado': df_original['nombre_empleado'].fillna('Sin nombre'),
        'fecha': df_original['fecha'],
        'fecha_str': df_original['fecha'].dt.strftime('%d/%m/%Y'),
        'dia_semana': df['dia_semana'].fillna(0).astype(int).map(dias_map),
        'mes': df['mes'].fillna(0).astype(int),
        'anio': df_original['fecha'].dt.year,
        'tardanza_min': df['tardanza_min'].fillna(0),
        'prediccion': predictions,
        'prob_presente': probabilities[:, 0],
        'prob_ausente': probabilities[:, 1],
        'prob_tardanza': probabilities[:, 2] if probabilities.shape[1] > 2 else [0] * len(probabilities)
    })

    # ✅ CALCULAR PROBABILIDAD MENSUAL POR EMPLEADO
    reporte_mensual = reporte.groupby(['empleado_id', 'nombre_empleado', 'mes', 'anio']).agg({
        'prob_ausente': 'mean',
        'prob_presente': 'mean',
        'prediccion': ['count', lambda x: (x == 1).sum()]
    }).reset_index()
    
    reporte_mensual.columns = ['empleado_id', 'nombre_empleado', 'mes', 'anio', 'prob_ausencia_promedio', 'prob_asistencia_promedio', 'total_dias', 'dias_ausente_predichos']
    reporte_mensual['prob_ausencia_promedio'] = reporte_mensual['prob_ausencia_promedio'] * 100
    reporte_mensual['prob_asistencia_promedio'] = reporte_mensual['prob_asistencia_promedio'] * 100
    reporte_mensual = reporte_mensual.sort_values('prob_ausencia_promedio', ascending=False)

    # Mapeo de meses
    meses_map = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
                 7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
    reporte_mensual['mes_nombre'] = reporte_mensual['mes'].map(meses_map)

    # Estadísticas
    total = len(reporte)
    presentes = (predictions == 0).sum()
    ausentes = (predictions == 1).sum()
    tardanzas = (predictions == 2).sum() if probabilities.shape[1] > 2 else 0

    # Generar HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Reporte de Predicción de Asistencia</title>
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
            .badge {{
                padding: 6px 12px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 11px;
                text-transform: uppercase;
                display: inline-block;
            }}
            .badge-presente {{
                background: #d4edda;
                color: #155724;
            }}
            .badge-ausente {{
                background: #f8d7da;
                color: #721c24;
            }}
            .badge-tardanza {{
                background: #fff3cd;
                color: #856404;
            }}
            .high-risk {{
                background-color: #fff5f5 !important;
            }}
            .prob-bar {{
                height: 8px;
                background: #ecf0f1;
                border-radius: 4px;
                overflow: hidden;
                margin-top: 5px;
            }}
            .prob-fill {{
                height: 100%;
                transition: width 0.3s;
            }}
            .prob-fill-red {{
                background: linear-gradient(90deg, #e74c3c, #c0392b);
            }}
            .prob-fill-yellow {{
                background: linear-gradient(90deg, #f39c12, #e67e22);
            }}
            .prob-fill-green {{
                background: linear-gradient(90deg, #2ecc71, #27ae60);
            }}
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
                    <h3>🔴 Ausentes</h3>
                    <div class="value" style="color: #e74c3c;">{ausentes:,}</div>
                    <div class="percentage">{ausentes/total*100:.1f}%</div>
                </div>
                <div class="card">
                    <h3>🟡 Tardanzas</h3>
                    <div class="value" style="color: #f39c12;">{tardanzas:,}</div>
                    <div class="percentage">{tardanzas/total*100:.1f}%</div>
                </div>
            </div>

            <div class="content">
                <h2>📅 Probabilidad de Asistencia y Ausencia Mensual por Empleado</h2>
                <p style="color: #7f8c8d; margin-bottom: 15px; font-size: 14px;">
                    📊 Esta tabla muestra el promedio de probabilidades de cada empleado por mes. 
                    <strong style="color: #27ae60;">Verde = Probabilidad de asistir</strong> | 
                    <strong style="color: #e74c3c;">Rojo = Probabilidad de faltar</strong>
                </p>
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Nombre Empleado</th>
                            <th>Mes</th>
                            <th>Año</th>
                            <th>🟢 Prob. Asistencia</th>
                            <th>🔴 Prob. Ausencia</th>
                            <th>Días Laborales</th>
                            <th>Ausencias Predichas</th>
                            <th>Nivel de Riesgo</th>
                        </tr>
                    </thead>
                    <tbody>
    """

    for _, row in reporte_mensual.iterrows():
        prob_ausencia_pct = row['prob_ausencia_promedio']
        prob_asistencia_pct = row['prob_asistencia_promedio']
        
        if prob_ausencia_pct >= 60:
            nivel_riesgo = "🔥 ALTO"
            row_class = "high-risk"
        elif prob_ausencia_pct >= 40:
            nivel_riesgo = "⚠️ MEDIO"
            row_class = ""
        else:
            nivel_riesgo = "✅ BAJO"
            row_class = ""
        
        html_content += f"""
                        <tr class="{row_class}">
                            <td><strong>{row['empleado_id']}</strong></td>
                            <td><strong>{row['nombre_empleado']}</strong></td>
                            <td>{row['mes_nombre']}</td>
                            <td>{int(row['anio'])}</td>
                            <td style="background: linear-gradient(to right, rgba(46, 204, 113, 0.1) 0%, rgba(46, 204, 113, 0.1) {min(prob_asistencia_pct, 100)}%, white {min(prob_asistencia_pct, 100)}%);">
                                <strong style="color: #27ae60; font-size: 16px;">{prob_asistencia_pct:.1f}%</strong>
                                <div class="prob-bar" style="margin-top: 8px;">
                                    <div class="prob-fill prob-fill-green" style="width: {min(prob_asistencia_pct, 100)}%"></div>
                                </div>
                            </td>
                            <td style="background: linear-gradient(to right, rgba(231, 76, 60, 0.1) 0%, rgba(231, 76, 60, 0.1) {min(prob_ausencia_pct, 100)}%, white {min(prob_ausencia_pct, 100)}%);">
                                <strong style="color: #e74c3c; font-size: 16px;">{prob_ausencia_pct:.1f}%</strong>
                                <div class="prob-bar" style="margin-top: 8px;">
                                    <div class="prob-fill prob-fill-{'red' if prob_ausencia_pct >= 60 else 'yellow' if prob_ausencia_pct >= 40 else 'green'}" style="width: {min(prob_ausencia_pct, 100)}%"></div>
                                </div>
                            </td>
                            <td><strong>{int(row['total_dias'])}</strong> días</td>
                            <td><strong style="color: #e74c3c;">{int(row['dias_ausente_predichos'])}</strong> días</td>
                            <td><strong>{nivel_riesgo}</strong></td>
                        </tr>
        """

    html_content += """
                    </tbody>
                </table>

                <h2>⚠️ Empleados en Alto Riesgo de Ausencia (Diario)</h2>
    """

    # Alto riesgo de ausencia
    alto_riesgo = reporte[reporte['prob_ausente'] > 0.6].sort_values('prob_ausente', ascending=False).head(20)
    
    if len(alto_riesgo) > 0:
        html_content += """
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Nombre</th>
                            <th>Fecha</th>
                            <th>Día</th>
                            <th>Predicción</th>
                            <th>Probabilidad Ausencia</th>
                            <th>Confianza</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for _, row in alto_riesgo.iterrows():
            pred = int(row['prediccion'])
            badge_class = ['badge-presente', 'badge-ausente', 'badge-tardanza'][pred]
            badge_text = ['PRESENTE', 'AUSENTE', 'TARDANZA'][pred]
            prob_pct = row['prob_ausente'] * 100
            
            html_content += f"""
                            <tr class="high-risk">
                                <td><strong>{row['empleado_id']}</strong></td>
                                <td><strong>{row['nombre_empleado']}</strong></td>
                                <td>{row['fecha_str']}</td>
                                <td>{row['dia_semana']}</td>
                                <td><span class="badge {badge_class}">{badge_text}</span></td>
                                <td>
                                    <strong>{prob_pct:.1f}%</strong>
                                    <div class="prob-bar">
                                        <div class="prob-fill prob-fill-red" style="width: {prob_pct}%"></div>
                                    </div>
                                </td>
                                <td>{'🔥 Alta' if prob_pct > 70 else '⚠️ Media'}</td>
                            </tr>
            """
        
        html_content += """
                    </tbody>
                </table>
        """
    else:
        html_content += "<p>✅ No hay empleados con alto riesgo de ausencia (>60%)</p>"

    # ✅ TODAS LAS TARDANZAS (sin filtro de probabilidad)
    html_content += """
                <h2>🟡 Empleados con Riesgo de Tardanza (Todas las Probabilidades)</h2>
                <p style="color: #7f8c8d; margin-bottom: 15px; font-size: 14px;">
                    ⚠️ Esta tabla muestra <strong>TODAS</strong> las tardanzas predichas, ordenadas por probabilidad descendente.
                </p>
    """
    
    # ✅ CAMBIO: Eliminar el filtro > 0.5 y mostrar TODAS las tardanzas
    tardanzas_pred = reporte[reporte['prob_tardanza'] > 0].sort_values('prob_tardanza', ascending=False)
    
    if len(tardanzas_pred) > 0:
        html_content += """
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
        """
        
        for _, row in tardanzas_pred.iterrows():
            pred = int(row['prediccion'])
            badge_class = ['badge-presente', 'badge-ausente', 'badge-tardanza'][pred]
            badge_text = ['PRESENTE', 'AUSENTE', 'TARDANZA'][pred]
            prob_pct = row['prob_tardanza'] * 100
            tardanza = row['tardanza_min']
            
            html_content += f"""
                            <tr>
                                <td><strong>{row['empleado_id']}</strong></td>
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
            """
        
        html_content += """
                    </tbody>
                </table>
        """
    else:
        html_content += "<p>✅ No hay tardanzas predichas</p>"

    html_content += """
            </div>
        </div>
    </body>
    </html>
    """

    # Guardar archivos
    import os
    os.makedirs("reports", exist_ok=True)
    
    with open("reports/reporte_ausencias.html", "w", encoding="utf-8") as f:
        f.write(html_content)

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