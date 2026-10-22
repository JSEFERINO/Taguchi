import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
import requests
import io

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="⚙️ Calculadora Taguchi Pro",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Calculadora de Metodología Taguchi Pro")
st.markdown("Análisis de Arreglos Ortogonales, Cociente S/N, Varianza y Medias.")
st.markdown("---")

# ============================================================
# MATRICES DE ARREGLOS ORTOGONALES
# ============================================================
OA_MATRICES = {
    "L4 (2^3) - 4 corridas": [[1,1,1], [1,2,2], [2,1,2], [2,2,1]],
    "L8 (2^7) - 8 corridas": [
        [1,1,1,1,1,1,1], [1,1,1,2,2,2,2], [1,2,2,1,1,2,2], [1,2,2,2,2,1,1],
        [2,1,2,1,2,1,2], [2,1,2,2,1,2,1], [2,2,1,1,2,2,1], [2,2,1,2,1,1,2]
    ],
    "L9 (3^4) - 9 corridas": [
        [1,1,1,1], [1,2,2,2], [1,3,3,3], [2,1,2,3], [2,2,3,1],
        [2,3,1,2], [3,1,3,2], [3,2,1,3], [3,3,2,1]
    ],
    "L12 (2^11) - 12 corridas": [
        [1,1,1,1,1,1,1,1,1,1,1], [1,1,1,1,1,2,2,2,2,2,2], [1,1,2,2,2,1,1,1,2,2,2],
        [1,2,1,2,2,1,2,2,1,1,2], [1,2,2,1,2,2,1,2,1,2,1], [1,2,2,2,1,2,2,1,2,1,1],
        [2,1,2,2,1,1,2,2,1,2,1], [2,1,2,1,2,2,2,1,1,1,1], [2,1,1,2,2,2,1,2,2,1,1],
        [2,2,2,1,1,1,2,2,2,1,1], [2,2,1,2,1,2,2,1,1,2,2], [2,2,1,1,2,1,1,2,2,2,2]
    ],
    "L16 (2^15) - 16 corridas": [
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,2,2,2,2,2,2,2,2],
        [1,1,1,2,2,2,2,1,1,1,1,2,2,2,2], [1,1,1,2,2,2,2,2,2,2,2,1,1,1,1],
        [1,2,2,1,1,2,2,1,1,2,2,1,1,2,2], [1,2,2,1,1,2,2,2,2,1,1,2,2,1,1],
        [1,2,2,2,2,1,1,1,1,2,2,2,2,1,1], [1,2,2,2,2,1,1,2,2,1,1,1,1,2,2],
        [2,1,2,1,2,1,2,1,2,1,2,1,2,1,2], [2,1,2,1,2,1,2,2,1,2,1,2,1,2,1],
        [2,1,2,2,1,2,1,1,2,1,2,2,1,2,1], [2,1,2,2,1,2,1,2,1,2,1,1,2,1,2],
        [2,2,1,1,2,2,1,1,2,2,1,1,2,2,1], [2,2,1,1,2,2,1,2,1,1,2,2,1,1,2],
        [2,2,1,2,1,1,2,1,2,1,2,2,1,1,2], [2,2,1,2,1,1,2,2,1,2,1,1,2,2,1]
    ],
    "L16 (4^5) - 16 corridas": [
        [1,1,1,1,1], [1,2,2,2,2], [1,3,3,3,3], [1,4,4,4,4],
        [2,1,2,3,4], [2,2,1,4,3], [2,3,4,1,2], [2,4,3,2,1],
        [3,1,3,4,2], [3,2,4,3,1], [3,3,1,2,4], [3,4,2,1,3],
        [4,1,4,2,3], [4,2,3,1,4], [4,3,2,4,1], [4,4,1,3,2]
    ]
}

def analizar_anova(df, formula, titulo="ANOVA"):
    try:
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category')
        modelo = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(modelo, typ=1)
        if 'mean_sq' not in anova_table.columns:
            anova_table['mean_sq'] = anova_table['sum_sq'] / anova_table['df']
        
        rename_map = {'sum_sq': 'Sum Sq', 'df': 'DF', 'mean_sq': 'Mean Sq', 'F': 'F-Ratio', 'PR(>F)': 'P-Value'}
        anova_table = anova_table.rename(columns=rename_map)
        
        st.success(f"✅ {titulo}")
        st.dataframe(anova_table.style.format({
            'Sum Sq': '{:.4f}', 'Mean Sq': '{:.4f}', 'F-Ratio': '{:.4f}', 'P-Value': '{:.6f}'
        }, na_rep="None"))
        st.write(f"**R² = {modelo.rsquared:.6f}** | **R² ajustado = {modelo.rsquared_adj:.6f}**")
        return modelo, anova_table
    except Exception as e:
        st.error(f"❌ Error en ANOVA: {str(e)}")
        return None, None

# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================

# 1. CONFIGURACIÓN DEL DISEÑO
with st.expander("📌 1. Configuración del Diseño y Factores", expanded=True):
    col_ar, col_nf = st.columns(2)
    with col_ar:
        tipo_arreglo = st.selectbox(
            "Selecciona el Arreglo Ortogonal:",
            list(OA_MATRICES.keys()),
            index=4,
            key="arreglo_select"
        )
    with col_nf:
        if "4^5" in tipo_arreglo:
            max_niveles, max_factores = 4, 5
        elif "3^4" in tipo_arreglo:
            max_niveles, max_factores = 3, 4
        elif "2^3" in tipo_arreglo:
            max_niveles, max_factores = 2, 3
        elif "2^7" in tipo_arreglo:
            max_niveles, max_factores = 2, 7
        elif "2^11" in tipo_arreglo:
            max_niveles, max_factores = 2, 11
        else:
            max_niveles, max_factores = 2, 15

        n_factores = st.number_input("Número de Factores a estudiar:", min_value=2, max_value=max_factores, value=min(max_factores, 3), step=1)

    st.markdown("---")
    nombres_factores = []
    cols_fact = st.columns(min(n_factores, 3))
    
    for i in range(n_factores):
        with cols_fact[i % len(cols_fact)]:
            st.markdown(f"**Factor {i+1}**")
            nombre = st.text_input(f"Nombre F{i+1}", value=f"Factor_{chr(65+i)}", key=f"f_name_{i}")
            niveles = []
            for n in range(1, max_niveles + 1):
                nivel_val = st.text_input(f"Nivel {n}", value=f"N{n}", key=f"f_lvl_{i}_{n}")
                niveles.append(nivel_val)
            nombres_factores.append({'nombre': nombre, 'niveles': niveles})

# 2. DEFINICIÓN DE RESPUESTAS Y RÉPLICAS
with st.expander("📌 2. Definición de Respuestas y Réplicas", expanded=True):
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        respuestas_input = st.text_input("Nombres de las respuestas (separadas por coma):", value="Respuesta", key="resp_input")
        lista_respuestas = [r.strip() for r in respuestas_input.split(',') if r.strip()]
    with col_r2:
        n_replicas = st.number_input("Número de Réplicas por corrida (para calcular varianza):", min_value=1, max_value=10, value=1, step=1)

# 3. GENERAR DISEÑO
if st.button("🎲 Generar Tabla del Diseño", type="primary", key="btn_generar"):
    oa_matrix = OA_MATRICES[tipo_arreglo]
    df_diseno = pd.DataFrame(oa_matrix, columns=[f"F{i+1}" for i in range(len(oa_matrix[0]))])
    df_diseno = df_diseno.iloc[:, :n_factores]
    df_diseno.columns = [f['nombre'] for f in nombres_factores]
    
    for i, f in enumerate(nombres_factores):
        mapeo = {j+1: f['niveles'][j] for j in range(len(f['niveles']))}
        df_diseno[f['nombre']] = df_diseno[f['nombre']].map(mapeo)
        
    df_diseno.insert(0, 'Run', range(1, len(df_diseno) + 1))
    
    # Crear columnas para las réplicas de cada respuesta
    for r in lista_respuestas:
        for rep in range(1, n_replicas + 1):
            df_diseno[f"{r}_R{rep}"] = np.nan
            
    st.session_state['df_taguchi'] = df_diseno
    st.session_state['factores_taguchi'] = [f['nombre'] for f in nombres_factores]
    st.session_state['respuestas_taguchi'] = lista_respuestas
    st.session_state['n_replicas'] = n_replicas
    st.rerun()

# 4. EDICIÓN Y ANÁLISIS
if 'df_taguchi' in st.session_state:
    st.markdown("---")
    st.subheader("📋 Carga de Datos")
    
    metodo_carga = st.radio(
        "Selecciona el método para ingresar tus datos:",
        ["✏️ Manual (Editar tabla generada)", "📁 Cargar archivo CSV", "🔗 Cargar desde URL"],
        horizontal=True
    )

    df_editado = st.session_state['df_taguchi'] # Por defecto el generado

    if metodo_carga == "✏️ Manual (Editar tabla generada)":
        st.info("💡 Edita directamente las celdas en la tabla. Si necesitas llenarlo en Excel, descarga la plantilla.")
        
        # Botón para descargar plantilla vacía
        csv_vacio = df_editado.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Descargar Plantilla Vacía (CSV)", 
            data=csv_vacio, 
            file_name="plantilla_taguchi.csv", 
            mime="text/csv"
        )
        
        df_editado = st.data_editor(df_editado, num_rows="fixed", use_container_width=True)
        st.session_state['df_taguchi'] = df_editado

    elif metodo_carga == "📁 Cargar archivo CSV":
        archivo = st.file_uploader("Sube tu archivo CSV", type=['csv'])
        if archivo is not None:
            try:
                contenido = archivo.getvalue().decode('utf-8')
                df_subido = pd.read_csv(io.StringIO(contenido))
                st.success("✅ Archivo CSV cargado exitosamente.")
                st.dataframe(df_subido)
                st.session_state['df_taguchi'] = df_subido
                df_editado = df_subido
            except Exception as e:
                st.error(f"❌ Error al leer el CSV: {e}")

    elif metodo_carga == "🔗 Cargar desde URL":
        url = st.text_input("Ingresa la URL directa del archivo CSV (ej. GitHub Raw):")
        if st.button("Cargar datos desde URL"):
            if url:
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    df_subido = pd.read_csv(io.StringIO(response.text))
                    st.success("✅ Datos cargados desde la URL.")
                    st.dataframe(df_subido)
                    st.session_state['df_taguchi'] = df_subido
                    df_editado = df_subido
                except Exception as e:
                    st.error(f"❌ Error al cargar desde URL: {e}")

    st.markdown("---")
    st.subheader("📊 Configuración del Análisis")
    
    col_rs, col_mm = st.columns(2)
    with col_rs:
        resp_sel = st.selectbox("Selecciona la Respuesta a analizar:", st.session_state['respuestas_taguchi'])
    with col_mm:
        performance_measure = st.selectbox(
            "Medida de Rendimiento (Performance Measure):", 
            [
                "Mean (Media)",
                "Log Variance",
                "Signal-to-Noise Ratio (-10 log s^2)",
                "Signal-to-Noise Ratio (Target Value)",
                "Signal-to-Noise Ratio (Larger is Better)",
                "Signal-to-Noise Ratio (Smaller is Better)"
            ]
        )
        
    target_val = 0.0
    if "Target Value" in performance_measure:
        target_val = st.number_input("Valor Objetivo (Target):", value=0.0)
        
    if st.button("🚀 Ejecutar Análisis Taguchi", type="primary"):
        df_analisis = st.session_state['df_taguchi'].copy()
        
        # Validar que las columnas de las réplicas existan
        cols_replicas = [c for c in df_analisis.columns if c.startswith(f"{resp_sel}_R")]
        
        if not cols_replicas:
            st.error(f"❌ No se encontraron columnas de réplicas para '{resp_sel}'. Asegúrate de que el CSV tenga el formato correcto (ej. {resp_sel}_R1, {resp_sel}_R2).")
        elif df_analisis[cols_replicas].isnull().any().any():
            st.error("❌ Faltan datos en las réplicas seleccionadas.")
        else:
            # Calcular métricas por fila (Run)
            medias = []
            varianzas = []
            metricas = []
            
            for idx, row in df_analisis.iterrows():
                y_vals = row[cols_replicas].values.astype(float)
                n = len(y_vals)
                mean_y = np.mean(y_vals)
                
                # Varianza (con corrección de muestra ddof=1 si n>1)
                var_y = np.var(y_vals, ddof=1) if n > 1 else 0.0
                
                medias.append(mean_y)
                varianzas.append(var_y)
                
                if performance_measure == "Mean (Media)":
                    metricas.append(mean_y)
                elif performance_measure == "Log Variance":
                    metricas.append(np.log(var_y + 1e-9))
                elif performance_measure == "Signal-to-Noise Ratio (-10 log s^2)":
                    metricas.append(-10 * np.log10(var_y + 1e-9))
                elif performance_measure == "Signal-to-Noise Ratio (Target Value)":
                    if n == 1:
                        st.warning(f"⚠️ Se requieren réplicas (n>1) para calcular la varianza en la corrida {row['Run']}. Usando varianza=0.")
                        var_y = 1e-9
                    metricas.append(10 * np.log10((mean_y**2) / (var_y + 1e-9)))
                elif performance_measure == "Signal-to-Noise Ratio (Larger is Better)":
                    metricas.append(-10 * np.log10(np.mean(1 / (y_vals**2 + 1e-9))))
                elif performance_measure == "Signal-to-Noise Ratio (Smaller is Better)":
                    metricas.append(-10 * np.log10(np.mean(y_vals**2)))
                    
            df_analisis['Metric'] = metricas
            df_analisis['Mean_Y'] = medias
            df_analisis['Var_Y'] = varianzas
            
            st.info(f"**Métrica seleccionada:** {performance_measure}")
            
            # ANOVA de la Métrica
            factores = st.session_state['factores_taguchi']
            formula = 'Metric ~ ' + ' + '.join(factores)
            modelo, anova = analizar_anova(df_analisis, formula, f"ANOVA de {performance_measure}")

            # Gráficos de Efectos Principales
            st.markdown("---")
            st.subheader("📈 Gráficos de Efectos Principales")
            efectos_optimos = {}
            
            cols_ef = st.columns(min(len(factores), 3))
            for i, f in enumerate(factores):
                with cols_ef[i % len(cols_ef)]:
                    medias_factor = df_analisis.groupby(f)['Metric'].mean().reset_index()
                    fig = px.line(medias_factor, x=f, y='Metric', markers=True, title=f"Efecto: {f}")
                    fig.update_layout(height=250, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Lógica de optimización según la métrica
                    if performance_measure in ["Mean (Media)", "Signal-to-Noise Ratio (Larger is Better)", "Signal-to-Noise Ratio (Target Value)"]:
                        mejor_nivel = medias_factor.loc[medias_factor['Metric'].idxmax(), f]
                    elif performance_measure in ["Signal-to-Noise Ratio (Smaller is Better)"]:
                        mejor_nivel = medias_factor.loc[medias_factor['Metric'].idxmin(), f]
                    else:
                        mejor_nivel = medias_factor.loc[medias_factor['Metric'].idxmin(), f]
                        
                    efectos_optimos[f] = mejor_nivel

            st.markdown("---")
            st.subheader("🏆 Predicción del Óptimo")
            st.write("**Niveles óptimos seleccionados:**")
            for f, nivel in efectos_optimos.items():
                st.write(f"- **{f}:** {nivel}")
                
            gran_media = df_analisis['Metric'].mean()
            suma_efectos = sum(df_analisis.groupby(f)['Metric'].mean()[nivel] - gran_media for f, nivel in efectos_optimos.items())
            predicho = gran_media + suma_efectos
            
            st.success(f"**Valor Predicho de la Métrica ({performance_measure}):** {predicho:.4f}")
            
            if "Larger is Better" in performance_measure:
                y_est = 10 ** (predicho / 20)
                st.info(f"**Respuesta Estimada (Y):** {y_est:.4f}")
            elif "Smaller is Better" in performance_measure:
                y_est = 10 ** (-predicho / 20)
                st.info(f"**Respuesta Estimada (Y):** {y_est:.4f}")
