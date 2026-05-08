import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="🍭 Drug Personality Predictor", page_icon="🧠", layout="centered")

DRUG_CONFIG = {
    "Cannabis": {"emoji": "", "dark": "#1a5c38", "badge": "#C8F5E1", "gradient": "#A8E6CF, #56CCF2"},
    "Nicotina": {"emoji": "", "dark": "#7a3410", "badge": "#FFE8D6", "gradient": "#FFD3B6, #FFA07A"},
    "Extasis":  {"emoji": "", "dark": "#5b1a94", "badge": "#EDD9FF", "gradient": "#D4A5F5, #FF79C6"},
    "Cocaina":  {"emoji": "", "dark": "#0e4a70", "badge": "#D0EEFF", "gradient": "#A0D8F1, #89F7FE"},
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@400;700;800&family=Nunito:wght@600;700&display=swap');

*, p, label, span, div { color: #2d2d2d; }

.stApp {
    background: linear-gradient(135deg, #fff9f0 0%, #f0f4ff 50%, #fff0f9 100%);
    font-family: 'Nunito', sans-serif;
}

/* Título */
.hero-title {
    text-align: center;
    font-family: 'Baloo 2', cursive;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff6b9d, #c44dff, #4d79ff, #00c9a7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 0.3rem;
}
.hero-sub { text-align:center; color:#666; font-size:1rem; margin-bottom:1.2rem; }

/* Pestañas */
div[data-testid="stTabs"] [role="tablist"] {
    gap: 8px; background: #FEF9FF; padding: 6px 10px;
    border-radius: 20px; border: 2px solid #E8D5FF;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}
div[data-testid="stTabs"] [role="tab"] {
    border-radius: 14px !important; font-family:'Baloo 2',cursive !important;
    font-weight:700 !important; font-size:0.95rem !important;
    padding:6px 18px !important; border:none !important; color:#7a5af8 !important;
}
div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #ff9de2, #a78bfa) !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(167,139,250,0.4);
}

/* Inputs */
div[data-testid="stNumberInput"] { background:#FEF9FF; border-radius:14px; }
div[data-testid="stNumberInput"] input {
    border-radius:12px !important; border:2px solid #D8B4FE !important;
    background:#FFF5FF !important; color:#2d2d2d !important;
    font-family:'Nunito',sans-serif !important; font-weight:600 !important;
}
div[data-testid="stNumberInput"] label { color:#4a2d7f !important; font-weight:700 !important; }

/* Radio */
div[data-testid="stRadio"] { background:#FEF9FF; border-radius:14px; padding:8px 12px; border:2px solid #E8D5FF; }
div[data-testid="stRadio"] label { color:#4a2d7f !important; font-weight:600 !important; }
div[data-testid="stRadio"] [role="radiogroup"] label span { color:#4a2d7f !important; }

/* Botón principal (submit) */
div[data-testid="stFormSubmitButton"] > button {
    width:100%; background:linear-gradient(135deg,#ff9de2,#a78bfa,#6ee7f7) !important;
    color:white !important; border:none !important; border-radius:20px !important;
    padding:14px !important; font-family:'Baloo 2',cursive !important;
    font-size:1.1rem !important; font-weight:800 !important;
    box-shadow:0 6px 20px rgba(167,139,250,0.4); transition:all .25s;
}
div[data-testid="stFormSubmitButton"] > button:hover { transform:translateY(-3px); }

/* Botón de Ejemplo */
div.stButton > button {
    background: linear-gradient(135deg, #FFD3B6, #FFA07A) !important;
    color: #7a3410 !important; border: 2px solid #FFB899 !important;
    border-radius: 16px !important; font-family:'Baloo 2',cursive !important;
    font-weight:700 !important; padding:8px 20px !important;
    box-shadow: 0 4px 12px rgba(255,160,122,0.3); transition:all .2s;
}
div.stButton > button:hover { transform:translateY(-2px); }

/* Uploader */
div[data-testid="stFileUploader"] {
    border: 2.5px dashed #c4a8ff !important; border-radius:20px !important;
    background:#fdf8ff !important; padding:20px !important;
}
div[data-testid="stFileUploader"] label { color:#5b1a94 !important; font-weight:700 !important; }
div[data-testid="stFileUploader"] span { color:#5b1a94 !important; }

/* Metric */
div[data-testid="stMetric"] {
    background:linear-gradient(135deg,#FFF3E0,#FFF9F0); border:2px solid #FFD180;
    border-radius:18px; padding:18px !important; box-shadow:0 4px 16px rgba(255,152,0,.1);
}
div[data-testid="stMetricLabel"] { color:#7a3410 !important; font-weight:700 !important; }
div[data-testid="stMetricValue"] {
    font-family:'Baloo 2',cursive !important; font-size:2rem !important; color:#c05000 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constantes ──────────────────────────────────────────────
FEATURE_COLS = ['Age','Gender','Education','Country','Ethnicity',
                'Nscore','Escore','Oscore','Ascore','Cscore','Impulsive','SS']

FEATURE_INFO = {
    'Age':       (" Edad",             -0.95, 2.59),
    'Gender':    (" Género",           -0.48, 0.48),
    'Education': (" Educación",        -2.44, 1.98),
    'Country':   (" País",             -0.57, 0.96),
    'Ethnicity': (" Etnia",           -0.32, 0.11),
    'Nscore':    (" Neuroticismo",     -3.46, 3.46),
    'Escore':    (" Extraversión",     -3.27, 3.27),
    'Oscore':    (" Apertura",         -3.27, 2.90),
    'Ascore':    (" Amabilidad",       -3.46, 3.46),
    'Cscore':    (" Responsabilidad",  -3.46, 3.46),
    'Impulsive': (" Impulsividad",     -2.55, 2.90),
    'SS':        (" Búsqueda sensac.", -1.92, 1.92),
}

# Valores de ejemplo: perfil de persona con alta apertura + impulsividad (tendencia Cannabis/Éxtasis)
EXAMPLE_VALUES = {
    'Age': -0.07854, 'Gender': -0.48246, 'Education': 1.16365, 'Country': 0.96082,
    'Ethnicity': -0.31685, 'Nscore': 0.73545, 'Escore': 1.93886, 'Oscore': 1.24033,
    'Ascore': -1.11396, 'Cscore': -1.37983, 'Impulsive': 1.29472, 'SS': 1.92173,
}

ALL_COLS = ['ID','Age','Gender','Education','Country','Ethnicity',
            'Nscore','Escore','Oscore','Ascore','Cscore','Impulsive','SS',
            'Alcohol','Amphet','Amyl','Benzos','Caff','Cannabis','Choc',
            'Coke','Crack','Ecstasy','Heroin','Ketamine','Legalh','LSD',
            'Meth','Mush','Nicotine','Semer','VSA']

TARGET_COL_MAP = {'Cannabis':'Cannabis','Nicotina':'Nicotine','Extasis':'Ecstasy','Cocaina':'Coke'}

# ── Carga de modelos ─────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelos... ")
def load_all_models():
    m = {}
    for name in DRUG_CONFIG:
        k = name.lower()
        m[name] = {
            "scaler": joblib.load(f"scaler_{k}.pkl"),
            "mlp":    joblib.load(f"mlp_{k}.pkl"),
            "logreg": joblib.load(f"logreg_{k}.pkl"),
        }
    return m

try:
    ALL_MODELS = load_all_models()
except Exception as e:
    st.error(f"Modelos no encontrados. Ejecuta `train_drugs.py` primero.\n\n{e}")
    st.stop()

def predict_all(inputs, model_key):
    results = {}
    for drug, pkg in ALL_MODELS.items():
        X = pkg["scaler"].transform([inputs])
        results[drug] = pkg[model_key].predict_proba(X)[0][1]
    return results

# ── Header ───────────────────────────────────────────────────
st.markdown('<div class="hero-title">🧠 Drug Personality Predictor 💊</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Ingresa un perfil de personalidad y la IA detectará la mayor predisposición.</div>', unsafe_allow_html=True)

badge_html = "".join(
    f'<span style="background:{cfg["badge"]};color:{cfg["dark"]};padding:6px 14px;'
    f'border-radius:30px;font-weight:800;font-family:Nunito,sans-serif;'
    f'font-size:0.9rem;margin:4px;display:inline-block;border:2px solid {cfg["dark"]}33;">'
    f'{cfg["emoji"]} {d}</span>'
    for d, cfg in DRUG_CONFIG.items()
)
st.markdown(f'<div style="text-align:center;margin-bottom:20px">{badge_html}</div>', unsafe_allow_html=True)

# ── Session state para ejemplo ───────────────────────────────
if "use_example" not in st.session_state:
    st.session_state.use_example = False

tab1, tab2 = st.tabs(["Predicción Individual", "Validación por Lotes"])

# ════════════════════════════════════════════════════════════
# TAB 1
# ════════════════════════════════════════════════════════════
with tab1:
    col_title, col_btn = st.columns([3, 1])
    with col_title:
        st.markdown("#### 🧬 Perfil de personalidad")
        st.caption("Valores cuantificados UCI (entre -3.0 y 3.0 aprox.)")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Ejemplo"):
            st.session_state.use_example = True
            st.rerun()

    model_choice = st.radio("Modelo:", ["Red Neuronal (MLP)", "Regresión Logística"], horizontal=True)
    model_key = "mlp" if "MLP" in model_choice else "logreg"

    with st.form("individual_form"):
        inputs = []
        cols_layout = st.columns(3)
        for i, feat in enumerate(FEATURE_COLS):
            label, mn, mx = FEATURE_INFO[feat]
            with cols_layout[i % 3]:
                # Si no hay ejemplo, el valor inicial es None para forzar el rellenado manual
                val = st.number_input(
                    label, 
                    value=float(EXAMPLE_VALUES[feat]) if st.session_state.use_example else None,
                    min_value=float(mn) - 0.1, max_value=float(mx) + 0.1,
                    step=0.01, format="%.4f",
                    placeholder="0.0000"
                )
                inputs.append(val)

        submitted = st.form_submit_button("Analizar perfil", use_container_width=True)

    if st.session_state.use_example:
        st.info("Datos de ejemplo cargados. Podés modificarlos o darle a Analizar perfil.")

    if submitted:
        if any(v is None for v in inputs):
            st.error("¡Faltan datos! Por favor, completa todos los campos de personalidad antes de continuar.")
        else:
            st.session_state.use_example = False
            results = predict_all(inputs, model_key)
            winner  = max(results, key=results.get)
            wcfg    = DRUG_CONFIG[winner]

        st.markdown("---")
        # Tarjeta ganadora simplificada
        st.markdown(
            f'<div style="background:linear-gradient(135deg,{wcfg["gradient"]});'
            f'border-radius:24px;padding:32px;text-align:center;'
            f'box-shadow:0 8px 24px rgba(0,0,0,0.1);margin-bottom:20px;">'
            f'<p style="font-family:Baloo 2,cursive;font-size:2.5rem;font-weight:800;'
            f'color:{wcfg["dark"]};margin:0;">Predicción: {winner}</p>'
            f'<p style="font-family:Nunito,sans-serif;font-size:1.2rem;font-weight:700;'
            f'color:{wcfg["dark"]};margin-top:10px;opacity:0.9;"></p>'
            f'</div>',
            unsafe_allow_html=True
        )



# ════════════════════════════════════════════════════════════
# TAB 2
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Validación por Lotes")
    st.markdown("Sube el dataset que tenga las 32 columnas requeridas. Se evaluarán las **4 sustancias** de una vez.")

    batch_model_type = st.radio("Modelo:", ["Red Neuronal (MLP)", "Regresión Logística"],
                                horizontal=True, key="batch_model")
    batch_key = "mlp" if "MLP" in batch_model_type else "logreg"


    uploaded_file = st.file_uploader("Selecciona tu archivo", type=['csv', 'data'])

    if uploaded_file:
        try:
            df_lote = pd.read_csv(uploaded_file, header=None)
            if df_lote.shape[1] != 32:
                st.error("El CSV debe tener 32 columnas (formato UCI).")
                st.stop()
            df_lote.columns = ALL_COLS
            X_lote = df_lote[FEATURE_COLS].values
            st.success(f"Archivo cargado: **{len(df_lote)} registros**.")

            # Unificación de predicciones para Matriz 4x4
            y_real_multi = []
            y_pred_multi = []
            
            # Función para obtener el nivel numérico
            def get_level(val): return int(val[2])

            for idx, row in df_lote.iterrows():
                # 1. Obtener Real (Sustancia dominante entre las 4)
                levels = {d: get_level(row[TARGET_COL_MAP[d]]) for d in DRUG_CONFIG}
                
                # Solo procesamos si consume al menos una de las 4 (CL2+)
                if max(levels.values()) <= 1: continue
                
                # Prioridad de desempate: Cocaina > Extasis > Cannabis > Nicotina
                prio_order = ["Cocaina", "Extasis", "Cannabis", "Nicotina"]
                real_winner = prio_order[0]
                max_l = -1
                for d in prio_order:
                    if levels[d] > max_l:
                        max_l = levels[d]
                        real_winner = d
                
                y_real_multi.append(real_winner)
                
                # 2. Obtener Predicción (Mayor probabilidad de los 4 modelos existentes)
                row_features = row[FEATURE_COLS].values
                probs = {}
                for drug in DRUG_CONFIG:
                    X_sc = ALL_MODELS[drug]["scaler"].transform([row_features])
                    probs[drug] = ALL_MODELS[drug][batch_key].predict_proba(X_sc)[0][1]
                
                pred_winner = max(probs, key=probs.get)
                y_pred_multi.append(pred_winner)

            if not y_real_multi:
                st.warning("No se encontraron usuarios de estas 4 sustancias en el archivo.")
            else:
                labels = ["Cannabis", "Nicotina", "Extasis", "Cocaina"]
                cm = confusion_matrix(y_real_multi, y_pred_multi, labels=labels)
                acc = accuracy_score(y_real_multi, y_pred_multi)

                st.markdown(
                    f'<div style="background:linear-gradient(135deg,#e0c3fc, #8ec5fc);'
                    f'border-radius:18px;padding:14px 22px;margin:24px 0 14px 0;text-align:center;">'
                    f'<span style="font-family:Baloo 2,cursive;font-size:1.6rem;'
                    f'font-weight:800;color:#2d2d2d;">Matriz de Confusión Unificada (4x4)</span><br>'
                    f'<span style="font-family:Baloo 2,cursive;font-size:1.3rem;'
                    f'font-weight:700;color:#2d2d2d;">Accuracy Global: {acc*100:.1f}%</span></div>',
                    unsafe_allow_html=True
                )

                fig, ax = plt.subplots(figsize=(6, 5))
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')
                
                im = ax.imshow(cm, interpolation='nearest', cmap='Purples')
                
                tick_marks = np.arange(len(labels))
                ax.set_xticks(tick_marks)
                ax.set_xticklabels(labels, rotation=45, ha="right", fontfamily='Nunito', fontweight='bold')
                ax.set_yticks(tick_marks)
                ax.set_yticklabels(labels, fontfamily='Nunito', fontweight='bold')

                # Anotar valores en las celdas
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, format(cm[i, j], 'd'),
                                ha="center", va="center",
                                color="white" if cm[i, j] > thresh else "black",
                                fontsize=14, fontweight='bold')

                ax.set_ylabel('Sustancia Real', fontsize=12, fontweight='bold', fontfamily='Baloo 2')
                ax.set_xlabel('Predicción', fontsize=12, fontweight='bold', fontfamily='Baloo 2')
                
                for spine in ax.spines.values(): spine.set_visible(False)
                
                st.pyplot(fig)
                plt.close(fig)
        except Exception as e:
            st.error(f"Error: {e}")