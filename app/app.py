import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, 'model', 'model.pkl'))

st.set_page_config(page_title="CKD Risk Predictor", page_icon="🩺", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #EAF6F6;
    color: #1A3A3A;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem !important; max-width: 100% !important; }

/* ── App background ── */
[data-testid="stAppViewContainer"] { background: #EAF6F6; }
[data-testid="stHeader"] { background: transparent; }

/* ── Header ── */
.ckd-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 0 14px 0;
    margin-bottom: 16px;
    border-bottom: 1px solid #D4EEF0;
}
.ckd-header-left { display: flex; align-items: center; gap: 12px; }
.ckd-logo {
    width: 38px; height: 38px;
    background: #2BBFBF;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; color: white;
}
.ckd-title { font-size: 1.1rem; font-weight: 700; color: #1A3A3A; margin: 0; }
.ckd-sub { font-size: 0.7rem; color: #7AADB0; margin: 2px 0 0 0; }
.ckd-badge {
    font-size: 0.68rem;
    background: #E0F5F5; color: #2BBFBF;
    border: 1px solid #B2E4E4;
    padding: 5px 12px; border-radius: 20px;
    font-weight: 600; letter-spacing: 0.5px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #F0FAFA !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid #D4EEF0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #7AADB0 !important;
    border-radius: 7px !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    padding: 6px 16px !important;
    border: none !important;
    font-family: 'Inter', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #2BBFBF !important;
    font-weight: 600 !important;
    border: 1px solid #D4EEF0 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Input fields ── */
.stNumberInput input, .stSelectbox > div > div {
    background: #F7FCFC !important;
    border: 1px solid #D4EEF0 !important;
    border-radius: 8px !important;
    color: #1A3A3A !important;
    font-size: 0.82rem !important;
    font-family: 'Inter', sans-serif !important;
    padding: 6px 10px !important;
}
.stNumberInput input:focus, .stSelectbox > div > div:focus-within {
    border-color: #2BBFBF !important;
    box-shadow: 0 0 0 3px rgba(43,191,191,0.12) !important;
    background: #fff !important;
}
label {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    color: #7AADB0 !important;
    letter-spacing: 0.4px !important;
    text-transform: uppercase !important;
}
[data-baseweb="select"] { background: #F7FCFC !important; }
[data-baseweb="popover"] { background: #fff !important; border: 1px solid #D4EEF0 !important; border-radius: 8px !important; }
[role="option"] { background: #fff !important; color: #1A3A3A !important; font-size: 0.82rem !important; }
[role="option"]:hover { background: #F0FAFA !important; color: #2BBFBF !important; }

/* ── Button ── */
.stButton > button {
    background: #2BBFBF !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 0 !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #22AAAA !important;
    transform: translateY(-1px) !important;
}

/* ── Cards / containers ── */
.result-panel {
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}
.result-low {
    background: #F0FAFA;
    border: 1px solid #D4EEF0;
}
.result-high {
    background: #FFF5F0;
    border: 1px solid #FFCFBA;
}
.result-idle {
    background: #F0FAFA;
    border: 1px dashed #B2E4E4;
    padding: 32px 20px;
}

/* ── Gauge ── */
.gauge-ring {
    width: 96px; height: 96px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 12px auto;
}
.ring-low  { background: conic-gradient(#2BBFBF var(--pct), #D4EEF0 0); }
.ring-high { background: conic-gradient(#FF7043 var(--pct), #FFE0D6 0); }
.ring-inner {
    width: 76px; height: 76px; border-radius: 50%;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.ring-inner-low  { background: #F0FAFA; }
.ring-inner-high { background: #FFF5F0; }

.result-label { font-size: 1rem; font-weight: 700; margin: 4px 0 2px 0; letter-spacing: 0.2px; }
.result-label-low  { color: #2BBFBF; }
.result-label-high { color: #FF7043; }
.result-desc { font-size: 0.72rem; color: #7AADB0; }

/* ── Signal rows ── */
.signal-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #EAF6F6;
    font-size: 0.75rem;
}
.signal-name { color: #7AADB0; font-size: 0.72rem; }
.signal-val  { color: #1A3A3A; font-weight: 600; font-size: 0.72rem; }
.signal-bar-wrap { width: 64px; height: 4px; background: #EAF6F6; border-radius: 2px; overflow: hidden; }
.signal-bar { height: 100%; border-radius: 2px; }

/* ── Section cards ── */
.section-card {
    background: #ffffff;
    border: 1px solid #D4EEF0;
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 12px;
}

/* spacing fixes */
.stNumberInput { margin-bottom: 0 !important; }
.stSelectbox  { margin-bottom: 0 !important; }
div[data-testid="column"] { padding: 0 4px !important; }
.sep { border: none; border-top: 1px solid #EAF6F6; margin: 10px 0; }
.disclaimer { font-size: 0.63rem; color: #A8CCCC; text-align: center; margin-top: 8px; }
.signals-heading {
    font-size: 0.62rem; font-weight: 700; color: #7AADB0;
    letter-spacing: 1.4px; text-transform: uppercase; margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ckd-header">
    <div class="ckd-header-left">
        <div class="ckd-logo">🩺</div>
        <div>
            <div class="ckd-title">CKD Risk Predictor</div>
        </div>
    </div>
    <div class="ckd-badge">⚡ Real-Time Analysis</div>
</div>
""", unsafe_allow_html=True)

# ── Layout ───────────────────────────────────────────────────────────────────
left, right = st.columns([1.65, 1], gap="large")

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🔬 Blood & Metabolic", "🫀 Vitals & Urine", "📋 Clinical History"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            hemo = st.number_input("Hemoglobin g/dL", 3.0, 18.0, 13.5, step=0.1)
            sc   = st.number_input("Serum Creatinine", 0.0, 15.0, 1.2, step=0.1)
        with c2:
            bgr  = st.number_input("Blood Glucose mg/dL", 50, 500, 120)
            bu   = st.number_input("Blood Urea mg/dL", 10, 200, 40)
        with c3:
            sod  = st.number_input("Sodium mEq/L", 100, 165, 137)
            pot  = st.number_input("Potassium mEq/L", 2.0, 10.0, 4.5, step=0.1)

    with tab2:
        c4, c5, c6 = st.columns(3)
        with c4:
            age  = st.number_input("Age (yrs)", 1, 100, 45)
            bp   = st.number_input("Blood Pressure", 50, 180, 80)
        with c5:
            sg   = st.number_input("Specific Gravity", 1.000, 1.030, 1.020, step=0.005, format="%.3f")
            al   = st.number_input("Albumin (0–5)", 0, 5, 0)
        with c6:
            su   = st.number_input("Sugar (0–5)", 0, 5, 0)

    with tab3:
        c7, c8 = st.columns(2)
        with c7:
            htn   = st.selectbox("Hypertension", ["no", "yes"])
            dm    = st.selectbox("Diabetes Mellitus", ["no", "yes"])
        with c8:
            cad   = st.selectbox("Coronary Artery Disease", ["no", "yes"])
            appet = st.selectbox("Appetite", ["good", "poor"])

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<hr class="sep">', unsafe_allow_html=True)
    predict = st.button("⚡ Analyze Patient Risk")

# ── Right panel ──────────────────────────────────────────────────────────────
with right:
    def encode(val): return 1 if val in ["yes", "poor"] else 0

    if predict:
        input_dict = {
            'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su,
            'rbc': 0, 'pc': 0, 'pcc': 0, 'ba': 0,
            'bgr': bgr, 'bu': bu, 'sc': sc, 'sod': sod, 'pot': pot,
            'hemo': hemo, 'pcv': 40, 'wc': 8000, 'rc': 5.0,
            'htn': encode(htn), 'dm': encode(dm), 'cad': encode(cad),
            'appet': encode(appet), 'pe': 0, 'ane': 0
        }
        input_df = pd.DataFrame([input_dict])
        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]

        pct_val = prob * 100
        pct_str = f"{pct_val:.1f}"
        conic_pct = f"{pct_val:.1f}%"

        if pred == 1:
            ring_class   = "ring-high"
            inner_class  = "ring-inner-high"
            result_class = "result-high"
            label_class  = "result-label result-label-high"
            label        = "High Risk"
            pct_color    = "#FF7043"
            bar_color    = "#FF7043"
            desc         = "Further clinical evaluation recommended"
        else:
            ring_class   = "ring-low"
            inner_class  = "ring-inner-low"
            result_class = "result-low"
            label_class  = "result-label result-label-low"
            label        = "Low Risk"
            pct_color    = "#2BBFBF"
            bar_color    = "#2BBFBF"
            desc         = "No significant CKD indicators detected"

        st.markdown(f"""
        <div class="result-panel {result_class}">
            <div class="gauge-ring {ring_class}" style="--pct:{conic_pct}">
                <div class="ring-inner {inner_class}">
                    <span style="font-size:1.15rem;font-weight:700;color:{pct_color};font-family:'Inter',sans-serif;">{pct_str}%</span>
                    <span style="font-size:0.58rem;color:#7AADB0;">PROB</span>
                </div>
            </div>
            <div class="{label_class}">{label}</div>
            <div class="result-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="sep" style="margin-top:12px;">', unsafe_allow_html=True)

        signals = [
            ("Hemoglobin",   hemo, 3,   18,   "g/dL"),
            ("Creatinine",   sc,   0,   15,   ""),
            ("Blood Urea",   bu,   10,  200,  "mg/dL"),
            ("Glucose",      bgr,  50,  500,  "mg/dL"),
            ("Sp. Gravity",  sg,   1.0, 1.03, ""),
        ]

        st.markdown('<div class="signals-heading">Key Signals</div>', unsafe_allow_html=True)
        for name, val, mn, mx, unit in signals:
            pct_bar = int(min(max((val - mn) / (mx - mn) * 100, 0), 100))
            display = f"{val} <span style='color:#A8CCCC;font-size:0.62rem;'>{unit}</span>" if unit else str(val)
            st.markdown(f"""
            <div class="signal-row">
                <span class="signal-name">{name}</span>
                <div class="signal-bar-wrap"><div class="signal-bar" style="width:{pct_bar}%;background:{bar_color};"></div></div>
                <span class="signal-val">{display}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<p class="disclaimer">For educational use only · Not a substitute for clinical diagnosis</p>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-panel result-idle" style="text-align:center;">
            <div style="font-size:2.2rem;margin-bottom:10px;color:#B2E4E4;">◎</div>
            <div style="font-size:0.82rem;font-weight:600;color:#7AADB0;">Awaiting Input</div>
            <div style="font-size:0.7rem;color:#A8CCCC;margin-top:4px;">Fill in patient data across<br>the three tabs, then analyze</div>
        </div>
        """, unsafe_allow_html=True)