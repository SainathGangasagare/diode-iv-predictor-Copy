import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="DiodeLab AI | I–V Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme and styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background:
      radial-gradient(circle at 8% 2%, rgba(56, 189, 248, .12), transparent 28%),
      radial-gradient(circle at 92% 10%, rgba(139, 92, 246, .14), transparent 30%),
      #080d18;
    color: #edf2ff;
}
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101a2e 0%, #0b1120 100%);
    border-right: 1px solid rgba(148,163,184,.16);
}
h1, h2, h3 { letter-spacing: -0.035em; }
.hero {
    padding: 2.2rem 2.3rem;
    border: 1px solid rgba(125, 211, 252, .25);
    border-radius: 24px;
    background: linear-gradient(120deg, rgba(14,165,233,.20), rgba(99,102,241,.18) 55%, rgba(139,92,246,.16));
    box-shadow: 0 18px 70px rgba(0,0,0,.22);
    animation: rise .7s ease-out both;
}
.eyebrow {
    display:inline-block; padding: .35rem .7rem; border-radius: 999px;
    border:1px solid rgba(125,211,252,.35); color:#a5e6ff;
    background:rgba(14,165,233,.10); font-size:.78rem; font-weight:700;
    letter-spacing:.08em; text-transform:uppercase;
}
.hero h1 { font-size: clamp(2.2rem, 5vw, 3.7rem); line-height:1.05; margin:.9rem 0; }
.gradient-text {
    background: linear-gradient(90deg, #67e8f9, #a5b4fc, #c4b5fd);
    -webkit-background-clip:text; background-clip:text; color:transparent;
}
.hero p { color:#cbd5e1; font-size:1.08rem; max-width:760px; line-height:1.75; }
.metric-card, .content-card {
    background: rgba(17, 27, 46, .78);
    border: 1px solid rgba(148,163,184,.17);
    border-radius: 18px; padding: 1.1rem 1.2rem;
    transition: transform .22s ease, border-color .22s ease;
}
.metric-card:hover, .content-card:hover { transform: translateY(-3px); border-color:rgba(103,232,249,.45); }
.metric-label { color:#9fb0c9; font-size:.85rem; }
.metric-value { color:#f8fafc; font-size:1.75rem; font-weight:800; margin-top:.25rem; }
.metric-note { color:#94a3b8; font-size:.8rem; margin-top:.2rem; }
.section-kicker { color:#67e8f9; text-transform:uppercase; font-size:.76rem; font-weight:800; letter-spacing:.13em; }
.small-muted { color:#9caec6; font-size:.9rem; }
div.stButton > button {
    border-radius: 11px; border: 1px solid rgba(125,211,252,.35);
    background: linear-gradient(90deg, #0284c7, #4f46e5); color:white;
    font-weight:700; padding:.65rem 1rem; transition: transform .2s ease, filter .2s ease;
}
div.stButton > button:hover { filter:brightness(1.12); transform:translateY(-1px); color:white; }
@keyframes rise { from {opacity:0; transform:translateY(14px)} to {opacity:1; transform:translateY(0)} }
hr { border-color:rgba(148,163,184,.18); }
[data-testid="stDataFrame"] { border:1px solid rgba(148,163,184,.2); border-radius:12px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)

# ---------- Load project assets ----------
@st.cache_resource
def load_models():
    loaded = {}
    errors = []
    for label, filename in [
        ("Random Forest", "random_forest_model.joblib"),
        ("Tuned SVR", "tuned_svr_model.joblib"),
    ]:
        if os.path.exists(filename):
            try:
                loaded[label] = joblib.load(filename)
            except Exception as exc:
                errors.append(f"{filename}: {exc}")
        else:
            errors.append(f"Missing file: {filename}")
    return loaded, errors

@st.cache_data
def load_data():
    path = "diode_iv_data.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Standardize the expected column names while preserving source values.
    if {"v_meas", "i_meas"}.issubset(df.columns):
        df = df[["v_meas", "i_meas"]].dropna().copy()
        df = df.sort_values("v_meas").reset_index(drop=True)
    return df

models, model_errors = load_models()
data = load_data()

if "page" not in st.session_state:
    st.session_state.page = "Home"

def go_to(page):
    st.session_state.page = page

with st.sidebar:
    st.markdown("## ⚡ DiodeLab AI")
    st.caption("Semiconductor ML explorer")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Home", "Predictor", "Model Comparison", "Dataset Explorer"],
        index=["Home", "Predictor", "Model Comparison", "Dataset Explorer"].index(st.session_state.page),
        key="navigation_radio",
    )
    st.session_state.page = page
    st.markdown("---")
    st.markdown("**PROJECT**")
    st.caption("1N4004 forward-bias I–V characteristics")
    st.caption("Machine learning • Regression • Experimental data")
    st.markdown("---")
    st.caption("Built with Python + Streamlit")

if model_errors:
    with st.expander("Asset loading notes", expanded=False):
        for error in model_errors:
            st.warning(error)

valid_data = not data.empty and {"v_meas", "i_meas"}.issubset(data.columns)
if valid_data:
    vmin = float(data["v_meas"].min())
    vmax = float(data["v_meas"].max())
    imin = float(data["i_meas"].min())
    imax = float(data["i_meas"].max())
else:
    vmin, vmax, imin, imax = 0.0, 0.815, 0.0, 0.05

# ---------- Shared plotting helper ----------
def draw_iv_curve(show_models=False, selected_model=None, highlight_voltage=None):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.patch.set_facecolor("#101827")
    ax.set_facecolor("#101827")
    if valid_data:
        ax.scatter(data["v_meas"], data["i_meas"], s=19, alpha=.78, label="Measured data")
    if show_models and models and valid_data:
        xs = np.linspace(vmin, vmax, 350).reshape(-1, 1)
        for label, model in models.items():
            if selected_model and label != selected_model:
                continue
            try:
                ys = np.asarray(model.predict(xs)).reshape(-1)
                ax.plot(xs.reshape(-1), ys, linewidth=2.2, label=f"{label} prediction")
            except Exception:
                pass
    if highlight_voltage is not None:
        ax.axvline(highlight_voltage, linestyle="--", linewidth=1.3, alpha=.8, label=f"Input: {highlight_voltage:.3f} V")
    ax.set_title("1N4004 forward-bias I–V characteristics", color="#e5edff", pad=12, fontsize=13)
    ax.set_xlabel("Measured voltage (V)", color="#cbd5e1")
    ax.set_ylabel("Measured current (A)", color="#cbd5e1")
    ax.tick_params(colors="#cbd5e1")
    for spine in ax.spines.values():
        spine.set_color("#475569")
    ax.grid(True, alpha=.18, color="#94a3b8")
    ax.legend(facecolor="#172235", edgecolor="#334155", labelcolor="#e5edff")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ---------- HOME ----------
if st.session_state.page == "Home":
    st.markdown("""
    <div class="hero">
      <span class="eyebrow">Experimental semiconductor data · Machine learning</span>
      <h1>Understand the diode.<br><span class="gradient-text">Predict its current.</span></h1>
      <p>Explore how a 1N4004 diode responds to forward voltage. Compare regression models trained on measured I–V data, make predictions, and inspect the dataset in one interactive workspace.</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Clean measurements</div><div class="metric-value">{len(data) if valid_data else "—"}</div><div class="metric-note">After removing missing rows</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Voltage range</div><div class="metric-value">{vmin:.3f} V</div><div class="metric-note">to {vmax:.3f} V</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Models available</div><div class="metric-value">{len(models)}</div><div class="metric-note">Saved trained estimators</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-label">Prediction type</div><div class="metric-value">I–V</div><div class="metric-note">Voltage in, current out</div></div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="section-kicker">Explore the workspace</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="content-card"><h3>⚡ Predictor</h3><p class="small-muted">Enter a voltage and estimate forward current using a saved model.</p></div>', unsafe_allow_html=True)
        if st.button("Open predictor →", key="home_predictor", use_container_width=True):
            go_to("Predictor")
            st.rerun()
    with b:
        st.markdown('<div class="content-card"><h3>📊 Model comparison</h3><p class="small-muted">Review held-out test metrics and compare prediction curves.</p></div>', unsafe_allow_html=True)
        if st.button("Compare models →", key="home_compare", use_container_width=True):
            go_to("Model Comparison")
            st.rerun()
    with c:
        st.markdown('<div class="content-card"><h3>🧪 Dataset explorer</h3><p class="small-muted">Filter measurements, inspect rows, and download the cleaned data.</p></div>', unsafe_allow_html=True)
        if st.button("Explore data →", key="home_data", use_container_width=True):
            go_to("Dataset Explorer")
            st.rerun()
    st.write("")
    st.markdown("### Measured diode response")
    if valid_data:
        draw_iv_curve()
    else:
        st.error("The dataset could not be loaded. Check that diode_iv_data.csv is in the repository root.")

# ---------- PREDICTOR ----------
elif st.session_state.page == "Predictor":
    st.markdown('<div class="section-kicker">Interactive inference</div>', unsafe_allow_html=True)
    st.title("Diode current predictor")
    st.write("Choose a trained model and enter a forward voltage. Predictions outside the measured voltage range are flagged because extrapolation may be unreliable.")
    if not models:
        st.error("No saved models loaded. Confirm the model .joblib files are in the repository root.")
    else:
        model_name = st.selectbox("Choose a model", list(models.keys()))
        safe_min = max(0.0, vmin)
        voltage = st.number_input("Measured voltage (V)", min_value=-0.1, max_value=max(1.5, vmax + 0.5), value=float(min(max(0.70, safe_min), vmax)), step=0.005, format="%.5f")
        if st.button("Predict current", key="predict_button"):
            try:
                current = float(np.asarray(models[model_name].predict(np.array([[voltage]]))).reshape(-1)[0])
                st.session_state.last_prediction = {"model": model_name, "voltage": voltage, "current": current}
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
        result = st.session_state.get("last_prediction")
        if result:
            st.markdown("### Prediction result")
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted current", f'{result["current"]:.6f} A')
            m2.metric("Equivalent current", f'{result["current"] * 1000:.3f} mA')
            m3.metric("Model", result["model"])
            if valid_data and not (vmin <= result["voltage"] <= vmax):
                st.warning(f'{result["voltage"]:.5f} V is outside the measured range ({vmin:.5f}–{vmax:.5f} V). Treat this prediction as an extrapolation and use caution.')
            else:
                st.success("Input voltage is within the measured voltage range. This is still a model estimate, not a guaranteed physical measurement.")
            if valid_data:
                st.markdown("### Model against measured data")
                draw_iv_curve(show_models=True, selected_model=result["model"], highlight_voltage=result["voltage"])
        elif valid_data:
            st.markdown("### Measured diode I–V curve")
            draw_iv_curve()

# ---------- MODEL COMPARISON ----------
elif st.session_state.page == "Model Comparison":
    st.markdown('<div class="section-kicker">Evaluation</div>', unsafe_allow_html=True)
    st.title("Model comparison")
    st.write("These are the recorded held-out random-test results from the project. Lower MAE and RMSE are better; R² closer to 1 indicates a better fit on this test set.")
    metrics = pd.DataFrame([
        {"Model": "Random Forest", "MAE (A)": 0.0001714702, "RMSE (A)": 0.0004224828, "R²": 0.9989055},
        {"Model": "Tuned SVR", "MAE (A)": 0.0002631377, "RMSE (A)": 0.0003458506, "R²": 0.9992665},
        {"Model": "Linear Regression", "MAE (A)": 0.006944, "RMSE (A)": 0.009559, "R²": 0.439677},
        {"Model": "Mean baseline", "MAE (A)": 0.008777, "RMSE (A)": 0.012815, "R²": -0.007052},
    ])
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Mean absolute error")
        st.bar_chart(metrics.set_index("Model")[["MAE (A)"]])
    with right:
        st.markdown("#### Root mean squared error")
        st.bar_chart(metrics.set_index("Model")[["RMSE (A)"]])
    st.info("Important limitation: these scores come from a random holdout within one measured voltage sweep. Nearby measurements may be correlated, and these results do not establish generalization to other diodes, temperatures, or measurement setups.")
    if valid_data and models:
        st.markdown("### Prediction curves across the measured range")
        selected = st.selectbox("Choose curve to display", list(models.keys()), key="comparison_model")
        draw_iv_curve(show_models=True, selected_model=selected)

# ---------- DATASET EXPLORER ----------
elif st.session_state.page == "Dataset Explorer":
    st.markdown('<div class="section-kicker">Experimental measurements</div>', unsafe_allow_html=True)
    st.title("Dataset explorer")
    if not valid_data:
        st.error("Could not load diode_iv_data.csv. Check that it is in the repository root and contains v_meas and i_meas columns.")
    else:
        st.write(f"The cleaned dataset contains **{len(data)} rows** with measured voltage and current.")
        lo, hi = st.slider("Filter voltage range (V)", min_value=float(vmin), max_value=float(vmax), value=(float(vmin), float(vmax)), step=max((vmax-vmin)/200, 0.0001), format="%.4f")
        filtered = data[data["v_meas"].between(lo, hi)].copy()
        q1, q2, q3 = st.columns(3)
        q1.metric("Rows shown", len(filtered))
        q2.metric("Voltage minimum", f"{filtered['v_meas'].min():.5f} V" if len(filtered) else "—")
        q3.metric("Maximum current", f"{filtered['i_meas'].max():.6f} A" if len(filtered) else "—")
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.download_button("Download filtered CSV", filtered.to_csv(index=False).encode("utf-8"), file_name="diode_iv_filtered.csv", mime="text/csv")
        st.markdown("### Filtered I–V curve")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.patch.set_facecolor("#101827")
        ax.set_facecolor("#101827")
        ax.scatter(filtered["v_meas"], filtered["i_meas"], s=24, color="#67e8f9", label="Filtered measurements")
        ax.set_xlabel("Measured voltage (V)", color="#cbd5e1")
        ax.set_ylabel("Measured current (A)", color="#cbd5e1")
        ax.set_title("Measured diode response", color="#e5edff")
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#475569")
        ax.grid(True, alpha=.18)
        ax.legend(facecolor="#172235", edgecolor="#334155", labelcolor="#e5edff")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

st.markdown("---")
st.markdown('<p class="small-muted">DiodeLab AI · Educational ML demonstration · Predictions are estimates based on one measured 1N4004 voltage sweep.</p>', unsafe_allow_html=True)

