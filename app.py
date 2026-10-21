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
    initial_sidebar_state="collapsed",
)

# ---------- Theme and styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
:root { color-scheme: dark; }
.stApp { background: radial-gradient(ellipse at 50% -18%, rgba(0,153,255,.22), transparent 48%), radial-gradient(circle at 85% 25%, rgba(0,210,255,.08), transparent 28%), #050b18; color:#eaf4ff; }
header[data-testid="stHeader"] { height: 0.25rem; background: transparent; }
.block-container { max-width: 1320px; padding-top: .55rem !important; padding-bottom: 2.5rem; }
[data-testid="stSidebar"] { display:none; }
h1,h2,h3,h4 { color:#f3f8ff; letter-spacing:-.025em; }
p, label, .stMarkdown { color:#b9cce3; }
.hero { position:relative; overflow:hidden; padding:clamp(2rem,5vw,4.5rem); border:1px solid rgba(70,174,255,.35); border-radius:24px; background:radial-gradient(ellipse at 78% 40%,rgba(0,149,255,.22),transparent 42%),linear-gradient(115deg,#0b1830 0%,#0b2445 58%,#071326 100%); box-shadow:0 20px 70px rgba(0,80,180,.16); animation:rise .7s ease-out both; }
.hero:after { content:' '; position:absolute; width:310px; height:310px; right:-85px; top:-115px; border-radius:50%; border:1px solid rgba(90,195,255,.18); box-shadow:0 0 0 28px rgba(90,195,255,.035),0 0 0 58px rgba(90,195,255,.025); pointer-events:none; }
.eyebrow { display:inline-block; padding:.4rem .75rem; border-radius:999px; border:1px solid rgba(83,190,255,.4); color:#79d8ff; background:rgba(10,100,180,.15); font-size:.72rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; }
.hero h1 { color:#f4f9ff; font-size:clamp(2.3rem,5.5vw,4.4rem); line-height:1.03; margin:1rem 0; }
.gradient-text { background:linear-gradient(90deg,#55d7ff,#4b91ff,#9b8cff); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero p { color:#b5cce5; font-size:1.05rem; max-width:780px; line-height:1.8; }
.metric-card,.content-card { background:linear-gradient(145deg,rgba(15,33,59,.96),rgba(8,20,39,.96)); border:1px solid #203b5d; border-radius:16px; padding:1.1rem 1.2rem; box-shadow:0 8px 24px rgba(0,0,0,.12); transition:transform .2s ease,box-shadow .2s ease,border-color .2s ease; }
.metric-card:hover,.content-card:hover { transform:translateY(-3px); border-color:#38bdf8; box-shadow:0 12px 30px rgba(0,145,255,.12); }
.metric-label { color:#9bb5d2; font-size:.85rem; }.metric-value { color:#f2f8ff; font-size:1.75rem; font-weight:800; margin-top:.25rem; }.metric-note { color:#7995b6; font-size:.8rem; margin-top:.2rem; }
.section-kicker { color:#5bd5ff; text-transform:uppercase; font-size:.76rem; font-weight:800; letter-spacing:.14em; }.small-muted { color:#a8bfd9; font-size:.9rem; }
div.stButton > button, div[data-testid="stDownloadButton"] button { border-radius:10px; border:1px solid #168de0; background:linear-gradient(100deg,#0878db,#05a6cf); color:#fff !important; font-weight:700; padding:.72rem 1rem; transition:transform .2s ease,filter .2s ease; }
div.stButton > button p, div[data-testid="stDownloadButton"] button p { color:#fff !important; font-weight:700; }
div.stButton > button:hover, div[data-testid="stDownloadButton"] button:hover { filter:brightness(1.15); transform:translateY(-2px); color:#fff !important; border-color:#5bd5ff; }
[data-testid="stMetric"] { background:#0c1c33; border:1px solid #203b5d; padding:1rem; border-radius:14px; }
[data-testid="stDataFrame"] { border:1px solid #203b5d; border-radius:12px; overflow:hidden; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background:#0c1c33; border-color:#315274; color:#eaf4ff; }
.stSlider [data-baseweb="slider"] div { color:#eaf4ff; }
[data-testid="stAlert"] { border-radius:12px; }
@keyframes rise { from {opacity:0; transform:translateY(12px)} to {opacity:1; transform:translateY(0)} }
hr { border-color:#203b5d; }
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
        ("Silicon SVR", "trained_models/silicon_svr.joblib"),
        ("SiC SVR", "trained_models/silicon_carbide_svr.joblib"),
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


@st.cache_data
def load_training_data():
    path = "data/processed/classified_diode_measurements.csv"

    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path)

    required_columns = {
        "device",
        "material",
        "voltage_V",
        "current_A",
        "temperature_K",
    }

    if not required_columns.issubset(df.columns):
        return pd.DataFrame()

    df = df.dropna(
        subset=["device", "voltage_V", "current_A", "temperature_K"]
    ).copy()

    return df


models, model_errors = load_models()
data = load_data()
training_data = load_training_data()

PAGES = ["Home", "Predictor", "Model Comparison", "Dataset Explorer"]

if "page" not in st.session_state:
    st.session_state.page = "Home"
if "navigation_radio" not in st.session_state:
    st.session_state.navigation_radio = st.session_state.page

def go_to(page):
    # Streamlit runs this callback before redrawing the page.
    st.session_state.page = page
    st.session_state.navigation_radio = page

def sync_page_from_radio():
    st.session_state.page = st.session_state.navigation_radio

# Compact top navigation replaces the crowded sidebar.
nav_cols = st.columns([1.45, 1, 1.25, 1.25, 1.15])
with nav_cols[0]:
    st.markdown("<div style='padding:.55rem 0;color:#eaf4ff;font-weight:800;font-size:1.05rem'>⚡ DiodeLab <span style='color:#54d5ff'>AI</span></div>", unsafe_allow_html=True)
for col, target, label in zip(nav_cols[1:], PAGES, ["⌂  Home", "⚡  Predictor", "◫  Model comparison", "⌁  Dataset"]):
    with col:
        st.button(label, key=f"topnav_{target}", use_container_width=True, on_click=go_to, args=(target,))

page = st.session_state.page

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
    fig.patch.set_facecolor("#081326")
    ax.set_facecolor("#081326")
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
    ax.set_title("1N4004 forward-bias I–V characteristics", color="#eaf4ff", pad=12, fontsize=13)
    ax.set_xlabel("Measured voltage (V)", color="#a9c4df")
    ax.set_ylabel("Measured current (A)", color="#a9c4df")
    ax.tick_params(colors="#a9c4df")
    for spine in ax.spines.values():
        spine.set_color("#294563")
    ax.grid(True, alpha=.25, color="#52708e")
    ax.legend(facecolor="#10223b", edgecolor="#294563", labelcolor="#eaf4ff")
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
        st.button("Open predictor →", key="home_predictor", use_container_width=True,
                  on_click=go_to, args=("Predictor",))
    with b:
        st.markdown('<div class="content-card"><h3>📊 Model comparison</h3><p class="small-muted">Review held-out test metrics and compare prediction curves.</p></div>', unsafe_allow_html=True)
        st.button("Compare models →", key="home_compare", use_container_width=True,
                  on_click=go_to, args=("Model Comparison",))
    with c:
        st.markdown('<div class="content-card"><h3>🧪 Dataset explorer</h3><p class="small-muted">Filter measurements, inspect rows, and download the cleaned data.</p></div>', unsafe_allow_html=True)
        st.button("Explore data →", key="home_data", use_container_width=True,
                  on_click=go_to, args=("Dataset Explorer",))
    st.write("")
    st.markdown("### Measured diode response")
    if valid_data:
        draw_iv_curve()
    else:
        st.error("The dataset could not be loaded. Check that diode_iv_data.csv is in the repository root.")

# ---------- PREDICTOR ----------

elif st.session_state.page == "Predictor":
    st.markdown(
        '<div class="section-kicker">Interactive inference</div>',
        unsafe_allow_html=True,
    )
    st.title("Diode current predictor")
    st.write(
        "Choose a trained model and enter a voltage. "
        "Predictions outside the measured voltage range may be unreliable."
    )

    if not models:
        st.error(
            "No saved models loaded. Confirm the model .joblib files "
            "are in the repository."
        )
    else:
        model_name = st.selectbox("Choose a model", list(models.keys()))

        # No explicit minimum/maximum: users may enter voltages outside
        # the measured range. The app warns about extrapolation below.
        voltage = st.number_input(
            "Measured voltage (V)",
            value=0.70,
            step=0.005,
            format="%.5f",
        )

       
        
        if model_name == "Silicon SVR":
            device_options = [
                "1N11877R",
                "1N1190",
                "1N3913",
                "45L120",
                "BYX661000",
                "BZW12360",
                "BZW1336",
                "BZW1347",
                "MBR4035",
            ]
        elif model_name == "SiC SVR":
            device_options = [
                "C6D20065A",
                "FFSH3065B",
                "FFSH3065B-F085",
                "STPSC12065DY",
            ]
        else:
            device_options = [
                "1N4004",
                "Generic silicon",
                "Silicon carbide",
                "GaN Schottky",
            ]

        device = st.selectbox("Device", device_options)


        temperature = st.number_input(
            "Temperature (K)",
            min_value=1.0,
            max_value=500.0,
            value=293.0,
            step=1.0,
        )
        
        if model_name in ["Silicon SVR", "SiC SVR"]:
            if not 77.0 <= temperature <= 293.0:
                st.warning(
                    "The SVR models were trained on data from "
                    "77 K to 293 K. Predictions outside this "
                    "temperature range may be unreliable."
                )


        
        
        
        
        if st.button("Predict current", key="predict_button"):
            # Clear the previous result before attempting a new prediction.
            st.session_state.pop("last_prediction", None)

            try:
                if model_name in ["Silicon SVR", "SiC SVR"]:
                    if training_data.empty:
                        st.error("The training dataset could not be loaded.")
                    else:
                        device_data = training_data[
                            (training_data["device"] == device)
                            & (training_data["temperature_K"] == temperature)
                            & (training_data["current_A"] > 0)
                        ]

                        if device_data.empty:
                            st.error(
                                f"No measured data is available for {device} "
                                f"at exactly {temperature:g} K. "
                                "Choose a temperature with measured data."
                            )
                        else:
                            min_voltage = float(
                                device_data["voltage_V"].min()
                            )
                            max_voltage = float(
                                device_data["voltage_V"].max()
                            )

                            # Allow prediction outside the measured voltage range,
                            # but explicitly warn that this is extrapolation.
                            if not min_voltage <= voltage <= max_voltage:
                                st.warning(
                                    f"{voltage:.5f} V is outside the measured range "
                                    f"for {device} at {temperature:g} K "
                                    f"({min_voltage:.5f} to {max_voltage:.5f} V). "
                                    "The model will still predict, but the result may be unreliable."
                                )

                            model_input = pd.DataFrame([{
                                "device": device,
                                "voltage_V": voltage,
                                "temperature_K": temperature,
                            }])

                            log_current = float(
                                models[model_name].predict(model_input)[0]
                            )
                            current = 10 ** log_current

                            st.session_state.last_prediction = {
                                "model": model_name,
                                "device": device,
                                "voltage": voltage,
                                "temperature": temperature,
                                "current": current,
                            }

                else:
                    # Original models use voltage as their only input.
                    current = float(
                        np.asarray(
                            models[model_name].predict(
                                np.array([[voltage]])
                            )
                        ).reshape(-1)[0]
                    )

                    if not np.isfinite(current):
                        st.error("The model returned a non-finite prediction.")
                    else:
                        st.session_state.last_prediction = {
                            "model": model_name,
                            "device": device,
                            "voltage": voltage,
                            "temperature": temperature,
                            "current": current,
                        }

            except Exception as exc:
                st.session_state.pop("last_prediction", None)
                st.error(f"Prediction failed: {exc}")

        # Show a result only when it matches the current inputs.
        result = st.session_state.get("last_prediction")

        if result and (
            result["model"] != model_name
            or result["device"] != device
            or result["voltage"] != voltage
            or result["temperature"] != temperature
        ):
            result = None

        if result:
            st.markdown("### Prediction result")

            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Predicted current",
                f'{result["current"]:.6g} A',
            )
            m2.metric(
                "Equivalent current",
                f'{result["current"] * 1000:.6g} mA',
            )
            m3.metric("Model", result["model"])

            if result["model"] in ["Silicon SVR", "SiC SVR"]:
                st.info(
                    f"Estimate for {result['device']} at "
                    f"{result['temperature']:g} K. "
                    "The voltage is within the measured range for this "
                    "device and temperature, but the estimate is not "
                    "a guaranteed physical measurement."
                )
            elif valid_data and not (
                vmin <= result["voltage"] <= vmax
            ):
                st.warning(
                    f'{result["voltage"]:.5f} V is outside the original '
                    f"1N4004 measured range ({vmin:.5f}–{vmax:.5f} V). "
                    "Treat this prediction as an extrapolation."
                )
            else:
                st.info(
                    "This is a model estimate, not a guaranteed "
                    "physical measurement."
                )

            with st.expander("How should I interpret this prediction?"):
                st.write(f"Model: {result['model']}")
                st.write(f"Device: {result['device']}")
                st.write(f"Voltage: {result['voltage']:.5f} V")
                st.write(f"Temperature: {result['temperature']:g} K")
                st.write(
                    f"Estimated current: {result['current']:.6g} A "
                    f"({result['current'] * 1000:.6g} mA)"
                )
                st.write(
                    "A prediction inside the measured voltage range can "
                    "still be inaccurate. These models are estimates, "
                    "not substitutes for physical measurements."
                )

            if valid_data:
                st.markdown("### Original 1N4004 measured I–V curve")
                draw_iv_curve(
                    highlight_voltage=result["voltage"]
                )

        elif valid_data:
            st.markdown("### Original 1N4004 measured I–V curve")
            draw_iv_curve()



# ---------- MODEL COMPARISON ----------
elif st.session_state.page == "Model Comparison":
    st.markdown('<div class="section-kicker">Evaluation</div>', unsafe_allow_html=True)
    st.title("Model comparison")
    st.write(
    "These are previously recorded held-out random-test results for the "
    "1N4004 dataset. The metrics are displayed from saved values rather "
    "than recalculated on each app run. Lower MAE and RMSE are better; "
    "R² closer to 1 indicates a better fit on this test set."
    )
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
    st.write(
        "Explore the experimental datasets included in this repository. "
        "Displaying a dataset here does not mean a prediction model was trained on it."
    )

    dataset_files = {
        "Original 1N4004 cleaned data": "diode_iv_data.csv",
        "1N4004 forward bias": "data/external/1n4004_forward.csv",
        "1N4004 reverse bias": "data/external/1n4004_reverse.csv",
        "1N4004 reverse bias (10 µA range)": "data/external/1n4004_reverse_10uA_range.csv",
        "1N5229B Zener reverse bias": "data/external/1n5229b_reverse.csv",
        "SB560 forward bias": "data/external/sb560_forward.csv",
        "SB560 reverse bias": "data/external/sb560_reverse.csv",
        "Cryogenic diode measurements": "data/external/cryogenic_diode_measurements.csv",
        "GaN Schottky measurements": "data/external/gan_schottky_measurements.csv",
        "Classified combined measurements": "data/processed/classified_diode_measurements.csv",
        "External measurements review": "data/processed/external_iv_measurements_review.csv",
    }

    selected_name = st.selectbox("Choose a dataset", list(dataset_files.keys()))
    selected_path = dataset_files[selected_name]

    if os.path.exists(selected_path):
        try:
            df = pd.read_csv(selected_path)
            st.write(f"**Dataset:** {selected_name}")
            st.write(f"**Rows:** {len(df)} · **Columns:** {len(df.columns)}")
            st.dataframe(df.head(1000), use_container_width=True, hide_index=True)
            st.download_button(
                "Download selected dataset as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=os.path.basename(selected_path),
                mime="text/csv",
            )

            voltage_candidates = ["v_meas", "voltage_V", "voltage", "Voltage", "V", "voltage (V)", "Vd", "voltage_v"]
            current_candidates = ["i_meas", "current_A", "current", "Current", "I", "current (A)", "Id", "current_a"]
            voltage_col = next((c for c in voltage_candidates if c in df.columns), None)
            current_col = next((c for c in current_candidates if c in df.columns), None)

            if voltage_col and current_col:
                plot_data = df[[voltage_col, current_col]].copy()
                plot_data[voltage_col] = pd.to_numeric(plot_data[voltage_col], errors="coerce")
                plot_data[current_col] = pd.to_numeric(plot_data[current_col], errors="coerce")
                plot_data = plot_data.dropna()
                if not plot_data.empty:
                    st.subheader("Measured I–V curve")
                    fig, ax = plt.subplots(figsize=(9, 4.5))
                    fig.patch.set_facecolor("#081326")
                    ax.set_facecolor("#081326")
                    ax.scatter(plot_data[voltage_col], plot_data[current_col], s=18, alpha=0.75, color="#38bdf8")
                    ax.set_xlabel(f"{voltage_col} (voltage)", color="#a9c4df")
                    ax.set_ylabel(f"{current_col} (current)", color="#a9c4df")
                    ax.set_title(selected_name, color="#eaf4ff")
                    ax.tick_params(colors="#a9c4df")
                    for spine in ax.spines.values():
                        spine.set_color("#294563")
                    ax.grid(True, alpha=0.25, color="#52708e")
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
            else:
                st.info(
                    "This dataset is available for inspection and download, but its column names "
                    "do not match the automatic voltage/current plotting patterns."
                )
        except Exception as exc:
            st.error(f"Could not read this dataset: {exc}")
    else:
        st.error(
            f"Dataset file not found: {selected_path}. Check that the file exists "
            "at this path in the GitHub repository."
        )

st.markdown("---")
st.markdown('<p class="small-muted">DiodeLab AI · Educational ML demonstration · Predictions are estimates based on one measured 1N4004 voltage sweep.</p>', unsafe_allow_html=True)


