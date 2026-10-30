import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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



# ---------- Discover all available measurement CSVs ----------
DATASET_FILES = {
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
    "Original cleaned 1N4004": "diode_iv_data.csv",
}

@st.cache_data
def read_dataset_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def voltage_current_columns(frame):
    """Return recognized voltage/current column names, or (None, None)."""
    pairs = [
        ("v_meas", "i_meas"), ("voltage_V", "current_A"),
        ("V_V", "I_A"), ("V", "I"), ("voltage", "current"),
    ]
    for v_col, i_col in pairs:
        if v_col in frame.columns and i_col in frame.columns:
            return v_col, i_col
    return None, None

def material_label(dataset_name, frame):
    if "material" in frame.columns:
        vals = frame["material"].dropna().astype(str).unique()
        if len(vals) == 1:
            return vals[0]
        if len(vals) > 1:
            return "Multiple materials"
    return dataset_name


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
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Clean measurements</div><div class="metric-value">{len(data) if valid_data else "—"}</div><div class="metric-note">After removing missing rows</div></div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Voltage range</div><div class="metric-value">{vmin:.3f} V</div><div class="metric-note">to {vmax:.3f} V</div></div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Models available</div><div class="metric-value">{len(models)}</div><div class="metric-note">Saved trained estimators</div></div>',
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">Prediction type</div><div class="metric-value">I–V</div><div class="metric-note">Voltage in, current out</div></div>',
            unsafe_allow_html=True
        )

    st.write("")
    st.markdown(
        '<div class="section-kicker">Explore the workspace</div>',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    with a:
        st.markdown(
            '<div class="content-card"><h3>⚡ Predictor</h3><p class="small-muted">Enter a voltage and estimate forward current using a saved model.</p></div>',
            unsafe_allow_html=True
        )
        st.button(
            "Open predictor →",
            key="home_predictor",
            use_container_width=True,
            on_click=go_to,
            args=("Predictor",)
        )

    with b:
        st.markdown(
            '<div class="content-card"><h3>📊 Model comparison</h3><p class="small-muted">Review held-out test metrics and compare prediction curves.</p></div>',
            unsafe_allow_html=True
        )
        st.button(
            "Compare models →",
            key="home_compare",
            use_container_width=True,
            on_click=go_to,
            args=("Model Comparison",)
        )

    with c:
        st.markdown(
            '<div class="content-card"><h3>🧪 Dataset explorer</h3><p class="small-muted">Filter measurements, inspect rows, and download the cleaned data.</p></div>',
            unsafe_allow_html=True
        )
        st.button(
            "Explore data →",
            key="home_data",
            use_container_width=True,
            on_click=go_to,
            args=("Dataset Explorer",)
        )

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

        # No min_value/max_value: the user can enter any voltage.
        # Predictions outside the training range are extrapolations and may be unreliable.
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





        st.markdown("### Interactive I–V graph")
        st.caption(
            "Use the graph toolbar to zoom, pan, or reset. Hover over a curve to see "
            "the exact voltage and current. The gold star marks the latest successful "
            "prediction for the current inputs. Choose Forward or Reverse view to avoid "
            "the negative-voltage range compressing the positive-voltage region."
        )

        # Keep the existing dataset choices; this update does not add or retrain datasets.
        compare_candidates = [
            "1N4004 forward bias",
            "1N4004 reverse bias",
            "1N4004 reverse bias (10 µA range)",
            "1N5229B Zener reverse bias",
            "SB560 forward bias",
            "SB560 reverse bias",
            "Cryogenic diode measurements",
            "GaN Schottky measurements",
        ]
        check_cols = st.columns(4)
        selected_curves = []
        for idx, dataset_name in enumerate(compare_candidates):
            with check_cols[idx % 4]:
                default_checked = dataset_name == (
                    "1N4004 forward bias" if voltage >= 0 else "1N4004 reverse bias"
                )
                if st.checkbox(
                    dataset_name,
                    value=default_checked,
                    key=f"curve_check_{dataset_name}",
                ):
                    selected_curves.append(dataset_name)

        # Load and plot only datasets selected above; the underlying CSV files are unchanged.
        curve_data = []
        for dataset_name in selected_curves:
            frame = read_dataset_csv(DATASET_FILES[dataset_name])
            v_col, i_col = voltage_current_columns(frame)
            if v_col is None:
                continue

            curve_label = dataset_name
            temperature_col = next(
                (col for col in ("temperature_K", "T_K") if col in frame.columns), None
            )
            if temperature_col is not None and dataset_name in (
                "Cryogenic diode measurements", "GaN Schottky measurements"
            ):
                frame = frame.copy()
                frame[temperature_col] = pd.to_numeric(frame[temperature_col], errors="coerce")
                available_temps = frame[temperature_col].dropna().unique()
                if len(available_temps):
                    nearest_temp = min(
                        available_temps,
                        key=lambda t: abs(float(t) - float(temperature)),
                    )
                    frame = frame[frame[temperature_col] == nearest_temp].copy()
                    curve_label = f"{dataset_name} ({nearest_temp:g} K)"

            curve = frame[[v_col, i_col]].copy()
            curve.columns = ["voltage", "current"]
            curve["voltage"] = pd.to_numeric(curve["voltage"], errors="coerce")
            curve["current"] = pd.to_numeric(curve["current"], errors="coerce")
            curve = (
                curve.replace([np.inf, -np.inf], np.nan)
                .dropna()
                .sort_values("voltage")
            )
            if not curve.empty:
                curve_data.append((curve_label, curve))

        result_for_graph = st.session_state.get("last_prediction")
        result_matches_inputs = bool(
            result_for_graph
            and result_for_graph.get("model") == model_name
            and result_for_graph.get("device") == device
            and result_for_graph.get("voltage") == voltage
            and result_for_graph.get("temperature") == temperature
        )
        if result_matches_inputs:
            try:
                result_matches_inputs = np.isfinite(float(result_for_graph["current"]))
            except (TypeError, ValueError, KeyError):
                result_matches_inputs = False

        all_voltages = []
        for _, curve in curve_data:
            all_voltages.extend(curve["voltage"].astype(float).tolist())
        if result_matches_inputs:
            all_voltages.append(float(result_for_graph["voltage"]))

        if all_voltages:
            data_vmin, data_vmax = float(min(all_voltages)), float(max(all_voltages))
            if data_vmin == data_vmax:
                data_vmin -= 0.1
                data_vmax += 0.1
        else:
            data_vmin, data_vmax = -1.0, 1.0

        view_mode = st.selectbox(
            "Voltage view",
            [
                "Full voltage range",
                "Forward bias (0 V and above)",
                "Reverse bias (below 0 V)",
                "Zoom around selected voltage",
                "Custom voltage range",
            ],
            key="iv_graph_view",
        )

        if view_mode == "Full voltage range":
            x_range = [data_vmin, data_vmax]
        elif view_mode == "Forward bias (0 V and above)":
            positive = [v for v in all_voltages if v >= 0]
            x_range = [0.0, max(positive)] if positive else [0.0, 1.0]
            if x_range[0] == x_range[1]:
                x_range[1] = x_range[0] + 0.1
        elif view_mode == "Reverse bias (below 0 V)":
            negative = [v for v in all_voltages if v < 0]
            x_range = [min(negative), 0.0] if negative else [-1.0, 0.0]
        elif view_mode == "Zoom around selected voltage":
            width = st.slider(
                "Voltage window width (V)",
                min_value=0.01,
                max_value=10.0,
                value=0.5,
                step=0.01,
                key="iv_zoom_width",
            )
            x_range = [float(voltage) - width / 2, float(voltage) + width / 2]
        else:
            # Number inputs allow a custom range even when data spans a wide voltage interval.
            left_col, right_col = st.columns(2)
            with left_col:
                x_min = st.number_input(
                    "Minimum visible voltage (V)",
                    value=float(data_vmin),
                    key="iv_custom_x_min",
                )
            with right_col:
                x_max = st.number_input(
                    "Maximum visible voltage (V)",
                    value=float(data_vmax),
                    key="iv_custom_x_max",
                )
            if x_min >= x_max:
                st.warning("The maximum voltage must be greater than the minimum voltage.")
                x_range = [data_vmin, data_vmax]
            else:
                x_range = [float(x_min), float(x_max)]

        # Plotly does not offer a native symmetric-log axis. Transform y positions for
        # display, while hover tooltips continue to show the original current in amperes.
        linthresh = 1e-9

        def symlog_transform(values):
            values = np.asarray(values, dtype=float)
            return np.sign(values) * np.log10(1.0 + np.abs(values) / linthresh)

        fig = go.Figure()
        for curve_label, curve in curve_data:
            currents = curve["current"].to_numpy(dtype=float)
            fig.add_trace(
                go.Scatter(
                    x=curve["voltage"],
                    y=symlog_transform(currents),
                    customdata=currents,
                    mode="lines+markers",
                    name=curve_label,
                    line={"width": 2},
                    marker={"size": 4},
                    hovertemplate=(
                        "Voltage: %{x:.6g} V<br>Current: %{customdata:.6g} A"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )

        if result_matches_inputs:
            pred_v = float(result_for_graph["voltage"])
            pred_i = float(result_for_graph["current"])
            fig.add_trace(
                go.Scatter(
                    x=[pred_v],
                    y=symlog_transform([pred_i]),
                    customdata=[pred_i],
                    mode="markers",
                    name="Predicted current",
                    marker={
                        "symbol": "star",
                        "size": 18,
                        "color": "#ffcc66",
                        "line": {"color": "white", "width": 1.5},
                    },
                    hovertemplate=(
                        "<b>MODEL PREDICTION</b><br>Voltage: %{x:.6g} V<br>"
                        "Predicted current: %{customdata:.6g} A<extra></extra>"
                    ),
                )
            )

        fig.add_vline(
            x=float(voltage),
            line_dash="dash",
            line_width=1.3,
            line_color="#9dbcf4",
            annotation_text=f"Selected: {voltage:.4g} V",
            annotation_position="top",
        )

        tick_currents = np.array([
            -1.0, -0.1, -0.01, -0.001, -1e-4, -1e-5, -1e-6,
            -1e-7, -1e-8, -1e-9, 0.0, 1e-9, 1e-8, 1e-7,
            1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1.0,
        ])
        fig.update_layout(
            title="Diode I–V characteristics",
            template="plotly_dark",
            height=650,
            autosize=True,
            margin={"l": 85, "r": 30, "t": 75, "b": 100},
            paper_bgcolor="#081326",
            plot_bgcolor="#081326",
            hovermode="closest",
            legend={
                "bgcolor": "rgba(16,34,59,0.9)",
                "bordercolor": "#294563",
                "borderwidth": 1,
            },
            xaxis={
                "title": "Voltage (V) — negative: reverse bias; positive: forward bias",
                "range": x_range,
                "showgrid": True,
                "gridcolor": "rgba(82,112,142,0.3)",
                "zeroline": True,
                "zerolinecolor": "#7890ad",
                "automargin": True,
            },
            yaxis={
                "title": "Current (A, symmetric-log scale)",
                "tickmode": "array",
                "tickvals": symlog_transform(tick_currents),
                "ticktext": [f"{value:.0e}" if value else "0" for value in tick_currents],
                "showgrid": True,
                "gridcolor": "rgba(82,112,142,0.3)",
                "zeroline": True,
                "zerolinecolor": "#7890ad",
                "automargin": True,
            },
        )

        if not curve_data and not result_matches_inputs:
            st.info(
                "Select one or more datasets above to display curves. "
                "The prediction marker appears after a successful prediction."
            )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"scrollZoom": True, "displaylogo": False, "responsive": True},
            key="interactive_iv_plot",
        )

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
    available = {
        name: path for name, path in DATASET_FILES.items()
        if os.path.exists(path)
    }
    if not available:
        st.error("No CSV datasets were found in the expected repository folders.")
    else:
        chosen = st.selectbox("Choose a dataset", list(available.keys()))
        frame = read_dataset_csv(available[chosen])
        if frame.empty:
            st.error("This CSV could not be read or contains no rows.")
        else:
            st.write(f"**{chosen}** — {len(frame):,} rows and {len(frame.columns)} columns.")
            st.dataframe(frame.head(500), use_container_width=True, hide_index=True)
            st.download_button(
                "Download this dataset",
                frame.to_csv(index=False).encode("utf-8"),
                file_name=os.path.basename(available[chosen]),
                mime="text/csv",
            )
            v_col, i_col = voltage_current_columns(frame)
            if v_col is not None:
                plot_frame = frame[[v_col, i_col]].copy()
                plot_frame[v_col] = pd.to_numeric(plot_frame[v_col], errors="coerce")
                plot_frame[i_col] = pd.to_numeric(plot_frame[i_col], errors="coerce")
                plot_frame = plot_frame.dropna().sort_values(v_col)
                st.markdown("### I–V curve")
                fig, ax = plt.subplots(figsize=(10, 4.8))
                fig.patch.set_facecolor("#081326")
                ax.set_facecolor("#081326")
                ax.plot(plot_frame[v_col], plot_frame[i_col], marker=".", markersize=3, linewidth=1.5, label=chosen)
                ax.set_xlabel("Voltage (V)", color="#a9c4df")
                ax.set_ylabel("Current (A)", color="#a9c4df")
                ax.set_title(f"{chosen}: measured I–V curve", color="#eaf4ff")
                ax.tick_params(colors="#a9c4df")
                for spine in ax.spines.values():
                    spine.set_color("#294563")
                ax.grid(True, alpha=.25, color="#52708e")
                ax.legend(facecolor="#10223b", edgecolor="#294563", labelcolor="#eaf4ff")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("This dataset does not contain a recognized voltage/current column pair.")

st.markdown("---")
st.markdown('<p class="small-muted">DiodeLab AI · Educational ML demonstration · Predictions are estimates based on one measured 1N4004 voltage sweep.</p>', unsafe_allow_html=True)



