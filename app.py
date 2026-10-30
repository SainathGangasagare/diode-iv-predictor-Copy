import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="DiodeLab AI | I–V Predictor", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp {background:radial-gradient(ellipse at 50% -18%,rgba(0,153,255,.18),transparent 48%),#050b18;color:#eaf4ff}
.block-container{max-width:1320px;padding-top:1rem!important;padding-bottom:2rem}
[data-testid="stSidebar"]{display:none} h1,h2,h3{color:#f3f8ff} p,label{color:#b9cce3}
div.stButton>button,div[data-testid="stDownloadButton"] button{border-radius:10px;border:1px solid #168de0;background:linear-gradient(100deg,#0878db,#05a6cf);color:white!important;font-weight:700}
[data-testid="stMetric"]{background:#0c1c33;border:1px solid #203b5d;padding:.8rem;border-radius:12px}
</style>
""", unsafe_allow_html=True)

# Older saved models refer to this function when unpickling.
def inverse_log_transform(values):
    return np.power(10.0, values)

ROOT = "."

@st.cache_data
def read_csv_safe(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

@st.cache_resource
def load_models():
    result, errors = {}, []
    paths = {
        "1N4004 · Random Forest": "random_forest_model.joblib",
        "1N4004 · Tuned SVR": "tuned_svr_model.joblib",
        "Silicon devices · SVR": "trained_models/silicon_svr.joblib",
        "SiC devices · SVR": "trained_models/silicon_carbide_svr.joblib",
        "GaN · Temperature-aware SVR": "trained_models/gan_temperature_aware_svr.joblib",
    }
    for label, path in paths.items():
        if not os.path.exists(path):
            errors.append(f"{label}: file not found ({path})")
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result[label] = joblib.load(path)
        except Exception as exc:
            errors.append(f"{label}: could not load model ({type(exc).__name__}: {exc})")
    return result, errors

models, model_errors = load_models()
main_data = read_csv_safe("diode_iv_data.csv")
if {"v_meas", "i_meas"}.issubset(main_data.columns):
    main_data = main_data[["v_meas", "i_meas"]].dropna().copy()
else:
    main_data = pd.DataFrame(columns=["v_meas", "i_meas"])

classified = read_csv_safe("data/processed/classified_diode_measurements.csv")
if not {"device", "material", "voltage_V", "current_A", "temperature_K"}.issubset(classified.columns):
    classified = pd.DataFrame(columns=["device", "material", "voltage_V", "current_A", "temperature_K"])
else:
    classified = classified.dropna(subset=["device", "material", "voltage_V", "current_A", "temperature_K"]).copy()

# Load additional measured curves, including reverse/negative-voltage measurements.
def load_1n4004_combined():
    fwd = read_csv_safe("data/external/1n4004_forward.csv")
    rev = read_csv_safe("data/external/1n4004_reverse.csv")
    frames = []
    for df in [fwd, rev]:
        if {"v_meas", "i_meas"}.issubset(df.columns):
            frames.append(df[["v_meas", "i_meas"]].dropna().rename(columns={"v_meas":"voltage_V", "i_meas":"current_A"}))
    if frames:
        return pd.concat(frames, ignore_index=True).drop_duplicates().sort_values("voltage_V")
    if not main_data.empty:
        return main_data.rename(columns={"v_meas":"voltage_V", "i_meas":"current_A"}).sort_values("voltage_V")
    return pd.DataFrame(columns=["voltage_V", "current_A"])

one_n4004 = load_1n4004_combined()

PAGES = ["Home", "Predictor", "Graphs & Data", "Model Comparison"]
if "page" not in st.session_state:
    st.session_state.page = "Home"
cols = st.columns([1.5, 1, 1, 1, 1])
with cols[0]: st.markdown("### ⚡ DiodeLab **AI**")
for c, p, label in zip(cols[1:], PAGES, ["Home", "Predictor", "Graphs & Data", "Model Comparison"]):
    with c:
        if st.button(label, key="nav_"+p, use_container_width=True): st.session_state.page = p
page = st.session_state.page

if model_errors:
    with st.expander(f"Model loading notes ({len(model_errors)})", expanded=False):
        st.caption("A listed model is unavailable to the predictor until its file loads successfully. Graphs of measured data can still work.")
        for e in model_errors: st.warning(e)

# Device-first mapping: model choices are shown only for the matching device/material family.
def choices_for_device(device, material):
    if device == "1N4004":
        return [x for x in ["1N4004 · Random Forest", "1N4004 · Tuned SVR"] if x in models]
    if material == "gallium nitride":
        return [x for x in ["GaN · Temperature-aware SVR"] if x in models]
    if material == "silicon carbide":
        return [x for x in ["SiC devices · SVR"] if x in models]
    if material == "silicon":
        return [x for x in ["Silicon devices · SVR"] if x in models]
    return []

def get_device_rows(device, material):
    if device == "1N4004":
        return one_n4004.copy()
    if classified.empty:
        return pd.DataFrame(columns=["voltage_V", "current_A", "temperature_K"])
    d = classified[(classified["device"].astype(str) == str(device)) & (classified["material"].astype(str) == str(material))].copy()
    return d

def apply_y_scale(ax, y, scale):
    y = np.asarray(y, dtype=float)
    if scale == "Log (positive current only)":
        positive = y[np.isfinite(y) & (y > 0)]
        if len(positive): ax.set_yscale("log")
        else: ax.text(.5,.5,"No positive current values to display on log scale",transform=ax.transAxes,ha="center")
    elif scale == "Symmetric log (shows zero/negative current)":
        ax.set_yscale("symlog", linthresh=1e-9)

def style_plot(ax, title, y_label, scale, yvalues):
    ax.set_title(title, color="#eaf4ff")
    ax.set_xlabel("Voltage (V)", color="#a9c4df")
    ax.set_ylabel(y_label, color="#a9c4df")
    ax.tick_params(colors="#a9c4df")
    ax.grid(True, alpha=.25)
    for sp in ax.spines.values(): sp.set_color("#294563")
    apply_y_scale(ax, yvalues, scale)

if page == "Home":
    st.markdown("# Understand the diode. Predict its current.")
    st.write("Explore measured current–voltage (I–V) data across diode devices, semiconductor materials, and available temperatures. Select a device first so the model choices and graphs stay relevant to that device.")
    a,b,c = st.columns(3)
    a.metric("Classified measurements", f"{len(classified):,}" if not classified.empty else "Not loaded")
    b.metric("Devices in classified data", f"{classified['device'].nunique():,}" if not classified.empty else "—")
    c.metric("Models loaded", len(models))
    st.info("The original 1N4004 dataset is one experiment. The GaN temperature-aware model is a separate experiment. A temperature field does not affect models trained with voltage alone.")
    if not main_data.empty:
        st.subheader("Original 1N4004 forward-bias measurements")
        scale = st.radio("Current-axis scale", ["Linear", "Log (positive current only)", "Symmetric log (shows zero/negative current)"], horizontal=True, key="home_scale")
        fig, ax = plt.subplots(figsize=(10,4.5)); fig.patch.set_facecolor("#081326"); ax.set_facecolor("#081326")
        ax.scatter(main_data.v_meas, main_data.i_meas, s=18, label="Measured forward-bias data")
        style_plot(ax, "1N4004 forward-bias I–V curve", "Current (A)", scale, main_data.i_meas)
        ax.legend(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

elif page == "Predictor":
    st.title("Device-first current predictor")
    st.write("First select the device. Then choose a model available for that device/material. The website does not offer a model for a device when no matching saved model is available.")
    if classified.empty:
        st.error("The classified dataset is missing or has unexpected columns.")
    else:
        device_material = classified[["device", "material"]].drop_duplicates().sort_values(["material", "device"])
        device_material = pd.concat([pd.DataFrame([{"device":"1N4004", "material":"silicon (1N4004)"}]), device_material], ignore_index=True).drop_duplicates()
        labels = [f"{r.device} — {r.material}" for r in device_material.itertuples()]
        selected_label = st.selectbox("1. Select device", labels, key="predict_device")
        selected_row = device_material.iloc[labels.index(selected_label)]
        device, material = str(selected_row.device), str(selected_row.material)
        model_choices = choices_for_device(device, material)
        if not model_choices:
            st.warning("No compatible saved prediction model is currently available for this exact device/material. You can still inspect its measured graph under Graphs & Data.")
        else:
            model_name = st.selectbox("2. Select a model trained for this device/material family", model_choices, key="predict_model")
            rows = get_device_rows(device, material)
            if device == "1N4004":
                voltage_min, voltage_max = float(one_n4004.voltage_V.min()), float(one_n4004.voltage_V.max())
                temps = []
            else:
                voltage_min, voltage_max = float(rows.voltage_V.min()), float(rows.voltage_V.max())
                temps = sorted(rows.temperature_K.unique().tolist())
            if "GaN" in model_name:
                temperature = st.number_input("3. Temperature (K)", min_value=40.0, max_value=400.0, value=300.0, step=10.0)
                voltage_min = max(voltage_min, -1.499998); voltage_max = min(voltage_max, 0.660000)
            elif model_name in ["Silicon devices · SVR", "SiC devices · SVR"]:
                if temps:
                    temperature = st.selectbox("3. Temperature (K)", temps, index=temps.index(293.0) if 293.0 in temps else 0)
                else: temperature = 293.0
            else:
                temperature = 293.0
            voltage = st.number_input("Voltage (V)", min_value=float(voltage_min), max_value=float(voltage_max), value=float(np.clip(0.5, voltage_min, voltage_max)), step=0.005, format="%.6f")
            st.caption(f"Measured voltage range for this selected dataset: {voltage_min:.5g} V to {voltage_max:.5g} V. This control stays within the available range; it does not imply accuracy at every point.")
            if st.button("Predict current", type="primary"):
                try:
                    model = models[model_name]
                    if model_name.startswith("1N4004"):
                        pred = float(np.asarray(model.predict(pd.DataFrame({"v_meas":[voltage]}))).reshape(-1)[0])
                    elif "GaN" in model_name:
                        pred = float(np.asarray(model.predict(pd.DataFrame([{"voltage_V":voltage,"temperature_K":temperature}]))).reshape(-1)[0])
                    else:
                        # These models were trained with device, voltage and temperature features.
                        pred_raw = float(np.asarray(model.predict(pd.DataFrame([{"device":device,"voltage_V":voltage,"temperature_K":temperature}]))).reshape(-1)[0])
                        pred = 10.0 ** pred_raw
                    st.session_state["prediction_result"] = {"device":device,"material":material,"model":model_name,"voltage":voltage,"temperature":temperature,"current":pred}
                except Exception as exc:
                    st.error(f"Prediction could not be calculated with this saved model: {type(exc).__name__}: {exc}")
            result = st.session_state.get("prediction_result")
            if result and result["device"] == device and result["model"] == model_name and result["voltage"] == voltage and result["temperature"] == temperature:
                st.subheader("Prediction result")
                st.metric("Predicted current", f"{result['current']:.6g} A")
                st.caption("This is a model estimate, not a physical measurement. It is valid only for the model's training setup and supported input format.")
                # Device-specific graph; negative voltages are retained if present in the selected data.
                graph_rows = rows.copy()
                if device != "1N4004" and "temperature_K" in graph_rows.columns:
                    graph_rows = graph_rows[graph_rows.temperature_K == temperature]
                if not graph_rows.empty:
                    scale = st.radio("Current-axis scale", ["Linear", "Log (positive current only)", "Symmetric log (shows zero/negative current)"], horizontal=True, key="pred_scale")
                    fig, ax = plt.subplots(figsize=(10,4.8)); fig.patch.set_facecolor("#081326"); ax.set_facecolor("#081326")
                    ax.scatter(graph_rows.voltage_V, graph_rows.current_A, s=20, alpha=.8, label="Measured data")
                    ax.axvline(voltage, linestyle="--", label=f"Input voltage: {voltage:g} V")
                    style_plot(ax, f"{device} measured I–V data at {temperature:g} K" if device != "1N4004" else "1N4004 measured I–V data", "Current (A)", scale, graph_rows.current_A)
                    ax.legend(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

elif page == "Graphs & Data":
    st.title("Measured I–V graphs")
    st.write("Choose the device first. The graphs use the actual measured data available in this repository, including negative voltage when the selected dataset contains it.")
    options = []
    options.append(("1N4004", "silicon (1N4004)"))
    if not classified.empty:
        options += list(classified[["device","material"]].drop_duplicates().sort_values(["material","device"]).itertuples(index=False, name=None))
    options = list(dict.fromkeys((str(a),str(b)) for a,b in options))
    labels = [f"{a} — {b}" for a,b in options]
    choice = st.selectbox("1. Select device", labels, key="graph_device")
    device, material = options[labels.index(choice)]
    rows = get_device_rows(device, material)
    if device != "1N4004" and not rows.empty:
        available_temps = sorted(rows.temperature_K.unique().tolist())
        temp = st.selectbox("2. Select measured temperature (K)", available_temps, index=available_temps.index(293.0) if 293.0 in available_temps else 0, key="graph_temp")
        plot_rows = rows[rows.temperature_K == temp].copy()
        st.caption(f"{len(plot_rows)} measurements at {temp:g} K; voltage range {plot_rows.voltage_V.min():.5g} to {plot_rows.voltage_V.max():.5g} V.")
    else:
        plot_rows = rows.copy()
        st.caption("1N4004 graph combines available forward- and reverse-bias measurements, so negative voltages are included where present.")
    scale = st.radio("3. Current-axis scale", ["Linear", "Log (positive current only)", "Symmetric log (shows zero/negative current)"], horizontal=True, key="graph_scale")
    if plot_rows.empty:
        st.warning("No measurements were found for this device selection.")
    else:
        plot_rows = plot_rows.sort_values("voltage_V")
        fig, ax = plt.subplots(figsize=(11,5)); fig.patch.set_facecolor("#081326"); ax.set_facecolor("#081326")
        ax.scatter(plot_rows.voltage_V, plot_rows.current_A, s=20, alpha=.82, label="Measured data")
        style_plot(ax, f"{device} ({material}) I–V characteristics", "Current (A)", scale, plot_rows.current_A)
        ax.legend(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.dataframe(plot_rows, use_container_width=True, hide_index=True)
        st.download_button("Download displayed measurements", plot_rows.to_csv(index=False).encode("utf-8"), file_name="selected_iv_measurements.csv", mime="text/csv")
        if plot_rows.voltage_V.min() < 0:
            st.success("Negative-voltage measurements are present in this graph.")
        else:
            st.info("This selected device/temperature dataset contains no negative-voltage measurements. Select another device or temperature to inspect reverse-bias data where available.")

elif page == "Model Comparison":
    st.title("Model comparison")
    st.write("These are previously recorded test metrics for the original 1N4004 experiment. They are not recalculated by this website.")
    metrics = pd.DataFrame([
        {"Model":"Random Forest","MAE (A)":0.0001714702,"RMSE (A)":0.0004224828,"R²":0.9989055},
        {"Model":"Tuned SVR","MAE (A)":0.0002631377,"RMSE (A)":0.0003458506,"R²":0.9992665},
        {"Model":"Linear Regression","MAE (A)":0.006944,"RMSE (A)":0.009559,"R²":0.439677},
        {"Model":"Mean baseline","MAE (A)":0.008777,"RMSE (A)":0.012815,"R²":-0.007052},
    ])
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    a,b = st.columns(2)
    with a: st.subheader("MAE (lower is better)"); st.bar_chart(metrics.set_index("Model")[["MAE (A)"]])
    with b: st.subheader("RMSE (lower is better)"); st.bar_chart(metrics.set_index("Model")[["RMSE (A)"]])
    st.warning("These scores came from a random holdout within one measured voltage sweep. They do not establish accuracy for other devices, temperatures, or voltage ranges. A separate voltage-ordered experiment showed poor extrapolation.")
    st.subheader("What the current scale means")
    st.write("Linear scale shows equal current increments. Log scale shows orders of magnitude but can display only positive current. Symmetric-log scale shows negative, zero, and positive current, with a linear region near zero.")

st.markdown("---")
st.caption("DiodeLab AI · Educational machine-learning demonstration. Measured data and model estimates are shown separately; predictions are not physical measurements.")

