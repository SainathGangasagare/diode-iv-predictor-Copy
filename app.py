import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diode I-V Predictor", page_icon="⚡")

st.title("Diode I-V Characteristics Predictor")
st.write("Predict the forward current of a 1N4004 diode from its measured voltage.")

# Load saved files
rf_model = joblib.load("random_forest_model.joblib")
svr_model = joblib.load("tuned_svr_model.joblib")
data = pd.read_csv("diode_iv_data.csv")

# Choose model
choice = st.selectbox("Choose a model", ["Random Forest", "Tuned SVR"])
model = rf_model if choice == "Random Forest" else svr_model

# Voltage input
voltage = st.number_input(
    "Measured voltage (V)",
    min_value=-0.01,
    max_value=0.90,
    value=0.70,
    step=0.01,
    format="%.5f"
)

if st.button("Predict current"):
    prediction = model.predict(pd.DataFrame({"v_meas": [voltage]}))[0]
    st.metric("Predicted current (A)", f"{prediction:.6f}")

    if voltage < data["v_meas"].min() or voltage > data["v_meas"].max():
        st.warning("Voltage is outside the measured data range; prediction may be unreliable.")

# Plot measured data
st.subheader("Measured diode I-V curve")
fig, ax = plt.subplots()
ax.scatter(data["v_meas"], data["i_meas"], s=18, label="Measured data")
ax.set_xlabel("Measured voltage (V)")
ax.set_ylabel("Measured current (A)")
ax.set_title("1N4004 forward-bias I-V characteristics")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Test-set results from the project evaluation
st.subheader("Model performance on the test set")
results = pd.DataFrame({
    "Model": ["Random Forest", "Tuned SVR", "Linear Regression", "Mean baseline"],
    "MAE (A)": [0.00017147, 0.00026314, 0.006944, 0.008777],
    "RMSE (A)": [0.00042248, 0.00034585, 0.009559, 0.012815],
    "R²": [0.998905, 0.999267, 0.439677, -0.007052]
})
st.dataframe(results, hide_index=True, use_container_width=True)

st.caption("Results are based on one measured voltage sweep and may not generalize to other diodes or measurement conditions.")
