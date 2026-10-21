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
