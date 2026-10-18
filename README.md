# DiodeLab AI — Diode I–V Predictor

## Project Overview

DiodeLab AI is an interactive application for exploring diode current–voltage (I–V) characteristics and predicting current using machine-learning models.

The project combines measured diode data, saved machine-learning models, and semiconductor physics calculations.

## Features

- Interactive diode current prediction.
- Comparison of recorded model evaluation metrics.
- Visualization of diode I–V curves.
- Saved Random Forest and Support Vector Regression (SVR) models.
- Additional Silicon and Silicon Carbide SVR models.
- Semiconductor diode physics calculations.

## Project Structure

- `app.py` — Streamlit application.
- `diode_iv_data.csv` — Original measured diode dataset.
- `random_forest_model.joblib` — Saved Random Forest model.
- `tuned_svr_model.joblib` — Saved tuned SVR model.
- `trained_models/` — Additional Silicon and Silicon Carbide SVR models.
- `diode_physics.py` — Diode physics calculations.
- `test_diode_physics.py` — Tests for diode physics calculations.
- `prepare_external_data.py` — Script for preparing external diode measurements.
- `data/` — Project datasets and processed measurements.
- `requirements.txt` — Python dependencies.

## Running the Application

1. Install Python and Git.
2. Clone this repository.
3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the application:

   ```bash
   streamlit run app.py
   ```

## Evaluation and Limitations

The application displays previously recorded evaluation metrics for selected models. These values are stored in the application and are not recalculated every time it runs.

The original evaluation uses a random holdout from a measured diode voltage sweep. Measurements from the same sweep may be correlated, so these results do not establish generalization to different diode devices, temperatures, or measurement setups.

The saved models have different input requirements. Their predictions should be interpreted within the data ranges and conditions applicable to each model.

## Reproducibility

The repository includes the application code, selected datasets, saved models, dependency information, and a data-preparation script.

The exact training procedures, random seeds, and complete model-training workflow still need to be documented where applicable. Evaluation metrics shown in the application should not be interpreted as independently reproduced results until the training and evaluation procedures are recorded.

## Future Improvements

- Add scripts to reproduce model training and evaluation.
- Record dataset sources, preprocessing steps, and train/test split methods.
- Document model-specific input ranges and limitations.
- Add automated tests for prediction and evaluation workflows.
