
from pathlib import Path
import pandas as pd

# Folder containing this script
PROJECT_DIR = Path(__file__).resolve().parent
EXTERNAL_DIR = PROJECT_DIR / "data" / "external"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

# External CSV files and their device labels
FILES = {
    "1n4004_forward.csv": {
        "device": "1N4004",
        "material": "silicon",
        "device_type": "rectifier",
        "bias": "forward",
    },
    "1n4004_reverse.csv": {
        "device": "1N4004",
        "material": "silicon",
        "device_type": "rectifier",
        "bias": "reverse",
    },
    "1n4004_reverse_10uA_range.csv": {
        "device": "1N4004",
        "material": "silicon",
        "device_type": "rectifier",
        "bias": "reverse",
    },
    "1n5229b_reverse.csv": {
        "device": "1N5229B",
        "material": "silicon",
        "device_type": "zener",
        "bias": "reverse_breakdown",
    },
    "sb560_forward.csv": {
        "device": "SB560",
        "material": "silicon",
        "device_type": "schottky",
        "bias": "forward",
    },
    "sb560_reverse.csv": {
        "device": "SB560",
        "material": "silicon",
        "device_type": "schottky",
        "bias": "reverse",
    },
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_tables = []

    for filename, labels in FILES.items():
        filepath = EXTERNAL_DIR / filename

        if not filepath.exists():
            print(f"SKIPPED: {filename} was not found.")
            continue

        df = pd.read_csv(filepath)

        required_columns = {"v_set", "v_meas", "i_meas"}
        if not required_columns.issubset(df.columns):
            print(f"SKIPPED: {filename} has unexpected columns.")
            continue

        # Convert measurement columns to numeric values.
        df["v_meas"] = pd.to_numeric(df["v_meas"], errors="coerce")
        df["i_meas"] = pd.to_numeric(df["i_meas"], errors="coerce")

        # Remove missing readings, including the invalid first row.
        df = df.dropna(subset=["v_meas", "i_meas"]).copy()

        # Use measured voltage/current, not the requested voltage.
        result = pd.DataFrame({
            "voltage_V": df["v_meas"],
            "current_A": df["i_meas"],
        })

        for column, value in labels.items():
            result[column] = value

        result["source_id"] = "blueberrye_io_diode_iv_measurements"
        result["evidence_type"] = "direct_measurement"
        result["temperature_C"] = pd.NA
        result["quality_flag"] = "review_required"

        cleaned_tables.append(result)
        print(f"Read {len(result)} valid rows from {filename}")

    if not cleaned_tables:
        raise RuntimeError(
            "No CSV files were processed. Check the data/external folder."
        )

    combined = pd.concat(cleaned_tables, ignore_index=True)

    output_file = OUTPUT_DIR / "external_iv_measurements.csv"
    combined.to_csv(output_file, index=False)

    print(f"\nSaved {len(combined)} rows to:")
    print(output_file)
    print("\nReview flags and measurement conditions before training.")


if __name__ == "__main__":
    main()
