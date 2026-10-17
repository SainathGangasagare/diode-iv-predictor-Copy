## External dataset source

**Dataset:** Diode I–V Measurements  
**Repository:** https://github.com/blueberrye-io/diode-iv-measurements/  
**Data license:** CC BY 4.0  
**Measurement instrument:** Rohde & Schwarz NGU411 source-measure unit  
**Units:** Voltage in volts (V); current in amperes (A)

### Imported files

- `1n4004_forward.csv` — silicon rectifier, forward bias
- `1n4004_reverse.csv` — silicon rectifier, reverse bias
- `1n4004_reverse_10uA_range.csv` — silicon rectifier, reverse bias, fixed current range
- `1n5229b_reverse.csv` — Zener diode, reverse bias
- `sb560_forward.csv` — Schottky diode, forward bias
- `sb560_reverse.csv` — Schottky diode, reverse bias

### Data handling notes

- The first row of each sweep contains an invalid reading (`nan`); skip it during cleaning.
- Treat readings below approximately 50 nA cautiously, as noted by the source.
- Use `v_meas` and `i_meas` for quantitative analysis.
- Preserve the original downloaded CSV files.
- These are external measurements, not physics simulations.
- Attribute the original dataset under its CC BY 4.0 license when redistributing or publishing results.



## Cryogenic Diode Measurements

- **Dataset:** Cryogenic and Superconducting Power Electronics Database
- **Source DOI:** https://doi.org/10.57745/2AHLA6
- **Measurement file:** `cryogenic_diode_measurements.csv`
- **Devices:** 70 diode references
- **Measurements:** 4,201 rows
- **Temperatures:** 293 K and 77 K
- **Units:** Voltage in volts (V), current in amperes (A)
- **Data type:** Experimental measurements
- **License:** Etalab Open License 2.0, according to the dataset's published license information

### Processing notes

The original downloaded ZIP is retained separately from the processed CSV. Each temperature sweep is stored as an independent observation. Missing voltage/current pairs are excluded independently for each temperature. Device technology labels are taken from the source metadata; unspecified values remain unspecified.

This dataset has not yet been used to retrain the project's existing machine-learning models.
