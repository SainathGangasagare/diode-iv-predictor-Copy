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



## GaN Schottky diode measurements

- **File:** `gan_schottky_measurements.csv`
- **Device:** Au/Ni/n-GaN Schottky diode
- **Material:** Gallium nitride (GaN)
- **Measurement rows:** 5,189
- **Temperature range:** 40–400 K, across 19 temperatures
- **Voltage column:** `voltage_V` (V)
- **Current column:** `current_A` (A)
- **Temperature column:** `temperature_K` (K)
- **Bias information:** Forward and reverse regions are identified in `Bias_Region`.
- **Evidence type:** Experimental measurements

**Source and provenance:** The dataset README attributes the experimental I–V series to Doğan and Elagöz, *Physica E* 63, 186–192 (2014), DOI: [10.1016/j.physe.2014.04.019](https://doi.org/10.1016/j.physe.2014.04.019). The reproducibility package also cites Torun and Doğan (2021), DOI: [10.1016/j.spmi.2021.107062](https://doi.org/10.1016/j.spmi.2021.107062).

**Processing notes:** The CSV preserves the original curve identifiers, source-file labels, bias-region labels, and temperature values, alongside standardized columns. The measurements represent one specific device across multiple temperatures, not 5,189 separate devices. Consult the source repository's license and citation instructions before redistributing the data.

This dataset has not yet been used to train or validate the project's existing prediction models.
