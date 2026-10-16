"""
Physics-based diode I-V simulator.

Independent of the Streamlit app and existing ML models.
Implements the ideal Shockley diode equation.

Limitations:
- Does not model reverse breakdown, series resistance, self-heating,
  or high-level injection.
- Parameters must be appropriate for the device being modeled.
"""
import math
from typing import Union

import numpy as np

Number = Union[int, float]

BOLTZMANN_J_PER_K = 1.380649e-23
ELEMENTARY_CHARGE_C = 1.602176634e-19


def thermal_voltage(temperature_c: Number) -> float:
    """Return kT/q in volts for a temperature in degrees Celsius."""
    temperature_k = float(temperature_c) + 273.15
    if not math.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("Temperature must be above absolute zero (-273.15 °C).")
    return BOLTZMANN_J_PER_K * temperature_k / ELEMENTARY_CHARGE_C


def diode_current(
    voltage_v: Number,
    saturation_current_a: Number = 1e-12,
    ideality_factor: Number = 1.0,
    temperature_c: Number = 25.0,
) -> float:
    """Calculate diode current in amperes using the Shockley equation.

    Positive voltage represents forward bias; negative voltage represents
    reverse bias. This ideal model does not simulate reverse breakdown.
    """
    voltage = float(voltage_v)
    isat = float(saturation_current_a)
    n = float(ideality_factor)

    if not math.isfinite(voltage):
        raise ValueError("Voltage must be finite.")
    if not math.isfinite(isat) or isat <= 0:
        raise ValueError("Saturation current must be finite and greater than zero.")
    if not math.isfinite(n) or n <= 0:
        raise ValueError("Ideality factor must be finite and greater than zero.")

    vt = thermal_voltage(temperature_c)
    exponent = min(voltage / (n * vt), 700.0)
    return isat * math.expm1(exponent)


def diode_current_array(
    voltages_v,
    saturation_current_a: Number = 1e-12,
    ideality_factor: Number = 1.0,
    temperature_c: Number = 25.0,
) -> np.ndarray:
    """Calculate currents for many voltages; returns a NumPy array in amperes."""
    voltages = np.asarray(voltages_v, dtype=float)
    if not np.all(np.isfinite(voltages)):
        raise ValueError("All voltages must be finite.")

    isat = float(saturation_current_a)
    n = float(ideality_factor)
    if not math.isfinite(isat) or isat <= 0:
        raise ValueError("Saturation current must be finite and greater than zero.")
    if not math.isfinite(n) or n <= 0:
        raise ValueError("Ideality factor must be finite and greater than zero.")

    vt = thermal_voltage(temperature_c)
    exponents = np.clip(voltages / (n * vt), -745.0, 700.0)
    return isat * np.expm1(exponents)
