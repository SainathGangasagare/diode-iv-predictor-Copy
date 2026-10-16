"""Basic tests for diode_physics.py.

Run from the repository root with:
    python test_diode_physics.py
"""
import numpy as np

from diode_physics import diode_current, diode_current_array, thermal_voltage


def main():
    reverse_current = diode_current(-0.2)
    zero_current = diode_current(0.0)
    forward_current = diode_current(0.5)

    print("Shockley diode model: basic checks")
    print("----------------------------------")
    print(f"Thermal voltage at 25 °C: {thermal_voltage(25):.6f} V")
    print(f"Current at -0.2 V (reverse bias): {reverse_current:.6e} A")
    print(f"Current at  0.0 V (zero bias):   {zero_current:.6e} A")
    print(f"Current at +0.5 V (forward bias): {forward_current:.6e} A")

    assert reverse_current < 0, "Reverse-bias current should be negative."
    assert zero_current == 0.0, "Current at zero voltage should be zero."
    assert forward_current > 0, "Forward-bias current should be positive."

    # With the same illustrative Is and ideality factor, increasing temperature
    # reduces the ideal-model current at this fixed forward voltage.
    current_25c = diode_current(0.5, temperature_c=25)
    current_75c = diode_current(0.5, temperature_c=75)
    print(f"Current at 0.5 V, 25 °C: {current_25c:.6e} A")
    print(f"Current at 0.5 V, 75 °C: {current_75c:.6e} A")

    voltages = np.array([-0.2, 0.0, 0.5])
    currents = diode_current_array(voltages)
    assert currents.shape == voltages.shape, "Array output shape should match input."
    assert np.all(np.isfinite(currents)), "Currents should be finite."

    # Invalid inputs should be rejected.
    for bad_temp in (-273.15, -300):
        try:
            thermal_voltage(bad_temp)
        except ValueError:
            pass
        else:
            raise AssertionError("Temperature at/below absolute zero should fail.")

    print("\nPASS: basic sign, zero-bias, array, and temperature-input checks passed.")
    print("Reminder: these are equation checks, not validation against measured diode data.")


if __name__ == "__main__":
    main()
