"""
Clamp simulation functions for the Hodgkin-Huxley model.

This module contains functions for voltage clamp and current clamp experiments
that can be performed on Hodgkin-Huxley neuron objects.
"""

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .hodgkin_huxley import HodgkinHuxley


def simulate_voltage_clamp(
    neuron: "HodgkinHuxley",
    voltage_protocol: np.ndarray | list[float],
    sampling_frequency: float = 100000.0,  # Hz (100 kHz default)
) -> pd.DataFrame:
    """
    Simulate a voltage clamp experiment using the Hodgkin-Huxley model.

    In a voltage clamp experiment, the membrane potential is held at specified
    values and the current required to maintain those voltages is measured. This
    function simulates this process by computing the ionic currents that would flow
    at each voltage step in the protocol.

    Parameters:
        neuron (HodgkinHuxley): The Hodgkin-Huxley neuron object to simulate.
        voltage_protocol (np.ndarray | list[float]): Voltage values in mV to clamp
            the membrane at for each time step. Must be an array/list for a
            time-varying voltage protocol. The length of the array determines the
            simulation duration.
        sampling_frequency (float): Sampling frequency in Hz for the simulation.
            Default is 100 kHz (0.01 ms time steps). Higher frequencies give
            finer temporal resolution but increase computation time.

    Returns:
        pd.DataFrame: DataFrame indexed by time in milliseconds (named 'time'),
            with columns: voltage, total_current, sodium_current,
            potassium_current, leak_current, potassium_activation,
            sodium_activation, sodium_inactivation.
    """
    # Convert voltage_protocol to numpy array if it's a list
    voltage_array = np.asarray(voltage_protocol, dtype=float)
    num_time_steps = len(voltage_array)

    if num_time_steps == 0:
        raise ValueError("voltage_protocol must not be empty.")
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive.")
    if not np.all(np.isfinite(voltage_array)):
        raise ValueError("voltage_protocol must not contain NaN or Inf values.")

    # Calculate time step from sampling frequency
    time_step = 1.0 / sampling_frequency * 1000.0  # Convert Hz to milliseconds

    # Calculate the actual simulation time and build time array
    actual_simulation_time = (num_time_steps - 1) * time_step
    time_array = np.linspace(0, actual_simulation_time, num_time_steps)

    # Pre-allocate output arrays
    n_arr = np.empty(num_time_steps)  # potassium activation
    m_arr = np.empty(num_time_steps)  # sodium activation
    h_arr = np.empty(num_time_steps)  # sodium inactivation
    I_Na = np.empty(num_time_steps)
    I_K = np.empty(num_time_steps)
    I_L = np.empty(num_time_steps)
    I_total = np.empty(num_time_steps)

    # Initialise gating variables at steady state for the first voltage
    V0 = voltage_array[0]
    an0, bn0 = neuron.alpha_n(V0), neuron.beta_n(V0)
    am0, bm0 = neuron.alpha_m(V0), neuron.beta_m(V0)
    ah0, bh0 = neuron.alpha_h(V0), neuron.beta_h(V0)
    n_arr[0] = an0 / (an0 + bn0)
    m_arr[0] = am0 / (am0 + bm0)
    h_arr[0] = ah0 / (ah0 + bh0)

    # Compute initial currents
    g_Na0 = neuron.g_Na * (m_arr[0] ** 3) * h_arr[0]
    g_K0 = neuron.g_K * (n_arr[0] ** 4)
    I_Na[0] = g_Na0 * (V0 - neuron.E_Na)
    I_K[0] = g_K0 * (V0 - neuron.E_K)
    I_L[0] = neuron.g_L * (V0 - neuron.E_L)
    I_total[0] = I_Na[0] + I_K[0] + I_L[0]

    # Main simulation loop — all state in plain numpy scalars
    for i in range(1, num_time_steps):
        V = voltage_array[i]
        n_prev, m_prev, h_prev = n_arr[i - 1], m_arr[i - 1], h_arr[i - 1]

        dn = neuron.alpha_n(V) * (1 - n_prev) - neuron.beta_n(V) * n_prev
        dm = neuron.alpha_m(V) * (1 - m_prev) - neuron.beta_m(V) * m_prev
        dh = neuron.alpha_h(V) * (1 - h_prev) - neuron.beta_h(V) * h_prev

        # Clip to physiological bounds; use clipped values for currents too
        n = float(np.clip(n_prev + dn * time_step, 0, 1))
        m = float(np.clip(m_prev + dm * time_step, 0, 1))
        h = float(np.clip(h_prev + dh * time_step, 0, 1))
        n_arr[i], m_arr[i], h_arr[i] = n, m, h

        g_Na = neuron.g_Na * (m**3) * h
        g_K = neuron.g_K * (n**4)
        I_Na[i] = g_Na * (V - neuron.E_Na)
        I_K[i] = g_K * (V - neuron.E_K)
        I_L[i] = neuron.g_L * (V - neuron.E_L)
        I_total[i] = I_Na[i] + I_K[i] + I_L[i]

    results = pd.DataFrame(
        {
            "voltage": voltage_array,
            "total_current": I_total,
            "sodium_current": I_Na,
            "potassium_current": I_K,
            "leak_current": I_L,
            "potassium_activation": n_arr,
            "sodium_activation": m_arr,
            "sodium_inactivation": h_arr,
        },
        index=time_array,
    )
    results.index.name = "time"
    return results


def simulate_current_clamp(
    neuron: "HodgkinHuxley",
    current_external: np.ndarray | list[float],
    sampling_frequency: float = 100000.0,  # Hz (100 kHz default)
) -> pd.DataFrame:
    """
    Simulate a current clamp experiment using the Hodgkin-Huxley model.

    In a current clamp experiment, current is injected into the cell membrane and
    the resulting voltage changes are recorded. This function simulates this process
    by computing the membrane voltage over time in response to the specified
    external current.

    Parameters:
        neuron (HodgkinHuxley): The Hodgkin-Huxley neuron object to simulate.
        current_external (np.ndarray | list[float]): External current in uA/cm^2.
            Must be an array/list for a time-varying current waveform.
            The length of the array determines the simulation duration.
        sampling_frequency (float): Sampling frequency in Hz for the simulation.
            Default is 100 kHz (0.01 ms time steps). Higher frequencies give
            finer temporal resolution but increase computation time.

    Returns:
        pd.DataFrame: DataFrame indexed by time in milliseconds (named 'time'),
            with columns: voltage, potassium_activation, sodium_activation,
            sodium_inactivation.
    """
    # Convert current_external to numpy array if it's a list
    current_array = np.asarray(current_external, dtype=float)
    num_time_steps = len(current_array)

    if num_time_steps == 0:
        raise ValueError("current_external must not be empty.")
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive.")
    if not np.all(np.isfinite(current_array)):
        raise ValueError("current_external must not contain NaN or Inf values.")

    # Calculate time step from sampling frequency
    time_step = 1.0 / sampling_frequency * 1000.0  # Convert Hz to milliseconds

    # Calculate the actual simulation time and build time array
    actual_simulation_time = (num_time_steps - 1) * time_step
    time_array = np.linspace(0, actual_simulation_time, num_time_steps)

    # Pre-allocate output arrays
    V_arr = np.empty(num_time_steps)
    n_arr = np.empty(num_time_steps)  # potassium activation
    m_arr = np.empty(num_time_steps)  # sodium activation
    h_arr = np.empty(num_time_steps)  # sodium inactivation

    # Define physiological limits for membrane voltage
    min_voltage = -100.0  # mV
    max_voltage = 60.0  # mV

    # Initialise gating variables at steady state for resting potential
    V0 = neuron.v_rest
    an0, bn0 = neuron.alpha_n(V0), neuron.beta_n(V0)
    am0, bm0 = neuron.alpha_m(V0), neuron.beta_m(V0)
    ah0, bh0 = neuron.alpha_h(V0), neuron.beta_h(V0)
    V_arr[0] = V0
    n_arr[0] = an0 / (an0 + bn0)
    m_arr[0] = am0 / (am0 + bm0)
    h_arr[0] = ah0 / (ah0 + bh0)

    # Main simulation loop — all state in plain numpy scalars
    for i in range(1, num_time_steps):
        V = V_arr[i - 1]
        n, m, h = n_arr[i - 1], m_arr[i - 1], h_arr[i - 1]

        g_Na = neuron.g_Na * (m**3) * h
        g_K = neuron.g_K * (n**4)

        I_Na = g_Na * (V - neuron.E_Na)
        I_K = g_K * (V - neuron.E_K)
        I_L = neuron.g_L * (V - neuron.E_L)

        dV = (current_array[i] - I_Na - I_K - I_L) / neuron.C_m
        dn = neuron.alpha_n(V) * (1 - n) - neuron.beta_n(V) * n
        dm = neuron.alpha_m(V) * (1 - m) - neuron.beta_m(V) * m
        dh = neuron.alpha_h(V) * (1 - h) - neuron.beta_h(V) * h

        V_arr[i] = float(np.clip(V + dV * time_step, min_voltage, max_voltage))
        n_arr[i] = float(np.clip(n + dn * time_step, 0, 1))
        m_arr[i] = float(np.clip(m + dm * time_step, 0, 1))
        h_arr[i] = float(np.clip(h + dh * time_step, 0, 1))

    results = pd.DataFrame(
        {
            "voltage": V_arr,
            "potassium_activation": n_arr,
            "sodium_activation": m_arr,
            "sodium_inactivation": h_arr,
        },
        index=time_array,
    )
    results.index.name = "time"
    return results
