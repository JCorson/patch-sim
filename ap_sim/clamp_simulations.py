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


def _setup_simulation(
    num_time_steps: int, sampling_frequency: float
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_step = 1.0 / sampling_frequency * 1000.0
    actual_simulation_time = (num_time_steps - 1) * time_step
    time_array = np.linspace(0, actual_simulation_time, num_time_steps)
    n_arr = np.empty(num_time_steps)
    m_arr = np.empty(num_time_steps)
    h_arr = np.empty(num_time_steps)
    return time_step, time_array, n_arr, m_arr, h_arr


def _initialize_gating_variables(
    neuron: "HodgkinHuxley", initial_voltage: float
) -> tuple[float, float, float]:
    V0 = initial_voltage
    an0, bn0 = neuron.alpha_n(V0), neuron.beta_n(V0)
    am0, bm0 = neuron.alpha_m(V0), neuron.beta_m(V0)
    ah0, bh0 = neuron.alpha_h(V0), neuron.beta_h(V0)
    return an0 / (an0 + bn0), am0 / (am0 + bm0), ah0 / (ah0 + bh0)


def _update_gating_variables(
    neuron: "HodgkinHuxley", V: float, n: float, m: float, h: float, dt: float
) -> tuple[float, float, float]:
    dn = neuron.alpha_n(V) * (1 - n) - neuron.beta_n(V) * n
    dm = neuron.alpha_m(V) * (1 - m) - neuron.beta_m(V) * m
    dh = neuron.alpha_h(V) * (1 - h) - neuron.beta_h(V) * h
    return (
        float(np.clip(n + dn * dt, 0, 1)),
        float(np.clip(m + dm * dt, 0, 1)),
        float(np.clip(h + dh * dt, 0, 1)),
    )


def simulate_voltage_clamp(
    neuron: "HodgkinHuxley",
    voltage_protocol: np.ndarray,
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
        voltage_protocol (np.ndarray): Voltage values in mV to clamp the
            membrane at for each time step. The length of the array determines
            the simulation duration.
        sampling_frequency (float): Sampling frequency in Hz for the simulation.
            Default is 100 kHz (0.01 ms time steps). Higher frequencies give
            finer temporal resolution but increase computation time.

    Returns:
        pd.DataFrame: DataFrame indexed by time in milliseconds (named 'time'),
            with columns: voltage, total_current, sodium_current,
            potassium_current, leak_current, potassium_activation,
            sodium_activation, sodium_inactivation.
    """
    num_time_steps = len(voltage_protocol)

    if num_time_steps == 0:
        raise ValueError("voltage_protocol must not be empty.")
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive.")
    if not np.all(np.isfinite(voltage_protocol)):
        raise ValueError("voltage_protocol must not contain NaN or Inf values.")

    time_step, time_array, n_arr, m_arr, h_arr = _setup_simulation(
        num_time_steps, sampling_frequency
    )

    # Pre-allocate voltage-clamp-specific output arrays
    I_Na = np.empty(num_time_steps)
    I_K = np.empty(num_time_steps)
    I_L = np.empty(num_time_steps)
    I_total = np.empty(num_time_steps)

    # Initialise gating variables at steady state for the first voltage
    n_arr[0], m_arr[0], h_arr[0] = _initialize_gating_variables(
        neuron, voltage_protocol[0]
    )

    # Compute initial currents
    V0 = voltage_protocol[0]
    g_Na0 = neuron.g_Na * (m_arr[0] ** 3) * h_arr[0]
    g_K0 = neuron.g_K * (n_arr[0] ** 4)
    I_Na[0] = g_Na0 * (V0 - neuron.E_Na)
    I_K[0] = g_K0 * (V0 - neuron.E_K)
    I_L[0] = neuron.g_L * (V0 - neuron.E_L)
    I_total[0] = I_Na[0] + I_K[0] + I_L[0]

    # Main simulation loop — all state in plain numpy scalars
    for i in range(1, num_time_steps):
        V = voltage_protocol[i]
        n_prev, m_prev, h_prev = n_arr[i - 1], m_arr[i - 1], h_arr[i - 1]

        n, m, h = _update_gating_variables(neuron, V, n_prev, m_prev, h_prev, time_step)
        n_arr[i], m_arr[i], h_arr[i] = n, m, h

        g_Na = neuron.g_Na * (m**3) * h
        g_K = neuron.g_K * (n**4)
        I_Na[i] = g_Na * (V - neuron.E_Na)
        I_K[i] = g_K * (V - neuron.E_K)
        I_L[i] = neuron.g_L * (V - neuron.E_L)
        I_total[i] = I_Na[i] + I_K[i] + I_L[i]

    results = pd.DataFrame(
        {
            "voltage": voltage_protocol,
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
    current_external: np.ndarray,
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
        current_external (np.ndarray): External current in uA/cm^2 for a
            time-varying current waveform. The length of the array determines
            the simulation duration.
        sampling_frequency (float): Sampling frequency in Hz for the simulation.
            Default is 100 kHz (0.01 ms time steps). Higher frequencies give
            finer temporal resolution but increase computation time.

    Returns:
        pd.DataFrame: DataFrame indexed by time in milliseconds (named 'time'),
            with columns: voltage, potassium_activation, sodium_activation,
            sodium_inactivation.
    """
    num_time_steps = len(current_external)

    if num_time_steps == 0:
        raise ValueError("current_external must not be empty.")
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive.")
    if not np.all(np.isfinite(current_external)):
        raise ValueError("current_external must not contain NaN or Inf values.")

    time_step, time_array, n_arr, m_arr, h_arr = _setup_simulation(
        num_time_steps, sampling_frequency
    )

    V_arr = np.empty(num_time_steps)

    # Define physiological limits for membrane voltage
    min_voltage = -100.0  # mV
    max_voltage = 60.0  # mV

    # Initialise gating variables at steady state for resting potential
    V_arr[0] = neuron.v_rest
    n_arr[0], m_arr[0], h_arr[0] = _initialize_gating_variables(neuron, neuron.v_rest)

    # Main simulation loop — all state in plain numpy scalars
    for i in range(1, num_time_steps):
        V = V_arr[i - 1]
        n, m, h = n_arr[i - 1], m_arr[i - 1], h_arr[i - 1]

        g_Na = neuron.g_Na * (m**3) * h
        g_K = neuron.g_K * (n**4)

        I_Na = g_Na * (V - neuron.E_Na)
        I_K = g_K * (V - neuron.E_K)
        I_L = neuron.g_L * (V - neuron.E_L)

        dV = (current_external[i] - I_Na - I_K - I_L) / neuron.C_m

        V_arr[i] = float(np.clip(V + dV * time_step, min_voltage, max_voltage))
        n_arr[i], m_arr[i], h_arr[i] = _update_gating_variables(
            neuron, V, n, m, h, time_step
        )

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
