"""Clamp simulation functions for the Hodgkin-Huxley model.

This module contains functions for voltage clamp and current clamp experiments
that can be performed on Hodgkin-Huxley neuron objects.
"""

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .hodgkin_huxley import HodgkinHuxley

#: Fixed simulation sampling frequency (Hz). dt = 1000 / SIM_SAMPLING_FREQ ms.
#: 40 kHz (dt = 0.025 ms) is standard for Hodgkin-Huxley models.
SIM_SAMPLING_FREQ: float = 40_000.0


def _setup_simulation(
    num_time_steps: int, sampling_frequency: float
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Allocate time array and gating variable output arrays for a simulation run.

    Args:
        num_time_steps: Total number of time steps in the simulation.
        sampling_frequency: Sampling frequency in Hz.

    Returns:
        Tuple of (time_step, time_array, n_arr, m_arr, h_arr) where time_step
        is in milliseconds and arrays are pre-allocated numpy arrays.
    """
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
    """Compute steady-state gating variables at a given initial voltage.

    Args:
        neuron: The Hodgkin-Huxley neuron model.
        initial_voltage: Initial membrane voltage in mV.

    Returns:
        Tuple of (n0, m0, h0) steady-state gating variable values.
    """
    V0 = initial_voltage
    an0, bn0 = neuron.alpha_n(V0), neuron.beta_n(V0)
    am0, bm0 = neuron.alpha_m(V0), neuron.beta_m(V0)
    ah0, bh0 = neuron.alpha_h(V0), neuron.beta_h(V0)
    return an0 / (an0 + bn0), am0 / (am0 + bm0), ah0 / (ah0 + bh0)


def _hh_derivatives(
    neuron: "HodgkinHuxley",
    V: float,
    n: float,
    m: float,
    h: float,
    I_ext: float,
) -> tuple[float, float, float, float]:
    """Compute HH ODE derivatives for the current-clamp system.

    Args:
        neuron: The Hodgkin-Huxley neuron model.
        V: Membrane voltage in mV.
        n: Potassium activation gating variable.
        m: Sodium activation gating variable.
        h: Sodium inactivation gating variable.
        I_ext: External current in uA/cm^2.

    Returns:
        Tuple of (dV/dt, dn/dt, dm/dt, dh/dt).
    """
    I_Na = neuron.g_Na * (m**3) * h * (V - neuron.E_Na)
    I_K = neuron.g_K * (n**4) * (V - neuron.E_K)
    I_L = neuron.g_L * (V - neuron.E_L)
    dV = (I_ext - I_Na - I_K - I_L) / neuron.C_m
    dn = neuron.alpha_n(V) * (1 - n) - neuron.beta_n(V) * n
    dm = neuron.alpha_m(V) * (1 - m) - neuron.beta_m(V) * m
    dh = neuron.alpha_h(V) * (1 - h) - neuron.beta_h(V) * h
    return dV, dn, dm, dh


def _gating_derivatives(
    neuron: "HodgkinHuxley",
    V: float,
    n: float,
    m: float,
    h: float,
) -> tuple[float, float, float]:
    """Compute gating variable derivatives for a prescribed voltage.

    Args:
        neuron: The Hodgkin-Huxley neuron model.
        V: Prescribed membrane voltage in mV.
        n: Potassium activation gating variable.
        m: Sodium activation gating variable.
        h: Sodium inactivation gating variable.

    Returns:
        Tuple of (dn/dt, dm/dt, dh/dt).
    """
    dn = neuron.alpha_n(V) * (1 - n) - neuron.beta_n(V) * n
    dm = neuron.alpha_m(V) * (1 - m) - neuron.beta_m(V) * m
    dh = neuron.alpha_h(V) * (1 - h) - neuron.beta_h(V) * h
    return dn, dm, dh


def _rk4_step_current_clamp(
    neuron: "HodgkinHuxley",
    V: float,
    n: float,
    m: float,
    h: float,
    I_ext: float,
    dt: float,
) -> tuple[float, float, float, float]:
    """Advance the current-clamp state by one RK4 step.

    Args:
        neuron: The Hodgkin-Huxley neuron model.
        V: Membrane voltage in mV.
        n: Potassium activation gating variable.
        m: Sodium activation gating variable.
        h: Sodium inactivation gating variable.
        I_ext: External current in uA/cm^2, held constant over the step.
        dt: Time step in milliseconds.

    Returns:
        Tuple of updated (V, n, m, h) with gating variables clipped to [0, 1].
    """
    dV1, dn1, dm1, dh1 = _hh_derivatives(neuron, V, n, m, h, I_ext)
    dV2, dn2, dm2, dh2 = _hh_derivatives(
        neuron,
        V + 0.5 * dt * dV1,
        n + 0.5 * dt * dn1,
        m + 0.5 * dt * dm1,
        h + 0.5 * dt * dh1,
        I_ext,
    )
    dV3, dn3, dm3, dh3 = _hh_derivatives(
        neuron,
        V + 0.5 * dt * dV2,
        n + 0.5 * dt * dn2,
        m + 0.5 * dt * dm2,
        h + 0.5 * dt * dh2,
        I_ext,
    )
    dV4, dn4, dm4, dh4 = _hh_derivatives(
        neuron,
        V + dt * dV3,
        n + dt * dn3,
        m + dt * dm3,
        h + dt * dh3,
        I_ext,
    )
    V_new = V + (dt / 6.0) * (dV1 + 2 * dV2 + 2 * dV3 + dV4)
    n_new = float(np.clip(n + (dt / 6.0) * (dn1 + 2 * dn2 + 2 * dn3 + dn4), 0, 1))
    m_new = float(np.clip(m + (dt / 6.0) * (dm1 + 2 * dm2 + 2 * dm3 + dm4), 0, 1))
    h_new = float(np.clip(h + (dt / 6.0) * (dh1 + 2 * dh2 + 2 * dh3 + dh4), 0, 1))
    return V_new, n_new, m_new, h_new


def _rk4_step_voltage_clamp(
    neuron: "HodgkinHuxley",
    V: float,
    n: float,
    m: float,
    h: float,
    dt: float,
) -> tuple[float, float, float]:
    """Advance voltage-clamp gating variables by one RK4 step.

    Args:
        neuron: The Hodgkin-Huxley neuron model.
        V: Prescribed membrane voltage in mV.
        n: Potassium activation gating variable.
        m: Sodium activation gating variable.
        h: Sodium inactivation gating variable.
        dt: Time step in milliseconds.

    Returns:
        Tuple of updated (n, m, h) gating variables clipped to [0, 1].
    """
    dn1, dm1, dh1 = _gating_derivatives(neuron, V, n, m, h)
    dn2, dm2, dh2 = _gating_derivatives(
        neuron,
        V,
        n + 0.5 * dt * dn1,
        m + 0.5 * dt * dm1,
        h + 0.5 * dt * dh1,
    )
    dn3, dm3, dh3 = _gating_derivatives(
        neuron,
        V,
        n + 0.5 * dt * dn2,
        m + 0.5 * dt * dm2,
        h + 0.5 * dt * dh2,
    )
    dn4, dm4, dh4 = _gating_derivatives(
        neuron,
        V,
        n + dt * dn3,
        m + dt * dm3,
        h + dt * dh3,
    )
    n_new = float(np.clip(n + (dt / 6.0) * (dn1 + 2 * dn2 + 2 * dn3 + dn4), 0, 1))
    m_new = float(np.clip(m + (dt / 6.0) * (dm1 + 2 * dm2 + 2 * dm3 + dm4), 0, 1))
    h_new = float(np.clip(h + (dt / 6.0) * (dh1 + 2 * dh2 + 2 * dh3 + dh4), 0, 1))
    return n_new, m_new, h_new


def simulate_voltage_clamp(
    neuron: "HodgkinHuxley",
    voltage_protocol: np.ndarray,
) -> pd.DataFrame:
    """Simulate a voltage clamp experiment using the Hodgkin-Huxley model.

    In a voltage clamp experiment, the membrane potential is held at specified
    values and the current required to maintain those voltages is measured. This
    function simulates this process by computing the ionic currents that would flow
    at each voltage step in the protocol.

    The simulation always uses :data:`SIM_SAMPLING_FREQ` (40 kHz, dt = 0.025 ms)
    as the integration time step.

    Args:
        neuron: The Hodgkin-Huxley neuron object to simulate.
        voltage_protocol: Voltage values in mV to clamp the membrane at for each
            time step. The length of the array determines the simulation duration.

    Returns:
        DataFrame indexed by time in milliseconds (named 'time'), with columns:
        voltage, total_current, sodium_current, potassium_current, leak_current,
        potassium_activation, sodium_activation, sodium_inactivation.
    """
    num_time_steps = len(voltage_protocol)

    if num_time_steps == 0:
        raise ValueError("voltage_protocol must not be empty.")
    if not np.all(np.isfinite(voltage_protocol)):
        raise ValueError("voltage_protocol must not contain NaN or Inf values.")

    time_step, time_array, n_arr, m_arr, h_arr = _setup_simulation(
        num_time_steps, SIM_SAMPLING_FREQ
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

        n, m, h = _rk4_step_voltage_clamp(neuron, V, n_prev, m_prev, h_prev, time_step)
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
) -> pd.DataFrame:
    """Simulate a current clamp experiment using the Hodgkin-Huxley model.

    In a current clamp experiment, current is injected into the cell membrane and
    the resulting voltage changes are recorded. This function simulates this process
    by computing the membrane voltage over time in response to the specified
    external current.

    The simulation always uses :data:`SIM_SAMPLING_FREQ` (40 kHz, dt = 0.025 ms)
    as the integration time step.

    Args:
        neuron: The Hodgkin-Huxley neuron object to simulate.
        current_external: External current in uA/cm^2 for a time-varying current
            waveform. The length of the array determines the simulation duration.

    Returns:
        DataFrame indexed by time in milliseconds (named 'time'), with columns:
        voltage, potassium_activation, sodium_activation, sodium_inactivation.
    """
    num_time_steps = len(current_external)

    if num_time_steps == 0:
        raise ValueError("current_external must not be empty.")
    if not np.all(np.isfinite(current_external)):
        raise ValueError("current_external must not contain NaN or Inf values.")

    time_step, time_array, n_arr, m_arr, h_arr = _setup_simulation(
        num_time_steps, SIM_SAMPLING_FREQ
    )

    V_arr = np.empty(num_time_steps)

    # Initialise gating variables at steady state for resting potential
    V_arr[0] = neuron.v_rest
    n_arr[0], m_arr[0], h_arr[0] = _initialize_gating_variables(neuron, neuron.v_rest)

    # Main simulation loop — all state in plain numpy scalars
    for i in range(1, num_time_steps):
        V = V_arr[i - 1]
        n, m, h = n_arr[i - 1], m_arr[i - 1], h_arr[i - 1]

        V_new, n_new, m_new, h_new = _rk4_step_current_clamp(
            neuron, V, n, m, h, current_external[i - 1], time_step
        )

        V_arr[i] = V_new
        n_arr[i], m_arr[i], h_arr[i] = n_new, m_new, h_new

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
