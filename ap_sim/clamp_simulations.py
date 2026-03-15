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
) -> tuple[float, float, float, dict[str, float]]:
    """Compute steady-state gating variables at a given initial voltage.

    Initialises both the classic HH gating variables (n, m, h) and the
    steady-state values for all optional channel gating variables.

    Args:
        neuron: The Hodgkin-Huxley neuron model.
        initial_voltage: Initial membrane voltage in mV.

    Returns:
        Tuple of (n0, m0, h0, opt_state) where opt_state maps each optional
        gating variable name to its steady-state value at initial_voltage.
    """
    V0 = initial_voltage
    an0, bn0 = neuron.alpha_n(V0), neuron.beta_n(V0)
    am0, bm0 = neuron.alpha_m(V0), neuron.beta_m(V0)
    ah0, bh0 = neuron.alpha_h(V0), neuron.beta_h(V0)
    n0 = an0 / (an0 + bn0)
    m0 = am0 / (am0 + bm0)
    h0 = ah0 / (ah0 + bh0)

    opt_state: dict[str, float] = {}
    for gv in neuron.all_optional_gating_variables():
        a, b = gv.alpha(V0), gv.beta(V0)
        opt_state[gv.name] = a / (a + b)

    return n0, m0, h0, opt_state


def _optional_gating_derivatives(
    neuron: "HodgkinHuxley",
    V: float,
    opt_state: dict[str, float],
) -> dict[str, float]:
    """Compute derivatives for all optional channel gating variables.

    Args:
        neuron: The Hodgkin-Huxley neuron model.
        V: Membrane voltage in mV.
        opt_state: Current gating state mapping name → value.

    Returns:
        Dict mapping each gating variable name to its dx/dt value.
    """
    derivs: dict[str, float] = {}
    for gv in neuron.all_optional_gating_variables():
        x = opt_state[gv.name]
        derivs[gv.name] = gv.alpha(V) * (1 - x) - gv.beta(V) * x
    return derivs


def _advance_opt_state(
    opt_state: dict[str, float],
    derivs: dict[str, float],
    dt: float,
) -> dict[str, float]:
    """Advance optional gating state by a scaled derivative step.

    Args:
        opt_state: Current gating state.
        derivs: Derivatives dict (same keys as opt_state).
        dt: Scaled time step (may be half-step for RK4 midpoints).

    Returns:
        New gating state with each value advanced by dt * deriv[key].
    """
    return {k: opt_state[k] + dt * derivs[k] for k in opt_state}


def _clip_opt_state(opt_state: dict[str, float]) -> dict[str, float]:
    """Clip all values in an optional gating state to [0, 1].

    Args:
        opt_state: Gating state to clip.

    Returns:
        New dict with all values clipped to [0, 1].
    """
    return {k: float(np.clip(v, 0.0, 1.0)) for k, v in opt_state.items()}


def _hh_derivatives(
    neuron: "HodgkinHuxley",
    V: float,
    n: float,
    m: float,
    h: float,
    opt_state: dict[str, float],
    I_ext: float,
    ca_i: float,
) -> tuple[float, float, float, float, dict[str, float], float]:
    """Compute HH ODE derivatives for the current-clamp system.

    Includes contributions from optional channels in the voltage derivative
    and returns their gating derivatives as a separate dict.  Also computes
    the Ca2+ concentration derivative when calcium dynamics are active.

    Args:
        neuron: The Hodgkin-Huxley neuron model.
        V: Membrane voltage in mV.
        n: Potassium activation gating variable.
        m: Sodium activation gating variable.
        h: Sodium inactivation gating variable.
        opt_state: Optional channel gating state (name → value).
        I_ext: External current in uA/cm^2.
        ca_i: Current intracellular Ca2+ concentration in mM.

    Returns:
        Tuple of (dV/dt, dn/dt, dm/dt, dh/dt, opt_derivs, dca_i/dt) where
        opt_derivs maps each optional gating variable name to its derivative
        and dca_i/dt is 0.0 when no calcium_dynamics are configured.
    """
    I_Na = neuron.g_Na * (m**3) * h * (V - neuron.E_Na)
    I_K = neuron.g_K * (n**4) * (V - neuron.E_K)
    I_L = neuron.g_L * (V - neuron.E_L)
    I_opt = sum(ch.compute_current(V, opt_state) for ch in neuron.optional_channels)
    dV = (I_ext - I_Na - I_K - I_L - I_opt) / neuron.C_m
    dn = neuron.alpha_n(V) * (1 - n) - neuron.beta_n(V) * n
    dm = neuron.alpha_m(V) * (1 - m) - neuron.beta_m(V) * m
    dh = neuron.alpha_h(V) * (1 - h) - neuron.beta_h(V) * h
    opt_derivs = _optional_gating_derivatives(neuron, V, opt_state)
    if neuron.calcium_dynamics is not None:
        I_Ca = neuron.calcium_current(V, opt_state)
        dca_i = neuron.calcium_dynamics.derivative(I_Ca, ca_i)
    else:
        dca_i = 0.0
    return dV, dn, dm, dh, opt_derivs, dca_i


def _gating_derivatives(
    neuron: "HodgkinHuxley",
    V: float,
    n: float,
    m: float,
    h: float,
    opt_state: dict[str, float],
    ca_i: float,
) -> tuple[float, float, float, dict[str, float], float]:
    """Compute gating variable derivatives for a prescribed voltage.

    Also computes the Ca2+ concentration derivative when calcium dynamics
    are active.

    Args:
        neuron: The Hodgkin-Huxley neuron model.
        V: Prescribed membrane voltage in mV.
        n: Potassium activation gating variable.
        m: Sodium activation gating variable.
        h: Sodium inactivation gating variable.
        opt_state: Optional channel gating state (name → value).
        ca_i: Current intracellular Ca2+ concentration in mM.

    Returns:
        Tuple of (dn/dt, dm/dt, dh/dt, opt_derivs, dca_i/dt) where
        opt_derivs maps each optional gating variable name to its derivative
        and dca_i/dt is 0.0 when no calcium_dynamics are configured.
    """
    dn = neuron.alpha_n(V) * (1 - n) - neuron.beta_n(V) * n
    dm = neuron.alpha_m(V) * (1 - m) - neuron.beta_m(V) * m
    dh = neuron.alpha_h(V) * (1 - h) - neuron.beta_h(V) * h
    opt_derivs = _optional_gating_derivatives(neuron, V, opt_state)
    if neuron.calcium_dynamics is not None:
        I_Ca = neuron.calcium_current(V, opt_state)
        dca_i = neuron.calcium_dynamics.derivative(I_Ca, ca_i)
    else:
        dca_i = 0.0
    return dn, dm, dh, opt_derivs, dca_i


def _rk4_step_current_clamp(
    neuron: "HodgkinHuxley",
    V: float,
    n: float,
    m: float,
    h: float,
    opt_state: dict[str, float],
    I_ext: float,
    dt: float,
    ca_i: float,
) -> tuple[float, float, float, float, dict[str, float], float]:
    """Advance the current-clamp state by one RK4 step.

    Args:
        neuron: The Hodgkin-Huxley neuron model.
        V: Membrane voltage in mV.
        n: Potassium activation gating variable.
        m: Sodium activation gating variable.
        h: Sodium inactivation gating variable.
        opt_state: Optional channel gating state (name → value).
        I_ext: External current in uA/cm^2, held constant over the step.
        dt: Time step in milliseconds.
        ca_i: Current intracellular Ca2+ concentration in mM.

    Returns:
        Tuple of updated (V, n, m, h, opt_state, ca_i) with gating variables
        clipped to [0, 1] and ca_i floored at 0.0.
    """
    dV1, dn1, dm1, dh1, dopt1, dca1 = _hh_derivatives(
        neuron, V, n, m, h, opt_state, I_ext, ca_i
    )
    opt2 = _advance_opt_state(opt_state, dopt1, 0.5 * dt)
    dV2, dn2, dm2, dh2, dopt2, dca2 = _hh_derivatives(
        neuron,
        V + 0.5 * dt * dV1,
        n + 0.5 * dt * dn1,
        m + 0.5 * dt * dm1,
        h + 0.5 * dt * dh1,
        opt2,
        I_ext,
        ca_i + 0.5 * dt * dca1,
    )
    opt3 = _advance_opt_state(opt_state, dopt2, 0.5 * dt)
    dV3, dn3, dm3, dh3, dopt3, dca3 = _hh_derivatives(
        neuron,
        V + 0.5 * dt * dV2,
        n + 0.5 * dt * dn2,
        m + 0.5 * dt * dm2,
        h + 0.5 * dt * dh2,
        opt3,
        I_ext,
        ca_i + 0.5 * dt * dca2,
    )
    opt4 = _advance_opt_state(opt_state, dopt3, dt)
    dV4, dn4, dm4, dh4, dopt4, dca4 = _hh_derivatives(
        neuron,
        V + dt * dV3,
        n + dt * dn3,
        m + dt * dm3,
        h + dt * dh3,
        opt4,
        I_ext,
        ca_i + dt * dca3,
    )
    V_new = V + (dt / 6.0) * (dV1 + 2 * dV2 + 2 * dV3 + dV4)
    n_new = float(np.clip(n + (dt / 6.0) * (dn1 + 2 * dn2 + 2 * dn3 + dn4), 0, 1))
    m_new = float(np.clip(m + (dt / 6.0) * (dm1 + 2 * dm2 + 2 * dm3 + dm4), 0, 1))
    h_new = float(np.clip(h + (dt / 6.0) * (dh1 + 2 * dh2 + 2 * dh3 + dh4), 0, 1))
    opt_new = _clip_opt_state(
        {
            k: opt_state[k]
            + (dt / 6.0) * (dopt1[k] + 2 * dopt2[k] + 2 * dopt3[k] + dopt4[k])
            for k in opt_state
        }
    )
    ca_new = max(0.0, ca_i + (dt / 6.0) * (dca1 + 2 * dca2 + 2 * dca3 + dca4))
    return V_new, n_new, m_new, h_new, opt_new, ca_new


def _rk4_step_voltage_clamp(
    neuron: "HodgkinHuxley",
    V: float,
    n: float,
    m: float,
    h: float,
    opt_state: dict[str, float],
    dt: float,
    ca_i: float,
) -> tuple[float, float, float, dict[str, float], float]:
    """Advance voltage-clamp gating variables by one RK4 step.

    Args:
        neuron: The Hodgkin-Huxley neuron model.
        V: Prescribed membrane voltage in mV.
        n: Potassium activation gating variable.
        m: Sodium activation gating variable.
        h: Sodium inactivation gating variable.
        opt_state: Optional channel gating state (name → value).
        dt: Time step in milliseconds.
        ca_i: Current intracellular Ca2+ concentration in mM.

    Returns:
        Tuple of updated (n, m, h, opt_state, ca_i) with gating variables
        clipped to [0, 1] and ca_i floored at 0.0.
    """
    dn1, dm1, dh1, dopt1, dca1 = _gating_derivatives(
        neuron, V, n, m, h, opt_state, ca_i
    )
    opt2 = _advance_opt_state(opt_state, dopt1, 0.5 * dt)
    dn2, dm2, dh2, dopt2, dca2 = _gating_derivatives(
        neuron,
        V,
        n + 0.5 * dt * dn1,
        m + 0.5 * dt * dm1,
        h + 0.5 * dt * dh1,
        opt2,
        ca_i + 0.5 * dt * dca1,
    )
    opt3 = _advance_opt_state(opt_state, dopt2, 0.5 * dt)
    dn3, dm3, dh3, dopt3, dca3 = _gating_derivatives(
        neuron,
        V,
        n + 0.5 * dt * dn2,
        m + 0.5 * dt * dm2,
        h + 0.5 * dt * dh2,
        opt3,
        ca_i + 0.5 * dt * dca2,
    )
    opt4 = _advance_opt_state(opt_state, dopt3, dt)
    dn4, dm4, dh4, dopt4, dca4 = _gating_derivatives(
        neuron,
        V,
        n + dt * dn3,
        m + dt * dm3,
        h + dt * dh3,
        opt4,
        ca_i + dt * dca3,
    )
    n_new = float(np.clip(n + (dt / 6.0) * (dn1 + 2 * dn2 + 2 * dn3 + dn4), 0, 1))
    m_new = float(np.clip(m + (dt / 6.0) * (dm1 + 2 * dm2 + 2 * dm3 + dm4), 0, 1))
    h_new = float(np.clip(h + (dt / 6.0) * (dh1 + 2 * dh2 + 2 * dh3 + dh4), 0, 1))
    opt_new = _clip_opt_state(
        {
            k: opt_state[k]
            + (dt / 6.0) * (dopt1[k] + 2 * dopt2[k] + 2 * dopt3[k] + dopt4[k])
            for k in opt_state
        }
    )
    ca_new = max(0.0, ca_i + (dt / 6.0) * (dca1 + 2 * dca2 + 2 * dca3 + dca4))
    return n_new, m_new, h_new, opt_new, ca_new


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

    When the neuron has optional channels, the DataFrame includes additional
    columns ``{channel_name}_current`` for each channel and
    ``{gating_var_name}`` for each optional gating variable.

    When the neuron has ``calcium_dynamics`` configured, a ``ca_i`` column
    containing intracellular Ca2+ concentration in mM is included.

    Args:
        neuron: The Hodgkin-Huxley neuron object to simulate.
        voltage_protocol: Voltage values in mV to clamp the membrane at for each
            time step. The length of the array determines the simulation duration.

    Returns:
        DataFrame indexed by time in milliseconds (named 'time'), with columns:
        voltage, total_current, sodium_current, potassium_current, leak_current,
        potassium_activation, sodium_activation, sodium_inactivation.
        Optional channel columns are appended when present.
        A ca_i column is appended when calcium_dynamics is configured.
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

    # Pre-allocate optional channel arrays
    opt_ch_currents: dict[str, np.ndarray] = {
        ch.name: np.empty(num_time_steps) for ch in neuron.optional_channels
    }
    opt_gating_arrs: dict[str, np.ndarray] = {
        gv.name: np.empty(num_time_steps)
        for gv in neuron.all_optional_gating_variables()
    }

    # Pre-allocate calcium array if dynamics are active
    ca_arr: np.ndarray | None = (
        np.empty(num_time_steps) if neuron.calcium_dynamics is not None else None
    )
    ca_i: float = (
        neuron.calcium_dynamics.ca_rest if neuron.calcium_dynamics is not None else 0.0
    )

    # Initialise gating variables at steady state for the first voltage
    n_arr[0], m_arr[0], h_arr[0], opt_state = _initialize_gating_variables(
        neuron, voltage_protocol[0]
    )

    # Record initial optional gating state
    for gv_name, val in opt_state.items():
        opt_gating_arrs[gv_name][0] = val

    if ca_arr is not None:
        ca_arr[0] = ca_i

    # Compute initial currents
    V0 = voltage_protocol[0]
    g_Na0 = neuron.g_Na * (m_arr[0] ** 3) * h_arr[0]
    g_K0 = neuron.g_K * (n_arr[0] ** 4)
    I_Na[0] = g_Na0 * (V0 - neuron.E_Na)
    I_K[0] = g_K0 * (V0 - neuron.E_K)
    I_L[0] = neuron.g_L * (V0 - neuron.E_L)
    for ch in neuron.optional_channels:
        opt_ch_currents[ch.name][0] = ch.compute_current(V0, opt_state)
    I_total[0] = (
        I_Na[0]
        + I_K[0]
        + I_L[0]
        + sum(opt_ch_currents[ch.name][0] for ch in neuron.optional_channels)
    )

    # Main simulation loop — all state in plain numpy scalars
    for i in range(1, num_time_steps):
        V = voltage_protocol[i]
        n_prev, m_prev, h_prev = n_arr[i - 1], m_arr[i - 1], h_arr[i - 1]

        n, m, h, opt_state, ca_i = _rk4_step_voltage_clamp(
            neuron, V, n_prev, m_prev, h_prev, opt_state, time_step, ca_i
        )
        n_arr[i], m_arr[i], h_arr[i] = n, m, h

        for gv_name, val in opt_state.items():
            opt_gating_arrs[gv_name][i] = val

        if ca_arr is not None:
            ca_arr[i] = ca_i

        g_Na = neuron.g_Na * (m**3) * h
        g_K = neuron.g_K * (n**4)
        I_Na[i] = g_Na * (V - neuron.E_Na)
        I_K[i] = g_K * (V - neuron.E_K)
        I_L[i] = neuron.g_L * (V - neuron.E_L)
        for ch in neuron.optional_channels:
            opt_ch_currents[ch.name][i] = ch.compute_current(V, opt_state)
        I_total[i] = (
            I_Na[i]
            + I_K[i]
            + I_L[i]
            + sum(opt_ch_currents[ch.name][i] for ch in neuron.optional_channels)
        )

    data: dict[str, np.ndarray] = {
        "voltage": voltage_protocol,
        "total_current": I_total,
        "sodium_current": I_Na,
        "potassium_current": I_K,
        "leak_current": I_L,
        "potassium_activation": n_arr,
        "sodium_activation": m_arr,
        "sodium_inactivation": h_arr,
    }
    for ch in neuron.optional_channels:
        data[f"{ch.name}_current"] = opt_ch_currents[ch.name]
    for gv_name, arr in opt_gating_arrs.items():
        data[gv_name] = arr
    if ca_arr is not None:
        data["ca_i"] = ca_arr

    results = pd.DataFrame(data, index=time_array)
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

    When the neuron has optional channels, the DataFrame includes additional
    columns ``{channel_name}_current`` for each channel and
    ``{gating_var_name}`` for each optional gating variable.

    When the neuron has ``calcium_dynamics`` configured, a ``ca_i`` column
    containing intracellular Ca2+ concentration in mM is included.

    Args:
        neuron: The Hodgkin-Huxley neuron object to simulate.
        current_external: External current in uA/cm^2 for a time-varying current
            waveform. The length of the array determines the simulation duration.

    Returns:
        DataFrame indexed by time in milliseconds (named 'time'), with columns:
        voltage, potassium_activation, sodium_activation, sodium_inactivation.
        Optional channel columns are appended when present.
        A ca_i column is appended when calcium_dynamics is configured.
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

    # Pre-allocate optional channel arrays
    opt_ch_currents: dict[str, np.ndarray] = {
        ch.name: np.empty(num_time_steps) for ch in neuron.optional_channels
    }
    opt_gating_arrs: dict[str, np.ndarray] = {
        gv.name: np.empty(num_time_steps)
        for gv in neuron.all_optional_gating_variables()
    }

    # Pre-allocate calcium array if dynamics are active
    ca_arr: np.ndarray | None = (
        np.empty(num_time_steps) if neuron.calcium_dynamics is not None else None
    )
    ca_i: float = (
        neuron.calcium_dynamics.ca_rest if neuron.calcium_dynamics is not None else 0.0
    )

    # Initialise gating variables at steady state for resting potential
    V_arr[0] = neuron.v_rest
    n_arr[0], m_arr[0], h_arr[0], opt_state = _initialize_gating_variables(
        neuron, neuron.v_rest
    )

    # Record initial optional gating state
    for gv_name, val in opt_state.items():
        opt_gating_arrs[gv_name][0] = val

    if ca_arr is not None:
        ca_arr[0] = ca_i

    # Main simulation loop — all state in plain numpy scalars
    for i in range(1, num_time_steps):
        V = V_arr[i - 1]
        n, m, h = n_arr[i - 1], m_arr[i - 1], h_arr[i - 1]

        V_new, n_new, m_new, h_new, opt_state, ca_i = _rk4_step_current_clamp(
            neuron, V, n, m, h, opt_state, current_external[i - 1], time_step, ca_i
        )

        V_arr[i] = V_new
        n_arr[i], m_arr[i], h_arr[i] = n_new, m_new, h_new
        for gv_name, val in opt_state.items():
            opt_gating_arrs[gv_name][i] = val
        if ca_arr is not None:
            ca_arr[i] = ca_i

    # Compute optional channel currents over the recorded voltage trace
    for i in range(num_time_steps):
        # Rebuild opt_state at each step from the recorded arrays
        step_opt: dict[str, float] = {
            gv_name: float(opt_gating_arrs[gv_name][i]) for gv_name in opt_gating_arrs
        }
        for ch in neuron.optional_channels:
            opt_ch_currents[ch.name][i] = ch.compute_current(float(V_arr[i]), step_opt)

    data: dict[str, np.ndarray] = {
        "voltage": V_arr,
        "potassium_activation": n_arr,
        "sodium_activation": m_arr,
        "sodium_inactivation": h_arr,
    }
    for ch in neuron.optional_channels:
        data[f"{ch.name}_current"] = opt_ch_currents[ch.name]
    for gv_name, arr in opt_gating_arrs.items():
        data[gv_name] = arr
    if ca_arr is not None:
        data["ca_i"] = ca_arr

    results = pd.DataFrame(data, index=time_array)
    results.index.name = "time"
    return results
