"""Clamp simulation functions for the Hodgkin-Huxley model.

This module contains functions for voltage clamp and current clamp experiments
that can be performed on Hodgkin-Huxley neuron objects.

Both clamp modes use a unified gating-state numpy array that covers all channels
(core Na/K/leak and any additional channels) and a single RK4 integrator path.
Gating variables are indexed by position using the ``gating_index`` mapping on
the neuron model, and channel conductance products are computed using the
pre-built ``_channel_gate_info`` index/power arrays.
"""

import logging
import numpy as np
import pandas as pd
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .hodgkin_huxley import HodgkinHuxley

logger = logging.getLogger(__name__)

#: Fixed simulation sampling frequency (Hz). dt = 1000 / SIM_SAMPLING_FREQ ms.
#: 40 kHz (dt = 0.025 ms) is standard for Hodgkin-Huxley models.
SIM_SAMPLING_FREQ: float = 40_000.0


def _setup_simulation(
    num_time_steps: int, sampling_frequency: float
) -> tuple[float, np.ndarray]:
    """Allocate time array for a simulation run and return the time step.

    Args:
        num_time_steps: Total number of time steps in the simulation.
        sampling_frequency: Sampling frequency in Hz.

    Returns:
        Tuple of (time_step, time_array) where time_step is in milliseconds.
    """
    time_step = 1.0 / sampling_frequency * 1000.0
    actual_simulation_time = (num_time_steps - 1) * time_step
    time_array = np.linspace(0, actual_simulation_time, num_time_steps)
    return time_step, time_array


def _initialize_gating_array(
    neuron: "HodgkinHuxley",
    initial_voltage: float,
    ca_i: float = 0.0,
) -> np.ndarray:
    """Compute steady-state gating variables at a given initial voltage.

    Initialises every gating variable (core Na/K/leak and additional channels)
    to its infinite-time steady-state value at *initial_voltage*, returned as
    a flat numpy array indexed by ``neuron.gating_index``.

    Args:
        neuron: The Hodgkin-Huxley neuron model.
        initial_voltage: Initial membrane voltage in mV.
        ca_i: Initial intracellular Ca²⁺ concentration in mM.

    Returns:
        1-D float64 array of length ``len(neuron.all_gating_variables)``
        with each entry set to its steady-state value.
    """
    state = np.empty(len(neuron.all_gating_variables), dtype=np.float64)
    for i, gv in enumerate(neuron.all_gating_variables):
        a = gv.alpha(initial_voltage, ca_i)
        b = gv.beta(initial_voltage, ca_i)
        state[i] = a / (a + b)
    return state


def _compute_channel_current(
    V: float,
    state: np.ndarray,
    neuron: "HodgkinHuxley",
    ch_idx: int,
) -> float:
    """Compute ionic current for a single channel using pre-built index arrays.

    Evaluates ``g_max * prod(state[indices] ** powers) * (V - E_rev)`` using
    the ``_channel_gate_info`` arrays on *neuron* to avoid per-step dict
    lookups.

    Args:
        V: Membrane voltage in mV.
        state: Flat gating state array indexed by ``neuron.gating_index``.
        neuron: The Hodgkin-Huxley neuron model.
        ch_idx: Index into ``neuron.all_channels`` for the channel to evaluate.

    Returns:
        Ionic current in µA/cm².
    """
    ch = neuron.all_channels[ch_idx]
    indices, powers = neuron._channel_gate_info[ch_idx]
    if len(indices):
        g = ch.g_max * float(np.prod(state[indices] ** powers))
    else:
        g = ch.g_max
    return g * (V - ch.reversal_potential(neuron))


def _gating_derivatives(
    neuron: "HodgkinHuxley",
    V: float,
    state: np.ndarray,
    ca_i: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Compute derivatives for all gating variables at a prescribed voltage.

    Used in voltage-clamp mode where V is held constant and only the gating
    variables evolve.  Also computes the Ca²⁺ concentration derivative when
    calcium dynamics are active.

    Args:
        neuron: The Hodgkin-Huxley neuron model.
        V: Membrane voltage in mV.
        state: Current flat gating state array.
        ca_i: Current intracellular Ca²⁺ concentration in mM.

    Returns:
        Tuple of (derivs, dca_i) where derivs is a float64 array of the same
        length as *state* and dca_i is 0.0 when no calcium_dynamics are
        configured.
    """
    derivs = np.empty_like(state)
    for i, gv in enumerate(neuron.all_gating_variables):
        x = state[i]
        derivs[i] = gv.alpha(V, ca_i) * (1.0 - x) - gv.beta(V, ca_i) * x
    if neuron.calcium_dynamics is not None:
        # Build a gating dict for the public calcium_current API
        gs = {gv.name: state[i] for i, gv in enumerate(neuron.all_gating_variables)}
        I_Ca = neuron.calcium_current(V, gs)
        dca_i = neuron.calcium_dynamics.derivative(I_Ca, ca_i)
    else:
        dca_i = 0.0
    return derivs, dca_i


def _hh_derivatives(
    neuron: "HodgkinHuxley",
    V: float,
    state: np.ndarray,
    I_ext: float,
    ca_i: float,
) -> tuple[float, np.ndarray, float]:
    """Compute HH ODE derivatives for the current-clamp system.

    Computes dV/dt by summing currents from all channels (core + additional),
    then delegates to :func:`_gating_derivatives` for the gating and Ca²⁺ ODEs.

    Args:
        neuron: The Hodgkin-Huxley neuron model.
        V: Membrane voltage in mV.
        state: Current flat gating state array.
        I_ext: External current in µA/cm².
        ca_i: Current intracellular Ca²⁺ concentration in mM.

    Returns:
        Tuple of (dV, derivs, dca_i) where dV is dV/dt in mV/ms, derivs is a
        float64 array of gating derivatives, and dca_i is 0.0 when no
        calcium_dynamics are configured.
    """
    I_total = sum(
        _compute_channel_current(V, state, neuron, ch_idx)
        for ch_idx in range(len(neuron.all_channels))
    )
    dV = (I_ext - I_total) / neuron.C_m
    derivs, dca_i = _gating_derivatives(neuron, V, state, ca_i)
    return dV, derivs, dca_i


def _rk4_step_voltage_clamp(
    neuron: "HodgkinHuxley",
    V: float,
    state: np.ndarray,
    dt: float,
    ca_i: float,
) -> tuple[np.ndarray, float]:
    """Advance voltage-clamp gating variables by one RK4 step.

    Voltage is held constant at *V*; only the gating variables and optional
    Ca²⁺ concentration evolve.

    Args:
        neuron: The Hodgkin-Huxley neuron model.
        V: Prescribed membrane voltage in mV.
        state: Current flat gating state array.
        dt: Time step in milliseconds.
        ca_i: Current intracellular Ca²⁺ concentration in mM.

    Returns:
        Tuple of (new_state, new_ca_i) with gating variables clipped to
        [0, 1] and ca_i floored at 0.0.
    """
    d1, dca1 = _gating_derivatives(neuron, V, state, ca_i)
    d2, dca2 = _gating_derivatives(
        neuron, V, state + 0.5 * dt * d1, ca_i + 0.5 * dt * dca1
    )
    d3, dca3 = _gating_derivatives(
        neuron, V, state + 0.5 * dt * d2, ca_i + 0.5 * dt * dca2
    )
    d4, dca4 = _gating_derivatives(neuron, V, state + dt * d3, ca_i + dt * dca3)
    new_state = np.clip(state + (dt / 6.0) * (d1 + 2.0 * d2 + 2.0 * d3 + d4), 0.0, 1.0)
    new_ca = max(0.0, ca_i + (dt / 6.0) * (dca1 + 2 * dca2 + 2 * dca3 + dca4))
    return new_state, new_ca


def _rk4_step_current_clamp(
    neuron: "HodgkinHuxley",
    V: float,
    state: np.ndarray,
    I_ext: float,
    dt: float,
    ca_i: float,
) -> tuple[float, np.ndarray, float]:
    """Advance the current-clamp state by one RK4 step.

    Both the membrane voltage and the gating variables evolve together.

    Args:
        neuron: The Hodgkin-Huxley neuron model.
        V: Membrane voltage in mV.
        state: Current flat gating state array.
        I_ext: External current in µA/cm², held constant over the step.
        dt: Time step in milliseconds.
        ca_i: Current intracellular Ca²⁺ concentration in mM.

    Returns:
        Tuple of (new_V, new_state, new_ca_i) with gating variables
        clipped to [0, 1] and ca_i floored at 0.0.
    """
    dV1, d1, dca1 = _hh_derivatives(neuron, V, state, I_ext, ca_i)
    dV2, d2, dca2 = _hh_derivatives(
        neuron, V + 0.5 * dt * dV1, state + 0.5 * dt * d1, I_ext, ca_i + 0.5 * dt * dca1
    )
    dV3, d3, dca3 = _hh_derivatives(
        neuron, V + 0.5 * dt * dV2, state + 0.5 * dt * d2, I_ext, ca_i + 0.5 * dt * dca2
    )
    dV4, d4, dca4 = _hh_derivatives(
        neuron, V + dt * dV3, state + dt * d3, I_ext, ca_i + dt * dca3
    )
    V_new = V + (dt / 6.0) * (dV1 + 2.0 * dV2 + 2.0 * dV3 + dV4)
    new_state = np.clip(state + (dt / 6.0) * (d1 + 2.0 * d2 + 2.0 * d3 + d4), 0.0, 1.0)
    new_ca = max(0.0, ca_i + (dt / 6.0) * (dca1 + 2 * dca2 + 2 * dca3 + dca4))
    return V_new, new_state, new_ca


def _simulate_voltage_clamp_core(
    neuron: "HodgkinHuxley",
    voltage_protocol: np.ndarray,
    gating_state: dict[str, float],
    ca_i: float,
) -> pd.DataFrame:
    """Run the voltage clamp simulation loop given pre-initialized state.

    This is the shared implementation called by both :func:`simulate_voltage_clamp`
    (steady-state initialization) and :func:`simulate_voltage_clamp_from_state`
    (caller-provided initialization).

    Args:
        neuron: The Hodgkin-Huxley neuron object to simulate.
        voltage_protocol: Voltage values in mV to clamp the membrane at for each
            time step. Must be non-empty and finite.
        gating_state: Initial gating variable values, keyed by variable name.
            Mutated during the simulation; callers should pass a copy if needed.
        ca_i: Initial intracellular Ca²⁺ concentration in mM.

    Returns:
        DataFrame indexed by time in milliseconds (named 'time').
    """
    num_time_steps = len(voltage_protocol)

    duration_ms = (num_time_steps - 1) / SIM_SAMPLING_FREQ * 1000.0
    logger.debug(
        "simulate_voltage_clamp: start — %d steps, %.2f ms, %d channels",
        num_time_steps,
        duration_ms,
        len(neuron.all_channels),
    )

    time_step, time_array = _setup_simulation(num_time_steps, SIM_SAMPLING_FREQ)

    n_gates = len(neuron.all_gating_variables)
    n_channels = len(neuron.all_channels)

    # Pre-allocate 2-D gating array (time × gates) and per-channel current arrays
    gating_arr = np.empty((num_time_steps, n_gates), dtype=np.float64)
    ch_current_arrs = np.empty((num_time_steps, n_channels), dtype=np.float64)
    I_total = np.empty(num_time_steps, dtype=np.float64)

    ca_arr: np.ndarray | None = (
        np.empty(num_time_steps) if neuron.calcium_dynamics is not None else None
    )

    # Initialize all gating variables at steady state for the first voltage
    state = _initialize_gating_array(neuron, voltage_protocol[0], ca_i)
    gating_arr[0, :] = state

    if ca_arr is not None:
        ca_arr[0] = ca_i

    # Compute initial currents
    V0 = voltage_protocol[0]
    ch_currents_0 = [
        _compute_channel_current(V0, state, neuron, j) for j in range(n_channels)
    ]
    ch_current_arrs[0, :] = ch_currents_0
    I_total[0] = sum(ch_currents_0)

    # Main simulation loop
    for i in range(1, num_time_steps):
        V = voltage_protocol[i]

        state, ca_i = _rk4_step_voltage_clamp(neuron, V, state, time_step, ca_i)

        gating_arr[i, :] = state

        if ca_arr is not None:
            ca_arr[i] = ca_i

        ch_currents_i = [
            _compute_channel_current(V, state, neuron, j) for j in range(n_channels)
        ]
        ch_current_arrs[i, :] = ch_currents_i
        I_total[i] = sum(ch_currents_i)

    # Assemble output DataFrame
    gating_index = neuron.gating_index
    data: dict[str, np.ndarray] = {
        "voltage": voltage_protocol,
        "total_current": I_total,
    }
    for j, ch in enumerate(neuron.all_channels):
        data[f"{ch.name}_current"] = ch_current_arrs[:, j]
    for gv in neuron.all_gating_variables:
        data[gv.name] = gating_arr[:, gating_index[gv.name]]
    if ca_arr is not None:
        data["ca_i"] = ca_arr

    results = pd.DataFrame(data, index=time_array)
    results.index.name = "time"
    logger.debug("simulate_voltage_clamp: complete — %d steps", num_time_steps)
    return results


def _simulate_current_clamp_core(
    neuron: "HodgkinHuxley",
    current_external: np.ndarray,
    initial_V: float,
    gating_state: dict[str, float],
    ca_i: float,
) -> pd.DataFrame:
    """Run the current clamp simulation loop given pre-initialized state.

    This is the shared implementation called by both :func:`simulate_current_clamp`
    (steady-state initialization) and :func:`simulate_current_clamp_from_state`
    (caller-provided initialization).

    Args:
        neuron: The Hodgkin-Huxley neuron object to simulate.
        current_external: External current in µA/cm² waveform. Must be non-empty
            and finite.
        initial_V: Initial membrane voltage in mV.
        gating_state: Initial gating variable values, keyed by variable name.
            Mutated during the simulation; callers should pass a copy if needed.
        ca_i: Initial intracellular Ca²⁺ concentration in mM.

    Returns:
        DataFrame indexed by time in milliseconds (named 'time').
    """
    num_time_steps = len(current_external)

    duration_ms = (num_time_steps - 1) / SIM_SAMPLING_FREQ * 1000.0
    logger.debug(
        "simulate_current_clamp: start — %d steps, %.2f ms, %d channels",
        num_time_steps,
        duration_ms,
        len(neuron.all_channels),
    )

    time_step, time_array = _setup_simulation(num_time_steps, SIM_SAMPLING_FREQ)

    n_gates = len(neuron.all_gating_variables)
    n_channels = len(neuron.all_channels)

    V_arr = np.empty(num_time_steps, dtype=np.float64)

    # Pre-allocate 2-D gating array (time × gates) and per-channel current arrays
    gating_arr = np.empty((num_time_steps, n_gates), dtype=np.float64)
    ch_current_arrs = np.empty((num_time_steps, n_channels), dtype=np.float64)
    I_total = np.empty(num_time_steps, dtype=np.float64)

    ca_arr: np.ndarray | None = (
        np.empty(num_time_steps) if neuron.calcium_dynamics is not None else None
    )

    # Initialize at resting potential
    V_arr[0] = neuron.v_rest
    state = _initialize_gating_array(neuron, neuron.v_rest, ca_i)
    gating_arr[0, :] = state

    if ca_arr is not None:
        ca_arr[0] = ca_i

    # Compute initial currents
    V0 = neuron.v_rest
    ch_currents_0 = [
        _compute_channel_current(V0, state, neuron, j) for j in range(n_channels)
    ]
    ch_current_arrs[0, :] = ch_currents_0
    I_total[0] = sum(ch_currents_0)

    # Main simulation loop
    for i in range(1, num_time_steps):
        V = V_arr[i - 1]

        V_new, state, ca_i = _rk4_step_current_clamp(
            neuron, V, state, current_external[i - 1], time_step, ca_i
        )

        V_arr[i] = V_new
        gating_arr[i, :] = state

        if ca_arr is not None:
            ca_arr[i] = ca_i

        ch_currents_i = [
            _compute_channel_current(V_new, state, neuron, j) for j in range(n_channels)
        ]
        ch_current_arrs[i, :] = ch_currents_i
        I_total[i] = sum(ch_currents_i)

    # Assemble output DataFrame
    gating_index = neuron.gating_index
    data: dict[str, np.ndarray] = {
        "voltage": V_arr,
    }
    for j, ch in enumerate(neuron.all_channels):
        data[f"{ch.name}_current"] = ch_current_arrs[:, j]
    data["total_current"] = I_total
    for gv in neuron.all_gating_variables:
        data[gv.name] = gating_arr[:, gating_index[gv.name]]
    if ca_arr is not None:
        data["ca_i"] = ca_arr

    results = pd.DataFrame(data, index=time_array)
    results.index.name = "time"
    logger.debug("simulate_current_clamp: complete — %d steps", num_time_steps)
    return results


def simulate_voltage_clamp(
    neuron: "HodgkinHuxley",
    voltage_protocol: np.ndarray,
) -> pd.DataFrame:
    """Simulate a voltage clamp experiment using the Hodgkin-Huxley model.

    In a voltage clamp experiment, the membrane potential is held at specified
    values and the current required to maintain those voltages is measured. This
    function simulates this process by computing the ionic currents that would
    flow at each voltage step in the protocol.

    The simulation always uses :data:`SIM_SAMPLING_FREQ` (40 kHz, dt = 0.025 ms)
    as the integration time step.

    All ion channels — core (Na, K, leak) and additional — are handled via a
    single unified loop. The DataFrame includes a ``{ch.name}_current`` column
    for every channel and a column for every gating variable.

    When the neuron has ``calcium_dynamics`` configured, a ``ca_i`` column
    containing intracellular Ca²⁺ concentration in mM is included.

    Args:
        neuron: The Hodgkin-Huxley neuron object to simulate.
        voltage_protocol: Voltage values in mV to clamp the membrane at for each
            time step. The length of the array determines the simulation duration.

    Returns:
        DataFrame indexed by time in milliseconds (named 'time'), with columns:
        voltage, total_current, Na_current, K_current, leak_current,
        potassium_activation, sodium_activation, sodium_inactivation.
        Additional channel columns are appended when present.
        A ca_i column is appended when calcium_dynamics is configured.
    """
    if len(voltage_protocol) == 0:
        raise ValueError("voltage_protocol must not be empty.")
    if not np.all(np.isfinite(voltage_protocol)):
        raise ValueError("voltage_protocol must not contain NaN or Inf values.")

    ca_i: float = (
        neuron.calcium_dynamics.ca_rest if neuron.calcium_dynamics is not None else 0.0
    )
    gating_state = _initialize_gating_variables(neuron, voltage_protocol[0], ca_i)
    return _simulate_voltage_clamp_core(neuron, voltage_protocol, gating_state, ca_i)


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

    All ion channels — core (Na, K, leak) and additional — are handled via a
    single unified loop.  The DataFrame includes a ``{ch.name}_current`` column
    for every channel (including core channels) and ``total_current``.

    When the neuron has ``calcium_dynamics`` configured, a ``ca_i`` column
    containing intracellular Ca²⁺ concentration in mM is included.

    Args:
        neuron: The Hodgkin-Huxley neuron object to simulate.
        current_external: External current in µA/cm² for a time-varying current
            waveform. The length of the array determines the simulation duration.

    Returns:
        DataFrame indexed by time in milliseconds (named 'time'), with columns:
        voltage, Na_current, K_current, leak_current, total_current,
        potassium_activation, sodium_activation, sodium_inactivation.
        Additional channel columns are appended when present.
        A ca_i column is appended when calcium_dynamics is configured.
    """
    if len(current_external) == 0:
        raise ValueError("current_external must not be empty.")
    if not np.all(np.isfinite(current_external)):
        raise ValueError("current_external must not contain NaN or Inf values.")

    ca_i: float = (
        neuron.calcium_dynamics.ca_rest if neuron.calcium_dynamics is not None else 0.0
    )
    gating_state = _initialize_gating_variables(neuron, neuron.v_rest, ca_i)
    return _simulate_current_clamp_core(
        neuron, current_external, neuron.v_rest, gating_state, ca_i
    )


def simulate_voltage_clamp_from_state(
    neuron: "HodgkinHuxley",
    voltage_protocol: np.ndarray,
    initial_gating_state: dict[str, float],
    initial_ca_i: float = 0.0,
) -> pd.DataFrame:
    """Simulate a voltage clamp experiment starting from an explicit gating state.

    Identical to :func:`simulate_voltage_clamp` except the initial gating
    variables and Ca²⁺ concentration are provided by the caller rather than
    initialised to steady-state at the first protocol voltage.  This is used
    by continuous simulation mode to carry neuron state across loop iterations.

    Args:
        neuron: The Hodgkin-Huxley neuron object to simulate.
        voltage_protocol: Voltage values in mV to clamp the membrane at for
            each time step.
        initial_gating_state: Gating variable values at the start of the run,
            keyed by variable name (e.g. ``{"m": 0.05, "h": 0.6, "n": 0.3}``).
        initial_ca_i: Initial intracellular Ca²⁺ concentration in mM.

    Returns:
        DataFrame with the same columns as :func:`simulate_voltage_clamp`.
    """
    if len(voltage_protocol) == 0:
        raise ValueError("voltage_protocol must not be empty.")
    if not np.all(np.isfinite(voltage_protocol)):
        raise ValueError("voltage_protocol must not contain NaN or Inf values.")

    num_time_steps = len(voltage_protocol)
    duration_ms = (num_time_steps - 1) / SIM_SAMPLING_FREQ * 1000.0
    logger.debug(
        "simulate_voltage_clamp_from_state: start — %d steps, %.2f ms, %d channels",
        num_time_steps,
        duration_ms,
        len(neuron.all_channels),
    )
    result = _simulate_voltage_clamp_core(
        neuron, voltage_protocol, dict(initial_gating_state), initial_ca_i
    )
    logger.debug(
        "simulate_voltage_clamp_from_state: complete — %d steps", num_time_steps
    )
    return result


def simulate_current_clamp_from_state(
    neuron: "HodgkinHuxley",
    current_external: np.ndarray,
    initial_V: float,
    initial_gating_state: dict[str, float],
    initial_ca_i: float = 0.0,
) -> pd.DataFrame:
    """Simulate a current clamp experiment starting from an explicit neuron state.

    Identical to :func:`simulate_current_clamp` except the initial membrane
    voltage, gating variables, and Ca²⁺ concentration are provided by the
    caller rather than initialised from ``neuron.v_rest``.  This is used by
    continuous simulation mode to carry neuron state across loop iterations.

    Args:
        neuron: The Hodgkin-Huxley neuron object to simulate.
        current_external: External current in µA/cm² for a time-varying current
            waveform.
        initial_V: Initial membrane voltage in mV.
        initial_gating_state: Gating variable values at the start of the run,
            keyed by variable name (e.g. ``{"m": 0.05, "h": 0.6, "n": 0.3}``).
        initial_ca_i: Initial intracellular Ca²⁺ concentration in mM.

    Returns:
        DataFrame with the same columns as :func:`simulate_current_clamp`.
    """
    if len(current_external) == 0:
        raise ValueError("current_external must not be empty.")
    if not np.all(np.isfinite(current_external)):
        raise ValueError("current_external must not contain NaN or Inf values.")

    num_time_steps = len(current_external)
    duration_ms = (num_time_steps - 1) / SIM_SAMPLING_FREQ * 1000.0
    logger.debug(
        "simulate_current_clamp_from_state: start — %d steps, %.2f ms, %d channels",
        num_time_steps,
        duration_ms,
        len(neuron.all_channels),
    )
    result = _simulate_current_clamp_core(
        neuron, current_external, initial_V, dict(initial_gating_state), initial_ca_i
    )
    logger.debug(
        "simulate_current_clamp_from_state: complete — %d steps", num_time_steps
    )
    return result


def simulate_batch(
    neuron: "HodgkinHuxley",
    protocols: Sequence[np.ndarray],
    simulate_fn: Callable[
        ["HodgkinHuxley", np.ndarray], pd.DataFrame
    ] = simulate_voltage_clamp,
    max_workers: int | None = None,
) -> Iterator[pd.DataFrame]:
    """Run multiple simulation protocols in parallel and yield results in order.

    Each protocol is submitted to a process pool and executed independently
    (fresh initial conditions per protocol).  Results are yielded in submission
    order so callers can process them progressively as simulations complete.

    An empty *protocols* sequence yields nothing.  A single-element sequence
    is equivalent to calling *simulate_fn* directly, but still routed through
    the pool.

    Args:
        neuron: The Hodgkin-Huxley neuron model to simulate.
        protocols: Sequence of protocol arrays, each passed individually to
            *simulate_fn*.
        simulate_fn: The simulation function to apply to each protocol.
            Defaults to :func:`simulate_voltage_clamp`; pass
            :func:`simulate_current_clamp` for current-clamp batch runs.
        max_workers: Maximum number of worker processes.  ``None`` (default)
            uses the CPU count.

    Yields:
        DataFrames in the same order as *protocols*, one per protocol.
    """
    if not protocols:
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(simulate_fn, neuron, protocol) for protocol in protocols
        ]
        for future in futures:
            yield future.result()
