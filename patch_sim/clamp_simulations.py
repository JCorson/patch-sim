"""Clamp simulation functions for the conductance-based neuron model.

This module contains functions for voltage clamp and current clamp experiments
that can be performed on :class:`~patch_sim.Neuron` objects.

Both clamp modes use a unified gating-state dictionary that covers all channels
(core Na/K/leak and any additional channels) and a single RK4 integrator path.
"""

import logging
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from typing import Any

    from .neuron import Neuron

    class _StructuredDtype:
        """Dtype stub for structured arrays — ``names`` is always set."""

        names: tuple[str, ...]

    class SimulationResult(np.ndarray[Any, Any]):
        """A NumPy structured array with named dtype fields.

        Each element represents one time step. Fields include ``"time"``,
        ``"voltage"``, ``"Itotal"``, per-channel currents (e.g. ``"INa"``,
        ``"IK"``), gating variables, and optionally ``"ca_i"``.
        """

        dtype: _StructuredDtype  # type: ignore[assignment]

else:
    SimulationResult = np.ndarray

logger = logging.getLogger(__name__)

#: Fixed simulation sampling frequency (Hz). dt = 1000 / SIM_SAMPLING_FREQ ms.
#: 40 kHz (dt = 0.025 ms) is standard for conductance-based neuron models.
SIM_SAMPLING_FREQ: float = 40_000.0

#: Voltage clamp bounds for the current-clamp RK4 integrator (mV).
#: Physiological voltages span ~[-90, +60] mV; ±150 mV covers all reversal
#: potentials (Ca²⁺ ~+120 mV, Na⁺ ~+60 mV, K⁺ ~-90 mV) with margin while
#: preventing numerical runaway during extreme stimulus inputs.
_V_CLAMP_MIN: float = -150.0
_V_CLAMP_MAX: float = 150.0


def _clamp_v(v: float) -> float:
    """Clamp membrane voltage to the numerical safety range.

    Constrains voltage to [_V_CLAMP_MIN, _V_CLAMP_MAX] to prevent runaway
    integration when extreme external currents are applied.

    Args:
        v: Membrane voltage in mV.

    Returns:
        Voltage clamped to [_V_CLAMP_MIN, _V_CLAMP_MAX] in mV.
    """
    return max(_V_CLAMP_MIN, min(_V_CLAMP_MAX, v))


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


def _initialize_gating_variables(
    neuron: "Neuron",
    initial_voltage: float,
    ca_i: float = 0.0,
) -> dict[str, float]:
    """Compute steady-state gating variables at a given initial voltage.

    Initialises every gating variable (core Na/K/leak and additional channels)
    to its infinite-time steady-state value at *initial_voltage*.

    Args:
        neuron: The conductance-based neuron model.
        initial_voltage: Initial membrane voltage in mV.
        ca_i: Initial intracellular Ca²⁺ concentration in mM.

    Returns:
        Dict mapping each gating variable name to its steady-state value.
    """
    state: dict[str, float] = {}
    for gv in neuron.all_gating_variables:
        a = gv.alpha(initial_voltage, ca_i)
        b = gv.beta(initial_voltage, ca_i)
        state[gv.name] = a / (a + b)
    return state


def _gating_derivatives(
    neuron: "Neuron",
    V: float,
    gating_state: dict[str, float],
    ca_i: float = 0.0,
) -> tuple[dict[str, float], float]:
    """Compute derivatives for all gating variables at a prescribed voltage.

    Used in voltage-clamp mode where V is held constant and only the gating
    variables evolve.  Also computes the Ca²⁺ concentration derivative when
    calcium dynamics are active.

    Args:
        neuron: The conductance-based neuron model.
        V: Membrane voltage in mV.
        gating_state: Current gating state mapping variable name → value.
        ca_i: Current intracellular Ca²⁺ concentration in mM.

    Returns:
        Tuple of (derivs, dca_i) where derivs maps each gating variable name
        to its dx/dt and dca_i is 0.0 when no calcium_dynamics are configured.
    """
    phi = neuron.q10_factor
    derivs: dict[str, float] = {}
    if neuron.calcium_dynamics is not None:
        i_ca_total = 0.0
        for ch in neuron.all_channels:
            if ch.carries_calcium:
                i_ca_total += ch.compute_current(V, gating_state, neuron)
            for gv in ch.gating_variables:
                x = gating_state[gv.name]
                derivs[gv.name] = phi * (
                    gv.alpha(V, ca_i) * (1 - x) - gv.beta(V, ca_i) * x
                )
        dca_i = neuron.calcium_dynamics.derivative(i_ca_total, ca_i)
    else:
        for gv in neuron.all_gating_variables:
            x = gating_state[gv.name]
            derivs[gv.name] = phi * (gv.alpha(V, ca_i) * (1 - x) - gv.beta(V, ca_i) * x)
        dca_i = 0.0
    return derivs, dca_i


def _hh_derivatives(
    neuron: "Neuron",
    V: float,
    gating_state: dict[str, float],
    I_ext: float,
    ca_i: float,
) -> tuple[float, dict[str, float], float]:
    """Compute HH ODE derivatives for the current-clamp system.

    Computes dV/dt by summing currents from all channels (core + additional),
    then delegates to :func:`_gating_derivatives` for the gating and Ca²⁺ ODEs.

    Args:
        neuron: The conductance-based neuron model.
        V: Membrane voltage in mV.
        gating_state: Current gating state mapping variable name → value.
        I_ext: External current in µA/cm².
        ca_i: Current intracellular Ca²⁺ concentration in mM.

    Returns:
        Tuple of (dV, derivs, dca_i) where dV is dV/dt in mV/ms, derivs maps
        each gating variable name to its derivative, and dca_i is 0.0 when no
        calcium_dynamics are configured.
    """
    I_total = 0.0
    for ch in neuron.all_channels:
        I_total += ch.compute_current(V, gating_state, neuron)
    dV = (I_ext - I_total) / neuron.C_m
    derivs, dca_i = _gating_derivatives(neuron, V, gating_state, ca_i)
    return dV, derivs, dca_i


def _advance_state(
    state: dict[str, float],
    derivs: dict[str, float],
    dt: float,
) -> dict[str, float]:
    """Advance gating state by a scaled derivative step.

    Args:
        state: Current gating state.
        derivs: Derivatives dict (same keys as state).
        dt: Scaled time step (may be a half-step for RK4 midpoints) in ms.

    Returns:
        New gating state with each value advanced by dt * deriv[key].
    """
    return {k: state[k] + dt * derivs[k] for k in state}


def _clip_state(state: dict[str, float]) -> dict[str, float]:
    """Clip all gating variable values to [0, 1].

    Args:
        state: Gating state to clip.

    Returns:
        New dict with all values clipped to [0, 1].
    """
    return {k: max(0.0, min(1.0, v)) for k, v in state.items()}


def _rk4_step_voltage_clamp(
    neuron: "Neuron",
    V: float,
    gating_state: dict[str, float],
    dt: float,
    ca_i: float,
) -> tuple[dict[str, float], float]:
    """Advance voltage-clamp gating variables by one RK4 step.

    Voltage is held constant at *V*; only the gating variables and optional
    Ca²⁺ concentration evolve.

    Args:
        neuron: The conductance-based neuron model.
        V: Prescribed membrane voltage in mV.
        gating_state: Current gating state mapping variable name → value.
        dt: Time step in milliseconds.
        ca_i: Current intracellular Ca²⁺ concentration in mM.

    Returns:
        Tuple of (new_gating_state, new_ca_i) with gating variables clipped to
        [0, 1] and ca_i floored at 0.0.
    """
    d1, dca1 = _gating_derivatives(neuron, V, gating_state, ca_i)
    s2 = _advance_state(gating_state, d1, 0.5 * dt)
    d2, dca2 = _gating_derivatives(neuron, V, s2, ca_i + 0.5 * dt * dca1)
    s3 = _advance_state(gating_state, d2, 0.5 * dt)
    d3, dca3 = _gating_derivatives(neuron, V, s3, ca_i + 0.5 * dt * dca2)
    s4 = _advance_state(gating_state, d3, dt)
    d4, dca4 = _gating_derivatives(neuron, V, s4, ca_i + dt * dca3)
    new_state = _clip_state(
        {
            k: gating_state[k] + (dt / 6.0) * (d1[k] + 2 * d2[k] + 2 * d3[k] + d4[k])
            for k in gating_state
        }
    )
    new_ca = max(0.0, ca_i + (dt / 6.0) * (dca1 + 2 * dca2 + 2 * dca3 + dca4))
    return new_state, new_ca


def _rk4_step_current_clamp(
    neuron: "Neuron",
    V: float,
    gating_state: dict[str, float],
    I_ext: float,
    dt: float,
    ca_i: float,
) -> tuple[float, dict[str, float], float]:
    """Advance the current-clamp state by one RK4 step.

    Both the membrane voltage and the gating variables evolve together.

    Args:
        neuron: The conductance-based neuron model.
        V: Membrane voltage in mV.
        gating_state: Current gating state mapping variable name → value.
        I_ext: External current in µA/cm², held constant over the step.
        dt: Time step in milliseconds.
        ca_i: Current intracellular Ca²⁺ concentration in mM.

    Returns:
        Tuple of (new_V, new_gating_state, new_ca_i) with gating variables
        clipped to [0, 1] and ca_i floored at 0.0.
    """
    dV1, d1, dca1 = _hh_derivatives(neuron, V, gating_state, I_ext, ca_i)
    s2 = _advance_state(gating_state, d1, 0.5 * dt)
    dV2, d2, dca2 = _hh_derivatives(
        neuron, _clamp_v(V + 0.5 * dt * dV1), s2, I_ext, ca_i + 0.5 * dt * dca1
    )
    s3 = _advance_state(gating_state, d2, 0.5 * dt)
    dV3, d3, dca3 = _hh_derivatives(
        neuron, _clamp_v(V + 0.5 * dt * dV2), s3, I_ext, ca_i + 0.5 * dt * dca2
    )
    s4 = _advance_state(gating_state, d3, dt)
    dV4, d4, dca4 = _hh_derivatives(
        neuron, _clamp_v(V + dt * dV3), s4, I_ext, ca_i + dt * dca3
    )
    V_new = _clamp_v(V + (dt / 6.0) * (dV1 + 2 * dV2 + 2 * dV3 + dV4))
    new_state = _clip_state(
        {
            k: gating_state[k] + (dt / 6.0) * (d1[k] + 2 * d2[k] + 2 * d3[k] + d4[k])
            for k in gating_state
        }
    )
    new_ca = max(0.0, ca_i + (dt / 6.0) * (dca1 + 2 * dca2 + 2 * dca3 + dca4))
    return V_new, new_state, new_ca


def _simulate_voltage_clamp_core(
    neuron: "Neuron",
    voltage_protocol: np.ndarray,
    gating_state: dict[str, float],
    ca_i: float,
) -> SimulationResult:
    """Run the voltage clamp simulation loop given pre-initialized state.

    This is the shared implementation called by both :func:`simulate_voltage_clamp`
    (steady-state initialization) and :func:`simulate_voltage_clamp_from_state`
    (caller-provided initialization).

    Args:
        neuron: The conductance-based neuron model to simulate.
        voltage_protocol: Voltage values in mV to clamp the membrane at for each
            time step. Must be non-empty and finite.
        gating_state: Initial gating variable values, keyed by variable name.
            Mutated during the simulation; callers should pass a copy if needed.
        ca_i: Initial intracellular Ca²⁺ concentration in mM.

    Returns:
        Structured array with one element per time step and a ``"time"`` field.
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

    ch_current_arrs: dict[str, np.ndarray] = {
        ch.name: np.empty(num_time_steps) for ch in neuron.all_channels
    }
    I_total = np.empty(num_time_steps)

    gating_arrs: dict[str, np.ndarray] = {
        gv.name: np.empty(num_time_steps) for gv in neuron.all_gating_variables
    }

    ca_arr: np.ndarray | None = (
        np.empty(num_time_steps) if neuron.calcium_dynamics is not None else None
    )

    # Record initial state. Callers must pass a gating_state whose keys exactly
    # match the neuron's gating variables; any mismatch (e.g. stale keys from a
    # neuron switch mid-iteration) is a programming error. Fail loudly rather
    # than leaving index 0 uninitialised or silently dropping values.
    if set(gating_state.keys()) != set(gating_arrs.keys()):
        raise ValueError(
            "gating_state keys do not match the neuron's gating variables: "
            f"got {sorted(gating_state.keys())}, "
            f"expected {sorted(gating_arrs.keys())}"
        )
    for gv_name, val in gating_state.items():
        gating_arrs[gv_name][0] = val

    if ca_arr is not None:
        ca_arr[0] = ca_i

    # Compute initial currents
    V0 = voltage_protocol[0]
    ch_currents_0 = [
        ch.compute_current(V0, gating_state, neuron) for ch in neuron.all_channels
    ]
    for ch, i_ch in zip(neuron.all_channels, ch_currents_0):
        ch_current_arrs[ch.name][0] = i_ch
    I_total[0] = sum(ch_currents_0)

    # Main simulation loop
    for i in range(1, num_time_steps):
        V = voltage_protocol[i]

        gating_state, ca_i = _rk4_step_voltage_clamp(
            neuron, V, gating_state, time_step, ca_i
        )

        for gv_name, val in gating_state.items():
            gating_arrs[gv_name][i] = val

        if ca_arr is not None:
            ca_arr[i] = ca_i

        ch_currents_i = [
            ch.compute_current(V, gating_state, neuron) for ch in neuron.all_channels
        ]
        for ch, i_ch in zip(neuron.all_channels, ch_currents_i):
            ch_current_arrs[ch.name][i] = i_ch
        I_total[i] = sum(ch_currents_i)

    # Assemble output structured array
    fields: list[tuple[str, type]] = [
        ("time", np.float64),
        ("voltage", np.float64),
        ("Itotal", np.float64),
    ]
    for ch in neuron.all_channels:
        fields.append((ch.current_name, np.float64))
    for gv in neuron.all_gating_variables:
        fields.append((gv.name, np.float64))
    if ca_arr is not None:
        fields.append(("ca_i", np.float64))

    results = np.empty(num_time_steps, dtype=np.dtype(fields))
    results["time"] = time_array
    results["voltage"] = voltage_protocol
    results["Itotal"] = I_total
    for ch in neuron.all_channels:
        results[ch.current_name] = ch_current_arrs[ch.name]
    for gv in neuron.all_gating_variables:
        results[gv.name] = gating_arrs[gv.name]
    if ca_arr is not None:
        results["ca_i"] = ca_arr

    logger.debug("simulate_voltage_clamp: complete — %d steps", num_time_steps)
    return cast("SimulationResult", results)


def _simulate_current_clamp_core(
    neuron: "Neuron",
    current_external: np.ndarray,
    initial_V: float,
    gating_state: dict[str, float],
    ca_i: float,
) -> SimulationResult:
    """Run the current clamp simulation loop given pre-initialized state.

    This is the shared implementation called by both :func:`simulate_current_clamp`
    (steady-state initialization) and :func:`simulate_current_clamp_from_state`
    (caller-provided initialization).

    Args:
        neuron: The conductance-based neuron model to simulate.
        current_external: External current in µA/cm² waveform. Must be non-empty
            and finite.
        initial_V: Initial membrane voltage in mV.
        gating_state: Initial gating variable values, keyed by variable name.
            Mutated during the simulation; callers should pass a copy if needed.
        ca_i: Initial intracellular Ca²⁺ concentration in mM.

    Returns:
        Structured array with one element per time step and a ``"time"`` field.
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

    V_arr = np.empty(num_time_steps)
    V_arr[0] = initial_V

    ch_current_arrs: dict[str, np.ndarray] = {
        ch.name: np.empty(num_time_steps) for ch in neuron.all_channels
    }
    I_total = np.empty(num_time_steps)

    gating_arrs: dict[str, np.ndarray] = {
        gv.name: np.empty(num_time_steps) for gv in neuron.all_gating_variables
    }

    ca_arr: np.ndarray | None = (
        np.empty(num_time_steps) if neuron.calcium_dynamics is not None else None
    )

    # Record initial state. Same contract as _simulate_voltage_clamp_core:
    # gating_state keys must match the neuron's gating variables exactly.
    if set(gating_state.keys()) != set(gating_arrs.keys()):
        raise ValueError(
            "gating_state keys do not match the neuron's gating variables: "
            f"got {sorted(gating_state.keys())}, "
            f"expected {sorted(gating_arrs.keys())}"
        )
    for gv_name, val in gating_state.items():
        gating_arrs[gv_name][0] = val

    if ca_arr is not None:
        ca_arr[0] = ca_i

    # Compute initial currents
    ch_currents_0 = [
        ch.compute_current(initial_V, gating_state, neuron)
        for ch in neuron.all_channels
    ]
    for ch, i_ch in zip(neuron.all_channels, ch_currents_0):
        ch_current_arrs[ch.name][0] = i_ch
    I_total[0] = sum(ch_currents_0)

    # Main simulation loop
    for i in range(1, num_time_steps):
        V = V_arr[i - 1]

        V_new, gating_state, ca_i = _rk4_step_current_clamp(
            neuron,
            V,
            gating_state,
            current_external[i - 1],
            time_step,
            ca_i,
        )

        V_arr[i] = V_new
        for gv_name, val in gating_state.items():
            gating_arrs[gv_name][i] = val
        if ca_arr is not None:
            ca_arr[i] = ca_i

        ch_currents_i = [
            ch.compute_current(V_new, gating_state, neuron)
            for ch in neuron.all_channels
        ]
        for ch, i_ch in zip(neuron.all_channels, ch_currents_i):
            ch_current_arrs[ch.name][i] = i_ch
        I_total[i] = sum(ch_currents_i)

    # Assemble output structured array
    fields: list[tuple[str, type]] = [("time", np.float64), ("voltage", np.float64)]
    for ch in neuron.all_channels:
        fields.append((ch.current_name, np.float64))
    fields.append(("Itotal", np.float64))
    for gv in neuron.all_gating_variables:
        fields.append((gv.name, np.float64))
    if ca_arr is not None:
        fields.append(("ca_i", np.float64))

    results = np.empty(num_time_steps, dtype=np.dtype(fields))
    results["time"] = time_array
    results["voltage"] = V_arr
    for ch in neuron.all_channels:
        results[ch.current_name] = ch_current_arrs[ch.name]
    results["Itotal"] = I_total
    for gv in neuron.all_gating_variables:
        results[gv.name] = gating_arrs[gv.name]
    if ca_arr is not None:
        results["ca_i"] = ca_arr

    logger.debug("simulate_current_clamp: complete — %d steps", num_time_steps)
    return cast("SimulationResult", results)


def simulate_voltage_clamp(
    neuron: "Neuron",
    voltage_protocol: np.ndarray,
) -> SimulationResult:
    """Simulate a voltage clamp experiment using the conductance-based neuron model.

    In a voltage clamp experiment, the membrane potential is held at specified
    values and the current required to maintain those voltages is measured. This
    function simulates this process by computing the ionic currents that would
    flow at each voltage step in the protocol.

    The simulation always uses :data:`SIM_SAMPLING_FREQ` (40 kHz, dt = 0.025 ms)
    as the integration time step.

    All ion channels — core (INa, IK, INaL, IKL) and additional — are handled via a
    single unified loop. The result includes a field named after each channel
    for every channel and a field for every gating variable.

    When the neuron has ``calcium_dynamics`` configured, a ``ca_i`` field
    containing intracellular Ca²⁺ concentration in mM is included.

    Args:
        neuron: The conductance-based neuron model to simulate.
        voltage_protocol: Voltage values in mV to clamp the membrane at for each
            time step. The length of the array determines the simulation duration.

    Returns:
        Structured array with one element per time step. Field ``"time"`` holds
        the time axis in milliseconds. Remaining fields: ``"voltage"``,
        ``"Itotal"``, per-channel current fields, gating variable fields, and
        optionally ``"ca_i"``.
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
    neuron: "Neuron",
    current_external: np.ndarray,
) -> SimulationResult:
    """Simulate a current clamp experiment using the conductance-based neuron model.

    In a current clamp experiment, current is injected into the cell membrane and
    the resulting voltage changes are recorded. This function simulates this process
    by computing the membrane voltage over time in response to the specified
    external current.

    The simulation always uses :data:`SIM_SAMPLING_FREQ` (40 kHz, dt = 0.025 ms)
    as the integration time step.

    All ion channels — core (INa, IK, INaL, IKL) and additional — are handled via a
    single unified loop. The result includes a field named after each channel
    for every channel (including core channels) and ``"Itotal"``.

    When the neuron has ``calcium_dynamics`` configured, a ``ca_i`` field
    containing intracellular Ca²⁺ concentration in mM is included.

    Args:
        neuron: The conductance-based neuron model to simulate.
        current_external: External current in µA/cm² for a time-varying current
            waveform. The length of the array determines the simulation duration.

    Returns:
        Structured array with one element per time step. Field ``"time"`` holds
        the time axis in milliseconds. Remaining fields: ``"voltage"``,
        per-channel current fields, ``"Itotal"``, gating variable fields, and
        optionally ``"ca_i"``.
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
    neuron: "Neuron",
    voltage_protocol: np.ndarray,
    initial_gating_state: dict[str, float],
    initial_ca_i: float = 0.0,
) -> SimulationResult:
    """Simulate a voltage clamp experiment starting from an explicit gating state.

    Identical to :func:`simulate_voltage_clamp` except the initial gating
    variables and Ca²⁺ concentration are provided by the caller rather than
    initialised to steady-state at the first protocol voltage.  This is used
    by continuous simulation mode to carry neuron state across loop iterations.

    Args:
        neuron: The conductance-based neuron model to simulate.
        voltage_protocol: Voltage values in mV to clamp the membrane at for
            each time step.
        initial_gating_state: Gating variable values at the start of the run,
            keyed by variable name (e.g. ``{"m": 0.05, "h": 0.6, "n": 0.3}``).
        initial_ca_i: Initial intracellular Ca²⁺ concentration in mM.

    Returns:
        Structured array with the same fields as :func:`simulate_voltage_clamp`.
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
    neuron: "Neuron",
    current_external: np.ndarray,
    initial_V: float,
    initial_gating_state: dict[str, float],
    initial_ca_i: float = 0.0,
) -> SimulationResult:
    """Simulate a current clamp experiment starting from an explicit neuron state.

    Identical to :func:`simulate_current_clamp` except the initial membrane
    voltage, gating variables, and Ca²⁺ concentration are provided by the
    caller rather than initialised from ``neuron.v_rest``.  This is used by
    continuous simulation mode to carry neuron state across loop iterations.

    Args:
        neuron: The conductance-based neuron model to simulate.
        current_external: External current in µA/cm² for a time-varying current
            waveform.
        initial_V: Initial membrane voltage in mV.
        initial_gating_state: Gating variable values at the start of the run,
            keyed by variable name (e.g. ``{"m": 0.05, "h": 0.6, "n": 0.3}``).
        initial_ca_i: Initial intracellular Ca²⁺ concentration in mM.

    Returns:
        Structured array with the same fields as :func:`simulate_current_clamp`.
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
    neuron: "Neuron",
    protocols: Sequence[np.ndarray],
    simulate_fn: Callable[
        ["Neuron", np.ndarray], SimulationResult
    ] = simulate_voltage_clamp,
    max_workers: int | None = None,
) -> Iterator[SimulationResult]:
    """Run multiple simulation protocols in parallel and yield results in order.

    Each protocol is submitted to a process pool and executed independently
    (fresh initial conditions per protocol).  Results are yielded in submission
    order so callers can process them progressively as simulations complete.

    An empty *protocols* sequence yields nothing.  A single-element sequence
    is equivalent to calling *simulate_fn* directly, but still routed through
    the pool.

    Args:
        neuron: The conductance-based neuron model to simulate.
        protocols: Sequence of protocol arrays, each passed individually to
            *simulate_fn*.
        simulate_fn: The simulation function to apply to each protocol.
            Defaults to :func:`simulate_voltage_clamp`; pass
            :func:`simulate_current_clamp` for current-clamp batch runs.
        max_workers: Maximum number of worker processes.  ``None`` (default)
            uses the CPU count.

    Yields:
        Structured arrays in the same order as *protocols*, one per protocol.
    """
    if not protocols:
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(simulate_fn, neuron, protocol) for protocol in protocols
        ]
        for future in futures:
            yield future.result()
