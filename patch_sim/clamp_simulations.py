"""Clamp simulation functions for the conductance-based neuron model.

This module contains functions for voltage clamp and current clamp experiments
that can be performed on :class:`~patch_sim.Neuron` objects.

Both clamp modes use a unified gating-state numpy array that covers all channels
(core Na/K/leak and any additional channels) and a single RK4 integrator path.
Gating variables are indexed by position using the ``gating_index`` mapping on
the neuron model, and channel conductance products are computed using the
pre-built ``_channel_gate_info`` index/power arrays.

When ``numba`` is installed, voltage-clamp and current-clamp simulations
automatically use JIT-compiled inner loops for significantly faster execution.
The Numba path is activated only when all gating variables use the six core
Hodgkin-Huxley rate functions and no calcium dynamics are configured; otherwise
the pure-Python path is used transparently.
"""

import logging
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np

try:
    from .numba_kernel import _jit_rk4_cc_loop, _jit_rk4_vc_loop

    HAS_NUMBA: bool = True
except ImportError:
    HAS_NUMBA = False

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


class _JitArrays(NamedTuple):
    """Flat numeric arrays extracted from a neuron model for the Numba JIT kernels."""

    func_ids: np.ndarray
    g_max_arr: np.ndarray
    e_rev_arr: np.ndarray
    gate_starts: np.ndarray
    gate_ends: np.ndarray
    gate_idx_flat: np.ndarray
    powers_flat: np.ndarray


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


def _use_jit(neuron: "Neuron") -> bool:
    """Return True when the Numba JIT path can be used for *neuron*.

    The JIT path requires:

    - numba installed (``HAS_NUMBA`` is True), **and**
    - all gating variables use one of the six core HH rate functions (all
      entries in ``_rate_func_ids`` are ≥ 0), **and**
    - no calcium dynamics (``calcium_dynamics`` is None).

    Args:
        neuron: The conductance-based neuron model to test.

    Returns:
        True if the JIT kernel should be used, False otherwise.
    """
    return (
        HAS_NUMBA
        and neuron.calcium_dynamics is None
        and bool(np.all(neuron._rate_func_ids >= 0))
    )


def _jit_arrays(neuron: "Neuron") -> _JitArrays:
    """Extract flat numeric arrays from *neuron* for the Numba JIT kernels.

    All returned arrays are cached properties on *neuron*, so this function
    performs no allocation — it just bundles references for the two call sites.

    Args:
        neuron: The conductance-based neuron model.

    Returns:
        Named tuple of numpy arrays ready for the JIT kernel.
    """
    return _JitArrays(
        func_ids=neuron._rate_func_ids,
        g_max_arr=neuron._g_max_arr,
        e_rev_arr=neuron._reversal_potentials,
        gate_starts=neuron._gate_starts,
        gate_ends=neuron._gate_ends,
        gate_idx_flat=neuron._gate_idx_flat,
        powers_flat=neuron._flat_powers,
    )


def _initialize_gating_array(
    neuron: "Neuron",
    initial_voltage: float,
    ca_i: float = 0.0,
) -> np.ndarray:
    """Compute steady-state gating variables at a given initial voltage.

    Initialises every gating variable (core Na/K/leak and additional channels)
    to its infinite-time steady-state value at *initial_voltage*, returned as
    a flat numpy array indexed by ``neuron.gating_index``.

    Args:
        neuron: The conductance-based neuron model.
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
    neuron: "Neuron",
    ch_idx: int,
) -> float:
    """Compute ionic current for a single channel using pre-built index arrays.

    Evaluates ``g_max * prod(state[i] ** power) * (V - E_rev)`` using the
    ``_channel_gate_info`` arrays and the pre-computed ``_reversal_potentials``
    on *neuron*.  The gate product is computed with a plain Python loop instead
    of ``np.prod`` to avoid numpy call overhead for 1–4 element arrays.

    Args:
        V: Membrane voltage in mV.
        state: Flat gating state array indexed by ``neuron.gating_index``.
        neuron: The conductance-based neuron model.
        ch_idx: Index into ``neuron.all_channels`` for the channel to evaluate.

    Returns:
        Ionic current in µA/cm².
    """
    ch = neuron.all_channels[ch_idx]
    indices, powers = neuron._channel_gate_info[ch_idx]
    g = ch.g_max
    for idx, pw in zip(indices, powers):
        g *= state[idx] ** pw
    return g * (V - neuron._reversal_potentials[ch_idx])


def _gating_derivatives(
    neuron: "Neuron",
    V: float,
    state: np.ndarray,
    ca_i: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Compute derivatives for all gating variables at a prescribed voltage.

    Used in voltage-clamp mode where V is held constant and only the gating
    variables evolve.  Also computes the Ca²⁺ concentration derivative when
    calcium dynamics are active.

    Args:
        neuron: The conductance-based neuron model.
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
    neuron: "Neuron",
    V: float,
    state: np.ndarray,
    I_ext: float,
    ca_i: float,
) -> tuple[float, np.ndarray, float]:
    """Compute HH ODE derivatives for the current-clamp system.

    Computes dV/dt by summing currents from all channels (core + additional),
    then delegates to :func:`_gating_derivatives` for the gating and Ca²⁺ ODEs.

    Args:
        neuron: The conductance-based neuron model.
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
    neuron: "Neuron",
    V: float,
    state: np.ndarray,
    dt: float,
    ca_i: float,
) -> tuple[np.ndarray, float]:
    """Advance voltage-clamp gating variables by one RK4 step.

    Voltage is held constant at *V*; only the gating variables and optional
    Ca²⁺ concentration evolve.

    Args:
        neuron: The conductance-based neuron model.
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
    neuron: "Neuron",
    V: float,
    state: np.ndarray,
    I_ext: float,
    dt: float,
    ca_i: float,
) -> tuple[float, np.ndarray, float]:
    """Advance the current-clamp state by one RK4 step.

    Both the membrane voltage and the gating variables evolve together.
    Intermediate voltages are clamped to ``[_V_CLAMP_MIN, _V_CLAMP_MAX]``
    to prevent numerical runaway under extreme stimulus inputs.

    Args:
        neuron: The conductance-based neuron model.
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
        neuron,
        _clamp_v(V + 0.5 * dt * dV1),
        state + 0.5 * dt * d1,
        I_ext,
        ca_i + 0.5 * dt * dca1,
    )
    dV3, d3, dca3 = _hh_derivatives(
        neuron,
        _clamp_v(V + 0.5 * dt * dV2),
        state + 0.5 * dt * d2,
        I_ext,
        ca_i + 0.5 * dt * dca2,
    )
    dV4, d4, dca4 = _hh_derivatives(
        neuron, _clamp_v(V + dt * dV3), state + dt * d3, I_ext, ca_i + dt * dca3
    )
    V_new = _clamp_v(V + (dt / 6.0) * (dV1 + 2.0 * dV2 + 2.0 * dV3 + dV4))
    new_state = np.clip(state + (dt / 6.0) * (d1 + 2.0 * d2 + 2.0 * d3 + d4), 0.0, 1.0)
    new_ca = max(0.0, ca_i + (dt / 6.0) * (dca1 + 2 * dca2 + 2 * dca3 + dca4))
    return V_new, new_state, new_ca


def _simulate_voltage_clamp_core(
    neuron: "Neuron",
    voltage_protocol: np.ndarray,
    state0: np.ndarray,
    ca_i: float,
) -> "SimulationResult":
    """Run the voltage clamp simulation loop given pre-initialized state.

    This is the shared implementation called by both :func:`simulate_voltage_clamp`
    (steady-state initialization) and :func:`simulate_voltage_clamp_from_state`
    (caller-provided initialization).

    Args:
        neuron: The conductance-based neuron model to simulate.
        voltage_protocol: Voltage values in mV to clamp the membrane at for each
            time step. Must be non-empty and finite.
        state0: Initial gating state as a 1-D float64 array indexed by
            ``neuron.gating_index``.
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

    n_channels = len(neuron.all_channels)

    if _use_jit(neuron):
        # --- Numba JIT path ---
        ja = _jit_arrays(neuron)
        gating_arr, ch_current_arrs = _jit_rk4_vc_loop(  # type: ignore[name-defined]
            voltage_protocol,
            time_step,
            state0,
            ja.func_ids,
            ja.g_max_arr,
            ja.e_rev_arr,
            ja.gate_starts,
            ja.gate_ends,
            ja.gate_idx_flat,
            ja.powers_flat,
        )
        ca_arr = None
    else:
        # --- Pure-Python path ---
        n_gates = len(neuron.all_gating_variables)

        gating_arr = np.empty((num_time_steps, n_gates), dtype=np.float64)
        ch_current_arrs = np.empty((num_time_steps, n_channels), dtype=np.float64)

        ca_arr = (
            np.empty(num_time_steps) if neuron.calcium_dynamics is not None else None
        )

        state = state0.copy()
        gating_arr[0, :] = state

        if ca_arr is not None:
            ca_arr[0] = ca_i

        V0 = voltage_protocol[0]
        ch_currents_0 = [
            _compute_channel_current(V0, state, neuron, j) for j in range(n_channels)
        ]
        ch_current_arrs[0, :] = ch_currents_0

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

    I_total: np.ndarray = ch_current_arrs.sum(axis=1)

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
    for j, ch in enumerate(neuron.all_channels):
        results[ch.current_name] = ch_current_arrs[:, j]
    for i, gv in enumerate(neuron.all_gating_variables):
        results[gv.name] = gating_arr[:, i]
    if ca_arr is not None:
        results["ca_i"] = ca_arr

    logger.debug("simulate_voltage_clamp: complete — %d steps", num_time_steps)
    return cast("SimulationResult", results)


def _simulate_current_clamp_core(
    neuron: "Neuron",
    current_external: np.ndarray,
    initial_V: float,
    state0: np.ndarray,
    ca_i: float,
) -> "SimulationResult":
    """Run the current clamp simulation loop given pre-initialized state.

    This is the shared implementation called by both :func:`simulate_current_clamp`
    (steady-state initialization) and :func:`simulate_current_clamp_from_state`
    (caller-provided initialization).

    Args:
        neuron: The conductance-based neuron model to simulate.
        current_external: External current in µA/cm² waveform. Must be non-empty
            and finite.
        initial_V: Initial membrane voltage in mV.
        state0: Initial gating state as a 1-D float64 array indexed by
            ``neuron.gating_index``.
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

    n_channels = len(neuron.all_channels)

    if _use_jit(neuron):
        # --- Numba JIT path ---
        ja = _jit_arrays(neuron)
        V_arr, gating_arr, ch_current_arrs = _jit_rk4_cc_loop(  # type: ignore[name-defined]
            current_external,
            time_step,
            initial_V,
            state0,
            neuron.C_m,
            ja.func_ids,
            ja.g_max_arr,
            ja.e_rev_arr,
            ja.gate_starts,
            ja.gate_ends,
            ja.gate_idx_flat,
            ja.powers_flat,
        )
        ca_arr = None
    else:
        # --- Pure-Python path ---
        n_gates = len(neuron.all_gating_variables)

        V_arr = np.empty(num_time_steps, dtype=np.float64)
        gating_arr = np.empty((num_time_steps, n_gates), dtype=np.float64)
        ch_current_arrs = np.empty((num_time_steps, n_channels), dtype=np.float64)

        ca_arr = (
            np.empty(num_time_steps) if neuron.calcium_dynamics is not None else None
        )

        V_arr[0] = initial_V
        state = state0.copy()
        gating_arr[0, :] = state

        if ca_arr is not None:
            ca_arr[0] = ca_i

        ch_currents_0 = [
            _compute_channel_current(initial_V, state, neuron, j)
            for j in range(n_channels)
        ]
        ch_current_arrs[0, :] = ch_currents_0

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
                _compute_channel_current(V_new, state, neuron, j)
                for j in range(n_channels)
            ]
            ch_current_arrs[i, :] = ch_currents_i

    I_total: np.ndarray = ch_current_arrs.sum(axis=1)

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
    for j, ch in enumerate(neuron.all_channels):
        results[ch.current_name] = ch_current_arrs[:, j]
    results["Itotal"] = I_total
    for i, gv in enumerate(neuron.all_gating_variables):
        results[gv.name] = gating_arr[:, i]
    if ca_arr is not None:
        results["ca_i"] = ca_arr

    logger.debug("simulate_current_clamp: complete — %d steps", num_time_steps)
    return cast("SimulationResult", results)


def simulate_voltage_clamp(
    neuron: "Neuron",
    voltage_protocol: np.ndarray,
) -> "SimulationResult":
    """Simulate a voltage clamp experiment using the conductance-based neuron model.

    In a voltage clamp experiment, the membrane potential is held at specified
    values and the current required to maintain those voltages is measured. This
    function simulates this process by computing the ionic currents that would
    flow at each voltage step in the protocol.

    The simulation always uses :data:`SIM_SAMPLING_FREQ` (40 kHz, dt = 0.025 ms)
    as the integration time step.

    All ion channels — core (INa, IK, Ileak) and additional — are handled via a
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
    state0 = _initialize_gating_array(neuron, voltage_protocol[0], ca_i)
    return _simulate_voltage_clamp_core(neuron, voltage_protocol, state0, ca_i)


def simulate_current_clamp(
    neuron: "Neuron",
    current_external: np.ndarray,
) -> "SimulationResult":
    """Simulate a current clamp experiment using the conductance-based neuron model.

    In a current clamp experiment, current is injected into the cell membrane and
    the resulting voltage changes are recorded. This function simulates this process
    by computing the membrane voltage over time in response to the specified
    external current.

    The simulation always uses :data:`SIM_SAMPLING_FREQ` (40 kHz, dt = 0.025 ms)
    as the integration time step.

    All ion channels — core (INa, IK, Ileak) and additional — are handled via a
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
    state0 = _initialize_gating_array(neuron, neuron.v_rest, ca_i)
    return _simulate_current_clamp_core(
        neuron, current_external, neuron.v_rest, state0, ca_i
    )


def simulate_voltage_clamp_from_state(
    neuron: "Neuron",
    voltage_protocol: np.ndarray,
    initial_gating_state: dict[str, float],
    initial_ca_i: float = 0.0,
) -> "SimulationResult":
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
    state0 = np.array(
        [initial_gating_state[gv.name] for gv in neuron.all_gating_variables],
        dtype=np.float64,
    )
    result = _simulate_voltage_clamp_core(
        neuron, voltage_protocol, state0, initial_ca_i
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
) -> "SimulationResult":
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
    state0 = np.array(
        [initial_gating_state[gv.name] for gv in neuron.all_gating_variables],
        dtype=np.float64,
    )
    result = _simulate_current_clamp_core(
        neuron, current_external, initial_V, state0, initial_ca_i
    )
    logger.debug(
        "simulate_current_clamp_from_state: complete — %d steps", num_time_steps
    )
    return result


def simulate_batch(
    neuron: "Neuron",
    protocols: Sequence[np.ndarray],
    simulate_fn: Callable[
        ["Neuron", np.ndarray], "SimulationResult"
    ] = simulate_voltage_clamp,
    max_workers: int | None = None,
) -> "Iterator[SimulationResult]":
    """Run multiple simulation protocols and yield results in order.

    Chooses between three execution strategies based on protocol count and
    whether the Numba JIT path is available:

    - **Single protocol**: runs directly in the calling process — a process
      pool adds pure overhead for one sweep.
    - **JIT sequential** (numba installed, standard HH neuron, no Ca²⁺
      dynamics): runs all protocols sequentially in the calling process.
      The JIT kernel is fast enough that per-sweep serialization overhead of
      a ``ProcessPoolExecutor`` exceeds the compute time.
    - **Parallel pool**: falls back to :class:`~concurrent.futures.ProcessPoolExecutor`
      when the above conditions are not met, parallelising across CPU cores.

    Each protocol is executed independently with fresh initial conditions.
    Results are yielded in submission order.

    An empty *protocols* sequence yields nothing.

    Args:
        neuron: The conductance-based neuron model to simulate.
        protocols: Sequence of protocol arrays, each passed individually to
            *simulate_fn*.
        simulate_fn: The simulation function to apply to each protocol.
            Defaults to :func:`simulate_voltage_clamp`; pass
            :func:`simulate_current_clamp` for current-clamp batch runs.
        max_workers: Maximum number of worker processes used by the parallel
            pool path.  ``None`` (default) uses the CPU count.  Ignored when
            the single-protocol or JIT sequential path is taken.

    Yields:
        Structured arrays in the same order as *protocols*, one per protocol.
    """
    if not protocols:
        return

    if len(protocols) == 1:
        logger.debug("simulate_batch: single-protocol fast-path (1 sweep)")
        yield simulate_fn(neuron, protocols[0])
        return

    _jit_fns = (simulate_voltage_clamp, simulate_current_clamp)
    if simulate_fn in _jit_fns and _use_jit(neuron):
        logger.debug("simulate_batch: JIT sequential path (%d sweeps)", len(protocols))
        for protocol in protocols:
            yield simulate_fn(neuron, protocol)
        return

    logger.debug(
        "simulate_batch: process pool path (%d sweeps, max_workers=%s)",
        len(protocols),
        max_workers,
    )
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(simulate_fn, neuron, protocol) for protocol in protocols
        ]
        for future in futures:
            yield future.result()
