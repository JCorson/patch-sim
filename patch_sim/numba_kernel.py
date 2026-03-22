"""Numba JIT-compiled simulation kernels for the Hodgkin-Huxley model.

This module provides ``@numba.njit``-compiled inner loops for voltage-clamp
and current-clamp simulations.  It is **optional** — ``clamp_simulations``
imports it with a ``try/except`` and falls back to pure-Python paths when
numba is not installed.

The six standard Hodgkin-Huxley rate functions (alpha/beta for n, m, h) are
inlined as pure math inside the JIT boundary using integer dispatch IDs that
correspond to the values stored in
:attr:`~patch_sim.hodgkin_huxley.HodgkinHuxley._rate_func_ids`.

Rate-function ID table
----------------------
- 0 → alpha_n
- 1 → beta_n
- 2 → alpha_m
- 3 → beta_m
- 4 → alpha_h
- 5 → beta_h

Any gating variable whose alpha or beta is not one of these six functions
receives ID ``-1`` (sentinel) and causes the JIT path to be skipped.
"""

import math

import numba
import numpy as np

#: Clipping limit for ``_safe_exp_jit``: matches ``SAFE_EXP_CLIP_MAX`` in utils.
_EXP_CLIP: float = 100.0

#: Singularity threshold for near-zero denominators in HH rate functions.
_SING: float = 1e-6


@numba.njit(cache=True)  # type: ignore[misc]
def _safe_exp_jit(x: float) -> float:
    """Numba-compatible exponential with overflow protection.

    Clips the argument to ``[-100, 100]`` before calling ``math.exp``,
    matching the behaviour of :func:`~patch_sim.utils.safe_exp`.

    Args:
        x: Exponent value.

    Returns:
        ``exp(clip(x, -100, 100))``.
    """
    if x > _EXP_CLIP:
        return math.exp(_EXP_CLIP)
    if x < -_EXP_CLIP:
        return math.exp(-_EXP_CLIP)
    return math.exp(x)


@numba.njit(cache=True)  # type: ignore[misc]
def _eval_rate(func_id: int, V: float) -> float:
    """Evaluate one of the six core HH rate functions by integer ID.

    Dispatches to the inlined rate equation corresponding to *func_id*.
    ID values are defined in the module docstring.

    Args:
        func_id: Integer ID in the range 0–5.
        V: Membrane voltage in mV.

    Returns:
        Rate value in 1/ms.
    """
    if func_id == 0:  # alpha_n
        if abs(V + 55.0) < _SING:
            return 0.1
        return 0.01 * (V + 55.0) / (1.0 - _safe_exp_jit(-(V + 55.0) / 10.0))
    elif func_id == 1:  # beta_n
        return 0.125 * _safe_exp_jit(-(V + 65.0) / 80.0)
    elif func_id == 2:  # alpha_m
        if abs(V + 40.0) < _SING:
            return 1.0
        return 0.1 * (V + 40.0) / (1.0 - _safe_exp_jit(-(V + 40.0) / 10.0))
    elif func_id == 3:  # beta_m
        return 4.0 * _safe_exp_jit(-(V + 65.0) / 18.0)
    elif func_id == 4:  # alpha_h
        return 0.07 * _safe_exp_jit(-(V + 65.0) / 20.0)
    else:  # func_id == 5  (beta_h) — only reachable when _use_jit guards are satisfied
        return 1.0 / (1.0 + _safe_exp_jit(-(V + 35.0) / 10.0))


@numba.njit(cache=True)  # type: ignore[misc]
def _jit_gating_derivatives(
    V: float,
    state: np.ndarray,
    func_ids: np.ndarray,
) -> np.ndarray:
    """Compute gating derivatives for all variables at a fixed voltage.

    Evaluates ``dx/dt = alpha(V) * (1 − x) − beta(V) * x`` for each gate
    using the integer ID dispatch table *func_ids*.

    Args:
        V: Membrane voltage in mV.
        state: 1-D float64 array of current gate values.
        func_ids: 2-D int32 array of shape ``(n_gates, 2)`` where column 0
            holds the alpha ID and column 1 the beta ID.

    Returns:
        1-D float64 array of derivatives, same length as *state*.
    """
    n = len(state)
    derivs = np.empty(n)
    for i in range(n):
        x = state[i]
        a = _eval_rate(func_ids[i, 0], V)
        b = _eval_rate(func_ids[i, 1], V)
        derivs[i] = a * (1.0 - x) - b * x
    return derivs


@numba.njit(cache=True)  # type: ignore[misc]
def _jit_channel_currents(
    V: float,
    state: np.ndarray,
    g_max_arr: np.ndarray,
    e_rev_arr: np.ndarray,
    gate_starts: np.ndarray,
    gate_ends: np.ndarray,
    gate_idx_flat: np.ndarray,
    powers_flat: np.ndarray,
    out: np.ndarray,
) -> float:
    """Compute per-channel currents and return the total.

    Writes each channel's ionic current into *out* in place and returns
    their sum.  Conductance is ``g_max * product(state[i]**power)`` over
    the gate slice defined by *gate_starts* / *gate_ends*.

    Args:
        V: Membrane voltage in mV.
        state: Current gating state array.
        g_max_arr: Maximum conductance per channel in mS/cm².
        e_rev_arr: Reversal potential per channel in mV.
        gate_starts: Start index into flat gate arrays for each channel.
        gate_ends: End index into flat gate arrays for each channel.
        gate_idx_flat: Flat array of state indices for every gate.
        powers_flat: Flat array of gate powers for every gate.
        out: Pre-allocated 1-D float64 output array, length n_channels.

    Returns:
        Sum of all channel currents in µA/cm².
    """
    n_channels = len(g_max_arr)
    total = 0.0
    for j in range(n_channels):
        g = g_max_arr[j]
        for k in range(gate_starts[j], gate_ends[j]):
            g *= state[gate_idx_flat[k]] ** powers_flat[k]
        current = g * (V - e_rev_arr[j])
        out[j] = current
        total += current
    return total


@numba.njit(cache=True)  # type: ignore[misc]
def _jit_hh_derivatives(
    V: float,
    state: np.ndarray,
    I_ext: float,
    C_m: float,
    func_ids: np.ndarray,
    g_max_arr: np.ndarray,
    e_rev_arr: np.ndarray,
    gate_starts: np.ndarray,
    gate_ends: np.ndarray,
    gate_idx_flat: np.ndarray,
    powers_flat: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Compute HH ODE derivatives for the full current-clamp system.

    Returns ``dV/dt`` and the gating-variable derivatives computed together
    in one pass to avoid redundant rate-function evaluations in the RK4 loop.

    Args:
        V: Membrane voltage in mV.
        state: Current gating state array.
        I_ext: External injected current in µA/cm².
        C_m: Membrane capacitance in µF/cm².
        func_ids: Rate-function ID table, shape ``(n_gates, 2)``.
        g_max_arr: Maximum conductance per channel in mS/cm².
        e_rev_arr: Reversal potential per channel in mV.
        gate_starts: Start index into flat gate arrays for each channel.
        gate_ends: End index into flat gate arrays for each channel.
        gate_idx_flat: Flat array of state indices for every gate.
        powers_flat: Flat array of gate powers for every gate.

    Returns:
        Tuple of ``(dV, gate_derivs)`` where ``dV`` is in mV/ms and
        ``gate_derivs`` is a 1-D float64 array.
    """
    n_channels = len(g_max_arr)
    I_total = 0.0
    for j in range(n_channels):
        g = g_max_arr[j]
        for k in range(gate_starts[j], gate_ends[j]):
            g *= state[gate_idx_flat[k]] ** powers_flat[k]
        I_total += g * (V - e_rev_arr[j])
    dV = (I_ext - I_total) / C_m
    gate_derivs = _jit_gating_derivatives(V, state, func_ids)
    return dV, gate_derivs


@numba.njit(cache=True)  # type: ignore[misc]
def _jit_rk4_vc_loop(
    voltage_protocol: np.ndarray,
    dt: float,
    state0: np.ndarray,
    func_ids: np.ndarray,
    g_max_arr: np.ndarray,
    e_rev_arr: np.ndarray,
    gate_starts: np.ndarray,
    gate_ends: np.ndarray,
    gate_idx_flat: np.ndarray,
    powers_flat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the full voltage-clamp RK4 simulation loop under Numba JIT.

    All parameters are flat numeric arrays so the loop can be compiled.
    Returns the same 2-D arrays that the Python path assembles, allowing
    the DataFrame construction code to be shared.

    Args:
        voltage_protocol: 1-D float64 array of clamped voltages in mV.
        dt: Integration time step in milliseconds.
        state0: Initial gating state, 1-D float64 of length ``n_gates``.
        func_ids: Rate-function ID table, shape ``(n_gates, 2)``, dtype int32.
        g_max_arr: Maximum conductance per channel in mS/cm².
        e_rev_arr: Reversal potential per channel in mV.
        gate_starts: Start index into flat gate arrays for each channel.
        gate_ends: End index into flat gate arrays for each channel.
        gate_idx_flat: Flat array of state indices for every gate.
        powers_flat: Flat array of gate powers for every gate.

    Returns:
        Tuple of ``(gating_arr, ch_currents_2d)`` where *gating_arr* has
        shape ``(n_steps, n_gates)`` and *ch_currents_2d* has shape
        ``(n_steps, n_channels)``.
    """
    n_steps = len(voltage_protocol)
    n_gates = len(state0)
    n_channels = len(g_max_arr)

    gating_arr = np.empty((n_steps, n_gates))
    ch_currents_2d = np.empty((n_steps, n_channels))

    state = state0.copy()
    gating_arr[0] = state

    # Initial channel currents
    _jit_channel_currents(
        voltage_protocol[0],
        state,
        g_max_arr,
        e_rev_arr,
        gate_starts,
        gate_ends,
        gate_idx_flat,
        powers_flat,
        ch_currents_2d[0],
    )

    dt_6 = dt / 6.0
    dt_half = 0.5 * dt

    for i in range(1, n_steps):
        V = voltage_protocol[i]

        d1 = _jit_gating_derivatives(V, state, func_ids)
        d2 = _jit_gating_derivatives(V, state + dt_half * d1, func_ids)
        d3 = _jit_gating_derivatives(V, state + dt_half * d2, func_ids)
        d4 = _jit_gating_derivatives(V, state + dt * d3, func_ids)

        new_state = state + dt_6 * (d1 + 2.0 * d2 + 2.0 * d3 + d4)
        for k in range(n_gates):
            if new_state[k] < 0.0:
                new_state[k] = 0.0
            elif new_state[k] > 1.0:
                new_state[k] = 1.0
        state = new_state
        gating_arr[i] = state

        _jit_channel_currents(
            V,
            state,
            g_max_arr,
            e_rev_arr,
            gate_starts,
            gate_ends,
            gate_idx_flat,
            powers_flat,
            ch_currents_2d[i],
        )

    return gating_arr, ch_currents_2d


@numba.njit(cache=True)  # type: ignore[misc]
def _jit_rk4_cc_loop(
    current_external: np.ndarray,
    dt: float,
    V0: float,
    state0: np.ndarray,
    C_m: float,
    func_ids: np.ndarray,
    g_max_arr: np.ndarray,
    e_rev_arr: np.ndarray,
    gate_starts: np.ndarray,
    gate_ends: np.ndarray,
    gate_idx_flat: np.ndarray,
    powers_flat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the full current-clamp RK4 simulation loop under Numba JIT.

    Integrates both membrane voltage and gating variables simultaneously.
    All parameters are flat numeric arrays for Numba compatibility.

    Args:
        current_external: 1-D float64 array of injected currents in µA/cm².
        dt: Integration time step in milliseconds.
        V0: Initial membrane voltage in mV.
        state0: Initial gating state, 1-D float64 of length ``n_gates``.
        C_m: Membrane capacitance in µF/cm².
        func_ids: Rate-function ID table, shape ``(n_gates, 2)``, dtype int32.
        g_max_arr: Maximum conductance per channel in mS/cm².
        e_rev_arr: Reversal potential per channel in mV.
        gate_starts: Start index into flat gate arrays for each channel.
        gate_ends: End index into flat gate arrays for each channel.
        gate_idx_flat: Flat array of state indices for every gate.
        powers_flat: Flat array of gate powers for every gate.

    Returns:
        Tuple of ``(V_arr, gating_arr, ch_currents_2d)`` where *V_arr* has
        length ``n_steps``, *gating_arr* has shape ``(n_steps, n_gates)``,
        and *ch_currents_2d* has shape ``(n_steps, n_channels)``.
    """
    n_steps = len(current_external)
    n_gates = len(state0)
    n_channels = len(g_max_arr)

    V_arr = np.empty(n_steps)
    gating_arr = np.empty((n_steps, n_gates))
    ch_currents_2d = np.empty((n_steps, n_channels))

    V = V0
    V_arr[0] = V
    state = state0.copy()
    gating_arr[0] = state

    # Initial channel currents
    _jit_channel_currents(
        V,
        state,
        g_max_arr,
        e_rev_arr,
        gate_starts,
        gate_ends,
        gate_idx_flat,
        powers_flat,
        ch_currents_2d[0],
    )

    dt_6 = dt / 6.0
    dt_half = 0.5 * dt

    for i in range(1, n_steps):
        # current_external[i-1] drives the step that produces V_arr[i],
        # matching the Python path in simulate_current_clamp.
        I_ext = current_external[i - 1]

        dV1, d1 = _jit_hh_derivatives(
            V,
            state,
            I_ext,
            C_m,
            func_ids,
            g_max_arr,
            e_rev_arr,
            gate_starts,
            gate_ends,
            gate_idx_flat,
            powers_flat,
        )
        dV2, d2 = _jit_hh_derivatives(
            V + dt_half * dV1,
            state + dt_half * d1,
            I_ext,
            C_m,
            func_ids,
            g_max_arr,
            e_rev_arr,
            gate_starts,
            gate_ends,
            gate_idx_flat,
            powers_flat,
        )
        dV3, d3 = _jit_hh_derivatives(
            V + dt_half * dV2,
            state + dt_half * d2,
            I_ext,
            C_m,
            func_ids,
            g_max_arr,
            e_rev_arr,
            gate_starts,
            gate_ends,
            gate_idx_flat,
            powers_flat,
        )
        dV4, d4 = _jit_hh_derivatives(
            V + dt * dV3,
            state + dt * d3,
            I_ext,
            C_m,
            func_ids,
            g_max_arr,
            e_rev_arr,
            gate_starts,
            gate_ends,
            gate_idx_flat,
            powers_flat,
        )

        V = V + dt_6 * (dV1 + 2.0 * dV2 + 2.0 * dV3 + dV4)
        new_state = state + dt_6 * (d1 + 2.0 * d2 + 2.0 * d3 + d4)
        for k in range(n_gates):
            if new_state[k] < 0.0:
                new_state[k] = 0.0
            elif new_state[k] > 1.0:
                new_state[k] = 1.0
        state = new_state
        V_arr[i] = V
        gating_arr[i] = state

        _jit_channel_currents(
            V,
            state,
            g_max_arr,
            e_rev_arr,
            gate_starts,
            gate_ends,
            gate_idx_flat,
            powers_flat,
            ch_currents_2d[i],
        )

    return V_arr, gating_arr, ch_currents_2d
