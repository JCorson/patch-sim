"""Equilibrium analysis utilities for conductance-based neuron models.

Provides tools for computing the true zero-current resting potential of a
neuron, accounting for the actual Nernst-derived reversal potentials and
steady-state gating variables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .neuron import Neuron


def _total_ionic_current(neuron: "Neuron", V: float, ca_i: float) -> float:
    """Compute the total ionic current at voltage V with steady-state gating.

    For each gating variable, the steady-state value is computed as
    ``alpha(V) / (alpha(V) + beta(V))``.  The current is then the sum of
    ``ch.compute_current(V, gating_state, neuron)`` across all channels.

    Args:
        neuron: The conductance-based neuron model.
        V: Membrane voltage in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM used when evaluating
            calcium-dependent gating variables.

    Returns:
        Total ionic current in µA/cm².
    """
    gating_state: dict[str, float] = {}
    for gv in neuron.all_gating_variables:
        a = gv.alpha(V, ca_i)
        b = gv.beta(V, ca_i)
        gating_state[gv.name] = a / (a + b)
    return sum(
        ch.compute_current(V, gating_state, neuron) for ch in neuron.all_channels
    )


def find_zero_current_voltage(
    neuron: "Neuron",
    v_min: float = -100.0,
    v_max: float = -20.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Find the membrane voltage where total ionic current is zero.

    Uses bisection to locate the voltage at which the sum of all ionic
    currents (with gating variables held at their steady-state values) is
    zero.  This is the true zero-current resting potential — the voltage
    the neuron will settle to when no external current is injected.

    Calcium dynamics are accounted for by using the neuron's declared
    ``ca_rest`` as the intracellular Ca²⁺ concentration when evaluating
    calcium-dependent gating variables, consistent with how
    :func:`~patch_sim.simulate_current_clamp` initialises the state.

    Args:
        neuron: The conductance-based neuron model.
        v_min: Lower bound of the voltage search range in mV.  Must bracket
            the root together with *v_max*.
        v_max: Upper bound of the voltage search range in mV.
        tol: Convergence tolerance in mV.  The search stops once the
            bracketing interval is smaller than this value.
        max_iter: Maximum number of bisection iterations.

    Returns:
        Voltage in mV at which total ionic current is zero.

    Raises:
        ValueError: If *v_min* and *v_max* do not bracket a sign change in
            the ionic-current function, or if *v_min* >= *v_max*.
    """
    if v_min >= v_max:
        raise ValueError(f"v_min ({v_min}) must be less than v_max ({v_max}).")

    ca_i = (
        neuron.calcium_dynamics.ca_rest if neuron.calcium_dynamics is not None else 0.0
    )

    f_min = _total_ionic_current(neuron, v_min, ca_i)
    f_max = _total_ionic_current(neuron, v_max, ca_i)

    if f_min * f_max > 0:
        raise ValueError(
            f"v_min={v_min} mV (I={f_min:.4f}) and v_max={v_max} mV "
            f"(I={f_max:.4f}) do not bracket a root.  Adjust the search "
            "range or verify that the neuron has a stable resting point."
        )

    lo, hi = v_min, v_max
    f_lo = f_min

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if (hi - lo) < tol:
            return mid
        f_mid = _total_ionic_current(neuron, mid, ca_i)
        if f_mid == 0.0:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo = mid
            f_lo = f_mid

    return (lo + hi) / 2.0
