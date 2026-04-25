"""Equilibrium analysis utilities for conductance-based neuron models.

Provides tools for computing the true zero-current resting potential of a
neuron, accounting for the actual Nernst-derived reversal potentials and
steady-state gating variables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scipy.optimize import brentq

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
        ch.compute_current(V, gating_state, neuron, ca_i=ca_i)
        for ch in neuron.all_channels
    )


def find_zero_current_voltage(
    neuron: "Neuron",
    v_min: float = -100.0,
    v_max: float = -20.0,
    tol: float = 1e-6,
    ca_i: float | None = None,
) -> float:
    """Find the membrane voltage where total ionic current is zero.

    Uses Brent's method to locate the voltage at which the sum of all ionic
    currents (with gating variables held at their steady-state values) is
    zero.  This is the true zero-current resting potential — the voltage
    the neuron will settle to when no external current is injected.

    Calcium dynamics are accounted for by using *ca_i* as the intracellular
    Ca²⁺ concentration when evaluating calcium-dependent gating variables.
    When *ca_i* is ``None`` (the default) the neuron's declared ``ca_rest``
    is used, consistent with how
    :func:`~patch_sim.simulate_current_clamp` historically initialised the
    state.  Pass an explicit value (e.g. from
    :func:`find_coupled_equilibrium`) when you need to evaluate the
    zero-current voltage at a specific Ca²⁺ concentration.

    Args:
        neuron: The conductance-based neuron model.
        v_min: Lower bound of the voltage search range in mV.  Must bracket
            the root together with *v_max*.
        v_max: Upper bound of the voltage search range in mV.
        tol: Convergence tolerance in mV.
        ca_i: Intracellular Ca²⁺ concentration in mM to use when evaluating
            calcium-dependent gating variables.  ``None`` defaults to
            ``neuron.calcium_dynamics.ca_rest`` (or 0.0 if no calcium
            dynamics are configured).

    Returns:
        Voltage in mV at which total ionic current is zero.

    Raises:
        ValueError: If *v_min* and *v_max* do not bracket a sign change in
            the ionic-current function, or if *v_min* >= *v_max*.
    """
    if v_min >= v_max:
        raise ValueError(f"v_min ({v_min}) must be less than v_max ({v_max}).")

    _ca_i: float = (
        ca_i
        if ca_i is not None
        else (
            neuron.calcium_dynamics.ca_rest
            if neuron.calcium_dynamics is not None
            else 0.0
        )
    )

    f_min = _total_ionic_current(neuron, v_min, _ca_i)
    f_max = _total_ionic_current(neuron, v_max, _ca_i)

    if f_min * f_max > 0:
        raise ValueError(
            f"v_min={v_min} mV (I={f_min:.4f}) and v_max={v_max} mV "
            f"(I={f_max:.4f}) do not bracket a root.  Adjust the search "
            "range or verify that the neuron has a stable resting point."
        )

    return brentq(
        lambda V: _total_ionic_current(neuron, V, _ca_i),
        v_min,
        v_max,
        xtol=tol,
    )


def find_coupled_equilibrium(
    neuron: "Neuron",
    v_min: float = -100.0,
    v_max: float = -20.0,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> tuple[float, float]:
    """Find the coupled (V_ss, ca_i_ss) equilibrium for a neuron.

    For neurons without calcium dynamics, equivalent to
    :func:`find_zero_current_voltage` with ``ca_i = 0`` — returns
    ``(V_ss, 0.0)``.

    For neurons *with* calcium dynamics, uses fixed-point iteration to
    simultaneously satisfy:

    * **Voltage equilibrium**: total ionic current is zero at *V_ss* (with
      gating variables at their steady-state values for that voltage and
      Ca²⁺ concentration).
    * **Calcium equilibrium**: the calcium ODE is at rest —
      ``ca_i_ss = ca_rest − α_ca · I_Ca(V_ss, ca_i_ss) · τ_ca``.

    Because the Ca²⁺ reversal potential depends on ``ca_i`` (dynamic E_Ca),
    the two equilibrium conditions are coupled: the resting ``ca_i`` shifts
    E_Ca, which shifts the zero-current voltage, which changes the Ca²⁺
    window current, which shifts ``ca_i`` again.  Iteration typically
    converges in fewer than ten steps for physiological parameters.

    The returned ``(V_ss, ca_i_ss)`` pair is the correct starting point for
    :func:`~patch_sim.simulate_current_clamp` so that the simulation does
    not drift when no external current is applied.

    Args:
        neuron: The conductance-based neuron model.
        v_min: Lower bound for the voltage root search in mV.
        v_max: Upper bound for the voltage root search in mV.
        tol: Convergence tolerance: iteration stops when both
            ``|V_new − V_old| < tol`` and ``|ca_i_new − ca_i_old| < 1e-10``.
        max_iter: Maximum number of fixed-point iterations before returning
            the best estimate found so far.

    Returns:
        ``(V_ss, ca_i_ss)`` — equilibrium membrane voltage in mV and
        equilibrium intracellular Ca²⁺ concentration in mM.

    Raises:
        ValueError: Propagated from :func:`find_zero_current_voltage` if the
            search range does not bracket a root.
    """
    if neuron.calcium_dynamics is None:
        return find_zero_current_voltage(neuron, v_min, v_max, tol), 0.0

    cd = neuron.calcium_dynamics
    ca_i: float = cd.ca_rest
    V_ss: float = find_zero_current_voltage(neuron, v_min, v_max, tol, ca_i=ca_i)

    for _ in range(max_iter):
        gating_state: dict[str, float] = {}
        for gv in neuron.all_gating_variables:
            a = gv.alpha(V_ss, ca_i)
            b = gv.beta(V_ss, ca_i)
            gating_state[gv.name] = a / (a + b)

        i_ca: float = sum(
            ch.compute_current(V_ss, gating_state, neuron, ca_i=ca_i)
            for ch in neuron.all_channels
            if ch.carries_calcium
        )
        ca_i_new = max(cd.ca_rest - cd.alpha_ca * i_ca * cd.tau_ca, 1e-9)
        V_ss_new = find_zero_current_voltage(neuron, v_min, v_max, tol, ca_i=ca_i_new)

        if abs(V_ss_new - V_ss) < tol and abs(ca_i_new - ca_i) < 1e-10:
            return V_ss_new, ca_i_new

        V_ss, ca_i = V_ss_new, ca_i_new

    return V_ss, ca_i
