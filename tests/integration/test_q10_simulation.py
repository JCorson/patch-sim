"""Behavioral tests for Q10 temperature scaling of gating kinetics.

Verifies that:
- Q10 scaling speeds up gating kinetics (faster m-gate rise at higher T).
- Q10 scaling does not shift steady-state gating curves (same x_inf at same V).
- Q10 = 1.0 produces identical gating traces regardless of temperature.
- Under current clamp, Q10 scaling produces a faster AP upstroke.

Voltage-clamp tests fix membrane potential so gating dynamics are isolated
from temperature-dependent reversal-potential shifts.  The current-clamp test
deliberately uses the same T (and hence the same reversal potentials) while
varying Q10 to isolate the kinetic effect.
"""

import dataclasses

import numpy as np
import pytest

from patch_sim.clamp_simulations import (
    SIM_SAMPLING_FREQ,
    simulate_current_clamp,
    simulate_voltage_clamp,
)
from patch_sim.neuron import Neuron
from patch_sim.presets import make_squid_giant_axon
from patch_sim.protocols import step_current, step_voltage


def _hh(**overrides: object) -> Neuron:
    """HH52 squid neuron with the given field overrides applied."""
    return dataclasses.replace(make_squid_giant_axon(), **overrides)


# Protocol must be generated at the simulation sampling frequency so that each
# array element maps to exactly one simulation time step (dt = 1/SIM_SAMPLING_FREQ ms).
_SF = SIM_SAMPLING_FREQ

# Reference temperature (22°C) — HH52 experimental reference.
_T_REF = 295.15
# T at exactly 10 K above reference → phi = Q10^1 = Q10.
_T_HOT = _T_REF + 10.0
# Physiological temperature (37°C).
_T_PHYS = 310.15


def _ms_to_idx(t_ms: float) -> int:
    """Convert a time in milliseconds to a sample index.

    Args:
        t_ms: Time in milliseconds.

    Returns:
        Nearest sample index at the given time.
    """
    return round(t_ms * SIM_SAMPLING_FREQ / 1000.0)


def test_q10_scaling_speeds_up_m_gate() -> None:
    """Higher temperature produces a faster m-gate rise under voltage clamp.

    Two neurons are constructed: one at T == T_ref (phi = 1) and one at
    T = T_ref + 10 K (phi = Q10 = 3).  Both are held at -70 mV, then
    stepped to 0 mV at t = 5 ms.  At 0 mV, tau_m ≈ 0.34 ms (cold) vs
    0.11 ms (hot with phi = 3).  At 0.1 ms after the step the hotter neuron
    should be noticeably further along toward steady state.
    """
    # Hold at -70 mV, then step to 0 mV at t=5 ms so gating starts from
    # the low m_inf at rest.  Protocol must be at SIM_SAMPLING_FREQ so that
    # each array element maps to exactly one simulation time step.
    v_protocol = step_voltage(
        duration=6.0,
        voltage_amplitude=0.0,
        step_start=5.0,
        holding_voltage=-70.0,
        sampling_frequency=_SF,
    )

    cold = _hh(T=_T_REF, T_ref=_T_REF, Q10=3.0)
    hot = _hh(T=_T_HOT, T_ref=_T_REF, Q10=3.0)

    cold_result = simulate_voltage_clamp(cold, voltage_protocol=v_protocol)
    hot_result = simulate_voltage_clamp(hot, voltage_protocol=v_protocol)

    # Compare 0.1 ms after the step — cold has ~0.3 tau elapsed, hot ~0.9 tau.
    compare_idx = _ms_to_idx(5.1)
    cold_m = float(cold_result["m"][compare_idx])
    hot_m = float(hot_result["m"][compare_idx])

    assert hot_m > cold_m, (
        "Expected hotter neuron (phi=Q10=3) to have a larger m gate at 0.1 ms "
        f"after the step, but got hot={hot_m:.4f} cold={cold_m:.4f}"
    )


def test_q10_scaling_preserves_steady_state_gating() -> None:
    """Q10 scaling does not change the steady-state gating variable values.

    At any voltage, x_inf = alpha / (alpha + beta).  Because Q10 multiplies
    both alpha and beta equally, the factor cancels and x_inf is temperature-
    independent.  Two neurons clamped at the same voltage for long enough must
    converge to the same steady-state m, h, and n.
    """
    # Clamp at -30 mV long enough for all gates to reach steady state.
    v_protocol = step_voltage(
        duration=200.0,
        voltage_amplitude=-30.0,
        step_start=0.0,
        sampling_frequency=_SF,
    )

    cold = _hh(T=_T_REF, T_ref=_T_REF, Q10=3.0)
    hot = _hh(T=_T_PHYS, T_ref=_T_REF, Q10=3.0)

    cold_result = simulate_voltage_clamp(cold, voltage_protocol=v_protocol)
    hot_result = simulate_voltage_clamp(hot, voltage_protocol=v_protocol)

    for gate in ("m", "h", "n"):
        cold_ss = float(cold_result[gate][-1])
        hot_ss = float(hot_result[gate][-1])
        assert cold_ss == pytest.approx(hot_ss, abs=0.01), (
            f"Steady-state {gate} differs: cold={cold_ss:.4f} hot={hot_ss:.4f}"
        )


def test_q10_of_one_produces_no_temperature_effect() -> None:
    """Q10 = 1.0 makes gating dynamics temperature-independent.

    When Q10 is set to 1.0 the q10_factor is always 1.0, so simulations at
    different temperatures produce identical gating variable traces when clamped
    at the same voltage.
    """
    v_protocol = step_voltage(
        duration=10.0,
        voltage_amplitude=0.0,
        step_start=0.0,
        sampling_frequency=_SF,
    )

    at_ref = _hh(T=_T_REF, T_ref=_T_REF, Q10=1.0)
    at_phys = _hh(T=_T_PHYS, T_ref=_T_REF, Q10=1.0)

    result_ref = simulate_voltage_clamp(at_ref, voltage_protocol=v_protocol)
    result_phys = simulate_voltage_clamp(at_phys, voltage_protocol=v_protocol)

    for gate in ("m", "h", "n"):
        np.testing.assert_allclose(
            result_ref[gate],
            result_phys[gate],
            atol=1e-10,
            err_msg=f"Q10=1 should give identical {gate} traces at any temperature",
        )


def test_q10_current_clamp_earlier_ap_onset() -> None:
    """Q10 scaling causes the first action potential to fire sooner.

    Two neurons are constructed at the same temperature (so reversal potentials
    are identical) but with Q10=3 vs Q10=1.  A suprathreshold current step is
    applied.  Faster gating kinetics (Q10=3) lower the effective threshold
    crossing time, so the first AP should occur at an earlier sample.
    """
    # Both neurons run at T=_T_REF so reversal potentials are identical.
    # Slow: T_ref == T → phi = 1.0 (no scaling).
    # Fast: T_ref = T - 10 → phi = Q10^1 = 3.0 (3× faster kinetics).
    # K_out=7.8 mM matches HH52 squid seawater; DEFAULT_K_OUT is 4.0 mM (mammalian).
    slow = _hh(T=_T_REF, T_ref=_T_REF, Q10=3.0)
    fast = _hh(T=_T_REF, T_ref=_T_REF - 10.0, Q10=3.0)

    # Near-threshold current so the subthreshold buildup is long enough for the
    # kinetic difference to shift AP onset by at least a few samples.
    protocol = step_current(duration=50.0, current_amplitude=7.5)

    slow_result = simulate_current_clamp(slow, current_external=protocol)
    fast_result = simulate_current_clamp(fast, current_external=protocol)

    def _first_ap_sample(voltage: np.ndarray) -> int:
        """Return the sample index of the first sample exceeding 0 mV.

        Args:
            voltage: 1-D voltage trace in mV.

        Returns:
            Sample index of the first AP, or -1 if no AP detected.
        """
        indices = np.nonzero(voltage > 0.0)[0]
        return int(indices[0]) if len(indices) else -1

    slow_idx = _first_ap_sample(slow_result["voltage"])
    fast_idx = _first_ap_sample(fast_result["voltage"])

    assert slow_idx != -1, "Slow (phi=1) neuron did not fire an AP"
    assert fast_idx != -1, "Fast (phi=3) neuron did not fire an AP"
    assert fast_idx < slow_idx, (
        f"Expected phi=3 neuron to fire first, but phi=3 AP at sample {fast_idx}, "
        f"phi=1 AP at sample {slow_idx}"
    )
