"""Integration tests for patch_sim.analysis.inactivation against real HH sims.

Runs a two-pulse steady-state-inactivation voltage-clamp protocol through the
full simulation pipeline and verifies that the derived h∞ curve and Boltzmann
fit are physiologically plausible.  Unit tests with synthetic data live in
tests/unit/test_inactivation.py.
"""

import numpy as np

import patch_sim
from patch_sim.analysis.inactivation import compute_inactivation
from patch_sim.analysis.iv_curve import analyze_iv


def test_inactivation_curve_integration_hh_neuron(hh_model):
    """h∞ from a real HH simulation is monotone-decreasing with a valid fit.

    Runs a two-pulse inactivation protocol (long conditioning prepulse swept
    across voltages, then a fixed depolarizing test pulse), measures the
    test-pulse peak inward current per sweep via the I-V analysis, then derives
    the h∞ curve.  Verifies that availability lies in [0, 1], decreases with
    prepulse depolarization, and that the decreasing Boltzmann fit converges
    with a physiologically plausible half-inactivation voltage (the classic HH
    Na+ h∞ has V½ ≈ −60 mV).
    """
    pre_ms, stim_ms = 150.0, 15.0
    min_v, max_v, step_v = -120.0, -20.0, 10.0

    protocol_2d = patch_sim.build_voltage_protocol(
        "Inactivation",
        sampling_frequency=40_000,
        pre_stimulus_duration=pre_ms,
        stimulus_duration=stim_ms,
        post_stimulus_duration=5.0,
        min_stimulus=min_v,
        max_stimulus=max_v,
        stimulus_step=step_v,
        test_pulse_voltage=0.0,
    )
    n_steps = round((max_v - min_v) / step_v) + 1
    prepulses = list(np.linspace(min_v, max_v, n_steps))
    protocols = [protocol_2d[i] for i in range(protocol_2d.shape[0])]

    results = list(
        patch_sim.simulate_batch(hh_model, protocols, patch_sim.simulate_voltage_clamp)
    )
    time = results[0]["time"]
    currents = [r["Itotal"] for r in results]
    iv = analyze_iv(time, currents, prepulses, pre_ms, pre_ms + stim_ms)
    result = compute_inactivation(iv)

    assert len(result.points) == n_steps

    # Availability is normalized to [0, 1] (allow a hair of float slack).
    h = result.h_normalized_values
    for value in h:
        assert 0.0 <= value <= 1.0 + 1e-9

    # The most-hyperpolarized prepulse leaves channels available; the
    # most-depolarized inactivates them.  Allow tiny numerical non-monotonicity.
    assert h[0] >= h[-1]
    assert np.all(np.diff(h) <= 0.05)

    # The decreasing Boltzmann fit should converge with a plausible V½ and k.
    assert result.boltzmann.converged is True
    assert -85.0 <= result.boltzmann.v_half <= -40.0
    assert 0.0 < result.boltzmann.k < 25.0
