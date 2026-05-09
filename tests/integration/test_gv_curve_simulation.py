"""Integration tests for patch_sim.analysis.gv_curve against real HH simulations.

Runs a voltage-clamp multi-sweep protocol through the full simulation pipeline
and verifies that the derived G-V curve and Boltzmann fit are physiologically
plausible.  Unit tests with synthetic data live in tests/unit/test_gv_curve.py.
"""

import numpy as np

import patch_sim
import patch_sim.channels
from patch_sim.analysis.gv_curve import compute_gv
from patch_sim.analysis.iv_curve import analyze_iv


def test_gv_curve_integration_hh_neuron(hh_model):
    """g-V curve from a real HH simulation has valid G/Gmax and Boltzmann fit.

    Runs a voltage-clamp multi-sweep protocol, computes the I-V curve, then
    derives the g-V curve.  Verifies that G/Gmax values lie in [0, 1] and
    that the Boltzmann fit converges with a physiologically plausible V_half.
    """
    pre_ms, stim_ms = 5.0, 30.0
    min_v, max_v, step_v = -80.0, 60.0, 10.0

    protocol_2d = patch_sim.build_voltage_protocol(
        "Step",
        sampling_frequency=40_000,
        pre_stimulus_duration=pre_ms,
        stimulus_duration=stim_ms,
        post_stimulus_duration=5.0,
        holding_voltage=-65.0,
        min_stimulus=min_v,
        max_stimulus=max_v,
        stimulus_step=step_v,
    )
    n_steps = round((max_v - min_v) / step_v) + 1
    voltage_steps = list(np.linspace(min_v, max_v, n_steps))
    protocols = [protocol_2d[i] for i in range(protocol_2d.shape[0])]

    results = list(
        patch_sim.simulate_batch(hh_model, protocols, patch_sim.simulate_voltage_clamp)
    )
    time = results[0]["time"]
    currents = [r["Itotal"] for r in results]
    iv = analyze_iv(time, currents, voltage_steps, pre_ms, pre_ms + stim_ms)

    na_channel = next(
        ch
        for ch in hh_model.channels
        if isinstance(ch.reversal_spec, patch_sim.channels.NernstSpec)
        and ch.reversal_spec.species is patch_sim.channels.IonSpecies.SODIUM
    )
    e_na = na_channel.reversal_potential(hh_model)
    result = compute_gv(iv, reversal_potential=e_na)

    # Should have several valid points
    assert len(result.points) >= 3

    # G/Gmax must be in [0, 1]
    for pt in result.points:
        assert 0.0 <= pt.g_normalized <= 1.0 + 1e-9

    # Boltzmann fit should converge
    assert result.boltzmann.converged is True

    # V_half for HH Na activation is typically in the range -50 to 0 mV
    assert -70.0 <= result.boltzmann.v_half <= 20.0
