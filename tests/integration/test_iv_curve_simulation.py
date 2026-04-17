"""Integration tests for patch_sim.analysis.iv_curve against real HH simulations.

Runs a multi-sweep voltage-clamp protocol through the full simulation pipeline
and verifies physiologically correct I-V curve properties.
Unit tests with synthetic data live in tests/unit/test_iv_curve.py.
"""

import numpy as np

import patch_sim
from patch_sim.analysis.iv_curve import analyze_iv


def test_iv_curve_integration_hh_neuron(hh_model):
    """I-V curve from a real HH simulation has inward current at intermediate voltages.

    The HH model should produce a region of inward (negative) sodium current
    at intermediate depolarised voltage steps (e.g. -10 to +20 mV), and
    predominantly outward potassium current at strong depolarisations.
    """
    pre_ms, stim_ms, post_ms = 5.0, 30.0, 5.0
    min_v, max_v, step_v = -80.0, 60.0, 20.0
    protocol_2d = patch_sim.build_voltage_protocol(
        "Step",
        sampling_frequency=40_000,
        pre_stimulus_duration=pre_ms,
        stimulus_duration=stim_ms,
        post_stimulus_duration=post_ms,
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

    # Should have one entry per voltage step, sorted
    assert len(iv.points) == len(voltage_steps)
    assert iv.voltage_steps == sorted(voltage_steps)

    # At moderately depolarised steps, total peak should have an inward component
    # (the sodium current drives total inward in the -40 to +10 mV range for HH)
    any_inward = any(p < 0 for p in iv.peak_inward_currents)
    assert any_inward, "Expected at least one inward peak current in the HH I-V curve"
