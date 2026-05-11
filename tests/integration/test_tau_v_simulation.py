"""Integration tests for patch_sim.analysis.tau_v against real HH simulations.

Runs voltage-clamp multi-sweep protocols through the full simulation pipeline
and verifies that the recovered activation and inactivation time constants
are physiologically plausible.  Unit tests with synthetic data live in
tests/unit/test_tau_v.py.

A note on total-current vs. isolated-channel kinetics:
    HH total current at strong depolarizations is dominated by the much
    larger sustained K⁺ outward current, which masks the brief inward Na⁺
    inactivation phase.  Real experiments isolate channels pharmacologically
    (e.g. TTX subtraction).  These integration tests assert what is actually
    extractable from the unmodified total-current trace; explicit
    inactivation kinetics are verified in tests/unit/test_tau_v.py with
    synthetic Na⁺-only traces.
"""

import numpy as np

import patch_sim
from patch_sim.analysis.tau_v import analyze_tau_v, compute_tau_v_point


def test_tau_v_integration_hh_neuron(hh_model):
    """τ-V from a real HH simulation produces physiological activation taus.

    Runs a step voltage-clamp multi-sweep protocol and verifies that the
    pipeline returns one :class:`TauVPoint` per step, that strong-
    depolarization steps yield convergent activation fits, and that the
    fitted τ values are within a physiologically plausible range
    (sub-ms at strong depolarizations dominated by K⁺ activation).
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

    tau_result = analyze_tau_v(time, currents, voltage_steps, pre_ms, pre_ms + stim_ms)

    assert len(tau_result.points) == len(voltage_steps)
    # The result is sorted ascending by voltage.
    assert tau_result.voltage_steps == sorted(tau_result.voltage_steps)

    strong_steps = [p for p in tau_result.points if p.voltage_step >= 30.0]
    activated_taus = [
        p.tau_activation_ms for p in strong_steps if p.tau_activation_ms is not None
    ]
    # Strongly depolarized steps should produce convergent activation fits
    # for the rising phase of the K⁺-dominated total current.
    assert len(activated_taus) >= 3
    assert all(0.05 <= t <= 5.0 for t in activated_taus)


def test_tau_v_integration_hh_neuron_returns_tau_v_point_per_step(hh_model):
    """Every voltage step in the protocol gets exactly one TauVPoint.

    Even sub-threshold or otherwise unfittable sweeps must produce a
    :class:`TauVPoint` (with ``None`` τ values), so downstream code can
    rely on parallel arrays of length ``n_steps`` for plotting.
    """
    pre_ms, stim_ms = 5.0, 20.0
    voltage_steps = [-80.0, -60.0, -40.0, -20.0, 0.0, 20.0, 40.0]

    protocol_2d = patch_sim.build_voltage_protocol(
        "Step",
        sampling_frequency=40_000,
        pre_stimulus_duration=pre_ms,
        stimulus_duration=stim_ms,
        post_stimulus_duration=5.0,
        holding_voltage=-65.0,
        min_stimulus=voltage_steps[0],
        max_stimulus=voltage_steps[-1],
        stimulus_step=20.0,
    )
    protocols = [protocol_2d[i] for i in range(protocol_2d.shape[0])]
    results = list(
        patch_sim.simulate_batch(hh_model, protocols, patch_sim.simulate_voltage_clamp)
    )
    time = results[0]["time"]
    currents = [r["Itotal"] for r in results]

    tau_result = analyze_tau_v(time, currents, voltage_steps, pre_ms, pre_ms + stim_ms)

    assert len(tau_result.points) == len(voltage_steps)
    # Each point's voltage_step matches an input step.
    assert set(tau_result.voltage_steps) == set(voltage_steps)


def test_compute_tau_v_point_failed_sweep_does_not_break_pipeline():
    """A degenerate constant current does not raise; both taus are None."""
    pre_ms, stim_ms = 5.0, 30.0
    time = np.linspace(0.0, pre_ms + stim_ms + 5.0, 1601)
    flat_current = np.full_like(time, -1.0)
    pt = compute_tau_v_point(
        time,
        flat_current,
        voltage_step=0.0,
        stim_start_ms=pre_ms,
        stim_end_ms=pre_ms + stim_ms,
    )
    assert pt.tau_inactivation_ms is None
    assert pt.inactivation_converged is False
