"""Integration tests for patch_sim.analysis.fi_curve against real HH simulations.

Runs batch current-clamp sweeps through the full simulation pipeline and verifies
F-I curve and rheobase properties against real HH dynamics.
Unit tests with synthetic traces live in tests/unit/test_fi_curve.py.
"""

import patch_sim
from patch_sim.analysis.fi_curve import estimate_rheobase

_STIM_START = 10.0
_STIM_DURATION = 50.0
_STIM_END = _STIM_START + _STIM_DURATION
_CURRENT_STEPS = [0.0, 5.0, 10.0, 20.0]


def _run_fi_sweeps(hh_model):
    """Run a batch of current-clamp sweeps and return (fi_result, time).

    Args:
        hh_model: Neuron instance to simulate.

    Returns:
        Tuple of (FIAnalysisResult, time array).
    """
    protocols = [
        patch_sim.step_current(
            duration=_STIM_START + _STIM_DURATION + 10.0,
            current_amplitude=amp,
            step_start=_STIM_START,
            step_duration=_STIM_DURATION,
        )
        for amp in _CURRENT_STEPS
    ]
    results = list(
        patch_sim.simulate_batch(hh_model, protocols, patch_sim.simulate_current_clamp)
    )
    voltages = [r["voltage"] for r in results]
    time = results[0]["time"]
    fi = patch_sim.analyze_fi(time, voltages, _CURRENT_STEPS, _STIM_START, _STIM_END)
    return fi, time


def test_fi_curve_integration_hh(hh_model):
    """F-I curve from real HH sweeps: firing rate increases with injected current.

    Runs a short batch of current clamp steps from subthreshold to suprathreshold
    and verifies that higher current steps produce higher mean firing rates.
    """
    fi, _ = _run_fi_sweeps(hh_model)

    # Subthreshold step should produce no spikes
    assert fi.points[0].spike_count == 0

    # Among spiking steps, all mean firing rates should be positive.
    spiking_rates = [
        p.mean_firing_rate for p in fi.points if p.mean_firing_rate is not None
    ]
    for rate in spiking_rates:
        assert rate > 0.0

    # Mean firing rates should be non-decreasing as injected current increases
    # (points are sorted by current_step).
    assert all(
        spiking_rates[i] <= spiking_rates[i + 1] for i in range(len(spiking_rates) - 1)
    )

    # The highest current step should have the most spikes
    spike_counts = [p.spike_count for p in fi.points]
    assert spike_counts[-1] >= max(spike_counts[:-1])


def test_estimate_rheobase_integration_hh(hh_model):
    """estimate_rheobase on real HH sweeps falls in the tested current range.

    Runs sweeps from clearly subthreshold to clearly suprathreshold and verifies
    that the estimated rheobase is positive and lies within the tested range.
    """
    fi, _ = _run_fi_sweeps(hh_model)
    rheobase = estimate_rheobase(fi)

    # At least one step must be suprathreshold for the HH model
    assert rheobase is not None

    # Rheobase must be positive (subthreshold at 0 µA/cm²)
    assert rheobase > 0.0

    # Rheobase must be within the tested range
    assert rheobase <= max(_CURRENT_STEPS)
