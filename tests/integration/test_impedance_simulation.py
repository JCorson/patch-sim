"""Integration tests for patch_sim.analysis.impedance against real simulations.

Runs a chirp current-clamp protocol through the full simulation pipeline and
verifies that the resulting impedance profile is structurally sound,
physiologically scaled, and only produced for subthreshold responses.
Unit tests with synthetic signals live in tests/unit/test_impedance.py.
"""

import numpy as np

import patch_sim
from patch_sim.clamp_simulations import SIM_SAMPLING_FREQ
from patch_sim.presets import make_ca1_pyramidal, make_thalamic_relay

_DURATION_MS = 500.0
_F_START = 1.0
_F_END = 100.0


def _run_chirp_profile(neuron, amplitude: float):
    """Run a chirp current-clamp sweep and return its impedance profile + voltage.

    Args:
        neuron: Neuron instance to simulate.
        amplitude: Chirp amplitude in µA/cm².

    Returns:
        Tuple of (ImpedanceProfile or None, voltage trace array in mV).
    """
    stimulus = patch_sim.chirp_current(
        duration=_DURATION_MS,
        dc_offset=0.0,
        amplitude=amplitude,
        start_frequency=_F_START,
        end_frequency=_F_END,
        sampling_frequency=SIM_SAMPLING_FREQ,
    )
    result = patch_sim.simulate_current_clamp(neuron, stimulus)
    time = np.asarray(result["time"])
    voltage = np.asarray(result["voltage"])
    profile = patch_sim.analyze_impedance(
        time, voltage, stimulus, 0.0, _DURATION_MS, _F_START, _F_END
    )
    return profile, voltage


def test_chirp_pipeline_produces_sane_impedance_profile(hh_model) -> None:
    """A subthreshold chirp on the squid axon yields a well-formed impedance profile.

    Checks the structural invariants of the returned profile: ascending
    in-band frequency axis, finite positive magnitudes, bounded phase, and
    consistent array lengths.
    """
    profile, voltage = _run_chirp_profile(hh_model, amplitude=1.0)

    assert voltage.max() < -20.0  # stayed subthreshold
    assert profile is not None
    freqs = np.asarray(profile.frequencies)
    mag = np.asarray(profile.magnitude)
    phase = np.asarray(profile.phase)
    assert len(freqs) == len(mag) == len(phase)
    assert len(freqs) >= 8
    assert np.all(np.diff(freqs) > 0.0)
    assert freqs.min() >= _F_START
    assert freqs.max() <= _F_END
    assert np.all(np.isfinite(mag)) and np.all(mag > 0.0)
    assert np.all(np.abs(phase) <= 180.0)


def test_ca1_chirp_shows_resonance_and_high_impedance(hh_model) -> None:
    """A hippocampal CA1 cell (with Ih) shows a low-frequency |Z| resonance.

    The CA1 pyramidal preset has an Ih (HCN) conductance, so a subthreshold
    chirp produces an interior peak in |Z(f)| in the few-Hz range, and its
    low-frequency impedance is much larger than the low-resistance squid axon.
    """
    ca1_profile, ca1_voltage = _run_chirp_profile(make_ca1_pyramidal(), amplitude=1.0)
    squid_profile, _ = _run_chirp_profile(hh_model, amplitude=1.0)

    assert ca1_voltage.max() < -20.0  # stayed subthreshold
    assert ca1_profile is not None and squid_profile is not None
    assert ca1_profile.resonance_frequency is not None
    assert 1.0 < ca1_profile.resonance_frequency < 30.0
    assert ca1_profile.peak_impedance is not None
    mag = np.asarray(ca1_profile.magnitude)
    assert ca1_profile.peak_impedance > mag[0]
    assert ca1_profile.peak_impedance > mag[-1]
    assert ca1_profile.magnitude[0] > 5.0 * squid_profile.magnitude[0]


def test_suprathreshold_chirp_returns_none() -> None:
    """A chirp that drives spiking yields no impedance profile (linear regime only).

    A thalamic relay neuron fires low-threshold Ca²⁺ spikes in response to even
    a small chirp around rest, so ``analyze_impedance`` declines to report a
    profile.
    """
    profile, voltage = _run_chirp_profile(make_thalamic_relay(), amplitude=1.0)

    assert voltage.max() > -20.0  # the cell spiked
    assert profile is None
