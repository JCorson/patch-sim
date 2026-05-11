"""Integration tests for patch_sim.analysis.impedance against real simulations.

Runs a chirp current-clamp protocol through the full simulation pipeline and
verifies that the resulting impedance profile is structurally sound and
physiologically scaled.
Unit tests with synthetic signals live in tests/unit/test_impedance.py.
"""

import numpy as np

import patch_sim
from patch_sim.clamp_simulations import SIM_SAMPLING_FREQ
from patch_sim.presets import make_thalamic_relay

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


def test_thalamic_relay_has_higher_low_frequency_impedance_than_squid(
    hh_model,
) -> None:
    """A thalamic relay neuron shows much larger low-frequency |Z| than the squid axon.

    Reflects the far higher input resistance of a small thalamic relay cell
    compared with the large, low-resistance squid giant axon.
    """
    relay_profile, _ = _run_chirp_profile(make_thalamic_relay(), amplitude=1.0)
    squid_profile, _ = _run_chirp_profile(hh_model, amplitude=1.0)

    assert relay_profile is not None and squid_profile is not None
    assert relay_profile.magnitude[0] > 5.0 * squid_profile.magnitude[0]
