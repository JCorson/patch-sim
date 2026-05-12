"""Integration tests for patch_sim.analysis.impedance against real simulations.

Runs a chirp current-clamp protocol through the full simulation pipeline and
verifies that the resulting impedance profile is structurally sound,
physiologically scaled, and only produced for subthreshold responses.
Unit tests with synthetic signals live in tests/unit/test_impedance.py.
"""

import numpy as np
import pytest

import patch_sim
from patch_sim.clamp_simulations import SIM_SAMPLING_FREQ
from patch_sim.constants import FREQUENCY_RESPONSE
from patch_sim.presets import (
    NEURON_PRESETS,
    make_ca1_pyramidal,
)
from patch_sim.presets.protocols import build_protocol_from_preset

# 1000 ms matches the tuned FREQUENCY_RESPONSE preset and gives 1 Hz frequency
# resolution — fine enough to bracket the −3 dB width of CA1's few-Hz Ih
# resonance (a 500 ms / 2 Hz-resolution chirp puts that peak too close to the
# 1 Hz band edge to bracket its low-side half-power crossing).
_DURATION_MS = 1000.0
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
    """A hippocampal CA1 cell (with Ih) shows a genuine low-frequency |Z| resonance.

    The CA1 pyramidal preset has an Ih (HCN) conductance, so a subthreshold
    chirp produces an interior peak in |Z(f)| in the few-Hz range that is both
    prominent and sharp enough to bracket a −3 dB width — so ``resonance_frequency``,
    ``peak_impedance`` *and* ``quality_factor`` are all populated.  Its
    low-frequency impedance is also much larger than the low-resistance squid axon.
    """
    ca1_profile, ca1_voltage = _run_chirp_profile(make_ca1_pyramidal(), amplitude=1.0)
    squid_profile, _ = _run_chirp_profile(hh_model, amplitude=1.0)

    assert ca1_voltage.max() < -20.0  # stayed subthreshold
    assert ca1_profile is not None and squid_profile is not None
    assert ca1_profile.resonance_frequency is not None
    assert 1.0 < ca1_profile.resonance_frequency < 30.0
    assert ca1_profile.peak_impedance is not None
    assert ca1_profile.quality_factor is not None
    assert ca1_profile.quality_factor > 0.0
    mag = np.asarray(ca1_profile.magnitude)
    assert ca1_profile.peak_impedance > mag[0]
    assert ca1_profile.peak_impedance > mag[-1]
    assert ca1_profile.magnitude[0] > 5.0 * squid_profile.magnitude[0]


# Integration-level coverage for the "no usable spike-free segment → None"
# branch lives at the unit level (tests/unit/test_impedance.py
# ``test_returns_none_when_spikes_throughout``) because the size/duration
# combination needed to wipe out every 50 ms gap is sensitive to preset
# excitability and gating kinetics, making a stable integration test brittle.
# The parametrized test below gives the cleaner positive-direction coverage —
# every shipped preset yields a profile out of the box.


@pytest.mark.parametrize("name", list(NEURON_PRESETS.keys()))
def test_frequency_response_preset_yields_profile_for_every_neuron(name: str) -> None:
    """``FREQUENCY_RESPONSE`` + per-neuron overrides recover a profile for every preset.

    Regression guard for the tuned global default plus the per-pacemaker
    holding-current overrides: building the chirp via
    ``build_protocol_from_preset`` (same code path the UI uses) and running it
    through ``analyze_impedance`` must yield a non-``None`` profile with finite,
    positive magnitudes for every preset.
    """
    neuron = NEURON_PRESETS[name]()
    stim = build_protocol_from_preset(
        FREQUENCY_RESPONSE,
        neuron_preset=name,
        sampling_frequency=SIM_SAMPLING_FREQ,
    )[0]
    result = patch_sim.simulate_current_clamp(neuron, stim)
    time = np.asarray(result["time"])
    voltage = np.asarray(result["voltage"])
    duration = float(time[-1])
    profile = patch_sim.analyze_impedance(
        time, voltage, stim, 0.0, duration, _F_START, _F_END, area_cm2=neuron.area_cm2
    )

    assert profile is not None, f"{name}: analyze_impedance returned None"
    mag = np.asarray(profile.magnitude)
    assert mag.size >= 8
    assert np.all(np.isfinite(mag)) and np.all(mag > 0.0)
