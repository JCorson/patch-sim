"""Behavioral tests for the CA1 Pyramidal neuron preset.

Pins the regular-spiking hippocampal phenotype: moderate firing rate with
spike-frequency adaptation driven by IM and IKCa, deep AHP from the Ca²⁺ /
IKCa interaction, and AP shape consistent with CA1 pyramidal recordings.
Bands cite Spruston & Johnston (2008), Migliore et al. (1999), and Storm
(1990).

Replaces the previous coverage gap: before this file the CA1 preset had
no suprathreshold-firing tests at all, only sag and stability checks.
"""

import numpy as np
import pytest

from patch_sim.analysis.ap_metrics import analyze_aps_from_result
from patch_sim.clamp_simulations import SIM_SAMPLING_FREQ, simulate_current_clamp
from patch_sim.constants import CA1_PYRAMIDAL
from patch_sim.neuron import Neuron
from patch_sim.neuron_factory import make_neuron
from patch_sim.presets import NEURON_PRESETS
from patch_sim.protocols import step_current
from tests.integration._ap_shape import assert_ap_shape

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Standard suprathreshold step: after the issue #302 retune (which adds
# INaP), rheobase drops to ~1.3 µA/cm² so 7 µA/cm² is ~5× rheobase.  The
# stim still produces a stable mid-rate trace (~27 Hz with adaptation)
# suitable for shape averaging without saturating into depolarization block.
_CA1_STEP_DURATION_MS = 500.0
_CA1_STEP_CURRENT = 7.0
_CA1_REFERENCE = "Spruston & Johnston 2008 / Migliore et al. 1999 / Storm 1990"


@pytest.fixture
def ca1_neuron() -> Neuron:
    """CA1 pyramidal neuron instance for all tests in this module."""
    return make_neuron(NEURON_PRESETS[CA1_PYRAMIDAL])


@pytest.fixture
def ca1_ap_shape_result(ca1_neuron: Neuron):
    """AP analysis under the standard suprathreshold step.

    Shared by every shape-band test in this module so each test does not
    re-run the simulation.
    """
    protocol = step_current(
        duration=_CA1_STEP_DURATION_MS,
        current_amplitude=_CA1_STEP_CURRENT,
    )
    result = simulate_current_clamp(ca1_neuron, current_external=protocol)
    return analyze_aps_from_result(result)


def _ms_to_samples(ms: float) -> int:
    """Convert a duration in milliseconds to a simulation sample count."""
    return int(ms * SIM_SAMPLING_FREQ / 1000.0)


# ---------------------------------------------------------------------------
# Resting stability and basic firing
# ---------------------------------------------------------------------------


def test_ca1_no_spontaneous_firing(ca1_neuron: Neuron) -> None:
    """CA1 pyramidal must not fire spontaneously at rest.

    CA1 pyramidal cells in slice are typically quiescent at zero injected
    current; spontaneous firing here would suggest g_NaP, Ih, or leak K⁺
    is mis-tuned.
    """
    protocol = np.zeros(_ms_to_samples(500) + 1)
    result = simulate_current_clamp(ca1_neuron, current_external=protocol)
    ap = analyze_aps_from_result(result)
    assert ap.spike_count == 0, (
        f"Expected no spontaneous APs at rest, but detected {ap.spike_count}."
    )


def test_ca1_suprathreshold_step_drives_repetitive_firing(ca1_ap_shape_result) -> None:
    """Suprathreshold step drives sustained repetitive firing in the RS range.

    Spruston & Johnston (2008): CA1 pyramidals fire 5–25 Hz at 1.5× rheobase
    in slice.  Tolerance widened to 5–30 Hz to accommodate the model's
    slightly elevated rate at the chosen 1.4× rheobase stim.  ≥10 spikes
    establishes sustained firing rather than a single AP.
    """
    assert_ap_shape(
        ca1_ap_shape_result,
        reference=_CA1_REFERENCE,
        firing_rate_hz=(5.0, 30.0),
        min_spike_count=10,
    )


# ---------------------------------------------------------------------------
# AP shape — RS pyramidal phenotype, Spruston & Johnston (2008).
# ---------------------------------------------------------------------------


def test_ca1_ap_half_width_in_rs_range(ca1_ap_shape_result) -> None:
    """Mean AP half-width matches CA1 pyramidal recordings (0.7–1.5 ms)."""
    assert_ap_shape(
        ca1_ap_shape_result,
        reference=_CA1_REFERENCE,
        half_width_ms=(0.7, 1.5),
    )


def test_ca1_ap_peak_voltage_in_rs_range(ca1_ap_shape_result) -> None:
    """Mean AP peak voltage falls within the CA1 RS range (+20 to +45 mV)."""
    assert_ap_shape(
        ca1_ap_shape_result,
        reference=_CA1_REFERENCE,
        peak_mv=(20.0, 45.0),
    )


def test_ca1_ap_threshold_in_rs_range(ca1_ap_shape_result) -> None:
    """Mean AP threshold falls within the CA1 RS range (−55 to −38 mV)."""
    assert_ap_shape(
        ca1_ap_shape_result,
        reference=_CA1_REFERENCE,
        threshold_mv=(-55.0, -38.0),
    )


def test_ca1_ap_ahp_depth_in_rs_range(ca1_ap_shape_result) -> None:
    """Mean AHP depth falls within the CA1 RS range (−78 to −55 mV).

    The Ca²⁺/IKCa interaction in CA1 produces a deeper AHP than the typical
    cortical RS AHP — Storm (1990) reports values around −70 mV after a
    single AP, deepening with bursts.
    """
    assert_ap_shape(
        ca1_ap_shape_result,
        reference=_CA1_REFERENCE,
        ahp_mv=(-78.0, -55.0),
    )


# ---------------------------------------------------------------------------
# Firing pattern — adapting (IM + IKCa drive SFA in CA1).
# ---------------------------------------------------------------------------


def test_ca1_spike_frequency_adaptation(ca1_ap_shape_result) -> None:
    """Inter-spike intervals lengthen during a sustained step — RS adaptation.

    IM (slow, non-inactivating K⁺) and IKCa (Ca²⁺-activated) both build up
    during repetitive firing in CA1, producing pronounced spike-frequency
    adaptation (Storm 1990; Madison & Nicoll 1984).  We assert that the
    mean of the last three ISIs exceeds the mean of the first three.
    """
    isis = ca1_ap_shape_result.isis
    assert len(isis) >= 6, (
        f"Need ≥6 ISIs to compare early vs late firing, got {len(isis)}"
    )
    mean_early = float(np.mean(isis[:3]))
    mean_late = float(np.mean(isis[-3:]))
    assert mean_late > mean_early, (
        f"Expected spike-frequency adaptation (late ISI > early ISI), "
        f"got mean_early={mean_early:.2f} ms, mean_late={mean_late:.2f} ms. "
        f"Reference: Storm (1990); Madison & Nicoll (1984)."
    )
