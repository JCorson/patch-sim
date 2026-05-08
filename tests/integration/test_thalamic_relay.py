"""Behavioral tests for the Thalamic Relay (TC) neuron preset.

Pins the TC tonic-firing AP shape under depolarising step and the
intra-burst AP shape during the LTS rebound burst.  The burst structure
itself (3–7 spikes, 200–500 Hz) is already pinned by
``test_thalamic_relay_step_release_produces_multi_spike_lts_burst`` in
``test_burst_metrics_simulation.py``; this module adds the per-spike
shape assertions that the burst-structure test does not cover.

Bands cite McCormick & Huguenard (1992).
"""

import numpy as np
import pytest

from patch_sim.analysis.ap_metrics import analyze_aps, analyze_aps_from_result
from patch_sim.analysis.burst_metrics import analyze_bursts_from_result
from patch_sim.clamp_simulations import SIM_SAMPLING_FREQ, simulate_current_clamp
from patch_sim.neuron import Neuron
from patch_sim.presets import make_thalamic_relay
from patch_sim.protocols import step_current
from tests.integration._ap_shape import assert_ap_shape

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Tonic-mode standard step: 1 µA/cm² is just above the rheobase for this
# preset and gives ~9 Hz repetitive firing — within the in-vitro tonic
# range reported for TC cells (McCormick & Huguenard 1992).
_TC_TONIC_STEP_DURATION_MS = 200.0
_TC_TONIC_STEP_CURRENT = 1.0
_TC_REFERENCE = "McCormick & Huguenard 1992"


@pytest.fixture
def tc_neuron() -> Neuron:
    """Thalamic relay neuron instance for all tests in this module."""
    return make_thalamic_relay()


@pytest.fixture
def tc_tonic_ap_result(tc_neuron: Neuron):
    """AP analysis under a 1 µA/cm² depolarising step (tonic mode)."""
    protocol = step_current(
        duration=_TC_TONIC_STEP_DURATION_MS,
        current_amplitude=_TC_TONIC_STEP_CURRENT,
    )
    result = simulate_current_clamp(tc_neuron, current_external=protocol)
    return analyze_aps_from_result(result)


@pytest.fixture
def tc_intra_burst_ap_result(tc_neuron: Neuron):
    """AP analysis of just the LTS rebound burst window.

    Mirrors the protocol used by
    ``test_thalamic_relay_step_release_produces_multi_spike_lts_burst``
    in ``test_burst_metrics_simulation.py`` so the AP-shape assertions
    apply to the same burst that the structural assertions cover.
    """
    pre = 50.0
    stim = 300.0
    post = 200.0
    protocol = step_current(
        duration=pre + stim + post,
        current_amplitude=-10.0,
        step_start=pre,
        step_duration=stim,
    )
    result = simulate_current_clamp(tc_neuron, current_external=protocol)
    burst_analysis = analyze_bursts_from_result(result, isi_threshold_ms=50.0)
    assert burst_analysis.burst_count >= 1, "LTS rebound burst not detected"
    burst = burst_analysis.bursts[0]

    time = result["time"]
    voltage = result["voltage"]
    # Pad the burst window slightly to capture the AHP after the last spike
    # in the burst.
    pad_ms = 5.0
    mask = (time >= burst.start_time - pad_ms) & (time <= burst.end_time + pad_ms)
    return analyze_aps(time[mask], voltage[mask])


def _ms_to_samples(ms: float) -> int:
    """Convert ms to simulation samples at SIM_SAMPLING_FREQ."""
    return int(ms * SIM_SAMPLING_FREQ / 1000.0)


# ---------------------------------------------------------------------------
# Resting state — TC cells are quiescent at rest in slice (no pacemaker
# mechanism; ICaT is inactivated near v_rest and Ih is too weak alone).
# ---------------------------------------------------------------------------


def test_tc_quiescent_at_rest(tc_neuron: Neuron) -> None:
    """TC fires at most a single transient spike with zero injected current.

    McCormick & Huguenard (1992) describe TC cells in slice as quiescent
    at rest; spontaneous tonic firing is absent because ICaT is inactivated
    near v_rest and Ih alone cannot drive repetitive firing.  We allow up
    to one transient spike from initial-condition relaxation.
    """
    zero_current = np.zeros(_ms_to_samples(1000) + 1)
    result = simulate_current_clamp(tc_neuron, current_external=zero_current)
    ap = analyze_aps_from_result(result)
    assert ap.spike_count <= 1, (
        f"Expected ≤1 transient spike at rest, but detected {ap.spike_count} "
        f"— TC should be quiescent (McCormick & Huguenard 1992)."
    )


# ---------------------------------------------------------------------------
# Tonic-mode AP shape — McCormick & Huguenard (1992).
# ---------------------------------------------------------------------------


def test_tc_tonic_firing_rate_in_tc_range(tc_tonic_ap_result) -> None:
    """Tonic firing rate at 1 µA/cm² falls in the TC range (5–50 Hz)."""
    assert_ap_shape(
        tc_tonic_ap_result,
        reference=_TC_REFERENCE,
        firing_rate_hz=(5.0, 50.0),
        min_spike_count=2,
    )


def test_tc_ap_half_width_in_tc_tonic_range(tc_tonic_ap_result) -> None:
    """Mean tonic AP half-width falls in the TC range (0.5–1.5 ms)."""
    assert_ap_shape(
        tc_tonic_ap_result,
        reference=_TC_REFERENCE,
        half_width_ms=(0.5, 1.5),
    )


def test_tc_ap_threshold_in_tc_tonic_range(tc_tonic_ap_result) -> None:
    """Mean tonic AP threshold falls in the TC range (−55 to −40 mV)."""
    assert_ap_shape(
        tc_tonic_ap_result,
        reference=_TC_REFERENCE,
        threshold_mv=(-55.0, -40.0),
    )


def test_tc_ap_peak_voltage_in_tc_tonic_range(tc_tonic_ap_result) -> None:
    """Mean tonic AP peak voltage falls within the TC range (+10 to +40 mV)."""
    assert_ap_shape(
        tc_tonic_ap_result,
        reference=_TC_REFERENCE,
        peak_mv=(10.0, 40.0),
    )


def test_tc_ap_ahp_depth_in_tc_tonic_range(tc_tonic_ap_result) -> None:
    """Mean tonic AHP depth falls within the TC range (−75 to −55 mV)."""
    assert_ap_shape(
        tc_tonic_ap_result,
        reference=_TC_REFERENCE,
        ahp_mv=(-75.0, -55.0),
    )


# ---------------------------------------------------------------------------
# Intra-burst AP shape during the LTS rebound burst.
# ---------------------------------------------------------------------------


def test_tc_intra_burst_ap_threshold_in_tc_range(tc_intra_burst_ap_result) -> None:
    """Intra-burst AP threshold falls in the TC range (−55 to −40 mV).

    Each Na⁺ spike during the LTS plateau still triggers from the same
    Pospischil/MH92 voltage threshold; the LTS just provides the
    depolarising drive that brings V there repeatedly.
    """
    assert_ap_shape(
        tc_intra_burst_ap_result,
        reference=_TC_REFERENCE,
        threshold_mv=(-55.0, -40.0),
    )


def test_tc_intra_burst_ap_peak_voltage_within_burst_range(
    tc_intra_burst_ap_result,
) -> None:
    """Intra-burst peak voltage stays positive (Na⁺ spikes survive on plateau).

    During the LTS plateau, partial Na⁺ inactivation reduces successive
    spike heights but they should still overshoot.  +10 mV lower bound is
    a sanity threshold; full RS-style overshoot is not expected mid-burst.
    """
    assert_ap_shape(
        tc_intra_burst_ap_result,
        reference=_TC_REFERENCE,
        peak_mv=(10.0, 50.0),
    )


def test_tc_intra_burst_ap_half_width_in_burst_range(
    tc_intra_burst_ap_result,
) -> None:
    """Intra-burst AP half-width falls in the narrow burst range (0.2–0.6 ms).

    Burst APs are narrower than tonic APs because partial Na⁺ inactivation
    reduces the upstroke amplitude relative to threshold, shrinking the
    half-amplitude window.
    """
    assert_ap_shape(
        tc_intra_burst_ap_result,
        reference=_TC_REFERENCE,
        half_width_ms=(0.2, 0.6),
    )
