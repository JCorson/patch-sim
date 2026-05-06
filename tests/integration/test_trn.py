"""Behavioral tests for the Thalamic Reticular Nucleus (TRN) neuron preset.

Pins the TRN spontaneous tonic AP shape (~3 Hz pacemaking) and the
intra-burst AP shape during the HP92 rebound burst.  The burst structure
(5–15 spikes, 200–600 Hz) is already pinned by
``test_trn_step_release_produces_hp92_rebound_burst`` in
``test_burst_metrics_simulation.py``; this module adds the per-spike
shape assertions that the burst-structure test does not cover.

Bands cite Huguenard & Prince (1992) and Bal & McCormick (1993).
"""

import numpy as np
import pytest

from patch_sim.analysis.ap_metrics import analyze_aps, analyze_aps_from_result
from patch_sim.analysis.burst_metrics import analyze_bursts_from_result
from patch_sim.clamp_simulations import SIM_SAMPLING_FREQ, simulate_current_clamp
from patch_sim.constants import TRN
from patch_sim.neuron import Neuron
from patch_sim.neuron_factory import make_neuron
from patch_sim.presets import NEURON_PRESETS
from patch_sim.protocols import step_current
from tests.integration._ap_shape import assert_ap_shape

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_TRN_SPONTANEOUS_DURATION_MS = 2000.0
_TRN_REFERENCE = "Huguenard & Prince 1992 / Bal & McCormick 1993"


@pytest.fixture
def trn_neuron() -> Neuron:
    """TRN neuron instance for all tests in this module."""
    return make_neuron(NEURON_PRESETS[TRN])


@pytest.fixture
def trn_spontaneous_ap_result(trn_neuron: Neuron):
    """AP analysis under spontaneous (zero-current) tonic firing.

    The TRN preset is documented as a ~3 Hz pacemaker driven by Ih and
    the elevated ICaT conductance (Huguenard & Prince 1992;
    Bal & McCormick 1993).
    """
    duration_samples = int(_TRN_SPONTANEOUS_DURATION_MS * SIM_SAMPLING_FREQ / 1000.0)
    zero_current = np.zeros(duration_samples + 1)
    result = simulate_current_clamp(trn_neuron, current_external=zero_current)
    return analyze_aps_from_result(result)


@pytest.fixture
def trn_intra_burst_ap_result(trn_neuron: Neuron):
    """AP analysis of just the HP92 rebound burst window.

    Mirrors the protocol used by
    ``test_trn_step_release_produces_hp92_rebound_burst`` in
    ``test_burst_metrics_simulation.py`` so the AP-shape assertions apply
    to the same burst that the structural assertions cover.
    """
    pre = 200.0
    stim = 500.0
    post = 200.0
    protocol = step_current(
        duration=pre + stim + post,
        current_amplitude=-4.0,
        step_start=pre,
        step_duration=stim,
    )
    result = simulate_current_clamp(trn_neuron, current_external=protocol)
    burst_analysis = analyze_bursts_from_result(result)
    assert burst_analysis.burst_count >= 1, "HP92 rebound burst not detected"
    burst = burst_analysis.bursts[0]

    time = result["time"]
    voltage = result["voltage"]
    pad_ms = 5.0
    mask = (time >= burst.start_time - pad_ms) & (time <= burst.end_time + pad_ms)
    return analyze_aps(time[mask], voltage[mask])


# ---------------------------------------------------------------------------
# Spontaneous tonic firing — Huguenard & Prince (1992); Bal & McCormick
# (1993). The preset is configured for ~3 Hz; we accept 1–15 Hz to allow
# some calibration slack while still flagging an over- or under-firing
# regime as biology drift.
# ---------------------------------------------------------------------------


def test_trn_spontaneous_tonic_pacing(trn_spontaneous_ap_result) -> None:
    """TRN fires autonomously at 1–15 Hz without injected current.

    The preset comment documents a target rate of ~3 Hz; the literature
    range covers spontaneous tonic firing observed in TRN slice work
    (Huguenard & Prince 1992; Bal & McCormick 1993).
    """
    assert_ap_shape(
        trn_spontaneous_ap_result,
        reference=_TRN_REFERENCE,
        firing_rate_hz=(1.0, 15.0),
        min_spike_count=2,
    )


# ---------------------------------------------------------------------------
# Spontaneous tonic AP shape.
# ---------------------------------------------------------------------------


def test_trn_ap_half_width_in_trn_tonic_range(trn_spontaneous_ap_result) -> None:
    """Mean tonic AP half-width falls in the TRN range (0.4–1.2 ms)."""
    assert_ap_shape(
        trn_spontaneous_ap_result,
        reference=_TRN_REFERENCE,
        half_width_ms=(0.4, 1.2),
    )


def test_trn_ap_threshold_in_trn_tonic_range(trn_spontaneous_ap_result) -> None:
    """Mean tonic AP threshold falls in the TRN range (−65 to −40 mV)."""
    assert_ap_shape(
        trn_spontaneous_ap_result,
        reference=_TRN_REFERENCE,
        threshold_mv=(-65.0, -40.0),
    )


def test_trn_ap_peak_voltage_in_trn_tonic_range(
    trn_spontaneous_ap_result,
) -> None:
    """Mean tonic AP peak voltage falls within the TRN range (+10 to +40 mV)."""
    assert_ap_shape(
        trn_spontaneous_ap_result,
        reference=_TRN_REFERENCE,
        peak_mv=(10.0, 40.0),
    )


def test_trn_ap_ahp_depth_in_trn_tonic_range(trn_spontaneous_ap_result) -> None:
    """Mean tonic AHP depth falls within the TRN range (−75 to −55 mV)."""
    assert_ap_shape(
        trn_spontaneous_ap_result,
        reference=_TRN_REFERENCE,
        ahp_mv=(-75.0, -55.0),
    )


# ---------------------------------------------------------------------------
# Intra-burst AP shape during HP92 rebound burst.
# ---------------------------------------------------------------------------


def test_trn_intra_burst_ap_threshold_in_trn_range(
    trn_intra_burst_ap_result,
) -> None:
    """Intra-burst AP threshold falls in the TRN range (−65 to −40 mV)."""
    assert_ap_shape(
        trn_intra_burst_ap_result,
        reference=_TRN_REFERENCE,
        threshold_mv=(-65.0, -40.0),
    )


def test_trn_intra_burst_ap_peak_voltage_within_burst_range(
    trn_intra_burst_ap_result,
) -> None:
    """Intra-burst peak voltage stays positive on the LTS plateau.

    Partial Na⁺ inactivation across the plateau reduces successive spike
    heights but Na⁺ spikes should still overshoot (+10 mV lower bound).
    """
    assert_ap_shape(
        trn_intra_burst_ap_result,
        reference=_TRN_REFERENCE,
        peak_mv=(10.0, 50.0),
    )


def test_trn_intra_burst_ap_half_width_in_burst_range(
    trn_intra_burst_ap_result,
) -> None:
    """Intra-burst AP half-width falls in the narrow burst range (0.2–0.6 ms)."""
    assert_ap_shape(
        trn_intra_burst_ap_result,
        reference=_TRN_REFERENCE,
        half_width_ms=(0.2, 0.6),
    )
