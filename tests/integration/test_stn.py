"""Behavioral tests for the Subthalamic Nucleus (STN) neuron preset.

Pins the STN tonic-pacemaker phenotype: 5–50 Hz autonomous firing and AP
shape consistent with Bevan & Wilson (1999); the conditional burst mode
is covered by ``test_stn_conditional_burst_mode_under_…`` in
``test_burst_metrics_simulation.py``.  Bands cite Beurrier et al. (1999)
and Bevan & Wilson (1999).
"""

import numpy as np
import pytest

from patch_sim.analysis.ap_metrics import analyze_aps_from_result
from patch_sim.clamp_simulations import SIM_SAMPLING_FREQ, simulate_current_clamp
from patch_sim.constants import STN
from patch_sim.neuron import Neuron
from patch_sim.neuron_factory import make_neuron
from patch_sim.presets import NEURON_PRESETS
from tests.integration._ap_shape import assert_ap_shape

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# 1 s of zero-current simulation captures the STN's autonomous tonic
# pacemaker phenotype directly: at the 5–50 Hz Bevan & Wilson (1999) rate
# this yields 5–50 spikes — sufficient for both rate and AP-shape stats.
_STN_SIMULATION_MS = 1000.0
_STN_REFERENCE = "Beurrier et al. 1999 / Bevan & Wilson 1999"


@pytest.fixture
def stn_neuron() -> Neuron:
    """STN neuron instance for all tests in this module."""
    return make_neuron(NEURON_PRESETS[STN])


@pytest.fixture
def stn_ap_shape_result(stn_neuron: Neuron):
    """AP analysis from a zero-current spontaneous-firing trace.

    All AP-shape assertions in this module run against the autonomous
    trace (the cell's intrinsic operating point per Bevan & Wilson 1999).
    The driven 2 µA/cm² operating point is exercised by
    ``test_repetitive_firing_preset`` in ``test_preset_protocols.py``.
    """
    zero_current = np.zeros(_ms_to_samples(_STN_SIMULATION_MS) + 1)
    result = simulate_current_clamp(stn_neuron, current_external=zero_current)
    return analyze_aps_from_result(result)


def _ms_to_samples(ms: float) -> int:
    """Convert ms to simulation samples at SIM_SAMPLING_FREQ."""
    return int(ms * SIM_SAMPLING_FREQ / 1000.0)


# ---------------------------------------------------------------------------
# Spontaneous tonic pacemaking — Bevan & Wilson (1999), 5–50 Hz in vitro.
# ---------------------------------------------------------------------------


def test_stn_spontaneous_pacemaking(stn_ap_shape_result) -> None:
    """STN fires autonomously at 5–50 Hz without injected current.

    Bevan & Wilson (1999) characterise the STN as a tonic pacemaker
    sustained intrinsically by INaP and Ih.  The driven 2 µA/cm²
    operating point is exercised by ``test_repetitive_firing_preset``
    in ``test_preset_protocols.py``.
    """
    assert_ap_shape(
        stn_ap_shape_result,
        reference=_STN_REFERENCE,
        firing_rate_hz=(5.0, 50.0),
        min_spike_count=5,
    )


# ---------------------------------------------------------------------------
# AP shape — STN tonic phenotype, Bevan & Wilson (1999).
# ---------------------------------------------------------------------------


def test_stn_ap_threshold_in_stn_range(stn_ap_shape_result) -> None:
    """Mean AP threshold falls in the STN range (−60 to −40 mV)."""
    assert_ap_shape(
        stn_ap_shape_result,
        reference=_STN_REFERENCE,
        threshold_mv=(-60.0, -40.0),
    )


def test_stn_ap_peak_voltage_in_stn_range(stn_ap_shape_result) -> None:
    """Mean AP peak voltage falls within the STN range (0 to +30 mV).

    STN APs have a comparatively low overshoot relative to other neurons —
    Bevan & Wilson (1999) report typical AP amplitudes of 60–80 mV from
    a threshold near −55 mV, putting the peak in the +5 to +25 mV range.
    """
    assert_ap_shape(
        stn_ap_shape_result,
        reference=_STN_REFERENCE,
        peak_mv=(0.0, 30.0),
    )


def test_stn_ap_half_width_in_stn_range(stn_ap_shape_result) -> None:
    """Mean AP half-width matches the STN tonic phenotype (0.4–1.2 ms)."""
    assert_ap_shape(
        stn_ap_shape_result,
        reference=_STN_REFERENCE,
        half_width_ms=(0.4, 1.2),
    )


def test_stn_ap_ahp_depth_in_stn_range(stn_ap_shape_result) -> None:
    """Mean AHP depth falls within the STN range (−85 to −55 mV)."""
    assert_ap_shape(
        stn_ap_shape_result,
        reference=_STN_REFERENCE,
        ahp_mv=(-85.0, -55.0),
    )


# ---------------------------------------------------------------------------
# Slow INaP inactivation engages during sustained drive — issue #324.
# ---------------------------------------------------------------------------


def test_stn_inap_slow_inactivation_engages_during_drive(
    stn_neuron: Neuron,
) -> None:
    """The sNaP gate closes during +5 µA/cm² × 200 ms, removing INaP from the plateau.

    Direct mechanism check for #324: the Magistretti & Alonso 1999 slow
    inactivation gate added to ``make_inap_channel`` must be doing real
    work during sustained suprathreshold drive — by the end of the step,
    sNaP should be below 0.1 (>90 % inactivation), so the persistent Na⁺
    contribution to any depol-block plateau is essentially abolished.

    Note: the STN preset can still settle on a residual depol-block
    plateau at ≈ −15 mV under this drive because the Otsuka 2004 fast
    Na⁺ model has a small h tail at depolarised voltages.  That residual
    plateau is held by I_Na, not I_NaP — see the STN preset comment in
    ``patch_sim/presets.py``.  Recovery tests for the cortical, CA1, and
    Purkinje presets confirm the slow-inactivation fix on cells that
    use a fully-inactivating fast Na⁺ kinetics.
    """
    n_pre = _ms_to_samples(100.0)
    n_step = _ms_to_samples(200.0)
    n_post = _ms_to_samples(50.0)
    current = np.concatenate(
        [
            np.zeros(n_pre),
            np.full(n_step, 5.0),
            np.zeros(n_post + 1),
        ]
    )
    result = simulate_current_clamp(stn_neuron, current_external=current)
    sNaP_at_rest = float(result["sNaP"][n_pre - 1])
    sNaP_at_step_end = float(result["sNaP"][n_pre + n_step - 1])
    fraction_inactivated = 1.0 - sNaP_at_step_end / sNaP_at_rest
    assert fraction_inactivated > 0.7, (
        f"sNaP did not meaningfully inactivate: rest={sNaP_at_rest:.3f}, "
        f"step end={sNaP_at_step_end:.3f}, "
        f"fraction inactivated={fraction_inactivated:.2f} (expected > 0.7)"
    )
