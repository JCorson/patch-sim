"""Behavioral tests for the Dopaminergic (SNc) neuron preset.

Pins the SNc DA pacemaker phenotype: slow autonomous firing (1–5 Hz)
sustained by Ih, broad APs, deep AHP, post-inhibitory rebound.  Bands
cite Grace & Bunney (1984) and Komendantov et al. (2004).  Several
metrics — most notably the absence of spontaneous pacemaking — currently
fall outside literature ranges and are marked ``xfail`` pending biology
fixes.

Replaces the previous coverage gap: before this file the DA preset had
no suprathreshold-firing test and no spontaneous-firing test, only sag
and rebound.
"""

import numpy as np
import pytest

from patch_sim.analysis.ap_metrics import analyze_aps_from_result
from patch_sim.clamp_simulations import SIM_SAMPLING_FREQ, simulate_current_clamp
from patch_sim.constants import DOPAMINERGIC
from patch_sim.neuron import Neuron
from patch_sim.neuron_factory import make_neuron
from patch_sim.presets import NEURON_PRESETS
from patch_sim.protocols import step_current
from tests.integration._ap_shape import assert_ap_shape

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Standard suprathreshold step: 1.5 µA/cm² (~1.5× rheobase for the model)
# gives a stable ~10 Hz trace under the Komendantov kinetics, suitable for
# averaging AP shape across spikes.
_DA_STEP_DURATION_MS = 1000.0
_DA_STEP_CURRENT = 1.5
_DA_REFERENCE = "Grace & Bunney 1984 / Komendantov et al. 2004"


@pytest.fixture
def da_neuron() -> Neuron:
    """Dopaminergic SNc neuron instance for all tests in this module."""
    return make_neuron(NEURON_PRESETS[DOPAMINERGIC])


@pytest.fixture
def da_ap_shape_result(da_neuron: Neuron):
    """AP analysis under the standard suprathreshold step."""
    protocol = step_current(
        duration=_DA_STEP_DURATION_MS,
        current_amplitude=_DA_STEP_CURRENT,
    )
    result = simulate_current_clamp(da_neuron, current_external=protocol)
    return analyze_aps_from_result(result)


def _ms_to_samples(ms: float) -> int:
    """Convert ms to simulation samples at SIM_SAMPLING_FREQ."""
    return int(ms * SIM_SAMPLING_FREQ / 1000.0)


# ---------------------------------------------------------------------------
# Spontaneous pacemaking — Grace & Bunney (1984), 1–5 Hz in vitro.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "DA preset is labelled an SNc pacemaker but produces ZERO spikes in "
        "2 s of zero-current simulation. Likely Ih is not strong enough at "
        "v_rest = -62.5 mV to drive spontaneous depolarisation to threshold, "
        "or g_NaP / Na window current is missing to support the pacemaker "
        "loop (Grace & Bunney 1984 report 1–5 Hz autonomous firing in "
        "vitro). Tracked in #304."
    ),
)
def test_da_spontaneous_pacemaking(da_neuron: Neuron) -> None:
    """SNc DA neurons fire autonomously at 1–5 Hz without injected current.

    Grace & Bunney (1984) report regular autonomous firing in slice; the
    rate is set by the interplay of Ih (HCN), Ca²⁺-activated K⁺, and
    intrinsic Na⁺ conductances.
    """
    duration_ms = 2000.0
    zero_current = np.zeros(_ms_to_samples(duration_ms) + 1)
    result = simulate_current_clamp(da_neuron, current_external=zero_current)
    ap = analyze_aps_from_result(result)
    assert_ap_shape(
        ap,
        reference=_DA_REFERENCE,
        firing_rate_hz=(1.0, 5.0),
        min_spike_count=2,
    )


# ---------------------------------------------------------------------------
# Driven firing under depolarising step — pinned even though the preset
# is supposed to pace at zero current. This makes the AP-shape battery
# meaningful while #pacemaking-issue is still open.
# ---------------------------------------------------------------------------


def test_da_suprathreshold_step_produces_repetitive_firing(
    da_ap_shape_result,
) -> None:
    """Mild depolarising step drives slow regular firing in the DA range.

    With the preset's current pacemaking gap, a small bias is required to
    elicit firing.  At 1.5 µA/cm², the model fires at ~10 Hz — within the
    range reported for SNc DA cells under modest depolarisation
    (Komendantov et al. 2004).
    """
    assert_ap_shape(
        da_ap_shape_result,
        reference=_DA_REFERENCE,
        firing_rate_hz=(4.0, 20.0),
        min_spike_count=10,
    )


# ---------------------------------------------------------------------------
# AP shape — DA pacemaker phenotype, Grace & Bunney (1984);
# Komendantov et al. (2004).
# ---------------------------------------------------------------------------


def test_da_ap_threshold_in_da_range(da_ap_shape_result) -> None:
    """Mean AP threshold falls in the DA range (−55 to −40 mV)."""
    assert_ap_shape(
        da_ap_shape_result,
        reference=_DA_REFERENCE,
        threshold_mv=(-55.0, -40.0),
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Mean AP half-width ~0.38 ms is dramatically below the 1.5–3.5 ms "
        "range reported for SNc DA cells (Grace & Bunney 1984). The "
        "Canavier/Komendantov Na⁺/K⁺ kinetics produce APs much narrower "
        "than DA recordings; tracked in #304."
    ),
)
def test_da_ap_half_width_in_da_range(da_ap_shape_result) -> None:
    """Mean AP half-width matches the broad DA phenotype (1.5–3.5 ms)."""
    assert_ap_shape(
        da_ap_shape_result,
        reference=_DA_REFERENCE,
        half_width_ms=(1.5, 3.5),
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Mean peak voltage ~+47 mV exceeds the +10 to +40 mV range reported "
        "for SNc DA cells (Komendantov et al. 2004). Likely g_Na too high; "
        "tracked in #304."
    ),
)
def test_da_ap_peak_voltage_in_da_range(da_ap_shape_result) -> None:
    """Mean AP peak voltage falls within the DA range (+10 to +40 mV)."""
    assert_ap_shape(
        da_ap_shape_result,
        reference=_DA_REFERENCE,
        peak_mv=(10.0, 40.0),
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Mean AHP depth ~−88 mV is far deeper than the −55 to −72 mV range "
        "reported for SNc DA cells (Grace & Bunney 1984). Suggests g_KL or "
        "K⁺ DR conductance over-tuned; tracked in #304."
    ),
)
def test_da_ap_ahp_depth_in_da_range(da_ap_shape_result) -> None:
    """Mean AHP depth falls within the DA range (−72 to −55 mV)."""
    assert_ap_shape(
        da_ap_shape_result,
        reference=_DA_REFERENCE,
        ahp_mv=(-72.0, -55.0),
    )
