"""Behavioral tests for the Dopaminergic (SNc) neuron preset.

Pins the SNc DA pacemaker phenotype: slow autonomous firing (1–5 Hz)
sustained by Ih + INaP, broad APs, post-spike hyperpolarisation,
post-inhibitory rebound.  Bands cite Grace & Bunney (1984) and
Komendantov et al. (2004).

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

# Spontaneous firing duration: 5 s of zero current produces ~10 spikes at the
# preset's ~2 Hz pacing rate, enough to average AP shape across spikes.
_DA_SPONT_DURATION_MS = 5000.0
# Modest depolarising step used to verify the cell sustains repetitive firing
# without entering depolarisation block; 0.5 µA/cm² was chosen empirically as
# the largest current the post-#304 preset can take before block.
_DA_STEP_DURATION_MS = 2000.0
_DA_STEP_CURRENT = 0.5
_DA_REFERENCE = "Grace & Bunney 1984 / Komendantov et al. 2004"


@pytest.fixture
def da_neuron() -> Neuron:
    """Dopaminergic SNc neuron instance for all tests in this module."""
    return make_neuron(NEURON_PRESETS[DOPAMINERGIC])


@pytest.fixture
def da_ap_shape_result(da_neuron: Neuron):
    """AP analysis under the autonomous (zero-current) pacing protocol.

    The DA preset is a tonic pacemaker — autonomous firing at ~2 Hz — so AP
    shape is averaged over the spontaneous train rather than a driven step.
    A 5 s window gives ~10 spikes for stable shape statistics.
    """
    duration_samples = int(_DA_SPONT_DURATION_MS * SIM_SAMPLING_FREQ / 1000.0)
    zero_current = np.zeros(duration_samples + 1)
    result = simulate_current_clamp(da_neuron, current_external=zero_current)
    return analyze_aps_from_result(result)


def _ms_to_samples(ms: float) -> int:
    """Convert ms to simulation samples at SIM_SAMPLING_FREQ."""
    return int(ms * SIM_SAMPLING_FREQ / 1000.0)


# ---------------------------------------------------------------------------
# Spontaneous pacemaking — Grace & Bunney (1984), 1–5 Hz in vitro.
# ---------------------------------------------------------------------------


def test_da_spontaneous_pacemaking(da_neuron: Neuron) -> None:
    """SNc DA neurons fire autonomously at 1–5 Hz without injected current.

    Grace & Bunney (1984) report regular autonomous firing in slice; the
    rate is set by the interplay of Ih (HCN), INaP (persistent Na), and
    intrinsic K⁺ conductances (Wilson & Callaway 2000).
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
# Modest depolarisation sustains firing (no depolarisation block).
# ---------------------------------------------------------------------------


def test_da_modest_step_sustains_firing(da_neuron: Neuron) -> None:
    """A small depolarising step keeps the cell firing without block.

    The DA preset is a tonic pacemaker with a narrow operating range —
    aggressive drive (≳1 µA/cm²) tips it into depolarisation block.  This
    test verifies that a modest 0.5 µA/cm² × 2 s step still produces
    sustained firing (≥4 APs, rate in 1–5 Hz), confirming the cell remains
    in the regular-firing regime under physiological synaptic-like drive.
    """
    protocol = step_current(
        duration=_DA_STEP_DURATION_MS,
        current_amplitude=_DA_STEP_CURRENT,
    )
    result = simulate_current_clamp(da_neuron, current_external=protocol)
    ap = analyze_aps_from_result(result)
    assert_ap_shape(
        ap,
        reference=_DA_REFERENCE,
        firing_rate_hz=(1.0, 5.0),
        min_spike_count=4,
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


def test_da_ap_half_width_in_da_range(da_ap_shape_result) -> None:
    """Mean AP half-width matches the broad DA phenotype (1.5–3.5 ms)."""
    assert_ap_shape(
        da_ap_shape_result,
        reference=_DA_REFERENCE,
        half_width_ms=(1.5, 3.5),
    )


def test_da_ap_peak_voltage_in_da_range(da_ap_shape_result) -> None:
    """Mean AP peak voltage falls within the DA range (+10 to +40 mV)."""
    assert_ap_shape(
        da_ap_shape_result,
        reference=_DA_REFERENCE,
        peak_mv=(10.0, 40.0),
    )


def test_da_ap_ahp_depth_in_da_range(da_ap_shape_result) -> None:
    """Mean AHP depth falls within the DA range (−72 to −55 mV)."""
    assert_ap_shape(
        da_ap_shape_result,
        reference=_DA_REFERENCE,
        ahp_mv=(-72.0, -55.0),
    )
