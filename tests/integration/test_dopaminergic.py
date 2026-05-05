"""Behavioural tests for the SNc Dopaminergic neuron preset.

Pins the SNc DA pacemaker phenotype: slow autonomous firing (1–5 Hz) sustained
by the Cav1.3 + INaP_SNc subthreshold ramp and SK-shaped AHP (Putzier 2009 +
Drion 2011 reconciliation), broad APs, and a biologically realistic
depolarisation-block boundary at ~5 µA/cm² (Tucker et al. 2012).
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

# 5 s zero-current trace gives ~10 spikes at the preset's pacing rate, enough
# for stable AP-shape statistics.
_DA_SPONT_DURATION_MS = 5000.0
# Modest depolarising step verifying sustained driven firing within the
# pre-block range; the new tuning supports steady firing up to ~4 µA/cm².
_DA_STEP_DURATION_MS = 2000.0
_DA_STEP_CURRENT = 1.0
_DA_REFERENCE = "Grace & Bunney 1984 / Putzier 2009 / Drion 2011 / Tucker 2012"


@pytest.fixture
def da_neuron() -> Neuron:
    """Dopaminergic SNc neuron instance for all tests in this module."""
    return make_neuron(NEURON_PRESETS[DOPAMINERGIC])


@pytest.fixture
def da_ap_shape_result(da_neuron: Neuron):
    """AP analysis under the autonomous (zero-current) pacing protocol."""
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

    Grace & Bunney (1984) report regular autonomous firing in slice; the rate
    is set by the Putzier+Drion subthreshold ramp (Cav1.3 with V½ = −31.1 mV
    plus INaP_SNc with V½ = −65 mV) balanced against the SK-shaped AHP.
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
# Modest depolarisation sustains firing.
# ---------------------------------------------------------------------------


def test_da_modest_step_sustains_firing(da_neuron: Neuron) -> None:
    """A 1 µA/cm² step keeps the cell firing without entering block.

    Real SNc DA neurons enter depolarisation block above ~100 pA injected
    current (Tucker et al. 2012); for a 7 pF cell that is roughly 5 µA/cm².
    A 1 µA/cm² step sits well inside the regular-firing range and should
    produce sustained firing in the 1–8 Hz band.
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
        firing_rate_hz=(1.0, 8.0),
        min_spike_count=4,
    )


# ---------------------------------------------------------------------------
# UI default F-I protocol — 200 ms steps over 0–12 µA/cm² in 1.5 µA/cm² steps.
# Encodes the literature-grounded depolarisation-block onset near 5 µA/cm²
# (Tucker et al. 2012): the cell fires across 0–4.5 µA/cm² and shows
# progressive block above that.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "amplitude,min_spikes,allow_block",
    [
        # Pacemaking range: cell fires throughout the step.
        (0.0, 1, False),
        (1.5, 2, False),
        (3.0, 2, False),
        # Transition: cell may begin to block toward the end of the step.
        (4.5, 1, True),
        # Block range: cell fires once or twice then parks at depolarised V.
        (6.0, 0, True),
        (9.0, 0, True),
    ],
)
def test_da_ui_fi_protocol(
    da_neuron: Neuron, amplitude: float, min_spikes: int, allow_block: bool
) -> None:
    """200 ms F-I step matches the UI default protocol — issue #304.

    SNc DA neurons enter depolarisation block above ~100 pA injected current
    (Tucker et al. 2012); for a 7 pF cell this is ~5 µA/cm².  Below that
    threshold the cell must fire at least one spike per 200 ms window without
    a sustained depol-block plateau; above it, progressive block is the
    expected biological phenotype rather than a model failure.
    """
    duration_ms = 200.0
    protocol = step_current(duration=duration_ms, current_amplitude=amplitude)
    result = simulate_current_clamp(da_neuron, current_external=protocol)
    ap = analyze_aps_from_result(result)
    assert ap.spike_count >= min_spikes, (
        f"At I = {amplitude} µA/cm², cell fired only {ap.spike_count} spike(s); "
        f"expected ≥{min_spikes} (Tucker 2012 depol-block onset ~5 µA/cm²)."
    )
    if not allow_block:
        # Below depol-block onset, no sustained plateau above −40 mV.
        voltage = np.asarray(result["voltage"])
        window = _ms_to_samples(150.0)
        kernel = np.ones(window) / window
        rolling_mean = np.convolve(voltage, kernel, mode="valid")
        assert np.max(rolling_mean) < -40.0, (
            f"At I = {amplitude} µA/cm², 150 ms rolling-mean V reached "
            f"{np.max(rolling_mean):.1f} mV — depolarisation block below onset."
        )


# ---------------------------------------------------------------------------
# AP shape — DA pacemaker phenotype, Grace & Bunney (1984);
# Komendantov et al. (2004); Lacey et al. (1989).
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
    """Mean AHP trough sits between the SK-shaped overshoot and the medium AHP.

    ``analyze_aps`` measures the *minimum* V between spike peak and the next
    threshold crossing — i.e. the brief K⁺ overshoot toward E_K immediately
    after the spike (fast AHP).  The literature −65 mV figure for SNc DA
    cells (Lacey et al. 1989) refers to the medium AHP plateau between fAHP
    and the next pacemaker ramp.  In the Cav1.3 + SK loop the fAHP is drawn
    close to E_K (≈ −95 mV) before SK closes; the band here permits that
    overshoot while still excluding a depol-block phenotype with no AHP at
    all (upper bound −55 mV).
    """
    assert_ap_shape(
        da_ap_shape_result,
        reference=_DA_REFERENCE,
        ahp_mv=(-95.0, -55.0),
    )
