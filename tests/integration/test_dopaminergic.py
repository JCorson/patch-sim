"""Behavioral tests for the Dopaminergic (SNc) neuron preset.

Pins the SNc DA pacemaker phenotype: slow autonomous firing (1–5 Hz)
sustained by the Cav1.3 ↔ SK loop (Putzier et al. 2009), with INaP and
Ih playing supporting roles; broad APs, post-spike hyperpolarisation,
post-inhibitory rebound.  Bands cite Grace & Bunney (1984), Lacey et al.
(1989), and Putzier et al. (2009).
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
    rate is set by the Cav1.3 ↔ SK loop (Putzier et al. 2009) — Cav1.3
    drives the slow inter-spike depolarisation, SK shapes the medium AHP —
    with Ih and INaP providing supporting depolarisation.
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
# Depolarisation block — the cell must not park at depolarised V at zero
# current.  The previous Cav1.3+SK tuning had a depol-block fixed point near
# −31 mV that the cell drifted to within the first 200 ms (issue #304).
# ---------------------------------------------------------------------------


def test_da_no_depolarisation_block_at_zero_current(da_neuron: Neuron) -> None:
    """At zero current the membrane never sits depolarised for 500 ms.

    Depolarisation block is the failure mode where the cell parks at a
    depolarised V (e.g. −25 to −31 mV) instead of cycling through AHPs.
    A 500 ms rolling-mean V above −40 mV indicates the cell is stuck above
    the AHP trough — i.e. not pacemaking properly.  Real SNc DA neurons
    spend most of each ISI hyperpolarised below −60 mV (Grace & Bunney 1984).
    """
    duration_ms = 2000.0
    zero_current = np.zeros(_ms_to_samples(duration_ms) + 1)
    result = simulate_current_clamp(da_neuron, current_external=zero_current)
    voltage = np.asarray(result["voltage"])
    window = _ms_to_samples(500.0)
    kernel = np.ones(window) / window
    rolling_mean = np.convolve(voltage, kernel, mode="valid")
    assert np.max(rolling_mean) < -40.0, (
        f"500 ms rolling-mean V reached {np.max(rolling_mean):.1f} mV; "
        f"the cell is in depolarisation block (Grace & Bunney 1984)."
    )


# ---------------------------------------------------------------------------
# Short-window F-I sanity — the cell must keep firing under modest drive
# without dropping to a single-spike-then-block phenotype.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("amplitude", [0.0, 1.0, 2.0, 3.0])
def test_da_short_window_keeps_firing(da_neuron: Neuron, amplitude: float) -> None:
    """500 ms steps at 0–3 µA/cm² produce repetitive firing, no block.

    The user-visible UI symptom of issue #304 was a single spike at the start
    of every F-I sweep followed by sustained depolarisation at ~−25 mV.  This
    parametrised test runs the same short-window sweeps the F-I curve uses
    and asserts the cell produces ≥2 spikes and never enters depol block
    (per the rolling-mean check above).
    """
    duration_ms = 500.0
    protocol = step_current(duration=duration_ms, current_amplitude=amplitude)
    result = simulate_current_clamp(da_neuron, current_external=protocol)
    ap = analyze_aps_from_result(result)
    assert ap.spike_count >= 2, (
        f"At I = {amplitude} µA/cm², cell fired only {ap.spike_count} spike(s) "
        f"— consistent with depolarisation block (issue #304)."
    )
    voltage = np.asarray(result["voltage"])
    window = _ms_to_samples(300.0)
    kernel = np.ones(window) / window
    rolling_mean = np.convolve(voltage, kernel, mode="valid")
    assert np.max(rolling_mean) < -40.0, (
        f"At I = {amplitude} µA/cm², 300 ms rolling-mean V reached "
        f"{np.max(rolling_mean):.1f} mV — depolarisation block."
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
    """Mean AHP trough falls within the SK-overshoot range (−95 to −55 mV).

    ``analyze_aps`` measures ``ahp_depth`` as the *minimum* V between the AP
    peak and the next threshold crossing — i.e. the brief K⁺ overshoot
    immediately after the spike (fast AHP / fAHP).  The literature −65 mV
    figure for SNc DA cells refers to the medium AHP that the cell holds
    *between* the fAHP and the ramp to the next spike.  In the Cav1.3+SK
    model the post-spike SK transient draws V close to E_K (≈ −95 mV) before
    Cav1.3 reactivates, so the measured trough is deeper than the literature
    mAHP.  The lower bound here permits the SK overshoot toward E_K; the
    upper bound (−55 mV) still excludes a depol-block phenotype where there
    is effectively no AHP at all.
    """
    assert_ap_shape(
        da_ap_shape_result,
        reference=_DA_REFERENCE,
        ahp_mv=(-95.0, -55.0),
    )
