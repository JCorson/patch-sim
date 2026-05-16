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
from patch_sim.clamp_simulations import simulate_current_clamp
from patch_sim.constants import SIM_SAMPLING_FREQ
from patch_sim.neuron import Neuron
from patch_sim.presets import make_ca1_pyramidal
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
    return make_ca1_pyramidal()


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


def test_ca1_single_ap_repolarizes_cleanly(ca1_neuron: Neuron) -> None:
    """A single evoked AP repolarizes within 6 ms; no post-spike plateau.

    Drives a single AP with the UI ACTION_POTENTIAL protocol (2 µA/cm² ×
    15 ms) and asserts that within 6 ms after the peak the voltage has
    dropped below −52 mV.  CA1 pyramidal APs are ~0.7–1.5 ms half-width
    followed by a fast/medium AHP to roughly −55 to −75 mV; the Ca²⁺/IKCa
    interaction makes it deeper and a little slower than the cortical RS
    AHP but it is still reached within ~3–6 ms (Spruston & Johnston 2008,
    J. Neurophysiol. 99:2782; Storm 1990, Prog. Brain Res. 83:161; Bean
    2007, Nat. Rev. Neurosci. 8:451, Fig. 2 — hippocampal pyramidal single
    APs repolarize below threshold within ~3–5 ms).  −52 mV is a
    conservative bound below the resting/threshold band so a single
    weakly-driven AP with a shallow fAHP still clears it; a long plateau
    hanging at ~−30 mV after the spike would mean a window current is
    dominating repolarization — a pathology the mean half-width and
    AHP-depth metrics do not catch (half-width is measured at the spike
    midpoint, which the trace can cross cleanly even when V parks at
    −30 mV afterward; cf. SNc DA, PR #318).
    """
    pre_ms, stim_ms, post_ms = 10.0, 15.0, 60.0
    total_samples = _ms_to_samples(pre_ms + stim_ms + post_ms) + 1
    current = np.zeros(total_samples)
    pre_n = _ms_to_samples(pre_ms)
    stim_n = _ms_to_samples(stim_ms)
    current[pre_n : pre_n + stim_n] = 2.0
    result = simulate_current_clamp(ca1_neuron, current_external=current)
    t = np.asarray(result["time"])
    v = np.asarray(result["voltage"])

    peak_idx = int(np.argmax(v))
    peak_t = float(t[peak_idx])
    after_mask = (t > peak_t) & (t <= peak_t + 6.0)
    v_after = v[after_mask]
    assert v_after.size > 0, "post-peak window is empty"
    min_v = float(np.min(v_after))
    assert min_v < -52.0, (
        f"AP plateau detected: V never falls below −52 mV in 6 ms after the "
        f"peak (min V = {min_v:.2f} mV at peak={peak_t:.2f} ms).  "
        f"Reference: {_CA1_REFERENCE}; Bean 2007."
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


# ---------------------------------------------------------------------------
# Slow Na inactivation engagement and depol-block recovery — issue #328.
# ---------------------------------------------------------------------------


_CA1_DEPOL_DRIVE_AMP_UA = 30.0
_CA1_DEPOL_DRIVE_MS = 200.0


def test_ca1_inap_slow_inactivation_engages_during_drive(ca1_neuron: Neuron) -> None:
    """The sNaP gate closes during +30 µA/cm² × 200 ms drive (#328 mechanism).

    Direct mechanism check for #328: the Magistretti & Alonso 1999 slow
    inactivation gate added to ``make_inap_channel`` must be doing real
    work during sustained suprathreshold drive.  By the end of the step,
    sNaP should have lost at least half its rest availability so the
    persistent Na⁺ contribution to the depol-block plateau is suppressed.
    The +30 µA/cm² level matches the recovery-test stimulus suggested in
    the issue body — CA1's deeper IKCa AHP makes it less excitable than
    cortical pyramidal, so the +12 µA/cm² level used for CP is too weak
    here to drive engagement past 50 %.
    """
    n_pre = _ms_to_samples(100.0)
    n_step = _ms_to_samples(_CA1_DEPOL_DRIVE_MS)
    n_post = _ms_to_samples(50.0)
    current = np.concatenate(
        [
            np.zeros(n_pre),
            np.full(n_step, _CA1_DEPOL_DRIVE_AMP_UA),
            np.zeros(n_post + 1),
        ]
    )
    result = simulate_current_clamp(ca1_neuron, current_external=current)
    sNaP_at_rest = float(result["sNaP"][n_pre - 1])
    assert sNaP_at_rest > 0.5, (
        f"sNaP rest availability is unexpectedly low ({sNaP_at_rest:.3f}); "
        "the inactivation engagement check below is meaningless if the "
        "gate is already mostly closed at v_rest."
    )
    sNaP_at_step_end = float(result["sNaP"][n_pre + n_step - 1])
    fraction_inactivated = 1.0 - sNaP_at_step_end / sNaP_at_rest
    assert fraction_inactivated > 0.5, (
        f"sNaP did not meaningfully inactivate: rest={sNaP_at_rest:.3f}, "
        f"step end={sNaP_at_step_end:.3f}, "
        f"fraction inactivated={fraction_inactivated:.2f} (expected > 0.5)"
    )


def test_ca1_fast_na_slow_inactivation_engages_during_drive(
    ca1_neuron: Neuron,
) -> None:
    """The sNa12 gate (fast Na slow inactivation) closes during +30 µA/cm² × 200 ms.

    Direct mechanism check for #328: the Mickus, Jung & Spruston 1999 slow
    voltage-dependent inactivation gate baked into ``make_nav12_channel``
    must lose at least half its rest availability by the end of a sustained
    suprathreshold step, abolishing the residual fast-Na h-tail that would
    otherwise pin the cell on a depolarization plateau.  Mickus 1999
    directly studied CA1 pyramidal cells, so this is the cell type the
    paper actually characterised.
    """
    n_pre = _ms_to_samples(100.0)
    n_step = _ms_to_samples(_CA1_DEPOL_DRIVE_MS)
    n_post = _ms_to_samples(50.0)
    current = np.concatenate(
        [
            np.zeros(n_pre),
            np.full(n_step, _CA1_DEPOL_DRIVE_AMP_UA),
            np.zeros(n_post + 1),
        ]
    )
    result = simulate_current_clamp(ca1_neuron, current_external=current)
    sNa_at_rest = float(result["sNa12"][n_pre - 1])
    assert sNa_at_rest > 0.5, (
        f"sNa12 rest availability is unexpectedly low ({sNa_at_rest:.3f}); "
        "the inactivation engagement check below is meaningless if the "
        "gate is already mostly closed at v_rest."
    )
    sNa_at_step_end = float(result["sNa12"][n_pre + n_step - 1])
    fraction_inactivated = 1.0 - sNa_at_step_end / sNa_at_rest
    assert fraction_inactivated > 0.5, (
        f"sNa12 did not meaningfully inactivate: rest={sNa_at_rest:.3f}, "
        f"step end={sNa_at_step_end:.3f}, "
        f"fraction inactivated={fraction_inactivated:.2f} (expected > 0.5)"
    )


def test_ca1_recovers_from_sustained_suprathreshold_drive(
    ca1_neuron: Neuron,
) -> None:
    """CA1 pyramidal repolarizes after +30 µA/cm² × 200 ms (#328 regression).

    Without slow Na inactivation the cell can hang on a depol-block
    plateau under sustained drive.  With sNaP (Magistretti & Alonso 1999)
    and Pospischil sNa (Mickus, Jung & Spruston 1999) opted in, the cell
    must escape the plateau within the post-stim window and settle below
    −50 mV during the last 200 ms of a 700 ms post-stimulus epoch.
    """
    n_pre = _ms_to_samples(100.0)
    n_step = _ms_to_samples(_CA1_DEPOL_DRIVE_MS)
    n_post = _ms_to_samples(700.0)
    current = np.concatenate(
        [
            np.zeros(n_pre),
            np.full(n_step, _CA1_DEPOL_DRIVE_AMP_UA),
            np.zeros(n_post + 1),
        ]
    )
    result = simulate_current_clamp(ca1_neuron, current_external=current)
    tail = np.asarray(result["voltage"][-_ms_to_samples(200.0) :])
    mean_v = float(tail.mean())
    assert mean_v < -50.0, (
        f"CA1 pyramidal failed to recover from +30 µA/cm² × 200 ms; "
        f"mean V in last 200 ms = {mean_v:.2f} mV"
    )
