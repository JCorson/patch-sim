"""Behavioral tests for the SNc Dopaminergic neuron preset.

Pins the SNc DA pacemaker phenotype: autonomous firing within the published
in-vitro range (Grace & Bunney 1984; Liss & Roeper 2008) sustained by the
Cav1.3 + INaP_SNc subthreshold ramp and SK-shaped AHP (Putzier 2009 + Drion
2011 reconciliation), broad APs, and tonic firing across the UI F-I sweep.

The somatic single-compartment model does not reproduce depolarization block:
at every amplitude tested up to 15 µA/cm² and every duration up to 10 s the
cell fires tonically with rolling-mean V below −70 mV.  Real SNc DA neurons
enter sustained block above ~100 pA (Tucker et al. 2012); reproducing that
phenotype requires dendritic Na inactivation that this representation lacks.
Tracked in #323.
"""

import numpy as np
import pytest

from patch_sim.analysis.ap_metrics import analyze_aps_from_result
from patch_sim.clamp_simulations import (
    SIM_SAMPLING_FREQ,
    simulate_current_clamp,
    simulate_voltage_clamp,
)
from patch_sim.neuron import Neuron
from patch_sim.presets import make_dopaminergic
from patch_sim.protocols import step_current, step_voltage
from tests.integration._ap_shape import assert_ap_shape

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# 5 s zero-current trace gives ~10 spikes at the preset's pacing rate, enough
# for stable AP-shape statistics.
_DA_SPONT_DURATION_MS = 5000.0
# Modest depolarizing step verifying sustained driven firing within the
# pre-block range; the new tuning supports steady firing up to ~4 µA/cm².
_DA_STEP_DURATION_MS = 2000.0
_DA_STEP_CURRENT = 1.0
_DA_REFERENCE = "Grace & Bunney 1984 / Putzier 2009 / Drion 2011 / Tucker 2012"


@pytest.fixture
def da_neuron() -> Neuron:
    """Dopaminergic SNc neuron instance for all tests in this module."""
    return make_dopaminergic()


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
    """SNc DA neurons fire autonomously within the published in-vitro range.

    SNc DA in-vitro firing rates span 1–8 Hz across published preparations:
    Grace & Bunney 1984: 1–5 Hz; Wilson & Callaway 2000: ~3 Hz; Liss &
    Roeper 2008: 1–8 Hz; Tucker et al. 2012 controls: 4–6 Hz.  The (4.0,
    8.0) band is a regression guard inside this published window.  The
    model measures ~7 Hz at zero current — the strong SK pull required to
    clear the post-spike Na window plateau (without that pull, V hangs
    pathologically at −30 mV for 20+ ms after each spike) places the cell
    at the upper end of the in-vitro range.  Pacemaking proceeds through
    the Putzier+Drion subthreshold ramp (Cav1.3 with V½ = −31.1 mV plus
    INaP_SNc with V½ = −65 mV) balanced against the SK-shaped medium AHP.
    """
    duration_ms = 5000.0
    zero_current = np.zeros(_ms_to_samples(duration_ms) + 1)
    result = simulate_current_clamp(da_neuron, current_external=zero_current)
    ap = analyze_aps_from_result(result)
    assert_ap_shape(
        ap,
        reference=_DA_REFERENCE,
        firing_rate_hz=(4.0, 8.0),
        min_spike_count=20,
    )


# ---------------------------------------------------------------------------
# Modest depolarization sustains firing.
# ---------------------------------------------------------------------------


def test_da_modest_step_sustains_firing(da_neuron: Neuron) -> None:
    """A 1 µA/cm² step keeps the cell firing without entering block.

    Real SNc DA neurons enter depolarization block above ~100 pA injected
    current at long sustained drive (Tucker et al. 2012); for a 50 pF cell
    that is roughly 2 µA/cm².  A 1 µA/cm² step sits well inside the
    regular-firing range; 2 s of 1 µA/cm² should produce sustained firing
    accelerated above the zero-current rate (~4 Hz) but still in a band
    consistent with regular tonic mode (1–15 Hz).
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
        firing_rate_hz=(1.0, 15.0),
        min_spike_count=4,
    )


# ---------------------------------------------------------------------------
# UI default F-I protocol — 200 ms steps over 0–12 µA/cm² in 1.5 µA/cm² steps.
# The somatic single-compartment model fires tonically across the entire
# sweep: empirical sweep (scratch/characterize_da_block.py) shows 4–7 spikes
# per 200 ms step at every amplitude, with 150 ms rolling-mean V never
# exceeding −70 mV.  Real SNc DA neurons enter depolarization block above
# ~100 pA sustained drive (Tucker et al. 2012, J. Neurophysiol. 108:288),
# but reproducing this requires the dendritic Na inactivation pool that a
# somatic single-compartment representation cannot supply.  Tracked in #323.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "amplitude,min_spikes",
    [
        # min_spikes floors set ≥1 below the empirically observed spike
        # counts at correct sampling (40 kHz, post-#348 alignment).  Prior
        # to the alignment, the 200 ms protocol was silently stretched to
        # 500 ms by the simulator, inflating the observed counts ~2.5× and
        # giving the original (3–5) floors.  At correct timing the SNc
        # somatic model fires 2 spikes at 0–3 µA/cm² and 3 spikes from
        # 4.5 µA/cm² onward, so the floors are 1–2.
        (0.0, 1),
        (1.5, 1),
        (3.0, 1),
        (4.5, 2),
        (6.0, 2),
        (7.5, 2),
        (9.0, 2),
        (10.5, 2),
        (12.0, 2),
    ],
)
def test_da_ui_fi_protocol(
    da_neuron: Neuron, amplitude: float, min_spikes: int
) -> None:
    """200 ms F-I step matches the UI default protocol — issue #304.

    The somatic single-compartment model fires tonically across the full
    UI F-I sweep — depolarization block is not reproduced (#323; missing
    dendritic Na inactivation, Tucker et al. 2012).  Each step must
    therefore produce at least ``min_spikes`` action potentials and must
    never park above −40 mV in the 150 ms rolling mean.  ``min_spikes``
    floors are set 1–2 below the empirically observed counts to allow for
    routine retuning without spurious failures.
    """
    duration_ms = 200.0
    protocol = step_current(duration=duration_ms, current_amplitude=amplitude)
    result = simulate_current_clamp(da_neuron, current_external=protocol)
    ap = analyze_aps_from_result(result)
    assert ap.spike_count >= min_spikes, (
        f"At I = {amplitude} µA/cm², cell fired only {ap.spike_count} spike(s); "
        f"expected ≥{min_spikes}."
    )
    voltage = np.asarray(result["voltage"])
    window = _ms_to_samples(150.0)
    kernel = np.ones(window) / window
    rolling_mean = np.convolve(voltage, kernel, mode="valid")
    assert np.max(rolling_mean) < -40.0, (
        f"At I = {amplitude} µA/cm², 150 ms rolling-mean V reached "
        f"{np.max(rolling_mean):.1f} mV — unexpected depolarization block "
        f"plateau (the somatic model is not expected to enter block; #323)."
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


def test_da_single_ap_repolarizes_cleanly(da_neuron: Neuron) -> None:
    """A single evoked AP repolarizes within ~5 ms; no Cav1.3-driven plateau.

    Drives a single AP with a 5 ms × 4 µA/cm² step (the UI ACTION_POTENTIAL
    protocol) and asserts that within 5 ms after the peak the voltage has
    dropped below −60 mV.  Real SNc DA APs are <3 ms wide with a clean
    medium AHP at −60 to −75 mV (Grace & Bunney 1984; Putzier et al. 2009).
    A long plateau hanging at ~−30 mV after the spike means Cav1.3 + Na
    window currents are dominating repolarization — a symptom of an
    unphysiologically large persistent inward current that the half-width
    metric does not catch (half-width is computed at the spike midpoint,
    which the trace can cross cleanly even when V parks at −30 mV after).
    """
    pre_ms, stim_ms, post_ms = 10.0, 5.0, 50.0
    total_samples = _ms_to_samples(pre_ms + stim_ms + post_ms) + 1
    current = np.zeros(total_samples)
    pre_n = _ms_to_samples(pre_ms)
    stim_n = _ms_to_samples(stim_ms)
    current[pre_n : pre_n + stim_n] = 4.0
    result = simulate_current_clamp(da_neuron, current_external=current)
    t = np.asarray(result["time"])
    v = np.asarray(result["voltage"])

    peak_idx = int(np.argmax(v))
    peak_t = float(t[peak_idx])
    # Within 5 ms after the peak, V must drop below -60 mV.
    after_mask = (t > peak_t) & (t <= peak_t + 5.0)
    v_after = v[after_mask]
    assert v_after.size > 0, "post-peak window is empty"
    min_v_5ms = float(np.min(v_after))
    assert min_v_5ms < -60.0, (
        f"AP plateau detected: V never falls below −60 mV in 5 ms after the "
        f"peak (min V = {min_v_5ms:.2f} mV at peak={peak_t:.2f} ms).  "
        f"Reference: Grace & Bunney 1984; Putzier 2009."
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


# ---------------------------------------------------------------------------
# Slow Na inactivation engagement and depol-block recovery — issue #330.
# ---------------------------------------------------------------------------

# F-I sweep upper bound from test_da_ui_fi_protocol; the issue specifies the
# sustained drive at the F-I sweep upper bound as the acceptance amplitude.
_DA_DEPOL_DRIVE_AMP_UA = 12.0
_DA_DEPOL_DRIVE_MS = 200.0


def test_da_inap_slow_inactivation_engages_under_depol_clamp(
    da_neuron: Neuron,
) -> None:
    """The sNaP_snc gate closes under sustained voltage clamp at −15 mV (#330).

    Direct mechanism check for #330: the Khaliq & Bean 2010 / Magistretti &
    Alonso 1999 slow inactivation gate baked into ``make_snc_inap_channel``
    must close significantly when the membrane is held depolarized.  Voltage
    clamp from a −75 mV hold (slow gate ≈ 0.94 available) to −15 mV for
    300 ms is enough time for the τ_floor ≈ 20 ms gate to settle below 5%
    availability.  Under current clamp the SNc cell continues firing through
    the F-I-upper-bound drive (the somatic model resists depol-block, #323),
    so the slow gate spends most of the cycle near hyperpolarized AHPs and
    does not engage strongly — voltage clamp is the cleaner mechanism check.
    """
    hold_ms, step_ms = 100.0, 300.0
    protocol = step_voltage(
        duration=hold_ms + step_ms,
        voltage_amplitude=-15.0,
        step_start=hold_ms,
        step_duration=step_ms,
        holding_voltage=-75.0,
    )
    result = simulate_voltage_clamp(da_neuron, protocol)
    sNaP_at_rest = float(result["sNaP_snc"][_ms_to_samples(hold_ms) - 1])
    assert sNaP_at_rest > 0.9, (
        f"sNaP_snc rest availability at −75 mV hold is unexpectedly low "
        f"({sNaP_at_rest:.3f}); the inactivation engagement check below "
        "would be meaningless if the gate were already partly closed."
    )
    sNaP_at_step_end = float(result["sNaP_snc"][-1])
    fraction_inactivated = 1.0 - sNaP_at_step_end / sNaP_at_rest
    assert fraction_inactivated > 0.5, (
        f"sNaP_snc did not meaningfully inactivate: rest={sNaP_at_rest:.3f}, "
        f"step end={sNaP_at_step_end:.3f}, "
        f"fraction inactivated={fraction_inactivated:.2f} (expected > 0.5)"
    )


def test_da_fast_na_slow_inactivation_engages_under_depol_clamp(
    da_neuron: Neuron,
) -> None:
    """The sNa_da gate closes under sustained voltage clamp at −15 mV (#330).

    Direct mechanism check for #330: the Khaliq & Bean 2010 slow voltage-
    dependent inactivation gate baked into ``make_dopaminergic_na_channel``
    must close significantly when the membrane is held depolarized.  Khaliq
    & Bean 2010 directly studied SNc DA neurons.  Voltage clamp at −15 mV
    is used (rather than current clamp) because the SNc cell continues
    firing through the F-I-upper-bound drive (#323) so the slow gate stays
    mostly open under current clamp.
    """
    hold_ms, step_ms = 100.0, 300.0
    protocol = step_voltage(
        duration=hold_ms + step_ms,
        voltage_amplitude=-15.0,
        step_start=hold_ms,
        step_duration=step_ms,
        holding_voltage=-75.0,
    )
    result = simulate_voltage_clamp(da_neuron, protocol)
    sNa_at_rest = float(result["sNa_da"][_ms_to_samples(hold_ms) - 1])
    assert sNa_at_rest > 0.9, (
        f"sNa_da rest availability at −75 mV hold is unexpectedly low "
        f"({sNa_at_rest:.3f}); the inactivation engagement check below "
        "would be meaningless if the gate were already partly closed."
    )
    sNa_at_step_end = float(result["sNa_da"][-1])
    fraction_inactivated = 1.0 - sNa_at_step_end / sNa_at_rest
    assert fraction_inactivated > 0.5, (
        f"sNa_da did not meaningfully inactivate: rest={sNa_at_rest:.3f}, "
        f"step end={sNa_at_step_end:.3f}, "
        f"fraction inactivated={fraction_inactivated:.2f} (expected > 0.5)"
    )


def test_da_recovers_from_sustained_suprathreshold_drive(
    da_neuron: Neuron,
) -> None:
    """SNc DA settles below −50 mV after +12 µA/cm² × 200 ms drive (#330).

    Regression guard rather than a fix for #323: the somatic single-
    compartment model already cycles around −60 mV under this drive
    (it never enters depol-block; see the module docstring and #323).
    With the new sNa_da and sNaP_snc gates the cell still must not park
    above −50 mV in the last 200 ms of the post-stim epoch.  This guards
    against the new gates *introducing* a depol-block plateau via
    misconfigured kinetics or an unintended interaction with the rest of
    the channel cocktail.
    """
    n_pre = _ms_to_samples(100.0)
    n_step = _ms_to_samples(_DA_DEPOL_DRIVE_MS)
    n_post = _ms_to_samples(700.0)
    current = np.concatenate(
        [
            np.zeros(n_pre),
            np.full(n_step, _DA_DEPOL_DRIVE_AMP_UA),
            np.zeros(n_post + 1),
        ]
    )
    result = simulate_current_clamp(da_neuron, current_external=current)
    tail = np.asarray(result["voltage"][-_ms_to_samples(200.0) :])
    mean_v = float(tail.mean())
    assert mean_v < -50.0, (
        f"SNc DA failed to recover from +12 µA/cm² × 200 ms; "
        f"mean V in last 200 ms = {mean_v:.2f} mV"
    )


def test_da_sNa_da_and_sNaP_snc_columns_present(da_neuron: Neuron) -> None:
    """``sNa_da`` and ``sNaP_snc`` columns appear in the simulation output.

    Direct check of the issue #330 acceptance criterion that both slow
    inactivation gating variables surface as named columns.  The SNc preset
    runs Cav1.3, SK, and Ih alongside the SNc-namespaced INaP_SNc; this
    test pins the SNc-specific gate naming (``sNa_da``, ``sNaP_snc``) so
    they cannot collide with the bare ``sNa``/``sNaP`` of other Na factories
    should a hypothetical preset combine them.
    """
    n_samples = _ms_to_samples(50.0) + 1
    zero_current = np.zeros(n_samples)
    result = simulate_current_clamp(da_neuron, current_external=zero_current)
    names = result.dtype.names or ()
    assert "sNa_da" in names, (
        f"Expected 'sNa_da' column for fast-Na slow inactivation; got {names!r}"
    )
    assert "sNaP_snc" in names, (
        f"Expected 'sNaP_snc' column for INaP_SNc slow inactivation; got {names!r}"
    )
    assert "pSNc" in names, (
        f"Expected 'pSNc' INaP_SNc activation column; got {names!r}.  Loss "
        "of this column would indicate a regression in INaP_SNc wiring."
    )
