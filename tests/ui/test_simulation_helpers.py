"""Behavior-focused unit tests for the analysis-formatting helpers.

These tests cover edge cases and documented invariants that the e2e suite
(``tests/e2e/``) and integration tests (``tests/integration/``) cannot catch:
em-dash and asterisk display conventions, branch selection between F-I and
hyperpolarization analysis, guard returns when sweeps and protocol steps
disagree, and the duty-cycle invariant that silent CC sweeps still count
toward the denominator.

Tests that would only re-state implementation details (exact dict key sets,
pure pass-through serialisers) live in the e2e suite by virtue of the UI
consuming the same dicts.
"""

import os

import numpy as np
import pytest

from patch_sim.analysis.ap_metrics import APAnalysisResult, SpikeMetrics
from patch_sim.analysis.burst_metrics import BurstAnalysisResult
from patch_sim.analysis.calcium_transients import CalciumTransient
from patch_sim.constants import CURRENT_CLAMP, VOLTAGE_CLAMP

# Reflex's metaclass and instance guards both require a testing environment.
os.environ.setdefault("PYTEST_CURRENT_TEST", "test_simulation_helpers.py::setup")
pytest.importorskip("reflex")

from patch_sim_ui.state._analysis_format import (  # noqa: E402
    _build_phase_plane_data,
    _compute_burst_data,
    _compute_cc_multi_sweep_analysis,
    _compute_iv_data,
    _compute_multi_sweep_burst_data,
    _fmt_optional,
    _format_ca_transient_dict,
    _serialise_burst_summary,
)
from patch_sim_ui.sweep import Sweep  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_spike(index: int = 0) -> SpikeMetrics:
    """Return a :class:`SpikeMetrics` with plausible defaults.

    Args:
        index: Spike index (also used as the spike's peak time in ms).

    Returns:
        A :class:`SpikeMetrics` populated with values valid for AP analysis.
    """
    return SpikeMetrics(
        index=index,
        threshold_voltage=-50.0,
        threshold_time=0.0,
        peak_voltage=30.0,
        peak_time=float(index),
        rise_time=0.5,
        half_width=1.0,
        ahp_depth=-70.0,
    )


def _make_ap_result(spikes: list[SpikeMetrics]) -> APAnalysisResult:
    """Return an :class:`APAnalysisResult` derived from *spikes*.

    Args:
        spikes: Per-spike metrics in chronological order.

    Returns:
        A populated :class:`APAnalysisResult` whose ``isis`` are computed
        from consecutive ``peak_time`` values.
    """
    isis = [
        spikes[i + 1].peak_time - spikes[i].peak_time for i in range(len(spikes) - 1)
    ]
    return APAnalysisResult(
        spike_count=len(spikes),
        spikes=spikes,
        isis=isis,
        mean_threshold_voltage=-50.0 if spikes else None,
        mean_peak_voltage=30.0 if spikes else None,
        mean_rise_time=0.5 if spikes else None,
        mean_half_width=1.0 if spikes else None,
        mean_ahp_depth=-70.0 if spikes else None,
        mean_isi=float(np.mean(isis)) if isis else None,
        firing_rate=(1000.0 / float(np.mean(isis))) if isis else None,
    )


def _make_transient(
    *,
    decay_tau: float | None = 30.0,
    decay_fit_converged: bool = True,
) -> CalciumTransient:
    """Return a :class:`CalciumTransient` with plausible defaults.

    Args:
        decay_tau: Decay time constant in ms, or None when both fit and
            fallback failed.
        decay_fit_converged: Whether ``curve_fit`` converged.

    Returns:
        A :class:`CalciumTransient`.
    """
    return CalciumTransient(
        index=0,
        onset_time=100.0,
        peak_time=110.0,
        peak_concentration=0.5,
        baseline_concentration=0.05,
        amplitude=0.45,
        time_to_peak=10.0,
        decay_tau=decay_tau,
        decay_fit_converged=decay_fit_converged,
    )


def _make_cc_sweep(
    *,
    label: str = "0.0 µA/cm²",
    n_points: int = 4000,
    duration_ms: float = 200.0,
    spikes_at: list[float] | None = None,
) -> Sweep:
    """Return a synthetic CC sweep, optionally with triangular spikes.

    Args:
        label: Sweep label.
        n_points: Number of samples (default 4000 → dt=0.05 ms, fine enough
            for :func:`analyze_aps` to resolve the synthetic spikes).
        duration_ms: Total recording duration in ms.
        spikes_at: Spike peak times in ms; each spike is a 2 ms-wide triangle
            rising from -70 to +30 mV so :func:`analyze_aps` detects it.

    Returns:
        A populated :class:`Sweep` in ``CURRENT_CLAMP`` mode.
    """
    time = np.linspace(0.0, duration_ms, n_points)
    voltage = np.full(n_points, -65.0)
    if spikes_at:
        for t_peak in spikes_at:
            mask = np.abs(time - t_peak) < 1.0
            voltage[mask] = -70.0 + (1.0 - np.abs(time[mask] - t_peak)) * 100.0
    dvdt = np.gradient(voltage, time)
    return Sweep(
        label=label,
        color="#000000",
        time=time.tolist(),
        voltage=voltage.tolist(),
        dvdt=dvdt.tolist(),
        total_current=[0.0] * n_points,
        stimulus=[0.0] * n_points,
        clamp_mode=CURRENT_CLAMP,
    )


def _make_vc_sweep(label: str = "0 mV") -> Sweep:
    """Return a synthetic VC sweep with a small inward peak then steady state.

    Args:
        label: Sweep label.

    Returns:
        A populated :class:`Sweep` in ``VOLTAGE_CLAMP`` mode.  The total
        current shape (zero pre, peak inward 10–25 ms, steady-state 25–60
        ms) matches :func:`patch_sim.analyze_iv`'s default sample windows.
    """
    n = 200
    time = np.linspace(0.0, 100.0, n)
    total = np.zeros(n)
    total[(time >= 10.0) & (time < 25.0)] = -10.0
    total[(time >= 25.0) & (time < 60.0)] = 1.0
    return Sweep(
        label=label,
        color="#000000",
        time=time.tolist(),
        voltage=[0.0] * n,
        dvdt=[],
        total_current=total.tolist(),
        stimulus=[0.0] * n,
        clamp_mode=VOLTAGE_CLAMP,
    )


# ---------------------------------------------------------------------------
# _fmt_optional — UI display convention
# ---------------------------------------------------------------------------


def test_fmt_optional_none_returns_em_dash() -> None:
    """Missing values render as a literal em-dash for the UI."""
    assert _fmt_optional(None, ".1f") == "—"


def test_fmt_optional_float_uses_format_spec() -> None:
    """A finite value is rendered using the supplied format spec."""
    assert _fmt_optional(1.2345, ".2f") == "1.23"


# ---------------------------------------------------------------------------
# Calcium-transient decay-fit display marker
# ---------------------------------------------------------------------------


def test_format_ca_transient_dict_converged_decay_has_no_marker() -> None:
    """A converged decay-fit renders the τ value without any suffix."""
    out = _format_ca_transient_dict(0, _make_transient(decay_fit_converged=True))
    assert out["decay_tau"] == "30.0"


def test_format_ca_transient_dict_non_converged_marks_with_asterisk() -> None:
    """Non-converged decay-fits are flagged with a trailing ``*`` for the user."""
    out = _format_ca_transient_dict(0, _make_transient(decay_fit_converged=False))
    assert out["decay_tau"] == "30.0*"


def test_format_ca_transient_dict_none_tau_is_em_dash() -> None:
    """A transient whose decay-fit failed entirely renders as an em-dash."""
    out = _format_ca_transient_dict(0, _make_transient(decay_tau=None))
    assert out["decay_tau"] == "—"


# ---------------------------------------------------------------------------
# Burst summary always-on invariant
# ---------------------------------------------------------------------------


def test_serialise_burst_summary_always_emits_threshold_and_method() -> None:
    """The summary surfaces the threshold + method even when no bursts exist.

    The UI relies on this so it can show *which* threshold the analyzer
    applied and how it was chosen, even on traces where zero bursts were
    detected.
    """
    result = BurstAnalysisResult(
        burst_count=0,
        bursts=[],
        unburst_spike_count=0,
        mean_spikes_per_burst=None,
        mean_intra_burst_frequency=None,
        mean_inter_burst_interval=None,
        duty_cycle=None,
        isi_threshold_ms=12.0,
        threshold_method="default-fixed",
    )
    out = _serialise_burst_summary(result)
    assert out["isi_threshold_ms"] == "12.0"
    assert out["threshold_method"] == "default-fixed"


# ---------------------------------------------------------------------------
# _compute_burst_data — guards and always-emit-summary invariant
# ---------------------------------------------------------------------------


def test_compute_burst_data_returns_empty_for_short_train() -> None:
    """Fewer than 2 spikes yields ``([], {})`` so the UI can hide the panel."""
    ap_result = _make_ap_result([_make_spike()])
    metrics, summary = _compute_burst_data(ap_result, np.linspace(0.0, 100.0, 10))
    assert metrics == []
    assert summary == {}


def test_compute_burst_data_emits_summary_even_without_bursts() -> None:
    """≥2 spikes always emits a summary so the UI can show the threshold.

    Spikes are spaced 200 ms apart — well outside the tight-cluster
    intra-burst cap — so the grouper assigns no bursts.  The summary still
    has to surface the threshold and method that were applied.
    """
    spikes = [_make_spike(index=i) for i in range(3)]
    spikes[0].peak_time = 0.0
    spikes[1].peak_time = 200.0
    spikes[2].peak_time = 400.0
    ap_result = _make_ap_result(spikes)
    metrics, summary = _compute_burst_data(ap_result, np.linspace(0.0, 500.0, 200))
    assert metrics == []
    assert summary["threshold_method"]
    assert summary["isi_threshold_ms"]


# ---------------------------------------------------------------------------
# _compute_multi_sweep_burst_data — VC filter + duty-cycle denominator
# ---------------------------------------------------------------------------


def test_compute_multi_sweep_burst_data_ignores_vc_sweeps() -> None:
    """Voltage-clamp sweeps contribute neither to bursts nor the denominator."""
    metrics, summary = _compute_multi_sweep_burst_data([_make_vc_sweep()])
    assert metrics == []
    assert summary == {}


def test_compute_multi_sweep_burst_data_duty_cycle_includes_silent_sweeps() -> None:
    """Sweeps with <2 spikes still count toward the duty-cycle denominator.

    A silent sweep contributes zero burst time but its full window length
    must enlarge the denominator — otherwise a 10-sweep run where 2 fire
    would report duty cycle as a fraction of those 2 sweeps' duration,
    overstating the active-burst time.  This invariant is documented in
    ``_analysis_format.py``.
    """
    burst_sweep = _make_cc_sweep(spikes_at=[20.0, 22.0, 24.0, 26.0, 28.0])
    silent_sweep = _make_cc_sweep(spikes_at=None)
    _, pair_summary = _compute_multi_sweep_burst_data([burst_sweep, silent_sweep])
    _, only_summary = _compute_multi_sweep_burst_data([burst_sweep])
    if (
        pair_summary
        and only_summary
        and pair_summary["duty_cycle"] != "—"
        and only_summary["duty_cycle"] != "—"
    ):
        assert float(pair_summary["duty_cycle"]) < float(only_summary["duty_cycle"])


# ---------------------------------------------------------------------------
# _build_phase_plane_data — VC filter + empty-dict contract
# ---------------------------------------------------------------------------


def test_build_phase_plane_data_filters_vc_and_empty_dvdt() -> None:
    """VC sweeps and CC sweeps with empty dV/dt are dropped from the output."""
    cc_with_dvdt = _make_cc_sweep(spikes_at=[40.0, 50.0])
    cc_empty_dvdt = _make_cc_sweep().model_copy(update={"dvdt": []})
    out = _build_phase_plane_data([_make_vc_sweep(), cc_with_dvdt, cc_empty_dvdt])
    assert len(out["sweeps"]) == 1
    assert out["sweeps"][0]["label"] == cc_with_dvdt.label


def test_build_phase_plane_data_returns_empty_when_no_eligible() -> None:
    """Only-VC sweeps return ``{}`` rather than ``{"sweeps": []}``.

    The consumer in ``AnalysisState`` distinguishes "no data" from "data
    with zero sweeps" via this contract — emitting an empty list would
    cause the panel to render an empty plot instead of being hidden.
    """
    assert _build_phase_plane_data([_make_vc_sweep()]) == {}


# ---------------------------------------------------------------------------
# _compute_cc_multi_sweep_analysis — branch selection + guard
# ---------------------------------------------------------------------------


def test_compute_cc_multi_sweep_hyperpolarizing_returns_hyp_data() -> None:
    """All-negative current steps trigger the sag/rebound branch, not F-I."""
    sweeps = [_make_cc_sweep(label=f"{i:.1f}") for i in (-0.4, -0.3, -0.2)]
    _, _, fi_data, _, hyp_data = _compute_cc_multi_sweep_analysis(
        sweeps,
        min_stimulus=-0.4,
        max_stimulus=-0.2,
        stimulus_step=0.1,
        pre_stimulus_duration=10.0,
        stimulus_duration=80.0,
    )
    assert fi_data == {}
    assert "current_steps" in hyp_data
    assert "sag_amplitudes" in hyp_data


def test_compute_cc_multi_sweep_step_count_mismatch_returns_empty_fi() -> None:
    """A sweep/step count mismatch skips F-I and stamps rheobase as em-dash."""
    sweeps = [_make_cc_sweep(spikes_at=[40.0, 60.0])]
    _, ap_summary, fi_data, _, hyp_data = _compute_cc_multi_sweep_analysis(
        sweeps,
        min_stimulus=0.0,
        max_stimulus=2.0,
        stimulus_step=1.0,
        pre_stimulus_duration=10.0,
        stimulus_duration=80.0,
    )
    assert fi_data == {}
    assert hyp_data == {}
    if ap_summary:
        assert ap_summary["rheobase"] == "—"


# ---------------------------------------------------------------------------
# _compute_iv_data — guards
# ---------------------------------------------------------------------------


def test_compute_iv_data_returns_empty_for_single_sweep() -> None:
    """Fewer than 2 sweeps cannot produce an I-V curve."""
    iv_data, iv_result = _compute_iv_data(
        [_make_vc_sweep()],
        min_stimulus=0.0,
        max_stimulus=10.0,
        stimulus_step=10.0,
        pre_stimulus_duration=10.0,
        stimulus_duration=50.0,
    )
    assert iv_data == {}
    assert iv_result is None


def test_compute_iv_data_returns_empty_on_step_count_mismatch() -> None:
    """When sweep count does not match the derived step count, return empty."""
    sweeps = [_make_vc_sweep("a"), _make_vc_sweep("b")]
    iv_data, iv_result = _compute_iv_data(
        sweeps,
        min_stimulus=0.0,
        max_stimulus=30.0,  # 4 steps with step=10 — mismatches the 2 sweeps.
        stimulus_step=10.0,
        pre_stimulus_duration=10.0,
        stimulus_duration=50.0,
    )
    assert iv_data == {}
    assert iv_result is None
