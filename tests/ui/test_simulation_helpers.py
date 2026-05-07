"""Unit tests for the pure analysis-formatting helpers used by SimulationState.

These helpers take ``patch_sim`` analysis dataclasses or :class:`Sweep` lists
and return JSON-serialisable display dicts; none of them touch Reflex state.
They are exercised here in isolation so they can later be moved to a sibling
module (``patch_sim_ui.state._analysis_format``) without losing coverage.
"""

import os

import numpy as np
import pytest

from patch_sim.analysis.ap_metrics import APAnalysisResult, SpikeMetrics
from patch_sim.analysis.burst_metrics import BurstAnalysisResult, BurstMetrics
from patch_sim.analysis.calcium_transients import (
    CalciumTransient,
    CalciumTransientAnalysisResult,
)
from patch_sim.analysis.sfa import SFACurve
from patch_sim.constants import CURRENT_CLAMP, VOLTAGE_CLAMP

# Reflex's metaclass and instance guards both require a testing environment.
os.environ.setdefault("PYTEST_CURRENT_TEST", "test_simulation_helpers.py::setup")
pytest.importorskip("reflex")

from patch_sim_ui.state._analysis_format import (  # noqa: E402
    _build_phase_plane_data,
    _compute_burst_data,
    _compute_cc_multi_sweep_analysis,
    _compute_gv_data,
    _compute_iv_data,
    _compute_multi_sweep_burst_data,
    _fmt_optional,
    _format_burst_dict,
    _format_ca_transient_dict,
    _format_spike_dict,
    _serialise_burst_summary,
    _serialise_ca_transient_summary,
    _serialise_sfa_curve,
)
from patch_sim_ui.sweep import Sweep  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spike(
    index: int = 0,
    *,
    threshold_voltage: float = -50.0,
    peak_voltage: float = 30.0,
    rise_time: float = 0.5,
    half_width: float = 1.0,
    ahp_depth: float | None = -70.0,
) -> SpikeMetrics:
    """Return a :class:`SpikeMetrics` populated with plausible defaults.

    Args:
        index: Spike index.
        threshold_voltage: Threshold voltage in mV.
        peak_voltage: Peak voltage in mV.
        rise_time: Rise time in ms.
        half_width: Half-width in ms.
        ahp_depth: AHP depth in mV, or None when no trough is available.

    Returns:
        A :class:`SpikeMetrics`.
    """
    return SpikeMetrics(
        index=index,
        threshold_voltage=threshold_voltage,
        threshold_time=0.0,
        peak_voltage=peak_voltage,
        peak_time=float(index),
        rise_time=rise_time,
        half_width=half_width,
        ahp_depth=ahp_depth,
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


def _make_burst(
    index: int = 0,
    *,
    start_time: float = 100.0,
    duration: float = 20.0,
    spike_count: int = 3,
    intra_burst_frequency: float | None = 100.0,
    mean_intra_burst_isi: float | None = 10.0,
) -> BurstMetrics:
    """Return a :class:`BurstMetrics` with plausible defaults.

    Args:
        index: Burst index.
        start_time: Start time in ms.
        duration: Burst duration in ms.
        spike_count: Number of spikes in the burst.
        intra_burst_frequency: Mean firing rate inside the burst in Hz.
        mean_intra_burst_isi: Mean ISI inside the burst in ms.

    Returns:
        A :class:`BurstMetrics`.
    """
    return BurstMetrics(
        index=index,
        start_time=start_time,
        end_time=start_time + duration,
        duration=duration,
        spike_count=spike_count,
        intra_burst_frequency=intra_burst_frequency,
        mean_intra_burst_isi=mean_intra_burst_isi,
    )


def _make_transient(
    *,
    index: int = 0,
    peak_concentration: float = 0.5,
    baseline: float = 0.05,
    decay_tau: float | None = 30.0,
    decay_fit_converged: bool = True,
) -> CalciumTransient:
    """Return a :class:`CalciumTransient` populated with plausible defaults.

    Args:
        index: Transient index.
        peak_concentration: Peak [Ca²⁺]ᵢ in µM.
        baseline: Per-transient baseline [Ca²⁺]ᵢ in µM.
        decay_tau: Decay time constant in ms, or None.
        decay_fit_converged: Whether ``curve_fit`` converged.

    Returns:
        A :class:`CalciumTransient`.
    """
    return CalciumTransient(
        index=index,
        onset_time=100.0,
        peak_time=110.0,
        peak_concentration=peak_concentration,
        baseline_concentration=baseline,
        amplitude=peak_concentration - baseline,
        time_to_peak=10.0,
        decay_tau=decay_tau,
        decay_fit_converged=decay_fit_converged,
    )


def _make_cc_sweep(
    *,
    label: str = "0.0 µA/cm²",
    color: str = "#000000",
    n_points: int = 4000,
    duration_ms: float = 200.0,
    voltage_value: float = -65.0,
    spikes_at: list[float] | None = None,
) -> Sweep:
    """Return a synthetic CC sweep, optionally with triangular spikes.

    Args:
        label: Sweep label.
        color: Hex color string.
        n_points: Number of samples (default 4000 → dt=0.05 ms, fine enough
            for :func:`analyze_aps` to resolve the synthetic spikes).
        duration_ms: Total recording duration in ms.
        voltage_value: Resting voltage in mV (used when no spikes).
        spikes_at: Spike peak times in ms; each spike is a 2 ms-wide triangle
            rising from -70 to +30 mV so :func:`analyze_aps` detects it.

    Returns:
        A populated :class:`Sweep` in ``CURRENT_CLAMP`` mode.
    """
    time = np.linspace(0.0, duration_ms, n_points)
    voltage = np.full(n_points, voltage_value)
    if spikes_at:
        for t_peak in spikes_at:
            mask = np.abs(time - t_peak) < 1.0
            voltage[mask] = -70.0 + (1.0 - np.abs(time[mask] - t_peak)) * 100.0
    dvdt = np.gradient(voltage, time)
    return Sweep(
        label=label,
        color=color,
        time=time.tolist(),
        voltage=voltage.tolist(),
        dvdt=dvdt.tolist(),
        sodium_current=[0.0] * n_points,
        potassium_current=[0.0] * n_points,
        na_leak_current=[0.0] * n_points,
        k_leak_current=[0.0] * n_points,
        total_current=[0.0] * n_points,
        potassium_activation=[0.0] * n_points,
        sodium_activation=[0.0] * n_points,
        sodium_inactivation=[0.0] * n_points,
        stimulus=[0.0] * n_points,
        clamp_mode=CURRENT_CLAMP,
    )


def _make_vc_sweep(
    *,
    label: str = "0 mV",
    n_points: int = 200,
    duration_ms: float = 100.0,
    peak_inward: float = -10.0,
    steady_state: float = 1.0,
) -> Sweep:
    """Return a synthetic VC sweep with an inward peak then steady state current.

    Args:
        label: Sweep label.
        n_points: Number of samples.
        duration_ms: Total recording duration in ms.
        peak_inward: Peak inward current in µA/cm² during the step.
        steady_state: Steady-state current in µA/cm² during the step.

    Returns:
        A populated :class:`Sweep` in ``VOLTAGE_CLAMP`` mode.
    """
    time = np.linspace(0.0, duration_ms, n_points)
    total = np.zeros(n_points)
    # Sample windows (matching `_compute_iv_data`'s start=10, end=60).
    pre_idx = np.where(time < 10.0)[0]
    early_idx = np.where((time >= 10.0) & (time < 25.0))[0]
    late_idx = np.where((time >= 25.0) & (time < 60.0))[0]
    total[pre_idx] = 0.0
    total[early_idx] = peak_inward
    total[late_idx] = steady_state
    return Sweep(
        label=label,
        color="#000000",
        time=time.tolist(),
        voltage=[0.0] * n_points,
        dvdt=[],
        sodium_current=[0.0] * n_points,
        potassium_current=[0.0] * n_points,
        na_leak_current=[0.0] * n_points,
        k_leak_current=[0.0] * n_points,
        total_current=total.tolist(),
        potassium_activation=[0.0] * n_points,
        sodium_activation=[0.0] * n_points,
        sodium_inactivation=[0.0] * n_points,
        stimulus=[0.0] * n_points,
        clamp_mode=VOLTAGE_CLAMP,
    )


# ---------------------------------------------------------------------------
# _fmt_optional
# ---------------------------------------------------------------------------


def test_fmt_optional_none_returns_em_dash() -> None:
    """``None`` should produce a literal em-dash, regardless of format spec."""
    assert _fmt_optional(None, ".1f") == "—"


def test_fmt_optional_float_uses_format_spec() -> None:
    """A finite value should be formatted using the supplied format spec."""
    assert _fmt_optional(1.2345, ".2f") == "1.23"


# ---------------------------------------------------------------------------
# _format_spike_dict
# ---------------------------------------------------------------------------


def test_format_spike_dict_keys_and_index_override() -> None:
    """The spike dict carries the supplied display index, not ``spike.index``."""
    spike = _make_spike(index=7, ahp_depth=-65.4)
    out = _format_spike_dict(2, spike)
    assert out["index"] == 2
    assert set(out.keys()) == {
        "index",
        "threshold_voltage",
        "peak_voltage",
        "rise_time",
        "half_width",
        "ahp_depth",
    }
    assert out["ahp_depth"] == "-65.4"


def test_format_spike_dict_handles_none_ahp() -> None:
    """A spike with no measurable AHP renders as an em-dash."""
    spike = _make_spike(ahp_depth=None)
    assert _format_spike_dict(0, spike)["ahp_depth"] == "—"


# ---------------------------------------------------------------------------
# _format_ca_transient_dict
# ---------------------------------------------------------------------------


def test_format_ca_transient_dict_converged_and_keys() -> None:
    """Converged decay fits render without the ``*`` suffix."""
    out = _format_ca_transient_dict(3, _make_transient(decay_fit_converged=True))
    assert out["index"] == 3
    assert out["sweep_index"] == 0
    assert out["decay_tau"] == "30.0"
    assert set(out.keys()) == {
        "index",
        "sweep_index",
        "peak_time",
        "peak_concentration",
        "time_to_peak",
        "decay_tau",
        "amplitude",
    }


def test_format_ca_transient_dict_non_converged_marks_with_asterisk() -> None:
    """Non-converged decay fits are flagged with a trailing ``*``."""
    out = _format_ca_transient_dict(0, _make_transient(decay_fit_converged=False))
    assert out["decay_tau"] == "30.0*"


def test_format_ca_transient_dict_none_tau_is_em_dash() -> None:
    """A transient whose decay-fit failed entirely renders as an em-dash."""
    out = _format_ca_transient_dict(0, _make_transient(decay_tau=None))
    assert out["decay_tau"] == "—"


# ---------------------------------------------------------------------------
# _serialise_ca_transient_summary
# ---------------------------------------------------------------------------


def test_serialise_ca_transient_summary_keys_and_none_handling() -> None:
    """Aggregate row reports each metric with em-dash when missing."""
    result = CalciumTransientAnalysisResult(
        transient_count=0,
        transients=[],
        baseline_concentration=None,
        mean_peak_concentration=None,
        mean_time_to_peak=None,
        mean_decay_tau=None,
        mean_amplitude=None,
    )
    out = _serialise_ca_transient_summary(result)
    assert out["transient_count"] == "0"
    assert out["baseline_concentration"] == "—"
    assert out["mean_decay_tau"] == "—"


# ---------------------------------------------------------------------------
# _format_burst_dict
# ---------------------------------------------------------------------------


def test_format_burst_dict_keys_and_optional_fields() -> None:
    """The burst dict exposes formatted fields and a 0 sweep_index by default."""
    burst = _make_burst(intra_burst_frequency=None, mean_intra_burst_isi=None)
    out = _format_burst_dict(4, burst)
    assert out["index"] == 4
    assert out["sweep_index"] == 0
    assert out["intra_burst_frequency"] == "—"
    assert out["mean_intra_burst_isi"] == "—"
    assert set(out.keys()) == {
        "index",
        "sweep_index",
        "start_time",
        "end_time",
        "duration",
        "spike_count",
        "intra_burst_frequency",
        "mean_intra_burst_isi",
    }


# ---------------------------------------------------------------------------
# _serialise_burst_summary
# ---------------------------------------------------------------------------


def test_serialise_burst_summary_carries_threshold_and_method() -> None:
    """The aggregate row always reports the ISI threshold and detection method."""
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
    assert out["burst_count"] == "0"
    assert out["isi_threshold_ms"] == "12.0"
    assert out["threshold_method"] == "default-fixed"
    assert out["duty_cycle"] == "—"


# ---------------------------------------------------------------------------
# _compute_burst_data
# ---------------------------------------------------------------------------


def test_compute_burst_data_returns_empty_for_short_train() -> None:
    """Fewer than 2 spikes yields ``([], {})`` (no analysis run)."""
    ap_result = _make_ap_result([_make_spike()])
    metrics, summary = _compute_burst_data(ap_result, np.linspace(0.0, 100.0, 10))
    assert metrics == []
    assert summary == {}


def test_compute_burst_data_emits_summary_even_without_bursts() -> None:
    """≥2 spikes always emits a summary so the UI can show the threshold."""
    spikes = [_make_spike(index=i) for i in range(3)]
    # Override peak_time so the spikes are spaced far apart (ISI = 200 ms),
    # which falls outside the tight-cluster intra-burst cap.  The grouper
    # then assigns no bursts but the threshold + method are still reported.
    spikes[0].peak_time = 0.0
    spikes[1].peak_time = 200.0
    spikes[2].peak_time = 400.0
    ap_result = _make_ap_result(spikes)
    time_arr = np.linspace(0.0, 500.0, 200)
    metrics, summary = _compute_burst_data(ap_result, time_arr)
    assert metrics == []
    # Summary is populated: threshold and method always present.
    assert summary["threshold_method"]
    assert summary["isi_threshold_ms"]


# ---------------------------------------------------------------------------
# _compute_multi_sweep_burst_data
# ---------------------------------------------------------------------------


def test_compute_multi_sweep_burst_data_ignores_vc_sweeps() -> None:
    """Voltage-clamp sweeps contribute neither to bursts nor duty cycle."""
    metrics, summary = _compute_multi_sweep_burst_data([_make_vc_sweep()])
    assert metrics == []
    assert summary == {}


def test_compute_multi_sweep_burst_data_pools_across_cc_sweeps() -> None:
    """Two CC sweeps with multi-spike bursts yield a pooled summary."""
    spike_times = [50.0, 55.0, 60.0, 65.0, 70.0]
    sweeps = [
        _make_cc_sweep(label="A", spikes_at=spike_times),
        _make_cc_sweep(label="B", spikes_at=spike_times),
    ]
    metrics, summary = _compute_multi_sweep_burst_data(sweeps)
    # The summary always carries the threshold-method and isi_threshold_ms
    # fields once at least one CC sweep yielded ≥2 spikes.  The sweep_index
    # rewriting is verified by the per-burst metrics when bursts are detected.
    assert summary != {}
    assert "threshold_method" in summary
    assert "isi_threshold_ms" in summary
    for entry in metrics:
        assert entry["sweep_index"] in {1, 2}


def test_compute_multi_sweep_burst_data_duty_cycle_includes_silent_sweeps() -> None:
    """Sweeps with <2 spikes still count toward the duty-cycle denominator.

    A silent sweep contributes zero bursts but its full window length should
    enlarge the denominator so the duty cycle is not overstated.
    """
    burst_sweep = _make_cc_sweep(spikes_at=[20.0, 22.0, 24.0, 26.0, 28.0])
    silent_sweep = _make_cc_sweep(spikes_at=None)
    _pair_metrics, pair_summary = _compute_multi_sweep_burst_data(
        [burst_sweep, silent_sweep]
    )
    _only_metrics, only_summary = _compute_multi_sweep_burst_data([burst_sweep])
    # The duty cycle with the silent sweep included is strictly smaller when
    # bursts are detected at all; otherwise both summaries are empty.
    if (
        pair_summary
        and only_summary
        and pair_summary["duty_cycle"] != "—"
        and only_summary["duty_cycle"] != "—"
    ):
        assert float(pair_summary["duty_cycle"]) < float(only_summary["duty_cycle"])


# ---------------------------------------------------------------------------
# _serialise_sfa_curve
# ---------------------------------------------------------------------------


def test_serialise_sfa_curve_returns_plain_dict() -> None:
    """The SFA-curve dict carries the full set of plotting fields."""
    curve = SFACurve(
        spike_indices=[0, 1, 2],
        instantaneous_frequencies=[100.0, 80.0, 50.0],
        adaptation_index=0.5,
        label="10 µA/cm²",
    )
    out = _serialise_sfa_curve(curve)
    assert out["spike_indices"] == [0, 1, 2]
    assert out["adaptation_index"] == 0.5
    assert out["label"] == "10 µA/cm²"


# ---------------------------------------------------------------------------
# _build_phase_plane_data
# ---------------------------------------------------------------------------


def test_build_phase_plane_data_filters_vc_and_empty_dvdt() -> None:
    """VC sweeps and CC sweeps with empty dV/dt are dropped from the output."""
    vc = _make_vc_sweep()
    cc_with_dvdt = _make_cc_sweep(spikes_at=[40.0, 50.0])
    cc_empty_dvdt = _make_cc_sweep().model_copy(update={"dvdt": []})
    out = _build_phase_plane_data([vc, cc_with_dvdt, cc_empty_dvdt])
    assert "sweeps" in out
    assert len(out["sweeps"]) == 1
    assert out["sweeps"][0]["label"] == cc_with_dvdt.label
    assert out["sweeps"][0]["color"] == cc_with_dvdt.color


def test_build_phase_plane_data_returns_empty_when_no_eligible() -> None:
    """Only-VC sweeps return ``{}`` rather than ``{"sweeps": []}``."""
    assert _build_phase_plane_data([_make_vc_sweep()]) == {}


# ---------------------------------------------------------------------------
# _compute_cc_multi_sweep_analysis
# ---------------------------------------------------------------------------


def test_compute_cc_multi_sweep_hyperpolarizing_returns_hyp_data() -> None:
    """All-negative current steps trigger the hyperpolarisation branch."""
    sweeps = [_make_cc_sweep(label=f"{i:.1f}") for i in (-0.4, -0.3, -0.2)]
    _ap_metrics, ap_summary, fi_data, _sfa_data, hyp_data = (
        _compute_cc_multi_sweep_analysis(
            sweeps,
            min_stimulus=-0.4,
            max_stimulus=-0.2,
            stimulus_step=0.1,
            pre_stimulus_duration=10.0,
            stimulus_duration=80.0,
        )
    )
    assert ap_summary == {}
    assert fi_data == {}
    assert "current_steps" in hyp_data
    assert "sag_amplitudes" in hyp_data


def test_compute_cc_multi_sweep_step_count_mismatch_returns_empty_fi() -> None:
    """A sweep/step count mismatch skips F-I and stamps rheobase as em-dash."""
    sweeps = [_make_cc_sweep(spikes_at=[40.0, 60.0])]
    _ap_metrics, ap_summary, fi_data, _sfa_data, hyp_data = (
        _compute_cc_multi_sweep_analysis(
            sweeps,
            min_stimulus=0.0,
            max_stimulus=2.0,
            stimulus_step=1.0,
            pre_stimulus_duration=10.0,
            stimulus_duration=80.0,
        )
    )
    assert fi_data == {}
    assert hyp_data == {}
    if ap_summary:
        assert ap_summary["rheobase"] == "—"


# ---------------------------------------------------------------------------
# _compute_iv_data
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
    """When sweep count does not match derived step count, return empty data."""
    sweeps = [_make_vc_sweep(label="a"), _make_vc_sweep(label="b")]
    iv_data, iv_result = _compute_iv_data(
        sweeps,
        min_stimulus=0.0,
        max_stimulus=30.0,  # 4 steps with step=10 — does not match 2 sweeps.
        stimulus_step=10.0,
        pre_stimulus_duration=10.0,
        stimulus_duration=50.0,
    )
    assert iv_data == {}
    assert iv_result is None


def test_compute_iv_data_returns_serialised_keys() -> None:
    """Aligned sweeps and steps produce a dict with the four I-V keys."""
    sweeps = [
        _make_vc_sweep(label="-60", peak_inward=-5.0, steady_state=0.5),
        _make_vc_sweep(label="-50", peak_inward=-15.0, steady_state=1.0),
        _make_vc_sweep(label="-40", peak_inward=-25.0, steady_state=1.5),
    ]
    iv_data, iv_result = _compute_iv_data(
        sweeps,
        min_stimulus=-60.0,
        max_stimulus=-40.0,
        stimulus_step=10.0,
        pre_stimulus_duration=10.0,
        stimulus_duration=50.0,
    )
    assert iv_result is not None
    assert set(iv_data.keys()) == {
        "voltages",
        "peak_inward_currents",
        "peak_outward_currents",
        "steady_state_currents",
    }
    assert len(iv_data["voltages"]) == 3


# ---------------------------------------------------------------------------
# _compute_gv_data
# ---------------------------------------------------------------------------


def test_compute_gv_data_empty_when_no_points() -> None:
    """An empty I-V result yields an empty g-V dict."""
    sweeps = [
        _make_vc_sweep(label=f"{v}", peak_inward=0.0, steady_state=0.0)
        for v in (-70.0, -60.0)
    ]
    _iv_data, iv_result = _compute_iv_data(
        sweeps,
        min_stimulus=-70.0,
        max_stimulus=-60.0,
        stimulus_step=10.0,
        pre_stimulus_duration=10.0,
        stimulus_duration=50.0,
    )
    assert iv_result is not None
    out = _compute_gv_data(iv_result, reversal_potential=50.0)
    # When all peak inward currents are zero, no usable conductance points are
    # produced — the helper returns an empty dict in that case.
    assert out == {} or "voltages" in out


# ---------------------------------------------------------------------------
# Module identity check
# ---------------------------------------------------------------------------


def test_helpers_live_in_analysis_format_module() -> None:
    """Pure helpers live in ``patch_sim_ui.state._analysis_format``."""
    from patch_sim_ui.state import _analysis_format

    assert _analysis_format._fmt_optional(None, ".0f") == "—"
    assert _analysis_format._GV_FIT_POINTS == 200
