"""Synchronous sweep executor for SimulationState.

Owns ``_SimResult`` (the data carrier) and ``_compute_simulation`` (the loop
that runs single- or multi-sweep current/voltage clamp simulations and
dispatches all post-hoc analysis).  Designed for thread-executor use: no
Reflex state mutation, no UI dependencies beyond figure building and the
side-channel trace store.
"""

import dataclasses
import uuid
from typing import Any

import numpy as np

import patch_sim
import patch_sim.channels
from patch_sim.constants import CURRENT_CLAMP, VOLTAGE_CLAMP
from patch_sim_ui import constants
from patch_sim_ui.api import traces
from patch_sim_ui.plotting import TraceVisibility, build_figure
from patch_sim_ui.state._analysis_format import (
    _build_phase_plane_data,
    _compute_burst_data,
    _compute_ca_transient_data,
    _compute_cc_multi_sweep_analysis,
    _compute_gv_data,
    _compute_iv_data,
    _compute_multi_sweep_burst_data,
    _compute_multi_sweep_ca_transient_data,
    _fmt_optional,
    _format_spike_dict,
    _serialise_sfa_curve,
)
from patch_sim_ui.sweep import Sweep


@dataclasses.dataclass(frozen=True)
class _SimResult:
    """Output of a complete simulation run, ready to apply to state.

    Produced by :func:`_compute_simulation` and consumed by
    :meth:`SimulationState._do_apply_simulation`.  All fields have
    empty defaults so callers only set what is relevant for the current
    clamp mode.
    """

    sweeps: list[Sweep]
    sim_token: str
    iv_data: dict[str, Any] = dataclasses.field(default_factory=dict)
    gv_data: dict[str, Any] = dataclasses.field(default_factory=dict)
    ap_metrics: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    ap_summary: dict[str, Any] = dataclasses.field(default_factory=dict)
    ap_is_multi_sweep: bool = False
    ca_transient_metrics: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    ca_transient_summary: dict[str, Any] = dataclasses.field(default_factory=dict)
    burst_metrics: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    burst_summary: dict[str, Any] = dataclasses.field(default_factory=dict)
    fi_data: dict[str, Any] = dataclasses.field(default_factory=dict)
    sfa_data: dict[str, Any] = dataclasses.field(default_factory=dict)
    hyperpolarization_data: dict[str, Any] = dataclasses.field(default_factory=dict)
    phase_plane_data: dict[str, Any] = dataclasses.field(default_factory=dict)


def _compute_simulation(
    neuron: "patch_sim.Neuron",
    protocols: "list[tuple[np.ndarray, str]]",
    mode: str,
    stored_traces: "list[Sweep]",
    show_hover: bool,
    min_stimulus: float,
    max_stimulus: float,
    stimulus_step: float,
    pre_stimulus_duration: float,
    stimulus_duration: float,
) -> _SimResult:
    """Run the simulation synchronously and compute all analysis.

    Designed to be called via ``run_in_executor`` in production (no state
    mutation) and directly in tests.  Raises :exc:`ValueError` on invalid
    protocol parameters so the caller can propagate the error to
    :attr:`~SimulationState.error_message`.

    Args:
        neuron: Built neuron model.
        protocols: List of ``(stimulus_array, label)`` tuples from
            :meth:`~patch_sim_ui.state.protocol.ProtocolState._build_protocols`.
        mode: ``"Current Clamp"`` or ``"Voltage Clamp"``.
        stored_traces: Snapshot of current stored traces for figure building.
        show_hover: Whether hover tooltips are enabled.
        min_stimulus: Minimum stimulus value for analysis range.
        max_stimulus: Maximum stimulus value for analysis range.
        stimulus_step: Stimulus step size for analysis range.
        pre_stimulus_duration: Pre-stimulus duration (ms) for analysis windows.
        stimulus_duration: Stimulus duration (ms) for analysis windows.

    Returns:
        A :class:`_SimResult` containing sweeps, figure token, and all
        analysis data ready for :meth:`~SimulationState._do_apply_simulation`.
    """
    sim_fn = (
        patch_sim.simulate_current_clamp
        if mode == CURRENT_CLAMP
        else patch_sim.simulate_voltage_clamp
    )
    is_multi = len(protocols) > 1

    if is_multi:
        new_sweeps: list[Sweep] = []
        for sweep_result, (protocol, label) in zip(
            patch_sim.simulate_batch(neuron, [p for p, _ in protocols], sim_fn),
            protocols,
        ):
            color_index = len(new_sweeps) % len(constants.SWEEP_COLORS)
            new_sweeps.append(
                Sweep.from_result(
                    sweep_result,
                    protocol,
                    label,
                    constants.SWEEP_COLORS[color_index],
                    mode,
                )
            )

        fig = build_figure(
            current_sweeps=new_sweeps,
            visibility=TraceVisibility(),
            clamp_mode=mode,
            stored_traces=stored_traces,
            show_hover=show_hover,
        )
        sim_token = uuid.uuid4().hex
        traces.put(sim_token, fig)

        if mode == VOLTAGE_CLAMP:
            iv_data, iv_result = _compute_iv_data(
                new_sweeps,
                min_stimulus,
                max_stimulus,
                stimulus_step,
                pre_stimulus_duration,
                stimulus_duration,
            )
            if iv_result is not None:
                na_channel = next(
                    (
                        ch
                        for ch in neuron.core_channels
                        if isinstance(ch.reversal_spec, patch_sim.channels.NernstSpec)
                        and ch.reversal_spec.species
                        is patch_sim.channels.IonSpecies.SODIUM
                    ),
                    None,
                )
                gv_data = (
                    _compute_gv_data(iv_result, na_channel.reversal_potential(neuron))
                    if na_channel is not None
                    else {}
                )
            else:
                gv_data = {}
            ms_ca_metrics, ms_ca_summary = _compute_multi_sweep_ca_transient_data(
                new_sweeps
            )
            return _SimResult(
                sweeps=new_sweeps,
                sim_token=sim_token,
                iv_data=iv_data,
                gv_data=gv_data,
                ca_transient_metrics=ms_ca_metrics,
                ca_transient_summary=ms_ca_summary,
            )

        ms_metrics, ms_summary, ms_fi, ms_sfa, ms_hyp = (
            _compute_cc_multi_sweep_analysis(
                new_sweeps,
                min_stimulus,
                max_stimulus,
                stimulus_step,
                pre_stimulus_duration,
                stimulus_duration,
            )
        )
        ms_ca_metrics, ms_ca_summary = _compute_multi_sweep_ca_transient_data(
            new_sweeps
        )
        ms_burst_metrics, ms_burst_summary = _compute_multi_sweep_burst_data(new_sweeps)
        return _SimResult(
            sweeps=new_sweeps,
            sim_token=sim_token,
            ap_metrics=ms_metrics,
            ap_summary=ms_summary,
            ap_is_multi_sweep=True,
            ca_transient_metrics=ms_ca_metrics,
            ca_transient_summary=ms_ca_summary,
            burst_metrics=ms_burst_metrics,
            burst_summary=ms_burst_summary,
            fi_data=ms_fi,
            sfa_data=ms_sfa,
            hyperpolarization_data=ms_hyp,
            phase_plane_data=_build_phase_plane_data(new_sweeps),
        )

    # Single sweep
    stimulus, _ = protocols[0]
    result = sim_fn(neuron, stimulus)
    sweep = Sweep.from_result(result, stimulus, "", "", mode)

    fig = build_figure(
        current_sweeps=[sweep],
        visibility=TraceVisibility(),
        clamp_mode=mode,
        stored_traces=stored_traces,
        show_hover=show_hover,
    )
    sim_token = uuid.uuid4().hex
    traces.put(sim_token, fig)

    if mode == CURRENT_CLAMP:
        ap_result = patch_sim.analyze_aps_from_result(result)
        sfa_curve = patch_sim.compute_sfa(ap_result)
        ca_metrics, ca_summary = _compute_ca_transient_data(result)
        burst_metrics, burst_summary = _compute_burst_data(
            ap_result, np.asarray(result["time"])
        )
        return _SimResult(
            sweeps=[sweep],
            sim_token=sim_token,
            ap_metrics=[_format_spike_dict(s.index, s) for s in ap_result.spikes],
            ap_summary={
                "spike_count": str(ap_result.spike_count),
                "mean_threshold_voltage": _fmt_optional(
                    ap_result.mean_threshold_voltage, ".1f"
                ),
                "mean_peak_voltage": _fmt_optional(ap_result.mean_peak_voltage, ".1f"),
                "mean_rise_time": _fmt_optional(ap_result.mean_rise_time, ".2f"),
                "mean_half_width": _fmt_optional(ap_result.mean_half_width, ".2f"),
                "mean_ahp_depth": _fmt_optional(ap_result.mean_ahp_depth, ".1f"),
                "mean_isi": _fmt_optional(ap_result.mean_isi, ".1f"),
                "firing_rate": _fmt_optional(ap_result.firing_rate, ".1f"),
                "adaptation_index": (
                    f"{sfa_curve.adaptation_index:.2f}"
                    if sfa_curve is not None
                    else "—"
                ),
                "rheobase": (
                    f"≤ {min_stimulus:.2f}" if ap_result.spike_count >= 1 else "—"
                ),
            },
            ca_transient_metrics=ca_metrics,
            ca_transient_summary=ca_summary,
            burst_metrics=burst_metrics,
            burst_summary=burst_summary,
            sfa_data=(
                {"curves": [_serialise_sfa_curve(sfa_curve)]}
                if sfa_curve is not None
                else {}
            ),
            phase_plane_data=_build_phase_plane_data([sweep]),
        )

    # Single VC sweep — IV/GV require multi-sweep, but calcium transients can
    # still appear (e.g. a depolarising VC step opens Ca channels).
    ca_metrics, ca_summary = _compute_ca_transient_data(result)
    return _SimResult(
        sweeps=[sweep],
        sim_token=sim_token,
        ca_transient_metrics=ca_metrics,
        ca_transient_summary=ca_summary,
    )
