"""Simulation state for the patch_sim web UI.

SimulationState owns simulation results, sweep collections, continuous mode,
and the figure computed var.  Cross-cutting state lives in the sibling
substates (NeuronState, ProtocolState, VisibilityState, AnalysisState, LogState).
"""

import asyncio
import logging
import time
from typing import Any, AsyncGenerator

import plotly.graph_objects as go
import reflex as rx

import patch_sim
import patch_sim.clamp_simulations
from patch_sim.constants import (
    CURRENT_CLAMP,
    VOLTAGE_CLAMP,
)
from patch_sim_ui import constants
from patch_sim_ui.plotting import (
    Sweep,
    TraceVisibility,
    build_figure,
    compute_trace_visibility_map,
)
from patch_sim_ui.state._common import (
    _ADDITIONAL_CURRENT_FIELD_MAP,
    _ADDITIONAL_GATING_FIELD_MAP,
    _LOG_SCROLL_JS,
    _PLOTLY_GD_JS,
    _SWEEP_HIGHLIGHT_JS,
    _compute_iv_data,
)
from patch_sim_ui.state.analysis import AnalysisState
from patch_sim_ui.state.log import LogState
from patch_sim_ui.state.neuron import NeuronState
from patch_sim_ui.state.protocol import ProtocolState
from patch_sim_ui.state.visibility import VisibilityState

logger = logging.getLogger("patch_sim_ui.state")


class SimulationState(rx.State):
    """State for simulation results, sweep collections, and figure rendering."""

    # Synced copy of NeuronState.active_neuron_type used by add_sweep /
    # store_trace for labelling (avoids making those handlers async).
    _label_neuron_type: str = "Squid Giant Axon (Classic HH)"
    # Synced copy of ProtocolState.clamp_mode used by figure_data and
    # _apply_visibility_js (both are synchronous, cannot call get_state).
    _figure_clamp_mode: str = CURRENT_CLAMP

    # ------------------------------------------------------------------ #
    # Simulation results                                                  #
    # ------------------------------------------------------------------ #
    current_sweeps: list[Sweep] = []  # Latest simulation result
    saved_sweeps: list[Sweep] = []  # User-saved sweeps for comparison overlay
    stored_traces: list[Sweep] = []  # Oscilloscope-style stored reference traces

    # ------------------------------------------------------------------ #
    # Continuous simulation mode                                        #
    # ------------------------------------------------------------------ #
    continuous_mode: bool = False  # user has toggled continuous on
    continuous_loop_running: bool = False  # background task is executing

    # Terminal neuron state carried between continuous loop iterations.
    # Prefixed with _ by convention; still full Reflex state vars.
    _cont_V: float = 0.0
    _cont_gating: dict[str, float] = {}
    _cont_ca_i: float = 0.0
    _cont_has_state: bool = False  # True once at least one iteration has run

    # ------------------------------------------------------------------ #
    # UI state                                                           #
    # ------------------------------------------------------------------ #
    is_running: bool = False
    error_message: str = ""
    show_hover: bool = True  # Whether plot hover tooltips are visible
    # Index of the last click-selected sweep seeded into the JS on figure
    # rebuild (-1 = none selected).  Selection state is managed entirely
    # client-side by ``window._psSweep``; this field is only written by
    # Python (reset on run/clamp-mode change) so that the correct seed value
    # is injected when the JS module is re-initialised after a figure rebuild.
    #
    # Known limitation: this field is never updated from the client side, so
    # any figure rebuild triggered by a Python state change (e.g. add_sweep)
    # while the user has a client-side selection active will re-seed JS with
    # -1, silently clearing the selection.
    selected_sweep: int = -1

    # ------------------------------------------------------------------ #
    # Analysis sidebar state                                             #
    # ------------------------------------------------------------------ #
    analysis_panel_open: bool = True

    # ------------------------------------------------------------------ #
    # Computed properties                                               #
    # ------------------------------------------------------------------ #
    @rx.var
    def has_result(self) -> bool:
        """Whether a simulation result is available."""
        return len(self.current_sweeps) > 0

    @rx.var
    def has_stored_traces(self) -> bool:
        """Whether any oscilloscope-stored reference traces exist."""
        return len(self.stored_traces) > 0

    @rx.var
    def continuous_active(self) -> bool:
        """True when continuous mode is enabled and the loop is running."""
        return self.continuous_mode and self.continuous_loop_running

    @rx.var
    def figure_data(self) -> go.Figure:
        """Plotly figure rebuilt when sweeps, clamp mode, or hover state change.

        All traces are built with full visibility; toggling show_* flags is
        handled client-side via ``Plotly.restyle`` so that figure rebuilds are
        not triggered by visibility changes.  The ``show_hover`` flag is
        respected here so that hovermode is baked into the figure data and takes
        effect immediately, even without a client-side relayout.

        Dark/light theming is applied client-side by the ``rx.plotly``
        component via its ``layout`` and ``template`` props, so no server-side
        colour mode state is needed.
        """
        return build_figure(
            current_sweeps=self.current_sweeps,
            saved_sweeps=self.saved_sweeps,
            visibility=TraceVisibility(),  # all visible; toggling handled client-side
            clamp_mode=self._figure_clamp_mode,
            stored_traces=self.stored_traces,
            show_hover=self.show_hover,
        )

    # ------------------------------------------------------------------ #
    # Event handlers                                                     #
    # ------------------------------------------------------------------ #
    def _clear_for_new_protocol(self) -> None:
        """Clear all simulation results and reset continuous state.

        Called by ProtocolState when the clamp mode or a protocol preset
        changes, ensuring stale results are not shown for the new protocol.
        """
        self.current_sweeps = []
        self.saved_sweeps = []
        self.stored_traces = []
        self._cont_has_state = False
        self.selected_sweep = -1

    async def reset_to_defaults(self) -> None:
        """Reset all state vars across all substates to their class-level defaults."""
        self.reset()
        neuron_st = await self.get_state(NeuronState)
        neuron_st.reset()
        proto_st = await self.get_state(ProtocolState)
        proto_st.reset()
        vis_st = await self.get_state(VisibilityState)
        vis_st.reset()
        analysis_st = await self.get_state(AnalysisState)
        analysis_st.reset()
        log_st = await self.get_state(LogState)
        log_st.reset()

    def toggle_analysis_panel(self) -> None:
        """Toggle the right-hand analysis sidebar open or closed."""
        self.analysis_panel_open = not self.analysis_panel_open

    def toggle_hover(self):
        """Toggle plot hover tooltips on or off.

        Flips ``show_hover`` and issues a client-side ``Plotly.relayout`` call
        to update the figure's ``hovermode`` without triggering a full rebuild.
        When hover is disabled, ``hovermode`` is set to ``false``.  When
        re-enabled it is restored to ``"x unified"`` for single-sweep traces or
        ``"x"`` for multi-sweep (I-V Curve) results.

        Returns:
            A ``rx.call_script`` event that applies the relayout in-browser.
        """
        self.show_hover = not self.show_hover
        if self.show_hover:
            hovermode = "x" if len(self.current_sweeps) > 1 else "x unified"
            hovermode_js = f'"{hovermode}"'
        else:
            hovermode_js = "false"
        js = (
            f"{_PLOTLY_GD_JS}"
            f"if(gd&&gd.layout)Plotly.relayout(gd,{{hovermode:{hovermode_js}}})"
        )
        return rx.call_script(js)

    # ------------------------------------------------------------------ #
    # Sweep management                                                   #
    # ------------------------------------------------------------------ #
    def _apply_visibility_js(self, vis_st: "VisibilityState") -> str | None:
        """Build a JS snippet to re-apply trace visibility, hover, and sweep highlight.

        Called after any operation that triggers a full figure rebuild (run
        simulation, add sweep, clear sweeps) so that traces the user has
        toggled off are correctly hidden again, the hover mode matches the
        current ``show_hover`` flag, and sweep highlight listeners are
        (re-)attached in multi-sweep mode.

        Args:
            vis_st: Current VisibilityState instance providing show_* values.

        Returns:
            A JS string that re-applies trace visibility, hover mode, and
            sweep highlight, or ``None`` when nothing needs to be applied.
        """
        import json

        trace_map = compute_trace_visibility_map(
            current_sweeps=self.current_sweeps,
            saved_sweeps=self.saved_sweeps,
            clamp_mode=self._figure_clamp_mode,
            additional_current_field_map=_ADDITIONAL_CURRENT_FIELD_MAP,
            additional_gating_field_map=_ADDITIONAL_GATING_FIELD_MAP,
            stored_traces=self.stored_traces,
        )
        hidden: list[int] = []
        for field_name, indices in trace_map.items():
            if not getattr(vis_st, field_name, True):
                hidden.extend(indices)

        parts: list[str] = []
        if hidden:
            parts.append(
                f"if(gd&&gd.data)Plotly.restyle(gd,"
                f"{{visible:false}},{json.dumps(hidden)});"
            )
        if not self.show_hover:
            parts.append("if(gd&&gd.layout)Plotly.relayout(gd,{hovermode:false});")

        # Inject sweep highlight listeners in multi-sweep mode.
        is_multi = len(self.current_sweeps) > 1
        if is_multi:
            parts.append(self._sweep_highlight_js())

        if not parts:
            return None
        body = "".join(parts)
        return (
            f"setTimeout(function(){{"
            f"var gd=document.querySelector('.js-plotly-plot');"
            f"{body}"
            f"}},0)"
        )

    def _sweep_highlight_js(self) -> str:
        """Return the sweep highlight JS with styling constants substituted.

        Returns:
            A self-executing JS function string.
        """
        return (
            _SWEEP_HIGHLIGHT_JS.replace(
                "/*DIM_OPACITY*/", str(constants.HIGHLIGHT_DIM_OPACITY)
            )
            .replace("/*HOVER_WIDTH*/", str(constants.HIGHLIGHT_HOVER_WIDTH))
            .replace("/*DIM_WIDTH*/", str(constants.HIGHLIGHT_DIM_WIDTH))
            .replace("/*SELECTED_SWEEP*/", str(self.selected_sweep))
        )

    def _do_add_sweep(self) -> None:
        """Append current sweeps to the saved overlay without applying visibility JS.

        Core logic extracted so tests can call it synchronously without
        triggering the async ``get_state`` path.
        """
        if not self.has_result:
            return
        logger.debug("Adding %d sweep(s) to overlay", len(self.current_sweeps))
        for sweep in self.current_sweeps:
            idx = len(self.saved_sweeps)
            color = constants.SWEEP_COLORS[idx % len(constants.SWEEP_COLORS)]
            self.saved_sweeps.append(
                sweep.model_copy(
                    update={
                        "color": color,
                        "label": f"Sweep {idx + 1} ({self._label_neuron_type})",
                    }
                )
            )

    async def add_sweep(self):
        """Promote current simulation result to the saved sweep overlay."""
        self._do_add_sweep()
        vis_st = await self.get_state(VisibilityState)
        js = self._apply_visibility_js(vis_st)
        if js:
            return rx.call_script(js)

    def _do_clear_sweeps(self) -> None:
        """Empty the saved sweep overlay without applying visibility JS.

        Core logic extracted so tests can call it synchronously.
        """
        logger.debug("Cleared %d saved sweep(s)", len(self.saved_sweeps))
        self.saved_sweeps = []

    async def clear_sweeps(self):
        """Remove all saved sweeps."""
        self._do_clear_sweeps()
        vis_st = await self.get_state(VisibilityState)
        js = self._apply_visibility_js(vis_st)
        if js:
            return rx.call_script(js)

    # ------------------------------------------------------------------ #
    # Stored trace management                                            #
    # ------------------------------------------------------------------ #
    def _do_store_trace(self) -> None:
        """Snapshot the current sweep into stored traces without applying visibility JS.

        Core logic extracted so tests can call it synchronously.
        """
        if not self.has_result:
            return
        idx = len(self.stored_traces)
        color = constants.STORED_TRACE_COLORS[idx % len(constants.STORED_TRACE_COLORS)]
        label = f"Stored {idx + 1} ({self._label_neuron_type})"
        self.stored_traces.append(
            self.current_sweeps[0].model_copy(update={"color": color, "label": label})
        )

    async def store_trace(self) -> None:
        """Snapshot the current sweep into the oscilloscope stored traces."""
        self._do_store_trace()
        vis_st = await self.get_state(VisibilityState)
        js = self._apply_visibility_js(vis_st)
        if js:
            return rx.call_script(js)

    def _do_clear_stored_traces(self) -> None:
        """Remove all stored traces without applying visibility JS.

        Core logic extracted so tests can call it synchronously.
        """
        self.stored_traces = []

    async def clear_stored_traces(self) -> None:
        """Remove all oscilloscope stored traces."""
        self._do_clear_stored_traces()
        vis_st = await self.get_state(VisibilityState)
        js = self._apply_visibility_js(vis_st)
        if js:
            return rx.call_script(js)

    # ------------------------------------------------------------------ #
    # Continuous simulation mode                                        #
    # ------------------------------------------------------------------ #
    def toggle_continuous_mode(self) -> None:
        """Enable or disable continuous simulation mode."""
        if self.continuous_loop_running:
            # Signal the running loop to stop
            self.continuous_mode = False
        else:
            self.continuous_mode = True
            return SimulationState.run_continuous  # type: ignore[return-value]

    @rx.event(background=True)
    async def run_continuous(self) -> AsyncGenerator[Any, None]:
        """Run simulations in a continuous loop until continuous_mode is False.

        Each iteration picks up the terminal neuron state (voltage, gating
        variables, Ca²⁺) from the previous iteration, giving true continuity.
        Parameter changes made while the loop runs take effect at the start
        of the next iteration.
        """
        async with self:
            self.continuous_loop_running = True
            self.error_message = ""

        async with self:
            proto_st = await self.get_state(ProtocolState)
        logger.info(
            "Continuous simulation started: mode=%s, protocol=%s",
            proto_st.clamp_mode,
            proto_st.protocol_type,
        )
        _iteration = 0
        try:
            while True:
                _iter_start = time.monotonic()
                # Snapshot all state needed for this iteration.
                async with self:
                    if not self.continuous_mode:
                        break

                    proto_st = await self.get_state(ProtocolState)
                    mode = proto_st.clamp_mode
                    ptype = proto_st.protocol_type
                    neuron_st = await self.get_state(NeuronState)
                    neuron = neuron_st._build_neuron()
                    stimulus = proto_st._build_protocols()[0][0]
                    use_prior_state = self._cont_has_state
                    prior_V = self._cont_V
                    prior_gating = dict(self._cont_gating)
                    prior_ca_i = self._cont_ca_i

                    _iteration += 1
                    logger.debug(
                        "Continuous iteration %d: mode=%s protocol=%s "
                        "use_prior_state=%s",
                        _iteration,
                        mode,
                        ptype,
                        use_prior_state,
                    )

                # Run simulation outside the state lock in a thread executor.
                loop = asyncio.get_running_loop()
                try:
                    if mode == CURRENT_CLAMP:
                        if use_prior_state:
                            result = await loop.run_in_executor(
                                None,
                                patch_sim.simulate_current_clamp_from_state,
                                neuron,
                                stimulus,
                                prior_V,
                                prior_gating,
                                prior_ca_i,
                            )
                        else:
                            result = await loop.run_in_executor(
                                None,
                                patch_sim.simulate_current_clamp,
                                neuron,
                                stimulus,
                            )
                    else:
                        if use_prior_state:
                            result = await loop.run_in_executor(
                                None,
                                patch_sim.simulate_voltage_clamp_from_state,
                                neuron,
                                stimulus,
                                prior_gating,
                                prior_ca_i,
                            )
                        else:
                            result = await loop.run_in_executor(
                                None,
                                patch_sim.simulate_voltage_clamp,
                                neuron,
                                stimulus,
                            )
                except ValueError as exc:
                    logger.exception("Continuous simulation error: %s", exc)
                    async with self:
                        self.error_message = str(exc)
                        self.continuous_mode = False
                    break

                # Extract terminal state for next iteration.
                last_V = float(result["voltage"][-1])
                gating_vars = {gv.name for gv in neuron.all_gating_variables}
                gating_cols = [col for col in result.dtype.names if col in gating_vars]
                last_gating = {col: float(result[col][-1]) for col in gating_cols}
                last_ca_i = (
                    float(result["ca_i"][-1]) if "ca_i" in result.dtype.names else 0.0
                )

                sweep = Sweep.from_result(result, stimulus, "", "", mode)

                async with self:
                    if not self.continuous_mode:
                        break
                    self.current_sweeps = [sweep]
                    self._cont_V = last_V
                    self._cont_gating = last_gating
                    self._cont_ca_i = last_ca_i
                    self._cont_has_state = True

                _iter_elapsed_ms = (time.monotonic() - _iter_start) * 1000
                if _iter_elapsed_ms > 500:
                    logger.warning(
                        "Continuous iteration %d took %.0f ms (>500 ms threshold)",
                        _iteration,
                        _iter_elapsed_ms,
                    )

                # Yield to allow UI events (parameter changes) to be processed.
                await asyncio.sleep(0)

        finally:
            async with self:
                self.continuous_mode = False
                self.continuous_loop_running = False
                logger.info(
                    "Continuous simulation stopped after %d iteration(s)", _iteration
                )
                log_st = await self.get_state(LogState)
                log_st._refresh_logs()
                vis_st = await self.get_state(VisibilityState)
            js = self._apply_visibility_js(vis_st)
            if js:
                yield rx.call_script(js)
            yield rx.call_script(_LOG_SCROLL_JS)

    @rx.event(background=True)
    async def run_simulation(self) -> AsyncGenerator[Any, None]:
        """Build protocol and run the simulation asynchronously.

        Uses background=True so the UI stays responsive during long runs.
        State is snapshotted inside the lock before any blocking work begins,
        and all blocking simulation calls are offloaded via run_in_executor so
        the event loop can flush the is_running=True update to the client
        before the computation starts.
        """
        async with self:
            self.is_running = True
            self.error_message = ""
            self.selected_sweep = -1
            neuron_st = await self.get_state(NeuronState)
            neuron = neuron_st._build_neuron()
            proto_st = await self.get_state(ProtocolState)
            mode = proto_st.clamp_mode
            ptype = proto_st.protocol_type
            try:
                protocols = proto_st._build_protocols()
            except ValueError as exc:
                logger.exception("Simulation error: %s", exc)
                self.error_message = str(exc)
                self.is_running = False
                return

        _start_ms = time.monotonic() * 1000
        logger.info(
            "Simulation started: mode=%s, protocol=%s",
            mode,
            ptype,
        )
        loop = asyncio.get_running_loop()
        sim_fn = (
            patch_sim.simulate_current_clamp
            if mode == CURRENT_CLAMP
            else patch_sim.simulate_voltage_clamp
        )
        is_multi = len(protocols) > 1
        try:
            if is_multi:
                # Run each sweep independently so gating variables are reset
                # between steps — matching real patch-clamp I-V experiments.
                def _run_batch() -> list[Sweep]:
                    """Run all sweeps via simulate_batch and assemble Sweep list."""
                    new_sweeps: list[Sweep] = []
                    for sweep_result, (protocol, label) in zip(
                        patch_sim.simulate_batch(
                            neuron, [p for p, _ in protocols], sim_fn
                        ),
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
                    return new_sweeps

                new_sweeps = await loop.run_in_executor(None, _run_batch)
                async with self:
                    self.current_sweeps = new_sweeps
                    analysis_st = await self.get_state(AnalysisState)
                    analysis_st.ap_metrics = []
                    analysis_st.ap_summary = {}
                    if mode == VOLTAGE_CLAMP:
                        analysis_st.iv_data = _compute_iv_data(
                            new_sweeps,
                            proto_st.min_stimulus,
                            proto_st.max_stimulus,
                            proto_st.stimulus_step,
                            proto_st.pre_stimulus_duration,
                            proto_st.stimulus_duration,
                        )
                    else:
                        analysis_st.iv_data = {}

            else:
                stimulus, _ = protocols[0]
                result = await loop.run_in_executor(None, sim_fn, neuron, stimulus)
                async with self:
                    self.current_sweeps = [
                        Sweep.from_result(result, stimulus, "", "", mode)
                    ]
                    analysis_st = await self.get_state(AnalysisState)
                    if mode == CURRENT_CLAMP:
                        ap_result = patch_sim.analyze_aps_from_result(result)
                        analysis_st.ap_metrics = [
                            {
                                "index": s.index,
                                "threshold_voltage": f"{s.threshold_voltage:.1f}",
                                "peak_voltage": f"{s.peak_voltage:.1f}",
                                "rise_time": f"{s.rise_time:.2f}",
                                "half_width": f"{s.half_width:.2f}",
                                "ahp_depth": (
                                    f"{s.ahp_depth:.1f}"
                                    if s.ahp_depth is not None
                                    else "\u2014"
                                ),
                            }
                            for s in ap_result.spikes
                        ]
                        analysis_st.ap_summary = {
                            "spike_count": str(ap_result.spike_count),
                            "mean_threshold_voltage": (
                                f"{ap_result.mean_threshold_voltage:.1f}"
                                if ap_result.mean_threshold_voltage is not None
                                else "\u2014"
                            ),
                            "mean_peak_voltage": (
                                f"{ap_result.mean_peak_voltage:.1f}"
                                if ap_result.mean_peak_voltage is not None
                                else "\u2014"
                            ),
                            "mean_rise_time": (
                                f"{ap_result.mean_rise_time:.2f}"
                                if ap_result.mean_rise_time is not None
                                else "\u2014"
                            ),
                            "mean_half_width": (
                                f"{ap_result.mean_half_width:.2f}"
                                if ap_result.mean_half_width is not None
                                else "\u2014"
                            ),
                            "mean_ahp_depth": (
                                f"{ap_result.mean_ahp_depth:.1f}"
                                if ap_result.mean_ahp_depth is not None
                                else "\u2014"
                            ),
                            "mean_isi": (
                                f"{ap_result.mean_isi:.1f}"
                                if ap_result.mean_isi is not None
                                else "\u2014"
                            ),
                            "firing_rate": (
                                f"{ap_result.firing_rate:.1f}"
                                if ap_result.firing_rate is not None
                                else "\u2014"
                            ),
                        }
                        analysis_st.iv_data = {}
                    else:
                        analysis_st.ap_metrics = []
                        analysis_st.ap_summary = {}
                        analysis_st.iv_data = {}

        except ValueError as exc:
            logger.exception("Simulation error: %s", exc)
            async with self:
                self.error_message = str(exc)
        else:
            elapsed = time.monotonic() * 1000 - _start_ms
            logger.info("Simulation complete: %.0f ms", elapsed)
        finally:
            async with self:
                self.is_running = False
                log_st = await self.get_state(LogState)
                log_st._refresh_logs()
                vis_st = await self.get_state(VisibilityState)
            js = self._apply_visibility_js(vis_st)
            if js:
                yield rx.call_script(js)
            yield rx.call_script(_LOG_SCROLL_JS)
